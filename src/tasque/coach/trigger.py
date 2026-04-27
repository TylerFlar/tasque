"""The single-source coach trigger queue.

Design:

- ``enqueue(bucket, reason, dedup_key=None)`` writes a row in
  ``coach_pending``. Two-phase dedup against ``(bucket, dedup_key)``:

    1. **Pending dedup, no time bound.** If any existing row with the
       same ``(bucket, dedup_key)`` has ``claimed_at IS NULL``, the call
       is a no-op. "Don't queue a duplicate of something already in the
       queue" — doesn't matter how long it's been sitting there.
    2. **Post-claim dedup, short window.** If any existing row was
       claimed within :data:`DEFAULT_DEDUP_SECONDS` (5 min by default),
       the call is also a no-op. This is the protection the user
       cares about: same trigger event arriving twice in quick
       succession (gateway reconnect replay, scheduler retry storm,
       accidental duplicate dispatch).

  Genuine re-triggers with the same dedup_key arriving more than 5 min
  after the previous one's claim time DO enqueue — the dedup is not a
  cooldown.

  ``dedup_key=None`` always enqueues — that's the explicit-wake path
  used by the CLI.

- ``run_drainer(...)`` is an asyncio task. It loops claiming the oldest
  unclaimed row. When a bucket is in flight, no second row for that
  bucket is claimed until the first finishes. On success the row is
  deleted; on failure the row keeps its ``claimed_at`` so the dedup
  window blocks an immediate retry storm — but only for 5 min, not
  hours.

- The dedup window defaults to 300 seconds and is overridable via
  ``TASQUE_COACH_DEDUP_SECONDS``.

This is intentionally the *only* path to firing a coach. Worker
completion and Discord replies MUST call ``enqueue`` rather than
invoking the coach directly.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from tasque.buckets import ALL_BUCKETS, Bucket
from tasque.coach.graph import BucketCoachState, run_bucket_coach
from tasque.memory.db import get_session
from tasque.memory.entities import CoachPending, utc_now_iso

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

log = structlog.get_logger(__name__)

DEFAULT_DEDUP_SECONDS = 300.0
DEFAULT_DRAINER_POLL_SECONDS = 1.0


def dedup_window() -> timedelta:
    """Return the post-claim dedup window, honoring ``TASQUE_COACH_DEDUP_SECONDS``.

    The window applies only to the post-claim phase — pending rows
    dedup permanently regardless of this value (see module docstring).
    """
    raw = os.environ.get("TASQUE_COACH_DEDUP_SECONDS")
    seconds = DEFAULT_DEDUP_SECONDS
    if raw is not None:
        try:
            seconds = float(raw)
        except ValueError:
            seconds = DEFAULT_DEDUP_SECONDS
    return timedelta(seconds=seconds)


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def enqueue(
    bucket: Bucket,
    reason: str,
    *,
    dedup_key: str | None = None,
) -> str | None:
    """Write a ``coach_pending`` row, or no-op if a recent matching row exists.

    Returns the new row's id, or ``None`` if dedup suppressed the write.

    ``dedup_key=None`` always enqueues. With a key, the call is a no-op
    if any row with the same ``(bucket, dedup_key)`` is either unclaimed
    (pending dedup, no time bound) OR was claimed within
    :func:`dedup_window` (post-claim dedup, default 5 min). See the
    module docstring for the rationale.
    """
    if bucket not in ALL_BUCKETS:
        raise ValueError(f"unknown bucket: {bucket!r}")

    cutoff = datetime.now(UTC) - dedup_window()

    with get_session() as sess:
        if dedup_key is not None:
            stmt = (
                select(CoachPending)
                .where(CoachPending.bucket == bucket)
                .where(CoachPending.dedup_key == dedup_key)
            )
            existing = list(sess.execute(stmt).scalars().all())
            for row in existing:
                if row.claimed_at is None:
                    return None
                claimed_dt = _parse_iso(row.claimed_at)
                if claimed_dt is not None and claimed_dt > cutoff:
                    return None

        new_row = CoachPending(
            bucket=bucket,
            reason=reason,
            dedup_key=dedup_key,
        )
        sess.add(new_row)
        sess.flush()
        new_id = new_row.id
        return new_id


def claim_one(*, exclude_buckets: set[str] | None = None) -> CoachPending | None:
    """Claim the oldest unclaimed row not in an in-flight bucket.

    Sets ``claimed_at`` and returns the row. Returns ``None`` if no
    eligible row exists.
    """
    blocked = exclude_buckets or set()
    with get_session() as sess:
        stmt = (
            select(CoachPending)
            .where(CoachPending.claimed_at.is_(None))
            .order_by(CoachPending.enqueued_at.asc())
        )
        rows = list(sess.execute(stmt).scalars().all())
        for row in rows:
            if row.bucket in blocked:
                continue
            row.claimed_at = utc_now_iso()
            sess.flush()
            sess.expunge(row)
            return row
    return None


def _delete_row(row_id: str) -> None:
    with get_session() as sess:
        obj = sess.get(CoachPending, row_id)
        if obj is not None:
            sess.delete(obj)


CoachRunner = Callable[[Bucket, str], Any]


def _default_runner(llm: BaseChatModel | None) -> CoachRunner:
    def _run(bucket: Bucket, reason: str) -> BucketCoachState:
        return run_bucket_coach(bucket, reason=reason, llm=llm)

    return _run


async def run_drainer(
    *,
    stop: asyncio.Event | None = None,
    runner: CoachRunner | None = None,
    llm: BaseChatModel | None = None,
    poll_seconds: float = DEFAULT_DRAINER_POLL_SECONDS,
    max_iterations: int | None = None,
) -> int:
    """Drain ``coach_pending`` until ``stop`` is set or the queue empties.

    One coach run executes at a time per bucket. Successful runs delete
    their row; failures leave the row claimed so the dedup window
    blocks an immediate retry storm (default 5 min).

    ``max_iterations`` caps the loop count (used by tests). ``runner``
    overrides the bucket-coach call (used by tests so an in-memory fake
    LLM can be threaded in without going through ``run_bucket_coach``).
    """
    actual_runner = runner if runner is not None else _default_runner(llm)
    in_flight: set[str] = set()
    iterations = 0
    runs = 0

    while True:
        if stop is not None and stop.is_set():
            return runs
        if max_iterations is not None and iterations >= max_iterations:
            return runs
        iterations += 1

        row = claim_one(exclude_buckets=in_flight)
        if row is None:
            # Nothing to do right now; wait briefly so tests can drive us
            # synchronously by setting ``stop`` instead of leaking time.
            if stop is not None and stop.is_set():
                return runs
            await asyncio.sleep(poll_seconds)
            continue

        bucket = row.bucket  # narrow type — Literal validated on enqueue
        in_flight.add(bucket)
        runs += 1
        log.info(
            "coach.drainer.claimed",
            row_id=row.id[:8],
            bucket=bucket,
            reason=(row.reason or "")[:120],
            dedup_key=row.dedup_key,
        )

        async def _execute(row_id: str = row.id, b: str = bucket, reason: str = row.reason) -> None:
            started = datetime.now(UTC)
            try:
                bucket_typed = _coerce_bucket(b)
                await asyncio.to_thread(actual_runner, bucket_typed, reason)
                _delete_row(row_id)
                log.info(
                    "coach.drainer.completed",
                    row_id=row_id[:8],
                    bucket=b,
                    duration_s=round((datetime.now(UTC) - started).total_seconds(), 1),
                )
            except Exception as exc:
                log.exception(
                    "coach.drainer.run_failed",
                    row_id=row_id,
                    bucket=b,
                    error=str(exc),
                )
                # Row stays with claimed_at set — the post-claim dedup
                # window blocks an immediate retry storm (default 5 min)
                # but doesn't lock the key out for hours. After the
                # window, a new event with the same dedup_key DOES
                # re-enqueue, which is what we want for transient
                # failures (next legitimate trigger gets a fresh shot).
            finally:
                in_flight.discard(b)

        # Await each run before claiming the next: serial drain.
        await _execute()

    return runs


def _coerce_bucket(value: str) -> Bucket:
    if value not in ALL_BUCKETS:
        raise ValueError(f"row has unknown bucket: {value!r}")
    return value  # type: ignore[return-value]


async def drain_until_empty(
    *,
    runner: CoachRunner | None = None,
    llm: BaseChatModel | None = None,
) -> int:
    """Synchronous-ish helper: claim and run until the queue is empty.

    Used by the CLI ``tasque coach wake`` command and by tests. Returns
    the number of coach runs that fired.
    """
    actual_runner = runner if runner is not None else _default_runner(llm)
    runs = 0
    in_flight: set[str] = set()
    while True:
        row = claim_one(exclude_buckets=in_flight)
        if row is None:
            return runs
        bucket = row.bucket
        in_flight.add(bucket)
        runs += 1
        try:
            bucket_typed = _coerce_bucket(bucket)
            await asyncio.to_thread(actual_runner, bucket_typed, row.reason)
            _delete_row(row.id)
        except Exception as exc:
            log.exception(
                "coach.drainer.run_failed",
                row_id=row.id,
                bucket=bucket,
                error=str(exc),
            )
        finally:
            in_flight.discard(bucket)

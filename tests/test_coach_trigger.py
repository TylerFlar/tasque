"""Tests for the coach trigger queue + drainer."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from tasque.buckets import Bucket
from tasque.coach.trigger import (
    claim_one,
    drain_until_empty,
    enqueue,
    run_drainer,
)
from tasque.memory.db import get_session
from tasque.memory.entities import CoachPending


def _count_pending() -> int:
    with get_session() as sess:
        return len(list(sess.execute(select(CoachPending)).scalars().all()))


# ------------------------------------------------------------- enqueue dedup

def test_enqueue_with_none_dedup_always_writes() -> None:
    a = enqueue("health", "first", dedup_key=None)
    b = enqueue("health", "second", dedup_key=None)
    c = enqueue("health", "third", dedup_key=None)
    assert a is not None and b is not None and c is not None
    assert _count_pending() == 3


def test_enqueue_dedup_suppresses_unclaimed_duplicate() -> None:
    first = enqueue("career", "x", dedup_key="job-42")
    second = enqueue("career", "x", dedup_key="job-42")
    assert first is not None
    assert second is None
    assert _count_pending() == 1


def test_enqueue_dedup_allows_after_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1-second window → a 5-second-old claim falls outside it.
    monkeypatch.setenv("TASQUE_COACH_DEDUP_SECONDS", "1")
    first = enqueue("home", "x", dedup_key="k")
    assert first is not None
    # Force-mark it claimed in the past so it falls out of the window.
    past = (datetime.now(UTC) - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with get_session() as sess:
        row = sess.get(CoachPending, first)
        assert row is not None
        row.claimed_at = past
    second = enqueue("home", "x", dedup_key="k")
    assert second is not None
    assert _count_pending() == 2


def test_enqueue_dedup_blocks_inside_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1-hour window — anything claimed in the last few seconds is inside it.
    monkeypatch.setenv("TASQUE_COACH_DEDUP_SECONDS", str(3600))
    first = enqueue("finance", "x", dedup_key="k")
    assert first is not None
    recent = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with get_session() as sess:
        row = sess.get(CoachPending, first)
        assert row is not None
        row.claimed_at = recent
    second = enqueue("finance", "x", dedup_key="k")
    assert second is None


def test_default_dedup_window_is_short() -> None:
    """Regression guard against accidentally re-introducing the multi-hour
    cooldown. The whole point of the rethink is that legitimate re-triggers
    of the same dedup_key (e.g. user replies to the same coach thread an
    hour later) MUST go through. Anything ≥ ~30 min would silently swallow
    those."""
    from tasque.coach.trigger import DEFAULT_DEDUP_SECONDS, dedup_window

    assert DEFAULT_DEDUP_SECONDS <= 30 * 60, (
        "default dedup window must stay short — it's anti-stutter, not a cooldown"
    )
    # And the env-var path returns the same default when unset.
    assert dedup_window().total_seconds() == DEFAULT_DEDUP_SECONDS


def test_enqueue_dedup_pending_is_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pending dedup has no time bound — even if the window is 0, a row
    that's still pending must suppress duplicates. Otherwise gateway
    replays could double-queue while the first row sits unclaimed."""
    monkeypatch.setenv("TASQUE_COACH_DEDUP_SECONDS", "0")
    first = enqueue("creative", "x", dedup_key="evt-1")
    assert first is not None
    second = enqueue("creative", "x", dedup_key="evt-1")
    assert second is None
    assert _count_pending() == 1


# ------------------------------------------------------------- claim ordering

def test_claim_one_returns_oldest_first() -> None:
    a = enqueue("health", "first", dedup_key=None)
    b = enqueue("career", "second", dedup_key=None)
    c = enqueue("home", "third", dedup_key=None)
    assert a and b and c
    first = claim_one()
    second = claim_one()
    third = claim_one()
    assert first is not None and second is not None and third is not None
    assert first.id == a
    assert second.id == b
    assert third.id == c
    assert claim_one() is None


def test_claim_one_skips_excluded_bucket() -> None:
    enqueue("health", "h1", dedup_key=None)
    enqueue("career", "c1", dedup_key=None)
    row = claim_one(exclude_buckets={"health"})
    assert row is not None
    assert row.bucket == "career"


# -------------------------------------------------------------- drainer runs


def _make_recording_runner() -> tuple[list[tuple[str, str]], Any]:
    log: list[tuple[str, str]] = []

    def runner(bucket: Bucket, reason: str) -> None:
        log.append((bucket, reason))

    return log, runner


def test_drainer_dedups_to_one_run_for_same_bucket() -> None:
    # Five rows for the same bucket, all with identical dedup_key.
    enqueued: list[str | None] = [
        enqueue("health", f"r{i}", dedup_key="same") for i in range(5)
    ]
    # First wins; rest are no-ops.
    assert enqueued.count(None) == 4
    assert _count_pending() == 1

    log, runner = _make_recording_runner()
    runs = asyncio.run(drain_until_empty(runner=runner))
    assert runs == 1
    assert log == [("health", "r0")]
    assert _count_pending() == 0


def test_drainer_runs_once_per_bucket_serially() -> None:
    # One row per bucket, five buckets.
    buckets: list[Bucket] = ["health", "career", "finance", "home", "creative"]
    for b in buckets:
        assert enqueue(b, f"reason-{b}", dedup_key=None) is not None
    assert _count_pending() == 5

    seen: list[str] = []
    in_flight_lock = threading.Lock()
    in_flight: set[str] = set()
    overlap_detected = False

    def runner(bucket: Bucket, reason: str) -> None:
        nonlocal overlap_detected
        with in_flight_lock:
            if in_flight:
                overlap_detected = True
            in_flight.add(bucket)
        seen.append(bucket)
        with in_flight_lock:
            in_flight.discard(bucket)

    runs = asyncio.run(drain_until_empty(runner=runner))
    assert runs == 5
    assert sorted(seen) == sorted(buckets)
    assert overlap_detected is False
    assert _count_pending() == 0


def test_drainer_keeps_row_on_failure() -> None:
    assert enqueue("personal", "boom", dedup_key=None) is not None

    def runner(bucket: Bucket, reason: str) -> None:
        raise RuntimeError("simulated failure")

    runs = asyncio.run(drain_until_empty(runner=runner))
    assert runs == 1
    # Row stays — claimed_at set, so the dedup window blocks reruns.
    assert _count_pending() == 1
    with get_session() as sess:
        row = sess.execute(select(CoachPending)).scalars().first()
        assert row is not None
        assert row.claimed_at is not None


def test_run_drainer_respects_max_iterations() -> None:
    enqueue("health", "x", dedup_key=None)
    log, runner = _make_recording_runner()

    async def go() -> int:
        return await run_drainer(
            runner=runner,
            poll_seconds=0.01,
            max_iterations=10,
        )

    runs = asyncio.run(go())
    assert runs == 1
    assert log == [("health", "x")]


def test_enqueue_rejects_unknown_bucket() -> None:
    with pytest.raises(ValueError):
        enqueue("not-a-bucket", "x", dedup_key=None)  # type: ignore[arg-type]

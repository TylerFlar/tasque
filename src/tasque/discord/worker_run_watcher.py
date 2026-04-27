"""Worker-run notification watcher.

Polls every ``poll_seconds`` for ``QueuedJob`` rows that have terminated
(``status IN ('completed', 'failed')``) but haven't yet been announced
to JOBS (``notified_at IS NULL``), and fires
:func:`tasque.discord.notify.notify_worker_run` for each. The notify
helper itself stamps ``QueuedJob.notified_at`` on success — this loop
just discovers eligible rows and dispatches the post.

Bridges the synchronous APScheduler scheduler to the asyncio bot loop
via DB polling — same pattern the chain-status watcher uses. Cross-
thread async bridging (``run_coroutine_threadsafe``) is intentionally
NOT used: the scheduler stamps the WorkerResult fields onto the row
synchronously, the watcher reads them on its next tick, and the post
happens inside the asyncio loop where the nextcord client lives.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select

from tasque.discord import notify, poster
from tasque.memory.db import get_session
from tasque.memory.entities import QueuedJob

log = structlog.get_logger(__name__)

DEFAULT_POLL_SECONDS = 10.0
MIN_POLL_SECONDS = 3.0  # Discord rate-limits message creates.

# How far back we look for un-notified terminal jobs. A job that
# completed more than this long ago without being announced — e.g. the
# daemon was offline when it finished and never recovered — stays silent
# rather than flooding JOBS on the next boot. Mirrors
# ``tasque.proxy.server.LOG_RETENTION_SECONDS`` (7 days) so retention
# semantics are uniform across the system.
NOTIFY_LOOKBACK_SECONDS = 7 * 24 * 60 * 60


_TERMINAL_STATUSES = ("completed", "failed")


def _decode_produces(raw: str | None) -> dict[str, Any]:
    """Round-trip the JSON blob the scheduler stashed in
    ``QueuedJob.last_produces_json``. Bad / missing payloads degrade to
    an empty dict — better than crashing the watcher tick over a
    corrupt row."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return decoded


async def _tick(*, notified_inflight: set[str]) -> int:
    """One pass over un-notified terminal jobs. Returns the number of
    notifications attempted this pass.

    ``notified_inflight`` carries the in-process job ids we've already
    tried this watcher run, so a transient post failure doesn't cause us
    to spam-retry every tick. Membership is dropped when the row's id
    falls out of the lookback window above.
    """
    cutoff_dt = datetime.now(UTC) - timedelta(seconds=NOTIFY_LOOKBACK_SECONDS)
    cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    with get_session() as sess:
        rows = list(
            sess.execute(
                select(QueuedJob)
                .where(QueuedJob.status.in_(_TERMINAL_STATUSES))
                .where(QueuedJob.notified_at.is_(None))
                .where(QueuedJob.visible.is_(True))
                .where(QueuedJob.updated_at >= cutoff_iso)
                .order_by(QueuedJob.updated_at.asc())
            ).scalars().all()
        )
        sess.expunge_all()

    seen_ids: set[str] = set()
    attempts = 0
    for job in rows:
        seen_ids.add(job.id)
        if job.id in notified_inflight:
            continue
        # Tag inflight BEFORE the post so a poster exception doesn't
        # cause every subsequent tick to retry; the row stays
        # un-notified persistently (notified_at column unset) but in-
        # memory we wait for the next watcher restart to retry.
        notified_inflight.add(job.id)
        attempts += 1

        produces = _decode_produces(job.last_produces_json)
        try:
            await notify.notify_worker_run(
                job,
                summary=job.last_summary or "",
                report=job.last_report or "",
                produces=produces,
                error=job.last_error,
            )
        except Exception:
            log.exception(
                "discord.worker_run_watcher.notify_failed",
                job_id=job.id[:8],
            )

    # Drop inflight ids for jobs no longer in the watch set.
    stale = [k for k in notified_inflight if k not in seen_ids]
    for k in stale:
        notified_inflight.discard(k)
    return attempts


async def run_worker_run_watcher(
    *,
    stop: asyncio.Event | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_iterations: int | None = None,
) -> int:
    """Long-running loop that announces completed / failed worker runs.

    Returns the number of ticks executed (used by tests). Each tick is
    silently skipped until the bot's ``on_ready`` has installed a poster
    client — same self-defer pattern as the chain status watcher so the
    very first tick doesn't race the gateway connect.
    """
    interval = max(poll_seconds, MIN_POLL_SECONDS)
    notified_inflight: set[str] = set()
    iters = 0
    while True:
        if stop is not None and stop.is_set():
            return iters
        if max_iterations is not None and iters >= max_iterations:
            return iters
        iters += 1
        if not poster.client_ready():
            log.debug("discord.worker_run_watcher.waiting_for_poster")
        else:
            try:
                await _tick(notified_inflight=notified_inflight)
            except Exception:
                log.exception("discord.worker_run_watcher.tick_failed")
        if max_iterations is not None and iters >= max_iterations:
            return iters
        if stop is not None and stop.is_set():
            return iters
        await asyncio.sleep(interval)


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "MIN_POLL_SECONDS",
    "NOTIFY_LOOKBACK_SECONDS",
    "run_worker_run_watcher",
]

"""DLQ notification watcher.

Polls every ``poll_seconds`` for ``FailedJob`` rows that haven't yet
been announced to the DLQ channel (``notified_at IS NULL``) and fires
:func:`tasque.discord.notify.notify_failed_job` for each — passing the
Retry + Resolve button view from
:func:`tasque.discord.chain_ui.build_retry_view`. The notify helper
itself stamps ``FailedJob.notified_at`` on success; this loop just
discovers eligible rows and dispatches the post.

Mirrors the ``worker_run_watcher`` pattern: a synchronous DB write
records the failure, the asyncio watcher picks it up on its next tick
and posts inside the bot's event loop where the nextcord client lives.
Already-resolved rows are skipped — the operator might mark a stale
row resolved before the daemon ever boots.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select

from tasque.discord import chain_ui, notify, poster
from tasque.memory.db import get_session
from tasque.memory.entities import FailedJob

log = structlog.get_logger(__name__)

DEFAULT_POLL_SECONDS = 10.0
MIN_POLL_SECONDS = 3.0  # Discord rate-limits message creates.

# How far back we look for un-notified failures. A row that landed more
# than this long ago without being announced — daemon was offline at
# the time and never recovered — stays silent rather than flooding DLQ
# on the next boot. Mirrors ``worker_run_watcher.NOTIFY_LOOKBACK_SECONDS``.
NOTIFY_LOOKBACK_SECONDS = 7 * 24 * 60 * 60


async def _tick(*, notified_inflight: set[str]) -> int:
    """One pass over un-notified, unresolved failures. Returns the
    number of notifications attempted this pass.

    ``notified_inflight`` carries the in-process FailedJob ids we've
    already tried this watcher run, so a transient post failure doesn't
    cause us to spam-retry every tick. Membership is dropped when the
    row's id falls out of the lookback window.
    """
    cutoff_dt = datetime.now(UTC) - timedelta(seconds=NOTIFY_LOOKBACK_SECONDS)
    cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    with get_session() as sess:
        rows = list(
            sess.execute(
                select(FailedJob)
                .where(FailedJob.notified_at.is_(None))
                .where(FailedJob.resolved.is_(False))
                .where(FailedJob.created_at >= cutoff_iso)
                .order_by(FailedJob.created_at.asc())
            ).scalars().all()
        )
        sess.expunge_all()

    seen_ids: set[str] = set()
    attempts = 0
    for fj in rows:
        seen_ids.add(fj.id)
        if fj.id in notified_inflight:
            continue
        # Tag inflight BEFORE the post so a poster exception doesn't
        # cause every subsequent tick to retry; the row stays
        # un-notified persistently (notified_at column unset) but in-
        # memory we wait for the next watcher restart to retry.
        notified_inflight.add(fj.id)
        attempts += 1

        try:
            view = chain_ui.build_retry_view(fj.id)
        except Exception:
            log.exception(
                "discord.dlq_watcher.build_view_failed",
                failed_job_id=fj.id[:8],
            )
            view = None

        try:
            await notify.notify_failed_job(fj, retry_view=view)
        except Exception:
            log.exception(
                "discord.dlq_watcher.notify_failed",
                failed_job_id=fj.id[:8],
            )

    # Drop inflight ids for failures no longer in the watch set.
    stale = [k for k in notified_inflight if k not in seen_ids]
    for k in stale:
        notified_inflight.discard(k)
    return attempts


async def run_dlq_watcher(
    *,
    stop: asyncio.Event | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_iterations: int | None = None,
) -> int:
    """Long-running loop that announces unresolved DLQ entries.

    Returns the number of ticks executed (used by tests). Each tick is
    silently skipped until the bot's ``on_ready`` has installed a
    poster client — same self-defer pattern as the other watchers so
    the very first tick doesn't race the gateway connect.
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
            log.debug("discord.dlq_watcher.waiting_for_poster")
        else:
            try:
                await _tick(notified_inflight=notified_inflight)
            except Exception:
                log.exception("discord.dlq_watcher.tick_failed")
        if max_iterations is not None and iters >= max_iterations:
            return iters
        if stop is not None and stop.is_set():
            return iters
        await asyncio.sleep(interval)


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "MIN_POLL_SECONDS",
    "NOTIFY_LOOKBACK_SECONDS",
    "run_dlq_watcher",
]

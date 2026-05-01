"""The three notification helpers wired into the daemon.

- :func:`notify_worker_run` posts a worker-run embed in the jobs
  channel and anchors a per-job thread to it. Replies / discussion
  about a specific worker run land inside that thread.
- :func:`notify_chain_event` posts a one-shot embed on chain start /
  completion / failure. There is no in-flight editing.
- :func:`notify_failed_job` posts a DLQ entry in the DLQ channel and
  anchors a per-failed-job thread to it (carrying the Retry button).

The ``jobs`` and ``dlq`` registry entries hold parent CHANNEL ids,
not thread ids — each notification creates its own thread off its
embed message. Bucket-thread routing for replies is handled by the
coach threads, which are separate.

These run inside the asyncio event loop and so must be awaited.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog

from tasque.discord import embeds, poster, threads
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun, FailedJob, QueuedJob, utc_now_iso

log = structlog.get_logger(__name__)

ChainEventKind = Literal["started", "completed", "failed", "stopped"]

# Discord caps thread names at 100 chars; we trim a hair shy to leave
# room for prefixes added by callers.
_THREAD_NAME_LIMIT = 95


# ----------------------------------------------------------------- helpers


def _trim_thread_name(name: str) -> str:
    cleaned = " ".join(name.split())
    if len(cleaned) <= _THREAD_NAME_LIMIT:
        return cleaned
    return cleaned[: _THREAD_NAME_LIMIT - 1] + "…"


def _job_thread_name(job: QueuedJob) -> str:
    bucket = f"{job.bucket}-" if job.bucket else ""
    short = (job.directive or "job").strip().splitlines()[0]
    return _trim_thread_name(f"job-{bucket}{short}")


def _chain_run_thread_name(chain_run: ChainRun) -> str:
    bucket = f"{chain_run.bucket}-" if chain_run.bucket else ""
    return _trim_thread_name(
        f"chain-{bucket}{chain_run.chain_name}-{chain_run.chain_id[:8]}"
    )


def _persist_chain_run_thread(chain_run_pk: str, thread_id: int) -> None:
    """Cache ``thread_id`` on the ChainRun row so subsequent posts about
    this chain (e.g. follow-up notifications, replies) reuse the same
    thread instead of spawning duplicates."""
    with get_session() as sess:
        row = sess.get(ChainRun, chain_run_pk)
        if row is None:
            return
        row.thread_id = str(thread_id)
        row.updated_at = utc_now_iso()


def _coerce_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _persist_queued_job_thread(job_id: str, thread_id: int) -> None:
    """Cache ``thread_id`` on the QueuedJob row so re-notifications reuse
    the same thread instead of spawning duplicates."""
    with get_session() as sess:
        row = sess.get(QueuedJob, job_id)
        if row is None:
            return
        row.thread_id = str(thread_id)
        row.updated_at = utc_now_iso()


def _final_step_report(state: dict[str, Any] | None) -> str:
    """Pick the report text from the final completed step in ``state``.

    Mirrors :func:`tasque.discord.embeds._pick_final_completed_step` so
    the overflow we post inside the thread is exactly the report whose
    truncated copy went into the embed. Falls back to the last completed
    step of any kind if all completed steps are fan-out children.

    When the chain has fan-out children, appends a per-branch outcome
    roll-up so the thread reader sees the actual leaf results rather
    than just the last passthrough's prose. Returns an empty string
    when no completed step has produced a report and no rollup is
    available.
    """
    if not state:
        return ""
    completed_raw = state.get("completed") or {}
    if not completed_raw:
        return ""
    completed: dict[str, dict[str, Any]] = dict(completed_raw)
    non_fanout = [(k, v) for k, v in completed.items() if "[" not in k]
    items = non_fanout if non_fanout else list(completed.items())
    body = ""
    if items:
        _step_id, output = items[-1]
        body = (output.get("report") or "").strip()
    rollup = embeds.build_fan_out_rollup(state)
    if rollup:
        return (body + "\n\n" + rollup).strip() if body else rollup
    return body


async def _post_full_report_to_thread(
    thread_id: int | None, report: str
) -> None:
    """Post the full ``report`` markdown body inside ``thread_id``.

    The embed only carries the agent's one-paragraph ``summary``; the
    full body lives in the anchored thread so it can be read end-to-end
    or quoted. ``post_long_message`` chunks at the 2000-char Discord
    limit so multi-page digests come through intact.

    Silently no-ops when there's nothing to post (no thread, empty
    report). A failure here is logged but never bubbles — the embed is
    already up.
    """
    if thread_id is None:
        return
    if not report:
        return
    try:
        await poster.post_long_message(thread_id, report)
    except Exception:
        log.exception(
            "discord.notify.thread_body_post_failed",
            thread_id=thread_id,
        )


def _persist_queued_job_notified_at(job_id: str) -> None:
    """Stamp ``QueuedJob.notified_at`` after a successful worker-run post.

    Mirrors :func:`_persist_chain_run_terminal_notified_at` — the
    persistent column survives a daemon restart so the watcher's first
    tick after boot doesn't re-announce a completion that landed before
    shutdown.
    """
    with get_session() as sess:
        row = sess.get(QueuedJob, job_id)
        if row is None:
            return
        row.notified_at = utc_now_iso()
        row.updated_at = utc_now_iso()


# ----------------------------------------------------------------- worker

async def notify_worker_run(
    job: QueuedJob,
    *,
    summary: str,
    report: str,
    produces: dict[str, object] | None = None,
    error: str | None = None,
) -> int | None:
    """Post a worker-run embed for a completed job and anchor a thread to it.

    The embed lives in the ``jobs`` channel (the registry holds that
    channel's id under :data:`tasque.discord.threads.PURPOSE_JOBS`).
    A per-job public thread is then created off the embed message so
    follow-up conversation about that specific job has a home. Returns
    the embed message id, or None if the job is silent / no jobs
    channel is configured.
    """
    if not job.visible:
        return None

    payload = embeds.worker_run_dict(
        job,
        summary=summary,
        report=report,
        produces=dict(produces or {}),
        error=error,
    )
    embed = embeds.build_worker_embed(payload)

    cached_thread = _coerce_int(job.thread_id)
    overflow_thread: int | None = None
    if cached_thread is not None:
        # Re-notification — post a follow-up embed in the existing
        # per-job thread instead of creating a duplicate.
        message_id = await poster.post_embed(cached_thread, embed)
        overflow_thread = cached_thread
    else:
        parent_channel_id = threads.get_thread_id(threads.PURPOSE_JOBS)
        if parent_channel_id is None:
            log.warning("discord.notify.worker.no_channel", job_id=job.id)
            return None

        message_id = await poster.post_embed(parent_channel_id, embed)
        try:
            new_thread_id = await poster.start_thread(
                parent_channel_id, message_id, _job_thread_name(job)
            )
            _persist_queued_job_thread(job.id, new_thread_id)
            overflow_thread = new_thread_id
        except Exception:
            # The embed is already posted; losing the thread is annoying but
            # not fatal — log and continue.
            log.exception(
                "discord.notify.worker.start_thread_failed",
                job_id=job.id,
                channel_id=parent_channel_id,
            )

    # The embed shows the summary; the thread body shows the full
    # report. Always post (when there's anything to post) so the user
    # has the complete output in one place.
    await _post_full_report_to_thread(overflow_thread, report)

    _persist_queued_job_notified_at(job.id)
    return message_id


# ----------------------------------------------------------------- chain

def _resolve_chain_target(chain_run: ChainRun) -> int | None:
    """Where chain-level Discord posts go.

    The chain's per-chain thread if set (created lazily by chain_ui for
    approval steps), otherwise the JOBS channel. JOBS is validated at
    startup so the registry entry is always present in a healthy daemon.
    """
    target: int | None = None
    if chain_run.thread_id:
        try:
            target = int(chain_run.thread_id)
        except (TypeError, ValueError):
            target = None
    if target is None:
        target = threads.get_thread_id(threads.PURPOSE_JOBS)
    return target


async def notify_chain_event(
    chain_run: ChainRun,
    kind: ChainEventKind,
) -> int | None:
    """Post a one-shot thin chain status embed (start / completed / failed).

    Lands in the chain's own thread if set, else the JOBS channel.
    Most callers want :func:`notify_chain_terminal` instead — that
    builds a richer embed pulling the final worker's report. This one
    stays for the rare case where chain state isn't available (e.g., a
    very early failure before the supervisor wrote any checkpoint).
    """
    target = _resolve_chain_target(chain_run)
    if target is None:
        log.error(
            "discord.notify.chain.no_target",
            chain_id=chain_run.chain_id,
            bucket=chain_run.bucket,
        )
        return None

    embed = embeds.build_chain_status_embed(chain_run, kind)
    return await poster.post_embed(target, embed)


async def notify_chain_terminal(
    chain_run: ChainRun,
    state: dict[str, Any] | None,
    kind: ChainEventKind,
) -> int | None:
    """Post the rich terminal embed for a finished chain run.

    ``state`` is the LangGraph checkpoint state dict (as returned by
    :func:`tasque.chains.manager.get_chain_state`). Pulled into the
    embed so the user sees the final worker's report and key produces
    keys, not just chain metadata. ``kind`` should be one of
    ``"completed"`` / ``"failed"`` / ``"stopped"``.

    Routing: posts inside the chain's own thread if one already exists
    (e.g. created by a previous notify on this chain). Otherwise posts
    to the JOBS channel and anchors a fresh thread to the embed so the
    user can reply about the run — same per-job thread pattern as
    :func:`notify_worker_run`. The new thread id is persisted on
    ``ChainRun.thread_id``.

    On a successful post, stamps ``ChainRun.terminal_notified_at`` so
    subsequent watcher ticks (in this process or after a daemon restart)
    don't re-announce the same terminal. The chain status watcher is
    the only intended caller; it owns the gate that decides which
    chains qualify for a terminal post in the first place.
    """
    embed = embeds.build_chain_terminal_embed(chain_run, state, kind)

    cached_thread = _coerce_int(chain_run.thread_id)
    overflow_thread: int | None = None
    if cached_thread is not None:
        # Already have a per-chain thread — post inside it.
        try:
            msg_id = await poster.post_embed(cached_thread, embed)
            overflow_thread = cached_thread
        except Exception:
            log.exception(
                "discord.notify.chain_terminal.thread_post_failed",
                chain_id=chain_run.chain_id,
                thread_id=cached_thread,
            )
            return None
    else:
        parent_channel_id = threads.get_thread_id(threads.PURPOSE_JOBS)
        if parent_channel_id is None:
            log.error(
                "discord.notify.chain_terminal.no_target",
                chain_id=chain_run.chain_id,
                bucket=chain_run.bucket,
            )
            return None
        msg_id = await poster.post_embed(parent_channel_id, embed)
        try:
            new_thread_id = await poster.start_thread(
                parent_channel_id, msg_id, _chain_run_thread_name(chain_run)
            )
            _persist_chain_run_thread(chain_run.id, new_thread_id)
            overflow_thread = new_thread_id
        except Exception:
            # The embed is posted; losing the thread is annoying but
            # not fatal. Log and continue — terminal_notified_at still
            # gets stamped below so we don't re-announce.
            log.exception(
                "discord.notify.chain_terminal.start_thread_failed",
                chain_id=chain_run.chain_id,
                channel_id=parent_channel_id,
            )

    # Embed carries the summary; thread body carries the full final-step
    # report. Same picking logic as the embed builder so the body
    # corresponds to the summary the user just saw.
    final_report = _final_step_report(state)
    await _post_full_report_to_thread(overflow_thread, final_report)

    with get_session() as sess:
        row = sess.get(ChainRun, chain_run.id)
        if row is not None:
            row.terminal_notified_at = utc_now_iso()
            row.updated_at = utc_now_iso()
    return msg_id


# ----------------------------------------------------------------- DLQ

def _persist_failed_job_notified_at(failed_job_id: str) -> None:
    """Stamp ``FailedJob.notified_at`` after a successful DLQ post.

    Persistent across restart so the watcher's first tick after boot
    doesn't re-announce a failure that landed before shutdown.
    """
    with get_session() as sess:
        row = sess.get(FailedJob, failed_job_id)
        if row is None:
            return
        row.notified_at = utc_now_iso()
        row.updated_at = utc_now_iso()


async def notify_failed_job(
    failed_job: FailedJob,
    *,
    retry_view: object | None = None,
) -> int | None:
    """Post a DLQ entry to the DLQ channel.

    The embed carries Retry + Resolve buttons (via ``retry_view``).
    Unlike worker-run / chain-terminal notifications, DLQ entries are
    status-only: no per-entry thread is created — there's no
    conversation to anchor. The registry entry under
    :data:`tasque.discord.threads.PURPOSE_DLQ` is the parent channel.
    On a successful post, stamps ``FailedJob.notified_at`` so the DLQ
    watcher won't re-announce. Returns the embed message id, or None
    if the DLQ channel isn't configured.
    """
    embed = embeds.build_failed_job_embed(failed_job)

    parent_channel_id = threads.get_thread_id(threads.PURPOSE_DLQ)
    if parent_channel_id is None:
        log.warning(
            "discord.notify.dlq.no_channel", failed_job_id=failed_job.id
        )
        return None

    message_id = await poster.post_embed(
        parent_channel_id, embed, view=retry_view
    )
    _persist_failed_job_notified_at(failed_job.id)
    return message_id


__all__ = [
    "ChainEventKind",
    "notify_chain_event",
    "notify_chain_terminal",
    "notify_failed_job",
    "notify_worker_run",
]

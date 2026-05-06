"""APScheduler poll loop that claims and runs queued jobs serially.

Architecture (single global drain):

    BackgroundScheduler ── tick every 5s ──▶ claim_and_run_one()
                                              │
                                              ├─ sweep_stuck_jobs()  (heartbeat > 5min → fail)
                                              ├─ claim_one()         (oldest pending whose fire_at is ready)
                                              ├─ start heartbeat thread (30s pump)
                                              ├─ run_worker(job)
                                              ├─ on success:
                                              │    mark_completed → schedule_next_if_recurring
                                              └─ on failure: dlq.record_failure(job, exc/error)

Key invariants:

- **Serial execution.** A single module-level lock (``_run_lock``) gates
  ``claim_and_run_one`` so at most one job is in flight, even if APScheduler
  fires overlapping ticks. If another tick is already running, the new
  tick is a no-op.
- **Heartbeat-tracked stuckness.** A claimed job's heartbeat is bumped
  every 30s by a daemon thread. ``sweep_stuck_jobs`` detects orphans
  whose heartbeat is older than 5 minutes (e.g. the worker crashed
  without flipping status) and routes them to the DLQ.
- **Recurrence inserts a new row, regardless of outcome.** Every recurring
  job — succeeded, failed, or runner-raised — spawns a fresh pending
  QueuedJob with the next fire-at before any failure bookkeeping runs.
  A single bad fire goes into the DLQ but the cron series keeps firing.
  The completed/failed row is left intact for history/auditing.
- **No implicit continuation.** Standalone worker completion is terminal.
  The worker-run watcher reports the result; follow-up work should come
  from an explicit user reply, Aim/Signal, recurrence, or chain step.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from tasque.jobs.cron import next_fire_at, to_iso
from tasque.jobs.dlq import record_failure
from tasque.jobs.runner import WorkerResult, run_worker
from tasque.memory.db import get_session
from tasque.memory.entities import QueuedJob, utc_now_iso
from tasque.memory.repo import bump_job_heartbeat, write_entity

log = structlog.get_logger(__name__)

DEFAULT_TICK_SECONDS = 5
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 300

# Single-process serial-execution lock — see module docstring.
_run_lock = threading.Lock()


WorkerRunner = Callable[[QueuedJob], WorkerResult]
CoachTrigger = Callable[[str, str], Any]


# ----------------------------------------------------------------- helpers

def _now_iso() -> str:
    return utc_now_iso()


def _default_worker_runner(job: QueuedJob) -> WorkerResult:
    return run_worker(job)


# ----------------------------------------------------------------- claiming

def claim_one() -> QueuedJob | None:
    """Atomically claim one pending QueuedJob whose ``fire_at`` is ready.

    Sets ``status="claimed"``, ``claimed_at=now``, ``last_heartbeat=now``.
    Returns the detached job, or ``None`` if no eligible row exists.
    """
    now_iso = _now_iso()
    with get_session() as sess:
        stmt = (
            select(QueuedJob)
            .where(QueuedJob.status == "pending")
            .where((QueuedJob.fire_at == "now") | (QueuedJob.fire_at <= now_iso))
            .order_by(QueuedJob.fire_at.asc(), QueuedJob.created_at.asc())
            .limit(1)
        )
        row = sess.execute(stmt).scalars().first()
        if row is None:
            return None
        row.status = "claimed"
        row.claimed_at = now_iso
        row.last_heartbeat = now_iso
        row.updated_at = now_iso
        sess.flush()
        sess.expunge(row)
        log.info(
            "jobs.scheduler.claimed",
            job_id=row.id[:8],
            bucket=row.bucket,
            kind=row.kind,
            directive=(row.directive or "").splitlines()[0][:80],
            chain_id=row.chain_id[:8] if row.chain_id else None,
            chain_step_id=row.chain_step_id,
            recurring=bool(row.recurrence),
        )
        return row


def _mark_completed(job_id: str) -> None:
    with get_session() as sess:
        row = sess.get(QueuedJob, job_id)
        if row is None:
            return
        row.status = "completed"
        row.updated_at = _now_iso()
        log.info(
            "jobs.scheduler.completed",
            job_id=job_id[:8],
            bucket=row.bucket,
            directive=(row.directive or "").splitlines()[0][:80],
        )


def _persist_worker_result(
    job_id: str,
    *,
    summary: str,
    report: str,
    produces: dict[str, Any] | None,
    error: str | None,
) -> None:
    """Cache the WorkerResult fields onto the QueuedJob row.

    Run BEFORE ``_mark_completed`` / ``record_failure`` so the worker-run
    watcher (which polls for ``status IN ('completed', 'failed')`` AND
    ``notified_at IS NULL``) finds these fields present the moment it
    notices the status flip. JSON-serialize ``produces`` because SQLite
    can't natively store it as a Python dict on this column.
    """
    try:
        produces_json = json.dumps(produces or {}, default=str)
    except (TypeError, ValueError):
        # Defensive: a non-serializable value shouldn't break the
        # scheduler. Fall back to a repr the watcher can at least show.
        produces_json = json.dumps({"_repr": repr(produces)})
    with get_session() as sess:
        row = sess.get(QueuedJob, job_id)
        if row is None:
            return
        row.last_summary = summary
        row.last_report = report
        row.last_produces_json = produces_json
        row.last_error = error
        row.updated_at = _now_iso()


# ----------------------------------------------------------------- heartbeat

def _pump_heartbeat(job_id: str, stop: threading.Event, interval: float) -> None:
    """Bump ``QueuedJob.last_heartbeat`` every ``interval`` seconds until stop fires.

    ``stop.wait(interval)`` returns True when the event is set; that's
    how the runner signals "job is done, stop pumping". The initial
    heartbeat is recorded by ``claim_one`` already, so this loop only
    needs to handle subsequent bumps.
    """
    while not stop.wait(interval):
        bump_job_heartbeat(job_id, _now_iso())


# ----------------------------------------------------------------- sweep

def sweep_stuck_jobs(
    timeout_seconds: int = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
) -> int:
    """Flip claimed jobs with stale heartbeats to ``failed``. Returns count.

    A "stale heartbeat" is one older than ``timeout_seconds`` ago. The
    job's worker thread may still be alive — we just stop trusting it.
    Each stuck job becomes a ``FailedJob`` with reason "stuck (no
    heartbeat)" via :func:`tasque.jobs.dlq.record_failure`.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=timeout_seconds)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    detached: list[QueuedJob] = []
    with get_session() as sess:
        stmt = select(QueuedJob).where(QueuedJob.status == "claimed")
        rows = list(sess.execute(stmt).scalars().all())
        for row in rows:
            hb = row.last_heartbeat or row.claimed_at
            if hb is None or hb >= cutoff_iso:
                continue
            sess.expunge(row)
            detached.append(row)

    for job in detached:
        record_failure(
            job,
            error_message="stuck (no heartbeat)",
            error_type="HeartbeatTimeout",
            original_trigger="scheduler",
        )
    return len(detached)


# ----------------------------------------------------------------- recurrence

def _schedule_next_if_recurring(job: QueuedJob) -> str | None:
    """If the job has a cron expression, insert a fresh pending row at the
    next firing time. Returns the new job id or ``None`` for non-recurring."""
    if not job.recurrence:
        return None
    try:
        nxt = next_fire_at(job.recurrence)
    except ValueError as exc:
        log.error(
            "jobs.scheduler.bad_recurrence",
            job_id=job.id,
            expr=job.recurrence,
            error=str(exc),
        )
        return None

    new_job = QueuedJob(
        kind=job.kind,
        bucket=job.bucket,
        directive=job.directive,
        reason=job.reason,
        fire_at=to_iso(nxt),
        status="pending",
        recurrence=job.recurrence,
        queued_by=job.queued_by,
        visible=job.visible,
        thread_id=job.thread_id,
        chain_id=None,
        chain_step_id=None,
        timeout_seconds=job.timeout_seconds,
        # Carry the parent's tier forward — every future firing of this
        # cron series will run at the same model tier. If the parent has
        # no tier (data corruption), the runner will refuse to run the
        # row and surface the error to the DLQ.
        tier=job.tier,
    )
    written = write_entity(new_job)
    return written.id


# ----------------------------------------------------------------- main tick

def claim_and_run_one(
    *,
    runner: WorkerRunner | None = None,
    coach_trigger: CoachTrigger | None = None,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    heartbeat_timeout: int = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
) -> str | None:
    """One scheduler tick: sweep stuck → claim one → run → schedule next.

    Returns the QueuedJob id that was processed, or ``None`` if nothing
    was eligible (or another tick is still in flight). Tests pass
    ``runner`` / ``coach_trigger`` to inject fakes.
    """
    if not _run_lock.acquire(blocking=False):
        # Another tick is still running — APScheduler's ``coalesce=True``
        # on the periodic job already prevents stacking, but we belt-and-
        # brace here in case of overlapping manual ticks.
        return None
    try:
        try:
            sweep_stuck_jobs(timeout_seconds=heartbeat_timeout)
        except Exception:
            log.exception("jobs.scheduler.sweep_failed")

        job = claim_one()
        if job is None:
            return None

        actual_runner = runner or _default_worker_runner
        actual_coach = coach_trigger

        stop_event = threading.Event()
        hb_thread = threading.Thread(
            target=_pump_heartbeat,
            args=(job.id, stop_event, heartbeat_interval),
            daemon=True,
            name=f"tasque-heartbeat-{job.id[:8]}",
        )
        hb_thread.start()

        runner_exc: BaseException | None = None
        result: WorkerResult | None = None
        try:
            result = actual_runner(job)
        except Exception as exc:
            log.exception("jobs.scheduler.runner_raised", job_id=job.id)
            runner_exc = exc
        finally:
            # Stop the heartbeat *before* doing any post-run bookkeeping
            # so the main thread has exclusive DB access for the rest of
            # the tick. Without this, on shared-connection SQLite (tests)
            # the bump and the post-run writes can collide.
            stop_event.set()
            hb_thread.join(timeout=2.0)

        # Schedule the next recurrence FIRST, regardless of outcome.
        # A single failed fire shouldn't kill the whole cron series — the
        # FailedJob row captures the failure for DLQ review while the
        # successor stays scheduled. Bracketed in its own try so a
        # scheduling bug can't break the rest of the tick.
        try:
            _schedule_next_if_recurring(job)
        except Exception:
            log.exception("jobs.scheduler.recurrence_schedule_failed", job_id=job.id)

        if runner_exc is not None:
            # Stash a synthetic WorkerResult so the worker-run watcher can
            # surface a "Worker run failed" embed with the exception
            # message. report stays empty — there's no real worker output
            # to relay.
            _persist_worker_result(
                job.id,
                summary="",
                report="",
                produces=None,
                error=f"{type(runner_exc).__name__}: {runner_exc}",
            )
            record_failure(job, exc=runner_exc, original_trigger="scheduler")
            return job.id

        assert result is not None  # for type narrowing
        err = result.get("error")
        if err:
            _persist_worker_result(
                job.id,
                summary=result.get("summary", "") or "",
                report=result.get("report", "") or "",
                produces=result.get("produces"),
                error=err,
            )
            record_failure(
                job,
                error_message=err,
                error_type="WorkerError",
                original_trigger="scheduler",
            )
            return job.id

        _persist_worker_result(
            job.id,
            summary=result.get("summary", "") or "",
            report=result.get("report", "") or "",
            produces=result.get("produces"),
            error=None,
        )
        _mark_completed(job.id)

        if job.bucket and actual_coach is not None:
            try:
                actual_coach(job.bucket, job.id)
            except Exception:
                log.exception("jobs.scheduler.coach_trigger_failed", job_id=job.id)
        return job.id
    finally:
        _run_lock.release()


# ----------------------------------------------------------------- lifecycle

def _fire_due_chain_templates_safe() -> None:
    """Wrapper for the chain-template poll callback that swallows exceptions
    so a single bad template doesn't kill the scheduler thread."""
    from tasque.chains.scheduler import fire_due_chain_templates

    try:
        fire_due_chain_templates()
    except Exception:
        log.exception("jobs.scheduler.chain_poll_failed")


def _sweep_nondurable_memory_safe() -> None:
    """Wrapper for the daily decay sweep that swallows exceptions so a
    single bad row doesn't kill the scheduler thread."""
    from tasque.memory.decay import sweep_nondurable_memory

    try:
        report = sweep_nondurable_memory()
    except Exception:
        log.exception("jobs.scheduler.decay_sweep_failed")
        return
    if report.total_archived or report.total_hard_deleted:
        log.info(
            "jobs.scheduler.decay_swept",
            archived_ephemeral_notes=report.archived_ephemeral_notes,
            archived_expired_signals=report.archived_expired_signals,
            archived_superseded_notes=report.archived_superseded_notes,
            hard_deleted_notes=report.hard_deleted_notes,
            hard_deleted_signals=report.hard_deleted_signals,
            hard_deleted_attachments=report.hard_deleted_attachments,
        )


def _reap_agent_results_safe() -> None:
    """Wrapper for the agent-result inbox reaper that swallows exceptions.

    The inbox is short-lived per-run state — agents delete their own
    rows on read. This sweep catches leftovers from a process crash
    between minting a token and reading it back.
    """
    from tasque.agents import result_inbox

    try:
        swept = result_inbox.reap_stale()
    except Exception:
        log.exception("jobs.scheduler.agent_results_reap_failed")
        return
    if swept:
        log.info("jobs.scheduler.agent_results_reaped", rows=swept)


def start_scheduler(*, tick_seconds: int = DEFAULT_TICK_SECONDS) -> BackgroundScheduler:
    """Start a BackgroundScheduler ticking every ``tick_seconds``.

    Returns the scheduler so callers can ``shutdown()`` it. ``coalesce``
    + ``max_instances=1`` ensure that if a tick takes longer than the
    interval, missed ticks collapse into one rather than stacking.
    """
    from tasque.config import get_settings

    settings = get_settings()

    # APScheduler logs a WARNING every tick when a run is skipped because
    # the previous one is still in flight. With our serial-execution lock
    # and 5s tick that fires constantly for any non-trivial job — pure
    # noise. Suppress just this logger; real scheduler errors still surface.
    logging.getLogger("apscheduler.scheduler").setLevel(logging.ERROR)

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        claim_and_run_one,
        trigger="interval",
        seconds=tick_seconds,
        id="tasque-tick",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    scheduler.add_job(
        _fire_due_chain_templates_safe,
        trigger="interval",
        seconds=max(tick_seconds, 30),
        id="tasque-chain-poll",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    decay_hours = max(1, settings.decay_sweep_interval_hours)
    scheduler.add_job(
        _sweep_nondurable_memory_safe,
        trigger="interval",
        hours=decay_hours,
        # Don't wait a full interval for the first sweep — run shortly
        # after startup so a long-stopped daemon catches up immediately.
        next_run_time=datetime.now(UTC) + timedelta(seconds=max(tick_seconds * 2, 30)),
        id="tasque-memory-decay",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    # Agent-result inbox reaper. Rows are normally consumed within the
    # same agent run that wrote them; this catches leftovers from a
    # process crash between minting a token and reading it back. Hourly
    # is well below the default ``DEFAULT_REAP_AGE_SECONDS`` (3600s).
    scheduler.add_job(
        _reap_agent_results_safe,
        trigger="interval",
        hours=1,
        next_run_time=datetime.now(UTC) + timedelta(seconds=max(tick_seconds * 2, 30)),
        id="tasque-agent-results-reap",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    scheduler.start()
    log.info(
        "jobs.scheduler.started",
        tick_seconds=tick_seconds,
        decay_sweep_hours=decay_hours,
    )
    return scheduler

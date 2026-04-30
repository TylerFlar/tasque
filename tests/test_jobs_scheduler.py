"""Tests for ``tasque.jobs.scheduler`` — claim, heartbeat, recurrence, coach hook."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from tasque.jobs import scheduler as scheduler_mod
from tasque.jobs.runner import WorkerResult
from tasque.jobs.scheduler import (
    DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    claim_and_run_one,
    claim_one,
    sweep_stuck_jobs,
)
from tasque.memory.db import get_session
from tasque.memory.entities import CoachPending, FailedJob, QueuedJob, utc_now_iso
from tasque.memory.repo import write_entity


def _ok_runner(produces: dict[str, Any] | None = None) -> Any:
    def _runner(job: QueuedJob) -> WorkerResult:
        return WorkerResult(
            report="r", summary="s", produces=produces or {}, error=None
        )

    return _runner


def _record_runner() -> tuple[list[QueuedJob], Any]:
    log: list[QueuedJob] = []

    def runner(job: QueuedJob) -> WorkerResult:
        log.append(job)
        return WorkerResult(report="r", summary="s", produces={}, error=None)

    return log, runner


def _make_job(**overrides: Any) -> QueuedJob:
    defaults: dict[str, Any] = dict(
        kind="worker",
        bucket="personal",
        directive="d",
        reason="r",
        fire_at="now",
        status="pending",
        queued_by="test",
        tier="haiku",
    )
    defaults.update(overrides)
    return write_entity(QueuedJob(**defaults))


def _record_coach() -> tuple[list[tuple[str, str]], Any]:
    log: list[tuple[str, str]] = []

    def coach(bucket: str, job_id: str) -> Any:
        log.append((bucket, job_id))
        return None

    return log, coach


# ---------------------------------------------------------------- claiming

def test_claim_one_picks_pending_due_job() -> None:
    a = _make_job(directive="first")
    claimed = claim_one()
    assert claimed is not None
    assert claimed.id == a.id
    assert claimed.status == "claimed"
    # Another claim returns None — only one pending was eligible.
    assert claim_one() is None


def test_claim_one_skips_future_fire_at() -> None:
    future = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    _make_job(directive="later", fire_at=future)
    assert claim_one() is None


def test_claim_one_skips_non_pending() -> None:
    _make_job(directive="claimed already", status="claimed")
    _make_job(directive="completed", status="completed")
    _make_job(directive="failed", status="failed")
    _make_job(directive="stopped", status="stopped")
    assert claim_one() is None


def test_claim_and_run_one_holds_serial_lock() -> None:
    """Two parallel ticks: only one runs the job."""
    _make_job(directive="solo")

    barrier = threading.Event()
    inflight = threading.Event()
    runs: list[str] = []

    def slow_runner(job: QueuedJob) -> WorkerResult:
        inflight.set()
        barrier.wait(timeout=2.0)
        runs.append(job.id)
        return WorkerResult(report="", summary="", produces={}, error=None)

    _coach_log, coach = _record_coach()

    results: dict[str, str | None] = {}

    def call_a() -> None:
        results["a"] = claim_and_run_one(runner=slow_runner, coach_trigger=coach)

    def call_b() -> None:
        # Wait until A is actually inflight before B fires.
        inflight.wait(timeout=2.0)
        results["b"] = claim_and_run_one(runner=slow_runner, coach_trigger=coach)

    ta = threading.Thread(target=call_a)
    tb = threading.Thread(target=call_b)
    ta.start()
    tb.start()
    # Let B observe the lock and bail, then release A.
    time.sleep(0.1)
    barrier.set()
    ta.join(timeout=3.0)
    tb.join(timeout=3.0)

    # Exactly one of the two ticks ran the job.
    ran = [v for v in results.values() if v is not None]
    assert len(runs) == 1
    assert len(ran) == 1


# ---------------------------------------------------------------- heartbeat

def test_heartbeat_updates_during_run() -> None:
    job = _make_job(directive="watch heartbeat")

    def slow_runner(j: QueuedJob) -> WorkerResult:
        # Sleep long enough for ≥1 heartbeat tick at 0.05s interval.
        time.sleep(0.25)
        return WorkerResult(report="r", summary="s", produces={}, error=None)

    _coach_log, coach = _record_coach()
    claim_and_run_one(
        runner=slow_runner,
        coach_trigger=coach,
        heartbeat_interval=0.05,
        heartbeat_timeout=DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    )

    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.status == "completed"
        assert row.last_heartbeat is not None
        assert row.claimed_at is not None
        # claim_one sets both timestamps to the same now_iso. The pump
        # thread should have bumped last_heartbeat strictly later.
        assert row.last_heartbeat > row.claimed_at


def test_sweep_stuck_jobs_marks_failed() -> None:
    job = _make_job(directive="orphan")
    # Force the row into a "claimed but stale" state.
    stale = (datetime.now(UTC) - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        row.status = "claimed"
        row.claimed_at = stale
        row.last_heartbeat = stale

    flipped = sweep_stuck_jobs(timeout_seconds=300)
    assert flipped == 1

    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.status == "failed"
        fjs = list(sess.execute(select(FailedJob)).scalars().all())
        assert len(fjs) == 1
        assert "stuck" in fjs[0].error_message
        assert fjs[0].error_type == "HeartbeatTimeout"


def test_sweep_skips_fresh_heartbeat() -> None:
    job = _make_job(directive="alive")
    fresh = utc_now_iso()
    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        row.status = "claimed"
        row.claimed_at = fresh
        row.last_heartbeat = fresh
    assert sweep_stuck_jobs(timeout_seconds=300) == 0


# ---------------------------------------------------------------- recurrence

def test_recurring_completed_job_inserts_next(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASQUE_TIMEZONE", "UTC")
    job = _make_job(directive="every 12h", recurrence="0 */12 * * *", bucket=None)

    _coach_log, coach = _record_coach()
    job_id = claim_and_run_one(
        runner=_ok_runner(),
        coach_trigger=coach,
        heartbeat_interval=0.05,
    )
    assert job_id == job.id

    with get_session() as sess:
        # Original job is completed.
        original = sess.get(QueuedJob, job.id)
        assert original is not None
        assert original.status == "completed"
        # A second pending row exists with the same directive + recurrence.
        rows = list(
            sess.execute(
                select(QueuedJob).where(QueuedJob.status == "pending")
            ).scalars().all()
        )
        assert len(rows) == 1
        nxt = rows[0]
        assert nxt.directive == "every 12h"
        assert nxt.recurrence == "0 */12 * * *"
        # fire_at is an ISO string in the future.
        assert nxt.fire_at != "now"
        assert nxt.fire_at.endswith("Z")


def test_non_recurring_does_not_insert_next() -> None:
    _make_job(directive="one-shot")
    _coach_log, coach = _record_coach()
    claim_and_run_one(runner=_ok_runner(), coach_trigger=coach, heartbeat_interval=0.05)
    with get_session() as sess:
        pendings = list(
            sess.execute(
                select(QueuedJob).where(QueuedJob.status == "pending")
            ).scalars().all()
        )
    assert pendings == []


# ---------------------------------------------------------------- coach hook

def test_success_enqueues_coach_trigger() -> None:
    job = _make_job(directive="bucket job", bucket="health")
    coach_log, coach = _record_coach()
    claim_and_run_one(runner=_ok_runner(), coach_trigger=coach, heartbeat_interval=0.05)
    assert coach_log == [("health", job.id)]


def test_success_with_no_bucket_skips_coach_trigger() -> None:
    _make_job(directive="bucketless", bucket=None)
    coach_log, coach = _record_coach()
    claim_and_run_one(runner=_ok_runner(), coach_trigger=coach, heartbeat_interval=0.05)
    assert coach_log == []


def test_default_coach_trigger_writes_coach_pending_row() -> None:
    job = _make_job(directive="real coach hook", bucket="career")
    # No coach_trigger passed → use default, which calls coach.trigger.enqueue.
    claim_and_run_one(runner=_ok_runner(), heartbeat_interval=0.05)
    with get_session() as sess:
        rows = list(sess.execute(select(CoachPending)).scalars().all())
    assert len(rows) == 1
    assert rows[0].bucket == "career"
    assert rows[0].dedup_key == f"job:{job.id}"
    assert rows[0].reason == f"job-completed:{job.id}"


# ---------------------------------------------------------------- failures

def test_runner_exception_routes_to_dlq() -> None:
    job = _make_job(directive="boom")

    def angry_runner(j: QueuedJob) -> WorkerResult:
        raise RuntimeError("kaboom")

    coach_log, coach = _record_coach()
    claim_and_run_one(runner=angry_runner, coach_trigger=coach, heartbeat_interval=0.05)
    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.status == "failed"
        fjs = list(sess.execute(select(FailedJob)).scalars().all())
    assert len(fjs) == 1
    assert fjs[0].error_type == "RuntimeError"
    assert "kaboom" in fjs[0].error_message
    # No coach trigger on failure.
    assert coach_log == []


def test_worker_error_result_routes_to_dlq() -> None:
    job = _make_job(directive="bad json")

    def malformed_runner(j: QueuedJob) -> WorkerResult:
        return WorkerResult(report="", summary="", produces={}, error="bad LLM output")

    claim_and_run_one(runner=malformed_runner, heartbeat_interval=0.05)
    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.status == "failed"
        fjs = list(sess.execute(select(FailedJob)).scalars().all())
    assert len(fjs) == 1
    assert fjs[0].error_type == "WorkerError"
    assert "bad LLM output" in fjs[0].error_message


def test_runner_exception_still_schedules_next_recurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single bad fire shouldn't kill the cron series — the failed
    row goes to the DLQ but the next firing must still be queued."""
    monkeypatch.setenv("TASQUE_TIMEZONE", "UTC")
    _make_job(directive="recurring fail", recurrence="0 */12 * * *", bucket="career")

    def angry_runner(j: QueuedJob) -> WorkerResult:
        raise RuntimeError("nope")

    claim_and_run_one(runner=angry_runner, heartbeat_interval=0.05)
    with get_session() as sess:
        pendings = list(
            sess.execute(
                select(QueuedJob).where(QueuedJob.status == "pending")
            ).scalars().all()
        )
        failures = list(sess.execute(select(FailedJob)).scalars().all())
    # Failure recorded in DLQ.
    assert len(failures) == 1
    assert failures[0].error_type == "RuntimeError"
    # And the cron series continues.
    assert len(pendings) == 1
    assert pendings[0].directive == "recurring fail"
    assert pendings[0].recurrence == "0 */12 * * *"


def test_worker_error_still_schedules_next_recurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as above but for the LLM-returned-error path (WorkerResult
    with a non-empty ``error`` field)."""
    monkeypatch.setenv("TASQUE_TIMEZONE", "UTC")
    _make_job(directive="recurring err", recurrence="0 */6 * * *", bucket="health")

    def err_runner(j: QueuedJob) -> WorkerResult:
        return WorkerResult(report="", summary="", produces={}, error="LLM bad output")

    claim_and_run_one(runner=err_runner, heartbeat_interval=0.05)
    with get_session() as sess:
        pendings = list(
            sess.execute(
                select(QueuedJob).where(QueuedJob.status == "pending")
            ).scalars().all()
        )
        failures = list(sess.execute(select(FailedJob)).scalars().all())
    assert len(failures) == 1
    assert failures[0].error_type == "WorkerError"
    assert len(pendings) == 1
    assert pendings[0].directive == "recurring err"
    assert pendings[0].recurrence == "0 */6 * * *"


def test_non_recurring_failure_does_not_create_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-shot job that fails should NOT spawn a pending row — only
    cron-bearing failures get auto-rescheduled."""
    monkeypatch.setenv("TASQUE_TIMEZONE", "UTC")
    _make_job(directive="one shot fail", recurrence=None, bucket="career")

    def angry_runner(j: QueuedJob) -> WorkerResult:
        raise RuntimeError("nope")

    claim_and_run_one(runner=angry_runner, heartbeat_interval=0.05)
    with get_session() as sess:
        pendings = list(
            sess.execute(
                select(QueuedJob).where(QueuedJob.status == "pending")
            ).scalars().all()
        )
    assert pendings == []


# ---------------------------------------------------------------- empty tick

def test_tick_with_no_pending_returns_none() -> None:
    assert claim_and_run_one() is None


# ---------------------------------------------------------------- start_scheduler

def test_start_scheduler_starts_and_shuts_down() -> None:
    sched = scheduler_mod.start_scheduler(tick_seconds=60)
    try:
        assert sched.running is True
        # The agent-result inbox reaper must be wired in alongside the
        # other periodic jobs — without it, leaked rows from crashed
        # agent runs accumulate forever.
        job_ids = {j.id for j in sched.get_jobs()}
        assert "tasque-agent-results-reap" in job_ids
    finally:
        sched.shutdown(wait=False)


def test_reap_agent_results_safe_invokes_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper used by APScheduler must call result_inbox.reap_stale
    and never let an exception propagate up into the scheduler thread."""
    from tasque.agents import result_inbox

    calls: list[None] = []

    def _fake_reap(*, max_age_seconds: int = result_inbox.DEFAULT_REAP_AGE_SECONDS) -> int:
        calls.append(None)
        return 0

    monkeypatch.setattr(result_inbox, "reap_stale", _fake_reap)
    scheduler_mod._reap_agent_results_safe()
    assert calls == [None]


def test_reap_agent_results_safe_swallows_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tasque.agents import result_inbox

    def _boom(*, max_age_seconds: int = result_inbox.DEFAULT_REAP_AGE_SECONDS) -> int:
        raise RuntimeError("db went away")

    monkeypatch.setattr(result_inbox, "reap_stale", _boom)
    # Must not raise — APScheduler thread would otherwise die silently.
    scheduler_mod._reap_agent_results_safe()

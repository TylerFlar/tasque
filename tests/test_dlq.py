"""Tests for ``tasque.jobs.dlq`` — record failures and retry."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from tasque.jobs.dlq import (
    list_unresolved,
    mark_resolved,
    record_failure,
    retry,
)
from tasque.memory.db import get_session
from tasque.memory.entities import FailedJob, QueuedJob
from tasque.memory.repo import write_entity


def _make_job(**overrides: Any) -> QueuedJob:
    defaults: dict[str, Any] = dict(
        kind="worker",
        bucket="personal",
        directive="boom",
        reason="why",
        fire_at="now",
        status="claimed",
        queued_by="test",
    )
    defaults.update(overrides)
    return write_entity(QueuedJob(**defaults))


# ----------------------------------------------------------- record_failure

def test_record_failure_from_exception_captures_traceback() -> None:
    job = _make_job(directive="errors")
    try:
        raise ValueError("kaboom")
    except ValueError as exc:
        fj = record_failure(job, exc=exc, original_trigger="scheduler")
    assert fj.error_type == "ValueError"
    assert "kaboom" in fj.error_message
    assert "ValueError" in fj.traceback
    assert fj.original_trigger == "scheduler"
    assert fj.job_id == job.id

    # And the QueuedJob is now status=failed.
    with get_session() as sess:
        refreshed = sess.get(QueuedJob, job.id)
        assert refreshed is not None
        assert refreshed.status == "failed"


def test_record_failure_from_message_only() -> None:
    job = _make_job(directive="explicit message")
    fj = record_failure(
        job,
        error_message="bad LLM output",
        error_type="WorkerError",
        original_trigger="scheduler",
    )
    assert fj.error_type == "WorkerError"
    assert fj.error_message == "bad LLM output"
    assert fj.traceback == ""


def test_record_failure_requires_exc_or_message_pair() -> None:
    job = _make_job()
    with pytest.raises(ValueError):
        record_failure(job)


def test_record_failure_propagates_chain_metadata() -> None:
    job = _make_job(
        directive="chain step", chain_id="chain-1", chain_step_id="step-9"
    )
    fj = record_failure(job, error_message="x", error_type="X", original_trigger="chain step")
    assert fj.chain_id == "chain-1"
    assert fj.plan_node_id == "step-9"


# ----------------------------------------------------------- list / resolve

def test_list_unresolved_orders_newest_first() -> None:
    j1 = _make_job(directive="first")
    fj1 = record_failure(j1, error_message="e1", error_type="E", original_trigger="scheduler")
    j2 = _make_job(directive="second")
    fj2 = record_failure(j2, error_message="e2", error_type="E", original_trigger="scheduler")

    rows = list_unresolved(limit=10)
    ids = [r.id for r in rows]
    # Newest-first
    assert ids[0] == fj2.id
    assert ids[1] == fj1.id


def test_mark_resolved_excludes_from_unresolved() -> None:
    j = _make_job()
    fj = record_failure(j, error_message="e", error_type="E", original_trigger="scheduler")
    assert mark_resolved(fj.id) is True
    rows = list_unresolved()
    assert all(r.id != fj.id for r in rows)


# ----------------------------------------------------------- retry standalone

def test_retry_standalone_inserts_fresh_pending_job() -> None:
    j = _make_job(
        directive="please retry me",
        bucket="health",
        reason="initial",
        recurrence=None,
    )
    fj = record_failure(j, error_message="e", error_type="E", original_trigger="scheduler")

    report = retry(fj.id)
    assert report["kind"] == "standalone"
    new_id = report["new_job_id"]
    assert isinstance(new_id, str) and new_id != j.id
    assert report["retry_count"] == 1

    with get_session() as sess:
        new_job = sess.get(QueuedJob, new_id)
        assert new_job is not None
        assert new_job.directive == "please retry me"
        assert new_job.bucket == "health"
        assert new_job.reason == "initial"
        assert new_job.fire_at == "now"
        assert new_job.status == "pending"
        assert new_job.queued_by == "dlq"
        assert new_job.chain_id is None

        # FailedJob retry_count was bumped.
        refreshed = sess.get(FailedJob, fj.id)
        assert refreshed is not None
        assert refreshed.retry_count == 1


def test_retry_standalone_preserves_recurrence() -> None:
    j = _make_job(directive="recurring", recurrence="0 9 * * MON-FRI")
    fj = record_failure(j, error_message="e", error_type="E", original_trigger="scheduler")
    report = retry(fj.id)
    with get_session() as sess:
        new_job = sess.get(QueuedJob, report["new_job_id"])
        assert new_job is not None
        assert new_job.recurrence == "0 9 * * MON-FRI"


def test_retry_unknown_id_raises() -> None:
    with pytest.raises(ValueError):
        retry("nope")


# ----------------------------------------------------------- retry chain step

def test_retry_chain_step_calls_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    j = _make_job(
        directive="chain step",
        chain_id="chain-abc",
        chain_step_id="step-42",
    )
    fj = record_failure(j, error_message="step boom", error_type="X", original_trigger="chain step")

    calls: list[tuple[str, str]] = []

    def fake_hook(chain_id: str, step_id: str) -> None:
        calls.append((chain_id, step_id))

    from tasque import chains
    monkeypatch.setattr(chains, "dlq_retry", fake_hook)

    report = retry(fj.id)
    assert report["kind"] == "chain-step"
    assert report["chain_id"] == "chain-abc"
    assert report["plan_node_id"] == "step-42"
    assert report["retry_count"] == 1
    assert calls == [("chain-abc", "step-42")]

    # No new QueuedJob inserted — the hook owns rescheduling.
    with get_session() as sess:
        rows = list(
            sess.execute(
                select(QueuedJob).where(QueuedJob.queued_by == "dlq")
            ).scalars().all()
        )
    assert rows == []


def test_retry_chain_step_with_no_checkpoint_returns_false() -> None:
    """``chains.dlq_retry`` returns False when the checkpoint is missing,
    rather than raising. The DLQ retry path still bumps ``retry_count``
    because the hook ran cleanly."""
    j = _make_job(
        directive="chain step",
        chain_id="chain-y",
        chain_step_id="step-3",
    )
    fj = record_failure(j, error_message="x", error_type="X", original_trigger="chain step")
    report = retry(fj.id)
    assert report["kind"] == "chain-step"
    assert report["chain_id"] == "chain-y"

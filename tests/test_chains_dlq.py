"""Test the chain DLQ retry path.

A failed step → ``dlq_retry`` → step re-runs and the chain proceeds. The
None-on-right reducer is the load-bearing piece — without it, the
supervisor would still see the old failure on the next pass.
"""

from __future__ import annotations

from typing import Any

import pytest

from tasque.chains import dlq_retry
from tasque.chains.manager import get_chain_state
from tasque.chains.scheduler import launch_chain_run
from tasque.jobs.runner import WorkerResult


def _spec_two_workers_then_approval() -> dict[str, Any]:
    return {
        "chain_name": "dlq-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "opus",
        "plan": [
            {"id": "scan", "kind": "worker", "directive": "scan", "tier": "haiku"},
            {
                "id": "filter",
                "kind": "worker",
                "directive": "filter",
                "depends_on": ["scan"],
                "consumes": ["scan"],
                "tier": "haiku",
            },
        ],
    }


def test_dlq_retry_clears_failure_and_step_runs_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Counter so we can fail the first invocation and succeed the second.
    state = {"filter_calls": 0}

    def _fake(job: Any, *, consumes: dict[str, Any] | None = None, **_: Any) -> WorkerResult:
        if job.chain_step_id == "scan":
            return WorkerResult(report="ok", summary="ok", produces={}, error=None)
        if job.chain_step_id == "filter":
            state["filter_calls"] += 1
            if state["filter_calls"] == 1:
                return WorkerResult(
                    report="",
                    summary="",
                    produces={},
                    error="filter failed first time",
                )
            return WorkerResult(
                report="ok 2nd time",
                summary="ok",
                produces={"x": "y"},
                error=None,
            )
        return WorkerResult(report="ok", summary="ok", produces={}, error=None)

    import tasque.chains.graph.worker as worker_mod
    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    chain_id = launch_chain_run(_spec_two_workers_then_approval())
    snap = get_chain_state(chain_id)
    assert snap is not None
    filter_node = next(n for n in snap["plan"] if n["id"] == "filter")
    assert filter_node["status"] == "failed"
    assert "filter" in snap["failures"]

    # Now retry the failed step.
    assert dlq_retry(chain_id, "filter") is True

    snap2 = get_chain_state(chain_id)
    assert snap2 is not None
    filter_after = next(n for n in snap2["plan"] if n["id"] == "filter")
    assert filter_after["status"] == "completed"
    # The None-on-right reducer should have cleared the failure.
    assert "filter" not in snap2["failures"]
    # Worker fake was called twice for filter (initial + retry).
    assert state["filter_calls"] == 2


def test_dlq_retry_returns_false_for_unknown_chain() -> None:
    assert dlq_retry("does-not-exist", "step") is False


def test_dlq_retry_returns_false_for_step_not_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake(job: Any, *, consumes: dict[str, Any] | None = None, **_: Any) -> WorkerResult:
        return WorkerResult(report="ok", summary="ok", produces={}, error=None)

    import tasque.chains.graph.worker as worker_mod
    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    spec = {
        "chain_name": "ok-chain",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "opus",
        "plan": [
            {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
            {"id": "b", "kind": "worker", "directive": "y", "depends_on": ["a"], "consumes": ["a"], "tier": "haiku"},
        ],
    }
    chain_id = launch_chain_run(spec)
    # Both completed successfully; retrying a completed step is a no-op.
    assert dlq_retry(chain_id, "a") is False


def test_dlq_retry_via_jobs_dlq_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a chain step's failure is recorded as a FailedJob,
    ``tasque.jobs.dlq.retry`` is called, which calls ``chains.dlq_retry``.
    """
    counter = {"calls": 0}

    def _fake(job: Any, *, consumes: dict[str, Any] | None = None, **_: Any) -> WorkerResult:
        counter["calls"] += 1
        if counter["calls"] == 1:
            return WorkerResult(report="", summary="", produces={}, error="first try fails")
        return WorkerResult(report="ok", summary="ok", produces={}, error=None)

    import tasque.chains.graph.worker as worker_mod
    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    spec = {
        "chain_name": "dlq-via-job",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "opus",
        "plan": [
            {"id": "only", "kind": "worker", "directive": "x", "tier": "haiku"},
        ],
    }
    chain_id = launch_chain_run(spec)

    # Now record a synthetic FailedJob with chain_id/plan_node_id and
    # call jobs.dlq.retry.
    from tasque.jobs.dlq import record_failure
    from tasque.jobs.dlq import retry as jobs_retry
    from tasque.memory.entities import QueuedJob
    from tasque.memory.repo import write_entity

    qj = write_entity(
        QueuedJob(
            kind="worker",
            bucket="personal",
            directive="x",
            reason="chain step",
            fire_at="now",
            status="failed",
            queued_by="chain",
            chain_id=chain_id,
            chain_step_id="only",
        )
    )
    fj = record_failure(qj, error_message="first try fails", error_type="WorkerError")
    report = jobs_retry(fj.id)
    assert report["kind"] == "chain-step"
    assert report["chain_id"] == chain_id

    snap = get_chain_state(chain_id)
    assert snap is not None
    only = next(n for n in snap["plan"] if n["id"] == "only")
    assert only["status"] == "completed"


def test_resume_interrupted_chains_retries_failed_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain wedged on a failed fan-out child must retry the child on
    the next ``resume_interrupted_chains`` pass — the regression that
    caused weekly-financial-sweep to sit in ``running`` forever after
    pull[1]/pull[2] hit transient APIConnectionError.
    """
    pull_calls = {"count": 0}

    def _fake(job: Any, *, consumes: dict[str, Any] | None = None, **_: Any) -> WorkerResult:
        if job.chain_step_id == "sources":
            return WorkerResult(
                report="ok",
                summary="ok",
                produces={"items": ["a", "b"]},
                error=None,
            )
        if job.chain_step_id and job.chain_step_id.startswith("pull["):
            pull_calls["count"] += 1
            # First two invocations (initial pull[0] + pull[1]) fail.
            if pull_calls["count"] <= 2:
                return WorkerResult(
                    report="",
                    summary="",
                    produces={},
                    error="LLM call failed: APIConnectionError: Connection error.",
                )
            return WorkerResult(
                report="ok",
                summary="ok",
                produces={"x": job.chain_step_id},
                error=None,
            )
        if job.chain_step_id == "consolidate":
            return WorkerResult(report="done", summary="done", produces={}, error=None)
        return WorkerResult(report="ok", summary="ok", produces={}, error=None)

    import tasque.chains.graph.worker as worker_mod

    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    spec: dict[str, Any] = {
        "chain_name": "wedge-on-fanout",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "opus",
        "plan": [
            {"id": "sources", "kind": "worker", "directive": "list", "tier": "haiku"},
            {
                "id": "pull",
                "kind": "worker",
                "directive": "pull one item",
                "depends_on": ["sources"],
                "consumes": ["sources"],
                "fan_out_on": "items",
                "tier": "haiku",
            },
            {
                "id": "consolidate",
                "kind": "worker",
                "directive": "roll up",
                "depends_on": ["pull"],
                "consumes": ["pull"],
                "tier": "haiku",
            },
        ],
    }

    from tasque.chains.scheduler import launch_chain_run, resume_interrupted_chains

    chain_id = launch_chain_run(spec)

    # Chain is wedged: both fan-out children failed, consolidate is pending,
    # ChainRun row is still 'running'.
    snap = get_chain_state(chain_id)
    assert snap is not None
    statuses = {n["id"]: n["status"] for n in snap["plan"]}
    assert statuses["pull[0]"] == "failed"
    assert statuses["pull[1]"] == "failed"
    assert statuses["consolidate"] == "pending"
    assert "pull[0]" in snap["failures"]

    from sqlalchemy import select

    from tasque.memory.db import get_session
    from tasque.memory.entities import ChainRun

    with get_session() as sess:
        row = sess.execute(
            select(ChainRun).where(ChainRun.chain_id == chain_id)
        ).scalars().one()
        assert row.status == "running"

    # Simulate a bot restart.
    resumed = resume_interrupted_chains()
    assert chain_id in resumed

    # Both pulls should have retried, succeeded, and consolidate should
    # have run. ChainRun row should have flipped to completed.
    snap2 = get_chain_state(chain_id)
    assert snap2 is not None
    statuses2 = {n["id"]: n["status"] for n in snap2["plan"]}
    assert statuses2["pull[0]"] == "completed"
    assert statuses2["pull[1]"] == "completed"
    assert statuses2["consolidate"] == "completed"
    assert "pull[0]" not in snap2["failures"]
    assert "pull[1]" not in snap2["failures"]

    with get_session() as sess:
        row2 = sess.execute(
            select(ChainRun).where(ChainRun.chain_id == chain_id)
        ).scalars().one()
        assert row2.status == "completed"

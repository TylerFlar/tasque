"""End-to-end chain run with a mocked worker function.

The worker function is monkeypatched to return canned WorkerResult dicts
so we don't go anywhere near a real LLM. The test runs through the
3-step shape (worker → worker → approval-interrupt) and resumes via
``Command(resume=...)``."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from tasque.chains.graph import get_compiled_chain_graph
from tasque.chains.scheduler import launch_chain_run
from tasque.jobs.runner import WorkerResult
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun


@pytest.fixture
def fake_worker_runner(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Monkeypatch ``tasque.chains.graph.worker.run_worker`` with a canned
    function and a recording dict the test can read."""
    record: dict[str, Any] = {"calls": []}

    def _fake(job: Any, *, consumes: dict[str, Any] | None = None, **_: Any) -> WorkerResult:
        record["calls"].append({
            "directive": job.directive,
            "chain_id": job.chain_id,
            "chain_step_id": job.chain_step_id,
            "consumes": dict(consumes or {}),
        })
        # Step-specific synthetic output.
        if job.chain_step_id == "scan":
            return WorkerResult(
                report="scanned ok",
                summary="scan summary",
                produces={"items": ["topic1", "topic2"]},
                error=None,
            )
        if job.chain_step_id and job.chain_step_id.startswith("filter["):
            idx = job.chain_step_id.split("[")[1][:-1]
            return WorkerResult(
                report=f"filter framing for {idx}",
                summary=f"filter[{idx}]",
                produces={"framing": f"f-{idx}"},
                error=None,
            )
        return WorkerResult(report="ok", summary="ok", produces={}, error=None)

    import tasque.chains.graph.worker as worker_mod
    monkeypatch.setattr(worker_mod, "run_worker", _fake)
    return record


def _3_step_spec() -> dict[str, Any]:
    return {
        "chain_name": "e2e-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {"id": "scan", "kind": "worker", "directive": "scan", "tier": "small"},
            {
                "id": "filter",
                "kind": "worker",
                "directive": "filter",
                "depends_on": ["scan"],
                "consumes": ["scan"],
                "fan_out_on": "items",
                "tier": "small",
            },
            {
                "id": "notify",
                "kind": "approval",
                "directive": "approve",
                "depends_on": ["filter"],
                "consumes": ["filter"],
            },
        ],
    }


def test_three_step_chain_runs_and_pauses_at_approval(fake_worker_runner: dict[str, Any]) -> None:
    chain_id = launch_chain_run(_3_step_spec())

    # The two filter children should have run.
    invoked_steps = {c["chain_step_id"] for c in fake_worker_runner["calls"]}
    assert "scan" in invoked_steps
    assert "filter[0]" in invoked_steps
    assert "filter[1]" in invoked_steps

    # The chain must have paused at the approval interrupt.
    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        assert row["status"] == "awaiting_approval"

    # And the notify approval should be in awaiting state per the
    # checkpoint (the snapshot reports "running" at the moment of dispatch
    # but the supervisor will re-resolve it on resume).
    from tasque.chains.manager import get_chain_state

    state = get_chain_state(chain_id)
    assert state is not None
    notify = next(n for n in state["plan"] if n["id"] == "notify")
    # Either running (just dispatched) or awaiting_user — both are
    # acceptable mid-interrupt states. What matters is that notify hasn't
    # completed yet.
    assert notify["status"] in ("running", "awaiting_user")
    assert "notify" not in state.get("completed", {})


def test_three_step_chain_resumes_from_approval(
    fake_worker_runner: dict[str, Any],
) -> None:
    chain_id = launch_chain_run(_3_step_spec())

    graph = get_compiled_chain_graph()
    cfg = RunnableConfig(configurable={"thread_id": chain_id})

    # Resume the interrupt with a user reply.
    graph.invoke(Command(resume="approved"), cfg)

    from tasque.chains.manager import get_chain_state
    from tasque.chains.scheduler import maybe_finalize_status

    maybe_finalize_status(chain_id)
    state = get_chain_state(chain_id)
    assert state is not None
    notify = next(n for n in state["plan"] if n["id"] == "notify")
    assert notify["status"] == "completed"
    assert state["completed"]["notify"]["produces"]["user_reply"] == "approved"

    # ChainRun should now be in completed state.
    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        assert row["status"] == "completed"
        assert row["ended_at"] is not None


def test_stop_chain_marks_awaiting_step_stopped(
    fake_worker_runner: dict[str, Any],
) -> None:
    from tasque.chains.manager import get_chain_state, stop_chain

    chain_id = launch_chain_run(_3_step_spec())
    assert stop_chain(chain_id) is True

    state = get_chain_state(chain_id)
    assert state is not None
    notify = next(n for n in state["plan"] if n["id"] == "notify")
    assert notify["status"] == "stopped"

    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        assert row["status"] == "stopped"
        assert row["ended_at"] is not None


def test_self_stop_inside_worker_prevents_downstream_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker may call chain_run_stop through MCP during its own turn.

    The active LangGraph invoke still merges that worker's result, but the
    next supervisor pass must honor the persisted stop request before it
    promotes downstream nodes.
    """
    calls: list[str | None] = []

    def _fake(
        job: Any,
        *,
        consumes: dict[str, Any] | None = None,
        **_: Any,
    ) -> WorkerResult:
        calls.append(job.chain_step_id)
        if job.chain_step_id == "gate":
            from tasque.chains.manager import stop_chain

            assert job.chain_id is not None
            assert stop_chain(job.chain_id) is True
            return WorkerResult(
                report="all tasks already satisfy the threshold",
                summary="no-op gate",
                produces={"failing_tasks": []},
                error=None,
            )
        return WorkerResult(
            report="should not run",
            summary="unexpected",
            produces={},
            error=None,
        )

    import tasque.chains.graph.worker as worker_mod

    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    spec = {
        "chain_name": "self-stop-demo",
        "bucket": "education",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {"id": "gate", "kind": "worker", "directive": "gate", "tier": "small"},
            {
                "id": "downstream",
                "kind": "worker",
                "directive": "downstream",
                "depends_on": ["gate"],
                "consumes": ["gate"],
                "tier": "small",
            },
        ],
    }

    chain_id = launch_chain_run(spec)

    assert calls == ["gate"]

    from tasque.chains.manager import get_chain_state

    state = get_chain_state(chain_id)
    assert state is not None
    by_id = {n["id"]: n for n in state["plan"]}
    assert by_id["gate"]["status"] == "completed"
    assert by_id["downstream"]["status"] == "stopped"
    assert "gate" in state["completed"]
    assert "downstream" not in state["completed"]

    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        assert row["status"] == "stopped"
        assert row["ended_at"] is not None


def test_launch_vars_reach_every_worker_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator-supplied ``vars`` at launch are merged over the spec's
    static ``vars`` and surface in every worker dispatch."""
    record: dict[str, Any] = {"calls": []}

    def _fake(
        job: Any,
        *,
        consumes: dict[str, Any] | None = None,
        vars: dict[str, Any] | None = None,
        **_: Any,
    ) -> WorkerResult:
        record["calls"].append({
            "step_id": job.chain_step_id,
            "vars": dict(vars or {}),
        })
        if job.chain_step_id == "scan":
            return WorkerResult(
                report="ok", summary="ok",
                produces={"items": ["x", "y"]}, error=None,
            )
        return WorkerResult(report="ok", summary="ok", produces={}, error=None)

    import tasque.chains.graph.worker as worker_mod
    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    spec: dict[str, Any] = {
        "chain_name": "vars-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        # Spec-level static vars.
        "vars": {"force": False, "tag": "static"},
        "plan": [
            {"id": "scan", "kind": "worker", "directive": "scan", "tier": "small"},
            {
                "id": "filter",
                "kind": "worker",
                "directive": "filter",
                "depends_on": ["scan"],
                "consumes": ["scan"],
                "fan_out_on": "items",
                "tier": "small",
            },
        ],
    }

    # Operator override at launch — wins on key collision with the spec.
    launch_chain_run(spec, vars={"force": True, "extra": 42})

    # Every dispatched worker (scan + two filter children) saw the
    # merged dict, with the operator override applied.
    assert len(record["calls"]) == 3
    for call in record["calls"]:
        assert call["vars"] == {"force": True, "tag": "static", "extra": 42}


def test_launch_with_wait_false_returns_quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    """``wait=False`` returns the chain_id without blocking on graph
    invocation. The daemon-owned runner claims the row later."""
    import time

    from tasque.chains import scheduler as sched

    invoked = False

    def _fake(
        job: Any, *,
        consumes: dict[str, Any] | None = None,
        vars: dict[str, Any] | None = None,
        **_: Any,
    ) -> WorkerResult:
        nonlocal invoked
        invoked = True
        # Simulate a slow worker so we can verify launch_chain_run
        # returned BEFORE the worker finished.
        time.sleep(0.2)
        return WorkerResult(report="ok", summary="ok", produces={}, error=None)

    import tasque.chains.graph.worker as worker_mod
    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    spec: dict[str, Any] = {
        "chain_name": "wait-false-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {"id": "only", "kind": "worker", "directive": "do thing", "tier": "small"},
        ],
    }

    t0 = time.monotonic()
    chain_id = launch_chain_run(spec, wait=False)
    elapsed = time.monotonic() - t0

    # Returned in well under the 0.2s worker sleep.
    assert elapsed < 0.15, f"launch_chain_run blocked for {elapsed:.3f}s with wait=False"
    assert chain_id
    assert invoked is False

    ran = sched.claim_and_run_ready_chains(owner_id="test-daemon", max_runs=1)
    assert ran == [chain_id]
    assert invoked is True

    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        assert row["status"] == "completed"


def test_resume_stale_chains_skips_fresh_checkpoints(
    fake_worker_runner: dict[str, Any],
) -> None:
    """A chain whose checkpoint was just written must NOT be resumed.

    Otherwise we'd race the in-flight runner thread and re-dispatch
    work the original invoke is still doing — exactly what the
    threshold filter is meant to prevent.
    """
    from tasque.chains import scheduler as sched

    chain_id = launch_chain_run(_3_step_spec())  # reaches awaiting_approval
    # Flip the row back to 'running' so the where clause picks it up.
    with get_session() as sess:
        sess.execute(
            ChainRun.__table__.update()
            .where(ChainRun.chain_id == chain_id)
            .values(status="running")
        )

    # Fresh checkpoint (just written by launch_chain_run) → not stale.
    resumed = sched.resume_stale_chains(threshold_seconds=30.0)
    assert resumed == []


def test_daemon_runner_picks_up_unstarted_running_row(
    fake_worker_runner: dict[str, Any],
) -> None:
    """wait=False only enqueues; the daemon runner claims and invokes it."""
    from tasque.chains import scheduler as sched

    spec: dict[str, Any] = {
        "chain_name": "queued-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {"id": "only", "kind": "worker", "directive": "do thing", "tier": "small"},
        ],
    }
    chain_id = launch_chain_run(spec, wait=False)

    ran = sched.claim_and_run_ready_chains(owner_id="test-daemon", max_runs=1)
    assert ran == [chain_id]
    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        assert row["status"] == "completed"


def test_resume_stale_chains_ignores_non_running_status(
    fake_worker_runner: dict[str, Any],
) -> None:
    """Only ``running`` rows are candidates — completed / paused /
    stopped chains are left alone."""
    from datetime import UTC, datetime, timedelta

    from tasque.chains import scheduler as sched

    # Single-worker chain runs to completion; ChainRun.status flips to
    # 'completed' via maybe_finalize_status. Non-running rows must be
    # skipped by the WHERE clause regardless of checkpoint age.
    spec: dict[str, Any] = {
        "chain_name": "completed-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {"id": "only", "kind": "worker", "directive": "do thing", "tier": "small"},
        ],
    }
    chain_id = launch_chain_run(spec)
    future = datetime.now(UTC) + timedelta(hours=1)
    resumed = sched.resume_stale_chains(threshold_seconds=0.0, now=future)
    assert chain_id not in resumed


def test_resume_stale_chains_skips_active_invoke_even_when_stale(
    fake_worker_runner: dict[str, Any],
) -> None:
    """Even when the checkpoint looks stale by mtime, a chain whose
    runner thread is currently alive in this process must not be
    re-invoked — that would double-dispatch work the live invoke is
    still doing (the original observed mid-LLM-turn race)."""
    from datetime import UTC, datetime, timedelta

    from tasque.chains import scheduler as sched

    chain_id = launch_chain_run(_3_step_spec())
    with get_session() as sess:
        sess.execute(
            ChainRun.__table__.update()
            .where(ChainRun.chain_id == chain_id)
            .values(status="running")
        )

    # Simulate an in-flight invoke for this chain. The registry is
    # process-local, so we mark it active manually and verify the
    # stale-resume path skips it despite a fully stale checkpoint.
    sched._mark_invoke_active(chain_id)
    try:
        future = datetime.now(UTC) + timedelta(hours=1)
        resumed = sched.resume_stale_chains(threshold_seconds=0.0, now=future)
    finally:
        sched._mark_invoke_inactive(chain_id)
    assert chain_id not in resumed


def test_resume_stale_chains_skips_fresh_persistent_heartbeat(
    fake_worker_runner: dict[str, Any],
) -> None:
    """A chain can be alive in another process, so stale resume must
    honor the persistent ChainRun heartbeat as well as the local active
    registry."""
    from datetime import UTC, datetime, timedelta

    from tasque.chains import scheduler as sched

    chain_id = launch_chain_run(_3_step_spec())
    future = datetime.now(UTC) + timedelta(hours=1)
    fresh_heartbeat = future.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with get_session() as sess:
        sess.execute(
            ChainRun.__table__.update()
            .where(ChainRun.chain_id == chain_id)
            .values(status="running", last_heartbeat=fresh_heartbeat)
        )

    resumed = sched.resume_stale_chains(threshold_seconds=30.0, now=future)
    assert resumed == []


def test_bump_chain_heartbeat_updates_persistent_liveness(
    fake_worker_runner: dict[str, Any],
) -> None:
    from datetime import UTC, datetime, timedelta

    from tasque.chains import scheduler as sched

    chain_id = launch_chain_run(_3_step_spec())
    stale = (datetime.now(UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    with get_session() as sess:
        sess.execute(
            ChainRun.__table__.update()
            .where(ChainRun.chain_id == chain_id)
            .values(last_heartbeat=stale)
        )

    assert sched._bump_chain_heartbeat(chain_id)

    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        assert row["last_heartbeat"] != stale


def test_active_invoke_registry_clears_after_launch_returns(
    fake_worker_runner: dict[str, Any],
) -> None:
    """``launch_chain_run(wait=True)`` must remove its chain from the
    active-invoke set on return so subsequent stale-resume passes can
    legitimately recover it if needed (e.g. after a daemon restart with
    a row left in 'running' from this very process — unusual, but the
    invariant should hold)."""
    from tasque.chains import scheduler as sched

    chain_id = launch_chain_run(_3_step_spec())
    assert not sched._is_invoke_active(chain_id)


def test_checkpoint_age_seconds_returns_none_when_no_checkpoint() -> None:
    from tasque.chains.scheduler import _checkpoint_age_seconds

    assert _checkpoint_age_seconds("nonexistent-chain-id") is None


def test_checkpoint_age_seconds_returns_seconds_when_checkpoint_exists(
    fake_worker_runner: dict[str, Any],
) -> None:
    from datetime import UTC, datetime, timedelta

    from tasque.chains.scheduler import _checkpoint_age_seconds

    chain_id = launch_chain_run(_3_step_spec())
    age = _checkpoint_age_seconds(chain_id)
    assert age is not None and age >= 0.0

    # Compute against a far-future reference: age should reflect that.
    far = datetime.now(UTC) + timedelta(seconds=3600)
    far_age = _checkpoint_age_seconds(chain_id, now=far)
    assert far_age is not None and far_age >= 3600.0


def test_produces_schema_violation_flips_step_to_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that returns the wrong shape must fail the step rather
    than silently propagate corrupt produces. This is the failure mode
    that motivated produces_schema: the trading-scan aggregate_scans
    LLM dropped ``trade_list`` from each list item, the chain reported
    ``completed``, and 7 buckets ended up in ``no_trades`` despite
    having real trades upstream."""

    def _fake(job: Any, **_: Any) -> WorkerResult:
        # Emit a list whose items are missing a required key.
        return WorkerResult(
            report="ok",
            summary="ok",
            produces={"branches": [{"bucket": "car"}, {"bucket": "home"}]},
            error=None,
        )

    import tasque.chains.graph.worker as worker_mod
    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    spec: dict[str, Any] = {
        "chain_name": "schema-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {
                "id": "agg",
                "kind": "worker",
                "directive": "aggregate",
                "tier": "small",
                "produces_schema": {
                    "required": ["branches"],
                    "list_items": {
                        "branches": {"required": ["bucket", "trade_list"]},
                    },
                },
            },
        ],
    }
    chain_id = launch_chain_run(spec)

    from tasque.chains.manager import get_chain_state

    state = get_chain_state(chain_id)
    assert state is not None
    agg = next(n for n in state["plan"] if n["id"] == "agg")
    assert agg["status"] == "failed"
    failure_reason = (state.get("failures") or {}).get("agg") or ""
    assert "produces_schema" in failure_reason
    assert "trade_list" in failure_reason


def test_launch_vars_default_to_empty_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """If neither the spec nor the launch call supplies vars, every
    worker sees an empty dict (never None or missing)."""
    seen_vars: list[dict[str, Any]] = []

    def _fake(
        job: Any, *,
        consumes: dict[str, Any] | None = None,
        vars: dict[str, Any] | None = None,
        **_: Any,
    ) -> WorkerResult:
        seen_vars.append(dict(vars or {}))
        return WorkerResult(report="ok", summary="ok", produces={}, error=None)

    import tasque.chains.graph.worker as worker_mod
    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    spec: dict[str, Any] = {
        "chain_name": "vars-default-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {"id": "only", "kind": "worker", "directive": "do thing", "tier": "small"},
        ],
    }
    launch_chain_run(spec)

    assert seen_vars == [{}]

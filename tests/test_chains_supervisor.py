"""Tests for the supervisor: reconcile, fan-out materialization, replan
routing, and the None-clears-failure reducer."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END

from tasque.chains.graph._common import _merge_failures
from tasque.chains.graph.supervisor import (
    APPROVAL_NODE,
    PLANNER_NODE,
    WORKER_NODE,
    _route_from_supervisor,
    supervisor,
)
from tasque.chains.spec import PlanNode


def _node(**kwargs: Any) -> PlanNode:
    base: dict[str, Any] = {
        "id": "x",
        "kind": "worker",
        "directive": "do",
        "depends_on": [],
        "consumes": [],
        "fan_out_on": None,
        "fan_out_concurrency": None,
        "status": "pending",
        "origin": "spec",
        "on_failure": "halt",
        "failure_reason": None,
        "fan_out_index": None,
        "fan_out_item": None,
        "tier": "haiku",
    }
    base.update(kwargs)
    # Approval steps must NOT carry a tier — the supervisor doesn't
    # use it for them and the validator rejects it.
    if base.get("kind") == "approval":
        base["tier"] = None
    return base  # type: ignore[return-value]


# ----------------------------------------------------------- _merge_failures

def test_merge_failures_none_on_right_clears_key() -> None:
    left = {"a": "boom", "b": "kapow"}
    right = {"a": None}
    out = _merge_failures(left, right)
    assert out == {"b": "kapow"}


def test_merge_failures_naive_merge_would_keep_old() -> None:
    """Sanity-check that {**left, **right} would NOT clear the key — this is
    the bug the contract calls out. ``_merge_failures`` must do better."""
    naive = {**{"a": "boom"}, **{"a": None}}
    assert naive == {"a": None}  # still present, not cleared
    out = _merge_failures({"a": "boom"}, {"a": None})
    assert "a" not in out


def test_merge_failures_right_value_wins_for_non_none() -> None:
    out = _merge_failures({"a": "old"}, {"a": "new"})
    assert out == {"a": "new"}


# ----------------------------------------------------------- supervisor

def test_supervisor_promotes_pending_to_running_when_deps_clear() -> None:
    plan = [
        _node(id="a"),
        _node(id="b", depends_on=["a"], consumes=["a"]),
    ]
    update = supervisor({"plan": plan, "completed": {}, "failures": {}})
    new_plan = update["plan"]
    statuses = {n["id"]: n["status"] for n in new_plan}
    assert statuses["a"] == "running"
    assert statuses["b"] == "pending"


def test_supervisor_reconciles_running_completed_step() -> None:
    plan = [_node(id="a", status="running")]
    completed = {"a": {"report": "done", "produces": {}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    statuses = {n["id"]: n["status"] for n in update["plan"]}
    assert statuses["a"] == "completed"


def test_supervisor_reconciles_running_failed_step_sets_failure_reason() -> None:
    plan = [_node(id="a", status="running")]
    failures = {"a": "kaboom"}
    update = supervisor({"plan": plan, "completed": {}, "failures": failures})
    a = next(n for n in update["plan"] if n["id"] == "a")
    assert a["status"] == "failed"
    assert a["failure_reason"] == "kaboom"


def test_supervisor_sets_replan_when_failed_step_has_replan() -> None:
    plan = [_node(id="a", status="running", on_failure="replan")]
    failures = {"a": "boom"}
    update = supervisor({"plan": plan, "completed": {}, "failures": failures})
    assert update.get("replan") is True


def test_supervisor_materializes_fan_out_with_good_list() -> None:
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="filter",
            kind="worker",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
        ),
    ]
    completed = {"scan": {"report": "ok", "produces": {"items": ["x", "y"]}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    new_plan = update["plan"]
    ids = sorted(n["id"] for n in new_plan)
    assert ids == ["filter", "filter[0]", "filter[1]", "scan"]
    f0 = next(n for n in new_plan if n["id"] == "filter[0]")
    assert f0["fan_out_index"] == 0
    assert f0["fan_out_item"] == "x"
    assert f0["fan_out_on"] is None
    template = next(n for n in new_plan if n["id"] == "filter")
    assert template["status"] == "completed"


def test_supervisor_chained_fan_out_synthesizes_items_from_upstream_template() -> None:
    """``fan_out_on`` may name an upstream fan-out template directly. The
    items are the upstream children's ``produces`` dicts in fan-out-index
    order, with no LLM passthrough aggregator in the middle. This is the
    structural fix for the trading-scan failure where an LLM aggregator
    hallucinated branch contents while the schema check passed (key
    shape was valid; values were fabricated)."""
    plan = [
        _node(id="scan", status="completed", fan_out_on="buckets"),
        # scan has two materialized children with real produces.
        _node(
            id="scan[0]",
            depends_on=["scan"],
            consumes=["scan"],
            status="completed",
            fan_out_index=0,
        ),
        _node(
            id="scan[1]",
            depends_on=["scan"],
            consumes=["scan"],
            status="completed",
            fan_out_index=1,
        ),
        # news_veto fans out over scan's children directly.
        _node(
            id="news_veto",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="scan",
        ),
    ]
    completed = {
        "scan[0]": {
            "report": "ok",
            "produces": {"bucket_id": "car", "trade_list": [1, 2, 3]},
        },
        "scan[1]": {
            "report": "ok",
            "produces": {"bucket_id": "home", "trade_list": []},
        },
    }
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    new_plan = update["plan"]
    ids = sorted(n["id"] for n in new_plan)
    assert "news_veto[0]" in ids
    assert "news_veto[1]" in ids
    nv0 = next(n for n in new_plan if n["id"] == "news_veto[0]")
    nv1 = next(n for n in new_plan if n["id"] == "news_veto[1]")
    # The fan_out_item is the upstream child's produces dict, byte-for-byte.
    assert nv0["fan_out_item"] == {"bucket_id": "car", "trade_list": [1, 2, 3]}
    assert nv1["fan_out_item"] == {"bucket_id": "home", "trade_list": []}
    template = next(n for n in new_plan if n["id"] == "news_veto")
    assert template["status"] == "completed"


def test_supervisor_chained_fan_out_single_pass_race() -> None:
    """When two fan-out templates chain (``scan`` → ``news_veto`` where
    news_veto.fan_out_on='scan'), the supervisor must NOT materialize
    both in a single pass. Otherwise the news_veto deps check fires
    before scan's children are added to plan, sees zero children, and
    materializes news_veto with zero items via the chained-fan-out
    path. The chain then "completes" with no work done.

    The fix: plan is extended in-place during the materialization loop
    so the second template sees scan's pending children and correctly
    defers to a later supervisor pass.

    This is the regression for the trading-scan run that completed at
    20:11 UTC on 2026-04-30 with scan[0..7] running but news_veto and
    dispatch templates marked completed with zero children — the chain
    silently skipped the entire trading dispatch phase."""
    # enumerate_buckets has just completed; scan template is pending
    # with deps satisfied. news_veto and dispatch templates also exist
    # in plan. completed dict contains enumerate_buckets only.
    plan = [
        _node(id="enumerate_buckets", status="completed"),
        _node(
            id="scan",
            depends_on=["enumerate_buckets"],
            consumes=["enumerate_buckets"],
            fan_out_on="buckets",
        ),
        _node(
            id="news_veto",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="scan",
        ),
        _node(
            id="dispatch",
            depends_on=["news_veto"],
            consumes=["news_veto"],
            fan_out_on="news_veto",
        ),
    ]
    completed = {
        "enumerate_buckets": {
            "report": "ok",
            "produces": {"buckets": [{"bucket_id": "car"}, {"bucket_id": "home"}]},
        },
    }
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    new_plan = update["plan"]

    # scan must have materialized 2 children.
    scan_children = [n for n in new_plan if n["id"].startswith("scan[")]
    assert len(scan_children) == 2, (
        f"expected scan to materialize 2 children, got {len(scan_children)}"
    )

    # news_veto must NOT have materialized in this same pass — its
    # deps include scan's just-added pending children.
    nv = next(n for n in new_plan if n["id"] == "news_veto")
    assert nv["status"] == "pending", (
        f"news_veto must defer materialization until scan's children "
        f"complete; got status={nv['status']}"
    )
    nv_children = [n for n in new_plan if n["id"].startswith("news_veto[")]
    assert nv_children == [], (
        f"news_veto must NOT have any children yet; got {len(nv_children)}"
    )

    dispatch = next(n for n in new_plan if n["id"] == "dispatch")
    assert dispatch["status"] == "pending"


def test_supervisor_chained_fan_out_requires_consumes() -> None:
    """If you fan out over an upstream fan-out template you must also
    consume it — the worker needs the items as ``consumes_payload``."""
    plan = [
        _node(id="scan", status="completed", fan_out_on="buckets"),
        _node(
            id="scan[0]",
            depends_on=["scan"],
            consumes=["scan"],
            status="completed",
            fan_out_index=0,
        ),
        _node(
            id="news_veto",
            depends_on=["scan"],
            consumes=[],  # <-- missing 'scan'
            fan_out_on="scan",
        ),
    ]
    completed = {"scan[0]": {"report": "ok", "produces": {}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    template = next(n for n in update["plan"] if n["id"] == "news_veto")
    assert template["status"] == "failed"
    assert "consumes" in (template.get("failure_reason") or "")


def test_supervisor_completes_template_on_empty_fan_out_list() -> None:
    """An empty list from upstream is a legitimate "nothing to do"
    outcome — the template completes with zero children rather than
    halting the chain."""
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="filter",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
        ),
    ]
    completed = {"scan": {"report": "ok", "produces": {"items": []}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    template = next(n for n in update["plan"] if n["id"] == "filter")
    assert template["status"] == "completed"
    # No fan-out children materialised.
    children = [n for n in update["plan"] if n["id"].startswith("filter[")]
    assert children == []
    # And no failure was synthesised.
    assert "failures" not in update or "filter" not in (update.get("failures") or {})


def test_supervisor_promotes_downstream_after_empty_fan_out() -> None:
    """After an empty fan-out completes, a downstream step that
    ``consumes`` the template should be promoted to running and
    receive ``[]`` for that dep — same shape it would see for a
    non-empty fan-out, just empty."""
    from tasque.chains.graph.supervisor import _gather_consumes_payload

    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="filter",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
        ),
        _node(
            id="apply",
            depends_on=["filter"],
            consumes=["filter"],
        ),
    ]
    completed = {"scan": {"report": "ok", "produces": {"items": []}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    new_plan = update["plan"]
    apply_node = next(n for n in new_plan if n["id"] == "apply")
    # Deps satisfied (filter completed with zero children) → promoted.
    assert apply_node["status"] == "running"
    payload = _gather_consumes_payload(new_plan, apply_node, completed)
    assert payload == {"filter": []}


def test_supervisor_fails_template_on_non_list_fan_out() -> None:
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="filter",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
        ),
    ]
    completed = {"scan": {"report": "ok", "produces": {"items": "nope"}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    template = next(n for n in update["plan"] if n["id"] == "filter")
    assert template["status"] == "failed"
    assert "not a list" in (template["failure_reason"] or "")


def test_supervisor_fails_template_when_fan_out_dep_not_in_consumes() -> None:
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="filter",
            depends_on=["scan"],
            consumes=[],
            fan_out_on="items",
        ),
    ]
    completed = {"scan": {"report": "ok", "produces": {"items": ["x"]}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    template = next(n for n in update["plan"] if n["id"] == "filter")
    assert template["status"] == "failed"


def test_supervisor_waits_on_fan_out_children_before_promoting_downstream() -> None:
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="filter",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
        ),
        _node(id="notify", depends_on=["filter"], consumes=["filter"]),
    ]
    completed = {"scan": {"report": "ok", "produces": {"items": ["x", "y"]}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    notify = next(n for n in update["plan"] if n["id"] == "notify")
    # Children just materialized, not yet completed → notify must wait.
    assert notify["status"] == "pending"


def test_supervisor_promotes_downstream_after_all_fan_out_children_completed() -> None:
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="filter",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
            status="completed",
        ),
        _node(
            id="filter[0]",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on=None,
            fan_out_index=0,
            fan_out_item="x",
            status="completed",
        ),
        _node(
            id="filter[1]",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on=None,
            fan_out_index=1,
            fan_out_item="y",
            status="completed",
        ),
        _node(id="notify", depends_on=["filter"], consumes=["filter"]),
    ]
    completed = {
        "scan": {"report": "ok", "produces": {"items": ["x", "y"]}},
        "filter[0]": {"report": "ok", "produces": {"framing": "X"}},
        "filter[1]": {"report": "ok", "produces": {"framing": "Y"}},
    }
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    notify = next(n for n in update["plan"] if n["id"] == "notify")
    assert notify["status"] == "running"


# --------------------------------------------------- fan_out_concurrency

def test_fan_out_concurrency_caps_initial_promotion() -> None:
    """A fan-out with concurrency=1 should only promote one child to
    running on the first supervisor pass even when several children are
    materialized at once and all their deps are satisfied."""
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="dispatch",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
            fan_out_concurrency=1,
        ),
    ]
    completed = {"scan": {"report": "ok", "produces": {"items": ["a", "b", "c"]}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    new_plan = update["plan"]
    children = [n for n in new_plan if n["id"].startswith("dispatch[")]
    assert len(children) == 3
    running = [c for c in children if c["status"] == "running"]
    pending = [c for c in children if c["status"] == "pending"]
    assert len(running) == 1
    assert len(pending) == 2


def test_fan_out_concurrency_inherits_to_children() -> None:
    """Children should carry the template's concurrency value so the
    supervisor can read it during later promotion passes (when the
    template node is already completed and unhelpful)."""
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="dispatch",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
            fan_out_concurrency=2,
        ),
    ]
    completed = {"scan": {"report": "ok", "produces": {"items": [1, 2, 3]}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    children = [n for n in update["plan"] if n["id"].startswith("dispatch[")]
    assert all(c["fan_out_concurrency"] == 2 for c in children)


def test_fan_out_concurrency_promotes_next_after_sibling_completes() -> None:
    """Once one running child finishes, the next supervisor pass should
    promote one more pending child — never exceeding the cap."""
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="dispatch",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
            fan_out_concurrency=1,
            status="completed",
        ),
        _node(
            id="dispatch[0]",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on=None,
            fan_out_concurrency=1,
            fan_out_index=0,
            fan_out_item="a",
            status="running",
        ),
        _node(
            id="dispatch[1]",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on=None,
            fan_out_concurrency=1,
            fan_out_index=1,
            fan_out_item="b",
            status="pending",
        ),
    ]
    # First sibling completed; reconcile should promote it, then promotion
    # step sees 0 running siblings and is free to promote one pending.
    completed = {
        "scan": {"report": "ok", "produces": {"items": ["a", "b"]}},
        "dispatch[0]": {"report": "ok", "produces": {}},
    }
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    new_plan = update["plan"]
    by_id = {n["id"]: n for n in new_plan}
    assert by_id["dispatch[0]"]["status"] == "completed"
    assert by_id["dispatch[1]"]["status"] == "running"


def test_fan_out_concurrency_none_means_unbounded() -> None:
    """The default behaviour (concurrency=None) must remain unbounded —
    every materialized child is promoted at once."""
    plan = [
        _node(id="scan", status="completed"),
        _node(
            id="dispatch",
            depends_on=["scan"],
            consumes=["scan"],
            fan_out_on="items",
            # fan_out_concurrency omitted — defaults to None via _node's base
        ),
    ]
    completed = {"scan": {"report": "ok", "produces": {"items": [1, 2, 3, 4]}}}
    update = supervisor({"plan": plan, "completed": completed, "failures": {}})
    children = [n for n in update["plan"] if n["id"].startswith("dispatch[")]
    assert len(children) == 4
    assert all(c["status"] == "running" for c in children)


# ----------------------------------------------------------- routing

def test_route_from_supervisor_routes_to_planner_on_replan() -> None:
    plan = [_node(id="a", status="failed")]
    target = _route_from_supervisor({"plan": plan, "replan": True})
    assert target == PLANNER_NODE


def test_route_from_supervisor_dispatches_running_workers() -> None:
    plan = [
        _node(id="a", status="running"),
        _node(id="b", status="pending"),
    ]
    target = _route_from_supervisor({"plan": plan, "completed": {}, "failures": {}})
    assert isinstance(target, list)
    assert len(target) == 1
    assert target[0].node == WORKER_NODE


def test_route_from_supervisor_dispatches_running_approvals() -> None:
    plan = [_node(id="a", kind="approval", status="running")]
    target = _route_from_supervisor({"plan": plan, "completed": {}, "failures": {}})
    assert isinstance(target, list)
    assert len(target) == 1
    assert target[0].node == APPROVAL_NODE


def test_route_from_supervisor_ends_when_nothing_in_flight() -> None:
    plan = [_node(id="a", status="completed")]
    target = _route_from_supervisor({"plan": plan, "completed": {}, "failures": {}})
    assert target == END

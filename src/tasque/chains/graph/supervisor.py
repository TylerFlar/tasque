"""The supervisor node and its conditional-edge router.

Responsibilities, in order, on every pass:

1. **Reconcile.** For each step in ``running``, see if a worker result has
   landed in ``completed`` or ``failures`` and flip the plan node to
   ``completed`` / ``failed``. If a failed step has ``on_failure="replan"``,
   set ``state.replan = True``.
2. **Materialize fan-outs.** A pending worker step with ``fan_out_on``
   whose deps are satisfied: mark the template ``completed``, append one
   child node per item in the upstream produces list. Bad fan-out shapes
   (upstream missing, not a list, empty) fail the template instead.
3. **Promote pending → running.** A pending worker/approval whose deps
   are satisfied gets flipped to ``running`` so the conditional edge
   dispatches it via :class:`~langgraph.types.Send`.

The conditional edge :func:`_route_from_supervisor` then either
recurses into the planner, dispatches all newly-running nodes, or returns
``END`` when there's nothing left to do.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END
from langgraph.types import Send

from tasque.chains.graph._common import (
    ChainStateSchema,
    _ApprovalInput,
    _node_by_id,
    _now_iso,
    _WorkerInput,
)
from tasque.chains.spec import HistoryEntry, PlanNode

SUPERVISOR_NODE = "supervisor"
WORKER_NODE = "worker"
APPROVAL_NODE = "approval"
PLANNER_NODE = "planner"


def _children_of(plan: list[PlanNode], template_id: str) -> list[PlanNode]:
    prefix = f"{template_id}["
    return [n for n in plan if n["id"].startswith(prefix)]


def _has_fan_out_children(plan: list[PlanNode], step_id: str) -> bool:
    return bool(_children_of(plan, step_id))


def _template_id_of(node_id: str) -> str | None:
    """Return the template id for a fan-out child, or None for a non-child.

    Fan-out children have ids of the form ``{template}[{i}]``; the
    template id is everything before the first ``[``.
    """
    bracket = node_id.find("[")
    if bracket == -1:
        return None
    return node_id[:bracket]


def _all_dep_ids_satisfied(plan: list[PlanNode], node: PlanNode) -> bool:
    """Stricter than ``_deps_satisfied`` from _common: a dep that is a
    fan-out template must have all materialized children completed too."""
    for dep_id in node["depends_on"]:
        direct = _node_by_id(plan, dep_id)
        if direct is None:
            return False
        if direct["status"] != "completed":
            return False
        for child in _children_of(plan, dep_id):
            if child["status"] != "completed":
                return False
    return True


def _gather_consumes_payload(
    plan: list[PlanNode],
    node: PlanNode,
    completed: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``consumes_payload`` dict the worker node injects into the
    LLM prompt.

    For a regular dep: ``payload[dep] = completed[dep]["produces"]``.
    For a fan-out template dep with N≥1 materialized children:
    ``payload[dep] = [child.produces for child in children]``.
    For a fan-out template dep that materialized zero children
    (legitimate empty fan-out): ``payload[dep] = []`` — same shape as
    a one-or-more fan-out, just empty, so the downstream worker can
    short-circuit on length.
    """
    payload: dict[str, Any] = {}
    for dep_id in node["consumes"]:
        children = _children_of(plan, dep_id)
        if children:
            payload[dep_id] = [
                completed.get(c["id"], {}).get("produces", {}) for c in children
            ]
            continue
        # No children. Distinguish "fan-out template with zero items"
        # from "regular non-fan-out dep" — the former should surface as
        # an empty list so the worker sees the same shape it would have
        # seen with N≥1 children.
        dep_node = _node_by_id(plan, dep_id)
        if dep_node is not None and dep_node.get("fan_out_on") is not None:
            payload[dep_id] = []
            continue
        cell = completed.get(dep_id)
        if cell is None:
            payload[dep_id] = {}
        else:
            payload[dep_id] = cell.get("produces", {})
    return payload


def _materialize_fan_out(
    template: PlanNode,
    plan: list[PlanNode],
    completed: dict[str, Any],
) -> tuple[list[PlanNode], str | None]:
    """Return (children, error_reason). On error, ``children`` is empty.

    Children inherit the template's ``directive``, ``consumes``, and
    ``on_failure``. Their ``id`` is ``template[i]``; ``fan_out_on`` is
    cleared to prevent recursive fan-out.

    Two resolution paths for the items list:

    1. **Upstream is itself a fan-out template** — when ``fan_out_on``
       names a dep id whose plan node is also a fan-out template, the
       items are the children's ``produces`` dicts in fan-out-index
       order. This is the chained-fan-out path (e.g. ``scan`` fans out
       per bucket → ``news_veto`` fans out per ``scan`` branch). It
       eliminates the need for an LLM passthrough aggregator step —
       the LLM-as-data-pipe pattern was load-bearingly fragile (the
       2026-04-30 trading-scan run had ``aggregate_news_veto`` (haiku)
       hallucinate every branch's ``bucket`` and ``trade_list`` while
       still producing a schema-valid shape, sending fabricated trades
       to dispatch).

    2. **Upstream produces has the key as a list** — the original
       behavior: any dep whose ``produces[fan_key]`` is a list serves
       as the source. Used when the items are computed inside a
       worker's directive (e.g. ``enumerate_buckets`` produces a
       ``buckets`` list that ``scan`` fans out over).
    """
    fan_key = template["fan_out_on"]
    assert fan_key is not None  # caller checks

    # Path 1: chained fan-out — upstream dep id matches and is a
    # fan-out template itself. Items are the synthesized child
    # produces list, in deterministic fan-out-index order.
    if fan_key in template["depends_on"]:
        fan_template = _node_by_id(plan, fan_key)
        if fan_template is not None and fan_template.get("fan_out_on") is not None:
            if fan_key not in template["consumes"]:
                return [], (
                    f"fan_out_on={fan_key!r} resolves to fan-out template "
                    f"{fan_key!r} but it's not in 'consumes'; add it"
                )
            children_of_fan = sorted(
                _children_of(plan, fan_key),
                key=lambda n: n.get("fan_out_index") or 0,
            )
            items: list[Any] = [
                completed.get(c["id"], {}).get("produces", {})
                for c in children_of_fan
            ]
            return _build_fan_out_children(template, items), None

    # Path 2: legacy — find fan_key inside an upstream's produces dict.
    upstream_id: str | None = None
    for dep in template["depends_on"]:
        cell = completed.get(dep)
        if cell is not None and fan_key in cell.get("produces", {}):
            upstream_id = dep
            break

    if upstream_id is None:
        return [], (
            f"fan_out_on={fan_key!r} not found in any upstream produces "
            f"and no dep matches a fan-out template "
            f"(consumes={template['consumes']!r})"
        )
    if upstream_id not in template["consumes"]:
        return [], (
            f"fan_out_on={fan_key!r} resolves via dep {upstream_id!r} which is "
            "not in 'consumes'; add it"
        )

    items_raw = completed[upstream_id]["produces"][fan_key]
    if not isinstance(items_raw, list):
        return [], (
            f"upstream {upstream_id!r} produces[{fan_key!r}] is "
            f"{type(items_raw).__name__}, not a list"
        )
    items = cast(list[Any], items_raw)
    # Empty list is a legitimate "nothing to do" outcome — the upstream
    # worker correctly produced zero items (e.g. no eligible buckets to
    # research today). Mark the template completed with zero children;
    # downstream steps that consume this template see ``[]`` and can
    # short-circuit. Failing here would halt the chain over a no-op.
    return _build_fan_out_children(template, items), None


def _build_fan_out_children(
    template: PlanNode, items: list[Any]
) -> list[PlanNode]:
    """Construct one child plan node per item. Pulled out of
    ``_materialize_fan_out`` so both resolution paths share the
    same shape."""
    children: list[PlanNode] = []
    for i, item in enumerate(items):
        child: PlanNode = {
            "id": f"{template['id']}[{i}]",
            "kind": template["kind"],
            "directive": template["directive"],
            "depends_on": list(template["depends_on"]),
            "consumes": list(template["consumes"]),
            "fan_out_on": None,
            # Children inherit the template's concurrency cap so the
            # supervisor's promotion step can read it from any sibling
            # without having to look up the (already-completed) template.
            "fan_out_concurrency": template.get("fan_out_concurrency"),
            "status": "pending",
            "origin": template["origin"],
            "on_failure": template["on_failure"],
            "failure_reason": None,
            "fan_out_index": i,
            "fan_out_item": item,
            # Children inherit the template's tier — every fan-out
            # branch runs at the same model size as its parent.
            "tier": template.get("tier"),
            "produces_schema": template.get("produces_schema"),
        }
        children.append(child)
    return children


# ----------------------------------------------------------------- supervisor

def supervisor(state: ChainStateSchema) -> dict[str, Any]:
    """Reconcile, materialize fan-outs, and promote pending → running.

    Returns a partial state update with the new ``plan`` list and any
    ``history`` entries to append. Does *not* set ``replan`` unless an
    ``on_failure="replan"`` step actually failed in this pass.
    """
    plan_in: list[PlanNode] = list(state.get("plan") or [])
    completed: dict[str, Any] = dict(state.get("completed") or {})
    failures: dict[str, str] = dict(state.get("failures") or {})

    # Deep-copy nodes we may mutate so we don't accidentally mutate the
    # in-state list (which other reducers may still be reading).
    plan: list[PlanNode] = [dict(n) for n in plan_in]  # type: ignore[misc]
    history_appends: list[HistoryEntry] = []
    set_replan = False

    # 1. Reconcile running → completed/failed/awaiting_user
    for n in plan:
        if n["status"] == "running":
            if n["id"] in failures:
                n["status"] = "failed"
                n["failure_reason"] = failures[n["id"]]
                history_appends.append({
                    "timestamp": _now_iso(),
                    "kind": "status",
                    "details": {"step": n["id"], "to": "failed"},
                })
                if n["on_failure"] == "replan":
                    set_replan = True
            elif n["id"] in completed:
                n["status"] = "completed"
                history_appends.append({
                    "timestamp": _now_iso(),
                    "kind": "status",
                    "details": {"step": n["id"], "to": "completed"},
                })

    # 2. Materialize fan-outs (pending workers with fan_out_on, deps OK).
    #
    # Plan extension is in-place during the loop so subsequent templates
    # see freshly-materialized children. This matters for chained
    # fan-outs (``scan`` → ``news_veto`` → ``dispatch``, where each step
    # has ``fan_out_on`` pointing at the previous fan-out template):
    # without in-place extension, all three templates' dep checks fire
    # against the original snapshot, scan's children aren't visible yet,
    # and ``_all_dep_ids_satisfied`` for news_veto trivially returns
    # True (zero unfinished children, because zero children at all).
    # news_veto then materializes with zero items via the chained path,
    # and dispatch follows. The chain "completes" with no work done.
    # Adding children to plan immediately means subsequent templates see
    # them as pending and correctly defer to a later supervisor pass.
    for n in list(plan):
        if (
            n["status"] != "pending"
            or n["fan_out_on"] is None
            or n["kind"] != "worker"
        ):
            continue
        if not _all_dep_ids_satisfied(plan, n):
            continue
        if _has_fan_out_children(plan, n["id"]):
            # already materialized
            continue
        children, err = _materialize_fan_out(n, plan, completed)
        if err is not None:
            n["status"] = "failed"
            n["failure_reason"] = err
            failures[n["id"]] = err
            history_appends.append({
                "timestamp": _now_iso(),
                "kind": "status",
                "details": {"step": n["id"], "to": "failed", "reason": err},
            })
            if n["on_failure"] == "replan":
                set_replan = True
            continue
        # Mark template completed; downstream prefix-match dep resolution
        # will require children to be done too.
        n["status"] = "completed"
        history_appends.append({
            "timestamp": _now_iso(),
            "kind": "mutation",
            "details": {
                "step": n["id"],
                "fan_out_count": len(children),
            },
        })
        plan.extend(children)

    # 3. Promote pending → running for steps whose deps are satisfied.
    # Track running siblings per fan-out template so we can enforce
    # ``fan_out_concurrency``: a child whose template is capped at K
    # must defer if K of its siblings are already running.
    running_per_template: dict[str, int] = {}
    for n in plan:
        if n["status"] == "running":
            tid = _template_id_of(n["id"])
            if tid is not None:
                running_per_template[tid] = running_per_template.get(tid, 0) + 1

    for n in plan:
        if n["status"] != "pending":
            continue
        if n["fan_out_on"] is not None and n["kind"] == "worker":
            # Fan-out templates aren't dispatched directly; they
            # materialize. Skip.
            continue
        if not _all_dep_ids_satisfied(plan, n):
            continue
        cap = n.get("fan_out_concurrency")
        if isinstance(cap, int) and cap > 0:
            tid = _template_id_of(n["id"])
            if tid is not None:
                current = running_per_template.get(tid, 0)
                if current >= cap:
                    # At cap; leave this child pending. The next
                    # supervisor pass (after a sibling completes) will
                    # promote it.
                    continue
                running_per_template[tid] = current + 1
        n["status"] = "running"
        history_appends.append({
            "timestamp": _now_iso(),
            "kind": "status",
            "details": {"step": n["id"], "to": "running"},
        })

    update: dict[str, Any] = {"plan": plan}
    # Make failures field reflect any new failures we synthesized (the
    # reducer merges with existing).
    new_failures = {k: failures[k] for k in failures if k not in (state.get("failures") or {})}
    if new_failures:
        update["failures"] = new_failures
    if history_appends:
        update["history"] = history_appends
    if set_replan and not state.get("replan"):
        update["replan"] = True
    return update


# ----------------------------------------------------------------- routing

def _build_send_for_step(
    state: ChainStateSchema, plan: list[PlanNode], step: PlanNode
) -> Send:
    chain_id = state.get("chain_id") or ""
    chain_name = state.get("chain_name") or ""
    bucket = state.get("bucket") or ""
    completed = state.get("completed") or {}
    payload = _gather_consumes_payload(plan, step, completed)
    if step["kind"] == "approval":
        approval_input: _ApprovalInput = {
            "chain_id": chain_id,
            "chain_name": chain_name,
            "bucket": bucket,
            "step_id": step["id"],
            "directive": step["directive"],
            "consumes_payload": payload,
        }
        return Send(APPROVAL_NODE, approval_input)
    tier = step.get("tier")
    if not isinstance(tier, str) or not tier:
        raise ValueError(
            f"chain {chain_id[:8]} step {step['id']!r} has no tier set; "
            f"validator should have caught this — refusing to dispatch."
        )
    chain_vars_raw = state.get("vars")
    chain_vars: dict[str, Any] = (
        dict(chain_vars_raw) if isinstance(chain_vars_raw, dict) else {}
    )
    worker_input: _WorkerInput = {
        "chain_id": chain_id,
        "chain_name": chain_name,
        "bucket": bucket,
        "step_id": step["id"],
        "directive": step["directive"],
        "consumes_payload": payload,
        "fan_out_item": step.get("fan_out_item"),
        "on_failure": step["on_failure"],
        "tier": tier,
        "vars": chain_vars,
        "produces_schema": step.get("produces_schema"),
    }
    return Send(WORKER_NODE, worker_input)


def _route_from_supervisor(state: ChainStateSchema) -> str | list[Send]:
    """Conditional-edge target after the supervisor.

    Order: replan first, then dispatch any running steps, then END if
    nothing else is in flight.
    """
    if state.get("replan"):
        return PLANNER_NODE

    plan: list[PlanNode] = list(state.get("plan") or [])
    sends: list[Send] = []
    for n in plan:
        if n["status"] == "running":
            sends.append(_build_send_for_step(state, plan, n))
    if sends:
        return sends

    # If no node is running and nothing is awaiting user, terminate.
    any_in_flight = any(
        n["status"] in ("running", "awaiting_user") for n in plan
    )
    if any_in_flight:
        # An awaiting_user step exists — the graph is interrupting; we
        # shouldn't END. Return an empty Send list to keep the loop
        # alive without dispatching.
        return END
    return END


# ----------------------------------------------------------------- public

# Re-exposed so tests and other modules can import without reaching into
# the private routing helpers.
__all__ = [
    "APPROVAL_NODE",
    "PLANNER_NODE",
    "SUPERVISOR_NODE",
    "WORKER_NODE",
    "_all_dep_ids_satisfied",
    "_gather_consumes_payload",
    "_materialize_fan_out",
    "_route_from_supervisor",
    "_template_id_of",
    "supervisor",
]

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
    """
    fan_key = template["fan_out_on"]
    assert fan_key is not None  # caller checks
    upstream_id: str | None = None
    for dep in template["depends_on"]:
        # Heuristic: pick the first upstream dep that has the key.
        cell = completed.get(dep)
        if cell is not None and fan_key in cell.get("produces", {}):
            upstream_id = dep
            break

    if upstream_id is None:
        return [], (
            f"fan_out_on={fan_key!r} not found in any upstream produces "
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

    children: list[PlanNode] = []
    for i, item in enumerate(items):
        child: PlanNode = {
            "id": f"{template['id']}[{i}]",
            "kind": template["kind"],
            "directive": template["directive"],
            "depends_on": list(template["depends_on"]),
            "consumes": list(template["consumes"]),
            "fan_out_on": None,
            "status": "pending",
            "origin": template["origin"],
            "on_failure": template["on_failure"],
            "failure_reason": None,
            "fan_out_index": i,
            "fan_out_item": item,
            # Children inherit the template's tier — every fan-out
            # branch runs at the same model size as its parent.
            "tier": template.get("tier"),
        }
        children.append(child)
    return children, None


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
    new_children: list[PlanNode] = []
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
        new_children.extend(children)
    plan.extend(new_children)

    # 3. Promote pending → running for steps whose deps are satisfied.
    for n in plan:
        if n["status"] != "pending":
            continue
        if n["fan_out_on"] is not None and n["kind"] == "worker":
            # Fan-out templates aren't dispatched directly; they
            # materialize. Skip.
            continue
        if not _all_dep_ids_satisfied(plan, n):
            continue
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
    "supervisor",
]

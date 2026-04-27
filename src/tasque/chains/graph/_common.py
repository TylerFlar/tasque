"""Helpers shared by chain graph nodes.

Reducer functions plus a typed dict for the worker invocation payload.
The reducers are the load-bearing piece — without
:func:`_merge_failures`, a DLQ retry that sets ``failures[step] = None``
would not actually clear the failure (the naive ``{**a, **b}`` merge
keeps the old value).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Any, NotRequired, TypedDict

from tasque.chains.spec import (
    CompletedOutput,
    HistoryEntry,
    PlanNode,
)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _merge_completed(
    left: dict[str, CompletedOutput] | None,
    right: dict[str, CompletedOutput] | None,
) -> dict[str, CompletedOutput]:
    out: dict[str, CompletedOutput] = dict(left or {})
    if right:
        out.update(right)
    return out


def _merge_failures(
    left: dict[str, str] | None,
    right: Mapping[str, str | None] | None,
) -> dict[str, str]:
    """Merge with None-on-right semantics: a key whose new value is None is
    dropped from the result. Required for DLQ retry to actually clear a
    prior failure."""
    out: dict[str, str] = dict(left or {})
    if right:
        for k, v in right.items():
            if v is None:
                out.pop(k, None)
            else:
                out[k] = v
    return out


def _append_history(
    left: list[HistoryEntry] | None,
    right: list[HistoryEntry] | None,
) -> list[HistoryEntry]:
    out = list(left or [])
    if right:
        out.extend(right)
    return out


def _merge_awaiting_posts(
    left: dict[str, str] | None,
    right: dict[str, str] | None,
) -> dict[str, str]:
    out = dict(left or {})
    if right:
        out.update(right)
    return out


def _replace_plan(
    _left: list[PlanNode] | None,
    right: list[PlanNode] | None,
) -> list[PlanNode]:
    """The supervisor and planner are the only nodes that emit a plan. They
    each return the *whole* new plan list, so the right side wins."""
    if right is not None:
        return right
    return list(_left or [])


# ----------------------------------------------------------------- ChainState

class ChainStateSchema(TypedDict, total=False):
    """The reducer-annotated TypedDict the LangGraph runtime reads.

    This is the schema bound to ``StateGraph(ChainStateSchema)`` — the
    user-facing :class:`tasque.chains.spec.ChainState` is the *plain*
    typed-dict form callers should reach for when reading state.
    """

    chain_id: str
    chain_name: str
    bucket: str
    thread_id: str | None
    plan: Annotated[list[PlanNode], _replace_plan]
    completed: Annotated[dict[str, CompletedOutput], _merge_completed]
    failures: Annotated[dict[str, str], _merge_failures]
    replan: bool
    history: Annotated[list[HistoryEntry], _append_history]
    approval_resume: NotRequired[str | None]
    awaiting_posts: Annotated[dict[str, str], _merge_awaiting_posts]
    planner_tier: str
    # Run-time overrides supplied at chain launch — merged over the
    # spec's static ``vars`` field. Workers see this dict in their
    # prompt context so directives can branch on operator overrides
    # (e.g. ``vars.force=true`` to skip an age gate). Frozen for the
    # duration of the run.
    vars: dict[str, Any]


# ----------------------------------------------------------------- helpers

def _node_by_id(plan: list[PlanNode], node_id: str) -> PlanNode | None:
    for n in plan:
        if n["id"] == node_id:
            return n
    return None


def _resolve_dep_status(plan: list[PlanNode], dep_id: str) -> str | None:
    """Find a dep's current status, with prefix-match for fan-out children.

    A dep ``"scan"`` may be satisfied by either the literal step
    ``"scan"`` (template) being completed *or* — when a fan-out is in
    progress — by all materialized children ``"scan[0]"``, ``"scan[1]"``,
    … all being completed/terminal. We treat the template's own status
    as authoritative; the supervisor flips it to ``completed`` once
    materialization finishes.
    """
    direct = _node_by_id(plan, dep_id)
    if direct is not None:
        return direct["status"]
    return None


def _deps_satisfied(plan: list[PlanNode], node: PlanNode) -> bool:
    """True iff every dep of ``node`` is in a terminal-success state.

    "Success" means ``completed``. A dep that is ``failed``, ``stopped``,
    or ``awaiting_user`` does not satisfy.
    """
    for dep_id in node["depends_on"]:
        status = _resolve_dep_status(plan, dep_id)
        if status != "completed":
            return False
    return True


def _is_terminal_step(node: PlanNode) -> bool:
    return node["status"] in ("completed", "failed", "stopped")


class _WorkerInput(TypedDict):
    """Payload Send-dispatched to the worker node."""

    chain_id: str
    chain_name: str
    bucket: str
    step_id: str
    directive: str
    consumes_payload: dict[str, Any]
    fan_out_item: Any
    on_failure: str
    tier: str
    vars: dict[str, Any]


class _ApprovalInput(TypedDict):
    """Payload Send-dispatched to the approval node."""

    chain_id: str
    chain_name: str
    bucket: str
    step_id: str
    directive: str
    consumes_payload: dict[str, Any]


__all__ = [
    "ChainStateSchema",
    "_ApprovalInput",
    "_WorkerInput",
    "_append_history",
    "_deps_satisfied",
    "_is_terminal_step",
    "_merge_awaiting_posts",
    "_merge_completed",
    "_merge_failures",
    "_node_by_id",
    "_now_iso",
    "_replace_plan",
]

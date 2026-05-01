"""Pause / resume / stop control for live ``ChainRun`` rows.

``stop_chain`` is the load-bearing one — it must flip both the
``ChainRun`` row *and* the plan nodes inside the LangGraph checkpoint so
the supervisor's next pass actually terminates. A row-only stop would be
overwritten on the next worker write. ``pause`` / ``resume`` flip the
row status only; the chain isn't actively dispatching while paused (the
:func:`resume_interrupted_chains` startup hook re-invokes only
``running`` rows).
"""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from tasque.chains.checkpointer import get_chain_checkpointer
from tasque.chains.graph._common import _now_iso
from tasque.chains.spec import HistoryEntry, PlanNode
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun, utc_now_iso


def _thread_config(chain_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": chain_id})


def _set_run_status(chain_id: str, status: str, *, ended: bool = False) -> bool:
    with get_session() as sess:
        stmt = select(ChainRun).where(ChainRun.chain_id == chain_id)
        row = sess.execute(stmt).scalars().first()
        if row is None:
            return False
        row.status = status
        row.lease_owner = None
        row.lease_expires_at = None
        row.updated_at = utc_now_iso()
        if ended:
            row.ended_at = utc_now_iso()
    return True


def pause_chain(chain_id: str) -> bool:
    """Mark a ChainRun as ``paused``. The supervisor doesn't actively
    pause — pausing keeps the row out of ``resume_interrupted_chains`` so
    a process restart won't accidentally resume it.
    """
    return _set_run_status(chain_id, "paused")


def resume_chain(chain_id: str) -> bool:
    """Flip a paused ``ChainRun`` back to ``running``."""
    return _set_run_status(chain_id, "running")


def _patch_checkpoint_plan(
    chain_id: str,
    *,
    new_status_for: tuple[str, ...],
    target_status: str,
    history_kind: str,
) -> int:
    """Walk the latest checkpoint and flip plan-node statuses in place.

    Returns the number of nodes flipped. ``new_status_for`` is the set of
    source statuses to rewrite (e.g. ``("pending", "running",
    "awaiting_user")``).
    """
    saver = get_chain_checkpointer()
    cfg = _thread_config(chain_id)
    snapshot = saver.get_tuple(cfg)
    if snapshot is None:
        return 0

    state: dict[str, Any] = snapshot.checkpoint.get("channel_values", {})
    plan: list[PlanNode] = list(state.get("plan") or [])
    flipped = 0
    new_plan: list[PlanNode] = []
    for n in plan:
        if n["status"] in new_status_for:
            new_n: PlanNode = dict(n)  # type: ignore[assignment]
            new_n["status"] = target_status  # type: ignore[typeddict-item]
            flipped += 1
            new_plan.append(new_n)
        else:
            new_plan.append(n)

    if flipped == 0:
        return 0

    # Apply the partial update via the compiled graph so the reducer
    # logic (and history append) flow through the proper channels.
    from tasque.chains.graph import get_compiled_chain_graph

    history: list[HistoryEntry] = [
        {
            "timestamp": _now_iso(),
            "kind": cast(Any, history_kind),
            "details": {"flipped": flipped, "to": target_status},
        }
    ]
    graph = get_compiled_chain_graph()
    graph.update_state(
        cfg,
        {"plan": new_plan, "history": history},
    )
    return flipped


def stop_chain(chain_id: str) -> bool:
    """Stop a chain immediately.

    Flips the ``ChainRun`` row to ``stopped`` and walks the LangGraph
    checkpoint to flip every ``pending`` / ``running`` / ``awaiting_user``
    plan node to ``stopped`` so the next supervisor pass terminates.
    """
    if not _set_run_status(chain_id, "stopped", ended=True):
        return False
    _patch_checkpoint_plan(
        chain_id,
        new_status_for=("pending", "running", "awaiting_user"),
        target_status="stopped",
        history_kind="pause",
    )
    return True


def render_plan_tree(chain_id: str, *, verbose: bool = False) -> str:
    """Return a multi-line text rendering of the chain's current plan.

    Used by ``tasque chain show``. Indentation reflects ``depends_on``;
    fan-out children render under their template id. ``verbose=True``
    appends each step's agent-supplied ``summary`` and, for fan-out
    templates, a per-outcome roll-up of the children's bucket ids /
    failure reasons — the forensic mode that makes silent fan-out
    failures (e.g. all dispatch legs returning ``no_trades`` while the
    user-facing chain status reads green) visible at a glance.
    """
    saver = get_chain_checkpointer()
    snapshot = saver.get_tuple(_thread_config(chain_id))
    if snapshot is None:
        return f"(no checkpoint state for chain {chain_id!r})"
    state: dict[str, Any] = snapshot.checkpoint.get("channel_values", {})
    plan: list[PlanNode] = list(state.get("plan") or [])
    if not plan:
        return "(empty plan)"

    completed: dict[str, dict[str, Any]] = dict(state.get("completed") or {})
    failures: dict[str, str] = dict(state.get("failures") or {})

    parents: dict[str, list[PlanNode]] = {}
    roots: list[PlanNode] = []
    children_by_template: dict[str, list[PlanNode]] = {}
    for n in plan:
        if "[" in n["id"] and n["id"].endswith("]"):
            template = n["id"].split("[", 1)[0]
            children_by_template.setdefault(template, []).append(n)
            continue
        if not n["depends_on"]:
            roots.append(n)
        else:
            for d in n["depends_on"]:
                parents.setdefault(d, []).append(n)

    lines: list[str] = []

    def _summary_of(step_id: str) -> str | None:
        cell = completed.get(step_id)
        if not cell:
            return None
        text = (cell.get("summary") or "").strip()
        if text:
            return text
        produces = cell.get("produces")
        if isinstance(produces, dict):
            inner = (produces.get("summary") or "").strip()
            if inner:
                return inner
        return None

    def _bucket_of_child(step_id: str) -> str | None:
        cell = completed.get(step_id)
        if not cell:
            return None
        produces = cell.get("produces") or {}
        direct = produces.get("bucket_id")
        if isinstance(direct, str) and direct:
            return direct
        nested = produces.get("bucket")
        if isinstance(nested, dict):
            nid = nested.get("bucket_id")
            if isinstance(nid, str) and nid:
                return nid
        return None

    def _fanout_rollup(template_id: str) -> list[str]:
        """Return outcome lines for a fan-out template's children. Empty
        list when the template has no completed children yet."""
        kids = children_by_template.get(template_id, [])
        if not kids:
            return []
        by_outcome: dict[str, list[str]] = {}
        no_outcome: list[str] = []
        failed: list[str] = []
        for kid in kids:
            kid_id = kid["id"]
            label = _bucket_of_child(kid_id) or kid_id
            if kid["status"] == "failed" or kid_id in failures:
                reason = failures.get(kid_id) or kid.get("failure_reason") or "?"
                failed.append(f"{label}: {reason}")
                continue
            cell = completed.get(kid_id)
            if not cell:
                no_outcome.append(f"{label} (status={kid['status']})")
                continue
            produces = cell.get("produces") or {}
            outcome = produces.get("outcome")
            if isinstance(outcome, str) and outcome:
                by_outcome.setdefault(outcome, []).append(label)
            else:
                no_outcome.append(label)
        out: list[str] = []
        for outcome in sorted(by_outcome.keys()):
            buckets = by_outcome[outcome]
            out.append(f"{outcome} ({len(buckets)}): " + ", ".join(buckets))
        if no_outcome:
            out.append(f"(no outcome) ({len(no_outcome)}): " + ", ".join(no_outcome))
        if failed:
            out.append(f"failed ({len(failed)}):")
            for f in failed:
                out.append(f"  - {f}")
        return out

    def _walk(node: PlanNode, depth: int) -> None:
        indent = "  " * depth
        lines.append(
            f"{indent}- {node['id']}  [{node['kind']} / {node['status']}]"
        )
        if verbose:
            summary = _summary_of(node["id"])
            if summary:
                lines.append(f"{indent}    summary: {summary}")
            reason = failures.get(node["id"]) or node.get("failure_reason")
            if reason:
                lines.append(f"{indent}    failure: {reason}")
            rollup = _fanout_rollup(node["id"])
            if rollup:
                lines.append(f"{indent}    fan-out outcomes:")
                for line in rollup:
                    lines.append(f"{indent}      - {line}")
        for child in children_by_template.get(node["id"], []):
            child_indent = "  " * (depth + 1)
            lines.append(
                f"{child_indent}- {child['id']}  "
                f"[{child['kind']} / {child['status']}]"
            )
            if verbose:
                summary = _summary_of(child["id"])
                if summary:
                    lines.append(f"{child_indent}    summary: {summary}")
                reason = failures.get(child["id"]) or child.get("failure_reason")
                if reason:
                    lines.append(f"{child_indent}    failure: {reason}")
        for nxt in parents.get(node["id"], []):
            _walk(nxt, depth + 1)

    for r in roots:
        _walk(r, 0)
    return "\n".join(lines)


def get_chain_state(chain_id: str) -> dict[str, Any] | None:
    """Return a snapshot dict of the chain's checkpoint state (or None)."""
    saver = get_chain_checkpointer()
    snapshot = saver.get_tuple(_thread_config(chain_id))
    if snapshot is None:
        return None
    raw: dict[str, Any] = snapshot.checkpoint.get("channel_values", {})
    return json.loads(json.dumps(raw, default=str))


__all__ = [
    "get_chain_state",
    "pause_chain",
    "render_plan_tree",
    "resume_chain",
    "stop_chain",
]

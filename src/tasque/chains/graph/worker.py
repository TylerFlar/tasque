"""The chain worker node.

A Send arrives with a :class:`_WorkerInput`. We synthesize a transient
``QueuedJob`` (not persisted), call :func:`tasque.jobs.runner.run_worker`
with the consumes payload, and write the outcome back to chain state:

* success → ``completed[step_id] = {report, produces}``
* failure → ``failures[step_id] = reason``

If the step had ``on_failure="replan"`` and failed, we also flip
``state.replan = True`` so the supervisor routes to the planner next pass.

The worker LLM gets the tasque MCP injected by the selected upstream (see
``src/tasque/mcp/server.py``). Side-effects — queueing a follow-up job,
writing a Note, firing a sub-chain, sending a Signal — happen
synchronously through MCP tool calls during the worker's turn. The
worker puts any ids or structured data downstream chain steps need
into ``produces`` for the chain engine's ``consumes`` resolution.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from tasque.chains.graph._common import _now_iso, _WorkerInput
from tasque.chains.spec import (
    CompletedOutput,
    HistoryEntry,
    ProducesSchema,
    validate_produces,
)
from tasque.jobs.runner import run_worker
from tasque.memory.entities import QueuedJob

log = structlog.get_logger(__name__)


def _augment_directive_with_fan_out(directive: str, item: Any) -> str:
    if item is None:
        return directive
    try:
        rendered = json.dumps(item, indent=2, default=str)
    except (TypeError, ValueError):
        rendered = repr(item)
    return (
        f"{directive}\n\n"
        f"Fan-out item (focus on this single element):\n{rendered}"
    )


def _augment_directive_with_schema(
    directive: str, schema: ProducesSchema | None
) -> str:
    """Append a produces_schema contract to the directive if one is set.

    The chain engine validates produces against this schema after the
    worker submits its result; surfacing the schema in the prompt lets
    the LLM produce a correctly-shaped dict on the first try instead of
    relying on the post-hoc check to catch (and fail) drift.
    """
    if not schema:
        return directive
    required = schema.get("required") or []
    list_items = schema.get("list_items") or {}
    parts: list[str] = ["Required produces shape (enforced after submit):"]
    if required:
        parts.append("- top-level required keys: " + ", ".join(required))
    for key, item_schema in list_items.items():
        item_required = item_schema.get("required") or []
        bounds: list[str] = []
        min_count = item_schema.get("min_count")
        if isinstance(min_count, int):
            bounds.append(f"min={min_count}")
        max_count = item_schema.get("max_count")
        if isinstance(max_count, int):
            bounds.append(f"max={max_count}")
        bounds_str = f" ({', '.join(bounds)})" if bounds else ""
        if item_required:
            parts.append(
                f"- {key!r} is a list of dicts; each item MUST contain "
                f"keys {item_required!r}{bounds_str}"
            )
        elif bounds:
            parts.append(f"- {key!r} is a list{bounds_str}")
    parts.append(
        "Drift on this shape (e.g. dropping a key from list items) is a "
        "hard failure — the chain rejects the step and routes to DLQ."
    )
    return f"{directive}\n\n" + "\n".join(parts)


def _build_synthetic_job(state: _WorkerInput) -> QueuedJob:
    """Create an in-memory QueuedJob for the worker graph; not persisted.

    The chain worker doesn't write a row to ``queued_jobs`` — chain steps
    flow through the chain checkpointer instead. The worker LLM still
    expects a QueuedJob shape, so we hand it one.
    """
    directive = _augment_directive_with_fan_out(
        state["directive"], state.get("fan_out_item")
    )
    directive = _augment_directive_with_schema(
        directive, state.get("produces_schema")
    )
    return QueuedJob(
        kind="worker",
        bucket=state["bucket"] or None,
        directive=directive,
        reason=f"chain={state['chain_id']}, step={state['step_id']}",
        fire_at="now",
        status="claimed",
        queued_by="chain",
        thread_id=None,
        chain_id=state["chain_id"] or None,
        chain_step_id=state["step_id"],
        tier=state["tier"],
    )


def worker(state: _WorkerInput) -> dict[str, Any]:
    """Run one chain worker step.

    The Send dispatches the worker with a :class:`_WorkerInput`, not the
    full ChainState — that's the langgraph map/reduce convention. We
    return only the slices of state we want to merge: ``completed`` /
    ``failures`` (via reducer) and an appended ``history`` entry.
    """
    step_id = state["step_id"]
    job = _build_synthetic_job(state)
    consumes = state.get("consumes_payload") or {}
    chain_vars = state.get("vars") or {}

    try:
        result = run_worker(job, consumes=consumes, vars=chain_vars)
    except Exception as exc:
        reason = f"worker raised: {type(exc).__name__}: {exc}"
        history: list[HistoryEntry] = [
            {
                "timestamp": _now_iso(),
                "kind": "status",
                "details": {"step": step_id, "to": "failed", "reason": reason},
            }
        ]
        update: dict[str, Any] = {
            "failures": {step_id: reason},
            "history": history,
        }
        if state["on_failure"] == "replan":
            update["replan"] = True
        return update

    err = result.get("error")
    if err:
        history = [
            {
                "timestamp": _now_iso(),
                "kind": "status",
                "details": {"step": step_id, "to": "failed", "reason": err},
            }
        ]
        update = {
            "failures": {step_id: err},
            "history": history,
        }
        if state["on_failure"] == "replan":
            update["replan"] = True
        return update

    produces_dict: dict[str, Any] = dict(result["produces"])

    schema_errors = validate_produces(state.get("produces_schema"), produces_dict)
    if schema_errors:
        # The worker thinks it succeeded, but its produces dict doesn't
        # satisfy the declared schema. Treat as a hard failure so the
        # bug surfaces instead of corrupting downstream state.
        reason = "produces_schema violation: " + "; ".join(schema_errors)
        history = [
            {
                "timestamp": _now_iso(),
                "kind": "status",
                "details": {"step": step_id, "to": "failed", "reason": reason},
            }
        ]
        log.warning(
            "chain.worker.produces_schema_violation",
            chain_id=state.get("chain_id"),
            step_id=step_id,
            errors=schema_errors,
        )
        update = {
            "failures": {step_id: reason},
            "history": history,
        }
        if state["on_failure"] == "replan":
            update["replan"] = True
        return update

    output: CompletedOutput = {
        "report": result["report"],
        "produces": produces_dict,
    }
    history = [
        {
            "timestamp": _now_iso(),
            "kind": "status",
            "details": {"step": step_id, "to": "completed"},
        }
    ]
    return {
        "completed": {step_id: output},
        "history": history,
    }


__all__ = ["worker"]

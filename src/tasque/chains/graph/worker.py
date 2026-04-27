"""The chain worker node.

A Send arrives with a :class:`_WorkerInput`. We synthesize a transient
``QueuedJob`` (not persisted), call :func:`tasque.jobs.runner.run_worker`
with the consumes payload, and write the outcome back to chain state:

* success → ``completed[step_id] = {report, produces}``
* failure → ``failures[step_id] = reason``

If the step had ``on_failure="replan"`` and failed, we also flip
``state.replan = True`` so the supervisor routes to the planner next pass.

The worker LLM gets the tasque MCP injected by ``claude --print`` (see
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
from tasque.chains.spec import CompletedOutput, HistoryEntry
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


def _build_synthetic_job(state: _WorkerInput) -> QueuedJob:
    """Create an in-memory QueuedJob for the worker graph; not persisted.

    The chain worker doesn't write a row to ``queued_jobs`` — chain steps
    flow through the chain checkpointer instead. The worker LLM still
    expects a QueuedJob shape, so we hand it one.
    """
    directive = _augment_directive_with_fan_out(
        state["directive"], state.get("fan_out_item")
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

    output: CompletedOutput = {
        "report": result["report"],
        "produces": dict(result["produces"]),
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

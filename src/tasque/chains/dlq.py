"""Chain DLQ retry.

Failed chain steps live in two places: a ``FailedJob`` row (so the user
can find them via ``tasque dlq list``) and a ``failures[step_id]`` entry
in the chain's checkpoint state. ``dlq_retry`` clears both and re-invokes
the meta-graph so the supervisor can re-dispatch the step.

The None-on-right semantics of :func:`_merge_failures` is what lets us
clear the entry by writing ``{step_id: None}``; without it, the
supervisor would still see the old failure on the next pass and never
promote the step back to ``running``.
"""

from __future__ import annotations

from typing import Any, cast

import structlog
from langchain_core.runnables import RunnableConfig

from tasque.chains.checkpointer import get_chain_checkpointer
from tasque.chains.graph import get_compiled_chain_graph
from tasque.chains.graph._common import _now_iso
from tasque.chains.spec import HistoryEntry, PlanNode

log = structlog.get_logger(__name__)


def _thread_config(chain_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": chain_id})


def dlq_retry(chain_id: str, step_id: str) -> bool:
    """Re-fire ``step_id`` of ``chain_id``.

    Returns True iff the chain checkpoint exists, the step was found in
    ``failed`` status, and we successfully patched the state. False if
    the chain or step doesn't exist; raises if the graph re-invocation
    fails.
    """
    saver = get_chain_checkpointer()
    cfg = _thread_config(chain_id)
    snapshot = saver.get_tuple(cfg)
    if snapshot is None:
        log.warning("chains.dlq.no_checkpoint", chain_id=chain_id)
        return False

    state: dict[str, Any] = snapshot.checkpoint.get("channel_values", {})
    plan_in: list[PlanNode] = list(state.get("plan") or [])
    target: PlanNode | None = next(
        (n for n in plan_in if n["id"] == step_id), None
    )
    if target is None:
        log.warning(
            "chains.dlq.unknown_step", chain_id=chain_id, step_id=step_id
        )
        return False
    if target["status"] != "failed":
        log.warning(
            "chains.dlq.step_not_failed",
            chain_id=chain_id,
            step_id=step_id,
            status=target["status"],
        )
        return False

    new_plan: list[PlanNode] = [dict(n) for n in plan_in]  # type: ignore[misc]
    for n in new_plan:
        if n["id"] == step_id:
            n["status"] = "pending"
            n["failure_reason"] = None

    history: list[HistoryEntry] = [
        {
            "timestamp": _now_iso(),
            "kind": "resume_pause",
            "details": {"step": step_id, "to": "pending", "via": "dlq"},
        }
    ]
    # Critical: the {step_id: None} value triggers the
    # _merge_failures None-clears-key reducer behavior. A naive merge
    # would leave the old failure entry, which is exactly the bug the
    # contract calls out.
    update: dict[str, Any] = {
        "plan": new_plan,
        "failures": cast(dict[str, str], {step_id: None}),
        "history": history,
        "replan": False,
    }

    graph = get_compiled_chain_graph()
    # Mark this update as if the worker had emitted it, so the graph's
    # next step is the worker→supervisor edge — supervisor re-promotes
    # our newly-pending step and dispatches it via Send.
    from tasque.chains.graph.supervisor import WORKER_NODE

    graph.update_state(cfg, update, as_node=WORKER_NODE)

    try:
        graph.invoke(cast(Any, None), cfg)
    except Exception:
        log.exception(
            "chains.dlq.reinvoke_failed", chain_id=chain_id, step_id=step_id
        )
        raise

    # Sync the ChainRun row terminal-status if needed.
    from tasque.chains.scheduler import maybe_finalize_status

    maybe_finalize_status(chain_id)
    return True


__all__ = ["dlq_retry"]

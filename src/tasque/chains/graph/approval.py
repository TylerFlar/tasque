"""The chain approval node.

The approval node yields control via :func:`langgraph.types.interrupt`
with a payload describing what's being asked. The graph pauses; outside
code (the Discord button handlers) calls
``graph.invoke(Command(resume=value), ...)`` and the node resumes with
``value`` returned from ``interrupt(...)``.

On resume we:
- mark the step ``completed``
- write the user's reply into the step's produces (so downstream consumers
  can see the decision)
- append a ``resume`` history entry

The LLM-roundtrip-on-resume is optional. The contract calls for
"applying the user's reply to the upstream proposal via an LLM call
(or pass-through if no LLM is needed)"; we go with pass-through here.
The planner is the place to do further LLM mediation if needed.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from tasque.chains.graph._common import _ApprovalInput, _now_iso
from tasque.chains.spec import CompletedOutput, HistoryEntry


def approval(state: _ApprovalInput) -> dict[str, Any]:
    """Suspend until the outside world resumes us with a reply.

    The interrupt payload carries enough context for a Discord button
    renderer to construct buttons (``approve`` / ``reject`` / freeform
    text). On resume, ``user_reply`` is whatever value was passed via
    ``Command(resume=...)``.
    """
    step_id = state["step_id"]
    interrupt_payload: dict[str, Any] = {
        "kind": "approval",
        "chain_id": state["chain_id"],
        "chain_name": state["chain_name"],
        "bucket": state["bucket"],
        "step_id": step_id,
        "directive": state["directive"],
        "consumes_payload": state["consumes_payload"],
    }
    user_reply = interrupt(interrupt_payload)

    output: CompletedOutput = {
        "report": f"Approval resolved by user: {user_reply!r}",
        "produces": {"user_reply": user_reply},
    }
    history: list[HistoryEntry] = [
        {
            "timestamp": _now_iso(),
            "kind": "resume",
            "details": {
                "step": step_id,
                "user_reply": user_reply if isinstance(user_reply, (str, int, float, bool)) else repr(user_reply),
            },
        }
    ]
    return {
        "completed": {step_id: output},
        "history": history,
        "approval_resume": str(user_reply) if user_reply is not None else None,
    }


__all__ = ["approval"]

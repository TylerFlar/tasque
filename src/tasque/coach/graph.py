"""The single LangGraph subgraph that powers all nine bucket coaches.

Five linear nodes:

    gather_context → build_prompt → call_llm → parse_response → persist_results

Same graph, parameterised by ``bucket`` carried in the state. The graph is
compiled once and reused. The ``llm`` argument on ``run_bucket_coach`` lets
tests inject a fake ``BaseChatModel`` returning canned JSON.

Writes happen via the tasque MCP injected by the selected upstream (see
``src/tasque/mcp/server.py``) during the LLM's turn. The structured JSON
the coach emits afterwards carries one field — ``thread_post`` — a
runtime-controlled signal asking the Discord bot to publish a message
to the bucket's thread on the coach's behalf.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast
from zoneinfo import ZoneInfo

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from tasque.agents import result_inbox
from tasque.buckets import Bucket
from tasque.coach.output import BucketCoachOutput
from tasque.coach.persist import persist_results
from tasque.coach.prompts import build_system_prompt
from tasque.config import get_settings
from tasque.llm.factory import get_chat_model
from tasque.memory.entities import Aim, Note, QueuedJob, Signal
from tasque.memory.repo import query_bucket, query_signals_for

# Tools the bucket-coach should not call from background/reactive runs.
# Aim decomposition is deliberately local to the coach: turn the Aim into
# one concrete job or a hand-written chain, instead of asking another LLM
# to synthesize a generic chain plan.
_ALWAYS_DISALLOWED_TOOLS: tuple[str, ...] = (
    "mcp__tasque__aim_plan_chain",
)

# Tools a consolidation-only bucket-coach pass MUST NOT call. The
# synchronous Discord reply path already had a chance to execute any
# user-initiated request. Without this gate, a second pass could see the
# same ephemeral note and duplicate a job or chain.
_CONSOLIDATION_DISALLOWED_TOOLS: tuple[str, ...] = (
    "mcp__tasque__chain_fire_template",
    "mcp__tasque__chain_queue_adhoc",
    "mcp__tasque__job_create",
)

# These triggers are bookkeeping reactions, not invitations to start the
# next worker. Keep them consolidation-only so a completed standalone job
# cannot chain itself into another near-identical job.
_REACTION_ONLY_TRIGGER_PREFIXES: tuple[str, ...] = (
    "job-completed:",
    "tool:note_create:",
    "tool:note_update:",
    "tool:note_supersede:",
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class BucketCoachState(TypedDict):
    """State threaded through the bucket-coach LangGraph.

    ``bucket`` and ``reason`` are required at invocation; the rest are
    populated by graph nodes and absent until the relevant node has run.
    """

    bucket: Bucket
    reason: str
    result_token: NotRequired[str]
    llm: NotRequired[BaseChatModel | None]
    notes: NotRequired[list[Note]]
    active_aims: NotRequired[list[Aim]]
    signals: NotRequired[list[Signal]]
    queued_jobs: NotRequired[list[QueuedJob]]
    messages: NotRequired[list[BaseMessage]]
    raw_response: NotRequired[str]
    parsed: NotRequired[BucketCoachOutput]
    persisted: NotRequired[dict[str, Any]]
    error: NotRequired[str]


def _gather_context(state: BucketCoachState) -> dict[str, Any]:
    bucket = state["bucket"]
    rows = query_bucket(bucket)
    notes = [
        r
        for r in rows
        if isinstance(r, Note)
        and not r.archived
        and (r.memory_kind or "") != "artifact"
    ]
    active_aims = [
        r for r in rows if isinstance(r, Aim) and r.status == "active"
    ]
    open_jobs = [
        r
        for r in rows
        if isinstance(r, QueuedJob) and r.status in {"pending", "claimed"}
    ]
    signals = query_signals_for(bucket)
    return {
        "notes": notes,
        "active_aims": active_aims,
        "signals": signals,
        "queued_jobs": open_jobs,
    }


_NOTE_CONTENT_TRUNCATE = 200
_RECENT_EPHEMERAL_CAP = 5


def _truncate(text: str, n: int = _NOTE_CONTENT_TRUNCATE) -> str:
    """Collapse internal whitespace and clip to ``n`` chars with an ellipsis."""
    flat = " ".join(text.split())
    if len(flat) <= n:
        return flat
    return flat[: n - 1].rstrip() + "…"


def _format_state_block(state: BucketCoachState) -> str:
    parts: list[str] = []

    notes = state.get("notes") or []
    behavioral = [n for n in notes if n.durability == "behavioral"]
    durable = [
        n
        for n in notes
        if n.durability == "durable"
        and (n.memory_kind or "") in {"", "fact", "preference", "policy", "summary"}
    ]
    ephemeral = sorted(
        (n for n in notes if n.durability == "ephemeral"),
        key=lambda n: n.updated_at,
        reverse=True,
    )

    if behavioral:
        parts.append("### Behavioral instructions (always honor)")
        for n in behavioral:
            parts.append(f"- ({n.id}) {n.content}")
    else:
        parts.append("### Behavioral instructions\n_(none)_")

    # Curated durable notes are NOT pre-injected. The bucket has potentially
    # hundreds of them; dumping all into every prompt floods the context.
    # Use the note_search_fts / note_search / note_list / note_get MCP tools
    # to pull the specific durable facts relevant to this run's trigger.
    parts.append(
        f"\n### Curated durable memory ({len(durable)} present in this bucket)"
        f"\n_Not pre-injected. Call `note_search_fts(query, durability='durable')` "
        f"or `note_list(durability='durable')` to retrieve specific facts "
        f"relevant to this run._"
    )

    ephemeral_total = len(ephemeral)
    ephemeral_shown = ephemeral[:_RECENT_EPHEMERAL_CAP]
    if ephemeral_shown:
        suffix = (
            f" — showing {_RECENT_EPHEMERAL_CAP} most recent of {ephemeral_total}"
            if ephemeral_total > _RECENT_EPHEMERAL_CAP
            else ""
        )
        parts.append(f"\n### Recent activity (ephemeral, most-recent first{suffix})")
        for n in ephemeral_shown:
            parts.append(f"- ({n.id}) {_truncate(n.content)}")
    else:
        parts.append("\n### Recent activity\n_(none)_")

    active_aims = sorted(
        state.get("active_aims") or [],
        key=lambda a: a.updated_at,
        reverse=True,
    )
    if active_aims:
        parts.append("\n### Active aims")
        for a in active_aims[:8]:
            target = f", target={a.target_date}" if a.target_date else ""
            parent = f", parent={a.parent_id}" if a.parent_id else ""
            desc = f" - {_truncate(a.description, 240)}" if a.description else ""
            parts.append(f"- ({a.id}) {a.title}{target}{parent}{desc}")
    else:
        parts.append("\n### Active aims\n_(none)_")

    signals = state.get("signals") or []
    if signals:
        parts.append("\n### Signals from other coaches")
        for s in signals:
            parts.append(
                f"- ({s.id}) from {s.from_bucket} → {s.to_bucket} | "
                f"{s.kind} | {s.urgency}: {s.summary}"
            )
    else:
        parts.append("\n### Signals from other coaches\n_(none)_")

    open_jobs = state.get("queued_jobs") or []
    if open_jobs:
        parts.append("\n### Open queued jobs")
        for j in open_jobs:
            parts.append(
                f"- ({j.id}) {j.directive} "
                f"[fire_at={j.fire_at}, status={j.status}]"
            )
    else:
        parts.append("\n### Open queued jobs\n_(none)_")

    return "\n".join(parts)


def _format_time_block(reason: str) -> tuple[str, str, str]:
    settings = get_settings()
    tz_name = settings.tasque_timezone or "UTC"
    local_tz: Any
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        # Windows boxes lacking the `tzdata` package can't resolve IANA
        # zone names — fall back to UTC so the coach still runs.
        local_tz = UTC
        tz_name = "UTC"
    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(local_tz)
    return (
        now_utc.strftime("%Y-%m-%d %H:%M:%S %Z"),
        now_local.strftime("%Y-%m-%d %H:%M:%S"),
        tz_name,
    )


def _is_reaction_only_trigger(reason: str) -> bool:
    return any(reason.startswith(prefix) for prefix in _REACTION_ONLY_TRIGGER_PREFIXES)


def _build_prompt(state: BucketCoachState) -> dict[str, Any]:
    bucket = state["bucket"]
    reason = state.get("reason", "")
    # System prompt = scaffold + bucket-mindset, with NO per-run substitutions
    # so upstream prefix caches can hit across runs of the same bucket when
    # supported. Time + reason live in the user message instead.
    system_text = build_system_prompt(bucket)
    now_utc_s, now_local_s, tz_name = _format_time_block(reason)
    state_block = _format_state_block(state)
    token = state.get("result_token") or result_inbox.mint_token()
    is_post_reply = reason == "reply"
    is_reaction_only = _is_reaction_only_trigger(reason)
    if is_post_reply:
        action_guidance = (
            "**Reply-consolidation pass.** The user just received a synchronous reply "
            "in this thread; that reply already executed any action the user "
            "asked for. Your job here is consolidation only — note_create, "
            "note_update, note_supersede, note_archive, signal_create, etc. "
            "The user-action tools "
            "(``chain_fire_template``, ``chain_queue_adhoc``, ``job_create``) "
            "are disabled for this pass; do not attempt them."
        )
    elif is_reaction_only:
        action_guidance = (
            "**Reaction-only pass.** This trigger came from bookkeeping "
            "(for example worker completion or a Note edit). Your job here "
            "is consolidation only: promote one compact fact/summary if it "
            "materially improves memory, archive handled Signals if needed, "
            "or do nothing. The user-action tools "
            "(``chain_fire_template``, ``chain_queue_adhoc``, ``job_create``) "
            "are disabled for this pass; do not attempt them."
        )
    else:
        action_guidance = (
            "Perform any writes via the tasque MCP now. For new Aims or "
            "`aim_added` signals, immediately translate the Aim into one "
            "concrete `job_create` call, or a small hand-written chain when "
            "one worker is not enough. Do not call `aim_plan_chain`; do not "
            "write a strategy essay first. If the Aim is blocked on missing "
            "user context, ask one concrete question in `thread_post` instead. "
            "Use `note_create`, `note_update`, and `signal_create` only when "
            "they materially help."
        )
    user_text = (
        "## Run context\n"
        f"- Bucket: {bucket}\n"
        f"- Current time (UTC): {now_utc_s}\n"
        f"- Current time (local): {now_local_s} ({tz_name})\n"
        f"- Trigger reason: {reason or '(no reason given)'}\n"
        f"- result_token: {token}  "
        "(pass this to submit_coach_result)\n\n"
        f"{state_block}\n\n"
        f"{action_guidance} When you're done, call "
        "``submit_coach_result(result_token=<above>, thread_post=...)`` "
        "exactly once — pass ``thread_post=None`` for the common 'no "
        "announcement' case, or a markdown string to ask the bot to post "
        "to this bucket's thread."
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_text),
    ]
    return {"messages": messages, "result_token": token}


def _call_llm(state: BucketCoachState) -> dict[str, Any]:
    llm = state.get("llm")
    if llm is None:
        reason = state.get("reason", "") or ""
        # Reply and bookkeeping triggers are consolidation passes. The
        # synchronous reply, explicit Aim/Signal path, scheduler, or chain
        # engine owns starting user-visible work.
        disallowed = list(_ALWAYS_DISALLOWED_TOOLS)
        if reason == "reply" or _is_reaction_only_trigger(reason):
            disallowed.extend(_CONSOLIDATION_DISALLOWED_TOOLS)
        llm = get_chat_model("coach", disallowed_tools=disallowed)
    messages = state.get("messages") or []
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        parts: list[str] = []
        for item in cast(list[Any], content):
            if isinstance(item, dict):
                d = cast(dict[str, Any], item)
                t = d.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(item, str):
                parts.append(item)
        text = "\n".join(parts)
    else:
        text = str(content)
    return {"raw_response": text}


def _parse_response(state: BucketCoachState) -> dict[str, Any]:
    token = state.get("result_token")
    if not token:
        return {"error": "coach run had no result_token in state"}
    payload = result_inbox.read_and_consume(token, agent_kind="coach")
    if payload is None:
        return {
            "error": (
                "coach did not call submit_coach_result during its turn — "
                "no structured result was deposited in the inbox"
            )
        }
    try:
        parsed = BucketCoachOutput.model_validate(payload)
    except ValidationError as exc:
        return {"error": f"invalid submit_coach_result payload: {exc}"}
    return {"parsed": parsed}


def _persist_results(state: BucketCoachState) -> dict[str, Any]:
    if "error" in state:
        return {}
    parsed = state.get("parsed")
    if parsed is None:
        return {"error": "no parsed output to persist"}
    bucket = state["bucket"]
    report = persist_results(bucket, parsed)
    return {"persisted": report}


_compiled: CompiledStateGraph[BucketCoachState, Any, BucketCoachState, BucketCoachState] | None = None


def build_graph() -> CompiledStateGraph[BucketCoachState, Any, BucketCoachState, BucketCoachState]:
    """Construct (and cache) the bucket-coach LangGraph."""
    global _compiled
    if _compiled is not None:
        return _compiled
    sg: StateGraph[BucketCoachState, Any, BucketCoachState, BucketCoachState] = StateGraph(
        BucketCoachState
    )
    sg.add_node("gather_context", _gather_context)
    sg.add_node("build_prompt", _build_prompt)
    sg.add_node("call_llm", _call_llm)
    sg.add_node("parse_response", _parse_response)
    sg.add_node("persist_results", _persist_results)
    sg.add_edge(START, "gather_context")
    sg.add_edge("gather_context", "build_prompt")
    sg.add_edge("build_prompt", "call_llm")
    sg.add_edge("call_llm", "parse_response")
    sg.add_edge("parse_response", "persist_results")
    sg.add_edge("persist_results", END)
    _compiled = sg.compile()
    return _compiled


def run_bucket_coach(
    bucket: Bucket,
    *,
    reason: str = "",
    llm: BaseChatModel | None = None,
) -> BucketCoachState:
    """Run the bucket coach for ``bucket`` once and return the final state.

    Tests pass ``llm=`` to inject a fake chat model returning canned
    JSON. Production callers leave it ``None`` so the factory wires a
    real ``ChatOpenAI`` against the proxy.
    """
    graph = build_graph()
    initial: BucketCoachState = {
        "bucket": bucket,
        "reason": reason,
        "llm": llm,
    }
    final = graph.invoke(initial)
    return cast(BucketCoachState, final)

"""The monitoring LangGraph for the strategist.

Five linear nodes mirror the bucket-coach pattern:

    gather_cross_bucket_context → build_prompt → call_llm
        → parse_response → persist_results

State is :class:`StrategistState`. The graph is compiled once and
reused. Tests inject a fake ``BaseChatModel`` via the ``llm`` argument
on :func:`run_monitoring`.

The decomposition flow does NOT use this graph — that one runs through
the parameterised reply runtime in :mod:`tasque.reply.strategist` so
the user can converse with the strategist normally. This graph is the
scheduled ambient pass.

The graph node only does DB writes. Posting the markdown summary to
Discord happens in :func:`run_monitoring_and_post`, an async wrapper
that orchestrates the synchronous graph and the async poster — keeping
the graph itself free of asyncio entanglement.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast
from zoneinfo import ZoneInfo

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError
from sqlalchemy import select

from tasque.buckets import ALL_BUCKETS, Bucket
from tasque.coach.output import extract_json_block
from tasque.config import get_settings
from tasque.llm.factory import get_chat_model
from tasque.memory.db import get_session
from tasque.memory.entities import (
    Aim,
    FailedJob,
    Note,
    QueuedJob,
    Signal,
)
from tasque.strategist.output import StrategistOutput
from tasque.strategist.persist import persist_results, post_summary

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

DEFAULT_HORIZON_DAYS = 14
DEFAULT_RECENT_NOTE_LIMIT = 8
DEFAULT_RECENT_FAILURE_LIMIT = 5
DEFAULT_RECENT_SIGNAL_LIMIT = 10

# Sentinel inside a chain step's ``directive`` that flags the worker to
# dispatch the strategist monitoring run instead of calling the worker
# LLM. Used by ``chains/templates/strategist-weekly.yaml``.
STRATEGIST_DIRECTIVE_SENTINEL = "[strategist:monitor]"


def _load_strategist_prompt() -> str:
    """Read ``prompts/strategist.md`` from the repo root."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "prompts" / "strategist.md"
    return path.read_text(encoding="utf-8")


class StrategistState(TypedDict):
    """State threaded through the strategist monitoring LangGraph."""

    reason: str
    horizon_days: NotRequired[int]
    llm: NotRequired[BaseChatModel | None]
    snapshot: NotRequired[dict[str, Any]]
    messages: NotRequired[list[BaseMessage]]
    raw_response: NotRequired[str]
    parsed: NotRequired[StrategistOutput]
    persisted: NotRequired[dict[str, Any]]
    posted_message_ids: NotRequired[list[int]]
    error: NotRequired[str]


# ----------------------------------------------------------------- gather


def _bucket_snapshot(bucket: Bucket) -> dict[str, Any]:
    with get_session() as sess:
        notes_stmt = (
            select(Note)
            .where(Note.bucket == bucket)
            .where(Note.archived.is_(False))
            .where(Note.source.in_(("coach", "coach-reply")))
            .order_by(Note.created_at.desc())
            .limit(DEFAULT_RECENT_NOTE_LIMIT)
        )
        notes = list(sess.execute(notes_stmt).scalars().all())

        aims_stmt = (
            select(Aim)
            .where(Aim.bucket == bucket)
            .where(Aim.status == "active")
            .order_by(Aim.created_at.desc())
        )
        aims = list(sess.execute(aims_stmt).scalars().all())

        jobs_stmt = (
            select(QueuedJob)
            .where(QueuedJob.bucket == bucket)
            .where(QueuedJob.status == "pending")
            .order_by(QueuedJob.created_at.desc())
        )
        pending_jobs = list(sess.execute(jobs_stmt).scalars().all())

        failures_stmt = (
            select(FailedJob)
            .where(FailedJob.bucket == bucket)
            .where(FailedJob.resolved.is_(False))
            .order_by(FailedJob.created_at.desc())
            .limit(DEFAULT_RECENT_FAILURE_LIMIT)
        )
        failures = list(sess.execute(failures_stmt).scalars().all())

        signals_stmt = (
            select(Signal)
            .where(
                (Signal.to_bucket == bucket)
                | (Signal.from_bucket == bucket)
            )
            .where(Signal.archived.is_(False))
            .order_by(Signal.created_at.desc())
            .limit(DEFAULT_RECENT_SIGNAL_LIMIT)
        )
        signals = list(sess.execute(signals_stmt).scalars().all())
        sess.expunge_all()

    return {
        "bucket": bucket,
        "recent_notes": [
            {
                "id": n.id,
                "content": n.content,
                "durability": n.durability,
                "source": n.source,
                "created_at": n.created_at,
            }
            for n in notes
        ],
        "active_aims": [
            {
                "id": a.id,
                "title": a.title,
                "scope": a.scope,
                "target_date": a.target_date,
                "parent_id": a.parent_id,
                "broken_down_at": dict(a.broken_down_at or {}),
                "source": a.source,
                "created_at": a.created_at,
            }
            for a in aims
        ],
        "pending_jobs": [
            {
                "id": j.id,
                "directive": j.directive,
                "fire_at": j.fire_at,
                "queued_by": j.queued_by,
                "created_at": j.created_at,
            }
            for j in pending_jobs
        ],
        "recent_failures": [
            {
                "id": f.id,
                "error_type": f.error_type,
                "error_message": f.error_message,
                "failure_timestamp": f.failure_timestamp,
            }
            for f in failures
        ],
        "recent_signals": [
            {
                "id": s.id,
                "from_bucket": s.from_bucket,
                "to_bucket": s.to_bucket,
                "kind": s.kind,
                "urgency": s.urgency,
                "summary": s.summary,
                "created_at": s.created_at,
            }
            for s in signals
        ],
    }


def _long_term_aims_snapshot() -> list[dict[str, Any]]:
    with get_session() as sess:
        stmt = (
            select(Aim)
            .where(Aim.scope == "long_term")
            .where(Aim.status == "active")
            .order_by(Aim.created_at.desc())
        )
        aims = list(sess.execute(stmt).scalars().all())
        sess.expunge_all()
    return [
        {
            "id": a.id,
            "title": a.title,
            "target_date": a.target_date,
            "broken_down_at": dict(a.broken_down_at or {}),
            "source": a.source,
            "created_at": a.created_at,
        }
        for a in aims
    ]


def _upcoming_target_dates(horizon_days: int) -> list[dict[str, Any]]:
    """All active Aims whose ``target_date`` falls inside the horizon."""
    today = datetime.now(UTC).date()
    horizon = today + timedelta(days=horizon_days)
    items: list[dict[str, Any]] = []
    with get_session() as sess:
        stmt = (
            select(Aim)
            .where(Aim.status == "active")
            .where(Aim.target_date.isnot(None))
        )
        aims = list(sess.execute(stmt).scalars().all())
        for a in aims:
            td = a.target_date
            if not td:
                continue
            try:
                td_date = datetime.strptime(td, "%Y-%m-%d").date()
            except ValueError:
                continue
            if td_date <= horizon:
                items.append(
                    {
                        "id": a.id,
                        "title": a.title,
                        "scope": a.scope,
                        "bucket": a.bucket,
                        "target_date": td,
                        "days_until": (td_date - today).days,
                    }
                )
        sess.expunge_all()
    items.sort(key=lambda x: cast(int, x["days_until"]))
    return items


def _gather_context(state: StrategistState) -> dict[str, Any]:
    horizon_days = state.get("horizon_days") or DEFAULT_HORIZON_DAYS
    snapshot: dict[str, Any] = {
        "horizon_days": horizon_days,
        "long_term_aims": _long_term_aims_snapshot(),
        "buckets": {b: _bucket_snapshot(b) for b in ALL_BUCKETS},
        "upcoming_target_dates": _upcoming_target_dates(horizon_days),
    }
    return {"snapshot": snapshot}


# ----------------------------------------------------------------- prompt


def _format_time_block() -> tuple[str, str, str]:
    settings = get_settings()
    tz_name = settings.tasque_timezone or "UTC"
    local_tz: Any
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = UTC
        tz_name = "UTC"
    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(local_tz)
    return (
        now_utc.strftime("%Y-%m-%d %H:%M:%S %Z"),
        now_local.strftime("%Y-%m-%d %H:%M:%S"),
        tz_name,
    )


_MONITORING_OUTPUT_INSTRUCTIONS = """\
You were invoked by a scheduled monitoring trigger (Mode 2). Read the
cross-bucket snapshot below and produce a single fenced JSON code block
matching this schema:

```json
{
  "summary": "<markdown post body that will be sent verbatim to the strategist Discord thread>",
  "new_aims": [
    {
      "title": "<aim title>",
      "scope": "long_term" | "bucket",
      "bucket": "<one of the 9 buckets>" | null,
      "target_date": "YYYY-MM-DD" | null,
      "description": "",
      "parent_id": "<id of an existing long-term Aim>" | null
    }
  ],
  "signals": [
    {
      "to_bucket": "<one of the 9 buckets>",
      "kind": "aim_added" | "strategist_alert" | "rebalance" | "fyi",
      "urgency": "low" | "normal" | "high",
      "summary": "<one-line>",
      "body": "<longer prose context>",
      "expires_at": "<ISO-8601 UTC>" | null
    }
  ],
  "aim_status_changes": [
    { "aim_id": "<existing Aim id>", "status": "completed" | "dropped" | "active", "reason": "" }
  ]
}
```

Rules:

- Every list MUST be present, even if empty.
- ``summary`` MUST be a non-empty markdown string. It is posted verbatim
  to the strategist Discord thread.
- Use empty lists if no Aims need to be created, no Signals need to be
  sent, and no statuses need to flip. That is a valid, common run.
- Do not include any field not listed above; extras are rejected.
- Do not queue worker jobs from here — that's the coaches' job. Send
  Signals so the coaches act on their next trigger.
"""


def _format_snapshot(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, default=str)


def _build_prompt(state: StrategistState) -> dict[str, Any]:
    base_prompt = _load_strategist_prompt().rstrip()
    now_utc_s, now_local_s, tz_name = _format_time_block()
    reason = state.get("reason") or "scheduled-monitoring"
    snapshot = state.get("snapshot") or {}

    system_text = (
        f"{base_prompt}\n\n"
        f"---\n\n"
        f"## Time block\n\n"
        f"Current time (UTC): {now_utc_s}\n"
        f"Current time (local): {now_local_s} ({tz_name})\n"
        f"Trigger reason: {reason}\n\n"
        f"---\n\n"
        f"{_MONITORING_OUTPUT_INSTRUCTIONS}"
    )
    user_text = (
        "Cross-bucket snapshot (JSON):\n\n"
        f"```json\n{_format_snapshot(snapshot)}\n```\n\n"
        "Respond now with the single JSON code block."
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_text),
    ]
    return {"messages": messages}


# ----------------------------------------------------------------- llm


def _call_llm(state: StrategistState) -> dict[str, Any]:
    llm = state.get("llm")
    if llm is None:
        llm = get_chat_model("strategist")
    messages = state.get("messages") or []
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in cast(list[Any], content):
            if isinstance(item, dict):
                d = cast(dict[str, Any], item)
                t = d.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
            elif isinstance(item, str):
                text_parts.append(item)
        text = "\n".join(text_parts)
    else:
        text = str(content)
    return {"raw_response": text}


# ----------------------------------------------------------------- parse


def _parse_response(state: StrategistState) -> dict[str, Any]:
    raw = state.get("raw_response", "")
    block = extract_json_block(raw)
    if block is None:
        return {"error": "no JSON block found in strategist LLM response"}
    try:
        parsed = StrategistOutput.model_validate_json(block)
    except ValidationError as exc:
        return {"error": f"invalid StrategistOutput JSON: {exc}"}
    return {"parsed": parsed}


# ----------------------------------------------------------------- persist


def _persist_results(state: StrategistState) -> dict[str, Any]:
    if "error" in state:
        return {}
    parsed = state.get("parsed")
    if parsed is None:
        return {"error": "no parsed strategist output to persist"}
    report = persist_results(parsed)
    return {"persisted": report}


# ----------------------------------------------------------------- assemble


_compiled: (
    CompiledStateGraph[StrategistState, Any, StrategistState, StrategistState]
    | None
) = None


def build_graph() -> CompiledStateGraph[
    StrategistState, Any, StrategistState, StrategistState
]:
    """Construct (and cache) the strategist monitoring LangGraph."""
    global _compiled
    if _compiled is not None:
        return _compiled
    sg: StateGraph[
        StrategistState, Any, StrategistState, StrategistState
    ] = StateGraph(StrategistState)
    sg.add_node("gather_cross_bucket_context", _gather_context)
    sg.add_node("build_prompt", _build_prompt)
    sg.add_node("call_llm", _call_llm)
    sg.add_node("parse_response", _parse_response)
    sg.add_node("persist_results", _persist_results)
    sg.add_edge(START, "gather_cross_bucket_context")
    sg.add_edge("gather_cross_bucket_context", "build_prompt")
    sg.add_edge("build_prompt", "call_llm")
    sg.add_edge("call_llm", "parse_response")
    sg.add_edge("parse_response", "persist_results")
    sg.add_edge("persist_results", END)
    _compiled = sg.compile()
    return _compiled


def reset_graph_cache() -> None:
    """Drop the compiled graph cache. Tests use this between runs."""
    global _compiled
    _compiled = None


def run_monitoring(
    *,
    reason: str = "scheduled-monitoring",
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    llm: BaseChatModel | None = None,
) -> StrategistState:
    """Run one strategist monitoring pass through the graph and return state.

    The graph performs DB writes (new Aims, Signals, Aim status flips)
    but does NOT post the summary to Discord — see
    :func:`run_monitoring_and_post` for the variant that also publishes.

    Tests pass ``llm=`` to inject a fake chat model that returns canned
    JSON. Production callers leave ``llm=None`` so the factory wires a
    real ``ChatOpenAI`` against the proxy.
    """
    graph = build_graph()
    initial: StrategistState = {
        "reason": reason,
        "horizon_days": horizon_days,
        "llm": llm,
    }
    final = graph.invoke(initial)
    return cast(StrategistState, final)


async def run_monitoring_and_post(
    *,
    reason: str = "scheduled-monitoring",
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    llm: BaseChatModel | None = None,
) -> StrategistState:
    """Run :func:`run_monitoring` and publish the summary to Discord.

    This is the entry point used by the chain-step bridge: the chain
    worker for a directive containing :data:`STRATEGIST_DIRECTIVE_SENTINEL`
    awaits this and returns the merged state.
    """
    state = run_monitoring(reason=reason, horizon_days=horizon_days, llm=llm)
    parsed = state.get("parsed")
    posted: list[int] = []
    if parsed is not None and parsed.summary.strip():
        posted = await post_summary(parsed.summary)
    state["posted_message_ids"] = posted
    return state


__all__ = [
    "DEFAULT_HORIZON_DAYS",
    "STRATEGIST_DIRECTIVE_SENTINEL",
    "StrategistState",
    "build_graph",
    "reset_graph_cache",
    "run_monitoring",
    "run_monitoring_and_post",
]

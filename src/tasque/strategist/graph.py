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
from sqlalchemy import or_, select

from tasque.agents import result_inbox
from tasque.buckets import ALL_BUCKETS, Bucket
from tasque.chains.spec import SpecError, validate_spec
from tasque.config import get_settings
from tasque.llm.factory import ALL_TIERS, get_chat_model, get_chat_model_for_tier
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
DEFAULT_AIM_PLAN_NOTE_LIMIT = 12
DEFAULT_AIM_PLAN_SIGNAL_LIMIT = 12
AIM_CHAIN_PLAN_AGENT_KIND = "aim_chain_plan"
STRATEGIST_DISALLOWED_TOOLS = [
    "mcp__tasque__aim_plan_chain",
    "mcp__tasque__chain_queue_adhoc",
    "mcp__tasque__chain_fire_template",
    "mcp__tasque__chain_template_create",
    "mcp__tasque__chain_template_update",
    "mcp__tasque__chain_template_delete",
    "mcp__tasque__job_create",
]
_AIM_CHAIN_PLAN_DISALLOWED_TOOLS = [
    *STRATEGIST_DISALLOWED_TOOLS,
]

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
    result_token: NotRequired[str]
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


def _serialize_aim_for_context(a: Aim) -> dict[str, Any]:
    return {
        "id": a.id,
        "title": a.title,
        "bucket": a.bucket,
        "scope": a.scope,
        "target_date": a.target_date,
        "description": a.description,
        "status": a.status,
        "parent_id": a.parent_id,
        "source": a.source,
        "broken_down_at": dict(a.broken_down_at or {}),
        "created_at": a.created_at,
        "updated_at": a.updated_at,
    }


def _serialize_note_for_aim_plan(n: Note) -> dict[str, Any]:
    return {
        "id": n.id,
        "bucket": n.bucket,
        "durability": n.durability,
        "source": n.source,
        "content": n.content,
        "created_at": n.created_at,
        "updated_at": n.updated_at,
    }


def _serialize_signal_for_aim_plan(s: Signal) -> dict[str, Any]:
    return {
        "id": s.id,
        "from_bucket": s.from_bucket,
        "to_bucket": s.to_bucket,
        "kind": s.kind,
        "urgency": s.urgency,
        "summary": s.summary,
        "body": s.body,
        "context": s.context,
        "expires_at": s.expires_at,
        "created_at": s.created_at,
    }


def _aim_chain_plan_context(aim_id: str) -> dict[str, Any] | None:
    """Gather the first-pass context used to turn one Aim into a chain spec."""
    with get_session() as sess:
        aim = sess.get(Aim, aim_id)
        if aim is None:
            return None

        child_aims = list(
            sess.execute(
                select(Aim)
                .where(Aim.parent_id == aim.id)
                .where(Aim.status == "active")
                .order_by(Aim.created_at.desc())
            ).scalars().all()
        )
        parent_aim: Aim | None = None
        if aim.parent_id:
            parent_aim = sess.get(Aim, aim.parent_id)

        relevant_buckets: set[str] = set()
        if aim.bucket:
            relevant_buckets.add(aim.bucket)
        relevant_buckets.update(
            a.bucket for a in child_aims if isinstance(a.bucket, str) and a.bucket
        )

        notes_stmt = select(Note).where(Note.archived.is_(False))
        if relevant_buckets:
            notes_stmt = notes_stmt.where(Note.bucket.in_(sorted(relevant_buckets)))
        notes = list(
            sess.execute(
                notes_stmt.order_by(Note.updated_at.desc()).limit(
                    DEFAULT_AIM_PLAN_NOTE_LIMIT
                )
            ).scalars().all()
        )

        signals_stmt = select(Signal).where(Signal.archived.is_(False))
        if relevant_buckets:
            buckets = sorted(relevant_buckets)
            signals_stmt = signals_stmt.where(
                or_(
                    Signal.to_bucket.in_(buckets),
                    Signal.from_bucket.in_(buckets),
                    Signal.to_bucket == "all",
                )
            )
        signals = list(
            sess.execute(
                signals_stmt.order_by(Signal.created_at.desc()).limit(
                    DEFAULT_AIM_PLAN_SIGNAL_LIMIT
                )
            ).scalars().all()
        )

        return {
            "aim": _serialize_aim_for_context(aim),
            "parent_aim": (
                _serialize_aim_for_context(parent_aim)
                if parent_aim is not None
                else None
            ),
            "child_aims": [_serialize_aim_for_context(a) for a in child_aims],
            "relevant_buckets": sorted(relevant_buckets),
            "recent_notes": [_serialize_note_for_aim_plan(n) for n in notes],
            "recent_signals": [_serialize_signal_for_aim_plan(s) for s in signals],
        }


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
cross-bucket snapshot below and call ``submit_strategist_result``
exactly once with the ``result_token`` from the run context:

    submit_strategist_result(
      result_token="<value from run context>",
      summary="<markdown post body sent verbatim to the strategist Discord thread>",
      new_aims=[
        {
          "title": "<aim title>",
          "scope": "long_term" | "bucket",
          "bucket": "<one of the 9 buckets>" | None,
          "target_date": "YYYY-MM-DD" | None,
          "description": "",
          "parent_id": "<id of an existing long-term Aim>" | None
        }
      ],
      signals=[
        {
          "to_bucket": "<one of the 9 buckets>",
          "kind": "aim_added" | "strategist_alert" | "rebalance" | "fyi",
          "urgency": "low" | "normal" | "high",
          "summary": "<one-line>",
          "body": "<longer prose context>",
          "expires_at": "<ISO-8601 UTC>" | None
        }
      ],
      aim_status_changes=[
        { "aim_id": "<existing Aim id>",
          "status": "completed" | "dropped" | "active",
          "reason": "" }
      ]
    )

Rules:

- ``summary`` MUST be a non-empty markdown string. It is posted
  verbatim to the strategist Discord thread.
- Pass empty lists for sections you don't need — that is a valid,
  common run. Do not omit the call.
- Do not include any field not listed above; extras are rejected.
- Do not queue worker jobs from here — that's the coaches' job. Send
  Signals so the coaches act on their next trigger.
"""


def _format_snapshot(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, default=str)


_AIM_CHAIN_PLAN_SYSTEM_PROMPT = """\
You are the tasque Aim-to-chain planner. Convert one long-horizon Aim
and its relevant context into a first-pass Tasque chain spec.

Call ``submit_aim_chain_plan_result`` exactly once with the
``result_token`` from the user message:

    submit_aim_chain_plan_result(
      result_token="<value from run context>",
      plan={
        "chain_name": "<stable, unique-ish slug>",
        "bucket": "<one of the 9 buckets>" | None,
        "recurrence": None,
        "planner_tier": "large" | "medium" | "small",
        "vars": {},
        "plan": [
          {
            "id": "<short_step_id>",
            "kind": "worker",
            "directive": "<specific worker instruction>",
            "depends_on": [],
            "consumes": [],
            "tier": "small" | "medium" | "large"
          }
        ]
      }
    )

Rules:
- Return a full chain spec, not planner mutations.
- Top-level keys must be only chain_name, bucket, recurrence,
  planner_tier, vars, and plan.
- Every worker step must include tier; approval steps must not include tier.
- consumes must be a subset of depends_on for the same step.
- Use recurrence=None unless the context clearly asks for a recurring
  template.
- Keep this to a practical first pass. No GOAP search, no hidden engine,
  no direct queueing or template creation. This tool will validate and
  decide whether to queue or save the spec after your result is deposited.
"""


def _build_aim_chain_plan_messages(
    *,
    context: dict[str, Any],
    result_token: str,
    mode: str,
) -> list[BaseMessage]:
    user_text = (
        "## Run context\n"
        f"- result_token: {result_token}  "
        "(pass this to submit_aim_chain_plan_result)\n"
        f"- requested mode: {mode}\n\n"
        "Aim planning context (JSON):\n\n"
        f"```json\n{json.dumps(context, indent=2, default=str)}\n```\n\n"
        "Produce the chain spec and call submit_aim_chain_plan_result with "
        "the result_token above."
    )
    return [
        SystemMessage(content=_AIM_CHAIN_PLAN_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]


def _attach_aim_vars(spec: dict[str, Any], aim: dict[str, Any]) -> None:
    vars_raw = spec.get("vars")
    if vars_raw is None:
        vars_value: dict[str, Any] = {}
    elif isinstance(vars_raw, dict):
        vars_value = dict(vars_raw)
    else:
        # Leave the invalid value intact so validate_spec reports the
        # producer's error clearly.
        return
    vars_value["aim_id"] = aim.get("id")
    vars_value["aim_title"] = aim.get("title")
    spec["vars"] = vars_value


def plan_chain_for_aim(
    aim_id: str,
    *,
    mode: str = "adhoc",
    planner_tier: str = "large",
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Ask the planner tier for a validated first-pass chain spec for an Aim.

    The returned dict is an MCP-friendly envelope. On success it contains
    ``{"ok": True, "spec": ...}``; on validation failure it contains
    ``validation_errors`` plus the candidate spec for inspection.
    """
    normalized_mode = mode.strip().lower()
    if normalized_mode not in ("adhoc", "template"):
        return {
            "ok": False,
            "error": "mode must be 'adhoc' or 'template'",
        }
    if planner_tier not in ALL_TIERS:
        return {
            "ok": False,
            "error": f"planner_tier must be one of {sorted(ALL_TIERS)!r}",
        }

    context = _aim_chain_plan_context(aim_id)
    if context is None:
        return {"ok": False, "error": f"no Aim with id={aim_id}"}
    aim = cast(dict[str, Any], context["aim"])

    token = result_inbox.mint_token()
    messages = _build_aim_chain_plan_messages(
        context=context,
        result_token=token,
        mode=normalized_mode,
    )
    if llm is None:
        llm = get_chat_model_for_tier(
            planner_tier,
            disallowed_tools=_AIM_CHAIN_PLAN_DISALLOWED_TOOLS,
        )
    try:
        llm.invoke(messages)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"aim chain planner LLM failed: {type(exc).__name__}: {exc}",
        }

    payload = result_inbox.read_and_consume(
        token,
        agent_kind=AIM_CHAIN_PLAN_AGENT_KIND,
    )
    if payload is None:
        return {
            "ok": False,
            "error": "aim chain planner did not call submit_aim_chain_plan_result",
        }
    raw_spec = payload.get("plan")
    if not isinstance(raw_spec, dict):
        return {
            "ok": False,
            "error": "submit_aim_chain_plan_result payload missing object 'plan'",
            "payload": payload,
        }

    spec = dict(cast(dict[str, Any], raw_spec))
    _attach_aim_vars(spec, aim)
    try:
        validate_spec(spec)
    except SpecError as exc:
        return {
            "ok": False,
            "error": "chain spec validation failed",
            "validation_errors": [str(exc)],
            "spec": spec,
        }

    return {
        "ok": True,
        "mode": normalized_mode,
        "aim": aim,
        "spec": spec,
    }


def _build_prompt(state: StrategistState) -> dict[str, Any]:
    base_prompt = _load_strategist_prompt().rstrip()
    now_utc_s, now_local_s, tz_name = _format_time_block()
    reason = state.get("reason") or "scheduled-monitoring"
    snapshot = state.get("snapshot") or {}
    token = state.get("result_token") or result_inbox.mint_token()

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
        "## Run context\n"
        f"- result_token: {token}  "
        "(pass this to submit_strategist_result)\n\n"
        "Cross-bucket snapshot (JSON):\n\n"
        f"```json\n{_format_snapshot(snapshot)}\n```\n\n"
        "Decide what to do and call submit_strategist_result with the "
        "result_token above."
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_text),
    ]
    return {"messages": messages, "result_token": token}


# ----------------------------------------------------------------- llm


def _call_llm(state: StrategistState) -> dict[str, Any]:
    llm = state.get("llm")
    if llm is None:
        llm = get_chat_model(
            "strategist",
            disallowed_tools=STRATEGIST_DISALLOWED_TOOLS,
        )
    messages = state.get("messages") or []
    llm.invoke(messages)
    # The LLM's text response is intentionally discarded — the
    # structured result lands in the inbox via submit_strategist_result.
    return {}


# ----------------------------------------------------------------- parse


def _parse_response(state: StrategistState) -> dict[str, Any]:
    token = state.get("result_token")
    if not token:
        return {"error": "strategist run had no result_token in state"}
    payload = result_inbox.read_and_consume(token, agent_kind="strategist")
    if payload is None:
        return {
            "error": (
                "strategist did not call submit_strategist_result during its "
                "turn — no structured result was deposited in the inbox"
            )
        }
    try:
        parsed = StrategistOutput.model_validate(payload)
    except ValidationError as exc:
        return {"error": f"invalid submit_strategist_result payload: {exc}"}
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
    "AIM_CHAIN_PLAN_AGENT_KIND",
    "DEFAULT_HORIZON_DAYS",
    "STRATEGIST_DIRECTIVE_SENTINEL",
    "STRATEGIST_DISALLOWED_TOOLS",
    "StrategistState",
    "build_graph",
    "plan_chain_for_aim",
    "reset_graph_cache",
    "run_monitoring",
    "run_monitoring_and_post",
]

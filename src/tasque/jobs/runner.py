"""The worker LangGraph: a single LLM call that executes one ``QueuedJob``.

Three linear nodes:

    build_prompt → call_llm → persist_run

The worker is intentionally narrow: assemble a prompt, call the haiku-tier
LLM, parse a JSON block of ``{report, summary, produces}``, persist a
``Note(durability=durable, source=worker)`` summarising the run.

Failures are surfaced via ``WorkerResult.error`` rather than raised — the
scheduler converts an error into a ``FailedJob`` row. The graph itself
never raises during normal operation; even a malformed LLM response is
caught and reported.

``run_worker(job)`` is a *pure function* over the QueuedJob: it does not
touch APScheduler, claim the job, or schedule recurrences. The scheduler
calls it after ``claim``-ing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from tasque.agents import result_inbox
from tasque.llm.factory import (
    ALL_TIERS,
    get_chat_model_for_tier,
)
from tasque.memory.entities import Note, QueuedJob
from tasque.memory.repo import write_entity

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


WORKER_SYSTEM_PROMPT = """\
You are a tasque worker. Execute exactly one directive and submit
your structured result by calling ``submit_worker_result`` exactly
once near the end of your turn. Tasque routes your output; you do
not post, format embeds, or talk about Discord/threads/channels.

## How to return your result

The run context section of the user message contains a
``result_token``. When the directive is complete, call:

    submit_worker_result(
      result_token="<value from run context>",
      report="<full markdown body — what you did, what you observed>",
      summary="<one or two sentences, max one short paragraph>",
      produces={"<key>": "<value>", ...},
      error=None,  # or "<short reason>" — see below
    )

Field roles — keep these strictly separated:

- ``report``: the full body the user reads in the thread. Markdown is
  fine. Long is fine; tasque chunks it.
- ``summary``: 1-2 sentences. The user sees this in a small embed
  description. Do NOT duplicate the report here. No more than one
  short paragraph.
- ``produces``: STRUCTURED data only — scalars, short lists, ids,
  status flags. It is internal — only consumed by downstream chain
  steps via their ``consumes`` resolution. It is NEVER rendered to
  the user. Do not put narrative, large strings, or embed-formatting
  payloads here. May be omitted (defaults to ``{}``) when the
  directive doesn't hand off to anything downstream.
- ``error``: short string identifying an application-level failure —
  pass it when your turn completed deterministically but the
  underlying action did not succeed (e.g. browser automation timed
  out, external API errored, gates rejected the proposal). Leave it
  ``None`` (the default) for normal successful runs. Setting it
  marks THIS chain step as failed; the chain engine then applies
  the step's ``on_failure`` policy and the overall ``ChainRun``
  finalises as ``failed`` if any step ends up failed. Still populate
  ``report``/``summary``/``produces`` so the user-facing thread post
  and downstream consumers see what happened.

You MUST call ``submit_worker_result`` exactly once per run. Do not
also emit a JSON code block — your text response is ignored. If you
forget to call the tool, the run is recorded as a worker failure.

## tasque MCP — write surface

The host injects the **tasque MCP** into this turn. Every side-effect
you perform — queueing a follow-up job, writing a Note, firing a chain,
sending a Signal — happens through these tools, synchronously, during
your turn. Pass the bucket from the run context above on each call.

- **Notes**: ``note_create``, ``note_get``, ``note_list``, ``note_search``,
  ``note_search_fts``, ``note_archive``.
- **Queued jobs**: ``job_create(directive, bucket, tier, fire_at,
  recurrence, ...)``, ``job_get``, ``job_update``, ``job_cancel``,
  ``job_list``. One-shot (``recurrence=None``) and recurring
  (5-field cron, alias DOW). ``tier`` is required: ``"haiku"`` for
  trivial nudges, ``"sonnet"`` for multi-step tool / scrape /
  summarize work, ``"opus"`` for agentic planning, code iteration,
  or deep creative generation.
- **Chain templates**: ``chain_template_create``, ``chain_template_get``,
  ``chain_template_list``, ``chain_template_update``, ``chain_template_delete``.
- **Chain runs**: ``chain_fire_template(name)`` to launch a saved
  template now; ``chain_queue_adhoc(plan_json)`` for an ad-hoc plan.
  ``chain_run_get``, ``chain_run_list``, ``chain_run_pause``,
  ``chain_run_resume``, ``chain_run_stop``.
- **Signals**: ``signal_create``, ``signal_list``, ``signal_archive``.
- **Aims**: ``aim_get``, ``aim_list`` (workers don't typically create
  Aims — that's the strategist's job).
- **Idle-silence claim**: ``claim_idle_silence(seconds, reason)``.
  Call this BEFORE any tool you expect to keep stdout silent for >2
  minutes (model training, large download, long Bash sleep, slow
  scrape). Tasque's proxy runs a stall watchdog that kills the
  ``claude --print`` subprocess after ~5 min of stdout silence by
  default — without this call, a legitimate 30-min training run looks
  identical to a hang. Honest estimate; over-budget still re-engages
  the watchdog so genuine hangs past your estimate still get caught.
  Returns ``{"ok": false}`` when not running under the proxy (e.g.
  during a unit test) — safe to ignore.

When you make an MCP call, capture any returned ids / structured
results that downstream chain steps need to see, and surface them in
your ``produces`` dict. The MCP performed the write; ``produces`` is
the structured hand-off to the next step.

If your directive is purely "produce a report / data for downstream
steps" — no side effects required — you may not need any other MCP
calls. Just compose the result and call ``submit_worker_result``. If
the directive requires a write, use the MCP: do not pretend the
write happened, and do not encode it as report-text.

Stay focused on the directive. Do not pad with unrelated commentary.
"""


class WorkerResult(TypedDict):
    """Return value of :func:`run_worker`.

    On success ``error`` is ``None`` and ``report``/``summary``/``produces``
    are populated. On failure ``error`` carries the message and the other
    fields default to empty.
    """

    report: str
    summary: str
    produces: dict[str, Any]
    error: str | None


class WorkerState(TypedDict):
    """State threaded through the worker LangGraph.

    Holds *primitives only* (strings, dicts, BaseMessage list) so the
    LangGraph default MemorySaver's msgpack serializer can checkpoint
    state at node boundaries. Earlier versions kept the full
    :class:`QueuedJob` ORM row here, which broke checkpointing as soon
    as langgraph started enforcing serialization on graph state.
    The :func:`run_worker` entrypoint extracts the relevant fields up
    front; the SQLAlchemy row never enters graph state.
    """

    job_id: str
    job_bucket: str | None
    job_directive: str
    job_reason: str
    job_chain_id: str | None
    job_chain_step_id: str | None
    job_tier: str
    result_token: str
    consumes: NotRequired[dict[str, Any]]
    vars: NotRequired[dict[str, Any]]
    llm: NotRequired[BaseChatModel | None]
    messages: NotRequired[list[BaseMessage]]
    raw_response: NotRequired[str]
    result: NotRequired[WorkerResult]


def _format_consumes(consumes: dict[str, Any] | None) -> str:
    if not consumes:
        return "(none — this is a standalone job, not a chain step)"
    try:
        return json.dumps(consumes, indent=2, default=str)
    except (TypeError, ValueError):
        return repr(consumes)


def _format_vars(chain_vars: dict[str, Any] | None) -> str:
    if not chain_vars:
        return "(none)"
    try:
        return json.dumps(chain_vars, indent=2, default=str)
    except (TypeError, ValueError):
        return repr(chain_vars)


def _build_prompt(state: WorkerState) -> dict[str, Any]:
    consumes = state.get("consumes") or {}
    chain_vars = state.get("vars") or {}
    bucket = state["job_bucket"] or "(no bucket)"
    reason = state["job_reason"] or "(no reason given)"
    user_text = (
        "## Run context\n"
        f"- Bucket: {bucket}\n"
        f"- Reason: {reason}\n"
        f"- result_token: {state['result_token']}  "
        "(pass this to submit_worker_result)\n\n"
        f"Directive:\n{state['job_directive']}\n\n"
        f"Consumes (from previous chain step, if any):\n{_format_consumes(consumes)}\n\n"
        f"Vars (run-time overrides supplied at chain launch):\n{_format_vars(chain_vars)}\n\n"
        "Execute the directive now. When done, call submit_worker_result "
        "exactly once with the result_token above and your structured result."
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=WORKER_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]
    return {"messages": messages}


def _extract_text(response: BaseMessage) -> str:
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
        return "\n".join(text_parts)
    return str(content)


_RETRY_REMINDER = (
    "You finished your turn without calling submit_worker_result. "
    "Call it now exactly once with the result_token from the run "
    "context above and your report / summary / produces. Do NOT redo "
    "any work you already performed via other MCP tools — those side "
    "effects already persisted; only the declarative end-of-turn "
    "payload is missing. If your earlier turn produced a usable "
    "report, reuse it verbatim."
)


def _call_llm(state: WorkerState) -> dict[str, Any]:
    llm = state.get("llm")
    if llm is None:
        tier = state["job_tier"]
        if tier not in ALL_TIERS:
            return {
                "result": WorkerResult(
                    report="",
                    summary="",
                    produces={},
                    error=(
                        f"job is missing a valid tier (got {tier!r}); "
                        f"every QueuedJob must have an explicit tier"
                    ),
                )
            }
        llm = get_chat_model_for_tier(tier)
    messages = state.get("messages") or []
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        return {
            "result": WorkerResult(
                report="",
                summary="",
                produces={},
                error=f"LLM call failed: {type(exc).__name__}: {exc}",
            )
        }
    text = _extract_text(response)

    # Tool-call miss recovery: the haiku tier in particular sometimes
    # finishes its turn with prose only and never invokes
    # submit_worker_result, sinking the whole job. Give it one
    # follow-up nudge before the persist node fails the run. The
    # reminder explicitly tells the LLM not to redo MCP side effects
    # so we don't double-write notes / queue duplicate jobs.
    if not result_inbox.peek(state["result_token"], agent_kind="worker"):
        try:
            retry_response = llm.invoke(
                [*list(messages), response, HumanMessage(content=_RETRY_REMINDER)]
            )
            text = _extract_text(retry_response) or text
            log.info(
                "jobs.runner.submit_worker_result_retry",
                job_id=state["job_id"],
                deposited=result_inbox.peek(
                    state["result_token"], agent_kind="worker"
                ),
            )
        except Exception as exc:
            # Don't fail the run on a retry-side error — let persist_run
            # surface the original "tool not called" message.
            log.warning(
                "jobs.runner.submit_worker_result_retry_failed",
                job_id=state["job_id"],
                error=f"{type(exc).__name__}: {exc}",
            )
    return {"raw_response": text}


def _persist_run(state: WorkerState) -> dict[str, Any]:
    # If call_llm already produced a terminal error result, pass it through.
    prior = state.get("result")
    if prior is not None and prior.get("error"):
        return {"result": prior}

    payload = result_inbox.read_and_consume(
        state["result_token"], agent_kind="worker"
    )
    if payload is None:
        return {
            "result": WorkerResult(
                report="",
                summary="",
                produces={},
                error=(
                    "worker did not call submit_worker_result during its turn — "
                    "no structured result was deposited in the inbox"
                ),
            )
        }
    report_v = payload.get("report")
    summary_v = payload.get("summary")
    produces_v = payload.get("produces") or {}
    if not isinstance(report_v, str) or not isinstance(summary_v, str):
        return {
            "result": WorkerResult(
                report="",
                summary="",
                produces={},
                error=(
                    "submit_worker_result payload missing required string "
                    "fields 'report' and/or 'summary'"
                ),
            )
        }
    if not isinstance(produces_v, dict):
        return {
            "result": WorkerResult(
                report="",
                summary="",
                produces={},
                error=(
                    f"submit_worker_result 'produces' must be an object, "
                    f"got {type(produces_v).__name__}"
                ),
            )
        }
    produces_d = cast(dict[str, Any], produces_v)
    # ``error`` is a worker-declared application-level failure: the turn
    # completed deterministically (we have report/summary/produces) but
    # the underlying action did not succeed. Surface it as the
    # WorkerResult's error so the chain engine flips the step to failed
    # and applies its on_failure policy. report/summary/produces stay
    # populated so downstream consumers and the user-facing thread post
    # still see the worker's full output.
    error_raw = payload.get("error")
    error_value: str | None = None
    if isinstance(error_raw, str):
        stripped = error_raw.strip()
        if stripped:
            error_value = stripped

    note = Note(
        content=summary_v,
        bucket=state["job_bucket"],
        durability="durable",
        source="worker",
        meta={
            "directive": state["job_directive"],
            "report": report_v,
            "produces": produces_d,
            "job_id": state["job_id"],
            "chain_id": state["job_chain_id"],
            "chain_step_id": state["job_chain_step_id"],
            "worker_error": error_value,
        },
    )
    write_entity(note)
    return {
        "result": WorkerResult(
            report=report_v,
            summary=summary_v,
            produces=produces_d,
            error=error_value,
        )
    }


_compiled: CompiledStateGraph[WorkerState, Any, WorkerState, WorkerState] | None = None


def build_graph() -> CompiledStateGraph[WorkerState, Any, WorkerState, WorkerState]:
    """Construct (and cache) the worker LangGraph."""
    global _compiled
    if _compiled is not None:
        return _compiled
    sg: StateGraph[WorkerState, Any, WorkerState, WorkerState] = StateGraph(WorkerState)
    sg.add_node("build_prompt", _build_prompt)
    sg.add_node("call_llm", _call_llm)
    sg.add_node("persist_run", _persist_run)
    sg.add_edge(START, "build_prompt")
    sg.add_edge("build_prompt", "call_llm")
    sg.add_edge("call_llm", "persist_run")
    sg.add_edge("persist_run", END)
    _compiled = sg.compile()
    return _compiled


def run_worker(
    job: QueuedJob,
    *,
    consumes: dict[str, Any] | None = None,
    vars: dict[str, Any] | None = None,
    llm: BaseChatModel | None = None,
) -> WorkerResult:
    """Run the worker LangGraph for a single ``QueuedJob`` and return the result.

    ``consumes`` is the produces-payload from the previous chain step
    (if any). ``vars`` is the operator-supplied run-time override dict
    set at chain launch (frozen for the run); the worker prompt
    surfaces it so directives can branch on it. ``llm`` lets tests
    inject a fake chat model. The function is pure with respect to the
    scheduler; it neither claims the job nor schedules a recurrence.

    Strategist bridge: a directive containing
    ``[strategist:monitor]`` short-circuits the worker LLM and dispatches
    the strategist monitoring run instead. The bridge produces a
    WorkerResult so the chain engine sees the step complete normally.
    """
    from tasque.strategist.graph import STRATEGIST_DIRECTIVE_SENTINEL

    if STRATEGIST_DIRECTIVE_SENTINEL in job.directive:
        return _run_strategist_monitoring(job, llm=llm)

    if not job.tier:
        return WorkerResult(
            report="",
            summary="",
            produces={},
            error=(
                f"QueuedJob {job.id} has no tier set; refusing to run. "
                f"Every job must declare its tier (opus / sonnet / haiku) "
                f"at insert time."
            ),
        )

    graph = build_graph()
    initial: WorkerState = {
        "job_id": job.id,
        "job_bucket": job.bucket,
        "job_directive": job.directive,
        "job_reason": job.reason or "",
        "job_chain_id": job.chain_id,
        "job_chain_step_id": job.chain_step_id,
        "job_tier": job.tier,
        "result_token": result_inbox.mint_token(),
        "consumes": consumes or {},
        "vars": vars or {},
        "llm": llm,
    }
    final = graph.invoke(initial)
    final_state = cast(WorkerState, final)
    result = final_state.get("result")
    if result is None:
        return WorkerResult(
            report="",
            summary="",
            produces={},
            error="worker graph produced no result",
        )
    return result


def _run_strategist_monitoring(
    job: QueuedJob, *, llm: BaseChatModel | None
) -> WorkerResult:
    """Bridge: dispatch a strategist monitoring run from a chain worker step.

    Imported lazily to avoid circular imports between the runner and
    the strategist module. The strategist's graph performs its own DB
    writes (Aims, Signals, Aim status flips) and the post-summary call
    publishes to Discord; here we just translate the result into a
    ``WorkerResult`` so the chain engine sees the step complete.
    """
    import asyncio

    from tasque.strategist.graph import run_monitoring_and_post

    reason = job.reason or "scheduled-monitoring"
    state = asyncio.run(run_monitoring_and_post(reason=reason, llm=llm))
    err = state.get("error")
    if err:
        return WorkerResult(report="", summary="", produces={}, error=err)
    parsed = state.get("parsed")
    persisted = state.get("persisted") or {}
    posted_ids = state.get("posted_message_ids") or []
    summary_text = parsed.summary if parsed is not None else ""
    return WorkerResult(
        report=summary_text,
        summary="strategist monitoring run completed",
        produces={
            "new_aim_ids": persisted.get("new_aim_ids", []),
            "signal_ids": persisted.get("signal_ids", []),
            "flipped_aim_ids": persisted.get("flipped_aim_ids", []),
            "posted_message_ids": posted_ids,
        },
        error=None,
    )

"""Tests for the bucket-coach LangGraph and helpers.

The coach now performs all writes via the tasque MCP during its turn
— the langgraph itself has no tool layer. ``BucketCoachOutput`` is
just ``{"thread_post": str | None}``: a runtime-controlled signal
asking the bot to publish a Discord message on the coach's behalf.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from tasque.agents import result_inbox
from tasque.coach.graph import (
    _build_prompt,
    _gather_context,
    _parse_response,
    run_bucket_coach,
)
from tasque.coach.output import BucketCoachOutput
from tasque.coach.persist import persist_results
from tasque.memory.entities import Note, QueuedJob, Signal
from tasque.memory.repo import write_entity

from .conftest import make_coach_result_chat_model

# ---------------------------------------------------------------- output

def test_bucket_coach_output_rejects_extra_fields() -> None:
    bad = json.dumps({"thread_post": None, "smuggled": "nope"})
    with pytest.raises(ValidationError):
        BucketCoachOutput.model_validate_json(bad)


def test_bucket_coach_output_defaults_thread_post_to_none() -> None:
    parsed = BucketCoachOutput.model_validate_json("{}")
    assert parsed.thread_post is None


# --------------------------------------------------------- gather_context

def test_gather_context_pulls_notes_signals_and_pending_jobs() -> None:
    write_entity(Note(content="active", bucket="health", durability="durable", source="user"))
    archived = write_entity(
        Note(content="gone", bucket="health", durability="durable", source="user", archived=True)
    )
    assert archived.archived is True
    write_entity(
        Signal(
            from_bucket="career",
            to_bucket="health",
            kind="fyi",
            urgency="whenever",
            summary="rest more",
        )
    )
    write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="walk",
            queued_by="health",
            status="pending",
        )
    )
    write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="old",
            queued_by="health",
            status="completed",
        )
    )

    result = _gather_context({"bucket": "health", "reason": "manual"})
    notes = result.get("notes") or []
    signals = result.get("signals") or []
    jobs = result.get("queued_jobs") or []
    assert [n.content for n in notes] == ["active"]
    assert [s.summary for s in signals] == ["rest more"]
    assert [j.directive for j in jobs] == ["walk"]


# --------------------------------------------------------- build_prompt

def test_build_prompt_includes_scaffold_bucket_state_and_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Durable notes are NOT pre-injected — the coach uses MCP search tools.
    write_entity(Note(content="hydrate", bucket="health", durability="durable", source="user"))
    # Behavioral notes ARE injected (always honor instructions).
    write_entity(
        Note(
            content="never auto-text the user",
            bucket="health",
            durability="behavioral",
            source="user",
        )
    )
    # Recent ephemeral notes ARE injected (capped, most-recent first).
    write_entity(
        Note(content="last run noticed dehydration", bucket="health", durability="ephemeral", source="coach")
    )
    write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="walk 20 min",
            queued_by="health",
            status="pending",
        )
    )
    monkeypatch.setenv("TASQUE_TIMEZONE", "UTC")
    state = _gather_context({"bucket": "health", "reason": "manual ping"})
    state["bucket"] = "health"
    state["reason"] = "manual ping"
    out = _build_prompt(state)
    messages = out.get("messages") or []
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    sys_text = str(messages[0].content)
    assert "tasque coach scaffolding" in sys_text
    assert "Sleep-and-recovery first" in sys_text  # health-bucket mindset present
    # System prompt is now stable across runs — time + reason are in the
    # user message, NOT the system prompt. This is what lets Claude CLI's
    # prefix cache hit on repeated runs of the same bucket.
    assert "{now_utc}" not in sys_text
    assert "{reason}" not in sys_text
    assert "manual ping" not in sys_text
    # The "## Run context" block lives in the user message, not as a header
    # in the system prompt. (The scaffold may mention the string in prose,
    # but it must not appear as a line-start markdown header.)
    assert "\n## Run context" not in sys_text
    assert "\n- Bucket: health" not in sys_text
    assert "\n- Trigger reason:" not in sys_text
    # No live timestamp anywhere in the system prompt.
    assert "Current time (UTC):" not in sys_text
    assert "## Time block" not in sys_text

    user_text = str(messages[1].content)
    # Time + reason + bucket live here.
    assert "## Run context" in user_text
    assert "Bucket: health" in user_text
    assert "manual ping" in user_text
    # The fresh result_token must be embedded so the LLM can pass it
    # back to submit_coach_result.
    assert "result_token:" in user_text
    assert "submit_coach_result" in user_text
    # And it must be propagated as state so _parse_response can read
    # the inbox row keyed on the same token.
    assert isinstance(out.get("result_token"), str) and out["result_token"]
    # Durable content does NOT appear in the prompt; only the count + a hint.
    assert "hydrate" not in user_text
    assert "1 present in this bucket" in user_text
    # Behavioral instructions and recent ephemeral notes DO appear.
    assert "never auto-text the user" in user_text
    assert "last run noticed dehydration" in user_text
    assert "walk 20 min" in user_text
    assert "Trigger reason: manual ping" in user_text


# --------------------------------------------------------- parse_response

def test_parse_response_reads_inbox_and_validates() -> None:
    token = result_inbox.mint_token()
    result_inbox.deposit(
        result_token=token,
        agent_kind="coach",
        payload={"thread_post": "queued one new task"},
    )
    out = _parse_response({"bucket": "health", "result_token": token})
    parsed = out.get("parsed")
    assert parsed is not None
    assert parsed.thread_post == "queued one new task"


def test_parse_response_accepts_null_thread_post() -> None:
    token = result_inbox.mint_token()
    result_inbox.deposit(
        result_token=token,
        agent_kind="coach",
        payload={"thread_post": None},
    )
    out = _parse_response({"bucket": "health", "result_token": token})
    parsed = out.get("parsed")
    assert parsed is not None
    assert parsed.thread_post is None


def test_parse_response_errors_when_tool_not_called() -> None:
    """An empty inbox means the coach LLM never called
    submit_coach_result during its turn."""
    token = result_inbox.mint_token()
    out = _parse_response({"bucket": "health", "result_token": token})
    assert "error" in out
    assert "submit_coach_result" in out["error"]


def test_parse_response_errors_on_missing_token() -> None:
    out = _parse_response({"bucket": "health"})
    assert "error" in out
    assert "result_token" in out["error"]


def test_parse_response_errors_on_invalid_payload() -> None:
    """A coach payload with extra fields fails BucketCoachOutput
    validation — surfaced as an error rather than silently dropped."""
    token = result_inbox.mint_token()
    result_inbox.deposit(
        result_token=token,
        agent_kind="coach",
        payload={"thread_post": None, "smuggled": "nope"},
    )
    out = _parse_response({"bucket": "health", "result_token": token})
    assert "error" in out


# --------------------------------------------------------- persist_results

def test_persist_results_surfaces_thread_post() -> None:
    parsed = BucketCoachOutput.model_validate({"thread_post": "queued one new task"})
    report = persist_results("health", parsed)
    assert report["bucket"] == "health"
    assert report["thread_post"] == "queued one new task"


def test_persist_results_passes_through_null() -> None:
    parsed = BucketCoachOutput.model_validate({"thread_post": None})
    report = persist_results("health", parsed)
    assert report["thread_post"] is None


# -------------------------------------------------------------- end-to-end

def test_run_bucket_coach_end_to_end_with_fake_llm() -> None:
    write_entity(
        Note(content="seed note", bucket="health", durability="durable", source="user")
    )
    fake = make_coach_result_chat_model(
        [{"thread_post": "ran without anything to announce"}]
    )
    final = run_bucket_coach("health", reason="test", llm=fake)
    persisted = final.get("persisted") or {}
    assert persisted.get("thread_post") == "ran without anything to announce"
    assert persisted.get("bucket") == "health"


def test_run_bucket_coach_end_to_end_null_post() -> None:
    fake = make_coach_result_chat_model([{"thread_post": None}])
    final = run_bucket_coach("health", reason="test", llm=fake)
    persisted = final.get("persisted") or {}
    assert persisted.get("thread_post") is None


# ----------------------------------------------- post-reply tool gating

def test_build_prompt_replies_with_post_reply_guidance() -> None:
    """When fired by the Discord router's post-reply trigger
    (reason="reply"), the prompt must NOT instruct the bucket coach to
    fire chains / queue jobs — the synchronous reply already had that
    chance, and a duplicate fire is the bug we're fixing."""
    out = _build_prompt({"bucket": "health", "reason": "reply"})
    user_text = str((out["messages"] or [])[1].content)
    assert "Post-reply pass" in user_text
    assert "synchronous reply" in user_text
    # Disabled action tools must be named so the LLM doesn't try them.
    assert "chain_fire_template" in user_text
    assert "do not attempt them" in user_text


def test_build_prompt_non_reply_keeps_full_action_surface() -> None:
    out = _build_prompt({"bucket": "health", "reason": "scheduled"})
    user_text = str((out["messages"] or [])[1].content)
    # Standard run still tells the coach it can write via MCP.
    assert "note_create" in user_text
    assert "Post-reply pass" not in user_text


def test_call_llm_passes_disallowed_tools_when_reply_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_bucket_coach`` with reason='reply' must build the LLM with
    the chain-fire / queue / job-create tools denylisted, so the
    proxy → claude --print path can't double-fire on the user's request."""
    from langchain_core.messages import AIMessage

    from tasque.coach import graph as coach_graph

    captured: dict[str, Any] = {}

    class _FakeLLM:
        def invoke(self, messages: list[Any]) -> AIMessage:
            return AIMessage(content="{}")

    def fake_factory(agent_kind: str, **kwargs: Any) -> _FakeLLM:
        captured["agent_kind"] = agent_kind
        captured["kwargs"] = kwargs
        return _FakeLLM()

    monkeypatch.setattr(coach_graph, "get_chat_model", fake_factory)
    coach_graph._call_llm({"bucket": "health", "reason": "reply", "messages": []})
    assert captured["agent_kind"] == "coach"
    assert captured["kwargs"].get("disallowed_tools") == [
        "mcp__tasque__chain_fire_template",
        "mcp__tasque__chain_queue_adhoc",
        "mcp__tasque__job_create",
    ]


def test_call_llm_omits_disallowed_tools_for_non_reply_triggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.messages import AIMessage

    from tasque.coach import graph as coach_graph

    captured: dict[str, Any] = {}

    class _FakeLLM:
        def invoke(self, messages: list[Any]) -> AIMessage:
            return AIMessage(content="{}")

    def fake_factory(agent_kind: str, **kwargs: Any) -> _FakeLLM:
        captured["kwargs"] = kwargs
        return _FakeLLM()

    monkeypatch.setattr(coach_graph, "get_chat_model", fake_factory)
    coach_graph._call_llm(
        {"bucket": "health", "reason": "job-completed:abc", "messages": []}
    )
    assert captured["kwargs"].get("disallowed_tools") is None

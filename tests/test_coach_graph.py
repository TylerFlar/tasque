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

from tasque.coach.graph import (
    _build_prompt,
    _gather_context,
    _parse_response,
    run_bucket_coach,
)
from tasque.coach.output import BucketCoachOutput, extract_json_block
from tasque.coach.persist import persist_results
from tasque.memory.entities import Note, QueuedJob, Signal
from tasque.memory.repo import write_entity

from .conftest import make_canned_chat_model

# ---------------------------------------------------------------- output

def test_extract_json_block_fenced() -> None:
    body = '{"thread_post": "hello"}'
    text = f"some preamble\n```json\n{body}\n```\nepilogue"
    extracted = extract_json_block(text)
    assert extracted is not None
    parsed = BucketCoachOutput.model_validate_json(extracted)
    assert parsed.thread_post == "hello"


def test_extract_json_block_unfenced_fallback() -> None:
    body = '{"thread_post": null}'
    text = f"hi {body} bye"
    extracted = extract_json_block(text)
    assert extracted == body


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
    # Durable content does NOT appear in the prompt; only the count + a hint.
    assert "hydrate" not in user_text
    assert "1 present in this bucket" in user_text
    # Behavioral instructions and recent ephemeral notes DO appear.
    assert "never auto-text the user" in user_text
    assert "last run noticed dehydration" in user_text
    assert "walk 20 min" in user_text
    assert "Trigger reason: manual ping" in user_text


# --------------------------------------------------------- parse_response

def test_parse_response_extracts_and_validates() -> None:
    payload = {"thread_post": "queued one new task"}
    raw = "preface\n```json\n" + json.dumps(payload) + "\n```\ntail"
    out = _parse_response({"bucket": "health", "raw_response": raw})
    parsed = out.get("parsed")
    assert parsed is not None
    assert parsed.thread_post == "queued one new task"


def test_parse_response_accepts_null_thread_post() -> None:
    raw = "```json\n" + json.dumps({"thread_post": None}) + "\n```"
    out = _parse_response({"bucket": "health", "raw_response": raw})
    parsed = out.get("parsed")
    assert parsed is not None
    assert parsed.thread_post is None


def test_parse_response_errors_on_missing_block() -> None:
    out = _parse_response({"bucket": "health", "raw_response": "no JSON here"})
    assert "error" in out


def test_parse_response_errors_on_malformed_json() -> None:
    raw = "```json\n{not real json}\n```"
    out = _parse_response({"bucket": "health", "raw_response": raw})
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
    fake = make_canned_chat_model([{"thread_post": "ran without anything to announce"}])
    final = run_bucket_coach("health", reason="test", llm=fake)
    persisted = final.get("persisted") or {}
    assert persisted.get("thread_post") == "ran without anything to announce"
    assert persisted.get("bucket") == "health"


def test_run_bucket_coach_end_to_end_null_post() -> None:
    fake = make_canned_chat_model([{"thread_post": None}])
    final = run_bucket_coach("health", reason="test", llm=fake)
    persisted = final.get("persisted") or {}
    assert persisted.get("thread_post") is None

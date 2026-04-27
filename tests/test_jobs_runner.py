"""Tests for the worker LangGraph (``tasque.jobs.runner``)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from tasque.jobs.runner import run_worker
from tasque.memory.db import get_session
from tasque.memory.entities import Note, QueuedJob
from tasque.memory.repo import write_entity

from .conftest import make_canned_chat_model


def _make_job(**overrides: Any) -> QueuedJob:
    defaults: dict[str, Any] = dict(
        kind="worker",
        bucket="personal",
        directive="say hi",
        reason="test",
        fire_at="now",
        status="pending",
        queued_by="test",
        tier="haiku",
    )
    defaults.update(overrides)
    return write_entity(QueuedJob(**defaults))


def test_run_worker_persists_note_on_success() -> None:
    job = _make_job(directive="introduce yourself", bucket="personal")
    fake = make_canned_chat_model(
        [
            {
                "report": "I am tasque worker, hello.",
                "summary": "Worker introduced itself.",
                "produces": {"greeting": "hi"},
            }
        ]
    )
    result = run_worker(job, llm=fake)
    assert result["error"] is None
    assert result["report"] == "I am tasque worker, hello."
    assert result["summary"] == "Worker introduced itself."
    assert result["produces"] == {"greeting": "hi"}

    with get_session() as sess:
        notes = list(
            sess.execute(
                select(Note).where(Note.source == "worker")
            ).scalars().all()
        )
    assert len(notes) == 1
    n = notes[0]
    assert n.content == "Worker introduced itself."
    assert n.durability == "durable"
    assert n.bucket == "personal"
    assert n.meta["directive"] == "introduce yourself"
    assert n.meta["report"] == "I am tasque worker, hello."
    assert n.meta["produces"] == {"greeting": "hi"}
    assert n.meta["job_id"] == job.id


def test_run_worker_handles_missing_json_block() -> None:
    job = _make_job(directive="malformed", bucket="personal")
    # Raw string payload that doesn't look like a json block.
    fake = make_canned_chat_model(["no JSON here, just prose"])
    result = run_worker(job, llm=fake)
    assert result["error"] is not None
    assert "no JSON block" in result["error"]
    # No Note should have been written.
    with get_session() as sess:
        notes = list(
            sess.execute(select(Note).where(Note.source == "worker")).scalars().all()
        )
    assert notes == []


def test_run_worker_handles_missing_required_field() -> None:
    job = _make_job(directive="bad shape", bucket="personal")
    fake = make_canned_chat_model([{"summary": "only summary, no report"}])
    result = run_worker(job, llm=fake)
    assert result["error"] is not None
    assert "report" in result["error"] or "summary" in result["error"]


def test_run_worker_threads_consumes_into_prompt() -> None:
    job = _make_job(directive="use the input", bucket="career")
    captured: dict[str, Any] = {}

    class Capturing:
        def invoke(self, messages: list[Any]) -> Any:
            captured["messages"] = messages
            from langchain_core.messages import AIMessage

            return AIMessage(
                content='```json\n{"report": "ok", "summary": "ok", "produces": {}}\n```'
            )

    result = run_worker(job, consumes={"prev_step_output": "abc"}, llm=Capturing())  # type: ignore[arg-type]
    assert result["error"] is None
    user_text = str(captured["messages"][1].content)
    assert "prev_step_output" in user_text
    assert "abc" in user_text


def test_run_worker_accepts_missing_produces() -> None:
    job = _make_job(directive="no produces field", bucket="personal")
    fake = make_canned_chat_model(
        [{"report": "did the thing", "summary": "done"}]
    )
    result = run_worker(job, llm=fake)
    assert result["error"] is None
    assert result["produces"] == {}


def test_run_worker_records_chain_metadata() -> None:
    job = _make_job(
        directive="chain step",
        bucket="personal",
        chain_id="chain-xyz",
        chain_step_id="step-1",
    )
    fake = make_canned_chat_model(
        [{"report": "r", "summary": "s", "produces": {"k": "v"}}]
    )
    result = run_worker(job, llm=fake)
    assert result["error"] is None
    with get_session() as sess:
        note = sess.execute(
            select(Note).where(Note.source == "worker")
        ).scalars().one()
        assert note.meta["chain_id"] == "chain-xyz"
        assert note.meta["chain_step_id"] == "step-1"

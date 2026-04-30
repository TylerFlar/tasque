"""Tests for the worker LangGraph (``tasque.jobs.runner``)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import select

from tasque.jobs.runner import run_worker
from tasque.memory.db import get_session
from tasque.memory.entities import Note, QueuedJob
from tasque.memory.repo import write_entity

from .conftest import make_worker_result_chat_model


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
    fake = make_worker_result_chat_model(
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


def test_run_worker_errors_when_tool_not_called() -> None:
    """If the LLM finishes its turn without calling
    ``submit_worker_result``, the inbox stays empty and the worker
    must surface a clear error rather than producing a phantom Note."""
    job = _make_job(directive="forgot the tool", bucket="personal")
    silent = FakeMessagesListChatModel(
        responses=[AIMessage(content="prose only, no tool call")]
    )
    result = run_worker(job, llm=silent)
    assert result["error"] is not None
    assert "submit_worker_result" in result["error"]
    with get_session() as sess:
        notes = list(
            sess.execute(select(Note).where(Note.source == "worker")).scalars().all()
        )
    assert notes == []


def test_run_worker_errors_on_missing_required_field() -> None:
    """Even when the LLM did call submit_worker_result, the payload
    must carry both ``report`` and ``summary`` strings."""
    job = _make_job(directive="bad shape", bucket="personal")
    fake = make_worker_result_chat_model(
        [{"summary": "only summary, no report"}]
    )
    result = run_worker(job, llm=fake)
    assert result["error"] is not None
    assert "report" in result["error"] or "summary" in result["error"]


def test_run_worker_threads_consumes_into_prompt() -> None:
    """Run-time ``consumes`` and ``vars`` must reach the user prompt
    so directives can branch on upstream output."""
    job = _make_job(directive="use the input", bucket="career")
    captured: dict[str, Any] = {}

    fake = make_worker_result_chat_model(
        [{"report": "ok", "summary": "ok", "produces": {}}]
    )
    original_generate = fake._generate

    def _capture(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        captured["messages"] = messages
        return original_generate(messages, *args, **kwargs)

    fake._generate = _capture  # type: ignore[method-assign]

    result = run_worker(job, consumes={"prev_step_output": "abc"}, llm=fake)
    assert result["error"] is None
    user_text = str(captured["messages"][1].content)
    assert "prev_step_output" in user_text
    assert "abc" in user_text
    assert "result_token" in user_text


def test_run_worker_accepts_missing_produces() -> None:
    job = _make_job(directive="no produces field", bucket="personal")
    fake = make_worker_result_chat_model(
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
    fake = make_worker_result_chat_model(
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


def test_run_worker_surfaces_application_error_field() -> None:
    """A worker that calls ``submit_worker_result(..., error="...")``
    after a deterministically-failed action (e.g. browser timeout) must
    have its WorkerResult.error populated so the chain engine flips
    THIS step to failed — even though ``report``/``summary``/``produces``
    are populated and the run otherwise succeeded."""
    job = _make_job(directive="dispatch fired but failed", bucket="finance")
    fake = make_worker_result_chat_model(
        [
            {
                "report": "Logged in, hit AAPL ticket, modal blocked preview.",
                "summary": "AAPL preview blocked by Fidelity modal.",
                "produces": {
                    "outcome": "executor_failure",
                    "n_fired": 0,
                    "n_total": 25,
                },
                "error": (
                    "fidelity_trade_ticket failed at preview-verify on "
                    "AAPL — 'information temporarily unavailable' modal"
                ),
            }
        ]
    )
    result = run_worker(job, llm=fake)
    assert result["error"] is not None
    assert "AAPL" in result["error"]
    # Report/summary/produces still propagate — downstream consumers
    # and the user-facing thread post need them.
    assert result["report"].startswith("Logged in")
    assert result["summary"].startswith("AAPL preview")
    assert result["produces"]["outcome"] == "executor_failure"


def test_run_worker_treats_empty_error_as_no_error() -> None:
    """A whitespace-only or empty ``error`` must NOT flip the run to
    failed — workers shouldn't have to remember to pass ``None`` vs
    ``""`` carefully."""
    job = _make_job(directive="ok run with empty-string error", bucket="personal")
    fake = make_worker_result_chat_model(
        [
            {
                "report": "all good",
                "summary": "ok",
                "produces": {},
                "error": "   ",
            }
        ]
    )
    result = run_worker(job, llm=fake)
    assert result["error"] is None

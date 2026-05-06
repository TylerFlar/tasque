"""Tests for the worker LangGraph (``tasque.jobs.runner``)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import select

from tasque.jobs.runner import run_worker
from tasque.memory.db import get_session
from tasque.memory.entities import Note, QueuedJob, WorkerPattern
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
        tier="small",
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
    assert n.durability == "ephemeral"
    assert n.memory_kind == "artifact"
    assert n.ttl_days == 3
    assert n.bucket == "personal"
    assert n.meta["directive"] == "introduce yourself"
    assert n.meta["report_chars"] == len("I am tasque worker, hello.")
    assert n.meta["produces_keys"] == ["greeting"]
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
    system_text = str(captured["messages"][0].content)
    assert "MCP tool discovery" not in system_text
    assert "rg -uuu" not in system_text
    user_text = str(captured["messages"][1].content)
    assert "prev_step_output" in user_text
    assert "abc" in user_text
    assert "result_token" in user_text


def test_run_worker_injects_relevant_past_patterns() -> None:
    write_entity(
        WorkerPattern(
            bucket="finance",
            source_kind="worker",
            key="worker:stripe-invoice",
            content=(
                "Directive: reconcile Stripe invoice exports\n"
                "Produces keys: invoice_ids\n"
                "Summary: Check Stripe before the ledger export."
            ),
            tags=["stripe", "invoice", "ledger"],
            success_count=3,
        )
    )
    write_entity(
        WorkerPattern(
            bucket="career",
            source_kind="worker",
            key="worker:unrelated",
            content="Directive: update resume bullets",
            tags=["resume"],
        )
    )
    job = _make_job(
        directive="Reconcile the Stripe invoice export and produce invoice_ids",
        bucket="finance",
    )
    captured: dict[str, Any] = {}
    fake = make_worker_result_chat_model(
        [{"report": "ok", "summary": "ok", "produces": {}}]
    )
    original_generate = fake._generate

    def _capture(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        captured["messages"] = messages
        return original_generate(messages, *args, **kwargs)

    fake._generate = _capture  # type: ignore[method-assign]

    result = run_worker(job, llm=fake)

    assert result["error"] is None
    user_text = str(captured["messages"][1].content)
    assert "## Relevant Past Patterns" in user_text
    assert "Check Stripe before the ledger export" in user_text
    assert "update resume bullets" not in user_text


def test_run_worker_writes_compact_pattern_after_success() -> None:
    job = _make_job(
        directive="Summarize pantry restock and return grocery_ids",
        bucket="home",
    )
    fake = make_worker_result_chat_model(
        [
            {
                "report": (
                    "Reviewed the pantry list.\n"
                    "Gotcha: Instacart hides unavailable replacements.\n"
                    "password=super-secret"
                ),
                "summary": "Pantry restock summary is ready.",
                "produces": {"grocery_ids": ["g-1", "g-2"]},
            }
        ]
    )

    result = run_worker(job, llm=fake)

    assert result["error"] is None
    with get_session() as sess:
        patterns = list(sess.execute(select(WorkerPattern)).scalars().all())
    assert len(patterns) == 1
    pattern = patterns[0]
    assert pattern.bucket == "home"
    assert pattern.source_kind == "worker"
    assert pattern.success_count == 1
    assert "Summarize pantry restock" in pattern.content
    assert "grocery_ids" in pattern.content
    assert "Pantry restock summary is ready" in pattern.content
    assert "Instacart hides unavailable replacements" in pattern.content
    assert "super-secret" not in pattern.content
    assert pattern.meta["produces_keys"] == ["grocery_ids"]


def test_run_worker_does_not_write_pattern_after_application_error() -> None:
    job = _make_job(directive="attempt blocked task", bucket="personal")
    fake = make_worker_result_chat_model(
        [
            {
                "report": "blocked",
                "summary": "blocked",
                "produces": {},
                "error": "external tool failed",
            }
        ]
    )

    result = run_worker(job, llm=fake)

    assert result["error"] == "external tool failed"
    with get_session() as sess:
        patterns = list(sess.execute(select(WorkerPattern)).scalars().all())
    assert patterns == []


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

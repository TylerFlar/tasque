"""Tests for the parameterised reply runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from tasque.reply.runtime import (
    AttachmentMarker,
    HistoryMessage,
    ReplyResult,
    run_reply,
)


class _FakeAgent:
    """Pretend ReAct agent that records its input and returns canned messages.

    Reproduces just enough of ``create_react_agent``'s output shape to
    let :func:`run_reply` extract the final text and tool calls.
    """

    def __init__(self, response_messages: list[Any], tool_calls_made: list[str] | None = None) -> None:
        self.response_messages = response_messages
        self.tool_calls_made = tool_calls_made or []
        self.invoked_with: dict[str, Any] | None = None

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.invoked_with = payload
        return {"messages": list(payload["messages"]) + list(self.response_messages)}


def _factory(messages: list[Any], tool_calls: list[str] | None = None):
    fake = _FakeAgent(messages, tool_calls)

    def factory(_llm: BaseChatModel, _tools: Sequence[Any]) -> _FakeAgent:
        return fake

    return fake, factory


def _llm() -> BaseChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content="unused")])


def test_run_reply_returns_final_ai_text() -> None:
    fake, factory = _factory([AIMessage(content="here is the answer")])
    result: ReplyResult = run_reply(
        llm=_llm(),
        tools=[],
        system_prompt="be brief",
        user_content="what's the weather",
        agent_factory=factory,
    )
    assert result["text"] == "here is the answer"
    assert fake.invoked_with is not None
    msgs = fake.invoked_with["messages"]
    # System + Human only on input.
    assert msgs[0].content == "be brief"
    assert "what's the weather" in msgs[1].content


def test_run_reply_includes_history_block() -> None:
    history: list[HistoryMessage] = [
        {"author": "tasque-bot", "content": "I queued one job", "is_bot": True},
        {"author": "tyler", "content": "ok thanks", "is_bot": False},
    ]
    fake, factory = _factory([AIMessage(content="ack")])
    run_reply(
        llm=_llm(),
        tools=[],
        system_prompt="x",
        user_content="actually no",
        history=history,
        agent_factory=factory,
    )
    assert fake.invoked_with is not None
    user_msg = fake.invoked_with["messages"][1].content
    assert "Recent thread history" in user_msg
    assert "I queued one job" in user_msg
    assert "actually no" in user_msg


def test_run_reply_truncates_long_history_messages() -> None:
    long = "x" * 1500
    history: list[HistoryMessage] = [
        {"author": "tasque-bot", "content": long, "is_bot": True},
    ]
    fake, factory = _factory([AIMessage(content="ack")])
    run_reply(
        llm=_llm(),
        tools=[],
        system_prompt="x",
        user_content="hi",
        history=history,
        agent_factory=factory,
    )
    assert fake.invoked_with is not None
    user_msg = fake.invoked_with["messages"][1].content
    # Should not contain the full 1500 'x' run; per-line cap is 500.
    assert "x" * 600 not in user_msg


def test_run_reply_caps_total_history_at_4000_chars() -> None:
    huge = "y" * 480
    history: list[HistoryMessage] = [
        {"author": f"u{i}", "content": huge, "is_bot": False} for i in range(20)
    ]
    fake, factory = _factory([AIMessage(content="ack")])
    run_reply(
        llm=_llm(),
        tools=[],
        system_prompt="x",
        user_content="hi",
        history=history,
        agent_factory=factory,
    )
    assert fake.invoked_with is not None
    user_msg = fake.invoked_with["messages"][1].content
    # The full 20 * (480+author overhead) = ~10k would blow 4000; ensure
    # the runtime stopped well short.
    assert user_msg.count("y" * 480) < 20


def test_run_reply_includes_attachment_block() -> None:
    attachments: list[AttachmentMarker] = [
        {
            "filename": "blood-panel.pdf",
            "local_path": "/tmp/abc.pdf",
            "content_type": "application/pdf",
        }
    ]
    fake, factory = _factory([AIMessage(content="ack")])
    run_reply(
        llm=_llm(),
        tools=[],
        system_prompt="x",
        user_content="here",
        attachments=attachments,
        agent_factory=factory,
    )
    assert fake.invoked_with is not None
    user_msg = fake.invoked_with["messages"][1].content
    assert "Attachments" in user_msg
    assert "blood-panel.pdf" in user_msg
    assert "/tmp/abc.pdf" in user_msg


def test_run_reply_extracts_tool_calls() -> None:
    ai_with_tool = AIMessage(
        content="calling",
        tool_calls=[
            {
                "name": "queue_job",
                "args": {"directive": "x"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    final = AIMessage(content="done")
    _, factory = _factory([ai_with_tool, final])
    result = run_reply(
        llm=_llm(),
        tools=[],
        system_prompt="x",
        user_content="hi",
        agent_factory=factory,
    )
    assert result["text"] == "done"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "queue_job"

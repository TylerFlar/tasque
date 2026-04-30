"""Per-test in-memory SQLite engine, isolated from any real data dir.

Also provides ``fake_coach_llm`` — a ``BaseChatModel`` that returns
pre-canned ``BucketCoachOutput`` JSON. Inject via the ``llm`` parameter on
``run_bucket_coach`` / ``run_drainer``.
"""

from __future__ import annotations

import contextlib
import re
import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from tasque.agents import result_inbox
from tasque.chains import checkpointer as chain_checkpointer
from tasque.chains import graph as chain_graph
from tasque.coach import prompts as coach_prompts
from tasque.memory import db as db_module


@pytest.fixture(autouse=True)
def isolated_db() -> Iterator[Engine]:
    """Replace the process-wide engine with a fresh in-memory SQLite for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    db_module.set_engine(engine)
    coach_prompts.reset_cache()

    # Fresh in-memory chain checkpointer per test so chain state doesn't
    # leak between cases. The compiled graph caches the saver, so reset
    # both.
    chain_conn = sqlite3.connect(":memory:", check_same_thread=False)
    chain_saver = SqliteSaver(chain_conn)
    chain_saver.setup()
    chain_checkpointer.set_chain_checkpointer(chain_saver)
    chain_graph.reset_compiled_chain_graph()

    try:
        yield engine
    finally:
        db_module.reset_engine()
        coach_prompts.reset_cache()
        chain_graph.reset_compiled_chain_graph()
        chain_checkpointer.reset_chain_checkpointer()
        with contextlib.suppress(sqlite3.Error):
            chain_conn.close()


def _wrap_in_fenced_block(payload: dict[str, Any] | str) -> str:
    import json as _json

    body = payload if isinstance(payload, str) else _json.dumps(payload)
    return f"```json\n{body}\n```"


def make_canned_chat_model(
    payloads: list[dict[str, Any] | str],
) -> FakeMessagesListChatModel:
    """Build a FakeMessagesListChatModel that returns ``payloads`` in order.

    Each payload is either a raw string (used as-is) or a dict (serialized
    inside a ```json fenced block). The chat model loops back to the start
    once exhausted.
    """
    messages: list[BaseMessage] = [AIMessage(content=_wrap_in_fenced_block(p)) for p in payloads]
    return FakeMessagesListChatModel(responses=messages)


@pytest.fixture
def fake_coach_llm() -> FakeMessagesListChatModel:
    """Default fake coach LLM — returns one all-empty BucketCoachOutput."""
    return make_canned_chat_model([{"thread_post": None}])


_RESULT_TOKEN_RE = re.compile(r"result_token:\s*([0-9a-fA-F]+)")


class _ResultDepositingLLM(BaseChatModel):
    """Test stand-in that simulates an agent LLM's MCP tool call.

    Production agents no longer parse the LLM's text output — they
    read the structured result from ``agent_results`` after the LLM
    calls ``submit_<agent>_result`` mid-turn. The langchain fake
    chat model can't actually invoke MCP tools, so this stand-in
    extracts the ``result_token`` from the prompt and writes the
    canned payload directly to the inbox before returning.

    ``agent_kind`` selects which agent's read will accept the row.
    Pass either a single payload (used for every call) or a list of
    payloads (consumed in order, looping after exhaustion).
    """

    agent_kind: str
    payloads: list[dict[str, Any]]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return f"test-{self.agent_kind}-result-depositing"

    def _generate(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        prompt_text = "\n".join(
            (m.content if isinstance(m.content, str) else str(m.content))
            for m in messages
        )
        match = _RESULT_TOKEN_RE.search(prompt_text)
        if match is not None:
            token = match.group(1)
            payload = self.payloads[self.cursor % len(self.payloads)]
            self.cursor += 1
            result_inbox.deposit(
                result_token=token,
                agent_kind=self.agent_kind,
                payload=payload,
            )
        message = AIMessage(
            content=f"ok — submit_{self.agent_kind}_result called"
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_result_depositing_chat_model(
    *,
    agent_kind: str,
    payloads: list[dict[str, Any]],
) -> _ResultDepositingLLM:
    """Build a fake chat model that simulates the inbox tool call for ``agent_kind``."""
    return _ResultDepositingLLM(agent_kind=agent_kind, payloads=list(payloads))


def make_worker_result_chat_model(
    payloads: list[dict[str, Any]],
) -> _ResultDepositingLLM:
    """Convenience shim for ``agent_kind='worker'`` — used by worker tests."""
    return make_result_depositing_chat_model(
        agent_kind="worker", payloads=payloads
    )


def make_coach_result_chat_model(
    payloads: list[dict[str, Any]],
) -> _ResultDepositingLLM:
    """Convenience shim for ``agent_kind='coach'``."""
    return make_result_depositing_chat_model(
        agent_kind="coach", payloads=payloads
    )


def make_planner_result_chat_model(
    payloads: list[dict[str, Any]],
) -> _ResultDepositingLLM:
    """Convenience shim for ``agent_kind='planner'``."""
    return make_result_depositing_chat_model(
        agent_kind="planner", payloads=payloads
    )


def make_strategist_result_chat_model(
    payloads: list[dict[str, Any]],
) -> _ResultDepositingLLM:
    """Convenience shim for ``agent_kind='strategist'``."""
    return make_result_depositing_chat_model(
        agent_kind="strategist", payloads=payloads
    )

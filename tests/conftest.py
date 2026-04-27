"""Per-test in-memory SQLite engine, isolated from any real data dir.

Also provides ``fake_coach_llm`` — a ``BaseChatModel`` that returns
pre-canned ``BucketCoachOutput`` JSON. Inject via the ``llm`` parameter on
``run_bucket_coach`` / ``run_drainer``.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

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

"""Strategist reply binding.

System prompt = ``prompts/strategist.md`` verbatim. Tools come from the
tasque MCP injected by the selected upstream (see ``src/tasque/mcp/server.py``).
A reply does its work synchronously via MCP tool calls. Bucket Aims and
Signals created during that reply wake the relevant bucket coaches.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from tasque.llm.factory import get_chat_model
from tasque.reply.runtime import (
    AttachmentMarker,
    HistoryMessage,
    ReplyResult,
    run_reply,
)
from tasque.strategist.graph import STRATEGIST_DISALLOWED_TOOLS

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRATEGIST_PROMPT_PATH = _REPO_ROOT / "prompts" / "strategist.md"


def build_strategist_reply_prompt() -> str:
    """Return the strategist reply system prompt — the full
    ``prompts/strategist.md`` body. The strategist's prompt covers both
    decomposition and monitoring modes; the runtime only invokes the
    decomposition path here, but the prompt's full text gives the LLM
    the same authoritative-voice grounding either way.
    """
    return DEFAULT_STRATEGIST_PROMPT_PATH.read_text(encoding="utf-8")


def run_strategist_reply(
    content: str,
    *,
    history: Sequence[HistoryMessage] | None = None,
    attachments: Sequence[AttachmentMarker] | None = None,
    llm: BaseChatModel | None = None,
    agent_factory: Callable[[BaseChatModel, Sequence[Any]], Any] | None = None,
) -> ReplyResult:
    """Run the strategist reply for ``content`` and return the result.

    ``llm`` defaults to :func:`tasque.llm.factory.get_chat_model("strategist")`.
    ``agent_factory`` is forwarded to the runtime so tests can inject a
    fake ReAct agent. Replies act synchronously through their tools.
    """
    chat = (
        llm
        if llm is not None
        else get_chat_model(
            "strategist",
            disallowed_tools=STRATEGIST_DISALLOWED_TOOLS,
        )
    )
    system_prompt = build_strategist_reply_prompt()
    return run_reply(
        llm=chat,
        tools=[],
        system_prompt=system_prompt,
        user_content=content,
        history=history,
        attachments=attachments,
        agent_factory=agent_factory,
    )


__all__ = ["build_strategist_reply_prompt", "run_strategist_reply"]

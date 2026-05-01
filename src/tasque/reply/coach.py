"""Coach reply binding.

System prompt = ``coach_prompts/<bucket>.md`` + ``prompts/coach_reply_scaffold.md``.
Tools come from the tasque MCP injected by the selected upstream (see
``src/tasque/mcp/server.py``). The post-reply hook is wired by the
Discord router so that a successful reply enqueues a coach trigger
keyed by the message id (see :func:`tasque.discord.router.route_message`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from tasque.buckets import ALL_BUCKETS, Bucket
from tasque.coach.prompts import load_bucket_prompt
from tasque.llm.factory import get_chat_model
from tasque.reply.runtime import (
    AttachmentMarker,
    HistoryMessage,
    ReplyResult,
    run_reply,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPLY_SCAFFOLD = _REPO_ROOT / "prompts" / "coach_reply_scaffold.md"


def _reply_scaffold_path() -> Path:
    raw = os.environ.get("TASQUE_COACH_REPLY_SCAFFOLD")
    return Path(raw) if raw else DEFAULT_REPLY_SCAFFOLD


def build_coach_reply_prompt(bucket: Bucket) -> str:
    """Return the composite coach reply system prompt.

    Layout: per-bucket mindset, then the reply scaffold. The scaffold
    gives the LLM its persona (a back-and-forth coach with tools);
    the bucket file gives it the bucket-specific perspective. The
    reactive single-shot scaffold (``coach_scaffold.md``) is *not*
    used here — that one is for the JSON-emitting bucket coach.
    """
    if bucket not in ALL_BUCKETS:
        raise ValueError(f"unknown bucket: {bucket!r}")
    mindset = load_bucket_prompt(bucket).strip()
    scaffold = _reply_scaffold_path().read_text(encoding="utf-8").strip()
    return f"{mindset}\n\n---\n\n{scaffold}\n"


def run_coach_reply(
    bucket: Bucket,
    content: str,
    *,
    history: Sequence[HistoryMessage] | None = None,
    attachments: Sequence[AttachmentMarker] | None = None,
    llm: BaseChatModel | None = None,
    agent_factory: Callable[[BaseChatModel, Sequence[Any]], Any] | None = None,
) -> ReplyResult:
    """Run the bucket coach reply for ``content`` and return the result.

    ``llm`` defaults to :func:`tasque.llm.factory.get_chat_model("coach")`.
    ``agent_factory`` is forwarded to the runtime so tests can inject a
    fake ReAct agent.
    """
    if bucket not in ALL_BUCKETS:
        raise ValueError(f"unknown bucket: {bucket!r}")
    chat = llm if llm is not None else get_chat_model("coach")
    system_prompt = build_coach_reply_prompt(bucket)
    return run_reply(
        llm=chat,
        tools=[],
        system_prompt=system_prompt,
        user_content=content,
        history=history,
        attachments=attachments,
        agent_factory=agent_factory,
    )


__all__ = ["build_coach_reply_prompt", "run_coach_reply"]

"""The single reply runtime. One function — :func:`run_reply` — that
all reply bindings (bucket coach, strategist) compose against.

Behaviour:

1. Build a message list: ``[SystemMessage(prompt), <history if any>,
   HumanMessage(user_content + attachment markers)]``.
2. Run a langgraph prebuilt ReAct agent over that with the tools bound.
3. Return the agent's final text content.

The runtime knows nothing about Discord, buckets, or coach trigger
queues. Bindings (e.g. :func:`tasque.reply.coach.run_coach_reply`)
compose the system prompt and choose the tool set.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent  # pyright: ignore[reportDeprecated]

ToolLike = BaseTool | Callable[..., Any]


class HistoryMessage(TypedDict):
    """One prior thread message, as supplied by the Discord layer."""

    author: str
    content: str
    is_bot: bool


class AttachmentMarker(TypedDict):
    """A short reference to a saved Discord attachment, included inline
    in the user message so the agent can reason about it."""

    filename: str
    local_path: str
    content_type: str


class ReplyResult(TypedDict):
    """Return value of :func:`run_reply`."""

    text: str
    tool_calls: list[dict[str, Any]]


def _format_history(history: Sequence[HistoryMessage]) -> str:
    """Render ``history`` as a transcript block, each line capped at 500
    chars, total capped at 4000 chars. Most recent messages first."""
    if not history:
        return ""
    lines: list[str] = []
    used = 0
    for msg in history:
        author = msg["author"]
        prefix = f"{author} (bot)" if msg["is_bot"] else author
        content = msg["content"].strip().replace("\n", " ")
        if len(content) > 500:
            content = content[:497] + "..."
        line = f"- {prefix}: {content}"
        if used + len(line) + 1 > 4000:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return "## Recent thread history\n\n" + "\n".join(lines)


def _format_attachments(attachments: Sequence[AttachmentMarker]) -> str:
    if not attachments:
        return ""
    lines = ["## Attachments saved by tasque"]
    for att in attachments:
        lines.append(
            f"- {att['filename']} ({att['content_type']}) → {att['local_path']}"
        )
    return "\n".join(lines)


def _build_user_text(
    user_content: str,
    history: Sequence[HistoryMessage],
    attachments: Sequence[AttachmentMarker],
) -> str:
    history_block = _format_history(history)
    attachments_block = _format_attachments(attachments)
    parts: list[str] = []
    if history_block:
        parts.append(history_block)
    if attachments_block:
        parts.append(attachments_block)
    parts.append("## User message\n\n" + user_content.strip())
    return "\n\n".join(parts)


ReactAgentFactory = Callable[[BaseChatModel, Sequence[ToolLike]], Any]


def _default_factory(llm: BaseChatModel, tools: Sequence[ToolLike]) -> Any:
    return create_react_agent(llm, list(tools))  # pyright: ignore[reportDeprecated]


def _extract_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in cast(list[Any], content):
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            d = cast(dict[str, Any], item)
            t = d.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "\n".join(parts)


def _extract_tool_calls(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        calls_raw = getattr(msg, "tool_calls", None)
        if not calls_raw:
            continue
        for tc in cast(list[Any], calls_raw):
            if isinstance(tc, dict):
                out.append(cast(dict[str, Any], tc))
    return out


def run_reply(
    *,
    llm: BaseChatModel,
    tools: Sequence[ToolLike],
    system_prompt: str,
    user_content: str,
    history: Sequence[HistoryMessage] | None = None,
    attachments: Sequence[AttachmentMarker] | None = None,
    agent_factory: ReactAgentFactory | None = None,
) -> ReplyResult:
    """Run a single reply turn through a ReAct agent and return its text.

    ``llm`` is a configured ``BaseChatModel`` (e.g. from
    :func:`tasque.llm.factory.get_chat_model`). ``tools`` is the list of
    tool callables to bind. ``history`` and ``attachments`` are
    optional context; both are formatted into the human message rather
    than passed as multimodal parts.

    ``agent_factory`` lets tests inject a fake agent that bypasses the
    LLM entirely.
    """
    factory = agent_factory or _default_factory
    agent = factory(llm, tools)

    user_text = _build_user_text(user_content, history or [], attachments or [])
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]

    final = agent.invoke({"messages": messages})
    final_messages_raw = (
        final.get("messages", []) if isinstance(final, dict) else []
    )
    final_messages: list[BaseMessage] = list(cast(list[BaseMessage], final_messages_raw))
    text = ""
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage):
            extracted = _extract_text(msg)
            if extracted.strip():
                text = extracted
                break
    tool_calls = _extract_tool_calls(final_messages)
    return {"text": text, "tool_calls": tool_calls}


__all__ = [
    "AttachmentMarker",
    "HistoryMessage",
    "ReplyResult",
    "run_reply",
]

"""Stateless REST helpers for posting to Discord.

These wrap the bot client's send / fetch / edit / upload primitives.
Callers pass channel/message ids explicitly — the poster keeps no
state of its own. The chain UI is the only caller allowed to use
:func:`edit_message`; everywhere else, posts are immutable.

The active client is set by :mod:`tasque.discord.bot` once on_ready
fires. Tests substitute a fake via :func:`set_client`.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any, Protocol, cast

from tasque.reply.runtime import HistoryMessage

DISCORD_CONTENT_LIMIT = 2000


class _ClientLike(Protocol):
    """Minimal subset of nextcord.Client / Bot used by the poster."""

    async def send_message(self, channel_id: int, content: str) -> int:
        ...

    async def send_embed(
        self, channel_id: int, embed: dict[str, Any], view: Any | None = None
    ) -> int:
        ...

    async def edit_message(
        self,
        channel_id: int,
        message_id: int,
        *,
        content: str | None = None,
        embed: dict[str, Any] | None = None,
        view: Any | None = None,
    ) -> None:
        ...

    async def upload_file(
        self,
        channel_id: int,
        path: Path,
        *,
        content: str | None = None,
    ) -> int:
        ...

    async def fetch_recent_messages(
        self, channel_id: int, limit: int
    ) -> list[HistoryMessage]:
        ...

    async def start_thread(
        self, channel_id: int, message_id: int, name: str
    ) -> int:
        ...

    def typing(self, channel_id: int) -> AbstractAsyncContextManager[None]:
        """Async context manager that shows ``Tasque is typing…`` in the
        target channel for the duration of the ``async with`` block.

        Discord's typing indicator self-expires after ~10s, so the
        nextcord adapter wraps the channel's own auto-refreshing
        ``Messageable.typing()`` context manager."""
        ...


_client: _ClientLike | None = None


def set_client(client: _ClientLike | None) -> None:
    """Install a poster client. The bot's on_ready handler calls this
    with a real nextcord-backed adapter; tests pass a fake."""
    global _client
    _client = client


def get_client() -> _ClientLike:
    """Return the active poster client. Raises if none is installed."""
    if _client is None:
        raise RuntimeError(
            "Discord poster client is not configured; "
            "call tasque.discord.poster.set_client(...) first"
        )
    return _client


def client_ready() -> bool:
    """True iff a poster client has been installed by ``set_client``.

    Background watchers consult this before posting so they can defer
    work until the bot's ``on_ready`` finishes wiring up the nextcord
    adapter — without this, the very first tick races the gateway
    connect and raises ``RuntimeError`` from :func:`get_client`.
    """
    return _client is not None


@asynccontextmanager
async def _noop_typing() -> AsyncGenerator[None, None]:
    yield


def typing_indicator(channel_id: int) -> AbstractAsyncContextManager[None]:
    """Return an async context manager that shows the typing indicator
    in ``channel_id`` while the wrapped block runs.

    Degrades to a no-op when no client is installed or the installed
    client doesn't expose ``typing`` — keeps the call site simple
    (``async with poster.typing_indicator(cid): ...`` always works)."""
    if _client is None:
        return _noop_typing()
    fn = getattr(_client, "typing", None)
    if fn is None:
        return _noop_typing()
    return fn(channel_id)


# ----------------------------------------------------------------- send

async def post_message(channel_id: int, content: str) -> int:
    """Send a single text message. Returns the new message id."""
    return await get_client().send_message(channel_id, content)


async def post_long_message(channel_id: int, content: str) -> list[int]:
    """Chunk ``content`` at the 2000-char limit and post each chunk.

    Returns the list of resulting message ids, in order. An empty input
    posts nothing.
    """
    if not content:
        return []
    chunks: list[str] = []
    remaining = content
    while remaining:
        if len(remaining) <= DISCORD_CONTENT_LIMIT:
            chunks.append(remaining)
            break
        # Try to break on a newline to keep things readable.
        cut = remaining.rfind("\n", 0, DISCORD_CONTENT_LIMIT)
        if cut < DISCORD_CONTENT_LIMIT // 2:
            cut = DISCORD_CONTENT_LIMIT
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    ids: list[int] = []
    for chunk in chunks:
        ids.append(await get_client().send_message(channel_id, chunk))
    return ids


async def post_embed(
    channel_id: int,
    embed: dict[str, Any],
    *,
    view: Any | None = None,
) -> int:
    """Send an embed (and optional View for buttons). Returns message id."""
    return await get_client().send_embed(channel_id, embed, view=view)


# ----------------------------------------------------------------- edit

async def edit_message(
    channel_id: int,
    message_id: int,
    *,
    content: str | None = None,
    embed: dict[str, Any] | None = None,
    view: Any | None = None,
) -> None:
    """Edit a previously-posted message. Used ONLY by the chain UI's
    button-click resolver to retire the buttons on an embed once the
    user has answered."""
    await get_client().edit_message(
        channel_id, message_id, content=content, embed=embed, view=view
    )


# ----------------------------------------------------------------- upload

async def upload_file(
    channel_id: int,
    path: Path,
    *,
    content: str | None = None,
) -> int:
    """Upload a local file as an attachment. Returns the new message id."""
    return await get_client().upload_file(channel_id, path, content=content)


# ----------------------------------------------------------------- fetch

async def fetch_recent_messages(
    channel_id: int, limit: int = 20
) -> list[HistoryMessage]:
    """Fetch up to ``limit`` recent messages in ``channel_id``, oldest
    first. Used by the reply runtime to build conversational context.
    """
    raw = await get_client().fetch_recent_messages(channel_id, limit)
    return list(cast(Sequence[HistoryMessage], raw))


async def start_thread(channel_id: int, message_id: int, name: str) -> int:
    """Anchor a public thread to ``message_id`` in ``channel_id``.

    Discord truncates thread names at 100 chars; the underlying client
    is responsible for that. Returns the new thread id.
    """
    return await get_client().start_thread(channel_id, message_id, name)


__all__ = [
    "DISCORD_CONTENT_LIMIT",
    "client_ready",
    "edit_message",
    "fetch_recent_messages",
    "get_client",
    "post_embed",
    "post_long_message",
    "post_message",
    "set_client",
    "start_thread",
    "upload_file",
]

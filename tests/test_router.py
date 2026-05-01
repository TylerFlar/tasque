"""Tests for the Discord message router."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import select

from tasque.discord import poster, threads
from tasque.discord.router import IncomingMessage, route_message
from tasque.memory.db import get_session
from tasque.memory.entities import Note
from tasque.reply.runtime import HistoryMessage, ReplyResult


@pytest.fixture(autouse=True)
def reset_threads_and_poster(tmp_path: Any) -> Any:
    """Per-test thread registry isolated from the user's data dir."""
    registry = tmp_path / "discord_threads.json"
    import os

    old = os.environ.get("TASQUE_DISCORD_THREAD_REGISTRY")
    os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = registry.as_posix()
    threads.reset_cache()
    poster.set_client(None)
    yield
    threads.reset_cache()
    poster.set_client(None)
    if old is None:
        os.environ.pop("TASQUE_DISCORD_THREAD_REGISTRY", None)
    else:
        os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = old


class _FakePoster:
    """Captures every poster call so the router can run without nextcord."""

    def __init__(self) -> None:
        self.sent_messages: list[tuple[int, str]] = []
        self.sent_embeds: list[tuple[int, dict[str, Any], Any]] = []
        self.fetched: list[tuple[int, int]] = []

    async def send_message(self, channel_id: int, content: str) -> int:
        self.sent_messages.append((channel_id, content))
        return 1000 + len(self.sent_messages)

    async def send_embed(
        self, channel_id: int, embed: dict[str, Any], view: Any | None = None
    ) -> int:
        self.sent_embeds.append((channel_id, embed, view))
        return 2000 + len(self.sent_embeds)

    async def edit_message(
        self,
        channel_id: int,
        message_id: int,
        *,
        content: str | None = None,
        embed: dict[str, Any] | None = None,
        view: Any | None = None,
    ) -> None:
        return None

    async def upload_file(
        self, channel_id: int, path: Any, *, content: str | None = None
    ) -> int:
        return 0

    async def fetch_recent_messages(
        self, channel_id: int, limit: int
    ) -> list[HistoryMessage]:
        self.fetched.append((channel_id, limit))
        return []

    async def start_thread(
        self, channel_id: int, message_id: int, name: str
    ) -> int:
        return 9000 + message_id


@dataclass
class _Att:
    filename: str
    content_type: str | None
    local_path: str


def _bucket_thread() -> int:
    threads.set_thread_id(threads.bucket_purpose("health"), 5555)
    return 5555


def _msg(content: str, channel_id: int = 5555, message_id: int = 7) -> IncomingMessage:
    return IncomingMessage(
        message_id=message_id,
        channel_id=channel_id,
        author="tyler",
        content=content,
        is_bot=False,
        attachments=[],
    )


def _fake_reply(
    text: str = "got it", record: dict[str, Any] | None = None
) -> Any:
    def fn(
        bucket: str,
        content: str,
        *,
        history: Sequence[HistoryMessage] | None = None,
        attachments: Sequence[Any] | None = None,
    ) -> ReplyResult:
        if record is not None:
            record["bucket"] = bucket
            record["content"] = content
            record["history"] = list(history or [])
            record["attachments"] = list(attachments or [])
        return {"text": text, "tool_calls": []}

    return fn


def _fake_trigger(record: dict[str, Any]) -> Any:
    def fn(bucket: str, reason: str, *, dedup_key: str | None = None) -> str:
        record["bucket"] = bucket
        record["reason"] = reason
        record["dedup_key"] = dedup_key
        return "trigger-1"

    return fn


# ------------------------------------------------------------- routing


@pytest.mark.asyncio
async def test_route_bucket_message_persists_note_and_enqueues_trigger() -> None:
    _bucket_thread()
    poster.set_client(_FakePoster())  # type: ignore[arg-type]

    reply_record: dict[str, Any] = {}
    trigger_record: dict[str, Any] = {}
    result = await route_message(
        _msg("how is my sleep tracking lately?"),
        coach_reply=_fake_reply("looking ok", reply_record),
        coach_trigger=_fake_trigger(trigger_record),
    )

    assert result.skipped is False
    assert result.bucket == "health"
    assert result.reply is not None and result.reply["text"] == "looking ok"
    assert result.posted_message_ids, "router should have posted the reply"
    assert result.coach_trigger_id == "trigger-1"

    # Note row created BEFORE the agent call.
    with get_session() as sess:
        rows = list(sess.execute(select(Note)).scalars().all())
    assert len(rows) == 1
    note = rows[0]
    assert note.source == "user"
    assert note.durability == "ephemeral"
    assert note.bucket == "health"
    assert note.content == "how is my sleep tracking lately?"

    # Coach trigger keyed by message id with dedup.
    assert trigger_record["dedup_key"] == "reply:7"

    # Reply binding actually called with the bucket + content.
    assert reply_record["bucket"] == "health"
    assert reply_record["content"] == "how is my sleep tracking lately?"


@pytest.mark.asyncio
async def test_route_skips_trivial_acks() -> None:
    _bucket_thread()
    poster.set_client(_FakePoster())  # type: ignore[arg-type]

    for ack in ("ok", "yes", "sure.", "Thx", "no"):
        result = await route_message(
            _msg(ack),
            coach_reply=_fake_reply("nope"),
            coach_trigger=_fake_trigger({}),
        )
        assert result.skipped is True
        assert result.reason == "trivial-ack"

    # And no Note rows from any of those.
    with get_session() as sess:
        rows = list(sess.execute(select(Note)).scalars().all())
    assert rows == []


@pytest.mark.asyncio
async def test_route_does_not_skip_short_meaningful_messages() -> None:
    """Short instructions like "do both", "approve", "go" are NOT
    trivial acks — they're directives. Earlier versions of the router
    dropped any message ≤10 chars, which silently lost user intent
    (concrete repro: a Discord conversation where the user replied
    "Do both" to a tasque proposal and got no response)."""
    _bucket_thread()
    poster.set_client(_FakePoster())  # type: ignore[arg-type]

    for content in ("do both", "approve", "go", "look", "fix it"):
        result = await route_message(
            _msg(content),
            coach_reply=_fake_reply("got it"),
            coach_trigger=_fake_trigger({}),
        )
        assert result.skipped is False, (
            f"message {content!r} was incorrectly skipped: {result.reason}"
        )


@pytest.mark.asyncio
async def test_route_skips_unknown_thread() -> None:
    poster.set_client(_FakePoster())  # type: ignore[arg-type]
    # No thread registered → channel id doesn't map to a purpose.
    result = await route_message(
        _msg("can you take a look at my queued jobs", channel_id=9999),
        coach_reply=_fake_reply("never called"),
        coach_trigger=_fake_trigger({}),
    )
    assert result.skipped is True
    assert result.reason == "unknown-thread"

    with get_session() as sess:
        rows = list(sess.execute(select(Note)).scalars().all())
    assert rows == []


@pytest.mark.asyncio
async def test_route_skips_bot_authored_messages() -> None:
    _bucket_thread()
    poster.set_client(_FakePoster())  # type: ignore[arg-type]
    msg = IncomingMessage(
        message_id=12,
        channel_id=5555,
        author="tasque-bot",
        content="hello world from the bot",
        is_bot=True,
    )
    result = await route_message(
        msg,
        coach_reply=_fake_reply("never called"),
        coach_trigger=_fake_trigger({}),
    )
    assert result.skipped is True
    assert result.reason == "bot-author"


@pytest.mark.asyncio
async def test_route_per_job_thread_dispatches_to_job_bucket_coach() -> None:
    """A reply in a per-job thread must route to the coach for that
    job's bucket — the user's example: art practice job → creative
    coach in the same thread."""
    from tasque.memory.entities import QueuedJob
    from tasque.memory.repo import write_entity

    poster.set_client(_FakePoster())  # type: ignore[arg-type]
    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="creative",
            directive="practice the brief",
            reason="weekly cycle",
            fire_at="now",
            status="completed",
            queued_by="creative",
            tier="medium",
            thread_id="7777",
        )
    )
    reply_record: dict[str, Any] = {}
    trigger_record: dict[str, Any] = {}

    msg = IncomingMessage(
        message_id=99,
        channel_id=7777,
        author="tyler",
        content="here is my piece, what do you think?",
    )
    result = await route_message(
        msg,
        coach_reply=_fake_reply("nice contrast", reply_record),
        coach_trigger=_fake_trigger(trigger_record),
    )

    assert result.skipped is False
    assert result.route == "queued-job-thread"
    assert result.bucket == "creative"
    assert reply_record["bucket"] == "creative"
    assert trigger_record["bucket"] == "creative"
    # Note carries the route tag so we can audit later.
    with get_session() as sess:
        rows = list(sess.execute(select(Note)).scalars().all())
    assert rows[0].meta.get("route") == "queued-job-thread"
    assert rows[0].bucket == "creative"
    # Sanity: job row exists (not optimised away).
    assert job.id


@pytest.mark.asyncio
async def test_route_per_chain_thread_dispatches_to_chain_bucket_coach() -> None:
    """A reply in a per-chain thread routes to the coach for that
    chain's bucket."""
    from tasque.memory.entities import ChainRun
    from tasque.memory.repo import write_entity

    poster.set_client(_FakePoster())  # type: ignore[arg-type]
    run = write_entity(
        ChainRun(
            chain_id="abc",
            chain_name="lycanworks-sweep",
            bucket="creative",
            status="completed",
            started_at="2026-04-26T00:00:00.000000Z",
            thread_id="8888",
        )
    )
    reply_record: dict[str, Any] = {}
    trigger_record: dict[str, Any] = {}
    msg = IncomingMessage(
        message_id=100,
        channel_id=8888,
        author="tyler",
        content="any updates from the sweep?",
    )
    result = await route_message(
        msg,
        coach_reply=_fake_reply("here's a recap", reply_record),
        coach_trigger=_fake_trigger(trigger_record),
    )
    assert result.skipped is False
    assert result.route == "chain-run-thread"
    assert result.bucket == "creative"
    assert reply_record["bucket"] == "creative"
    assert trigger_record["bucket"] == "creative"
    assert run.id  # sanity


@pytest.mark.asyncio
async def test_route_strategist_thread_dispatches_to_strategist_reply() -> None:
    """A reply in the strategist thread routes to the strategist binding,
    bypasses bucket-coach trigger queues."""
    threads.set_thread_id(threads.PURPOSE_STRATEGIST, 6666)
    poster.set_client(_FakePoster())  # type: ignore[arg-type]

    strategist_record: dict[str, Any] = {}

    def fake_strategist(
        content: str,
        *,
        history: Sequence[HistoryMessage] | None = None,
        attachments: Sequence[Any] | None = None,
    ) -> ReplyResult:
        strategist_record["content"] = content
        return {"text": "strategy reply", "tool_calls": []}

    trigger_record: dict[str, Any] = {}
    msg = IncomingMessage(
        message_id=101,
        channel_id=6666,
        author="tyler",
        content="add a long-term goal: ship lycanworks v2 by june",
    )
    result = await route_message(
        msg,
        strategist_reply=fake_strategist,
        coach_trigger=_fake_trigger(trigger_record),
    )
    assert result.skipped is False
    assert result.route == "registry-strategist"
    assert result.bucket is None
    assert strategist_record["content"].startswith("add a long-term goal")
    # Strategist replies don't enqueue a coach trigger — there's no
    # bucket coach to wake.
    assert "bucket" not in trigger_record


@pytest.mark.asyncio
async def test_route_per_job_thread_skips_when_bucketless_job() -> None:
    """A bucketless job has no coach to route to; skip with
    ``unknown-thread`` rather than fabricating a destination."""
    from tasque.memory.entities import QueuedJob
    from tasque.memory.repo import write_entity

    poster.set_client(_FakePoster())  # type: ignore[arg-type]
    write_entity(
        QueuedJob(
            kind="worker",
            bucket=None,
            directive="x",
            reason="",
            fire_at="now",
            status="completed",
            queued_by="cli",
            tier="small",
            thread_id="4444",
        )
    )
    msg = IncomingMessage(
        message_id=110, channel_id=4444, author="tyler", content="any updates"
    )
    result = await route_message(msg, coach_reply=_fake_reply("never"))
    assert result.skipped is True
    assert result.reason == "unknown-thread"


@pytest.mark.asyncio
async def test_route_passes_attachments_to_reply() -> None:
    _bucket_thread()
    poster.set_client(_FakePoster())  # type: ignore[arg-type]
    reply_record: dict[str, Any] = {}
    msg = IncomingMessage(
        message_id=42,
        channel_id=5555,
        author="tyler",
        content="here are my labs",
        attachments=[_Att("labs.pdf", "application/pdf", "/tmp/labs.pdf")],
    )
    await route_message(
        msg,
        coach_reply=_fake_reply("got it", reply_record),
        coach_trigger=_fake_trigger({}),
    )
    atts = reply_record["attachments"]
    assert len(atts) == 1
    assert atts[0]["filename"] == "labs.pdf"
    assert atts[0]["local_path"] == "/tmp/labs.pdf"
    assert atts[0]["content_type"] == "application/pdf"

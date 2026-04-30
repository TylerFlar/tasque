"""Route an inbound Discord message to the right reply binding.

Routing fallback chain (first hit wins):

1. Registry purpose ``bucket:<bucket>`` → bucket coach reply.
2. Registry purpose ``strategist`` → strategist reply.
3. ``QueuedJob.thread_id == channel_id`` → coach for ``job.bucket``
   (per-job worker-run thread; the bucket coach owns the conversation
   about that job because it owns the memory the user is talking about).
4. ``ChainRun.thread_id == channel_id`` → coach for ``chain_run.bucket``
   (per-chain thread, same reasoning).
5. Anything else → skipped with ``unknown-thread``.

DLQ threads are intentionally not in the fallback chain — they're
status-only embeds with retry/resolve buttons; replies in them have
nowhere meaningful to go.

Trivial acks are skipped *before* persisting the ephemeral note —
otherwise a "ok" reply would still create a note row. The ephemeral
note IS persisted before invoking the agent for non-trivial messages,
so a crash mid-reply doesn't lose the input.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import structlog
from sqlalchemy import select

from tasque.buckets import ALL_BUCKETS, Bucket
from tasque.coach.trigger import enqueue as enqueue_coach_trigger
from tasque.discord import poster, threads
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun, Note, QueuedJob
from tasque.memory.repo import write_entity
from tasque.reply.coach import run_coach_reply
from tasque.reply.runtime import AttachmentMarker, HistoryMessage, ReplyResult
from tasque.reply.strategist import run_strategist_reply

log = structlog.get_logger(__name__)

TRIVIAL_ACK_RE = re.compile(r"^(ok|thx|yes|no|sure)\.?$", re.IGNORECASE)
TRIVIAL_ACK_LIMIT = 10


@runtime_checkable
class _AttachmentLike(Protocol):
    filename: str
    content_type: str | None
    local_path: str


@dataclass
class IncomingMessage:
    """The minimal slice of a Discord message the router needs.

    The bot's on_message handler converts a ``nextcord.Message`` into
    one of these so the router stays free of nextcord coupling.
    """

    message_id: int
    channel_id: int
    author: str
    content: str
    is_bot: bool = False
    attachments: Sequence[_AttachmentLike] = field(default_factory=list)


@dataclass
class RouteResult:
    """What the router did with a message — used by tests."""

    skipped: bool
    reason: str
    bucket: Bucket | None = None
    # How we resolved the destination: "registry-bucket", "registry-strategist",
    # "queued-job-thread", "chain-run-thread", or None when skipped.
    route: str | None = None
    note_id: str | None = None
    coach_trigger_id: str | None = None
    reply: ReplyResult | None = None
    posted_message_ids: list[int] = field(default_factory=list)


# ----------------------------------------------------------------- helpers


def _is_trivial(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    if len(stripped) <= TRIVIAL_ACK_LIMIT:
        return True
    return bool(TRIVIAL_ACK_RE.match(stripped))


def _resolve_registry_purpose(channel_id: int) -> str | None:
    for purpose, tid in threads.all_thread_ids().items():
        if tid == channel_id:
            return purpose
    return None


def _bucket_from_purpose(purpose: str) -> Bucket | None:
    if not purpose.startswith("bucket:"):
        return None
    name = purpose.split(":", 1)[1]
    if name not in ALL_BUCKETS:
        return None
    return name  # type: ignore[return-value]


def _coerce_bucket(value: str | None) -> Bucket | None:
    if value is None or value not in ALL_BUCKETS:
        return None
    return value  # type: ignore[return-value]


def _bucket_from_queued_job_thread(channel_id: int) -> Bucket | None:
    """If ``channel_id`` matches a ``QueuedJob.thread_id``, return that
    job's bucket. Returns None if no row matches or the row is bucketless.
    """
    cid = str(channel_id)
    with get_session() as sess:
        stmt = (
            select(QueuedJob.bucket)
            .where(QueuedJob.thread_id == cid)
            .limit(1)
        )
        bucket = sess.execute(stmt).scalars().first()
    return _coerce_bucket(bucket)


def _bucket_from_chain_run_thread(channel_id: int) -> Bucket | None:
    """If ``channel_id`` matches a ``ChainRun.thread_id``, return that
    chain's bucket. Returns None if no row matches or the row is
    bucketless.
    """
    cid = str(channel_id)
    with get_session() as sess:
        stmt = (
            select(ChainRun.bucket)
            .where(ChainRun.thread_id == cid)
            .limit(1)
        )
        bucket = sess.execute(stmt).scalars().first()
    return _coerce_bucket(bucket)


@dataclass
class _Destination:
    """Where a message should be routed.

    Exactly one of ``bucket`` / ``strategist`` is set.
    """

    route: str
    bucket: Bucket | None = None
    strategist: bool = False


def _resolve_destination(channel_id: int) -> _Destination | None:
    """Walk the routing fallback chain. Returns None if the channel has
    no resolvable destination — caller skips with ``unknown-thread``.
    """
    purpose = _resolve_registry_purpose(channel_id)
    if purpose is not None:
        bucket = _bucket_from_purpose(purpose)
        if bucket is not None:
            return _Destination(route="registry-bucket", bucket=bucket)
        if purpose == threads.PURPOSE_STRATEGIST:
            return _Destination(route="registry-strategist", strategist=True)
        # Other registry purposes (jobs, dlq, ops_panel) are not
        # reply destinations — they're channel parents for embeds.
        return None

    bucket = _bucket_from_queued_job_thread(channel_id)
    if bucket is not None:
        return _Destination(route="queued-job-thread", bucket=bucket)

    bucket = _bucket_from_chain_run_thread(channel_id)
    if bucket is not None:
        return _Destination(route="chain-run-thread", bucket=bucket)

    return None


def _to_attachment_markers(
    attachments: Sequence[_AttachmentLike],
) -> list[AttachmentMarker]:
    out: list[AttachmentMarker] = []
    for a in attachments:
        out.append(
            {
                "filename": a.filename,
                "local_path": a.local_path,
                "content_type": a.content_type or "application/octet-stream",
            }
        )
    return out


CoachReplyFn = Callable[..., ReplyResult]
StrategistReplyFn = Callable[..., ReplyResult]
CoachTriggerFn = Callable[..., Any]


# ----------------------------------------------------------------- main entry


async def route_message(
    message: IncomingMessage,
    *,
    coach_reply: CoachReplyFn | None = None,
    strategist_reply: StrategistReplyFn | None = None,
    coach_trigger: CoachTriggerFn | None = None,
    fetch_history: Callable[[int, int], Awaitable[list[HistoryMessage]]] | None = None,
) -> RouteResult:
    """Route an inbound message. Returns a :class:`RouteResult`.

    ``coach_reply`` defaults to :func:`tasque.reply.coach.run_coach_reply`,
    ``strategist_reply`` to :func:`tasque.reply.strategist.run_strategist_reply`;
    tests pass fakes to bypass the LLM. ``coach_trigger`` defaults to
    :func:`tasque.coach.trigger.enqueue` for the post-reply hook (only
    fired when a bucket coach handles the reply — strategist replies act
    synchronously through their tools and have no trigger queue).
    """
    if message.is_bot:
        return RouteResult(skipped=True, reason="bot-author")

    if _is_trivial(message.content):
        return RouteResult(skipped=True, reason="trivial-ack")

    dest = _resolve_destination(message.channel_id)
    if dest is None:
        return RouteResult(skipped=True, reason="unknown-thread")

    # Persist the user's content as an ephemeral Note BEFORE invoking
    # the agent so a crash doesn't lose the input. Strategist replies
    # are cross-bucket and don't have a natural bucket to attach the
    # note to — we still record it under ``meta`` so the message isn't
    # lost, just without a bucket binding.
    note = Note(
        content=message.content,
        bucket=dest.bucket,
        durability="ephemeral",
        source="user",
        meta={
            "discord_message_id": str(message.message_id),
            "discord_channel_id": str(message.channel_id),
            "author": message.author,
            "route": dest.route,
        },
    )
    written = write_entity(note)

    history: list[HistoryMessage] = []
    if fetch_history is not None:
        try:
            raw = await fetch_history(message.channel_id, 20)
            # Drop the user's current message if Discord returned it.
            history = [
                h
                for h in raw
                if h.get("content", "").strip() != message.content.strip()
            ]
        except Exception:
            log.exception("discord.router.history_fetch_failed")
    elif _has_default_fetcher():
        try:
            raw = await poster.fetch_recent_messages(message.channel_id, 20)
            history = [
                h
                for h in raw
                if h.get("content", "").strip() != message.content.strip()
            ]
        except Exception:
            log.exception("discord.router.history_fetch_failed")

    attachments = _to_attachment_markers(message.attachments)

    import asyncio

    reply: ReplyResult
    try:
        async with poster.typing_indicator(message.channel_id):
            if dest.strategist:
                sfn: StrategistReplyFn = strategist_reply or run_strategist_reply
                reply = await asyncio.to_thread(
                    sfn,
                    message.content,
                    history=history,
                    attachments=attachments,
                )
            else:
                assert dest.bucket is not None  # type narrowing
                cfn: CoachReplyFn = coach_reply or run_coach_reply
                reply = await asyncio.to_thread(
                    cfn,
                    dest.bucket,
                    message.content,
                    history=history,
                    attachments=attachments,
                )
    except Exception:
        log.exception(
            "discord.router.reply_failed",
            route=dest.route,
            bucket=dest.bucket,
            message_id=message.message_id,
        )
        raise

    posted_ids: list[int] = []
    text = reply.get("text") or ""
    if text.strip():
        posted_ids = await poster.post_long_message(message.channel_id, text)

    trigger_id: str | None = None
    if dest.bucket is not None:
        trigger_fn: CoachTriggerFn = coach_trigger or enqueue_coach_trigger
        try:
            trigger_id = trigger_fn(
                dest.bucket,
                "reply",
                dedup_key=f"reply:{message.message_id}",
            )
        except Exception:
            log.exception(
                "discord.router.coach_trigger_failed",
                route=dest.route,
                bucket=dest.bucket,
                message_id=message.message_id,
            )

    return RouteResult(
        skipped=False,
        reason="ok",
        bucket=dest.bucket,
        route=dest.route,
        note_id=written.id,
        coach_trigger_id=trigger_id,
        reply=reply,
        posted_message_ids=posted_ids,
    )


def _has_default_fetcher() -> bool:
    """True iff the poster has a client installed (i.e. running in a
    real bot context). Tests without a poster client skip history."""
    try:
        poster.get_client()
        return True
    except RuntimeError:
        return False


__all__ = [
    "IncomingMessage",
    "RouteResult",
    "route_message",
]

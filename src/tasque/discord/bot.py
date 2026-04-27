"""nextcord client wiring for the tasque bot.

One bot. Three event handlers (``on_ready``, ``on_message``,
``on_interaction``). Brings up the chain UI watcher background task on
start. The bot adapts nextcord's API to the stateless poster
interface so the rest of the discord layer doesn't import nextcord
directly.

Configuration is via env vars:
- ``TASQUE_DISCORD_TOKEN`` — required.
- ``TASQUE_DISCORD_COACH_CHANNEL_ID`` — coach channel id (required so
  per-bucket / per-chain threads can be created underneath it).
- ``TASQUE_DISCORD_DLQ_CHANNEL_ID`` — optional override; if set, the
  DLQ thread is created under this channel.
- ``TASQUE_DISCORD_JOBS_CHANNEL_ID`` — optional fallback for
  bucket-less worker reports.
- ``TASQUE_ATTACHMENTS_DIR`` — where to save user-uploaded files
  (default: ``<data_dir>/attachments``).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nextcord
import structlog

from tasque.config import get_settings
from tasque.discord import chain_ui, ops_panel, ops_watcher, poster, threads
from tasque.discord.router import IncomingMessage, route_message
from tasque.memory.entities import Attachment
from tasque.memory.repo import write_entity
from tasque.reply.runtime import HistoryMessage

log = structlog.get_logger(__name__)


# ----------------------------------------------------------------- attachments

@dataclass
class _SavedAttachment:
    filename: str
    content_type: str | None
    local_path: str


def _attachments_dir() -> Path:
    raw = os.environ.get("TASQUE_ATTACHMENTS_DIR")
    if raw:
        return Path(raw)
    return get_settings().data_dir / "attachments"


async def _save_one_attachment(
    raw_attachment: Any,
    *,
    bucket: str | None,
    discord_message_id: int,
    discord_channel_id: int,
) -> _SavedAttachment:
    """Read a nextcord ``Attachment`` and write it to the attachments dir
    under its sha256. Records a ``Attachment`` row.
    """
    data: bytes = await raw_attachment.read()
    digest = hashlib.sha256(data).hexdigest()
    suffix = Path(raw_attachment.filename).suffix
    target_dir = _attachments_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{digest}{suffix}"
    if not target.exists():
        target.write_bytes(data)
    row = Attachment(
        filename=raw_attachment.filename,
        content_type=raw_attachment.content_type or "application/octet-stream",
        size_bytes=len(data),
        bucket=bucket,
        source="user",
        local_path=target.as_posix(),
        sha256=digest,
        discord_message_id=str(discord_message_id),
        discord_channel_id=str(discord_channel_id),
    )
    written = write_entity(row)
    return _SavedAttachment(
        filename=written.filename,
        content_type=written.content_type,
        local_path=written.local_path,
    )


# ----------------------------------------------------------------- adapter

class _NextcordPosterAdapter:
    """Adapts the live nextcord client to the poster's protocol.

    Each call resolves the channel via ``client.fetch_channel`` (which
    works for both regular channels and threads) and dispatches the
    appropriate nextcord call.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    async def _channel(self, channel_id: int) -> Any:
        chan = self._client.get_channel(channel_id)
        if chan is None:
            chan = await self._client.fetch_channel(channel_id)
        return chan

    async def send_message(self, channel_id: int, content: str) -> int:
        chan = await self._channel(channel_id)
        msg = await chan.send(content=content)
        return cast(int, msg.id)

    async def send_embed(
        self, channel_id: int, embed: dict[str, Any], view: Any | None = None
    ) -> int:
        chan = await self._channel(channel_id)
        embed_obj = nextcord.Embed.from_dict(embed)
        kwargs: dict[str, Any] = {"embed": embed_obj}
        if view is not None:
            kwargs["view"] = view
        msg = await chan.send(**kwargs)
        return cast(int, msg.id)

    async def edit_message(
        self,
        channel_id: int,
        message_id: int,
        *,
        content: str | None = None,
        embed: dict[str, Any] | None = None,
        view: Any | None = None,
    ) -> None:
        chan = await self._channel(channel_id)
        msg = await chan.fetch_message(message_id)
        kwargs: dict[str, Any] = {}
        if content is not None:
            kwargs["content"] = content
        if embed is not None:
            kwargs["embed"] = nextcord.Embed.from_dict(embed)
        kwargs["view"] = view
        await msg.edit(**kwargs)

    async def upload_file(
        self,
        channel_id: int,
        path: Path,
        *,
        content: str | None = None,
    ) -> int:
        chan = await self._channel(channel_id)
        msg = await chan.send(content=content, file=nextcord.File(path.as_posix()))
        return cast(int, msg.id)

    async def fetch_recent_messages(
        self, channel_id: int, limit: int
    ) -> list[HistoryMessage]:
        chan = await self._channel(channel_id)
        out: list[HistoryMessage] = []
        async for msg in chan.history(limit=limit):
            out.append(
                {
                    "author": msg.author.display_name,
                    "content": msg.content,
                    "is_bot": bool(getattr(msg.author, "bot", False)),
                }
            )
        out.reverse()  # oldest first
        return out

    async def start_thread(
        self, channel_id: int, message_id: int, name: str
    ) -> int:
        chan = await self._channel(channel_id)
        msg = await chan.fetch_message(message_id)
        thread = await msg.create_thread(
            name=name[:100], auto_archive_duration=10080
        )
        return cast(int, thread.id)

    @asynccontextmanager
    async def typing(self, channel_id: int) -> AsyncGenerator[None, None]:
        """Show the typing indicator in ``channel_id`` for the duration
        of the wrapped ``async with`` block.

        nextcord's ``Messageable.typing()`` is itself an async context
        manager that auto-refreshes the indicator (Discord's raw event
        decays after ~10s); we forward to it via ``AsyncExitStack`` so
        that channel-resolution or typing-setup failures degrade to a
        no-op rather than breaking the wrapped reply path.
        """
        async with AsyncExitStack() as stack:
            try:
                chan = await self._channel(channel_id)
                await stack.enter_async_context(chan.typing())
            except Exception:
                log.exception("discord.bot.typing_failed", channel_id=channel_id)
            yield


# ----------------------------------------------------------------- bot factory

# Every chain/job/coach/ops/dlq channel env var is REQUIRED. There are
# no fallbacks: a missing or unparseable id is a startup error so the
# operator sees the misconfiguration before the bot connects, instead
# of finding silently-misrouted messages later. ``require_channel_ids``
# is the single point of validation, called from ``build_bot``.

_REQUIRED_CHANNEL_ENV_VARS: tuple[str, ...] = (
    "TASQUE_DISCORD_COACH_CHANNEL_ID",
    "TASQUE_DISCORD_DLQ_CHANNEL_ID",
    "TASQUE_DISCORD_JOBS_CHANNEL_ID",
    "TASQUE_DISCORD_OPS_CHANNEL_ID",
    "TASQUE_DISCORD_CHAINS_CHANNEL_ID",
)


def require_channel_ids() -> dict[str, int]:
    """Read every required channel-id env var or raise.

    Returns a mapping from env-var name to its parsed int. Raises
    ``RuntimeError`` listing every missing or unparseable variable so
    the operator can fix them all in one pass instead of one-at-a-time.
    """
    out: dict[str, int] = {}
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    for name in _REQUIRED_CHANNEL_ENV_VARS:
        raw = os.environ.get(name)
        if not raw:
            missing.append(name)
            continue
        try:
            out[name] = int(raw)
        except ValueError:
            invalid.append((name, raw))
    if missing or invalid:
        parts: list[str] = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if invalid:
            parts.append(
                "not an integer: "
                + ", ".join(f"{n}={v!r}" for n, v in invalid)
            )
        raise RuntimeError(
            "tasque Discord channel configuration is incomplete — every "
            "channel id is required, no fallbacks. " + "; ".join(parts)
        )
    return out


def _coach_channel_id() -> int:
    return require_channel_ids()["TASQUE_DISCORD_COACH_CHANNEL_ID"]


def _dlq_channel_id() -> int:
    return require_channel_ids()["TASQUE_DISCORD_DLQ_CHANNEL_ID"]


def _jobs_channel_id() -> int:
    return require_channel_ids()["TASQUE_DISCORD_JOBS_CHANNEL_ID"]


def _bot_token() -> str:
    raw = os.environ.get("TASQUE_DISCORD_TOKEN")
    if not raw:
        raise RuntimeError(
            "TASQUE_DISCORD_TOKEN is required to start the Discord bot"
        )
    return raw


def _guild_ids() -> list[int]:
    """Optional guild scope for slash commands. Setting
    ``TASQUE_DISCORD_GUILD_ID`` registers commands instantly to that
    guild; otherwise they're global (Discord may take up to an hour to
    propagate)."""
    raw = os.environ.get("TASQUE_DISCORD_GUILD_ID")
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        try:
            out.append(int(part.strip()))
        except ValueError:
            continue
    return out


@dataclass
class TasqueBotHandle:
    """The pieces of a running bot the serve loop holds onto."""

    client: Any
    token: str
    watcher_task: asyncio.Task[int] | None = None
    ops_watcher_task: asyncio.Task[int] | None = None
    stop: asyncio.Event | None = None


def build_bot(
    *,
    on_message_handler: Callable[[Any], Awaitable[None]] | None = None,
) -> TasqueBotHandle:
    """Construct (but do not start) a TasqueBotHandle.

    ``on_message_handler`` overrides the default router-dispatch path —
    used by integration tests that want to assert on the bot's
    behaviour without going through nextcord's gateway.
    """
    # Validate every required channel-id env var up front so a
    # misconfigured deployment fails before the bot even connects.
    require_channel_ids()

    intents = nextcord.Intents.default()
    intents.message_content = True
    client = nextcord.Client(intents=intents)
    handle = TasqueBotHandle(client=client, token=_bot_token())

    @client.event
    async def on_ready() -> None:  # pyright: ignore[reportUnusedFunction]
        log.info("discord.bot.ready", user=str(client.user))
        poster.set_client(_NextcordPosterAdapter(client))

        await _ensure_known_threads(client, _coach_channel_id())

        # Start the chain UI watcher.
        if handle.stop is None:
            handle.stop = asyncio.Event()
        handle.watcher_task = asyncio.create_task(
            chain_ui.run_watcher(
                stop=handle.stop,
                view_factory=chain_ui.build_approval_view,
            ),
            name="tasque-chain-ui-watcher",
        )

        # Start the live ops-panel watcher. The ops channel id was
        # validated at build time, so this always starts.
        handle.ops_watcher_task = asyncio.create_task(
            ops_watcher.run_ops_watcher(stop=handle.stop),
            name="tasque-ops-panel-watcher",
        )

    @client.event
    async def on_message(message: Any) -> None:  # pyright: ignore[reportUnusedFunction]
        if on_message_handler is not None:
            await on_message_handler(message)
            return
        if message.author == client.user:
            return
        await _route_nextcord_message(message)

    @client.event
    async def on_interaction(interaction: Any) -> None:  # pyright: ignore[reportUnusedFunction]
        await _handle_interaction(client, interaction)

    _register_status_command(client)

    return handle


def _register_status_command(client: Any) -> None:
    """Register the ``/status`` ops-panel slash command on ``client``.

    Posts a fresh ephemeral embed each invocation — no live editing, no
    pinned message. Honors the Phase 6 contract by being one-shot.
    """
    guild_ids = _guild_ids()
    decorator_kwargs: dict[str, Any] = {
        "name": "status",
        "description": "Show the tasque ops panel (jobs, chains, DLQ, scheduler).",
    }
    if guild_ids:
        decorator_kwargs["guild_ids"] = guild_ids

    @client.slash_command(**decorator_kwargs)
    async def status_cmd(  # pyright: ignore[reportUnusedFunction]
        interaction: nextcord.Interaction[Any],
    ) -> None:
        try:
            snapshot = await asyncio.to_thread(ops_panel.build_ops_snapshot)
        except Exception:
            log.exception("discord.bot.ops_panel_failed")
            await interaction.response.send_message(
                content="failed to build ops panel — see daemon logs",
                ephemeral=True,
            )
            return
        embed_dict = ops_panel.build_ops_embed(snapshot)
        embed_obj = nextcord.Embed.from_dict(embed_dict)
        await interaction.response.send_message(embed=embed_obj, ephemeral=True)


# ----------------------------------------------------------------- helpers


async def _ensure_known_threads(client: Any, coach_channel_id: int) -> None:
    """Populate the thread registry.

    For each bucket purpose, if not already registered: send an intro
    message in the coach channel and anchor a public thread to it. The
    intro message keeps the thread visible (Discord otherwise hides
    threads created without a starter).

    For ``jobs`` and ``dlq``: record the configured channel id (NOT a
    thread id). Each worker run / failed job creates its own thread off
    its embed in that channel — see :func:`tasque.discord.notify`.

    For ``strategist``: anchor a thread under the coach channel, same
    shape as bucket coach threads. The strategist sits above the
    coaches conceptually but lives in the same workspace area.
    """
    coach_chan = await client.fetch_channel(coach_channel_id)

    async def _create(purpose: str, label: str) -> int:
        if purpose == threads.PURPOSE_DLQ:
            return _dlq_channel_id()
        if purpose == threads.PURPOSE_JOBS:
            return _jobs_channel_id()
        if purpose == threads.PURPOSE_STRATEGIST:
            intro = await coach_chan.send(
                content=(
                    "**Strategist**\n"
                    "Reply in this thread to talk to the strategist about "
                    "long-horizon goals or cross-bucket coordination."
                )
            )
            thread = await intro.create_thread(
                name="strategist", auto_archive_duration=10080
            )
            log.info(
                "discord.bot.thread_created",
                purpose=purpose,
                thread_id=thread.id,
                anchor_message_id=intro.id,
            )
            return cast(int, thread.id)
        # Bucket purpose: post intro message, anchor thread to it.
        bucket_name = (
            label.removeprefix("coach-") if label.startswith("coach-") else label
        )
        intro = await coach_chan.send(
            content=(
                f"**Coach: {bucket_name}**\n"
                f"Reply in this thread to talk to the {bucket_name} coach."
            )
        )
        thread = await intro.create_thread(
            name=f"coach-{bucket_name}"[:100], auto_archive_duration=10080
        )
        log.info(
            "discord.bot.thread_created",
            purpose=purpose,
            thread_id=thread.id,
            anchor_message_id=intro.id,
        )
        return cast(int, thread.id)

    await threads.ensure_threads(create_missing=_create)


async def _route_nextcord_message(message: Any) -> None:
    """Convert a ``nextcord.Message`` into an ``IncomingMessage`` and route it.

    Saves any attachments to ``data/attachments/`` first so the agent
    can refer to them by local path.
    """
    purpose = _resolve_thread_purpose(message.channel.id)
    bucket: str | None = None
    if purpose is not None and purpose.startswith("bucket:"):
        bucket = purpose.split(":", 1)[1]

    saved: list[_SavedAttachment] = []
    for att in getattr(message, "attachments", []):
        try:
            saved.append(
                await _save_one_attachment(
                    att,
                    bucket=bucket,
                    discord_message_id=message.id,
                    discord_channel_id=message.channel.id,
                )
            )
        except Exception:
            log.exception("discord.bot.attachment_save_failed", filename=att.filename)

    incoming = IncomingMessage(
        message_id=message.id,
        channel_id=message.channel.id,
        author=message.author.display_name,
        content=message.content,
        is_bot=bool(getattr(message.author, "bot", False)),
        attachments=cast(Any, saved),
    )
    try:
        await route_message(incoming)
    except Exception:
        log.exception(
            "discord.bot.route_failed",
            message_id=message.id,
            channel_id=message.channel.id,
        )


def _resolve_thread_purpose(channel_id: int) -> str | None:
    for purpose, tid in threads.all_thread_ids().items():
        if tid == channel_id:
            return purpose
    return None


async def _safe_defer(interaction: Any, *, custom_id: str) -> None:
    """Defer ``interaction`` if it hasn't already been acknowledged.

    Component interactions can land on us already-acked: a queued click
    that arrived during the daemon-startup gateway handshake, a retry
    from Discord's side after a slow first response, or nextcord's
    persistent-view dispatch firing first. ``interaction.response.is_done()``
    is the canonical "did anyone already ack this?" check; if yes,
    ``defer`` would raise 40060 ("Interaction has already been
    acknowledged") and we silently skip — ``followup.send`` works
    against the original token regardless.
    """
    response = getattr(interaction, "response", None)
    if response is not None:
        try:
            if response.is_done():
                return
        except Exception:
            # If is_done() itself fails, fall through to the defer
            # attempt; better to try and log than to silently skip.
            pass
        try:
            await response.defer()
        except Exception:
            log.warning(
                "discord.bot.defer_skipped",
                custom_id=custom_id,
                note="defer raised after is_done() said False; continuing to followup",
            )


async def _handle_interaction(client: Any, interaction: Any) -> None:
    """Dispatch button clicks. Supported custom_id shapes:

    - ``tasque-approve:<chain>:<step>`` / ``tasque-decline:<chain>:<step>``
      — chain approval resolution
    - ``tasque-retry:<failed_job_id>`` / ``tasque-resolve:<failed_job_id>``
      — DLQ entry actions
    - ``tasque-chain-pause:<chain>`` / ``tasque-chain-resume:<chain>``
      / ``tasque-chain-stop:<chain>`` — chain status panel controls
    """
    data = getattr(interaction, "data", None) or {}
    custom_id = data.get("custom_id") if isinstance(data, dict) else None
    if not isinstance(custom_id, str):
        return

    if custom_id.startswith("tasque-approve:") or custom_id.startswith("tasque-decline:"):
        kind, _, rest = custom_id.partition(":")
        chain_id, _, step_id = rest.partition(":")
        if not chain_id or not step_id:
            return
        resolution = "approved" if kind == "tasque-approve" else "declined"
        message = interaction.message
        await _safe_defer(interaction, custom_id=custom_id)
        try:
            await chain_ui.resolve_approval(
                chain_id,
                step_id,
                resolution,
                posted_channel_id=message.channel.id,
                posted_message_id=message.id,
            )
        except Exception:
            log.exception(
                "discord.bot.resolve_failed", chain_id=chain_id, step_id=step_id
            )
        return

    if custom_id.startswith("tasque-retry:"):
        _, _, failed_job_id = custom_id.partition(":")
        if not failed_job_id:
            return
        await _safe_defer(interaction, custom_id=custom_id)
        try:
            report = await chain_ui.handle_dlq_retry(failed_job_id)
            await interaction.followup.send(
                content=f"Retry queued: {report}", ephemeral=True
            )
        except Exception:
            log.exception(
                "discord.bot.retry_failed", failed_job_id=failed_job_id
            )
        return

    if custom_id.startswith("tasque-resolve:"):
        _, _, failed_job_id = custom_id.partition(":")
        if not failed_job_id:
            return
        await _safe_defer(interaction, custom_id=custom_id)
        try:
            from tasque.jobs.dlq import mark_resolved

            ok = await asyncio.to_thread(mark_resolved, failed_job_id)
            await interaction.followup.send(
                content=(
                    "Marked resolved." if ok else "No FailedJob with that id."
                ),
                ephemeral=True,
            )
        except Exception:
            log.exception(
                "discord.bot.resolve_failed", failed_job_id=failed_job_id
            )
        return

    if custom_id.startswith("tasque-chain-"):
        action, _, chain_id = custom_id.partition(":")
        if not chain_id:
            return
        await _safe_defer(interaction, custom_id=custom_id)
        try:
            await _handle_chain_control(action, chain_id, interaction)
        except Exception:
            log.exception(
                "discord.bot.chain_control_failed",
                action=action,
                chain_id=chain_id,
            )


async def _handle_chain_control(
    action: str, chain_id: str, interaction: Any
) -> None:
    """Apply a chain status-panel button action (pause / resume / stop).

    Keeps the interaction handler thin — sync DB writes go through
    ``asyncio.to_thread`` and the user gets an ephemeral confirmation.
    Resume re-invokes the chain graph so a paused chain actually starts
    making progress again; pause/stop just flip status (the supervisor
    isn't running mid-step in this architecture).
    """
    from tasque.chains.manager import (
        pause_chain,
        resume_chain,
        stop_chain,
    )

    if action == "tasque-chain-pause":
        ok = await asyncio.to_thread(pause_chain, chain_id)
        msg = "Paused." if ok else "Chain not found."
    elif action == "tasque-chain-stop":
        ok = await asyncio.to_thread(stop_chain, chain_id)
        msg = "Stopped." if ok else "Chain not found."
    elif action == "tasque-chain-resume":
        ok = await asyncio.to_thread(resume_chain, chain_id)
        if ok:
            # Kick the graph forward so the resumed chain actually
            # progresses — same pattern as resume_interrupted_chains.
            try:
                await asyncio.to_thread(_resume_chain_graph, chain_id)
                msg = "Resumed."
            except Exception:
                log.exception(
                    "discord.bot.chain_resume_invoke_failed",
                    chain_id=chain_id,
                )
                msg = "Resumed (status flipped, but graph re-invoke failed — see logs)."
        else:
            msg = "Chain not found."
    else:
        msg = f"Unknown chain action: {action}"

    try:
        await interaction.followup.send(content=msg, ephemeral=True)
    except Exception:
        log.exception("discord.bot.chain_control_followup_failed", action=action)


def _resume_chain_graph(chain_id: str) -> None:
    """Re-invoke the chain LangGraph for a paused chain. Mirrors the
    body of :func:`tasque.chains.scheduler.resume_interrupted_chains`
    for a single chain."""
    from langchain_core.runnables import RunnableConfig

    from tasque.chains.graph import get_compiled_chain_graph
    from tasque.chains.scheduler import maybe_finalize_status

    graph = get_compiled_chain_graph()
    cfg = RunnableConfig(configurable={"thread_id": chain_id})
    graph.invoke(None, cfg)  # type: ignore[arg-type]
    maybe_finalize_status(chain_id)


async def run_bot(handle: TasqueBotHandle) -> None:
    """Start ``handle.client`` and block until cancelled."""
    handle.stop = asyncio.Event()
    try:
        await handle.client.start(handle.token)
    finally:
        handle.stop.set()
        if handle.watcher_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await handle.watcher_task
        if handle.ops_watcher_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await handle.ops_watcher_task
        with contextlib.suppress(Exception):
            await handle.client.close()
        poster.set_client(None)


__all__ = [
    "TasqueBotHandle",
    "build_bot",
    "run_bot",
]

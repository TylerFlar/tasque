"""Live-edited ops panel watcher.

Maintains exactly ONE embed message in a dedicated ops channel. On
every tick the watcher rebuilds the snapshot and edits the message
in place. The message id is cached in the threads registry under
:data:`tasque.discord.threads.PURPOSE_OPS_PANEL` so a daemon restart
finds and reuses the same message instead of spamming a new one.

Lifecycle:

- First tick (no cached message id) → post a fresh embed, record the
  message id.
- Subsequent ticks → edit the cached message.
- Edit raises (typically the user deleted the message) → log, post a
  fresh embed, overwrite the cached id.

This is the one explicit exception to the "no live-edited dashboard"
constraint from the original Phase 6 spec — the user opted in.
The slash command and CLI snapshot are still available as one-shot
fallbacks.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import structlog

from tasque.discord import poster, threads
from tasque.discord.ops_panel import build_ops_embed, build_ops_snapshot

log = structlog.get_logger(__name__)

DEFAULT_POLL_SECONDS = 30.0
MIN_POLL_SECONDS = 5.0  # Discord rate-limits message edits.


def ops_channel_id() -> int:
    """Return the configured ops-panel channel id.

    Reads ``TASQUE_DISCORD_OPS_CHANNEL_ID``. The variable is required —
    there is no fallback. Raises ``RuntimeError`` if it is unset or not
    an integer. Daemon startup catches this through
    :func:`tasque.discord.bot.require_channel_ids` so a misconfigured
    deployment fails before the bot connects.
    """
    raw = os.environ.get("TASQUE_DISCORD_OPS_CHANNEL_ID")
    if not raw:
        raise RuntimeError(
            "TASQUE_DISCORD_OPS_CHANNEL_ID is required — set it to the "
            "channel hosting the live ops panel."
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"TASQUE_DISCORD_OPS_CHANNEL_ID is not an integer: {raw!r}"
        ) from exc


def _payload_signature(embed: dict[str, Any]) -> str:
    """Stable hash of the parts of an embed that matter for change
    detection. The footer's ``snapshot ...`` timestamp is excluded so
    we don't edit on every tick when nothing has actually changed."""
    return json.dumps(
        {"fields": embed.get("fields"), "color": embed.get("color"), "title": embed.get("title")},
        sort_keys=True,
        default=str,
    )


async def _post_or_edit(channel_id: int, embed: dict[str, Any]) -> int:
    """Post a fresh ops-panel message or edit the existing one.

    Returns the message id that was used (newly created or pre-existing).
    Falls back to a fresh post if the cached message can't be edited
    (most often because the user deleted it).
    """
    cached_message_id = threads.get_thread_id(threads.PURPOSE_OPS_PANEL)
    if cached_message_id is not None:
        try:
            await poster.edit_message(
                channel_id, cached_message_id, embed=embed, view=None
            )
            return cached_message_id
        except Exception:
            log.exception(
                "discord.ops_watcher.edit_failed",
                channel_id=channel_id,
                message_id=cached_message_id,
            )

    new_id = await poster.post_embed(channel_id, embed)
    threads.set_thread_id(threads.PURPOSE_OPS_PANEL, new_id)
    log.info(
        "discord.ops_watcher.posted",
        channel_id=channel_id,
        message_id=new_id,
    )
    return new_id


async def _tick(channel_id: int, last_signature: str | None) -> str:
    """Build the snapshot, render the embed, post or edit. Returns the
    new payload signature so the caller can skip identical ticks."""
    snapshot = await asyncio.to_thread(build_ops_snapshot)
    embed = build_ops_embed(snapshot)
    signature = _payload_signature(embed)
    if signature == last_signature and threads.get_thread_id(
        threads.PURPOSE_OPS_PANEL
    ) is not None:
        # Nothing meaningful has changed AND we already have a posted
        # message — skip the edit to stay under Discord's rate limit.
        return signature
    await _post_or_edit(channel_id, embed)
    return signature


async def run_ops_watcher(
    *,
    stop: asyncio.Event | None = None,
    channel_id: int | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_iterations: int | None = None,
) -> int:
    """Long-running task that maintains the live ops panel.

    Returns the number of ticks executed (used by tests). Resolves the
    ops-panel channel via :func:`ops_channel_id`, which raises if the
    env var is unset — daemon startup catches that through
    :func:`tasque.discord.bot.require_channel_ids`.
    """
    target = channel_id if channel_id is not None else ops_channel_id()

    interval = max(poll_seconds, MIN_POLL_SECONDS)
    last_signature: str | None = None
    iters = 0
    while True:
        if stop is not None and stop.is_set():
            return iters
        if max_iterations is not None and iters >= max_iterations:
            return iters
        iters += 1
        try:
            last_signature = await _tick(target, last_signature)
        except Exception:
            log.exception("discord.ops_watcher.tick_failed", channel_id=target)
        # Skip the trailing sleep if we're about to exit anyway — keeps
        # tests fast and shutdown snappy.
        if max_iterations is not None and iters >= max_iterations:
            return iters
        if stop is not None and stop.is_set():
            return iters
        await asyncio.sleep(interval)


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "MIN_POLL_SECONDS",
    "ops_channel_id",
    "run_ops_watcher",
]

"""Live chain-status watcher.

Polls every ``poll_seconds`` for ``ChainRun`` rows and maintains a single
status embed per chain in ``TASQUE_DISCORD_CHAINS_CHANNEL_ID`` — that
env var is required, there are no fallbacks. The message id is
persisted on ``ChainRun.status_message_id`` so a daemon restart finds
and edits the existing message instead of spamming a new one.

Terminal chains (completed / failed / stopped) get one final edit
showing the terminal status, then are no longer touched. Active chains
are re-rendered on every change to plan, completed, or failures.

Edits are skipped when the rendered embed signature hasn't changed —
Discord rate-limits message edits, and a chain that's mid-step but
hasn't transitioned otherwise generates no useful UI churn.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import structlog
from sqlalchemy import select

from tasque.chains.checkpointer import get_chain_checkpointer
from tasque.discord import notify, poster
from tasque.discord.chain_status_panel import (
    ChainStatusSnapshot,
    build_chain_status_embed,
    build_chain_status_snapshot,
    is_terminal_run,
)
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun, utc_now_iso

log = structlog.get_logger(__name__)

DEFAULT_POLL_SECONDS = 10.0
MIN_POLL_SECONDS = 3.0  # Discord rate-limits message edits.

# Discord refuses edits to messages older than 1 hour (error 30046).
# Skip the edit attempt past this age and post a fresh message instead —
# otherwise nextcord retries 5x on every tick before giving up.
EDIT_MAX_MESSAGE_AGE_SECONDS = 55 * 60
DISCORD_EPOCH_MS = 1_420_070_400_000


def _snowflake_age_seconds(message_id: int, *, now_ms: int) -> float:
    """Return the age in seconds of a Discord snowflake id."""
    created_ms = (message_id >> 22) + DISCORD_EPOCH_MS
    return max(0.0, (now_ms - created_ms) / 1000.0)

# Statuses we re-render on. Terminal chains are also picked up for one
# final edit (the watcher diffs signatures and quietly skips after the
# terminal embed has been written).
_WATCHED_STATUSES = (
    "running",
    "awaiting_approval",
    "awaiting_user",
    "paused",
    "completed",
    "failed",
    "stopped",
)


def _load_state(chain_id: str) -> dict[str, Any] | None:
    saver = get_chain_checkpointer()
    try:
        snapshot = saver.get_tuple({"configurable": {"thread_id": chain_id}})  # type: ignore[arg-type]
    except Exception:
        log.exception("discord.chain_status.load_state_failed", chain_id=chain_id)
        return None
    if snapshot is None:
        return None
    raw: dict[str, Any] = snapshot.checkpoint.get("channel_values", {})
    return raw


def _signature(embed: dict[str, Any]) -> str:
    """Stable hash key for the parts of the embed that should drive an
    edit. Field timestamps that move every tick (none currently — the
    panel uses static started/ended) are excluded by construction."""
    return json.dumps(
        {
            "title": embed.get("title"),
            "description": embed.get("description"),
            "color": embed.get("color"),
            "fields": embed.get("fields"),
        },
        sort_keys=True,
        default=str,
    )


def chains_channel_id() -> int:
    """Return the dedicated chain-status channel id.

    Reads ``TASQUE_DISCORD_CHAINS_CHANNEL_ID``. The variable is required
    — there is no fallback. Raises ``RuntimeError`` if it is unset or
    cannot be parsed as an int. Daemon startup catches this through
    :func:`tasque.discord.bot.require_channel_ids` so a misconfigured
    deployment fails before the bot connects, not during the first chain
    run.
    """
    raw = os.environ.get("TASQUE_DISCORD_CHAINS_CHANNEL_ID")
    if not raw:
        raise RuntimeError(
            "TASQUE_DISCORD_CHAINS_CHANNEL_ID is required — set it to the "
            "channel where live chain status panels should be posted."
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"TASQUE_DISCORD_CHAINS_CHANNEL_ID is not an integer: {raw!r}"
        ) from exc


def _resolve_target_channel(chain_run: ChainRun) -> int:
    """Return the channel id every chain status embed lands in.

    All live chain status panels go to ``TASQUE_DISCORD_CHAINS_CHANNEL_ID``
    — one message per chain, edited in place. There are no fallbacks.
    Per-chain threads (created under JOBS for approval embeds) and bucket
    coach threads are intentionally NOT used here.
    """
    return chains_channel_id()


def _persist_message_id(chain_run_pk: str, message_id: int) -> None:
    """Store ``message_id`` on the ``ChainRun`` row so subsequent ticks
    edit the same Discord message instead of posting a new one."""
    with get_session() as sess:
        row = sess.get(ChainRun, chain_run_pk)
        if row is None:
            return
        row.status_message_id = str(message_id)
        row.updated_at = utc_now_iso()


def _clear_message_id(chain_run_pk: str) -> None:
    """Drop the cached status message id once the chain has finished.

    A terminal chain gets exactly one final edit (the embed flips to
    completed/failed/stopped) and is then immutable — there's no further
    state worth re-rendering. Clearing the id keeps the row from looking
    "live", and the watcher's "skip terminal-with-no-status_message_id"
    guard then short-circuits future ticks for this chain instead of
    re-fetching its checkpoint state every poll forever.
    """
    with get_session() as sess:
        row = sess.get(ChainRun, chain_run_pk)
        if row is None:
            return
        if row.status_message_id is None:
            return
        row.status_message_id = None
        row.updated_at = utc_now_iso()


def _coerce_message_id(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def _post_or_edit(
    *,
    chain_run: ChainRun,
    embed: dict[str, Any],
    run_status: str,
) -> int | None:
    """Post a fresh status message or edit the cached one.

    Returns the resulting message id, or ``None`` if Discord refused
    both an edit and a fresh post — the caller retries on the next tick.
    The Pause / Resume / Stop control view is attached based on
    ``run_status``; terminal statuses get ``view=None`` so the panel
    stops being interactive once the chain has finished.
    """
    from tasque.discord.chain_ui import build_chain_control_view

    target = _resolve_target_channel(chain_run)
    view = build_chain_control_view(chain_run.chain_id, run_status)
    cached = _coerce_message_id(chain_run.status_message_id)
    if cached is not None:
        now_ms = int(time.time() * 1000)
        age_s = _snowflake_age_seconds(cached, now_ms=now_ms)
        if age_s >= EDIT_MAX_MESSAGE_AGE_SECONDS:
            log.info(
                "discord.chain_status.edit_skipped_age",
                chain_id=chain_run.chain_id[:8],
                message_id=cached,
                age_s=round(age_s, 1),
            )
            if run_status in ("completed", "failed", "stopped"):
                # Terminal chain whose cached panel fell out of editable
                # age — typically observed on daemon restart after a
                # long downtime. Re-posting a fresh terminal panel here
                # would just flood the chains channel with duplicates of
                # runs that are already finished. Clear the stale id and
                # let the "skip-retroactive" guard short-circuit future
                # ticks for this chain.
                _clear_message_id(chain_run.id)
                return None
            # Active chains still need a live panel; fall through to a
            # fresh post.
        else:
            try:
                await poster.edit_message(target, cached, embed=embed, view=view)
                return cached
            except Exception:
                log.exception(
                    "discord.chain_status.edit_failed",
                    chain_id=chain_run.chain_id,
                    channel_id=target,
                    message_id=cached,
                )
                # Fall through to a fresh post — the user probably deleted
                # the message.

    try:
        new_id = await poster.post_embed(target, embed, view=view)
    except Exception:
        log.exception(
            "discord.chain_status.post_failed",
            chain_id=chain_run.chain_id,
            channel_id=target,
        )
        return None
    _persist_message_id(chain_run.id, new_id)
    log.info(
        "discord.chain_status.posted",
        chain_id=chain_run.chain_id,
        channel_id=target,
        message_id=new_id,
    )
    return new_id


# ----------------------------------------------------------------- tick


async def _tick(
    *,
    last_signatures: dict[str, str],
    notified_terminal: set[str],
) -> int:
    """One pass over the watched chains. Returns the number of edits or
    posts performed this pass.

    ``notified_terminal`` is a per-watcher in-memory set of chain ids
    we've already fired the rich one-shot terminal embed for. Membership
    survives only as long as the watcher loop runs — a daemon restart
    re-resets it, but the "must have seen it non-terminal first" gate
    (see below) keeps that from flooding the JOBS channel with stale
    completions on boot.
    """
    with get_session() as sess:
        rows = list(
            sess.execute(
                select(ChainRun).where(ChainRun.status.in_(_WATCHED_STATUSES))
            ).scalars().all()
        )
        sess.expunge_all()

    written = 0
    seen_ids: set[str] = set()
    for chain_run in rows:
        seen_ids.add(chain_run.chain_id)

        # Don't retroactively post for chains that already finished WITHOUT
        # ever having a live status message. Those completed (or failed,
        # or got stopped) before the chain status panel feature existed,
        # or before the watcher caught up — surfacing them now would be
        # noise, not signal. They stay in the audit trail (chain_runs
        # row, checkpoints) but never get a Discord embed.
        if (
            chain_run.status in ("completed", "failed", "stopped")
            and chain_run.status_message_id is None
        ):
            continue

        state = _load_state(chain_run.chain_id)
        snapshot: ChainStatusSnapshot = build_chain_status_snapshot(
            chain_run, state
        )
        embed = build_chain_status_embed(snapshot)
        sig = _signature(embed)

        prev_sig = last_signatures.get(chain_run.chain_id)
        if (
            prev_sig == sig
            and chain_run.status_message_id is not None
        ):
            # Nothing meaningful changed AND there's already a posted
            # message — skip the live-panel edit. Terminal notification
            # uses ChainRun.terminal_notified_at for de-dup, so it can't
            # be skipped by this short-circuit either way.
            continue

        # If the chain is terminal AND we've already done one edit at
        # this terminal signature, leave it alone forever. The signature
        # check above handles "one edit and stop" naturally because the
        # terminal embed's signature is stable.
        result = await _post_or_edit(
            chain_run=chain_run,
            embed=embed,
            run_status=snapshot["run_status"],
        )
        if result is None:
            continue
        last_signatures[chain_run.chain_id] = sig
        written += 1

        if is_terminal_run(snapshot):
            log.debug(
                "discord.chain_status.terminal_finalized",
                chain_id=chain_run.chain_id,
                status=snapshot["run_status"],
            )
            # Fire the one-shot rich terminal embed in the JOBS channel.
            # Two layered gates:
            #  - in-memory ``notified_terminal`` for fast same-process
            #    de-dup across consecutive ticks.
            #  - persistent ``ChainRun.terminal_notified_at`` so a daemon
            #    restart between observing the transition and posting
            #    doesn't double-fire on the next boot. The earlier
            #    "skip terminal-with-no-status_message_id" guard already
            #    ensures we never retroactively post for a chain we
            #    didn't track live (those stay silent forever).
            if (
                chain_run.chain_id not in notified_terminal
                and chain_run.terminal_notified_at is None
            ):
                try:
                    await notify.notify_chain_terminal(
                        chain_run, state, snapshot["run_status"],
                    )
                    log.info(
                        "discord.chain_status.terminal_notified",
                        chain_id=chain_run.chain_id[:8],
                        chain_name=chain_run.chain_name,
                        status=snapshot["run_status"],
                    )
                except Exception:
                    log.exception(
                        "discord.chain_status.terminal_notify_failed",
                        chain_id=chain_run.chain_id,
                    )
                # Mark notified regardless of post outcome so a transient
                # Discord error doesn't cause us to spam-retry every tick.
                notified_terminal.add(chain_run.chain_id)

            # Drop the cached status_message_id now that the panel won't
            # be edited again. Future ticks for this chain hit the
            # "skip terminal-with-no-status_message_id" guard above and
            # bail before doing any work.
            _clear_message_id(chain_run.id)

    # Drop signatures for chains that are no longer in the watch set —
    # keeps the in-memory map from growing unboundedly across a long
    # daemon run. Same cleanup for notified_terminal — once a chain has
    # rolled out of the watch set, we don't need its membership.
    stale = [k for k in last_signatures if k not in seen_ids]
    for k in stale:
        del last_signatures[k]
    stale_terminal = [k for k in notified_terminal if k not in seen_ids]
    for k in stale_terminal:
        notified_terminal.discard(k)
    return written


async def run_chain_status_watcher(
    *,
    stop: asyncio.Event | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_iterations: int | None = None,
) -> int:
    """Long-running loop that maintains live chain status embeds.

    Returns the number of ticks executed (used by tests). The watcher
    runs forever until ``stop`` is set. Each tick is silently skipped
    until the bot's ``on_ready`` has installed a poster client —
    otherwise the very first tick races the gateway connect and crashes
    on the unset client.
    """
    interval = max(poll_seconds, MIN_POLL_SECONDS)
    last_signatures: dict[str, str] = {}
    notified_terminal: set[str] = set()
    iters = 0
    while True:
        if stop is not None and stop.is_set():
            return iters
        if max_iterations is not None and iters >= max_iterations:
            return iters
        iters += 1
        if not poster.client_ready():
            # Bot hasn't connected yet — skip this tick instead of
            # spamming "client not configured" tracebacks.
            log.debug("discord.chain_status.waiting_for_poster")
        else:
            try:
                await _tick(
                    last_signatures=last_signatures,
                    notified_terminal=notified_terminal,
                )
            except Exception:
                log.exception("discord.chain_status.tick_failed")
        if max_iterations is not None and iters >= max_iterations:
            return iters
        if stop is not None and stop.is_set():
            return iters
        await asyncio.sleep(interval)


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "MIN_POLL_SECONDS",
    "run_chain_status_watcher",
]

"""Chain UI watcher.

Polls every ``poll_seconds`` for ``ChainRun`` rows in
``awaiting_approval`` / ``awaiting_user`` status. For each one it walks
the LangGraph checkpoint, finds approval steps that haven't been
posted yet, and posts an embed with Approve / Decline buttons in the
chains channel. The embed's message id is recorded in
``ChainState.awaiting_posts[step_id]``.

A button click handler invokes ``graph.invoke(Command(resume=value),
config)`` and edits the embed *once* to a static "resolved" form. That
is the ONLY allowed live edit in the whole Discord layer; it's
inherent to button interactions.

Approvals now post directly into the chains channel (no per-chain
thread). The only per-chain thread the daemon still creates is the one
:func:`tasque.discord.notify.notify_chain_terminal` anchors under JOBS
to hold the final report — that's the conversation-after-completion
thread, routed to the bucket coach.

This module also exposes :func:`build_retry_view` for the DLQ Retry +
Resolve buttons, and :func:`build_chain_control_view` for the
Pause / Resume / Stop buttons attached to the live status panel.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sqlalchemy import select

from tasque.chains.checkpointer import get_chain_checkpointer
from tasque.chains.spec import PlanNode
from tasque.discord import embeds, poster
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)

DEFAULT_POLL_SECONDS = 5.0


def _thread_config(chain_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": chain_id})


def _load_chain_state(chain_id: str) -> dict[str, Any] | None:
    saver = get_chain_checkpointer()
    snapshot = saver.get_tuple(_thread_config(chain_id))
    if snapshot is None:
        return None
    raw: dict[str, Any] = snapshot.checkpoint.get("channel_values", {})
    return raw


def _approval_steps_to_post(state: dict[str, Any]) -> list[PlanNode]:
    """Return approval-kind plan nodes whose step_id is NOT already in
    ``awaiting_posts``."""
    plan_raw = state.get("plan") or []
    plan: list[PlanNode] = list(plan_raw)
    posted: dict[str, str] = dict(state.get("awaiting_posts") or {})
    out: list[PlanNode] = []
    for n in plan:
        if n.get("kind") != "approval":
            continue
        if n.get("status") not in ("running", "awaiting_user"):
            continue
        if n["id"] in posted:
            continue
        out.append(n)
    return out


def _consumes_for_step(
    state: dict[str, Any], step: PlanNode
) -> dict[str, Any]:
    """Reconstruct the ``consumes_payload`` for a step from ``completed``."""
    completed = state.get("completed") or {}
    payload: dict[str, Any] = {}
    for dep in step.get("consumes") or []:
        cell = completed.get(dep)
        if cell is None:
            payload[dep] = {}
        else:
            payload[dep] = cell.get("produces", {}) if isinstance(cell, dict) else cell
    return payload


def _record_awaiting_post(chain_id: str, step_id: str, message_id: int) -> None:
    """Append ``{step_id: str(message_id)}`` to the chain's awaiting_posts.

    The state schema reducer merges this into the existing dict.
    """
    from tasque.chains.graph import get_compiled_chain_graph

    graph = get_compiled_chain_graph()
    cfg = _thread_config(chain_id)
    graph.update_state(cfg, {"awaiting_posts": {step_id: str(message_id)}})


def _set_chain_status(chain_id: str, status: str) -> None:
    from tasque.memory.entities import utc_now_iso

    with get_session() as sess:
        stmt = select(ChainRun).where(ChainRun.chain_id == chain_id)
        row = sess.execute(stmt).scalars().first()
        if row is None:
            return
        row.status = status
        row.updated_at = utc_now_iso()


# ----------------------------------------------------------------- post side


def _build_view_factory_default(
    chain_id: str, step_id: str
) -> Any:
    """Default View factory: returns None unless nextcord is installed
    AND the bot has wired up a real factory. Tests pass None.
    """
    try:
        return _make_approval_view(chain_id, step_id)
    except Exception:
        return None


def _make_approval_view(chain_id: str, step_id: str) -> Any:
    """Construct an Approve / Decline ``nextcord.ui.View``.

    Defined here rather than in bot.py so the chain UI watcher can
    rebuild views on restart (each ``View`` instance is short-lived;
    nextcord's persistent-view API is opt-in via ``custom_id``).
    """
    import nextcord
    from nextcord import ButtonStyle
    from nextcord.ui import Button, View

    class _ApprovalView(View):
        def __init__(self) -> None:
            super().__init__(timeout=None)
            approve_btn: Any = Button(
                label="Approve",
                style=ButtonStyle.success,
                custom_id=f"tasque-approve:{chain_id}:{step_id}",
            )
            decline_btn: Any = Button(
                label="Decline",
                style=ButtonStyle.danger,
                custom_id=f"tasque-decline:{chain_id}:{step_id}",
            )
            self.add_item(approve_btn)
            self.add_item(decline_btn)

    _ = nextcord  # keep import-time import for nicer error if missing
    return _ApprovalView()


def _make_retry_view(failed_job_id: str) -> Any:
    """Construct a DLQ ``nextcord.ui.View`` with Retry + Resolve buttons.

    Both buttons share the view because every DLQ embed needs both
    actions: Retry re-fires the job, Resolve closes the entry without
    re-firing (e.g. fixed manually).
    """
    import nextcord
    from nextcord import ButtonStyle
    from nextcord.ui import Button, View

    class _RetryView(View):
        def __init__(self) -> None:
            super().__init__(timeout=None)
            retry_btn: Any = Button(
                label="Retry",
                style=ButtonStyle.primary,
                custom_id=f"tasque-retry:{failed_job_id}",
            )
            resolve_btn: Any = Button(
                label="Resolve",
                style=ButtonStyle.secondary,
                custom_id=f"tasque-resolve:{failed_job_id}",
            )
            self.add_item(retry_btn)
            self.add_item(resolve_btn)

    _ = nextcord
    return _RetryView()


def build_approval_view(chain_id: str, step_id: str) -> Any:
    """Public factory for the approval-buttons View. Returns None if
    nextcord cannot be imported (tests / headless mode)."""
    try:
        return _make_approval_view(chain_id, step_id)
    except Exception:
        return None


def build_retry_view(failed_job_id: str) -> Any:
    """Public factory for the DLQ retry-button View."""
    try:
        return _make_retry_view(failed_job_id)
    except Exception:
        return None


# Chain run statuses where control buttons are meaningful. Terminal
# statuses (completed / failed / stopped) get no buttons — there's
# nothing to control.
_CONTROL_STATUSES: frozenset[str] = frozenset(
    {"running", "awaiting_approval", "awaiting_user", "paused"}
)


def _make_chain_control_view(chain_id: str, run_status: str) -> Any:
    """Construct a Pause / Resume / Stop ``nextcord.ui.View`` for the
    chain status panel.

    Buttons shown depend on ``run_status``:

    - ``running`` / ``awaiting_approval`` / ``awaiting_user`` → Pause + Stop
    - ``paused`` → Resume + Stop
    - terminal states → returns ``None`` (no view; status panel is final)
    """
    if run_status not in _CONTROL_STATUSES:
        return None

    import nextcord
    from nextcord import ButtonStyle
    from nextcord.ui import Button, View

    class _ChainControlView(View):
        def __init__(self) -> None:
            super().__init__(timeout=None)
            if run_status == "paused":
                resume_btn: Any = Button(
                    label="Resume",
                    style=ButtonStyle.success,
                    custom_id=f"tasque-chain-resume:{chain_id}",
                )
                self.add_item(resume_btn)
            else:
                pause_btn: Any = Button(
                    label="Pause",
                    style=ButtonStyle.secondary,
                    custom_id=f"tasque-chain-pause:{chain_id}",
                )
                self.add_item(pause_btn)
            stop_btn: Any = Button(
                label="Stop",
                style=ButtonStyle.danger,
                custom_id=f"tasque-chain-stop:{chain_id}",
            )
            self.add_item(stop_btn)

    _ = nextcord
    return _ChainControlView()


def build_chain_control_view(chain_id: str, run_status: str) -> Any:
    """Public factory for the chain-status control View. Returns None if
    nextcord can't be imported or the status doesn't warrant controls."""
    try:
        return _make_chain_control_view(chain_id, run_status)
    except Exception:
        return None


# ----------------------------------------------------------------- watcher


from collections.abc import Callable


def _chains_channel_id() -> int | None:
    """Lazy lookup of the chains channel — returns None if unset so the
    watcher can no-op cleanly during early startup or in tests that don't
    set the env var."""
    from tasque.discord.chain_status_watcher import chains_channel_id

    try:
        return chains_channel_id()
    except RuntimeError:
        return None


async def _scan_once(
    *,
    view_factory: Callable[[str, str], Any] | None = None,
    target_channel_id: int | None = None,
) -> int:
    """Walk all chains in awaiting_* status and post approval embeds for
    any steps that don't yet have one. Returns the number of embeds
    posted in this pass.

    All approval embeds land in the chains channel (no per-chain
    threads). ``target_channel_id`` overrides the env-derived channel
    for tests; production callers leave it None.
    """
    posted = 0
    with get_session() as sess:
        stmt = select(ChainRun).where(
            ChainRun.status.in_(
                ("awaiting_approval", "awaiting_user", "running")
            )
        )
        rows = list(sess.execute(stmt).scalars().all())
        sess.expunge_all()

    channel_id = (
        target_channel_id
        if target_channel_id is not None
        else _chains_channel_id()
    )
    if channel_id is None:
        # Chains channel unset — daemon misconfigured; skip silently and
        # let the env-validation guard surface the problem elsewhere.
        return 0

    for row in rows:
        state = _load_chain_state(row.chain_id)
        if state is None:
            continue
        steps = _approval_steps_to_post(state)
        if not steps:
            continue

        for step in steps:
            proposal = _consumes_for_step(state, step)
            embed = embeds.build_approval_embed(step, proposal, chain_run=row)
            view = (
                view_factory(row.chain_id, step["id"])
                if view_factory is not None
                else None
            )
            try:
                msg_id = await poster.post_embed(channel_id, embed, view=view)
            except Exception:
                log.exception(
                    "discord.chain_ui.post_failed",
                    chain_id=row.chain_id,
                    step_id=step["id"],
                )
                continue
            _record_awaiting_post(row.chain_id, step["id"], msg_id)
            _set_chain_status(row.chain_id, "awaiting_approval")
            posted += 1
    return posted


async def run_watcher(
    *,
    stop: asyncio.Event | None = None,
    view_factory: Callable[[str, str], Any] | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_iterations: int | None = None,
    target_channel_id: int | None = None,
) -> int:
    """Long-running chain UI watcher loop. Returns the total embeds posted.

    Tests pass ``max_iterations`` and ``target_channel_id`` to drive a
    single pass without polling and without touching the env var.
    """
    actual_view = view_factory if view_factory is not None else _build_view_factory_default

    iters = 0
    total = 0
    while True:
        if stop is not None and stop.is_set():
            return total
        if max_iterations is not None and iters >= max_iterations:
            return total
        iters += 1
        try:
            total += await _scan_once(
                view_factory=actual_view,
                target_channel_id=target_channel_id,
            )
        except Exception:
            log.exception("discord.chain_ui.scan_failed")
        if stop is not None and stop.is_set():
            return total
        await asyncio.sleep(poll_seconds)


# ----------------------------------------------------------------- resolver

async def resolve_approval(
    chain_id: str,
    step_id: str,
    resolution: str,
    *,
    posted_channel_id: int,
    posted_message_id: int,
) -> None:
    """Resume an interrupted chain with ``resolution`` and edit the
    embed to its resolved form.

    ``resolution`` is what the button click maps to — typically
    ``"approved"`` or ``"declined"`` for buttons, or freeform text from
    a modal. Whatever value is passed becomes the worker's
    ``user_reply`` produces field downstream.
    """
    from tasque.chains.graph import get_compiled_chain_graph
    from tasque.chains.scheduler import maybe_finalize_status

    graph = get_compiled_chain_graph()
    cfg = _thread_config(chain_id)
    try:
        graph.invoke(Command(resume=resolution), cfg)
    except Exception:
        log.exception(
            "discord.chain_ui.resume_failed",
            chain_id=chain_id,
            step_id=step_id,
        )
        raise
    maybe_finalize_status(chain_id)

    # Load the resolved step's snapshot so we can build the static embed.
    state = _load_chain_state(chain_id)
    chain_run: ChainRun | None = None
    with get_session() as sess:
        chain_run = sess.execute(
            select(ChainRun).where(ChainRun.chain_id == chain_id)
        ).scalars().first()
        if chain_run is not None:
            sess.expunge(chain_run)

    if state is None or chain_run is None:
        log.warning(
            "discord.chain_ui.no_state_after_resume",
            chain_id=chain_id,
            step_id=step_id,
        )
        return

    plan = cast(list[PlanNode], state.get("plan") or [])
    step = next((n for n in plan if n["id"] == step_id), None)
    if step is None:
        log.warning(
            "discord.chain_ui.step_missing",
            chain_id=chain_id,
            step_id=step_id,
        )
        return

    embed = embeds.build_resolved_approval_embed(
        step, resolution, chain_run=chain_run
    )
    try:
        await poster.edit_message(
            posted_channel_id, posted_message_id, embed=embed, view=None
        )
    except Exception:
        log.exception(
            "discord.chain_ui.edit_failed",
            chain_id=chain_id,
            step_id=step_id,
            channel_id=posted_channel_id,
            message_id=posted_message_id,
        )


async def handle_dlq_retry(failed_job_id: str) -> dict[str, Any]:
    """Run :func:`tasque.jobs.dlq.retry` in a worker thread (it touches
    SQLite synchronously). Used by the DLQ Retry button handler."""
    from tasque.jobs.dlq import retry as _retry

    return await asyncio.to_thread(_retry, failed_job_id)


__all__ = [
    "build_approval_view",
    "build_chain_control_view",
    "build_retry_view",
    "handle_dlq_retry",
    "resolve_approval",
    "run_watcher",
]

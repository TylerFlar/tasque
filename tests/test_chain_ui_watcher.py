"""Tests for the chain UI watcher: post embeds for awaiting approvals,
relay button clicks back into the chain via Command(resume=...), and
edit the embed once on resolution."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from tasque.chains.checkpointer import get_chain_checkpointer
from tasque.chains.scheduler import launch_chain_run
from tasque.discord import chain_ui, poster, threads
from tasque.discord.chain_ui import build_chain_control_view
from tasque.jobs.runner import WorkerResult
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun


@pytest.fixture(autouse=True)
def reset_threads_and_poster(tmp_path: Any) -> Any:
    import os

    registry = tmp_path / "discord_threads.json"
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
    def __init__(self) -> None:
        self.embeds_posted: list[tuple[int, dict[str, Any], Any]] = []
        self.messages_posted: list[tuple[int, str]] = []
        self.edits: list[tuple[int, int, dict[str, Any] | None, Any]] = []

    async def send_message(self, channel_id: int, content: str) -> int:
        self.messages_posted.append((channel_id, content))
        return 1000 + len(self.messages_posted)

    async def send_embed(
        self, channel_id: int, embed: dict[str, Any], view: Any | None = None
    ) -> int:
        self.embeds_posted.append((channel_id, embed, view))
        return 9000 + len(self.embeds_posted)

    async def edit_message(
        self,
        channel_id: int,
        message_id: int,
        *,
        content: str | None = None,
        embed: dict[str, Any] | None = None,
        view: Any | None = None,
    ) -> None:
        self.edits.append((channel_id, message_id, embed, view))

    async def upload_file(
        self, channel_id: int, path: Any, *, content: str | None = None
    ) -> int:
        return 0

    async def fetch_recent_messages(self, channel_id: int, limit: int) -> list[Any]:
        return []

    async def start_thread(
        self, channel_id: int, message_id: int, name: str
    ) -> int:
        return 9999


def _spec() -> dict[str, Any]:
    return {
        "chain_name": "approval-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {"id": "scan", "kind": "worker", "directive": "scan", "tier": "small"},
            {
                "id": "decide",
                "kind": "approval",
                "directive": "should we proceed",
                "depends_on": ["scan"],
                "consumes": ["scan"],
            },
        ],
    }


@pytest.fixture
def fake_worker(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    record: dict[str, Any] = {"calls": []}

    def _fake(job: Any, *, consumes: dict[str, Any] | None = None, **_: Any) -> WorkerResult:
        record["calls"].append(job.chain_step_id)
        if job.chain_step_id == "scan":
            return WorkerResult(
                report="ok", summary="scan ok", produces={"items": ["a", "b"]}, error=None
            )
        return WorkerResult(report="", summary="", produces={}, error=None)

    import tasque.chains.graph.worker as worker_mod

    monkeypatch.setattr(worker_mod, "run_worker", _fake)
    return record


@pytest.mark.asyncio
async def test_watcher_posts_embed_for_awaiting_approval(fake_worker: dict[str, Any]) -> None:
    chain_id = launch_chain_run(_spec())
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    n = await chain_ui.run_watcher(
        target_channel_id=7777,
        view_factory=lambda _c, _s: None,
        max_iterations=1,
        poll_seconds=0,
    )
    assert n == 1, "expected one embed posted for the pending approval step"
    assert fake.embeds_posted, "expected the watcher to post an embed"
    channel_id, embed, _ = fake.embeds_posted[0]
    assert channel_id == 7777
    assert embed["title"] == "Approval needed"

    # ChainRun.status should now be awaiting_approval.
    with get_session() as sess:
        run = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert run is not None
        assert run["status"] == "awaiting_approval"

    # And the post id should be recorded in the chain checkpoint state.
    state = _load_state(chain_id)
    assert state is not None
    assert state.get("awaiting_posts", {}).get("decide") is not None


@pytest.mark.asyncio
async def test_watcher_does_not_repost_after_recording(fake_worker: dict[str, Any]) -> None:
    launch_chain_run(_spec())
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    async def _ensure_thread(_run: ChainRun) -> int:
        return 7777

    await chain_ui.run_watcher(
        target_channel_id=7777,
        view_factory=lambda _c, _s: None,
        max_iterations=1,
        poll_seconds=0,
    )
    posted_first = len(fake.embeds_posted)

    # Second pass: should NOT post again — the awaiting_posts entry is
    # already there.
    await chain_ui.run_watcher(
        target_channel_id=7777,
        view_factory=lambda _c, _s: None,
        max_iterations=1,
        poll_seconds=0,
    )
    assert len(fake.embeds_posted) == posted_first


@pytest.mark.asyncio
async def test_resolve_approval_resumes_chain_and_edits_embed(
    fake_worker: dict[str, Any],
) -> None:
    chain_id = launch_chain_run(_spec())
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    async def _ensure_thread(_run: ChainRun) -> int:
        return 7777

    await chain_ui.run_watcher(
        target_channel_id=7777,
        view_factory=lambda _c, _s: None,
        max_iterations=1,
        poll_seconds=0,
    )

    posted_msg_id = 9001
    await chain_ui.resolve_approval(
        chain_id,
        "decide",
        "approved",
        posted_channel_id=7777,
        posted_message_id=posted_msg_id,
    )

    # Embed editing was the watcher's resolve step.
    assert fake.edits, "expected one edit on resolution"
    edit_channel, edit_msg_id, edit_embed, edit_view = fake.edits[0]
    assert edit_channel == 7777
    assert edit_msg_id == posted_msg_id
    assert edit_embed is not None
    assert edit_embed["title"] == "Approval resolved"
    assert edit_view is None

    # Chain should now be completed and the produces should reflect the
    # user's reply.
    state = _load_state(chain_id)
    assert state is not None
    decide = next(n for n in state["plan"] if n["id"] == "decide")
    assert decide["status"] == "completed"
    assert state["completed"]["decide"]["produces"]["user_reply"] == "approved"


# ----------------------------------------------------------------- control view


def _custom_ids(view: Any) -> set[str]:
    """Pull the ``custom_id`` of every Button child on a view."""
    return {
        getattr(c, "custom_id", None)
        for c in getattr(view, "children", [])
        if getattr(c, "custom_id", None) is not None
    }


@pytest.mark.asyncio
async def test_chain_control_view_running_has_pause_and_stop() -> None:
    # nextcord.ui.View constructor needs a running event loop, so the
    # view-construction tests live inside @pytest.mark.asyncio.
    view = build_chain_control_view("abc123", "running")
    assert view is not None
    ids = _custom_ids(view)
    assert ids == {"tasque-chain-pause:abc123", "tasque-chain-stop:abc123"}


@pytest.mark.asyncio
async def test_chain_control_view_paused_has_resume_and_stop() -> None:
    view = build_chain_control_view("abc123", "paused")
    assert view is not None
    ids = _custom_ids(view)
    assert ids == {"tasque-chain-resume:abc123", "tasque-chain-stop:abc123"}


@pytest.mark.asyncio
async def test_chain_control_view_awaiting_states_get_pause_and_stop() -> None:
    for status in ("awaiting_approval", "awaiting_user"):
        view = build_chain_control_view("abc123", status)
        assert view is not None, f"expected control view for {status}"
        ids = _custom_ids(view)
        assert "tasque-chain-pause:abc123" in ids
        assert "tasque-chain-stop:abc123" in ids


def test_chain_control_view_terminal_returns_none() -> None:
    # No event loop needed — terminal statuses short-circuit before
    # touching nextcord.
    for status in ("completed", "failed", "stopped"):
        assert build_chain_control_view("abc123", status) is None, status


def _load_state(chain_id: str) -> dict[str, Any] | None:
    saver = get_chain_checkpointer()
    snapshot = saver.get_tuple(RunnableConfig(configurable={"thread_id": chain_id}))
    if snapshot is None:
        return None
    raw: dict[str, Any] = snapshot.checkpoint.get("channel_values", {})
    return raw


_ = SqliteSaver  # touched to keep the import in case the saver type is needed

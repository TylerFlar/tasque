"""Tests for the live-edited ops panel watcher."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tasque.discord import ops_watcher, poster, threads
from tasque.memory.entities import QueuedJob
from tasque.memory.repo import write_entity


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path: Path) -> Iterator[Path]:
    registry = tmp_path / "discord_threads.json"
    old = os.environ.get("TASQUE_DISCORD_THREAD_REGISTRY")
    os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = registry.as_posix()
    threads.reset_cache()
    poster.set_client(None)
    try:
        yield registry
    finally:
        threads.reset_cache()
        poster.set_client(None)
        if old is None:
            os.environ.pop("TASQUE_DISCORD_THREAD_REGISTRY", None)
        else:
            os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = old


class _FakePoster:
    """Minimal poster client that records edit/post traffic."""

    def __init__(self, *, edit_should_raise: bool = False) -> None:
        self.posted: list[tuple[int, dict[str, Any], Any]] = []
        self.edited: list[tuple[int, int, dict[str, Any] | None, Any]] = []
        self.edit_should_raise = edit_should_raise

    async def send_message(self, channel_id: int, content: str) -> int:
        return 0

    async def send_embed(
        self, channel_id: int, embed: dict[str, Any], view: Any | None = None
    ) -> int:
        self.posted.append((channel_id, embed, view))
        return 7000 + len(self.posted)

    async def edit_message(
        self,
        channel_id: int,
        message_id: int,
        *,
        content: str | None = None,
        embed: dict[str, Any] | None = None,
        view: Any | None = None,
    ) -> None:
        if self.edit_should_raise:
            raise RuntimeError("simulated edit failure (message deleted)")
        self.edited.append((channel_id, message_id, embed, view))

    async def upload_file(self, *args: Any, **kwargs: Any) -> int:
        return 0

    async def fetch_recent_messages(self, channel_id: int, limit: int) -> list[Any]:
        return []

    async def start_thread(self, channel_id: int, message_id: int, name: str) -> int:
        return 0


# ----------------------------------------------------------------- behaviour


@pytest.mark.asyncio
async def test_first_tick_posts_new_message_and_caches_id() -> None:
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    iters = await ops_watcher.run_ops_watcher(
        channel_id=1234, max_iterations=1, poll_seconds=0.0
    )
    assert iters == 1
    assert len(fake.posted) == 1
    assert fake.posted[0][0] == 1234
    assert fake.edited == []
    # Message id is cached for the next tick.
    assert threads.get_thread_id(threads.PURPOSE_OPS_PANEL) == 7001


@pytest.mark.asyncio
async def test_subsequent_tick_edits_cached_message() -> None:
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    # First tick → post.
    await ops_watcher.run_ops_watcher(
        channel_id=1234, max_iterations=1, poll_seconds=0.0
    )
    # Make the snapshot change so the dedup doesn't suppress the edit.
    write_entity(QueuedJob(
        kind="worker", bucket="health", directive="d",
        reason="", fire_at="now", status="pending", queued_by="cli",
    ))

    # Second tick → edit.
    await ops_watcher.run_ops_watcher(
        channel_id=1234, max_iterations=1, poll_seconds=0.0
    )
    assert len(fake.posted) == 1, "should not post again on second tick"
    assert len(fake.edited) == 1
    edit_channel, edit_msg, edit_embed, _ = fake.edited[0]
    assert edit_channel == 1234
    assert edit_msg == 7001
    assert edit_embed is not None
    assert edit_embed["title"] == "tasque ops panel"


@pytest.mark.asyncio
async def test_failed_edit_falls_back_to_fresh_post() -> None:
    """Simulate the cached message having been deleted: the edit raises,
    the watcher posts a new message and updates the cached id."""
    fake = _FakePoster(edit_should_raise=True)
    poster.set_client(fake)  # type: ignore[arg-type]
    # Pretend a previous run left a stale message id behind.
    threads.set_thread_id(threads.PURPOSE_OPS_PANEL, 999_999)

    await ops_watcher.run_ops_watcher(
        channel_id=4242, max_iterations=1, poll_seconds=0.0
    )

    # We tried to edit the stale id once...
    assert len(fake.edited) == 0, "edit raised, so it shouldn't be recorded"
    # ...and recovered by posting a fresh message.
    assert len(fake.posted) == 1
    new_id = fake.posted[0][0]
    assert new_id == 4242
    cached = threads.get_thread_id(threads.PURPOSE_OPS_PANEL)
    assert cached == 7001
    assert cached != 999_999, "cached id should have been overwritten"


@pytest.mark.asyncio
async def test_tick_skipped_when_payload_unchanged() -> None:
    """Two ticks within one long-running call: with no DB changes
    between them, the second tick is suppressed by the dedup. Keeps us
    under Discord's per-channel edit rate limit."""
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    iters = await ops_watcher.run_ops_watcher(
        channel_id=1234, max_iterations=2, poll_seconds=0.0
    )
    assert iters == 2
    assert len(fake.posted) == 1, "only the first tick should post"
    assert len(fake.edited) == 0, "second tick should be deduped (no changes)"


@pytest.mark.asyncio
async def test_no_channel_configured_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ops channel id is required — the watcher must raise loudly
    rather than silently no-op when nothing is configured."""
    monkeypatch.delenv("TASQUE_DISCORD_OPS_CHANNEL_ID", raising=False)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="TASQUE_DISCORD_OPS_CHANNEL_ID"):
        await ops_watcher.run_ops_watcher(
            channel_id=None, max_iterations=5, poll_seconds=0.0
        )
    assert fake.posted == []
    assert fake.edited == []


@pytest.mark.asyncio
async def test_stop_event_breaks_loop() -> None:
    import asyncio

    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]
    stop = asyncio.Event()
    stop.set()  # already set; loop must exit before any tick
    iters = await ops_watcher.run_ops_watcher(
        channel_id=1234, stop=stop, poll_seconds=0.0
    )
    assert iters == 0
    assert fake.posted == []


def test_ops_channel_id_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASQUE_DISCORD_OPS_CHANNEL_ID", "987654321")
    assert ops_watcher.ops_channel_id() == 987654321


def test_ops_channel_id_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASQUE_DISCORD_OPS_CHANNEL_ID", raising=False)
    with pytest.raises(RuntimeError, match="TASQUE_DISCORD_OPS_CHANNEL_ID"):
        ops_watcher.ops_channel_id()


def test_ops_channel_id_raises_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASQUE_DISCORD_OPS_CHANNEL_ID", "garbage")
    with pytest.raises(RuntimeError, match="not an integer"):
        ops_watcher.ops_channel_id()

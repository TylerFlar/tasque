"""Tests for ``tasque.discord.dlq_watcher``.

Covers:
- unresolved + un-notified FailedJob row picked up + notify fired with the Retry view
- successful post stamps ``FailedJob.notified_at`` so subsequent ticks don't repeat
- already-resolved rows are filtered out
- rows older than the lookback window are skipped
- watcher self-defers when poster client is not ready
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tasque.discord import dlq_watcher, poster, threads
from tasque.memory.db import get_session
from tasque.memory.entities import FailedJob
from tasque.memory.repo import write_entity


@pytest.fixture(autouse=True)
def reset_threads_and_poster(tmp_path: Any) -> Any:
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
        self.embeds: list[tuple[int, dict[str, Any], Any]] = []
        self.next_message_id = 5000

    async def send_message(self, channel_id: int, content: str) -> int:
        return 1000

    async def send_embed(
        self, channel_id: int, embed: dict[str, Any], view: Any | None = None
    ) -> int:
        self.next_message_id += 1
        self.embeds.append((channel_id, embed, view))
        return self.next_message_id

    async def edit_message(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def upload_file(self, *args: Any, **kwargs: Any) -> int:
        return 0

    async def fetch_recent_messages(self, channel_id: int, limit: int) -> list[Any]:
        return []

    async def start_thread(
        self, channel_id: int, message_id: int, name: str
    ) -> int:
        return 8000


def _make_failure(**overrides: Any) -> FailedJob:
    """Persist a FailedJob row in the state the scheduler leaves after
    ``record_failure`` — unresolved, un-notified."""
    defaults: dict[str, Any] = dict(
        job_id="job-1",
        agent_kind="worker",
        bucket="finance",
        failure_timestamp="2026-04-26T00:00:00.000000Z",
        error_type="APIConnectionError",
        error_message="Connection error.",
        traceback="",
        retry_count=0,
        original_trigger="scheduler",
        resolved=False,
        notified_at=None,
    )
    defaults.update(overrides)
    return write_entity(FailedJob(**defaults))


# ----------------------------------------------------------------- watcher

@pytest.mark.asyncio
async def test_watcher_posts_dlq_embed_with_retry_view_and_stamps_notified_at() -> None:
    threads.set_thread_id(threads.PURPOSE_DLQ, 7777)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    fj = _make_failure()

    await dlq_watcher.run_dlq_watcher(max_iterations=1, poll_seconds=0)

    # One DLQ embed posted in the DLQ channel with a non-None view.
    assert len(fake.embeds) == 1
    channel_id, embed, view = fake.embeds[0]
    assert channel_id == 7777
    assert "APIConnectionError" in embed["title"]
    assert view is not None  # The Retry + Resolve View

    # notified_at stamped — a second tick must not re-fire.
    with get_session() as sess:
        row = sess.get(FailedJob, fj.id)
        assert row is not None
        assert row.notified_at is not None

    embeds_before = len(fake.embeds)
    await dlq_watcher.run_dlq_watcher(max_iterations=1, poll_seconds=0)
    assert len(fake.embeds) == embeds_before


@pytest.mark.asyncio
async def test_watcher_skips_resolved_failures() -> None:
    threads.set_thread_id(threads.PURPOSE_DLQ, 7777)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    _make_failure(resolved=True)

    await dlq_watcher.run_dlq_watcher(max_iterations=1, poll_seconds=0)
    assert fake.embeds == []


@pytest.mark.asyncio
async def test_watcher_skips_already_notified_rows() -> None:
    threads.set_thread_id(threads.PURPOSE_DLQ, 7777)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    _make_failure(notified_at="2026-04-26T19:00:00.000000Z")

    await dlq_watcher.run_dlq_watcher(max_iterations=1, poll_seconds=0)
    assert fake.embeds == []


@pytest.mark.asyncio
async def test_watcher_skips_rows_older_than_lookback() -> None:
    """A FailedJob older than the lookback window stays silent — the
    operator missed announcing it during downtime; flooding DLQ on
    boot is worse than swallowing it."""
    threads.set_thread_id(threads.PURPOSE_DLQ, 7777)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    old = (
        datetime.now(UTC)
        - timedelta(seconds=dlq_watcher.NOTIFY_LOOKBACK_SECONDS + 60)
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    _make_failure(created_at=old, failure_timestamp=old)

    await dlq_watcher.run_dlq_watcher(max_iterations=1, poll_seconds=0)
    assert fake.embeds == []


@pytest.mark.asyncio
async def test_watcher_self_defers_until_poster_ready() -> None:
    """No client installed → no post attempted, but the loop keeps
    ticking so the watcher can pick up later once on_ready fires."""
    threads.set_thread_id(threads.PURPOSE_DLQ, 7777)
    poster.set_client(None)

    _make_failure()

    iters = await dlq_watcher.run_dlq_watcher(max_iterations=2, poll_seconds=0)
    assert iters == 2
    # No post attempted — and notified_at remained None.
    with get_session() as sess:
        rows = sess.query(FailedJob).all()
        assert all(r.notified_at is None for r in rows)


@pytest.mark.asyncio
async def test_watcher_keeps_running_when_a_post_fails() -> None:
    """A poster exception on one row shouldn't strand subsequent
    failures — the watcher logs and moves on."""
    threads.set_thread_id(threads.PURPOSE_DLQ, 7777)

    class _BrokenThenWorkingPoster(_FakePoster):
        def __init__(self) -> None:
            super().__init__()
            self._call = 0

        async def send_embed(
            self, channel_id: int, embed: dict[str, Any], view: Any | None = None
        ) -> int:
            self._call += 1
            if self._call == 1:
                raise RuntimeError("simulated discord 5xx")
            return await super().send_embed(channel_id, embed, view)

    fake = _BrokenThenWorkingPoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    fj_first = _make_failure(error_type="FirstError")
    fj_second = _make_failure(error_type="SecondError")

    await dlq_watcher.run_dlq_watcher(max_iterations=1, poll_seconds=0)
    # Second failure posted successfully.
    titles = [embed["title"] for _, embed, _ in fake.embeds]
    assert any("SecondError" in t for t in titles)

    with get_session() as sess:
        first = sess.get(FailedJob, fj_first.id)
        second = sess.get(FailedJob, fj_second.id)
        assert first is not None
        assert second is not None
        # First row never stamped (post raised before notify_failed_job's
        # stamp call); second row stamped because its post succeeded.
        assert first.notified_at is None
        assert second.notified_at is not None

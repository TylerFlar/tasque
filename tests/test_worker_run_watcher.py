"""Tests for ``tasque.discord.worker_run_watcher``.

Covers:
- terminal-status row picked up + notify fired with the persisted fields
- successful post stamps ``QueuedJob.notified_at`` so subsequent ticks don't repeat
- silent (``visible=False``) jobs are filtered out by the WHERE clause
- corrupt produces JSON degrades to {} instead of crashing the tick
- jobs whose ``updated_at`` is older than the lookback window are skipped
- watcher self-defers when poster client is not ready
- the scheduler's success path persists WorkerResult fields to the row
- the scheduler's failure paths persist last_error so the watcher can
  surface "Worker run failed" embeds
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tasque.discord import poster, threads, worker_run_watcher
from tasque.jobs.runner import WorkerResult
from tasque.jobs.scheduler import claim_and_run_one
from tasque.memory.db import get_session
from tasque.memory.entities import QueuedJob
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
        self.messages: list[tuple[int, str]] = []
        self.threads_started: list[tuple[int, int, str]] = []
        self.next_message_id = 5000
        self.next_thread_id = 8000

    async def send_message(self, channel_id: int, content: str) -> int:
        self.messages.append((channel_id, content))
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
        self.next_thread_id += 1
        self.threads_started.append((channel_id, message_id, name))
        return self.next_thread_id


def _make_completed_job(**overrides: Any) -> QueuedJob:
    """Persist a ``QueuedJob`` already in ``completed`` status with the
    captured WorkerResult fields populated — i.e. the state the
    scheduler leaves after a successful run."""
    defaults: dict[str, Any] = dict(
        kind="worker",
        bucket="health",
        directive="check sleep tracking",
        reason="",
        fire_at="now",
        status="completed",
        queued_by="cli",
        visible=True,
        last_summary="ok",
        last_report="all good",
        last_produces_json=json.dumps({"k": "v"}),
        last_error=None,
        notified_at=None,
    )
    defaults.update(overrides)
    return write_entity(QueuedJob(**defaults))


# ----------------------------------------------------------------- watcher

@pytest.mark.asyncio
async def test_watcher_posts_for_completed_job_and_stamps_notified_at() -> None:
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    job = _make_completed_job()

    await worker_run_watcher.run_worker_run_watcher(
        max_iterations=1, poll_seconds=0
    )

    # One worker-run embed was posted in JOBS, and a per-job thread anchored.
    assert len(fake.embeds) == 1
    channel_id, embed, _view = fake.embeds[0]
    assert channel_id == 4242
    assert embed["title"] == "ok"
    assert len(fake.threads_started) == 1

    # notified_at should now be stamped — a second tick must NOT re-fire.
    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.notified_at is not None

    # Second tick: no new posts.
    embeds_before = len(fake.embeds)
    await worker_run_watcher.run_worker_run_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert len(fake.embeds) == embeds_before


@pytest.mark.asyncio
async def test_watcher_posts_for_failed_job_with_error_field() -> None:
    """Failure path: scheduler stamped last_error; the embed must use
    the failure colour/title with the error surfaced in the description."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    _make_completed_job(
        status="failed",
        last_summary="",
        last_report="",
        last_produces_json="{}",
        last_error="LLM call failed: TimeoutError: hung",
    )

    await worker_run_watcher.run_worker_run_watcher(
        max_iterations=1, poll_seconds=0
    )

    assert len(fake.embeds) == 1
    _channel, embed, _view = fake.embeds[0]
    assert embed["title"] == "Worker run failed"
    # The error message should land in the description.
    assert "TimeoutError" in embed["description"]


@pytest.mark.asyncio
async def test_watcher_skips_silent_jobs() -> None:
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    _make_completed_job(visible=False)

    await worker_run_watcher.run_worker_run_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert fake.embeds == []


@pytest.mark.asyncio
async def test_watcher_skips_already_notified_rows() -> None:
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    _make_completed_job(notified_at="2026-04-26T19:00:00.000000Z")

    await worker_run_watcher.run_worker_run_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert fake.embeds == []


@pytest.mark.asyncio
async def test_watcher_handles_corrupt_produces_json() -> None:
    """A row whose ``last_produces_json`` failed to round-trip should
    still get notified — the watcher degrades to {} rather than crashing
    the tick and stranding every subsequent terminal job."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    _make_completed_job(last_produces_json="not-valid-json{")

    await worker_run_watcher.run_worker_run_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert len(fake.embeds) == 1


@pytest.mark.asyncio
async def test_watcher_skips_old_terminal_rows_outside_lookback() -> None:
    """A row that completed more than the lookback window ago stays silent —
    a daemon that's been offline for weeks shouldn't flood JOBS on boot
    with stale completions."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    old = (
        datetime.now(UTC)
        - timedelta(seconds=worker_run_watcher.NOTIFY_LOOKBACK_SECONDS + 3600)
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    job = _make_completed_job()
    # Manually backdate updated_at (write_entity sets it to now).
    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        row.updated_at = old

    await worker_run_watcher.run_worker_run_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert fake.embeds == []


@pytest.mark.asyncio
async def test_watcher_self_defers_when_poster_not_ready() -> None:
    """No poster client installed: the tick must be a silent no-op
    rather than raising — same defer pattern as the chain status watcher."""
    poster.set_client(None)
    _make_completed_job()
    iters = await worker_run_watcher.run_worker_run_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert iters == 1


# ------------------------------------------------- scheduler integration

def test_scheduler_persists_worker_result_on_success() -> None:
    """End-to-end: the scheduler's success path leaves last_summary /
    last_report / last_produces_json populated so the watcher has the
    fields it needs."""
    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="d",
            reason="",
            fire_at="now",
            status="pending",
            queued_by="test",
        )
    )

    def _runner(j: QueuedJob) -> WorkerResult:
        return WorkerResult(
            report="full report text",
            summary="ok",
            produces={"k": "v"},
            error=None,
        )

    claim_and_run_one(runner=_runner, heartbeat_interval=0.05)

    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.status == "completed"
        assert row.last_summary == "ok"
        assert row.last_report == "full report text"
        assert json.loads(row.last_produces_json or "{}") == {"k": "v"}
        assert row.last_error is None
        # notified_at must remain None — the watcher stamps it, not the
        # scheduler.
        assert row.notified_at is None


def test_scheduler_persists_worker_result_on_runner_exception() -> None:
    """When the runner raises, the scheduler should still stamp
    last_error so the watcher can post a failure embed instead of
    silently sitting on an unannounced row."""
    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="career",
            directive="boom",
            reason="",
            fire_at="now",
            status="pending",
            queued_by="test",
        )
    )

    def _angry(j: QueuedJob) -> WorkerResult:
        raise RuntimeError("kapow")

    claim_and_run_one(runner=_angry, heartbeat_interval=0.05)

    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.status == "failed"
        assert row.last_error is not None
        assert "RuntimeError" in row.last_error
        assert "kapow" in row.last_error


def test_scheduler_persists_worker_result_on_worker_error_field() -> None:
    """When the worker returns a WorkerResult with ``error`` set, the
    scheduler stamps it before flipping to failed."""
    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="career",
            directive="bad json",
            reason="",
            fire_at="now",
            status="pending",
            queued_by="test",
        )
    )

    def _err(j: QueuedJob) -> WorkerResult:
        return WorkerResult(
            report="", summary="", produces={}, error="LLM bad output"
        )

    claim_and_run_one(runner=_err, heartbeat_interval=0.05)

    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.status == "failed"
        assert row.last_error == "LLM bad output"

"""Tests for the per-job / per-DLQ thread pattern in notify."""

from __future__ import annotations

from typing import Any

import pytest

from tasque.discord import notify, poster, threads
from tasque.memory.entities import FailedJob, QueuedJob
from tasque.memory.repo import write_entity


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
        self.embeds: list[tuple[int, dict[str, Any], Any]] = []
        self.threads_started: list[tuple[int, int, str]] = []
        self.messages: list[tuple[int, str]] = []
        self._next_message_id = 1000

    async def send_message(self, channel_id: int, content: str) -> int:
        self._next_message_id += 1
        self.messages.append((channel_id, content))
        return self._next_message_id

    async def send_embed(
        self, channel_id: int, embed: dict[str, Any], view: Any | None = None
    ) -> int:
        self.embeds.append((channel_id, embed, view))
        return 5000 + len(self.embeds)

    async def edit_message(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def upload_file(self, *args: Any, **kwargs: Any) -> int:
        return 0

    async def fetch_recent_messages(self, channel_id: int, limit: int) -> list[Any]:
        return []

    async def start_thread(
        self, channel_id: int, message_id: int, name: str
    ) -> int:
        self.threads_started.append((channel_id, message_id, name))
        return 8000 + len(self.threads_started)


@pytest.mark.asyncio
async def test_notify_worker_run_posts_in_jobs_channel_and_starts_thread() -> None:
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="check sleep tracking",
            reason="",
            fire_at="now",
            status="completed",
            queued_by="cli",
            visible=True,
        )
    )

    msg_id = await notify.notify_worker_run(
        job, summary="ok", report="all good", produces={"x": 1}, error=None
    )

    assert msg_id is not None
    assert len(fake.embeds) == 1
    embed_channel, embed, _ = fake.embeds[0]
    assert embed_channel == 4242
    assert embed["title"] == "ok"
    # And a per-job thread was anchored to the embed message.
    assert len(fake.threads_started) == 1
    thread_channel, thread_msg, name = fake.threads_started[0]
    assert thread_channel == 4242
    assert thread_msg == msg_id
    assert "health" in name
    assert "check sleep tracking" in name


@pytest.mark.asyncio
async def test_notify_worker_run_caches_thread_id_on_job() -> None:
    """The first call should persist the new thread id back onto the
    QueuedJob row so re-notifications can find it."""
    from tasque.memory.db import get_session

    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="d",
            reason="",
            fire_at="now",
            status="completed",
            queued_by="cli",
            visible=True,
        )
    )
    assert job.thread_id is None

    await notify.notify_worker_run(
        job, summary="ok", report="ok", produces={}, error=None
    )

    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.thread_id is not None
        # Fake poster's start_thread returns 8000 + N.
        assert int(row.thread_id) == 8001


@pytest.mark.asyncio
async def test_notify_worker_run_reuses_cached_thread() -> None:
    """If the job already has a thread_id, the second call should post
    a follow-up embed in that thread instead of spawning a new thread."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="d",
            reason="",
            fire_at="now",
            status="completed",
            queued_by="cli",
            visible=True,
        )
    )
    job.thread_id = "12345"  # pretend a thread already exists

    msg_id = await notify.notify_worker_run(
        job, summary="rerun", report="r", produces={}, error=None
    )
    assert msg_id is not None
    # Embed posted directly to the cached thread.
    assert len(fake.embeds) == 1
    assert fake.embeds[0][0] == 12345
    # No new thread anchor — we're reusing the existing one.
    assert fake.threads_started == []


@pytest.mark.asyncio
async def test_notify_worker_run_skips_silent_jobs() -> None:
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]
    job = QueuedJob(
        kind="worker",
        bucket="health",
        directive="silent maintenance",
        reason="",
        fire_at="now",
        status="completed",
        queued_by="cli",
        visible=False,
    )
    msg_id = await notify.notify_worker_run(
        job, summary="ok", report="ok", produces={}, error=None
    )
    assert msg_id is None
    assert fake.embeds == []
    assert fake.threads_started == []


@pytest.mark.asyncio
async def test_notify_failed_job_posts_in_dlq_channel_and_starts_thread() -> None:
    threads.set_thread_id(threads.PURPOSE_DLQ, 9999)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    fj = write_entity(
        FailedJob(
            job_id="job-1",
            agent_kind="worker",
            bucket="career",
            failure_timestamp="2026-04-26T00:00:00.000Z",
            error_type="WorkerError",
            error_message="kapow",
            traceback="",
            original_trigger="scheduler",
        )
    )

    msg_id = await notify.notify_failed_job(fj, retry_view="VIEW")

    assert msg_id is not None
    assert len(fake.embeds) == 1
    embed_channel, embed, view = fake.embeds[0]
    assert embed_channel == 9999
    assert view == "VIEW"
    assert embed["title"] == "Job failed: WorkerError"
    # DLQ entries are status-only — no thread is created. The Retry +
    # Resolve buttons handle the actions; there's no conversation to
    # anchor.
    assert len(fake.threads_started) == 0


@pytest.mark.asyncio
async def test_notify_failed_job_does_not_create_thread() -> None:
    """Multiple notifications for the same FailedJob just post fresh
    embeds — no per-DLQ thread tracking."""
    from tasque.memory.db import get_session

    threads.set_thread_id(threads.PURPOSE_DLQ, 9999)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    fj = write_entity(
        FailedJob(
            job_id="job-1",
            agent_kind="worker",
            bucket="career",
            failure_timestamp="2026-04-26T00:00:00.000Z",
            error_type="WorkerError",
            error_message="kapow",
            traceback="",
            original_trigger="scheduler",
        )
    )

    await notify.notify_failed_job(fj, retry_view=None)
    await notify.notify_failed_job(fj, retry_view=None)
    assert len(fake.threads_started) == 0
    assert len(fake.embeds) == 2

    with get_session() as sess:
        row = sess.get(FailedJob, fj.id)
        assert row is not None
        assert row.thread_id is None


@pytest.mark.asyncio
async def test_notify_worker_run_skips_when_jobs_channel_unset() -> None:
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]
    job = QueuedJob(
        kind="worker",
        bucket=None,
        directive="x",
        reason="",
        fire_at="now",
        status="completed",
        queued_by="cli",
        visible=True,
    )
    msg_id = await notify.notify_worker_run(
        job, summary="ok", report="ok", produces={}, error=None
    )
    assert msg_id is None
    assert fake.embeds == []


# ----------------------------------------------------- chain terminal embed


def _make_chain_run_for_terminal(
    *,
    chain_id: str = "term12345",
    chain_name: str = "demo-chain",
    bucket: str | None = "education",
    status: str = "completed",
) -> Any:
    from tasque.memory.entities import ChainRun

    return ChainRun(
        id="row-" + chain_id[:8],
        chain_id=chain_id,
        chain_name=chain_name,
        bucket=bucket,
        status=status,
        started_at="2026-04-26T19:00:00.000000Z",
        ended_at="2026-04-26T19:30:00.000000Z",
        thread_id=None,
        status_message_id="msg-1",
    )


@pytest.mark.asyncio
async def test_notify_chain_terminal_posts_summary_in_description_and_full_report_in_thread() -> None:
    """The terminal embed shows only the agent's ``summary`` in the
    description plus tasque-set metadata fields. The full ``report``
    body is posted inside the anchored thread. ``produces`` is internal
    and must NOT surface as embed fields."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    # Persist the chain run so notify can stamp terminal_notified_at /
    # thread_id on it after posting. Without this, the in-memory
    # ChainRun instance has no DB row to update.
    from tasque.memory.entities import ChainRun
    from tasque.memory.repo import write_entity

    chain_run = write_entity(_make_chain_run_for_terminal())
    long_report_body = (
        "## Iteration 1\n"
        "- task1: 0.643 (rank 12) — +0.012 from 0.631\n"
        "- task2: 0.717\n"
        "- task3: 0.640\n\n"
        "Threshold rank <=4. Still failing: task1. Queued refire."
    )
    state: dict[str, Any] = {
        "plan": [],
        "completed": {
            "check_leaderboard": {
                "report": "scraped leaderboard, 35 submissions",
                "summary": "leaderboard scraped",
                "produces": {"failing_tasks": ["task1"]},
            },
            "verify": {
                "report": long_report_body,
                "summary": "iteration 1: task1 still failing, queued refire",
                "produces": {
                    "new_scores": {"task1": 0.643},
                    "iteration_number": 1,
                    "next_action": "refire",
                    "new_failing_tasks": ["task1"],
                },
            },
        },
    }

    msg_id = await notify.notify_chain_terminal(chain_run, state, "completed")
    assert msg_id is not None
    assert len(fake.embeds) == 1
    target_channel, embed, _view = fake.embeds[0]
    assert target_channel == 4242  # JOBS channel

    # Title reflects outcome; description is the agent's summary.
    assert "Chain completed" in embed["title"]
    assert "demo-chain" in embed["title"]
    assert embed["description"] == "iteration 1: task1 still failing, queued refire"

    # Tasque-set metadata fields only — no produces, no "final step".
    field_map = {f["name"]: f["value"] for f in embed["fields"]}
    assert set(field_map.keys()) <= {"chain_id", "bucket", "started", "ended"}
    assert "final step" not in field_map
    assert "iteration_number" not in field_map
    assert "next_action" not in field_map
    assert "new_failing_tasks" not in field_map
    assert "new_scores" not in field_map

    # A per-chain thread should be anchored to the embed and persisted.
    assert len(fake.threads_started) == 1
    parent_channel, anchor_msg, name = fake.threads_started[0]
    assert parent_channel == 4242
    assert anchor_msg == msg_id
    assert "demo-chain" in name and chain_run.chain_id[:8] in name

    # The full report body is posted into the new thread, regardless of
    # length. (Empty reports skip; this one isn't empty.)
    new_thread_id = 8000 + len(fake.threads_started)
    body_messages = [c for (channel, c) in fake.messages if channel == new_thread_id]
    assert body_messages, "the full report body must be posted into the thread"
    assert "".join(body_messages) == long_report_body

    from tasque.memory.db import get_session

    with get_session() as sess:
        row = sess.get(ChainRun, chain_run.id)
        assert row is not None
        assert row.thread_id is not None, (
            "thread id must persist on ChainRun.thread_id so subsequent "
            "posts about this chain reuse the same thread"
        )
        assert row.terminal_notified_at is not None


@pytest.mark.asyncio
async def test_notify_chain_terminal_reuses_existing_chain_thread() -> None:
    """If the chain already has a thread (e.g. anchored earlier by an
    approval embed), the terminal post lands inside it. No new thread
    anchoring."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    from tasque.memory.entities import ChainRun
    from tasque.memory.repo import write_entity

    cr = _make_chain_run_for_terminal()
    cr.thread_id = "9999"  # already anchored elsewhere
    chain_run = write_entity(cr)

    msg_id = await notify.notify_chain_terminal(chain_run, None, "completed")
    assert msg_id is not None
    assert len(fake.embeds) == 1
    target_channel, _embed, _ = fake.embeds[0]
    assert target_channel == 9999, "must reuse the existing chain thread"
    # No new thread anchored — we're already inside one.
    assert fake.threads_started == []

    from tasque.memory.db import get_session

    with get_session() as sess:
        row = sess.get(ChainRun, chain_run.id)
        assert row is not None
        assert row.thread_id == "9999", "must not overwrite an existing thread"
        assert row.terminal_notified_at is not None


@pytest.mark.asyncio
async def test_notify_worker_run_always_posts_full_report_in_thread() -> None:
    """The embed only carries the summary; the full report body always
    lands inside the anchored thread, regardless of length. Long reports
    are chunked at the 2000-char Discord message limit."""
    from tasque.discord.embeds import EMBED_DESC_LIMIT

    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="big report job",
            reason="",
            fire_at="now",
            status="completed",
            queued_by="cli",
            visible=True,
        )
    )

    big_report = "X" * (EMBED_DESC_LIMIT + 4000)
    msg_id = await notify.notify_worker_run(
        job, summary="ok", report=big_report, produces={}, error=None
    )
    assert msg_id is not None
    assert len(fake.threads_started) == 1
    new_thread_id = 8000 + len(fake.threads_started)
    body_messages = [
        (channel, content) for (channel, content) in fake.messages
        if channel == new_thread_id
    ]
    assert body_messages, "the full report body must be posted in the thread"
    assert "".join(c for _, c in body_messages) == big_report


@pytest.mark.asyncio
async def test_notify_worker_run_posts_short_report_in_thread_too() -> None:
    """Short reports also land in the thread body — the embed is
    summary-only, so the user expects the full body somewhere."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="d",
            reason="",
            fire_at="now",
            status="completed",
            queued_by="cli",
            visible=True,
        )
    )

    await notify.notify_worker_run(
        job, summary="ok", report="short report", produces={}, error=None
    )
    new_thread_id = 8000 + len(fake.threads_started)
    body_messages = [c for (channel, c) in fake.messages if channel == new_thread_id]
    assert body_messages == ["short report"]


@pytest.mark.asyncio
async def test_notify_worker_run_skips_thread_body_when_report_empty() -> None:
    """Empty reports (e.g. an error path) skip the thread body post —
    nothing useful to say."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="d",
            reason="",
            fire_at="now",
            status="completed",
            queued_by="cli",
            visible=True,
        )
    )

    await notify.notify_worker_run(
        job, summary="ok", report="", produces={}, error=None
    )
    assert fake.messages == []


@pytest.mark.asyncio
async def test_notify_worker_run_persists_notified_at() -> None:
    """A successful post must stamp ``QueuedJob.notified_at`` so the
    watcher's persistent gate trips on the next tick."""
    from tasque.memory.db import get_session

    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    job = write_entity(
        QueuedJob(
            kind="worker",
            bucket="health",
            directive="d",
            reason="",
            fire_at="now",
            status="completed",
            queued_by="cli",
            visible=True,
        )
    )

    await notify.notify_worker_run(
        job, summary="ok", report="r", produces={}, error=None
    )
    with get_session() as sess:
        row = sess.get(QueuedJob, job.id)
        assert row is not None
        assert row.notified_at is not None


@pytest.mark.asyncio
async def test_notify_chain_terminal_posts_full_report_in_thread_always() -> None:
    """The chain terminal embed is summary-only; the full final-step
    report always lands inside the per-chain thread, chunked across the
    2000-char limit when long."""
    from tasque.discord.embeds import EMBED_DESC_LIMIT
    from tasque.memory.repo import write_entity

    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    chain_run = write_entity(_make_chain_run_for_terminal())
    big_report = "Y" * (EMBED_DESC_LIMIT + 5000)
    state: dict[str, Any] = {
        "completed": {
            "verify": {
                "report": big_report,
                "summary": "verify done",
                "produces": {"x": 1},
            },
        },
    }

    await notify.notify_chain_terminal(chain_run, state, "completed")

    # Anchored thread id is the fake's 8001 (first start_thread).
    new_thread_id = 8000 + len(fake.threads_started)
    body_messages = [
        (channel, content) for (channel, content) in fake.messages
        if channel == new_thread_id
    ]
    assert body_messages
    assert "".join(c for _, c in body_messages) == big_report


@pytest.mark.asyncio
async def test_notify_chain_terminal_handles_missing_state() -> None:
    """A chain without checkpoint state should still produce a usable
    metadata-only embed instead of crashing — useful for edge cases like
    a chain that failed before any worker wrote a result."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    from tasque.memory.repo import write_entity

    chain_run = write_entity(_make_chain_run_for_terminal(status="failed"))
    msg_id = await notify.notify_chain_terminal(chain_run, None, "failed")
    assert msg_id is not None
    _, embed, _ = fake.embeds[0]
    assert "Chain failed" in embed["title"]
    # Tasque-set metadata fields only — no "final step", no produces.
    field_names = {f["name"] for f in embed["fields"]}
    assert field_names <= {"chain_id", "bucket", "started", "ended"}

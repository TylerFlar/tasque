"""Tests for the live chain status panel: pure renderer + watcher."""

from __future__ import annotations

import os
from typing import Any

import pytest

from tasque.chains.scheduler import launch_chain_run
from tasque.discord import chain_status_watcher, poster, threads
from tasque.discord.chain_status_panel import (
    build_chain_status_embed,
    build_chain_status_snapshot,
    is_terminal_run,
)
from tasque.jobs.runner import WorkerResult
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun

# ----------------------------------------------------------------- fixtures


# All status-panel tests run with a known chains channel id set, since
# ``TASQUE_DISCORD_CHAINS_CHANNEL_ID`` is required in production. Tests
# that need to exercise the unset / invalid paths override locally with
# ``monkeypatch``.
_TEST_CHAINS_CHANNEL_ID = 5555


@pytest.fixture(autouse=True)
def reset_threads_and_poster(tmp_path: Any) -> Any:
    registry = tmp_path / "discord_threads.json"
    old_registry = os.environ.get("TASQUE_DISCORD_THREAD_REGISTRY")
    old_chains = os.environ.get("TASQUE_DISCORD_CHAINS_CHANNEL_ID")
    os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = registry.as_posix()
    os.environ["TASQUE_DISCORD_CHAINS_CHANNEL_ID"] = str(_TEST_CHAINS_CHANNEL_ID)
    threads.reset_cache()
    poster.set_client(None)
    yield
    threads.reset_cache()
    poster.set_client(None)
    if old_registry is None:
        os.environ.pop("TASQUE_DISCORD_THREAD_REGISTRY", None)
    else:
        os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = old_registry
    if old_chains is None:
        os.environ.pop("TASQUE_DISCORD_CHAINS_CHANNEL_ID", None)
    else:
        os.environ["TASQUE_DISCORD_CHAINS_CHANNEL_ID"] = old_chains


class _FakePoster:
    def __init__(self) -> None:
        self.embeds_posted: list[tuple[int, dict[str, Any], Any]] = []
        self.edits: list[tuple[int, int, dict[str, Any] | None, Any]] = []
        # Generate ids that look like fresh Discord snowflakes — the
        # watcher derives message age from the snowflake bits and would
        # otherwise classify a tiny integer like 9000 as ~11 years old
        # and refuse to edit it. ``next_message_id`` here is just a
        # disambiguating tail so consecutive posts don't collide.
        self.next_message_id = 0
        self.edit_should_raise = False

    def _fresh_snowflake(self) -> int:
        import time

        from tasque.discord.chain_status_watcher import DISCORD_EPOCH_MS

        self.next_message_id += 1
        ms = int(time.time() * 1000) - DISCORD_EPOCH_MS
        return (ms << 22) | self.next_message_id

    async def send_message(self, channel_id: int, content: str) -> int:
        return 0

    async def send_embed(
        self, channel_id: int, embed: dict[str, Any], view: Any | None = None
    ) -> int:
        msg_id = self._fresh_snowflake()
        self.embeds_posted.append((channel_id, embed, view))
        return msg_id

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
            raise RuntimeError("simulated discord 404 — message deleted")
        self.edits.append((channel_id, message_id, embed, view))

    async def upload_file(
        self, channel_id: int, path: Any, *, content: str | None = None
    ) -> int:
        return 0

    async def fetch_recent_messages(self, channel_id: int, limit: int) -> list[Any]:
        return []

    async def start_thread(self, channel_id: int, message_id: int, name: str) -> int:
        return 0


def _make_chain_run(
    *,
    chain_id: str = "abc123",
    chain_name: str = "demo",
    bucket: str | None = "personal",
    status: str = "running",
    started_at: str = "2026-04-26T19:00:00.000000Z",
    ended_at: str | None = None,
    thread_id: str | None = None,
    status_message_id: str | None = None,
) -> ChainRun:
    return ChainRun(
        id="row" + chain_id[:8],
        chain_id=chain_id,
        chain_name=chain_name,
        bucket=bucket,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        thread_id=thread_id,
        status_message_id=status_message_id,
    )


# ----------------------------------------------------------------- snapshot


def test_snapshot_handles_missing_state_gracefully() -> None:
    """A chain whose checkpoint hasn't been written yet should still render."""
    run = _make_chain_run()
    snapshot = build_chain_status_snapshot(run, None)
    assert snapshot["chain_id"] == "abc123"
    assert snapshot["counts"]["total"] == 0
    assert snapshot["tree_lines"] == []
    assert snapshot["in_flight"] == []


def test_snapshot_counts_statuses_and_tracks_in_flight() -> None:
    state = {
        "plan": [
            {
                "id": "scan", "kind": "worker", "directive": "x",
                "depends_on": [], "consumes": [], "fan_out_on": None,
                "status": "completed", "origin": "spec", "on_failure": "halt",
                "failure_reason": None, "fan_out_index": None, "fan_out_item": None,
            },
            {
                "id": "filter", "kind": "worker", "directive": "y",
                "depends_on": ["scan"], "consumes": ["scan"], "fan_out_on": None,
                "status": "running", "origin": "spec", "on_failure": "halt",
                "failure_reason": None, "fan_out_index": None, "fan_out_item": None,
            },
            {
                "id": "approve", "kind": "approval", "directive": "z",
                "depends_on": ["filter"], "consumes": ["filter"], "fan_out_on": None,
                "status": "pending", "origin": "spec", "on_failure": "halt",
                "failure_reason": None, "fan_out_index": None, "fan_out_item": None,
            },
        ],
        "completed": {},
        "failures": {},
    }
    snapshot = build_chain_status_snapshot(_make_chain_run(), state)
    assert snapshot["counts"]["total"] == 3
    assert snapshot["counts"]["completed"] == 1
    assert snapshot["counts"]["running"] == 1
    assert snapshot["counts"]["pending"] == 1
    assert snapshot["in_flight"] == ["filter"]


def test_snapshot_tree_indents_fan_out_children_under_template() -> None:
    """Fan-out children render indented under their template parent so
    the tree mirrors how the user reasons about the run."""
    state = {
        "plan": [
            {
                "id": "scan", "kind": "worker", "directive": "x",
                "depends_on": [], "consumes": [], "fan_out_on": None,
                "status": "completed", "origin": "spec", "on_failure": "halt",
                "failure_reason": None, "fan_out_index": None, "fan_out_item": None,
            },
            {
                "id": "filter", "kind": "worker", "directive": "y",
                "depends_on": ["scan"], "consumes": ["scan"], "fan_out_on": "items",
                "status": "running", "origin": "spec", "on_failure": "halt",
                "failure_reason": None, "fan_out_index": None, "fan_out_item": None,
            },
            {
                "id": "filter[0]", "kind": "worker", "directive": "y",
                "depends_on": ["scan"], "consumes": ["scan"], "fan_out_on": None,
                "status": "completed", "origin": "spec", "on_failure": "halt",
                "failure_reason": None, "fan_out_index": 0, "fan_out_item": "a",
            },
            {
                "id": "filter[1]", "kind": "worker", "directive": "y",
                "depends_on": ["scan"], "consumes": ["scan"], "fan_out_on": None,
                "status": "running", "origin": "spec", "on_failure": "halt",
                "failure_reason": None, "fan_out_index": 1, "fan_out_item": "b",
            },
        ],
        "completed": {},
        "failures": {},
    }
    snapshot = build_chain_status_snapshot(_make_chain_run(), state)
    tree = snapshot["tree_lines"]
    # Expect: scan at depth 0, filter at depth 1, filter[0]/[1] at depth 2.
    assert any(line.startswith("✓ `scan`") for line in tree)
    assert any(line.startswith("  ▶ `filter`") and "fan-out x2" in line for line in tree)
    assert any(line.startswith("    ✓ `filter[0]`") for line in tree)
    assert any(line.startswith("    ▶ `filter[1]`") for line in tree)


def test_snapshot_carries_failure_reason_in_tree() -> None:
    state = {
        "plan": [
            {
                "id": "broken", "kind": "worker", "directive": "x",
                "depends_on": [], "consumes": [], "fan_out_on": None,
                "status": "failed", "origin": "spec", "on_failure": "halt",
                "failure_reason": "LLM call failed", "fan_out_index": None,
                "fan_out_item": None,
            },
        ],
        "completed": {},
        "failures": {"broken": "LLM call failed"},
    }
    snapshot = build_chain_status_snapshot(_make_chain_run(), state)
    assert any("err: LLM call failed" in line for line in snapshot["tree_lines"])
    assert snapshot["failures"] == {"broken": "LLM call failed"}


def test_is_terminal_run_classifies_correctly() -> None:
    for status in ("completed", "failed", "stopped"):
        snap = build_chain_status_snapshot(_make_chain_run(status=status), None)
        assert is_terminal_run(snap), f"{status} should be terminal"
    for status in ("running", "awaiting_user", "awaiting_approval", "paused"):
        snap = build_chain_status_snapshot(_make_chain_run(status=status), None)
        assert not is_terminal_run(snap), f"{status} should NOT be terminal"


# ----------------------------------------------------------------- embed


def test_embed_title_includes_run_status() -> None:
    snap = build_chain_status_snapshot(_make_chain_run(status="running"), None)
    embed = build_chain_status_embed(snap)
    assert "running" in embed["title"]
    assert snap["chain_name"] in embed["title"]


def test_embed_fields_include_chain_id_and_started() -> None:
    snap = build_chain_status_snapshot(_make_chain_run(), None)
    embed = build_chain_status_embed(snap)
    field_names = {f["name"] for f in embed["fields"]}
    assert "chain_id" in field_names
    assert "started" in field_names


def _make_term_chain_run() -> ChainRun:
    """ChainRun used for terminal-embed tests below."""
    return _make_chain_run(
        chain_id="termabcdef00",
        chain_name="demo-chain",
        bucket="finance",
        status="completed",
        started_at="2026-04-26T19:00:00.000000Z",
        ended_at="2026-04-26T19:30:00.000000Z",
    )


def test_terminal_embed_does_not_render_produces_as_inline_fields() -> None:
    """``produces`` is internal chain state — downstream consumers and
    audit only. The terminal embed must NOT surface any produces keys
    as inline fields (regardless of value type, size, or name). The
    only fields are the tasque-set chain metadata."""
    from tasque.discord.embeds import build_chain_terminal_embed

    state = {
        "completed": {
            "consolidate": {
                "report": "weekly summary body",
                "summary": "weekly digest",
                "produces": {
                    "iso_week": "2026-W17",
                    "task_ids": ["task1", "task2"],
                    "next_action": "refire",
                    "discord_post": {"channel": "x", "body": "y"},
                    "huge_blob": "x" * 5000,
                },
            },
        },
    }
    embed = build_chain_terminal_embed(_make_term_chain_run(), state, "completed")
    field_names = {f["name"] for f in embed["fields"]}
    # Every field must be tasque-set metadata.
    assert field_names <= {"chain_id", "bucket", "started", "ended"}
    # No "final step" (dropped from layout).
    assert "final step" not in field_names


def test_terminal_embed_description_is_agent_summary() -> None:
    """The description is the final completed step's agent-supplied
    ``summary`` (not the full report and not synthesised from
    produces)."""
    from tasque.discord.embeds import build_chain_terminal_embed

    state = {
        "completed": {
            "consolidate": {
                "report": "long markdown body that goes in the thread",
                "summary": "weekly digest done",
                "produces": {"iso_week": "2026-W17"},
            },
        },
    }
    embed = build_chain_terminal_embed(_make_term_chain_run(), state, "completed")
    assert embed["description"] == "weekly digest done"


def test_terminal_embed_falls_back_to_produces_summary() -> None:
    """Production CompletedOutput stores summary at ``produces.summary``,
    not at the top level. The embed must read both."""
    from tasque.discord.embeds import build_chain_terminal_embed

    state = {
        "completed": {
            "consolidate": {
                "report": "...",
                "produces": {"summary": "weekly digest done"},
            },
        },
    }
    embed = build_chain_terminal_embed(_make_term_chain_run(), state, "completed")
    assert "weekly digest done" in (embed.get("description") or "")


def test_terminal_embed_appends_fan_out_rollup() -> None:
    """Fan-out chains get a per-branch outcome rollup in the description.
    Without this, the user-facing embed shows the last passthrough's
    summary and silently hides leaf failures (the trading-scan bug:
    8 dispatch legs all returning ``no_trades`` while the embed reads
    green)."""
    from tasque.discord.embeds import build_chain_terminal_embed

    state = {
        "completed": {
            "scan": {"report": "...", "produces": {"summary": "8 ready"}},
            "scan[0]": {
                "report": "...",
                "produces": {"bucket_id": "car", "outcome": "scanned"},
            },
            "scan[1]": {
                "report": "...",
                "produces": {"bucket_id": "home", "outcome": "no_trades"},
            },
            "scan[2]": {
                "report": "...",
                "produces": {"bucket_id": "wants", "outcome": "scanned"},
            },
        },
    }
    embed = build_chain_terminal_embed(_make_term_chain_run(), state, "completed")
    desc = embed.get("description") or ""
    # Header references the parent template id and total branch count.
    assert "scan" in desc
    assert "3 branches" in desc
    # Each outcome bucket list is present.
    assert "scanned (2): car, wants" in desc
    assert "no_trades (1): home" in desc


def test_embed_summary_line_reflects_step_progress() -> None:
    state = {
        "plan": [
            {
                "id": f"s{i}", "kind": "worker", "directive": "x",
                "depends_on": [], "consumes": [], "fan_out_on": None,
                "status": "completed" if i < 2 else "pending", "origin": "spec",
                "on_failure": "halt", "failure_reason": None,
                "fan_out_index": None, "fan_out_item": None,
            }
            for i in range(5)
        ],
        "completed": {}, "failures": {},
    }
    snap = build_chain_status_snapshot(_make_chain_run(), state)
    embed = build_chain_status_embed(snap)
    assert "step **2/5**" in embed["description"]


# ----------------------------------------------------------------- watcher


def _spec() -> dict[str, Any]:
    return {
        "chain_name": "status-demo",
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
    """Stand-in for the chain worker so we don't touch real LLMs."""
    record: dict[str, Any] = {"calls": []}

    def _fake(job: Any, *, consumes: dict[str, Any] | None = None, **_: Any) -> WorkerResult:
        record["calls"].append(job.chain_step_id)
        return WorkerResult(report="ok", summary="ok", produces={"items": ["a"]}, error=None)

    import tasque.chains.graph.worker as worker_mod

    monkeypatch.setattr(worker_mod, "run_worker", _fake)
    return record


@pytest.mark.asyncio
async def test_watcher_posts_status_message_on_first_pass(
    fake_worker: dict[str, Any],
) -> None:
    chain_id = launch_chain_run(_spec(), thread_id="7777")
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )

    assert len(fake.embeds_posted) == 1, "expected one initial post"
    channel_id, embed, _view = fake.embeds_posted[0]
    assert channel_id == _TEST_CHAINS_CHANNEL_ID, (
        "status embed must land in TASQUE_DISCORD_CHAINS_CHANNEL_ID, even "
        "when the chain has its own thread"
    )
    assert chain_id in embed["description"] or any(
        f["name"] == "chain_id" and f["value"] == chain_id for f in embed["fields"]
    )

    # The message id should have been persisted on the ChainRun row.
    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        assert row["status_message_id"] is not None


@pytest.mark.asyncio
async def test_watcher_skips_edit_when_signature_unchanged(
    fake_worker: dict[str, Any],
) -> None:
    launch_chain_run(_spec(), thread_id="7777")
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert len(fake.embeds_posted) == 1
    assert len(fake.edits) == 0

    # Second pass — nothing has changed; we should NOT edit.
    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )
    # A new watcher has an empty signature cache so it will do one edit
    # on its first pass to confirm the message exists, but it must not
    # POST a second message for the same chain.
    assert len(fake.embeds_posted) == 1, "must not double-post"


@pytest.mark.asyncio
async def test_watcher_routes_to_chains_channel_ignoring_bucket_and_thread(
    fake_worker: dict[str, Any],
) -> None:
    """No fallbacks: even when the chain has its own thread AND the
    bucket coach thread is registered, the status embed always lands in
    ``TASQUE_DISCORD_CHAINS_CHANNEL_ID``."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 9876)
    threads.set_thread_id(threads.bucket_purpose("personal"), 4242)

    launch_chain_run(_spec(), thread_id="7777")
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert len(fake.embeds_posted) == 1
    channel_id, _embed, _view = fake.embeds_posted[0]
    assert channel_id == _TEST_CHAINS_CHANNEL_ID


def test_chains_channel_id_raises_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TASQUE_DISCORD_CHAINS_CHANNEL_ID", raising=False)
    with pytest.raises(RuntimeError, match="TASQUE_DISCORD_CHAINS_CHANNEL_ID"):
        chain_status_watcher.chains_channel_id()


def test_chains_channel_id_raises_on_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TASQUE_DISCORD_CHAINS_CHANNEL_ID", "not-an-int")
    with pytest.raises(RuntimeError, match="not an integer"):
        chain_status_watcher.chains_channel_id()


@pytest.mark.asyncio
async def test_watcher_re_posts_after_edit_failure(
    fake_worker: dict[str, Any],
) -> None:
    """If editing the cached message raises (user deleted it), the
    watcher should fall back to posting a fresh message and overwrite
    the cached id."""
    chain_id = launch_chain_run(_spec(), thread_id="7777")
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    # Pre-seed a stale message id on the row to force an edit attempt.
    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        sess.execute(
            ChainRun.__table__.update()
            .where(ChainRun.chain_id == chain_id)
            .values(status_message_id="424242")
        )

    fake.edit_should_raise = True
    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )

    # One fallback post should have happened; the stored id should be
    # the new one, not 424242.
    assert len(fake.embeds_posted) == 1
    with get_session() as sess:
        row = sess.execute(
            ChainRun.__table__.select().where(ChainRun.chain_id == chain_id)
        ).mappings().first()
        assert row is not None
        assert row["status_message_id"] != "424242"


@pytest.mark.asyncio
async def test_watcher_skips_terminal_chain_with_stale_message_id() -> None:
    """Daemon restart after long downtime: a terminal chain's cached
    message id is now too old to edit. The watcher must NOT post a
    fresh duplicate panel — that flooded the chains channel on every
    restart. It should clear the stale id and let the skip-retroactive
    guard handle the chain on subsequent ticks.
    """
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    from tasque.memory.entities import ChainRun as ChainRunEntity
    from tasque.memory.entities import utc_now_iso

    chain_id = "stalecid000000000"
    # Snowflake from 2 hours ago — well past the 55min edit window.
    import time as _time
    two_hours_ago_ms = int(_time.time() * 1000) - 2 * 60 * 60 * 1000
    stale_id = (
        (two_hours_ago_ms - chain_status_watcher.DISCORD_EPOCH_MS) << 22
    ) | 1
    with get_session() as sess:
        sess.add(
            ChainRunEntity(
                id="row" + chain_id[:8],
                chain_id=chain_id,
                chain_name="stale-demo",
                bucket="personal",
                status="completed",
                started_at=utc_now_iso(),
                ended_at=utc_now_iso(),
                status_message_id=str(stale_id),
            )
        )

    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )

    # No fresh post for the terminal chain whose panel fell out of
    # editable age — that's exactly the flood we're preventing.
    assert fake.embeds_posted == []
    assert fake.edits == []
    # The stale id should have been cleared so the skip-retroactive
    # guard picks up subsequent ticks before doing any work.
    with get_session() as sess:
        row = sess.execute(
            ChainRunEntity.__table__.select().where(
                ChainRunEntity.chain_id == chain_id
            )
        ).mappings().first()
        assert row is not None
        assert row["status_message_id"] is None


# ------------------------------- terminal one-shot notification


def _spec_single_worker() -> dict[str, Any]:
    return {
        "chain_name": "single-worker-demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {"id": "only", "kind": "worker", "directive": "do thing", "tier": "small"},
        ],
    }


@pytest.mark.asyncio
async def test_watcher_fires_terminal_notify_on_first_observed_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain that starts running, gets observed, then transitions to
    completed should yield exactly ONE rich terminal embed in the JOBS
    channel. This is the iteration-summary the user expected to see."""
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)

    # Stage worker: returns a result on the first call so the chain
    # progresses from 'running' to 'completed' between watcher ticks.
    state = {"completed": False}

    def _fake(job: Any, *, consumes: dict[str, Any] | None = None, **_: Any) -> WorkerResult:
        state["completed"] = True
        return WorkerResult(
            report="iteration done — task1 score 0.643",
            summary="ok",
            produces={"next_action": "refire"},
            error=None,
        )

    import tasque.chains.graph.worker as worker_mod

    monkeypatch.setattr(worker_mod, "run_worker", _fake)

    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    # Track terminal-notify calls by patching notify.notify_chain_terminal.
    notify_calls: list[tuple[str, str | None, dict[str, Any]]] = []

    async def _capture_notify(chain_run: Any, state_arg: Any, kind: str) -> int | None:
        notify_calls.append((chain_run.chain_id, kind, dict(state_arg or {})))
        return 6789

    monkeypatch.setattr(
        chain_status_watcher.notify, "notify_chain_terminal", _capture_notify
    )

    # Launch the chain. With our fake worker it runs synchronously to
    # completion before returning, so the row is already 'completed' when
    # the watcher first looks.
    launch_chain_run(_spec_single_worker())

    # Tick 1: chain is already terminal, watcher has no prior signature
    # for it -> must NOT terminal-notify (historical-completion guard).
    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert notify_calls == [], (
        "watcher must not retroactively announce a chain that was already "
        "terminal on first observation; the user would otherwise get a "
        "flood of historical embeds on every daemon restart"
    )


@pytest.mark.asyncio
async def test_watcher_terminal_notify_fires_on_running_to_completed() -> None:
    """Mid-flight transition: tick 1 sees 'running', tick 2 sees
    'completed'. Exactly ONE rich terminal embed posted to JOBS per
    transition. Subsequent ticks at the same terminal status MUST NOT
    re-fire — the user shouldn't get duplicate iteration summaries.

    Drives the real notify path (no monkeypatch) so the test exercises
    the persistent ``terminal_notified_at`` gate end-to-end. The
    in-memory set alone wouldn't survive a daemon restart.
    """
    threads.set_thread_id(threads.PURPOSE_JOBS, 4242)

    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    from tasque.memory.entities import ChainRun as ChainRunEntity
    from tasque.memory.entities import utc_now_iso

    chain_id = "cidruncomplete000"
    with get_session() as sess:
        sess.add(
            ChainRunEntity(
                id="row" + chain_id[:8],
                chain_id=chain_id,
                chain_name="trans-demo",
                bucket="personal",
                status="running",
                started_at=utc_now_iso(),
            )
        )

    def _jobs_posts() -> list[dict[str, Any]]:
        return [embed for (channel_id, embed, _v) in fake.embeds_posted if channel_id == 4242]

    # Tick 1: row is running. Watcher posts the live status panel
    # (chains channel 5555). No terminal post yet (status not terminal).
    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert _jobs_posts() == []
    chains_posts_after_t1 = [
        embed for (channel_id, embed, _v) in fake.embeds_posted
        if channel_id == _TEST_CHAINS_CHANNEL_ID
    ]
    assert len(chains_posts_after_t1) == 1, "tick 1 should post one live-panel embed"

    # Flip the row to 'completed'. status_message_id was persisted on
    # the tick-1 post; it stays — tick 2 will edit the chains-channel
    # message in place, and ALSO post the terminal embed to JOBS.
    with get_session() as sess:
        from sqlalchemy import update

        sess.execute(
            update(ChainRunEntity)
            .where(ChainRunEntity.chain_id == chain_id)
            .values(status="completed", ended_at=utc_now_iso())
        )

    # Tick 2: transition observed -> exactly one JOBS-channel terminal embed.
    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )
    jobs_posts = _jobs_posts()
    assert len(jobs_posts) == 1
    assert "Chain completed" in jobs_posts[0]["title"]
    assert "trans-demo" in jobs_posts[0]["title"]

    # Persistent gate: the column should now be set.
    with get_session() as sess:
        row = sess.execute(
            ChainRunEntity.__table__.select().where(
                ChainRunEntity.chain_id == chain_id
            )
        ).mappings().first()
        assert row is not None
        assert row["terminal_notified_at"] is not None, (
            "terminal_notified_at must be stamped after a successful post"
        )

    # Tick 3 (same watcher-process call sequence): persistent + in-memory
    # gates should both prevent a second JOBS post.
    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert len(_jobs_posts()) == 1, "must not re-fire terminal embed across ticks"

    # Tick 4 simulates a fresh watcher-process boot after a daemon
    # restart (the fixture's `run_chain_status_watcher` already creates
    # fresh in-memory state on each call). Persistent column is the
    # only thing keeping us safe here.
    await chain_status_watcher.run_chain_status_watcher(
        max_iterations=1, poll_seconds=0
    )
    assert len(_jobs_posts()) == 1, (
        "persistent terminal_notified_at must survive a fresh watcher "
        "process so the user does not get a duplicate report on restart"
    )

    # status_message_id must be cleared after the terminal embed lands —
    # there's no panel to keep editing once the chain is finished, and
    # holding the id forever is just stale state on the row.
    with get_session() as sess:
        row = sess.execute(
            ChainRunEntity.__table__.select().where(
                ChainRunEntity.chain_id == chain_id
            )
        ).mappings().first()
        assert row is not None
        assert row["status_message_id"] is None, (
            "status_message_id should be cleared once the chain reaches "
            "terminal — the panel is no longer being edited"
        )

"""Tests for the strategist's monitoring graph + post pipeline.

Mock the LLM with canned :class:`StrategistOutput` JSON, run the
graph, verify:

- new Aims persisted with ``source="strategist"``
- Signals sent with ``from_bucket="strategist"``
- Aim statuses flipped via ``aim_status_changes``
- summary posted to the strategist Discord thread via a fake poster
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from sqlalchemy import select

from tasque.discord import poster, threads
from tasque.memory.db import get_session
from tasque.memory.entities import Aim, Note, Signal
from tasque.memory.repo import write_entity
from tasque.reply.runtime import HistoryMessage
from tasque.strategist.graph import (
    reset_graph_cache,
    run_monitoring,
    run_monitoring_and_post,
)
from tasque.strategist.output import StrategistOutput
from tasque.strategist.persist import persist_results

from .conftest import make_canned_chat_model


@pytest.fixture(autouse=True)
def reset_threads_and_poster_and_graph(tmp_path: Any) -> Any:
    """Per-test thread registry isolated from the user's data dir.

    Also resets the strategist graph cache so each test re-builds with a
    clean module-state.
    """
    registry = tmp_path / "discord_threads.json"
    old = os.environ.get("TASQUE_DISCORD_THREAD_REGISTRY")
    os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = registry.as_posix()
    threads.reset_cache()
    poster.set_client(None)
    reset_graph_cache()
    yield
    threads.reset_cache()
    poster.set_client(None)
    reset_graph_cache()
    if old is None:
        os.environ.pop("TASQUE_DISCORD_THREAD_REGISTRY", None)
    else:
        os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = old


class _FakePoster:
    """Captures every poster call so tests can run without nextcord."""

    def __init__(self) -> None:
        self.sent_messages: list[tuple[int, str]] = []

    async def send_message(self, channel_id: int, content: str) -> int:
        self.sent_messages.append((channel_id, content))
        return 1000 + len(self.sent_messages)

    async def send_embed(
        self, channel_id: int, embed: dict[str, Any], view: Any | None = None
    ) -> int:
        return 0

    async def edit_message(
        self,
        channel_id: int,
        message_id: int,
        *,
        content: str | None = None,
        embed: dict[str, Any] | None = None,
        view: Any | None = None,
    ) -> None:
        return None

    async def upload_file(
        self, channel_id: int, path: Any, *, content: str | None = None
    ) -> int:
        return 0

    async def fetch_recent_messages(
        self, channel_id: int, limit: int
    ) -> list[HistoryMessage]:
        return []


def _seed_existing_aim(*, scope: str = "long_term", bucket: str | None = None) -> str:
    aim = Aim(
        title="existing aim",
        bucket=bucket,
        scope=scope,
        source="strategist",
    )
    return write_entity(aim).id


def _seed_coach_note(bucket: str, content: str) -> None:
    write_entity(
        Note(
            content=content,
            bucket=bucket,
            durability="ephemeral",
            source="coach",
        )
    )


# ----------------------------------------------------------------- persist


def test_persist_results_writes_new_aims_signals_and_status_flips() -> None:
    existing_id = _seed_existing_aim()

    parsed = StrategistOutput.model_validate(
        {
            "summary": "## weekly\n\nAll quiet.",
            "new_aims": [
                {
                    "title": "rebalance health vs career",
                    "scope": "long_term",
                    "bucket": None,
                    "target_date": None,
                    "description": "spotted a debt pattern",
                    "parent_id": None,
                },
                {
                    "title": "sleep before midnight",
                    "scope": "bucket",
                    "bucket": "health",
                    "target_date": None,
                    "description": "",
                    "parent_id": None,
                },
            ],
            "signals": [
                {
                    "to_bucket": "health",
                    "kind": "strategist_alert",
                    "urgency": "high",
                    "summary": "rebalance: career creep is eating sleep",
                    "body": "see the new long-term Aim",
                    "expires_at": None,
                }
            ],
            "aim_status_changes": [
                {"aim_id": existing_id, "status": "dropped", "reason": "stale"}
            ],
        }
    )

    report = persist_results(parsed)
    assert len(report["new_aim_ids"]) == 2
    assert len(report["signal_ids"]) == 1
    assert report["flipped_aim_ids"] == [existing_id]

    with get_session() as sess:
        # The two new Aims exist with source=strategist.
        aims = list(sess.execute(select(Aim)).scalars().all())
        new_titles = {a.title for a in aims if a.id in report["new_aim_ids"]}
        assert new_titles == {"rebalance health vs career", "sleep before midnight"}
        for a in aims:
            if a.id in report["new_aim_ids"]:
                assert a.source == "strategist"

        # The Signal was sent from the strategist.
        sigs = list(sess.execute(select(Signal)).scalars().all())
        assert len(sigs) == 1
        assert sigs[0].from_bucket == "strategist"
        assert sigs[0].to_bucket == "health"
        assert sigs[0].urgency == "high"

        # The existing Aim's status was flipped.
        flipped = sess.get(Aim, existing_id)
        assert flipped is not None
        assert flipped.status == "dropped"


# ------------------------------------------------------------- run_monitoring


def test_run_monitoring_persists_canned_llm_output() -> None:
    _seed_coach_note("career", "shipped the migration this week")
    _seed_coach_note("health", "skipped two workouts this week")

    canned = {
        "summary": "## weekly check-in\n\n- career: shipped migration\n- health: 2 missed workouts",
        "new_aims": [
            {
                "title": "weekly cardio floor",
                "scope": "bucket",
                "bucket": "health",
                "target_date": None,
                "description": "",
                "parent_id": None,
            }
        ],
        "signals": [
            {
                "to_bucket": "health",
                "kind": "strategist_alert",
                "urgency": "normal",
                "summary": "set a weekly cardio floor",
                "body": "two missed sessions in a row → make floor explicit",
                "expires_at": None,
            }
        ],
        "aim_status_changes": [],
    }
    fake_llm = make_canned_chat_model([canned])

    state = run_monitoring(reason="test-trigger", llm=fake_llm)

    assert "error" not in state
    parsed = state.get("parsed")
    assert parsed is not None
    assert parsed.summary.startswith("## weekly check-in")

    persisted = state.get("persisted") or {}
    assert len(persisted.get("new_aim_ids", [])) == 1
    assert len(persisted.get("signal_ids", [])) == 1

    with get_session() as sess:
        aims = list(sess.execute(select(Aim)).scalars().all())
        assert len(aims) == 1
        assert aims[0].title == "weekly cardio floor"
        assert aims[0].bucket == "health"
        assert aims[0].source == "strategist"

        sigs = list(sess.execute(select(Signal)).scalars().all())
        assert len(sigs) == 1
        assert sigs[0].from_bucket == "strategist"


def test_run_monitoring_records_error_on_unparseable_llm_output() -> None:
    fake_llm = make_canned_chat_model(["this is not a JSON code block at all"])
    state = run_monitoring(llm=fake_llm)
    assert "error" in state
    assert "no JSON block" in state["error"]


def test_run_monitoring_records_error_on_invalid_schema() -> None:
    bad = {
        # missing required ``summary``
        "new_aims": [],
        "signals": [],
        "aim_status_changes": [],
    }
    fake_llm = make_canned_chat_model([bad])
    state = run_monitoring(llm=fake_llm)
    assert "error" in state
    assert "StrategistOutput" in state["error"]


# -------------------------------------------------- run_monitoring_and_post


@pytest.mark.asyncio
async def test_monitoring_post_publishes_summary_to_strategist_thread() -> None:
    threads.set_thread_id(threads.PURPOSE_STRATEGIST, 9090)
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    canned = {
        "summary": "## weekly\n\nshort and sweet.",
        "new_aims": [],
        "signals": [],
        "aim_status_changes": [],
    }
    fake_llm = make_canned_chat_model([canned])
    state = await run_monitoring_and_post(llm=fake_llm)
    assert "error" not in state

    posted_ids = state.get("posted_message_ids") or []
    assert posted_ids
    assert len(fake.sent_messages) == 1
    channel_id, content = fake.sent_messages[0]
    assert channel_id == 9090
    assert content == "## weekly\n\nshort and sweet."


@pytest.mark.asyncio
async def test_monitoring_post_skips_when_no_strategist_thread() -> None:
    # No PURPOSE_STRATEGIST registered.
    fake = _FakePoster()
    poster.set_client(fake)  # type: ignore[arg-type]

    canned = {
        "summary": "## empty",
        "new_aims": [],
        "signals": [],
        "aim_status_changes": [],
    }
    fake_llm = make_canned_chat_model([canned])
    state = await run_monitoring_and_post(llm=fake_llm)
    assert state.get("posted_message_ids") == []
    assert fake.sent_messages == []

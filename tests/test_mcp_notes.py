"""Tests for LLM-facing Note MCP tools."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from tasque.memory.db import get_session
from tasque.memory.entities import CoachPending, Note
from tasque.memory.repo import write_entity


def _call_tool(name: str, **kwargs: object) -> dict[str, Any] | list[Any]:
    from tasque.mcp.server import build_server

    server = build_server()
    tool = server._tool_manager.get_tool(name)
    fn = getattr(tool, "fn", None)
    if fn is None:
        raise RuntimeError(f"expected {name} to be a FastMCP-decorated tool")
    return json.loads(fn(**kwargs))  # type: ignore[operator]


def test_note_update_patches_existing_note_and_wakes_bucket() -> None:
    original = Note(
        content="sleep goal is 7 hours",
        bucket="health",
        durability="durable",
        source="user",
        meta={"confidence": "low"},
    )
    write_entity(original)

    out = _call_tool(
        "note_update",
        note_id=original.id,
        content="sleep goal is 8 hours",
        durability="behavioral",
        meta={"confidence": "high"},
    )

    assert out["ok"] is True
    assert out["id"] == original.id
    assert out["note"]["content"] == "sleep goal is 8 hours"
    assert out["note"]["durability"] == "behavioral"
    assert out["note"]["meta"] == {"confidence": "high"}

    with get_session() as sess:
        note = sess.get(Note, original.id)
        assert note is not None
        assert note.content == "sleep goal is 8 hours"
        assert note.durability == "behavioral"
        assert note.meta == {"confidence": "high"}
        pending = list(sess.execute(select(CoachPending)).scalars().all())

    assert len(pending) == 1
    assert pending[0].bucket == "health"
    assert pending[0].dedup_key == "tool:note_update:health"


def test_note_update_can_move_note_between_buckets() -> None:
    original = Note(
        content="tax folder lives in the cabinet",
        bucket="home",
        durability="durable",
        source="user",
    )
    write_entity(original)

    out = _call_tool("note_update", note_id=original.id, bucket="finance")

    assert out["ok"] is True
    assert out["note"]["bucket"] == "finance"

    with get_session() as sess:
        note = sess.get(Note, original.id)
        assert note is not None
        assert note.bucket == "finance"
        pending = list(sess.execute(select(CoachPending)).scalars().all())

    assert {row.bucket for row in pending} == {"home", "finance"}
    assert {row.dedup_key for row in pending} == {
        "tool:note_update:home",
        "tool:note_update:finance",
    }


def test_note_update_rejects_empty_patch() -> None:
    out = _call_tool("note_update", note_id="missing")
    assert out == {"ok": False, "error": "provide at least one field to update"}


def test_note_supersede_archives_old_note_and_returns_replacement() -> None:
    old = Note(
        content="prefers evening workouts",
        bucket="health",
        durability="durable",
        source="user",
        meta={"observed": "2026-04"},
    )
    write_entity(old)

    out = _call_tool(
        "note_supersede",
        note_id=old.id,
        content="prefers morning workouts",
        meta={"observed": "2026-05"},
    )

    assert out["ok"] is True
    new_id = out["id"]
    assert out["superseded_id"] == old.id
    assert out["note"]["content"] == "prefers morning workouts"
    assert out["note"]["bucket"] == "health"
    assert out["note"]["durability"] == "durable"
    assert out["note"]["source"] == "user"
    assert out["note"]["meta"] == {"observed": "2026-05"}

    with get_session() as sess:
        old_row = sess.get(Note, old.id)
        new_row = sess.get(Note, new_id)
        assert old_row is not None
        assert new_row is not None
        assert old_row.archived is True
        assert old_row.superseded_by == new_id
        assert new_row.archived is False
        pending = list(sess.execute(select(CoachPending)).scalars().all())

    assert len(pending) == 1
    assert pending[0].bucket == "health"
    assert pending[0].dedup_key == "tool:note_supersede:health"

    listed = _call_tool(
        "note_list",
        intent="current durable health preferences",
        bucket="health",
        durability="durable",
    )
    assert isinstance(listed, list)
    assert [row["id"] for row in listed] == [new_id]

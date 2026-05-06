"""Tests for ``tasque.mcp.tool_triggers.dispatch_tool_event``.

These cover the dispatcher in isolation — we replace the module's
``enqueue_fn`` attribute with a recording fake, so no real coach
trigger queue is touched. Each test asserts on the calls captured.
"""

from __future__ import annotations

from typing import Any

import pytest

from tasque.buckets import ALL_BUCKETS
from tasque.mcp import tool_triggers


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_enqueue(bucket: str, *, reason: str, dedup_key: str | None = None) -> str | None:
        calls.append({"bucket": bucket, "reason": reason, "dedup_key": dedup_key})
        return f"row-{len(calls)}"

    monkeypatch.setattr(tool_triggers, "enqueue_fn", fake_enqueue)
    return calls


def test_note_create_does_not_wake_target_bucket(
    captured_calls: list[dict[str, Any]],
) -> None:
    tool_triggers.dispatch_tool_event(
        "note_create", bucket="personal", source="mcp"
    )
    assert captured_calls == []


def test_note_create_with_unknown_bucket_is_silent(
    captured_calls: list[dict[str, Any]],
) -> None:
    tool_triggers.dispatch_tool_event("note_create", bucket="not-a-bucket")
    assert captured_calls == []


def test_signal_create_wakes_to_bucket_only(
    captured_calls: list[dict[str, Any]],
) -> None:
    tool_triggers.dispatch_tool_event(
        "signal_create", from_bucket="finance", to_bucket="career"
    )
    assert len(captured_calls) == 1
    assert captured_calls[0]["bucket"] == "career"
    assert captured_calls[0]["dedup_key"] == "tool:signal_create:career"


def test_strategist_aim_added_signal_dedups_with_aim_create(
    captured_calls: list[dict[str, Any]],
) -> None:
    tool_triggers.dispatch_tool_event(
        "signal_create",
        from_bucket="strategist",
        to_bucket="relationships",
        kind="aim_added",
    )
    assert len(captured_calls) == 1
    assert captured_calls[0]["bucket"] == "relationships"
    assert captured_calls[0]["reason"] == "tool:signal_create:relationships"
    assert captured_calls[0]["dedup_key"] == "tool:aim_create:relationships"


def test_signal_create_broadcast_wakes_all_buckets(
    captured_calls: list[dict[str, Any]],
) -> None:
    tool_triggers.dispatch_tool_event(
        "signal_create", from_bucket="strategist", to_bucket="all"
    )
    assert len(captured_calls) == len(ALL_BUCKETS)
    woken = {c["bucket"] for c in captured_calls}
    assert woken == set(ALL_BUCKETS)
    # Each broadcast call gets its own dedup key per bucket so a same-
    # window single-bucket trigger from another tool isn't masked by it.
    dedup_keys = {c["dedup_key"] for c in captured_calls}
    assert len(dedup_keys) == len(ALL_BUCKETS)
    for key in dedup_keys:
        assert key.startswith("tool:signal_create:all:")


def test_signal_create_invalid_to_bucket_is_silent(
    captured_calls: list[dict[str, Any]],
) -> None:
    tool_triggers.dispatch_tool_event(
        "signal_create", from_bucket="finance", to_bucket="garbage"
    )
    assert captured_calls == []


def test_aim_create_long_term_does_not_wake_any_bucket(
    captured_calls: list[dict[str, Any]],
) -> None:
    # Long-term aims have no bucket; there's nothing to wake.
    tool_triggers.dispatch_tool_event(
        "aim_create", bucket=None, scope="long_term"
    )
    assert captured_calls == []


def test_aim_create_bucket_scope_wakes_owner(
    captured_calls: list[dict[str, Any]],
) -> None:
    tool_triggers.dispatch_tool_event(
        "aim_create", bucket="health", scope="bucket"
    )
    assert len(captured_calls) == 1
    assert captured_calls[0]["bucket"] == "health"
    assert captured_calls[0]["dedup_key"] == "tool:aim_create:health"


def test_unknown_tool_is_a_no_op(captured_calls: list[dict[str, Any]]) -> None:
    tool_triggers.dispatch_tool_event(
        "chain_template_create", bucket="career", scope="bucket"
    )
    assert captured_calls == []


def test_handler_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> str | None:
        raise RuntimeError("db went away")

    monkeypatch.setattr(tool_triggers, "enqueue_fn", boom)

    # Should NOT raise — the underlying mutation already succeeded; a
    # failed trigger must not propagate to the caller.
    tool_triggers.dispatch_tool_event(
        "note_create", bucket="personal", source="mcp"
    )


# -------- end-to-end: MCP tool invocation actually reaches the dispatcher

def test_note_create_via_mcp_does_not_enqueue_coach_trigger() -> None:
    """Notes are memory edits; they should not start another coach pass."""
    import asyncio

    from sqlalchemy import select

    from tasque.mcp.server import build_server
    from tasque.memory.db import get_session
    from tasque.memory.entities import CoachPending, Note

    server = build_server()

    async def _call() -> Any:
        return await server.call_tool(
            "note_create",
            {"content": "hello world", "bucket": "career", "source": "mcp"},
        )

    asyncio.run(_call())

    with get_session() as sess:
        notes = list(sess.execute(select(Note)).scalars().all())
        pending = list(sess.execute(select(CoachPending)).scalars().all())

    assert len(notes) == 1
    assert notes[0].bucket == "career"
    assert pending == []


def test_signal_create_via_mcp_to_all_enqueues_per_bucket() -> None:
    """Broadcast signal wakes all nine bucket coaches."""
    import asyncio

    from sqlalchemy import select

    from tasque.mcp.server import build_server
    from tasque.memory.db import get_session
    from tasque.memory.entities import CoachPending

    server = build_server()

    async def _call() -> Any:
        return await server.call_tool(
            "signal_create",
            {
                "from_bucket": "strategist",
                "to_bucket": "all",
                "kind": "rebalance",
                "urgency": "normal",
                "summary": "global rebalance",
            },
        )

    asyncio.run(_call())

    with get_session() as sess:
        pending = list(sess.execute(select(CoachPending)).scalars().all())

    assert len(pending) == len(ALL_BUCKETS)
    woken = {p.bucket for p in pending}
    assert woken == set(ALL_BUCKETS)

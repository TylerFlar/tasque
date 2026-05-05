"""Tests for the agent result inbox + the submit_worker_result MCP tool."""

from __future__ import annotations

import json

from sqlalchemy import select

from tasque.agents import result_inbox
from tasque.memory.db import get_session
from tasque.memory.entities import AgentResult


def test_mint_token_returns_unique_hex_strings() -> None:
    a = result_inbox.mint_token()
    b = result_inbox.mint_token()
    assert a != b
    assert len(a) == 32 and all(c in "0123456789abcdef" for c in a)
    assert len(b) == 32 and all(c in "0123456789abcdef" for c in b)


def test_deposit_and_consume_round_trip() -> None:
    token = result_inbox.mint_token()
    result_inbox.deposit(
        result_token=token,
        agent_kind="worker",
        payload={"report": "r", "summary": "s", "produces": {"k": "v"}},
    )
    payload = result_inbox.read_and_consume(token, agent_kind="worker")
    assert payload == {"report": "r", "summary": "s", "produces": {"k": "v"}}
    # Row must be gone after read.
    with get_session() as sess:
        assert sess.get(AgentResult, token) is None


def test_consume_returns_none_for_unknown_token() -> None:
    assert result_inbox.read_and_consume("never-deposited", agent_kind="worker") is None


def test_consume_rejects_mismatched_agent_kind() -> None:
    """A row deposited by one agent must not satisfy another agent's
    read — even though we still consume the row to prevent stale data."""
    token = result_inbox.mint_token()
    result_inbox.deposit(
        result_token=token,
        agent_kind="coach",
        payload={"thread_post": "hi"},
    )
    payload = result_inbox.read_and_consume(token, agent_kind="worker")
    assert payload is None
    # Row should still have been deleted to avoid lingering bad data.
    with get_session() as sess:
        assert sess.get(AgentResult, token) is None


def test_deposit_overwrites_on_repeat_token() -> None:
    """Idempotent on token: a second deposit replaces the first."""
    token = result_inbox.mint_token()
    result_inbox.deposit(
        result_token=token, agent_kind="worker", payload={"report": "v1"}
    )
    result_inbox.deposit(
        result_token=token, agent_kind="worker", payload={"report": "v2"}
    )
    with get_session() as sess:
        rows = list(sess.execute(select(AgentResult)).scalars().all())
    assert len(rows) == 1
    assert json.loads(rows[0].payload_json) == {"report": "v2"}


def test_reap_stale_removes_old_rows_only() -> None:
    fresh_token = result_inbox.mint_token()
    result_inbox.deposit(
        result_token=fresh_token, agent_kind="worker", payload={"r": "fresh"}
    )
    # Insert a stale row with a backdated created_at.
    stale_token = result_inbox.mint_token()
    with get_session() as sess:
        sess.add(
            AgentResult(
                result_token=stale_token,
                agent_kind="worker",
                payload_json='{"r":"stale"}',
                created_at="1970-01-01T00:00:00.000000Z",
            )
        )
    swept = result_inbox.reap_stale(max_age_seconds=60)
    assert swept == 1
    with get_session() as sess:
        assert sess.get(AgentResult, stale_token) is None
        assert sess.get(AgentResult, fresh_token) is not None


# ------------------------------------------------ submit_worker_result tool


def _call_tool(tool: object, **kwargs: object) -> str:
    """FastMCP wraps decorated callables; the underlying ``fn`` is the
    original Python function. Call it directly so tests don't have to
    spin up the full MCP server."""
    fn = getattr(tool, "fn", None)
    if fn is None:
        raise RuntimeError("expected FastMCP-decorated tool with .fn attribute")
    return fn(**kwargs)  # type: ignore[no-any-return]


def _get_tool(name: str) -> object:
    from tasque.mcp.server import build_server

    server = build_server()
    return server._tool_manager.get_tool(name)


def _get_submit_worker_result() -> object:
    return _get_tool("submit_worker_result")


def test_submit_worker_result_writes_inbox_row() -> None:
    tool = _get_submit_worker_result()
    token = result_inbox.mint_token()
    out = _call_tool(
        tool,
        result_token=token,
        report="full body",
        summary="one liner",
        produces={"k": "v"},
    )
    assert json.loads(out) == {"ok": True}
    payload = result_inbox.read_and_consume(token, agent_kind="worker")
    assert payload == {
        "report": "full body",
        "summary": "one liner",
        "produces": {"k": "v"},
        # Default-omitted ``error`` is normalised to None on deposit so
        # the runner can read it back uniformly.
        "error": None,
    }


def test_submit_worker_result_defaults_produces_to_empty_dict() -> None:
    tool = _get_submit_worker_result()
    token = result_inbox.mint_token()
    out = _call_tool(
        tool,
        result_token=token,
        report="r",
        summary="s",
        produces=None,
    )
    assert json.loads(out) == {"ok": True}
    payload = result_inbox.read_and_consume(token, agent_kind="worker")
    assert payload is not None
    assert payload["produces"] == {}


def test_submit_worker_result_rejects_blank_token() -> None:
    tool = _get_submit_worker_result()
    out = _call_tool(tool, result_token="   ", report="r", summary="s")
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert "result_token" in parsed["error"]


def test_submit_worker_result_carries_error_field_through() -> None:
    """The optional ``error`` parameter must round-trip through the
    inbox so the runner can surface it as a chain-step failure even
    when report/summary/produces are populated."""
    tool = _get_submit_worker_result()
    token = result_inbox.mint_token()
    out = _call_tool(
        tool,
        result_token=token,
        report="hit the API, got a 503",
        summary="upstream 503",
        produces={"outcome": "executor_failure"},
        error="upstream returned 503",
    )
    assert json.loads(out) == {"ok": True}
    payload = result_inbox.read_and_consume(token, agent_kind="worker")
    assert payload is not None
    assert payload["error"] == "upstream returned 503"


def test_submit_worker_result_normalizes_blank_error_to_none() -> None:
    """Whitespace-only ``error`` should normalise to ``None`` so the
    runner doesn't have to special-case it. Workers must explicitly
    pass a non-empty string to flip the step to failed."""
    tool = _get_submit_worker_result()
    token = result_inbox.mint_token()
    _call_tool(
        tool,
        result_token=token,
        report="r",
        summary="s",
        error="   ",
    )
    payload = result_inbox.read_and_consume(token, agent_kind="worker")
    assert payload is not None
    assert payload["error"] is None


def test_submit_worker_result_rejects_non_string_error() -> None:
    tool = _get_submit_worker_result()
    out = _call_tool(
        tool,
        result_token=result_inbox.mint_token(),
        report="r",
        summary="s",
        error=42,  # type: ignore[arg-type]
    )
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert "error" in parsed["error"]


# ------------------------------------ submit_coach_result tool


def test_submit_coach_result_writes_inbox_row() -> None:
    tool = _get_tool("submit_coach_result")
    token = result_inbox.mint_token()
    out = _call_tool(tool, result_token=token, thread_post="hello")
    assert json.loads(out) == {"ok": True}
    payload = result_inbox.read_and_consume(token, agent_kind="coach")
    assert payload == {"thread_post": "hello"}


def test_submit_coach_result_accepts_null_thread_post() -> None:
    tool = _get_tool("submit_coach_result")
    token = result_inbox.mint_token()
    out = _call_tool(tool, result_token=token, thread_post=None)
    assert json.loads(out) == {"ok": True}
    payload = result_inbox.read_and_consume(token, agent_kind="coach")
    assert payload == {"thread_post": None}


# ------------------------------------ submit_planner_result tool


def test_submit_planner_result_writes_inbox_row() -> None:
    tool = _get_tool("submit_planner_result")
    token = result_inbox.mint_token()
    out = _call_tool(
        tool,
        result_token=token,
        mutations=[{"op": "abort_chain"}],
    )
    assert json.loads(out) == {"ok": True}
    payload = result_inbox.read_and_consume(token, agent_kind="planner")
    assert payload == {"mutations": [{"op": "abort_chain"}]}


def test_submit_planner_result_defaults_to_empty_list() -> None:
    tool = _get_tool("submit_planner_result")
    token = result_inbox.mint_token()
    _call_tool(tool, result_token=token, mutations=None)
    payload = result_inbox.read_and_consume(token, agent_kind="planner")
    assert payload == {"mutations": []}


# ------------------------------------ submit_aim_chain_plan_result tool


def test_submit_aim_chain_plan_result_writes_inbox_row() -> None:
    tool = _get_tool("submit_aim_chain_plan_result")
    token = result_inbox.mint_token()
    plan = {
        "chain_name": "demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [{"id": "a", "directive": "do A", "tier": "small"}],
    }
    out = _call_tool(tool, result_token=token, plan=plan)
    assert json.loads(out) == {"ok": True}
    payload = result_inbox.read_and_consume(token, agent_kind="aim_chain_plan")
    assert payload == {"plan": plan}


# ------------------------------------ submit_strategist_result tool


def test_submit_strategist_result_writes_inbox_row() -> None:
    tool = _get_tool("submit_strategist_result")
    token = result_inbox.mint_token()
    out = _call_tool(
        tool,
        result_token=token,
        summary="## weekly\nshort.",
        new_aims=[],
        signals=[],
        aim_status_changes=[],
    )
    assert json.loads(out) == {"ok": True}
    payload = result_inbox.read_and_consume(token, agent_kind="strategist")
    assert payload == {
        "summary": "## weekly\nshort.",
        "new_aims": [],
        "signals": [],
        "aim_status_changes": [],
    }


def test_submit_strategist_result_rejects_empty_summary() -> None:
    tool = _get_tool("submit_strategist_result")
    out = _call_tool(
        tool,
        result_token=result_inbox.mint_token(),
        summary="   ",
    )
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert "summary" in parsed["error"]

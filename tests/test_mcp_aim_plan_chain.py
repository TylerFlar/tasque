"""MCP tests for planning an Aim into a Tasque chain spec."""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy import select

from tasque.memory.db import get_session
from tasque.memory.entities import Aim, ChainRun, ChainTemplate, Note, Signal, utc_now_iso
from tasque.memory.repo import write_entity
from tasque.strategist import graph as strategist_graph

from .conftest import make_result_depositing_chat_model


def _call_tool(name: str, **kwargs: object) -> dict[str, Any]:
    from tasque.mcp.server import build_server

    server = build_server()
    tool = server._tool_manager.get_tool(name)
    fn = getattr(tool, "fn", None)
    if fn is None:
        raise RuntimeError(f"expected {name} to be a FastMCP-decorated tool")
    return json.loads(fn(**kwargs))  # type: ignore[operator]


def _seed_aim() -> Aim:
    aim = Aim(
        title="Build a durable tax prep system",
        bucket="finance",
        scope="bucket",
        target_date="2026-10-15",
        description="Make annual tax prep calm and repeatable.",
        source="user",
    )
    written = write_entity(aim)
    write_entity(
        Note(
            content="Tax documents are scattered across email and the cabinet.",
            bucket="finance",
            durability="durable",
            source="user",
        )
    )
    write_entity(
        Signal(
            from_bucket="career",
            to_bucket="finance",
            kind="deadline",
            urgency="normal",
            summary="RSU forms arrive in February",
            body="Include employer equity docs in the prep checklist.",
        )
    )
    return written


def _valid_spec() -> dict[str, Any]:
    return {
        "chain_name": "aim-tax-prep-system",
        "bucket": "finance",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {
                "id": "inventory",
                "kind": "worker",
                "directive": "Inventory existing tax prep documents and gaps.",
                "tier": "small",
            },
            {
                "id": "plan",
                "kind": "worker",
                "directive": "Turn the inventory into a repeatable prep checklist.",
                "depends_on": ["inventory"],
                "consumes": ["inventory"],
                "tier": "medium",
            },
        ],
    }


def _patch_planner_llm(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    fake_llm = make_result_depositing_chat_model(
        agent_kind=strategist_graph.AIM_CHAIN_PLAN_AGENT_KIND,
        payloads=[payload],
    )
    monkeypatch.setattr(
        strategist_graph,
        "get_chat_model_for_tier",
        lambda *args, **kwargs: fake_llm,
    )


def test_aim_plan_chain_returns_validation_errors_without_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aim = _seed_aim()
    bad_spec = _valid_spec()
    del bad_spec["plan"][0]["tier"]
    _patch_planner_llm(monkeypatch, {"plan": bad_spec})

    out = _call_tool("aim_plan_chain", aim_id=aim.id)

    assert out["ok"] is False
    assert out["error"] == "chain spec validation failed"
    assert any("tier" in err for err in out["validation_errors"])
    assert out["spec"]["vars"]["aim_id"] == aim.id
    assert out["spec"]["vars"]["aim_title"] == aim.title
    with get_session() as sess:
        assert list(sess.execute(select(ChainRun)).scalars().all()) == []


def test_aim_plan_chain_queues_valid_adhoc_chain_with_aim_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aim = _seed_aim()
    _patch_planner_llm(monkeypatch, {"plan": _valid_spec()})

    out = _call_tool("aim_plan_chain", aim_id=aim.id, mode="adhoc")

    assert out["ok"] is True
    assert out["mode"] == "adhoc"
    assert out["chain_name"] == "aim-tax-prep-system"
    assert out["spec"]["vars"]["aim_id"] == aim.id
    assert out["spec"]["vars"]["aim_title"] == aim.title

    with get_session() as sess:
        row = sess.execute(select(ChainRun)).scalars().one()
        assert row.chain_id == out["chain_id"]
        assert row.status == "running"
        initial = json.loads(row.initial_state_json or "{}")

    assert initial["vars"]["aim_id"] == aim.id
    assert initial["vars"]["aim_title"] == aim.title


def test_aim_plan_chain_reuses_active_adhoc_chain_for_same_aim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aim = _seed_aim()
    _patch_planner_llm(monkeypatch, {"plan": _valid_spec()})

    first = _call_tool("aim_plan_chain", aim_id=aim.id, mode="adhoc")
    second = _call_tool("aim_plan_chain", aim_id=aim.id, mode="adhoc")

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["deduped"] is True
    assert second["existing_status"] == "running"
    assert second["chain_id"] == first["chain_id"]
    with get_session() as sess:
        rows = list(sess.execute(select(ChainRun)).scalars().all())
    assert len(rows) == 1


def test_aim_plan_chain_reuses_recent_terminal_chain_for_same_aim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aim = _seed_aim()
    _patch_planner_llm(monkeypatch, {"plan": _valid_spec()})

    first = _call_tool("aim_plan_chain", aim_id=aim.id, mode="adhoc")
    with get_session() as sess:
        row = sess.execute(select(ChainRun)).scalars().one()
        row.status = "completed"
        row.ended_at = utc_now_iso()
        row.updated_at = row.ended_at

    second = _call_tool("aim_plan_chain", aim_id=aim.id, mode="adhoc")

    assert second["ok"] is True
    assert second["deduped"] is True
    assert second["existing_status"] == "completed"
    assert second["chain_id"] == first["chain_id"]
    with get_session() as sess:
        rows = list(sess.execute(select(ChainRun)).scalars().all())
    assert len(rows) == 1


def test_aim_plan_chain_template_mode_defaults_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aim = _seed_aim()
    _patch_planner_llm(monkeypatch, {"plan": _valid_spec()})

    out = _call_tool("aim_plan_chain", aim_id=aim.id, mode="template")

    assert out["ok"] is True
    assert out["mode"] == "template"
    assert out["enabled"] is False
    with get_session() as sess:
        row = sess.execute(select(ChainTemplate)).scalars().one()
        assert row.chain_name == "aim-tax-prep-system"
        assert row.enabled is False

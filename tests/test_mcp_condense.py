"""Tests for ``tasque.mcp.condense.maybe_condense``.

The condense layer is best-effort: it must never raise, and it must
preserve the original payload unless every precondition is met (size
over threshold, non-empty intent, small-tier call succeeds, valid envelope
produced). These tests pin those invariants without requiring a live
proxy — the small-tier call is monkeypatched per test.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tasque.mcp import condense


def test_under_threshold_returns_original_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(condense, "DEFAULT_THRESHOLD_BYTES", 1_000)
    monkeypatch.delenv("TASQUE_MCP_CONDENSE_THRESHOLD", raising=False)
    payload = json.dumps({"hello": "world"})
    out = condense.maybe_condense(payload, intent="anything", tool_name="note_get")
    assert out == payload


def test_over_threshold_with_intent_invokes_condenser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = json.dumps({"data": "x" * 70_000})
    captured: dict[str, Any] = {}

    def fake_condense(result_json: str, *, intent: str, tool_name: str) -> str:
        captured["result_json"] = result_json
        captured["intent"] = intent
        captured["tool_name"] = tool_name
        return "[condensed]"

    monkeypatch.setattr(condense, "_condense_via_small", fake_condense)
    monkeypatch.delenv("TASQUE_MCP_CONDENSE_THRESHOLD", raising=False)

    out = condense.maybe_condense(
        big, intent="show me the head of x", tool_name="note_search"
    )
    assert captured["intent"] == "show me the head of x"
    assert captured["tool_name"] == "note_search"

    parsed = json.loads(out)
    assert parsed["_condensed"] is True
    assert parsed["_intent"] == "show me the head of x"
    assert parsed["_tool"] == "note_search"
    assert parsed["summary"] == "[condensed]"
    assert parsed["_original_bytes"] == len(big.encode("utf-8"))


def test_over_threshold_with_blank_intent_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = json.dumps({"data": "x" * 70_000})

    def boom(result_json: str, *, intent: str, tool_name: str) -> str:
        raise AssertionError("should not be called when intent is blank")

    monkeypatch.setattr(condense, "_condense_via_small", boom)
    monkeypatch.delenv("TASQUE_MCP_CONDENSE_THRESHOLD", raising=False)

    out = condense.maybe_condense(big, intent="   ", tool_name="note_search")
    assert out == big


def test_small_tier_failure_returns_original_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = json.dumps({"data": "x" * 70_000})

    def explode(result_json: str, *, intent: str, tool_name: str) -> str:
        raise RuntimeError("proxy unreachable")

    monkeypatch.setattr(condense, "_condense_via_small", explode)
    monkeypatch.delenv("TASQUE_MCP_CONDENSE_THRESHOLD", raising=False)

    out = condense.maybe_condense(big, intent="anything", tool_name="job_list")
    assert out == big


def test_threshold_env_override_lowers_trigger_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TASQUE_MCP_CONDENSE_THRESHOLD", "10")

    monkeypatch.setattr(
        condense,
        "_condense_via_small",
        lambda result_json, *, intent, tool_name: "[tiny-condensed]",
    )

    out = condense.maybe_condense(
        '{"a": "bb"}', intent="something", tool_name="aim_get"
    )
    parsed = json.loads(out)
    assert parsed["_condensed"] is True
    assert parsed["summary"] == "[tiny-condensed]"


def test_threshold_env_override_invalid_value_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TASQUE_MCP_CONDENSE_THRESHOLD", "not-a-number")

    def boom(result_json: str, *, intent: str, tool_name: str) -> str:
        raise AssertionError("default threshold should keep us under")

    monkeypatch.setattr(condense, "_condense_via_small", boom)
    out = condense.maybe_condense(
        json.dumps({"x": "y"}), intent="i", tool_name="note_get"
    )
    assert out == json.dumps({"x": "y"})


def test_threshold_env_override_zero_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TASQUE_MCP_CONDENSE_THRESHOLD", "0")

    def boom(result_json: str, *, intent: str, tool_name: str) -> str:
        raise AssertionError("zero is invalid; default keeps us under threshold")

    monkeypatch.setattr(condense, "_condense_via_small", boom)
    out = condense.maybe_condense('{"x": 1}', intent="i", tool_name="note_get")
    assert out == '{"x": 1}'

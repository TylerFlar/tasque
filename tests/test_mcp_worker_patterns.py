"""Tests for worker-pattern MCP tools."""

from __future__ import annotations

import json
from typing import Any

from tasque.memory.repo import upsert_worker_pattern


def _call_tool(name: str, **kwargs: object) -> dict[str, Any] | list[Any]:
    from tasque.mcp.server import build_server

    server = build_server()
    tool = server._tool_manager.get_tool(name)
    fn = getattr(tool, "fn", None)
    if fn is None:
        raise RuntimeError(f"expected {name} to be a FastMCP-decorated tool")
    return json.loads(fn(**kwargs))  # type: ignore[operator]


def test_worker_pattern_search_returns_keyword_matches() -> None:
    upsert_worker_pattern(
        bucket="finance",
        source_kind="worker",
        key="worker:earnings-scan",
        content=(
            "Directive: scan earnings calendar\n"
            "Produces keys: tickers\n"
            "Summary: Filter by after-hours reports first."
        ),
        tags=["earnings", "tickers"],
        meta={"produces_keys": ["tickers"]},
    )

    out = _call_tool(
        "worker_pattern_search",
        intent="find earnings scan precedent",
        query="earnings tickers",
        bucket="finance",
        limit=5,
    )

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["key"] == "worker:earnings-scan"
    assert out[0]["success_count"] == 1
    assert "after-hours reports" in out[0]["content"]


def test_worker_pattern_search_validates_bucket() -> None:
    out = _call_tool(
        "worker_pattern_search",
        intent="bad bucket",
        query="anything",
        bucket="not-a-bucket",
    )

    assert out["ok"] is False
    assert "bucket must be one of" in out["error"]

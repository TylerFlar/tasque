"""DB-backed inbox for an agent's structured LLM result.

Pattern (replaces fragile post-hoc JSON parsing):

    token = mint_token()
    # ... include token in the LLM prompt; instruct LLM to call
    # ``submit_<agent>_result(result_token=token, ...)`` exactly once ...
    llm.invoke(messages)
    payload = read_and_consume(token, agent_kind="worker")
    if payload is None:
        # the model didn't call the tool — error path

The MCP-side ``submit_<agent>_result`` tool calls :func:`deposit` with
the agent_kind it owns. The token namespace is global; agent_kind is
recorded both for audit and as a defensive read-side filter so a
mistakenly-routed payload from another agent can't poison this read.

Tokens are short-lived. The agent deletes its own row on read; an idle
reaper sweeps stragglers (process crash mid-run, model never called
the tool) based on ``created_at`` age.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, select

from tasque.memory.db import get_session
from tasque.memory.entities import AgentResult, utc_now_iso

DEFAULT_REAP_AGE_SECONDS = 60 * 60  # 1 hour — well past any agent run.


def mint_token() -> str:
    """Generate a fresh result token. Hex UUID, no separators."""
    return uuid4().hex


def deposit(
    *,
    result_token: str,
    agent_kind: str,
    payload: dict[str, Any],
) -> None:
    """Persist ``payload`` so the matching agent run can read it back.

    Idempotent on the token: a second deposit overwrites the first.
    The serializer falls back to ``default=str`` to handle stray
    non-JSON-native values (e.g. datetimes) without dropping the run.
    """
    payload_json = json.dumps(payload, default=str)
    with get_session() as sess:
        existing = sess.get(AgentResult, result_token)
        if existing is not None:
            existing.agent_kind = agent_kind
            existing.payload_json = payload_json
            existing.created_at = utc_now_iso()
            return
        sess.add(
            AgentResult(
                result_token=result_token,
                agent_kind=agent_kind,
                payload_json=payload_json,
            )
        )


def peek(result_token: str, *, agent_kind: str) -> bool:
    """Return True if ``agent_kind`` has deposited under ``result_token``.

    Non-consuming — leaves the row in place so the subsequent
    :func:`read_and_consume` still finds it. Used by the worker runner
    to detect a missed tool call mid-graph and decide whether to retry
    the LLM turn before falling through to the failure path.
    """
    with get_session() as sess:
        row = sess.get(AgentResult, result_token)
        if row is None:
            return False
        return row.agent_kind == agent_kind


def read_and_consume(
    result_token: str, *, agent_kind: str
) -> dict[str, Any] | None:
    """Fetch and delete the row for ``result_token``.

    Returns the parsed payload dict, or ``None`` if no row exists or
    the row was written by a different agent. The row is always
    deleted before returning — even on agent_kind mismatch — so a
    stale entry can't survive subsequent reads.
    """
    with get_session() as sess:
        row = sess.get(AgentResult, result_token)
        if row is None:
            return None
        kind = row.agent_kind
        payload_json = row.payload_json
        sess.delete(row)
    if kind != agent_kind:
        return None
    try:
        parsed = json.loads(payload_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed  # type: ignore[return-value]


def reap_stale(*, max_age_seconds: int = DEFAULT_REAP_AGE_SECONDS) -> int:
    """Delete inbox rows older than ``max_age_seconds`` and return the count.

    Called on an idle schedule; protects against accumulation when an
    agent crashed between minting a token and reading the result.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with get_session() as sess:
        rows = list(
            sess.execute(
                select(AgentResult).where(AgentResult.created_at < cutoff_iso)
            ).scalars().all()
        )
        if not rows:
            return 0
        sess.execute(
            delete(AgentResult).where(AgentResult.created_at < cutoff_iso)
        )
    return len(rows)


__all__ = [
    "DEFAULT_REAP_AGE_SECONDS",
    "deposit",
    "mint_token",
    "peek",
    "read_and_consume",
    "reap_stale",
]

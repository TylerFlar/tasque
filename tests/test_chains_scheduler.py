"""Tests for ``tasque.chains.scheduler._due_templates`` cron polling."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tasque.chains.scheduler import _due_templates
from tasque.config import reset_settings_cache
from tasque.memory.db import get_session
from tasque.memory.entities import ChainTemplate, utc_now_iso


@pytest.fixture(autouse=True)
def _utc_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the project timezone to UTC for these tests so cron math is
    deterministic regardless of whatever the developer's ``.env`` says.
    Resets the settings cache on entry and exit so we don't poison other
    tests that read different env vars (e.g. ``test_cron.py`` which also
    expects UTC)."""
    monkeypatch.setenv("TASQUE_TIMEZONE", "UTC")
    reset_settings_cache()
    yield
    reset_settings_cache()


def _seed_template(
    *,
    name: str,
    recurrence: str | None,
    enabled: bool = True,
    last_fired_at: str | None = None,
    created_at: str | None = None,
) -> str:
    spec: dict[str, Any] = {
        "chain_name": name,
        "bucket": "personal",
        "recurrence": recurrence,
        "planner_tier": "opus",
        "plan": [{"id": "a", "kind": "worker", "directive": "do A", "tier": "haiku"}],
    }
    with get_session() as sess:
        row = ChainTemplate(
            chain_name=name,
            bucket="personal",
            recurrence=recurrence,
            enabled=enabled,
            plan_json=json.dumps(spec),
            last_fired_at=last_fired_at,
        )
        if created_at is not None:
            row.created_at = created_at
            row.updated_at = created_at
        sess.add(row)
        sess.flush()
        return row.id


_NOW = datetime(2026, 4, 26, 19, 0, 0, tzinfo=UTC)  # Sunday 19:00 UTC
_YESTERDAY = (_NOW - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def test_unfired_template_with_past_creation_is_due() -> None:
    """Regression: a never-fired template whose cron should have fired since
    creation must be reported due. The previous implementation anchored at
    ``now`` when ``last_fired_at`` was None, which made ``next_fire_at``
    return a future time and the template would never become due."""
    # Created 24h ago; next 07:00 UTC fire is at 2026-04-26 07:00 UTC,
    # which is before _NOW (19:00) — the template is overdue.
    _seed_template(
        name="daily-7am",
        recurrence="0 7 * * *",
        last_fired_at=None,
        created_at=_YESTERDAY,
    )
    due = _due_templates(now=_NOW)
    assert [t.chain_name for t in due] == ["daily-7am"]


def test_unfired_template_created_after_today_fire_is_not_due() -> None:
    """A template created AFTER today's would-be fire-time should wait for
    the next firing (tomorrow), not fire immediately on the next poll."""
    # Created today at 08:00 UTC, after today's 07:00 UTC fire window.
    created_at = (_NOW - timedelta(hours=11)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    _seed_template(
        name="daily-7am-late-create",
        recurrence="0 7 * * *",
        last_fired_at=None,
        created_at=created_at,
    )
    assert _due_templates(now=_NOW) == []


def test_already_fired_template_uses_last_fired_at_as_anchor() -> None:
    """When ``last_fired_at`` is set, it is the anchor — not creation time."""
    one_minute_ago = (_NOW - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    long_ago = (_NOW - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    _seed_template(
        name="daily-7am-fresh-fire",
        recurrence="0 7 * * *",
        last_fired_at=one_minute_ago,
        created_at=long_ago,
    )
    # Last fire was 1 minute ago; next is tomorrow 07:00 UTC — not due.
    assert _due_templates(now=_NOW) == []


def test_disabled_template_is_skipped() -> None:
    _seed_template(
        name="disabled",
        recurrence="0 7 * * *",
        enabled=False,
        last_fired_at=None,
        created_at=_YESTERDAY,
    )
    assert _due_templates(now=_NOW) == []


def test_template_without_recurrence_is_skipped() -> None:
    _seed_template(
        name="oneshot",
        recurrence=None,
        last_fired_at=None,
        created_at=_YESTERDAY,
    )
    assert _due_templates(now=_NOW) == []


def test_invalid_cron_is_skipped_not_raised() -> None:
    """A bad cron expression on one row must not break the whole poll."""
    # Pure-numeric DOW is rejected by validate_cron — see jobs/cron.py.
    _seed_template(
        name="bad-cron",
        recurrence="0 7 * * 1-5",
        last_fired_at=None,
        created_at=_YESTERDAY,
    )
    _seed_template(
        name="good-cron",
        recurrence="0 7 * * *",
        last_fired_at=None,
        created_at=_YESTERDAY,
    )
    due = _due_templates(now=_NOW)
    assert [t.chain_name for t in due] == ["good-cron"]


def test_default_now_uses_wall_clock() -> None:
    """When ``now`` is omitted, the function should use the current UTC
    time. A template created an hour ago with a ``* * * * *`` cron is
    always due — exercise the default-arg branch."""
    created_at = (datetime.now(UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    _seed_template(
        name="every-minute",
        recurrence="* * * * *",
        last_fired_at=None,
        created_at=created_at,
    )
    due = _due_templates()
    assert [t.chain_name for t in due] == ["every-minute"]


def test_corrupt_anchor_string_is_skipped() -> None:
    """Defensive: a malformed ``created_at`` should skip the row, not raise."""
    _seed_template(
        name="corrupt",
        recurrence="0 7 * * *",
        last_fired_at=None,
        created_at="not-a-real-iso-string",
    )
    assert utc_now_iso()  # sanity
    assert _due_templates(now=_NOW) == []

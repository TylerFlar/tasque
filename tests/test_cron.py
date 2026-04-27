"""Tests for ``tasque.jobs.cron`` — validation + next_fire_at math."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tasque.jobs.cron import next_fire_at, to_iso, validate_cron

# ----------------------------------------------------------- validate_cron

def test_validate_cron_accepts_alias_dow_forms() -> None:
    assert validate_cron("0 9 * * MON-FRI") is None
    assert validate_cron("0 9 * * SUN") is None
    assert validate_cron("0 9 * * mon,wed,fri") is None
    assert validate_cron("0 9 * * *") is None


def test_validate_cron_rejects_pure_numeric_dow() -> None:
    err = validate_cron("0 15 * * 1-5")
    assert err is not None
    assert "MON-FRI" in err or "alias" in err.lower()


def test_validate_cron_rejects_single_digit_dow() -> None:
    err = validate_cron("0 9 * * 0")
    assert err is not None


def test_validate_cron_rejects_comma_numeric_dow() -> None:
    err = validate_cron("0 9 * * 0,6")
    assert err is not None


def test_validate_cron_accepts_step_form_in_other_fields() -> None:
    # */12 in hour position is fine; DOW is "*" so no rejection.
    assert validate_cron("0 */12 * * *") is None


def test_validate_cron_rejects_wrong_field_count() -> None:
    err = validate_cron("0 9 * *")
    assert err is not None
    assert "5" in err


def test_validate_cron_rejects_empty() -> None:
    assert validate_cron("") is not None
    assert validate_cron("   ") is not None


def test_validate_cron_rejects_bogus_field() -> None:
    err = validate_cron("0 99 * * *")  # hour 99 is invalid
    assert err is not None


# ----------------------------------------------------------- next_fire_at

def test_next_fire_at_mon_fri_skips_weekend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASQUE_TIMEZONE", "UTC")
    # A known Saturday (2026-04-25 was a Saturday).
    saturday_noon = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    nxt = next_fire_at("0 9 * * MON-FRI", after=saturday_noon)
    # Next firing should be Monday 2026-04-27 at 09:00 UTC.
    assert nxt.weekday() == 0  # Monday
    assert nxt.hour == 9
    assert nxt.year == 2026 and nxt.month == 4 and nxt.day == 27


def test_next_fire_at_sun_lands_on_sunday(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASQUE_TIMEZONE", "UTC")
    monday = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    nxt = next_fire_at("0 9 * * SUN", after=monday)
    assert nxt.weekday() == 6  # Sunday
    assert nxt.hour == 9


def test_next_fire_at_every_12_hours(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASQUE_TIMEZONE", "UTC")
    base = datetime(2026, 4, 25, 6, 30, tzinfo=UTC)
    nxt = next_fire_at("0 */12 * * *", after=base)
    # Fires at 0:00 and 12:00 of each day in UTC; next after 06:30 is 12:00 same day.
    assert nxt.year == 2026 and nxt.month == 4 and nxt.day == 25
    assert nxt.hour == 12 and nxt.minute == 0


def test_next_fire_at_raises_on_invalid() -> None:
    with pytest.raises(ValueError):
        next_fire_at("0 15 * * 1-5")  # pure-numeric DOW rejected


def test_next_fire_at_is_strictly_after_anchor() -> None:
    """Calling ``next_fire_at(expr, after=fire_time)`` must NOT return
    ``fire_time`` itself — otherwise an enumeration loop never advances.

    The original APScheduler-backed implementation returned at-or-after,
    which broke schedule-enumeration callers.
    """
    # Use a daily-at-noon-UTC cron and pass an anchor exactly at noon.
    base = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    nxt = next_fire_at("0 12 * * *", after=base)
    assert nxt > base, f"expected strictly after {base}, got {nxt}"
    # Specifically: should be the next day's noon, not the same day.
    assert nxt.day == 26
    assert nxt.hour == 12


def test_next_fire_at_loop_advances_through_window() -> None:
    """A loop that feeds the previous result back in as ``after`` should
    enumerate distinct fires, not infinite-loop on the same one."""
    base = datetime(2026, 4, 25, 0, 0, tzinfo=UTC)
    horizon = base + timedelta(days=3)
    cur = base
    fires: list[datetime] = []
    for _ in range(10):
        cur = next_fire_at("0 12 * * *", after=cur)
        if cur > horizon:
            break
        fires.append(cur)
    # 3 distinct fires (one per day) within the 3-day window.
    assert len(fires) == 3
    assert len(set(fires)) == 3


def test_to_iso_round_trips_utc() -> None:
    dt = datetime(2026, 4, 27, 9, 0, tzinfo=UTC)
    s = to_iso(dt)
    assert s.startswith("2026-04-27T09:00:00")
    assert s.endswith("Z")

"""Cron expression parsing wrapped around ``apscheduler.triggers.cron``.

Why this module exists rather than calling APScheduler directly:

APScheduler's ``CronTrigger`` treats day-of-week ``0`` as **Monday**,
whereas standard Unix cron treats ``0`` as **Sunday**. A naive spec like
``0 15 * * 1-5`` typed by a human means "Mon-Fri" in cron-speak but fires
Tue-Sat under APScheduler. Rather than silently accept the off-by-one,
``validate_cron`` rejects pure-numeric day-of-week fields and requires
the alias form (``MON-FRI``, ``SUN``, etc.), where APScheduler's parser
has unambiguous semantics.

``next_fire_at`` returns the next firing time as a UTC-naive ISO 8601
string suitable for storing in ``QueuedJob.fire_at``. The cron expression
is interpreted in the configured ``tasque_timezone``; the result is
converted back to UTC.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from tasque.config import get_settings

# Pure-numeric DOW: any of N, N-N, N,N, */N, N/N where N is a single digit
# (or two digits) and no MON/TUE/.../SUN tokens appear. This rejects the
# common "1-5" / "0-6" / "0,6" forms that humans intend as Unix cron but
# APScheduler parses with Monday=0.
_DOW_HAS_ALPHA = re.compile(r"[A-Za-z]")


def _has_pure_numeric_dow(dow_field: str) -> bool:
    """Return True if the day-of-week field has any digit and no alphabetic tokens.

    ``*``, ``?`` and pure-symbolic forms are fine. ``MON-FRI``, ``mon,fri``
    are fine. ``1-5``, ``0,6``, ``*/2`` (no digits other than the step?) —
    we treat any digit-bearing, no-letter expression as pure-numeric and
    reject it.
    """
    if _DOW_HAS_ALPHA.search(dow_field):
        return False
    return any(ch.isdigit() for ch in dow_field)


def validate_cron(expr: str) -> str | None:
    """Return ``None`` if ``expr`` is valid; otherwise an error message string.

    Accepts the standard 5-field form ``"min hour dom month dow"``. Rejects
    pure-numeric ``dow`` (e.g. ``1-5``, ``0,6``) because APScheduler treats
    ``0`` as Monday rather than Sunday — silent off-by-one bugs.
    """
    if not expr or not expr.strip():
        return "cron expression must be a non-empty string"
    parts = expr.split()
    if len(parts) != 5:
        return f"cron expression must have 5 fields, got {len(parts)}: {expr!r}"
    dow_field = parts[4]
    if _has_pure_numeric_dow(dow_field):
        return (
            f"day-of-week field {dow_field!r} is pure-numeric — APScheduler treats 0 "
            "as Monday, not Sunday. Use the alias form instead (e.g. MON-FRI, SUN, MON,WED)."
        )
    try:
        CronTrigger.from_crontab(expr)
    except (ValueError, KeyError) as exc:
        return f"invalid cron expression {expr!r}: {exc}"
    return None


def _resolve_timezone() -> Any:
    """Return a tzinfo for the configured tasque timezone, falling back to UTC."""
    tz_name = get_settings().tasque_timezone or "UTC"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return UTC


def next_fire_at(expr: str, *, after: datetime | None = None) -> datetime:
    """Return the next firing datetime for ``expr`` strictly after ``after``.

    The cron expression is interpreted in the configured timezone; the
    returned datetime is converted back to UTC. Raises ``ValueError`` if
    the expression is invalid (call :func:`validate_cron` first if you
    want a soft check).

    Note: APScheduler's ``get_next_fire_time(None, now)`` returns the fire
    time AT-or-after ``now`` — so passing exactly a fire-time timestamp
    would return the same timestamp. We bump by 1µs so the result is
    truly strictly-after, matching the docstring contract. Without this
    bump, an enumeration loop ``cur = next_fire_at(..., after=cur)``
    would never advance past a fire instant.
    """
    err = validate_cron(expr)
    if err is not None:
        raise ValueError(err)
    tz = _resolve_timezone()
    trigger = CronTrigger.from_crontab(expr, timezone=tz)
    base = after if after is not None else datetime.now(UTC)
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    base = base + timedelta(microseconds=1)
    base_local = base.astimezone(tz)
    nxt = trigger.get_next_fire_time(None, base_local)
    if nxt is None:
        raise ValueError(f"cron expression {expr!r} has no future firing time")
    return nxt.astimezone(UTC)


def to_iso(dt: datetime) -> str:
    """Format a UTC-aware datetime as the project's ISO-8601 string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

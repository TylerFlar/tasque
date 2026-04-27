"""Ops panel — a snapshot of *actionable* daemon state.

Backs the live panel watcher, the ``/status`` Discord slash command,
and the ``tasque status`` CLI. Everything shown here is something the
user can act on: pending or claimed jobs, active chains, unclaimed
coach triggers, unresolved DLQ entries, scheduler liveness. Terminal
counts (completed / failed / stopped historicals) are deliberately
NOT surfaced — they're audit trail, not a to-do list.

Two layers:

- :func:`build_ops_snapshot` — pure query: returns a ``dict`` shaped for
  JSON serialisation. Used by the CLI directly and by the embed builder.
- :func:`build_ops_embed` — pure transform: stats dict → Discord embed
  dict, ready to hand to the poster.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict, cast
from zoneinfo import ZoneInfo

from sqlalchemy import func, select

from tasque.config import get_settings
from tasque.jobs.cron import next_fire_at, validate_cron
from tasque.memory.db import get_session
from tasque.memory.entities import (
    ChainRun,
    ChainTemplate,
    CoachPending,
    FailedJob,
    QueuedJob,
)

# Plan-node statuses that mean the chain is still actionable. Anything
# else (completed/failed/stopped) is terminal and is intentionally
# omitted from the panel.
_ACTIVE_CHAIN_STATUSES = (
    "running",
    "awaiting_approval",
    "awaiting_user",
    "paused",
)


class OpsSnapshot(TypedDict):
    """The shape :func:`build_ops_snapshot` returns."""

    generated_at: str
    queued_jobs: dict[str, Any]
    coach_queue: dict[str, Any]
    chains: dict[str, Any]
    chain_templates: dict[str, Any]
    dlq: dict[str, Any]
    scheduler: dict[str, Any]
    next_upcoming: dict[str, Any] | None


# ----------------------------------------------------------------- helpers


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _local_tz() -> Any:
    """Return the configured local timezone, falling back to UTC."""
    tz_name = get_settings().tasque_timezone or "UTC"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return UTC


def _format_local(iso_or_dt: str | datetime | None) -> str:
    """Render a UTC ISO string or datetime as a short local-time label
    (e.g. ``"Sun 12:00 PM PDT"``). Falls back to the raw value if parse
    fails so we never crash the panel over a formatting issue."""
    if iso_or_dt is None:
        return "(none)"
    if isinstance(iso_or_dt, datetime):
        dt = iso_or_dt
    else:
        try:
            dt = datetime.strptime(iso_or_dt, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
        except ValueError:
            return iso_or_dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    local = dt.astimezone(_local_tz())
    return local.strftime("%a %m-%d %I:%M %p %Z").replace(" 0", " ")


def _count_by_column(
    cls: type[Any], column: Any, *, where: list[Any] | None = None
) -> dict[str, int]:
    """Return ``{column_value: count}`` for all rows in ``cls`` (optionally
    filtered)."""
    with get_session() as sess:
        stmt = select(column, func.count()).group_by(column)
        for clause in where or []:
            stmt = stmt.where(clause)
        rows = sess.execute(stmt).all()
    out: dict[str, int] = {}
    for value, count in rows:
        key = str(value) if value is not None else "(none)"
        out[key] = int(count)
    return out


# ----------------------------------------------------------------- snapshot


def build_ops_snapshot() -> OpsSnapshot:
    """Query the DB once and return a snapshot of *actionable* daemon state.

    Deliberately omits terminal counts (completed / failed / stopped)
    and recent-window stats — those are historical, not actionable. If
    you want them, query the tables directly.
    """
    now = _utc_now()

    # ------- queued jobs (only pending + claimed; the rest are terminal)
    # Split pending into ready-to-fire vs future-scheduled. A pile of
    # future cron recurrences is NOT something the user can act on; only
    # ``ready`` jobs that haven't been picked up are signal.
    now_iso = _iso(now)
    is_ready = (QueuedJob.fire_at == "now") | (QueuedJob.fire_at <= now_iso)
    with get_session() as sess:
        ready_count = sess.execute(
            select(func.count())
            .select_from(QueuedJob)
            .where(QueuedJob.status == "pending")
            .where(is_ready)
        ).scalar_one()
        scheduled_count = sess.execute(
            select(func.count())
            .select_from(QueuedJob)
            .where(QueuedJob.status == "pending")
            .where(~is_ready)
        ).scalar_one()
        claimed_count = sess.execute(
            select(func.count())
            .select_from(QueuedJob)
            .where(QueuedJob.status == "claimed")
        ).scalar_one()
        # ``func.min`` over String returns ``str | None`` at runtime
        # (None when the filter matches no rows) but the stubs type it as
        # ``str``. Cast so the None-check below isn't flagged unreachable.
        next_scheduled_fire_at = cast(
            "str | None",
            sess.execute(
                select(func.min(QueuedJob.fire_at))
                .where(QueuedJob.status == "pending")
                .where(~is_ready)
            ).scalar_one(),
        )
    ready_by_bucket = _count_by_column(
        QueuedJob,
        QueuedJob.bucket,
        where=[QueuedJob.status == "pending", is_ready],
    )
    with get_session() as sess:
        in_flight_rows = list(
            sess.execute(
                select(QueuedJob)
                .where(QueuedJob.status == "claimed")
                .order_by(QueuedJob.claimed_at.desc())
                .limit(10)
            ).scalars().all()
        )
        in_flight = [
            {
                "id": j.id,
                "bucket": j.bucket,
                "directive": (j.directive or "")[:120],
                "claimed_at": j.claimed_at,
                "last_heartbeat": j.last_heartbeat,
                "chain_id": j.chain_id,
            }
            for j in in_flight_rows
        ]

    queued_jobs: dict[str, Any] = {
        "ready": int(ready_count or 0),
        "claimed": int(claimed_count or 0),
        "scheduled": int(scheduled_count or 0),
        "ready_by_bucket": ready_by_bucket,
        # `next_scheduled_fire_at` retained for any caller wanting a
        # job-only metric; the `next_upcoming` field at the top level
        # folds chain template fires too.
        "next_scheduled_fire_at": next_scheduled_fire_at,
        "in_flight": in_flight,
    }

    # ------- coach trigger queue
    with get_session() as sess:
        unclaimed = sess.execute(
            select(func.count())
            .select_from(CoachPending)
            .where(CoachPending.claimed_at.is_(None))
        ).scalar_one()
        in_flight_coach = sess.execute(
            select(func.count())
            .select_from(CoachPending)
            .where(CoachPending.claimed_at.is_not(None))
        ).scalar_one()
    unclaimed_by_bucket = _count_by_column(
        CoachPending,
        CoachPending.bucket,
        where=[CoachPending.claimed_at.is_(None)],
    )
    coach_queue: dict[str, Any] = {
        "unclaimed": int(unclaimed or 0),
        "in_flight": int(in_flight_coach or 0),
        "unclaimed_by_bucket": unclaimed_by_bucket,
    }

    # ------- chains (active only — terminal runs are audit history)
    with get_session() as sess:
        active_rows = list(
            sess.execute(
                select(ChainRun)
                .where(ChainRun.status.in_(_ACTIVE_CHAIN_STATUSES))
                .order_by(ChainRun.started_at.desc())
                .limit(10)
            ).scalars().all()
        )
        active_chains = [
            {
                "chain_id": r.chain_id,
                "chain_name": r.chain_name,
                "bucket": r.bucket,
                "status": r.status,
                "started_at": r.started_at,
            }
            for r in active_rows
        ]
    by_active_status: dict[str, int] = {}
    for row in active_chains:
        status_str = str(row.get("status"))
        by_active_status[status_str] = by_active_status.get(status_str, 0) + 1
    chains: dict[str, Any] = {
        "active": active_chains,
        "by_status": by_active_status,
    }

    # ------- templates (system-shape sanity check; toggle is actionable)
    with get_session() as sess:
        total_templates = sess.execute(
            select(func.count()).select_from(ChainTemplate)
        ).scalar_one()
        enabled_templates = sess.execute(
            select(func.count())
            .select_from(ChainTemplate)
            .where(ChainTemplate.enabled.is_(True))
        ).scalar_one()
        scheduled_templates = sess.execute(
            select(func.count())
            .select_from(ChainTemplate)
            .where(ChainTemplate.enabled.is_(True))
            .where(ChainTemplate.recurrence.is_not(None))
        ).scalar_one()
    templates_block: dict[str, Any] = {
        "total": int(total_templates or 0),
        "enabled": int(enabled_templates or 0),
        "scheduled": int(scheduled_templates or 0),
    }

    # ------- DLQ (unresolved only)
    with get_session() as sess:
        unresolved = sess.execute(
            select(func.count())
            .select_from(FailedJob)
            .where(FailedJob.resolved.is_(False))
        ).scalar_one()
        recent_unresolved = list(
            sess.execute(
                select(FailedJob)
                .where(FailedJob.resolved.is_(False))
                .order_by(FailedJob.created_at.desc())
                .limit(5)
            ).scalars().all()
        )
        recent_failures = [
            {
                "id": f.id,
                "bucket": f.bucket,
                "error_type": f.error_type,
                "error_message": (f.error_message or "")[:200],
                "retry_count": f.retry_count,
                "created_at": f.created_at,
            }
            for f in recent_unresolved
        ]
    dlq: dict[str, Any] = {
        "unresolved": int(unresolved or 0),
        "recent": recent_failures,
    }

    # ------- scheduler liveness
    with get_session() as sess:
        latest_heartbeat = sess.execute(
            select(func.max(QueuedJob.last_heartbeat))
            .where(QueuedJob.status == "claimed")
        ).scalar_one()
    scheduler: dict[str, Any] = {
        "last_in_flight_heartbeat": latest_heartbeat,
        "queue_depth": int(ready_count or 0),
    }

    # ------- next-upcoming across BOTH pending jobs AND chain templates
    # The Jobs section only sees QueuedJobs, but chain templates
    # materialize into ChainRuns at cron time without going through the
    # QueuedJob table — so the user's "what's the daemon doing next?"
    # has to fold both.
    next_upcoming: dict[str, Any] | None = None
    candidates: list[tuple[datetime, str, str]] = []  # (dt, kind, label)

    if next_scheduled_fire_at is not None:
        try:
            job_dt = datetime.strptime(
                next_scheduled_fire_at, "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=UTC)
            # Find the directive of the earliest scheduled job for the label.
            with get_session() as sess:
                earliest = sess.execute(
                    select(QueuedJob)
                    .where(QueuedJob.status == "pending")
                    .where(~is_ready)
                    .where(QueuedJob.fire_at == next_scheduled_fire_at)
                    .limit(1)
                ).scalars().first()
                label = (
                    (earliest.directive or "").strip().splitlines()[0][:80]
                    if earliest is not None
                    else "(job)"
                )
            candidates.append((job_dt, "job", label))
        except ValueError:
            pass

    with get_session() as sess:
        active_templates = list(
            sess.execute(
                select(ChainTemplate)
                .where(ChainTemplate.enabled.is_(True))
                .where(ChainTemplate.recurrence.is_not(None))
            ).scalars().all()
        )
    for t in active_templates:
        if t.recurrence is None or validate_cron(t.recurrence) is not None:
            continue
        try:
            tnxt = next_fire_at(t.recurrence, after=now)
        except ValueError:
            continue
        candidates.append((tnxt, "chain", t.chain_name))

    if candidates:
        candidates.sort(key=lambda c: c[0])
        nxt_dt, nxt_kind, nxt_label = candidates[0]
        next_upcoming = {
            "at": _iso(nxt_dt),
            "kind": nxt_kind,
            "label": nxt_label,
        }

    return {
        "generated_at": _iso(now),
        "queued_jobs": queued_jobs,
        "coach_queue": coach_queue,
        "chains": chains,
        "chain_templates": templates_block,
        "dlq": dlq,
        "scheduler": scheduler,
        "next_upcoming": next_upcoming,
    }


# ----------------------------------------------------------------- embed


_COLOR_OK = 0x2ECC71
_COLOR_WARN = 0xF1C40F
_COLOR_ALERT = 0xE74C3C


def _summary_color(snapshot: OpsSnapshot) -> int:
    """Red when DLQ has unresolved entries; yellow when something is
    actually in flight or ready-but-unclaimed; green when truly idle.
    Future-scheduled cron recurrences do NOT trigger yellow — they're
    not actionable, the scheduler will pick them up on time."""
    if snapshot["dlq"]["unresolved"] > 0:
        return _COLOR_ALERT
    has_active = (
        snapshot["queued_jobs"]["ready"] > 0
        or snapshot["queued_jobs"]["claimed"] > 0
        or snapshot["coach_queue"]["in_flight"] > 0
        or len(snapshot["chains"]["active"]) > 0
    )
    return _COLOR_WARN if has_active else _COLOR_OK


def _format_qj_block(qj: dict[str, Any]) -> str:
    ready = qj.get("ready", 0)
    claimed = qj.get("claimed", 0)
    scheduled = qj.get("scheduled", 0)
    if ready == 0 and claimed == 0:
        # Truly nothing to act on. Note future-scheduled separately so
        # the user knows the queue isn't *empty*, just quiet. The actual
        # "next at" lives in the dedicated "Next up" field — it folds in
        # chain template fires too, which often beat the next job.
        if scheduled > 0:
            return f"_(idle — {scheduled} scheduled)_"
        return "_(idle)_"
    ready_buckets = qj.get("ready_by_bucket", {})
    bucket_summary = (
        ", ".join(f"{b}:{n}" for b, n in sorted(ready_buckets.items()))
        if ready_buckets
        else "—"
    )
    line1 = f"ready **{ready}** • claimed **{claimed}**"
    line2 = f"by bucket (ready): {bucket_summary}"
    if scheduled > 0:
        line2 += f"  •  scheduled: {scheduled}"
    return f"{line1}\n{line2}"


def _format_next_upcoming(nu: dict[str, Any] | None) -> str:
    if nu is None:
        return "_(nothing scheduled)_"
    when = _format_local(nu.get("at"))
    kind = nu.get("kind", "?")
    label = nu.get("label", "")
    return f"**{when}** — [{kind}] {label}"


def _format_in_flight_block(in_flight: list[dict[str, Any]]) -> str:
    if not in_flight:
        return "_(none)_"
    lines: list[str] = []
    for j in in_flight[:5]:
        bucket = j.get("bucket") or "—"
        directive = (j.get("directive") or "").strip().splitlines()[0][:80]
        chain = f" • chain `{j['chain_id'][:8]}`" if j.get("chain_id") else ""
        lines.append(f"`{j['id'][:8]}` [{bucket}] {directive}{chain}")
    if len(in_flight) > 5:
        lines.append(f"… +{len(in_flight) - 5} more")
    return "\n".join(lines)


def _format_coach_block(coach: dict[str, Any]) -> str:
    if coach["unclaimed"] == 0 and coach["in_flight"] == 0:
        return "_(idle)_"
    by_bucket = coach.get("unclaimed_by_bucket", {})
    bucket_summary = (
        ", ".join(f"{b}:{n}" for b, n in sorted(by_bucket.items()))
        if by_bucket
        else "—"
    )
    return (
        f"unclaimed **{coach['unclaimed']}** • in flight **{coach['in_flight']}**\n"
        f"by bucket (unclaimed): {bucket_summary}"
    )


def _format_active_chains(active: list[dict[str, Any]]) -> str:
    if not active:
        return "_(idle)_"
    lines: list[str] = []
    for c in active[:5]:
        bucket = c.get("bucket") or "—"
        lines.append(
            f"`{c['chain_id'][:8]}` [{bucket}] **{c['chain_name']}** ({c['status']})"
        )
    if len(active) > 5:
        lines.append(f"… +{len(active) - 5} more")
    return "\n".join(lines)


def _format_dlq_block(dlq: dict[str, Any]) -> str:
    if dlq["unresolved"] == 0:
        return "_(none)_"
    lines = [f"unresolved **{dlq['unresolved']}**"]
    for f in dlq.get("recent", [])[:3]:
        bucket = f.get("bucket") or "—"
        msg = (f.get("error_message") or "").strip().splitlines()[0][:80]
        lines.append(f"`{f['id'][:8]}` [{bucket}] {f['error_type']}: {msg}")
    return "\n".join(lines)


def _format_templates_block(t: dict[str, Any]) -> str:
    return (
        f"total **{t['total']}** • enabled **{t['enabled']}** • "
        f"scheduled **{t['scheduled']}**"
    )


def build_ops_embed(snapshot: OpsSnapshot) -> dict[str, Any]:
    """Render an :class:`OpsSnapshot` as a Discord embed dict.

    Sections only appear when they have something actionable to say —
    idle areas collapse to ``_(idle)_`` so they don't dominate the
    panel when the daemon is quiet.
    """
    qj = snapshot["queued_jobs"]
    fields: list[dict[str, Any]] = [
        {
            "name": "Next up",
            "value": _format_next_upcoming(snapshot.get("next_upcoming")),
            "inline": False,
        },
        {
            "name": "Jobs",
            "value": _format_qj_block(qj),
            "inline": False,
        },
        {
            "name": "In flight",
            "value": _format_in_flight_block(cast(list[dict[str, Any]], qj["in_flight"])),
            "inline": False,
        },
        {
            "name": "Coach queue",
            "value": _format_coach_block(snapshot["coach_queue"]),
            "inline": False,
        },
        {
            "name": "Active chains",
            "value": _format_active_chains(
                cast(list[dict[str, Any]], snapshot["chains"]["active"])
            ),
            "inline": False,
        },
        {
            "name": "Templates",
            "value": _format_templates_block(snapshot["chain_templates"]),
            "inline": False,
        },
        {
            "name": "DLQ",
            "value": _format_dlq_block(snapshot["dlq"]),
            "inline": False,
        },
    ]

    sched = snapshot["scheduler"]
    last_hb_raw = sched.get("last_in_flight_heartbeat")
    last_hb = _format_local(last_hb_raw) if last_hb_raw else "_(no in-flight job)_"
    footer_text = (
        f"queue depth {sched['queue_depth']}  •  "
        f"last heartbeat {last_hb}  •  "
        f"snapshot {_format_local(snapshot['generated_at'])}"
    )

    return {
        "title": "tasque ops panel",
        "color": _summary_color(snapshot),
        "fields": fields,
        "footer": {"text": footer_text[:2048]},
    }


__all__ = [
    "OpsSnapshot",
    "build_ops_embed",
    "build_ops_snapshot",
]

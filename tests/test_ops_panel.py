"""Tests for the ops panel — snapshot query + embed builder."""

from __future__ import annotations

from typing import Any

from tasque.discord.ops_panel import build_ops_embed, build_ops_snapshot
from tasque.memory.entities import (
    ChainRun,
    ChainTemplate,
    CoachPending,
    FailedJob,
    QueuedJob,
)
from tasque.memory.repo import write_entity

# ----------------------------------------------------------------- snapshot


def test_snapshot_empty_db_returns_zeros() -> None:
    snap = build_ops_snapshot()
    assert snap["queued_jobs"]["ready"] == 0
    assert snap["queued_jobs"]["claimed"] == 0
    assert snap["queued_jobs"]["scheduled"] == 0
    assert snap["queued_jobs"]["in_flight"] == []
    assert snap["coach_queue"]["unclaimed"] == 0
    assert snap["coach_queue"]["in_flight"] == 0
    assert snap["chains"]["active"] == []
    assert snap["chains"]["by_status"] == {}
    assert snap["chain_templates"]["total"] == 0
    assert snap["dlq"]["unresolved"] == 0
    assert snap["scheduler"]["queue_depth"] == 0


def test_snapshot_splits_pending_into_ready_and_scheduled() -> None:
    """A pending job whose fire_at is in the future is ``scheduled``
    (informational); one with fire_at='now' or in the past is ``ready``
    (the scheduler should pick it up on the next tick)."""
    # Two ready: one with fire_at='now', one with a past timestamp.
    write_entity(QueuedJob(
        kind="worker", bucket="health", directive="d1",
        reason="", fire_at="now", status="pending", queued_by="cli",
    ))
    write_entity(QueuedJob(
        kind="worker", bucket="career", directive="d2",
        reason="", fire_at="2020-01-01T00:00:00.000000Z",
        status="pending", queued_by="cli",
    ))
    # Two scheduled in the future.
    write_entity(QueuedJob(
        kind="worker", bucket="education", directive="d3",
        reason="", fire_at="2099-01-01T00:00:00.000000Z",
        status="pending", queued_by="cli",
    ))
    write_entity(QueuedJob(
        kind="worker", bucket="education", directive="d4",
        reason="", fire_at="2099-06-01T00:00:00.000000Z",
        status="pending", queued_by="cli",
    ))
    # One claimed.
    write_entity(QueuedJob(
        kind="worker", bucket="career", directive="d5",
        reason="", fire_at="now", status="claimed",
        queued_by="cli", claimed_at="2026-04-26T00:00:00.000Z",
    ))

    snap = build_ops_snapshot()

    assert snap["queued_jobs"]["ready"] == 2
    assert snap["queued_jobs"]["scheduled"] == 2
    assert snap["queued_jobs"]["claimed"] == 1
    assert snap["queued_jobs"]["ready_by_bucket"] == {
        "health": 1, "career": 1
    }
    # Earliest future fire_at surfaces.
    assert snap["queued_jobs"]["next_scheduled_fire_at"] == "2099-01-01T00:00:00.000000Z"
    # Queue depth tracks ready, NOT scheduled — those will run later.
    assert snap["scheduler"]["queue_depth"] == 2


def test_snapshot_excludes_terminal_jobs() -> None:
    """Completed/failed/stopped QueuedJobs don't show up anywhere."""
    write_entity(QueuedJob(
        kind="worker", bucket=None, directive="d1",
        reason="", fire_at="now", status="completed", queued_by="cli",
    ))
    write_entity(QueuedJob(
        kind="worker", bucket="finance", directive="d2",
        reason="", fire_at="now", status="failed", queued_by="cli",
    ))
    write_entity(QueuedJob(
        kind="worker", bucket="health", directive="d3",
        reason="", fire_at="now", status="stopped", queued_by="cli",
    ))
    snap = build_ops_snapshot()
    assert snap["queued_jobs"]["ready"] == 0
    assert snap["queued_jobs"]["scheduled"] == 0
    assert snap["queued_jobs"]["claimed"] == 0
    assert snap["queued_jobs"]["in_flight"] == []


def test_snapshot_counts_coach_queue() -> None:
    write_entity(CoachPending(bucket="health", reason="r"))
    write_entity(CoachPending(bucket="career", reason="r"))
    write_entity(CoachPending(
        bucket="finance", reason="r",
        claimed_at="2026-04-26T00:00:00.000Z",
    ))
    snap = build_ops_snapshot()
    assert snap["coach_queue"]["unclaimed"] == 2
    assert snap["coach_queue"]["in_flight"] == 1
    assert snap["coach_queue"]["unclaimed_by_bucket"] == {
        "health": 1, "career": 1
    }


def test_snapshot_only_lists_active_chains() -> None:
    """Terminal ChainRuns (completed/failed/stopped) are deliberately
    omitted — they're history."""
    write_entity(ChainRun(
        chain_id="c1", chain_name="demo-a", bucket="personal",
        status="running", started_at="2026-04-26T00:00:00.000Z",
    ))
    write_entity(ChainRun(
        chain_id="c2", chain_name="demo-b", bucket="health",
        status="awaiting_approval", started_at="2026-04-26T00:00:00.000Z",
    ))
    # These three should NOT show up.
    write_entity(ChainRun(
        chain_id="c3", chain_name="demo-c", bucket="career",
        status="completed", started_at="2026-04-26T00:00:00.000Z",
        ended_at="2026-04-26T00:01:00.000Z",
    ))
    write_entity(ChainRun(
        chain_id="c4", chain_name="demo-d", bucket="creative",
        status="failed", started_at="2026-04-26T00:00:00.000Z",
        ended_at="2026-04-26T00:01:00.000Z",
    ))
    write_entity(ChainRun(
        chain_id="c5", chain_name="demo-e", bucket="home",
        status="stopped", started_at="2026-04-26T00:00:00.000Z",
        ended_at="2026-04-26T00:01:00.000Z",
    ))

    snap = build_ops_snapshot()
    active_ids = {c["chain_id"] for c in snap["chains"]["active"]}
    assert active_ids == {"c1", "c2"}
    # by_status only over the active rows.
    assert snap["chains"]["by_status"] == {
        "running": 1, "awaiting_approval": 1
    }


def test_snapshot_counts_templates() -> None:
    import json as _json

    plan = _json.dumps({
        "chain_name": "t1", "bucket": None, "recurrence": None,
        "planner_tier": "large",
        "plan": [{"id": "a", "kind": "worker", "directive": "x", "tier": "small"}],
    })
    write_entity(ChainTemplate(
        chain_name="t1", bucket=None, recurrence=None,
        enabled=True, plan_json=plan,
    ))
    write_entity(ChainTemplate(
        chain_name="t2", bucket=None, recurrence="0 */6 * * *",
        enabled=True, plan_json=plan,
    ))
    write_entity(ChainTemplate(
        chain_name="t3", bucket=None, recurrence=None,
        enabled=False, plan_json=plan,
    ))
    snap = build_ops_snapshot()
    assert snap["chain_templates"]["total"] == 3
    assert snap["chain_templates"]["enabled"] == 2
    assert snap["chain_templates"]["scheduled"] == 1


def test_snapshot_counts_dlq() -> None:
    write_entity(FailedJob(
        job_id="j1", agent_kind="worker", bucket="health",
        failure_timestamp="2026-04-26T00:00:00.000Z",
        error_type="WorkerError", error_message="boom",
    ))
    write_entity(FailedJob(
        job_id="j2", agent_kind="worker", bucket="career",
        failure_timestamp="2026-04-26T00:00:00.000Z",
        error_type="HeartbeatTimeout", error_message="slow",
        resolved=True,
    ))
    snap = build_ops_snapshot()
    assert snap["dlq"]["unresolved"] == 1
    assert len(snap["dlq"]["recent"]) == 1
    assert snap["dlq"]["recent"][0]["error_type"] == "WorkerError"


# ----------------------------------------------------------------- embed


def test_embed_renders_only_actionable_sections() -> None:
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    assert embed["title"] == "tasque ops panel"
    field_names = {f["name"] for f in embed["fields"]}
    # No "Chains" by-status summary section anymore — collapsed into Active chains.
    assert field_names == {
        "Next up",
        "Jobs",
        "In flight",
        "Coach queue",
        "Active chains",
        "Templates",
        "DLQ",
    }
    assert "footer" in embed
    assert embed["footer"]["text"]


def test_embed_color_warns_when_in_flight() -> None:
    write_entity(QueuedJob(
        kind="worker", bucket="health", directive="d",
        reason="", fire_at="now", status="claimed",
        queued_by="cli", claimed_at="2026-04-26T00:00:00.000Z",
    ))
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    # warn = yellow; ok = green; alert = red
    assert embed["color"] == 0xF1C40F


def test_embed_color_alerts_when_dlq_unresolved() -> None:
    write_entity(FailedJob(
        job_id="j1", agent_kind="worker", bucket="health",
        failure_timestamp="2026-04-26T00:00:00.000Z",
        error_type="WorkerError", error_message="boom",
    ))
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    assert embed["color"] == 0xE74C3C


def test_embed_color_ok_on_idle_db() -> None:
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    assert embed["color"] == 0x2ECC71


def test_embed_color_stays_green_with_only_terminal_state() -> None:
    """Failed/completed/stopped jobs and chains shouldn't paint the
    panel red or yellow — they're not actionable."""
    write_entity(QueuedJob(
        kind="worker", bucket="health", directive="d",
        reason="", fire_at="now", status="failed", queued_by="cli",
    ))
    write_entity(ChainRun(
        chain_id="c1", chain_name="x", bucket=None,
        status="failed", started_at="2026-04-26T00:00:00.000Z",
    ))
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    assert embed["color"] == 0x2ECC71


def test_embed_jobs_section_collapses_to_idle_when_zero() -> None:
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    jobs_field = next(f for f in embed["fields"] if f["name"] == "Jobs")
    assert "_(idle)_" in jobs_field["value"]
    coach_field = next(f for f in embed["fields"] if f["name"] == "Coach queue")
    assert "_(idle)_" in coach_field["value"]


def test_embed_jobs_section_notes_scheduled_count_when_nothing_ready() -> None:
    """If nothing is ready or claimed but there are future-scheduled
    runs, the jobs section says ``N scheduled`` so the user knows the
    queue isn't empty (the actual ``next at`` lives in the dedicated
    Next-up field, which folds chain template fires too)."""
    write_entity(QueuedJob(
        kind="worker", bucket="health", directive="d",
        reason="", fire_at="2099-01-01T00:00:00.000000Z",
        status="pending", queued_by="cli",
    ))
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    jobs_field = next(f for f in embed["fields"] if f["name"] == "Jobs")
    val: str = jobs_field["value"]
    assert "1 scheduled" in val
    # Color must stay green — future cron isn't actionable.
    assert embed["color"] == 0x2ECC71


def test_next_upcoming_is_populated_for_pending_job() -> None:
    """A pending future job should produce a non-null next_upcoming with
    the matching label."""
    write_entity(QueuedJob(
        kind="worker", bucket="health", directive="check sleep tracking",
        reason="", fire_at="2099-01-01T00:00:00.000000Z",
        status="pending", queued_by="cli",
    ))
    snap = build_ops_snapshot()
    nu = snap["next_upcoming"]
    assert nu is not None
    assert nu["kind"] == "job"
    assert nu["at"] == "2099-01-01T00:00:00.000000Z"
    assert "check sleep tracking" in nu["label"]


def test_next_upcoming_includes_chain_template_fires() -> None:
    """If a chain template fires sooner than the next pending job, the
    template should win — that's the whole point of the fold."""
    import json as _json

    # Pending job in the far future.
    write_entity(QueuedJob(
        kind="worker", bucket="health", directive="far-out job",
        reason="", fire_at="2099-01-01T00:00:00.000000Z",
        status="pending", queued_by="cli",
    ))
    # Chain template that fires every minute (will always be sooner).
    plan = _json.dumps({
        "chain_name": "frequent-chain", "bucket": "personal",
        "recurrence": "*/1 * * * *",
        "planner_tier": "large",
        "plan": [{"id": "a", "kind": "worker", "directive": "x", "tier": "small"}],
    })
    write_entity(ChainTemplate(
        chain_name="frequent-chain", bucket="personal",
        recurrence="*/1 * * * *", enabled=True, plan_json=plan,
    ))

    snap = build_ops_snapshot()
    nu = snap["next_upcoming"]
    assert nu is not None
    assert nu["kind"] == "chain"
    assert nu["label"] == "frequent-chain"


def test_next_upcoming_is_none_with_no_schedule() -> None:
    snap = build_ops_snapshot()
    assert snap["next_upcoming"] is None
    embed = build_ops_embed(snap)
    nu_field = next(f for f in embed["fields"] if f["name"] == "Next up")
    assert "nothing scheduled" in nu_field["value"]


def test_embed_renders_local_timestamps_in_footer() -> None:
    """Footer timestamps must be human-readable local time, not raw UTC ISO."""
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    footer = embed["footer"]["text"]
    # No raw ISO timestamps in the footer.
    assert "T" not in footer or "snapshot" in footer  # guard against false positives
    # The snapshot timestamp should look like a local time.
    assert "snapshot" in footer
    # Should NOT contain the literal ISO Z-suffix shape we'd see if
    # someone forgot to format.
    assert ".000000Z" not in footer


def test_embed_color_warns_when_ready_but_not_claimed() -> None:
    """A ready-to-fire job that hasn't been picked up is signal — the
    scheduler may be down or backed up. Yellow."""
    write_entity(QueuedJob(
        kind="worker", bucket="health", directive="d",
        reason="", fire_at="now", status="pending", queued_by="cli",
    ))
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    assert embed["color"] == 0xF1C40F


def test_embed_in_flight_section_lists_jobs() -> None:
    job = write_entity(QueuedJob(
        kind="worker", bucket="health",
        directive="check sleep tracking lately",
        reason="", fire_at="now", status="claimed",
        queued_by="cli", claimed_at="2026-04-26T00:00:00.000Z",
    ))
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    in_flight_field = next(f for f in embed["fields"] if f["name"] == "In flight")
    val: str = in_flight_field["value"]
    assert job.id[:8] in val
    assert "health" in val
    assert "check sleep tracking" in val


def test_embed_in_flight_truncates_to_5_with_overflow_marker() -> None:
    for i in range(7):
        write_entity(QueuedJob(
            kind="worker", bucket="health", directive=f"d{i}",
            reason="", fire_at="now", status="claimed",
            queued_by="cli",
            claimed_at=f"2026-04-26T00:0{i}:00.000Z",
        ))
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    in_flight_field = next(f for f in embed["fields"] if f["name"] == "In flight")
    val: str = in_flight_field["value"]
    # Five lines + overflow marker.
    assert "+" in val and "more" in val


def test_embed_active_chains_section_lists_active_chains() -> None:
    write_entity(ChainRun(
        chain_id="abcdef0123456789", chain_name="approval-demo",
        bucket="personal", status="awaiting_approval",
        started_at="2026-04-26T00:00:00.000Z",
    ))
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    active_field = next(f for f in embed["fields"] if f["name"] == "Active chains")
    val: str = active_field["value"]
    assert "abcdef01" in val
    assert "approval-demo" in val
    assert "awaiting_approval" in val


def test_embed_dlq_section_lists_recent_failures() -> None:
    write_entity(FailedJob(
        job_id="j1", agent_kind="worker", bucket="career",
        failure_timestamp="2026-04-26T00:00:00.000Z",
        error_type="WorkerError",
        error_message="something broke spectacularly",
    ))
    snap = build_ops_snapshot()
    embed = build_ops_embed(snap)
    dlq_field = next(f for f in embed["fields"] if f["name"] == "DLQ")
    val: str = dlq_field["value"]
    assert "WorkerError" in val
    assert "something broke" in val


# ----------------------------------------------------------------- types


def test_snapshot_keys_are_all_jsonable() -> None:
    """Make sure the snapshot can round-trip through json.dumps — the
    CLI relies on this for ``tasque status``."""
    import json as _json

    snap = build_ops_snapshot()
    raw = _json.dumps(snap, default=str)
    again: dict[str, Any] = _json.loads(raw)
    assert "queued_jobs" in again
    assert "generated_at" in again

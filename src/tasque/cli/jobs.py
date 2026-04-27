"""``tasque jobs`` subcommands: queue, list, stop, tick.

``queue`` validates --cron at insert time so a bad expression (including
the pure-numeric DOW trap) fails immediately rather than at firing.
``tick`` drives one scheduler tick from the CLI, useful for end-to-end
testing without keeping ``tasque serve`` up.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated

import typer
from sqlalchemy import select

from tasque.buckets import ALL_BUCKETS, Bucket
from tasque.jobs.cron import next_fire_at, to_iso, validate_cron
from tasque.jobs.scheduler import claim_and_run_one
from tasque.llm.factory import ALL_TIERS
from tasque.memory.db import get_session
from tasque.memory.entities import QueuedJob, utc_now_iso
from tasque.memory.repo import update_entity_status, write_entity

jobs_app = typer.Typer(
    help="Queued-job commands: enqueue, list, stop, manually tick the scheduler.",
    no_args_is_help=True,
    add_completion=False,
)


def _maybe_validate_bucket(value: str | None) -> Bucket | None:
    if value is None:
        return None
    if value not in ALL_BUCKETS:
        valid = ", ".join(ALL_BUCKETS)
        raise typer.BadParameter(f"unknown bucket {value!r}; expected one of: {valid}")
    return value  # type: ignore[return-value]


def _resolve_fire_at(fire_at: str, cron: str | None) -> str:
    """Return the canonical ``fire_at`` string for a queue command.

    For one-shot jobs (no cron), pass the user-supplied value through.
    For recurring jobs, compute the first firing time from the cron.
    """
    if cron is None:
        return fire_at
    nxt = next_fire_at(cron, after=datetime.now(UTC))
    return to_iso(nxt)


@jobs_app.command("queue")
def cmd_queue(
    directive: Annotated[str, typer.Argument(help="What the worker should do.")],
    tier: Annotated[
        str,
        typer.Option(
            "--tier",
            help=(
                "Model tier the worker runs at: 'opus', 'sonnet', or "
                "'haiku'. REQUIRED — every job must declare its tier."
            ),
        ),
    ],
    bucket: Annotated[
        str | None,
        typer.Option("--bucket", "-b", help="Bucket this job belongs to (optional)."),
    ] = None,
    fire_at: Annotated[
        str,
        typer.Option(
            "--fire-at",
            help="When to fire: 'now' or an ISO 8601 timestamp. Ignored if --cron is set.",
        ),
    ] = "now",
    cron: Annotated[
        str | None,
        typer.Option(
            "--cron",
            help=(
                "Recurring schedule, 5-field cron. Pure-numeric day-of-week "
                "(e.g. '1-5') is rejected — use alias form like 'MON-FRI'."
            ),
        ),
    ] = None,
    reason: Annotated[
        str,
        typer.Option("--reason", help="Why this job exists — recorded with the row."),
    ] = "",
    silent: Annotated[
        bool,
        typer.Option("--silent", help="Set visible=False (job won't appear in coach view)."),
    ] = False,
    queued_by: Annotated[
        str,
        typer.Option("--queued-by", help="Who queued this job — defaults to 'cli'."),
    ] = "cli",
) -> None:
    """Insert a new pending QueuedJob."""
    if tier not in ALL_TIERS:
        raise typer.BadParameter(
            f"--tier must be one of {sorted(ALL_TIERS)!r}, got {tier!r}"
        )

    typed_bucket = _maybe_validate_bucket(bucket)

    if cron is not None:
        err = validate_cron(cron)
        if err is not None:
            raise typer.BadParameter(err)

    canonical_fire_at = _resolve_fire_at(fire_at, cron)

    job = QueuedJob(
        kind="worker",
        bucket=typed_bucket,
        directive=directive,
        reason=reason,
        fire_at=canonical_fire_at,
        status="pending",
        recurrence=cron,
        queued_by=queued_by,
        visible=not silent,
        tier=tier,
    )
    written = write_entity(job)
    typer.echo(
        json.dumps(
            {
                "id": written.id,
                "bucket": written.bucket,
                "fire_at": written.fire_at,
                "recurrence": written.recurrence,
                "status": written.status,
                "tier": written.tier,
            }
        )
    )


@jobs_app.command("list")
def cmd_list(
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter by status (pending, claimed, completed, failed, stopped)."),
    ] = None,
    bucket: Annotated[
        str | None,
        typer.Option("--bucket", "-b", help="Filter by bucket."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Max rows to list."),
    ] = 50,
) -> None:
    """List QueuedJobs, newest first."""
    if bucket is not None:
        _maybe_validate_bucket(bucket)
    with get_session() as sess:
        stmt = select(QueuedJob)
        if status is not None:
            stmt = stmt.where(QueuedJob.status == status)
        if bucket is not None:
            stmt = stmt.where(QueuedJob.bucket == bucket)
        stmt = stmt.order_by(QueuedJob.created_at.desc()).limit(limit)
        rows = list(sess.execute(stmt).scalars().all())
        out = [
            {
                "id": r.id,
                "bucket": r.bucket,
                "directive": r.directive,
                "fire_at": r.fire_at,
                "status": r.status,
                "recurrence": r.recurrence,
                "queued_by": r.queued_by,
                "tier": r.tier,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    typer.echo(json.dumps(out, indent=2))


@jobs_app.command("stop")
def cmd_stop(
    job_id: Annotated[str, typer.Argument(help="QueuedJob id to stop.")],
) -> None:
    """Mark a QueuedJob as stopped (terminal — won't fire even if pending)."""
    with get_session() as sess:
        row = sess.get(QueuedJob, job_id)
        if row is None:
            typer.echo(f"no QueuedJob with id={job_id}", err=True)
            raise typer.Exit(code=2)
        if row.status in ("completed", "failed", "stopped"):
            typer.echo(
                f"job {job_id} is already in terminal status {row.status!r}",
                err=True,
            )
            raise typer.Exit(code=1)
    update_entity_status(job_id, "stopped")
    # Touch updated_at — repo helper already does this, but be explicit.
    with get_session() as sess:
        row = sess.get(QueuedJob, job_id)
        if row is not None:
            row.updated_at = utc_now_iso()
    typer.echo(json.dumps({"id": job_id, "status": "stopped"}))


@jobs_app.command("tick")
def cmd_tick() -> None:
    """Run one scheduler tick synchronously. Useful for testing without a daemon."""
    job_id = claim_and_run_one()
    typer.echo(json.dumps({"ran": job_id}))

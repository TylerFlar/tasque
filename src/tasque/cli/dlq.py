"""``tasque dlq`` subcommands: inspect and retry failed jobs."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from tasque.jobs.dlq import get_failure, list_unresolved, mark_resolved, retry

dlq_app = typer.Typer(
    help="Dead-letter queue: list, show, retry, resolve failed worker runs.",
    no_args_is_help=True,
    add_completion=False,
)


@dlq_app.command("list")
def cmd_list(
    limit: Annotated[
        int,
        typer.Option("--limit", help="Max rows to list (newest unresolved first)."),
    ] = 20,
) -> None:
    """List unresolved FailedJobs, newest first."""
    rows = list_unresolved(limit=limit)
    out = [
        {
            "id": r.id,
            "job_id": r.job_id,
            "agent_kind": r.agent_kind,
            "bucket": r.bucket,
            "error_type": r.error_type,
            "error_message": r.error_message,
            "retry_count": r.retry_count,
            "chain_id": r.chain_id,
            "plan_node_id": r.plan_node_id,
            "failure_timestamp": r.failure_timestamp,
        }
        for r in rows
    ]
    typer.echo(json.dumps(out, indent=2))


@dlq_app.command("show")
def cmd_show(failed_id: Annotated[str, typer.Argument(help="FailedJob id to display.")]) -> None:
    """Print one FailedJob in full, including the traceback."""
    fj = get_failure(failed_id)
    if fj is None:
        typer.echo(f"no FailedJob with id={failed_id}", err=True)
        raise typer.Exit(code=2)
    payload = {
        "id": fj.id,
        "job_id": fj.job_id,
        "agent_kind": fj.agent_kind,
        "bucket": fj.bucket,
        "failure_timestamp": fj.failure_timestamp,
        "error_type": fj.error_type,
        "error_message": fj.error_message,
        "traceback": fj.traceback,
        "retry_count": fj.retry_count,
        "original_trigger": fj.original_trigger,
        "resolved": fj.resolved,
        "chain_id": fj.chain_id,
        "plan_node_id": fj.plan_node_id,
    }
    typer.echo(json.dumps(payload, indent=2))


@dlq_app.command("retry")
def cmd_retry(
    failed_id: Annotated[str, typer.Argument(help="FailedJob id to retry.")],
) -> None:
    """Re-fire a FailedJob (insert fresh QueuedJob, or call chain hook)."""
    try:
        report = retry(failed_id)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except NotImplementedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2))


@dlq_app.command("resolve")
def cmd_resolve(
    failed_id: Annotated[str, typer.Argument(help="FailedJob id to mark resolved.")],
) -> None:
    """Mark a FailedJob as resolved (manual closeout)."""
    if not mark_resolved(failed_id):
        typer.echo(f"no FailedJob with id={failed_id}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps({"id": failed_id, "resolved": True}))

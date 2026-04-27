"""``tasque coach`` subcommands."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated

import typer

from tasque.buckets import ALL_BUCKETS, Bucket
from tasque.coach.trigger import drain_until_empty, enqueue

coach_app = typer.Typer(
    help="Bucket-coach commands: enqueue a coach run, drain the queue.",
    no_args_is_help=True,
    add_completion=False,
)


def _validate_bucket(value: str) -> Bucket:
    if value not in ALL_BUCKETS:
        valid = ", ".join(ALL_BUCKETS)
        raise typer.BadParameter(f"unknown bucket {value!r}; expected one of: {valid}")
    return value  # type: ignore[return-value]


@coach_app.command("wake")
def cmd_wake(
    bucket: Annotated[str, typer.Argument(help="Bucket to run the coach for.")],
    reason: Annotated[
        str,
        typer.Option("--reason", help="Why this wake happened — recorded with the row."),
    ] = "manual wake",
) -> None:
    """Enqueue a no-dedup coach run for ``bucket`` and drain until it finishes."""
    typed = _validate_bucket(bucket)
    row_id = enqueue(typed, reason, dedup_key=None)
    if row_id is None:
        # dedup_key=None always enqueues, so this should not happen.
        typer.echo("error: enqueue suppressed unexpectedly", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"enqueued row {row_id} for bucket={bucket}")
    runs = asyncio.run(drain_until_empty())
    typer.echo(json.dumps({"runs": runs}))

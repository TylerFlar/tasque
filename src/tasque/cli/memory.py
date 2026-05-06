"""``tasque memory`` subcommands: import, export, wipe, prune, stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import func, select

from tasque.config import get_settings
from tasque.memory.db import get_session, reset_engine
from tasque.memory.entities import (
    Aim,
    Attachment,
    ChainRun,
    ChainTemplate,
    ContextItem,
    FailedJob,
    Intent,
    Note,
    QueuedJob,
    Signal,
    WorkerPattern,
    WorkItem,
)
from tasque.memory.exporters import export_jsonl
from tasque.memory.importers import import_jsonl, import_markdown_dir
from tasque.memory.repo import sweep_nondurable_memory

memory_app = typer.Typer(
    help="Memory layer: import, export, prune, stats.",
    no_args_is_help=True,
    add_completion=False,
)


@memory_app.command("import")
def cmd_import(
    path: Annotated[Path, typer.Argument(help="JSONL file or directory of .md files.")],
) -> None:
    """Import a JSONL file or a directory of markdown notes."""
    if not path.exists():
        typer.echo(f"path not found: {path}", err=True)
        raise typer.Exit(code=2)
    report = import_markdown_dir(path) if path.is_dir() else import_jsonl(path)
    typer.echo(json.dumps(report, indent=2))


@memory_app.command("export")
def cmd_export(
    path: Annotated[Path, typer.Argument(help="Destination JSONL file.")],
) -> None:
    """Export all entities to a JSONL file."""
    report = export_jsonl(path)
    typer.echo(json.dumps(report, indent=2))


@memory_app.command("wipe")
def cmd_wipe(
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Required confirmation flag — irreversible."),
    ] = False,
) -> None:
    """Delete the SQLite database file. Schema is recreated on next access."""
    if not yes:
        typer.echo("refusing to wipe without --yes", err=True)
        raise typer.Exit(code=2)
    settings = get_settings()
    db = settings.db_path
    reset_engine()
    if db.exists():
        db.unlink()
    typer.echo(f"wiped {db}")


@memory_app.command("prune")
def cmd_prune(
    notes_days: Annotated[
        int | None,
        typer.Option(
            "--notes-days",
            help="Override the ephemeral-Note age cutoff (defaults to settings).",
        ),
    ] = None,
    superseded_days: Annotated[
        int | None,
        typer.Option(
            "--superseded-days",
            help="Override the superseded-Note age cutoff (defaults to settings).",
        ),
    ] = None,
    hard_delete_days: Annotated[
        int | None,
        typer.Option(
            "--hard-delete-days",
            help="If set, hard-delete archived rows older than this many days. "
            "Defaults to settings.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report counts without changing anything."),
    ] = False,
) -> None:
    """Archive decayed lifecycle Notes, expired Signals, superseded Notes,
    and duplicate canonical summaries. Hard-delete long-archived rows when
    configured. Runs against settings defaults unless overridden."""
    report = sweep_nondurable_memory(
        notes_cutoff_days=notes_days,
        superseded_cutoff_days=superseded_days,
        # The function uses ``-1`` as "use settings"; ``None`` means
        # "never hard-delete." Distinguish them so a CLI omission honors
        # the configured default.
        hard_delete_cutoff_days=-1 if hard_delete_days is None else hard_delete_days,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2))


@memory_app.command("stats")
def cmd_stats() -> None:
    """Print row counts per entity type as JSON."""
    counts: dict[str, int] = {}
    classes = (
        Note,
        Aim,
        Signal,
        QueuedJob,
        WorkerPattern,
        FailedJob,
        ChainTemplate,
        ChainRun,
        Attachment,
        Intent,
        ContextItem,
        WorkItem,
    )
    with get_session() as sess:
        for cls in classes:
            n = sess.scalar(select(func.count()).select_from(cls))
            counts[cls.__name__] = int(n or 0)
    typer.echo(json.dumps(counts, indent=2))

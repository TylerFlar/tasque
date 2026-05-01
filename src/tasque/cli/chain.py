"""``tasque chain`` subcommands: reload, export, queue, list, show, pause,
resume, stop, delete, templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import select

from tasque.chains import (
    delete_chain_template,
    export_template_to_yaml,
    get_chain_state,
    get_chain_template,
    launch_chain_run,
    list_chain_templates,
    pause_chain,
    reload_templates,
    render_plan_tree,
    resume_chain,
    stop_chain,
)
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun

chain_app = typer.Typer(
    help="Chain commands: reload templates, queue runs, manage live chains.",
    no_args_is_help=True,
    add_completion=False,
)


@chain_app.command("reload")
def cmd_reload(
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            help="Override the templates directory (default: chains/templates).",
        ),
    ] = None,
) -> None:
    """Scan ``chains/templates/*.yaml`` and upsert each into the DB."""
    report = reload_templates(templates_dir=path)
    typer.echo(json.dumps(report, indent=2))


@chain_app.command("export")
def cmd_export(
    name: Annotated[str, typer.Argument(help="Chain template name to export.")],
    path: Annotated[
        Path | None,
        typer.Argument(help="Optional override path; defaults to the row's seed_path."),
    ] = None,
) -> None:
    """Write a ChainTemplate row's current state back to YAML."""
    try:
        out = export_template_to_yaml(name, path)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps({"name": name, "path": out.as_posix()}))


@chain_app.command("queue")
def cmd_queue(
    name_or_path: Annotated[
        str,
        typer.Argument(
            help="Template name (looked up by chain_name) or a JSON-spec file path."
        ),
    ],
    vars_json: Annotated[
        str | None,
        typer.Option(
            "--vars",
            help=(
                "JSON object of run-time overrides merged over the spec's "
                "static `vars` (caller wins on key collision). Workers see "
                "this dict in their prompt; directives can branch on it "
                "(e.g. --vars '{\"force\": true}' to skip an age gate)."
            ),
        ),
    ] = None,
) -> None:
    """One-shot launch — by template name or by ad-hoc JSON spec file."""
    spec: dict[str, object]
    candidate_path = Path(name_or_path)
    if candidate_path.exists() and candidate_path.is_file():
        spec_raw = json.loads(candidate_path.read_text(encoding="utf-8"))
        if not isinstance(spec_raw, dict):
            typer.echo(
                f"spec at {candidate_path} did not parse to an object", err=True
            )
            raise typer.Exit(code=2)
        spec = spec_raw
        template_id: str | None = None
    else:
        row = get_chain_template(name_or_path)
        if row is None:
            typer.echo(f"no chain template named {name_or_path!r}", err=True)
            raise typer.Exit(code=2)
        plan = row["plan"]
        if not isinstance(plan, dict):
            typer.echo(
                f"chain template {name_or_path!r} has malformed plan_json",
                err=True,
            )
            raise typer.Exit(code=2)
        spec = plan
        template_id = row.get("id") if isinstance(row.get("id"), str) else None

    runtime_vars: dict[str, object] | None = None
    if vars_json is not None:
        try:
            parsed_vars = json.loads(vars_json)
        except json.JSONDecodeError as exc:
            typer.echo(f"--vars did not parse as JSON: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        if not isinstance(parsed_vars, dict):
            typer.echo("--vars must encode a JSON object", err=True)
            raise typer.Exit(code=2)
        runtime_vars = parsed_vars

    chain_id = launch_chain_run(
        spec,
        template_id=template_id,
        vars=runtime_vars,
        wait=False,
    )
    typer.echo(json.dumps({"chain_id": chain_id}))


@chain_app.command("list")
def cmd_list(
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter by ChainRun status."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows.")] = 50,
) -> None:
    """List ChainRuns, newest first."""
    with get_session() as sess:
        stmt = select(ChainRun)
        if status is not None:
            stmt = stmt.where(ChainRun.status == status)
        stmt = stmt.order_by(ChainRun.created_at.desc()).limit(limit)
        rows = list(sess.execute(stmt).scalars().all())
        out = [
            {
                "chain_id": r.chain_id,
                "chain_name": r.chain_name,
                "bucket": r.bucket,
                "status": r.status,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
            }
            for r in rows
        ]
    typer.echo(json.dumps(out, indent=2))


@chain_app.command("show")
def cmd_show(
    chain_id: Annotated[str, typer.Argument(help="ChainRun chain_id.")],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help=(
                "Include per-step summaries, failure reasons, and a "
                "fan-out outcome roll-up under each fan-out template. "
                "Use this to forensically diagnose chain runs whose "
                "user-facing status looks green but whose leaf workers "
                "actually failed silently."
            ),
        ),
    ] = False,
) -> None:
    """Render the chain's plan tree from the live checkpoint."""
    state = get_chain_state(chain_id)
    if state is None:
        typer.echo(f"no checkpoint for chain {chain_id!r}", err=True)
        raise typer.Exit(code=2)
    typer.echo(render_plan_tree(chain_id, verbose=verbose))


@chain_app.command("templates")
def cmd_templates(
    enabled_only: Annotated[
        bool,
        typer.Option("--enabled-only", help="Filter to enabled templates only."),
    ] = False,
) -> None:
    """List ChainTemplate rows."""
    rows = list_chain_templates(enabled_only=enabled_only)
    out = [
        {
            "chain_name": r["chain_name"],
            "bucket": r["bucket"],
            "recurrence": r["recurrence"],
            "enabled": r["enabled"],
            "last_fired_at": r["last_fired_at"],
            "seed_path": r["seed_path"],
        }
        for r in rows
    ]
    typer.echo(json.dumps(out, indent=2))


@chain_app.command("pause")
def cmd_pause(
    chain_id: Annotated[str, typer.Argument(help="ChainRun chain_id.")],
) -> None:
    """Mark a ChainRun ``paused`` (skipped on restart)."""
    if not pause_chain(chain_id):
        typer.echo(f"no chain run {chain_id!r}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps({"chain_id": chain_id, "status": "paused"}))


@chain_app.command("resume")
def cmd_resume(
    chain_id: Annotated[str, typer.Argument(help="ChainRun chain_id.")],
) -> None:
    """Flip a paused ChainRun back to ``running``."""
    if not resume_chain(chain_id):
        typer.echo(f"no chain run {chain_id!r}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps({"chain_id": chain_id, "status": "running"}))


@chain_app.command("stop")
def cmd_stop(
    chain_id: Annotated[str, typer.Argument(help="ChainRun chain_id.")],
) -> None:
    """Stop a chain run AND mark its non-terminal plan nodes ``stopped``."""
    if not stop_chain(chain_id):
        typer.echo(f"no chain run {chain_id!r}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps({"chain_id": chain_id, "status": "stopped"}))


@chain_app.command("delete")
def cmd_delete(
    name: Annotated[str, typer.Argument(help="Chain template name.")],
) -> None:
    """Hard-delete a ChainTemplate. ChainRuns referencing it have their
    ``template_id`` nulled out (history preserved)."""
    if not delete_chain_template(name):
        typer.echo(f"no chain template named {name!r}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps({"name": name, "deleted": True}))

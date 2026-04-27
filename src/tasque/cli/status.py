"""``tasque status`` — print the ops snapshot.

Same data the ``/status`` Discord slash command shows, but rendered as
JSON for piping into ``jq`` or similar tools. Pass ``--text`` for a
human-readable rendering.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from tasque.discord.ops_panel import build_ops_embed, build_ops_snapshot

status_app = typer.Typer(
    help="Snapshot of daemon state: jobs, chains, DLQ, coach queue, scheduler.",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)


_ASCII_SUBS = {
    "**": "",
    "`": "",
    "•": "-",
    "✓": "OK",
    "✗": "X",
    "…": "...",
    "—": "-",
}


def _to_ascii(value: str) -> str:
    """Strip Discord markdown and Unicode glyphs the embed uses so the
    output renders on Windows' default cp1252 console too."""
    out = value
    for src, dst in _ASCII_SUBS.items():
        out = out.replace(src, dst)
    return out


def _render_text(snapshot: dict[str, object]) -> str:
    """Flatten the embed dict back into plain lines for terminal display."""
    embed = build_ops_embed(snapshot)  # type: ignore[arg-type]
    lines: list[str] = [str(embed.get("title") or "tasque ops panel"), ""]
    for field in embed.get("fields", []):  # type: ignore[union-attr]
        if not isinstance(field, dict):
            continue
        name = field.get("name", "")
        value = _to_ascii(str(field.get("value", "") or ""))
        lines.append(f"== {name} ==")
        lines.append(value)
        lines.append("")
    footer = embed.get("footer")
    if isinstance(footer, dict) and footer.get("text"):
        lines.append(_to_ascii(str(footer["text"])))
    return "\n".join(lines)


@status_app.callback()
def cmd_status(
    text: Annotated[
        bool,
        typer.Option(
            "--text", help="Render as human-readable text instead of JSON."
        ),
    ] = False,
) -> None:
    """Print the daemon's current ops snapshot."""
    snapshot = build_ops_snapshot()
    if text:
        typer.echo(_render_text(dict(snapshot)))
    else:
        typer.echo(json.dumps(snapshot, indent=2, default=str))


__all__ = ["status_app"]

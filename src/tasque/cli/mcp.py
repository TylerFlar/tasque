"""``tasque mcp`` — run the tasque stdio MCP server.

Registered in the user's ``~/.claude.json`` so every ``claude --print``
invocation routed through the tasque proxy inherits the tasque tool
catalog. Spawned per request by the host (claude CLI), so this command
stays alive only for the duration of one MCP session.

Example registration in ``~/.claude.json`` (under ``mcpServers``):

    "tasque": {
        "command": "tasque",
        "args": ["mcp"],
        "env": {}
    }
"""

from __future__ import annotations

import typer

from tasque.mcp.server import run_stdio

mcp_app = typer.Typer(
    help="Run the tasque stdio MCP server (registered in ~/.claude.json).",
    invoke_without_command=True,
    add_completion=False,
)


@mcp_app.callback()
def cmd_mcp() -> None:
    """Start the stdio MCP server. Blocks until the parent closes the pipe."""
    run_stdio()

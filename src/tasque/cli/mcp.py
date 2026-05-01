"""``tasque mcp`` — run the tasque stdio MCP server.

Registered in the user's host MCP config so every upstream invocation
routed through the tasque proxy inherits the tasque tool catalog.
Spawned per request by the host CLI, so this command stays alive only
for the duration of one MCP session.

Example registration under ``mcpServers``:

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
    help="Run the tasque stdio MCP server (registered in your MCP host).",
    invoke_without_command=True,
    add_completion=False,
)


@mcp_app.callback()
def cmd_mcp() -> None:
    """Start the stdio MCP server. Blocks until the parent closes the pipe."""
    run_stdio()

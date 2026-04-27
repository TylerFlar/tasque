"""``tasque`` CLI entry point. Wires together all subcommand groups."""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from tasque.cli.chain import chain_app
from tasque.cli.coach import coach_app
from tasque.cli.dlq import dlq_app
from tasque.cli.jobs import jobs_app
from tasque.cli.mcp import mcp_app
from tasque.cli.memory import memory_app
from tasque.cli.proxy import proxy_app
from tasque.cli.serve import serve_app
from tasque.cli.status import status_app
from tasque.logging_setup import configure_logging

# pydantic-settings only loads .env into Settings; os.environ.get() call
# sites (bot token, channel ids, model overrides) miss it without this.
load_dotenv()

# All ``tasque <subcommand>`` invocations share one logging config —
# coloured console renderer when stderr is a TTY, key=value otherwise.
# Override the default level with ``TASQUE_LOG_LEVEL=DEBUG`` etc.
configure_logging()

app = typer.Typer(
    help="tasque — single-user task-orchestration daemon.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(memory_app, name="memory", help="Memory layer: import, export, prune, stats.")
app.add_typer(proxy_app, name="proxy", help="Start the host-side LLM proxy.")
app.add_typer(mcp_app, name="mcp", help="Run the tasque stdio MCP server.")
app.add_typer(coach_app, name="coach", help="Bucket-coach trigger queue + manual wake.")
app.add_typer(jobs_app, name="jobs", help="Queued-job lifecycle: queue, list, stop, tick.")
app.add_typer(dlq_app, name="dlq", help="Dead-letter queue: list, show, retry, resolve.")
app.add_typer(chain_app, name="chain", help="Chain templates + live runs: reload, queue, pause, stop.")
app.add_typer(serve_app, name="serve", help="Run the tasque daemon (bot + scheduler + drainer).")
app.add_typer(status_app, name="status", help="Snapshot of daemon state: jobs, chains, DLQ, scheduler.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

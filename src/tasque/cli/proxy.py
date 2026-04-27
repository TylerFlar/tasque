"""``tasque proxy`` — start the host-side LLM proxy."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tasque.proxy.server import DEFAULT_HOST, DEFAULT_PORT, serve

proxy_app = typer.Typer(
    help="Start the OpenAI-compat proxy that wraps `claude --print`.",
    invoke_without_command=True,
    add_completion=False,
)


@proxy_app.callback()
def cmd_proxy(
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option("--port", help="TCP port.")] = DEFAULT_PORT,
    max_concurrent: Annotated[
        int | None,
        typer.Option("--max-concurrent", help="Cap on in-flight claude --print calls."),
    ] = None,
    log_dir: Annotated[
        Path | None,
        typer.Option("--log-dir", help="Directory for per-request stream-json logs."),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="Per-request wall-clock cap (seconds)."),
    ] = None,
) -> None:
    """Start the proxy. Blocks until interrupted."""
    serve(
        host=host,
        port=port,
        max_concurrent=max_concurrent,
        log_dir=log_dir,
        timeout=timeout,
    )

"""Centralised logging configuration for the daemon and the CLI.

Calling :func:`configure_logging` once at process start hooks structlog
into stdlib ``logging`` with a coloured ConsoleRenderer, sets the root
log level, and silences the third-party loggers that spam INFO at us
(nextcord, httpx, urllib3, asyncio cancellation chatter, APScheduler's
"job skipped" warnings).

The CLI wires this in :func:`tasque.cli.__main__.main` so every
subcommand (``serve``, ``proxy``, ``coach wake``, …) gets the same
output format. Tests don't call it; they leave structlog at its default
"capturing" config so log assertions stay deterministic.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog

_DEFAULT_LEVEL = "INFO"

# Loggers we never want at INFO. They're all third-party plumbing
# whose INFO messages drown out the daemon's own events.
_SILENCED_LOGGERS: tuple[tuple[str, int], ...] = (
    # APScheduler emits a WARNING every tick when a coalesced job is
    # skipped (which is normal under our serial-execution lock).
    ("apscheduler.scheduler", logging.ERROR),
    ("apscheduler.executors.default", logging.WARNING),
    # nextcord HTTP and gateway chatter — keep WARNING so reconnects
    # surface, but kill the per-call INFO traces.
    ("nextcord", logging.WARNING),
    ("nextcord.client", logging.WARNING),
    ("nextcord.gateway", logging.WARNING),
    ("nextcord.http", logging.WARNING),
    # HTTP client libraries fire INFO on every request/response. Way
    # too noisy when the bot is doing dozens of REST calls per minute.
    ("httpx", logging.WARNING),
    ("httpcore", logging.WARNING),
    ("urllib3", logging.WARNING),
    # asyncio is otherwise fine, but emits noisy "task was destroyed
    # but it is pending!" tracebacks during cooperative shutdown.
    ("asyncio", logging.WARNING),
)


def _resolve_level(default: str) -> int:
    """``TASQUE_LOG_LEVEL`` env var wins; otherwise the caller's default."""
    raw = os.environ.get("TASQUE_LOG_LEVEL", default).upper().strip()
    return getattr(logging, raw, logging.INFO)


def configure_logging(*, level: str = _DEFAULT_LEVEL, force: bool = False) -> None:
    """Wire structlog into stdlib logging with a coloured console renderer.

    Idempotent — calling twice is safe but the second call is a no-op
    unless ``force=True``. ``level`` is the floor for the root logger
    (overridden by ``TASQUE_LOG_LEVEL`` if set).
    """
    if not force and getattr(configure_logging, "_done", False):
        return

    log_level = _resolve_level(level)

    # stdlib root: emit to stderr at the chosen level. We deliberately
    # do NOT call ``logging.basicConfig(force=True)`` more than once;
    # the handler list below is replaced in-place so re-runs (e.g. in
    # tests) don't accumulate duplicate handlers.
    root = logging.getLogger()
    root.setLevel(log_level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)

    for name, lvl in _SILENCED_LOGGERS:
        logging.getLogger(name).setLevel(lvl)

    # structlog: render with colors when stderr is a TTY, fall back to
    # plain key=value on file/pipe.
    is_tty = sys.stderr.isatty()
    renderer: structlog.types.Processor
    if is_tty:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        renderer = structlog.processors.KeyValueRenderer(
            key_order=["timestamp", "level", "event"],
            drop_missing=True,
        )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    configure_logging._done = True


__all__ = ["configure_logging"]

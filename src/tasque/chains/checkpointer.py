"""Singleton ``SqliteSaver`` for chain LangGraph checkpoints.

The contract called for ``AsyncSqliteSaver``; the rest of tasque (APScheduler
ticks, CLI commands) runs synchronously. We use the sync ``SqliteSaver`` so
chain invocation can stay sync without spinning an event loop per call;
``langgraph-checkpoint-sqlite`` ships both variants from the same package
and they share the on-disk schema.

The checkpoint database lives next to the main tasque DB
(``<data_dir>/tasque-chains.db``) so a wipe of one doesn't accidentally
take the other with it. WAL is enabled on first open so reads from the
CLI don't block the supervisor and vice-versa.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from threading import Lock

from langgraph.checkpoint.sqlite import SqliteSaver

from tasque.config import get_settings

_saver: SqliteSaver | None = None
_conn: sqlite3.Connection | None = None
_lock = Lock()


def _checkpoint_db_path() -> Path:
    settings = get_settings()
    return settings.db_path.parent / "tasque-chains.db"


def get_chain_checkpointer() -> SqliteSaver:
    """Return the process-wide ``SqliteSaver``, creating it on first call.

    Enables WAL + ``synchronous=NORMAL`` for concurrent readers, and uses
    ``check_same_thread=False`` so APScheduler's worker thread and the
    Typer CLI process can share the connection within a single tasque
    process.
    """
    global _saver, _conn
    if _saver is not None:
        return _saver
    with _lock:
        if _saver is None:
            path = _checkpoint_db_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                path.as_posix(),
                check_same_thread=False,
                isolation_level=None,
            )
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            saver = SqliteSaver(conn)
            saver.setup()
            _conn = conn
            _saver = saver
    assert _saver is not None
    return _saver


def reset_chain_checkpointer() -> None:
    """Close and drop the cached saver. Tests use this between runs."""
    global _saver, _conn
    if _conn is not None:
        with contextlib.suppress(sqlite3.Error):
            _conn.close()
    _saver = None
    _conn = None


def set_chain_checkpointer(saver: SqliteSaver) -> None:
    """Override the process-wide saver — primarily for tests using
    ``SqliteSaver(sqlite3.connect(":memory:"))``."""
    global _saver
    _saver = saver


__all__ = [
    "get_chain_checkpointer",
    "reset_chain_checkpointer",
    "set_chain_checkpointer",
]

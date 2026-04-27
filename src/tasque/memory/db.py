"""SQLite engine + sessionmaker, plus the idempotent ``_ensure_schema`` bootstrap.

Anything outside ``tasque.memory`` should not import SQLAlchemy. Use the
``tasque.memory.repo`` facade instead — that is the seam that lets us
evolve schema cheaply.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock
from typing import cast

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from tasque.config import get_settings
from tasque.memory.entities import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None
_lock = Lock()


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first call."""
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is None:
            settings = get_settings()
            settings.db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{settings.db_path.as_posix()}"
            engine = create_engine(
                url,
                future=True,
                echo=False,
                connect_args={"check_same_thread": False},
            )
            _ensure_schema(engine)
            _engine = engine
            _SessionLocal = sessionmaker(
                bind=engine,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
                future=True,
            )
    assert _engine is not None
    return _engine


def set_engine(engine: Engine) -> None:
    """Override the process-wide engine — primarily for tests."""
    global _engine, _SessionLocal
    _engine = engine
    _SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    _ensure_schema(engine)


def reset_engine() -> None:
    """Drop the cached engine. Closes connections so files can be removed."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a Session; commit on clean exit, rollback on exception."""
    if _SessionLocal is None:
        get_engine()
    factory = cast(sessionmaker[Session], _SessionLocal)
    sess = factory()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def _ensure_schema(engine: Engine) -> None:
    """Create tables and ALTER any missing columns into existing tables.

    Idempotent. This is the only schema-bootstrap path. Type changes and
    column removals require export → wipe → import, by design.

    Also creates the FTS5 virtual table ``notes_fts`` and the triggers that
    keep it in sync with ``notes``. Existing rows from before FTS5 was added
    are NOT auto-backfilled — see ``migration/backfill_notes_fts.py``.
    """
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.tables.values():
            try:
                existing = {c["name"] for c in insp.get_columns(table.name)}
            except Exception:
                continue
            for col in table.columns:
                if col.name in existing:
                    continue
                col_type_sql = col.type.compile(dialect=engine.dialect)
                # Add as nullable to avoid SQLite ADD COLUMN NOT NULL pain;
                # Python defaults populate the value on subsequent writes.
                conn.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type_sql}')
                )
    _ensure_notes_fts(engine)


def _ensure_notes_fts(engine: Engine) -> None:
    """Create the ``notes_fts`` FTS5 virtual table and its sync triggers.

    Skips silently if the SQLite build doesn't have FTS5 (rare on modern
    Python distributions but possible). Idempotent — uses ``IF NOT EXISTS``
    on every DDL statement.
    """
    with engine.begin() as conn:
        try:
            conn.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5("
                    "note_id UNINDEXED, "
                    "content, "
                    "bucket UNINDEXED, "
                    "durability UNINDEXED, "
                    "archived UNINDEXED, "
                    "tokenize = 'porter unicode61'"
                    ")"
                )
            )
        except Exception:
            # FTS5 unavailable. Search-by-substring still works via the
            # existing tools; FTS-specific tools will return an error.
            return
        conn.execute(
            text(
                "CREATE TRIGGER IF NOT EXISTS notes_fts_ai AFTER INSERT ON notes BEGIN "
                "INSERT INTO notes_fts(note_id, content, bucket, durability, archived) "
                "VALUES (new.id, new.content, new.bucket, new.durability, new.archived); "
                "END"
            )
        )
        conn.execute(
            text(
                "CREATE TRIGGER IF NOT EXISTS notes_fts_ad AFTER DELETE ON notes BEGIN "
                "DELETE FROM notes_fts WHERE note_id = old.id; "
                "END"
            )
        )
        conn.execute(
            text(
                "CREATE TRIGGER IF NOT EXISTS notes_fts_au AFTER UPDATE ON notes BEGIN "
                "DELETE FROM notes_fts WHERE note_id = old.id; "
                "INSERT INTO notes_fts(note_id, content, bucket, durability, archived) "
                "VALUES (new.id, new.content, new.bucket, new.durability, new.archived); "
                "END"
            )
        )

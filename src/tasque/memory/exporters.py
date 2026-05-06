"""JSONL exporter — the inverse of ``importers.import_jsonl``.

Round-trip JSONL → import → export is an identity, modulo
``created_at`` / ``updated_at`` for newly created rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from tasque.memory.db import get_session
from tasque.memory.entities import (
    Aim,
    Attachment,
    Base,
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

# Per-class Python-attribute → JSON-key renames (inverse of importer).
_FIELD_RENAMES: dict[str, dict[str, str]] = {
    "Note": {"meta": "metadata"},
    "WorkerPattern": {"meta": "metadata"},
    "Intent": {"meta": "metadata"},
    "ContextItem": {"meta": "metadata"},
    "WorkItem": {"meta": "metadata"},
}

# Order entities are written in — keep stable so test assertions are stable.
_EXPORT_ORDER: tuple[type[Base], ...] = (
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


def _to_dict(obj: Base) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prop in obj.__mapper__.column_attrs:
        result[prop.key] = getattr(obj, prop.key)
    type_name = type(obj).__name__
    renames = _FIELD_RENAMES.get(type_name, {})
    for attr_key, json_key in renames.items():
        if attr_key in result:
            result[json_key] = result.pop(attr_key)
    return result


def export_jsonl(path: Path) -> dict[str, Any]:
    """Write all entities to ``path`` as JSONL. Returns a counts report."""
    counts: dict[str, int] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f, get_session() as sess:
        for cls in _EXPORT_ORDER:
            rows = sess.execute(select(cls)).scalars().all()
            for r in rows:
                d = _to_dict(r)
                d = {"type": cls.__name__, **d}
                f.write(json.dumps(d, ensure_ascii=False, sort_keys=True))
                f.write("\n")
                counts[cls.__name__] = counts.get(cls.__name__, 0) + 1
    return {"counts": counts}

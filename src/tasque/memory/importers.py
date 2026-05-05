"""Source-agnostic importers for the memory layer.

JSONL: one JSON object per line, ``{"type": "<EntityName>", ...fields...}``.
Markdown: each ``.md`` file becomes a durable Note; if the file lives in a
bucket-named subdirectory, that bucket is recorded.

These functions know nothing about any upstream system. Adapters live
upstream of this layer, not inside it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tasque.buckets import ALL_BUCKETS
from tasque.memory.db import get_session
from tasque.memory.entities import ENTITY_BY_NAME, Note

# Per-class JSON-key → Python-attribute renames. Currently only Note
# (because ``metadata`` is reserved on SQLAlchemy DeclarativeBase).
_FIELD_RENAMES: dict[str, dict[str, str]] = {
    "Note": {"metadata": "meta"},
    "WorkerPattern": {"metadata": "meta"},
}


def _rename_in(type_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    renames = _FIELD_RENAMES.get(type_name, {})
    if not renames:
        return payload
    out = dict(payload)
    for json_key, attr_key in renames.items():
        if json_key in out:
            out[attr_key] = out.pop(json_key)
    return out


def _stable_id_from_path(rel: Path) -> str:
    norm = str(rel).replace("\\", "/")
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def import_jsonl(path: Path) -> dict[str, Any]:
    """Import entities from a JSONL file. Idempotent on the same input.

    Returns a report: ``{counts: {...}, skipped: int, errors: int,
    error_lines: [...]}``.
    """
    counts: dict[str, int] = {}
    skipped = 0
    errors = 0
    error_lines: list[dict[str, Any]] = []

    with get_session() as sess, path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors += 1
                error_lines.append({"line": lineno, "error": str(e)})
                continue
            if not isinstance(obj, dict):
                errors += 1
                error_lines.append({"line": lineno, "error": "not a JSON object"})
                continue
            type_name = obj.pop("type", None)
            if not isinstance(type_name, str):
                errors += 1
                error_lines.append({"line": lineno, "error": "missing 'type' field"})
                continue
            cls = ENTITY_BY_NAME.get(type_name)
            if cls is None:
                skipped += 1
                continue
            payload = _rename_in(type_name, obj)
            valid_attrs = set(cls.__mapper__.attrs.keys())
            filtered = {k: v for k, v in payload.items() if k in valid_attrs}
            inst = cls(**filtered)
            sess.merge(inst)
            counts[type_name] = counts.get(type_name, 0) + 1

    return {
        "counts": counts,
        "skipped": skipped,
        "errors": errors,
        "error_lines": error_lines,
    }


def import_markdown_dir(path: Path) -> dict[str, Any]:
    """Walk ``path`` for ``.md`` files; each becomes a durable Note.

    A file at ``<path>/<bucket>/.../foo.md`` (where ``<bucket>`` is one of
    the canonical buckets) inherits ``bucket=<bucket>``; otherwise bucket
    is None. The relative path lands in ``metadata.path``. Idempotent
    via stable ids hashed from the relative path.
    """
    counts: dict[str, int] = {"Note": 0}
    skipped = 0
    errors = 0

    base = path.resolve()
    if not base.is_dir():
        return {
            "counts": counts,
            "skipped": skipped,
            "errors": 1,
            "error_lines": [{"path": str(path), "error": "not a directory"}],
        }

    with get_session() as sess:
        for md in sorted(base.rglob("*.md")):
            try:
                content = md.read_text(encoding="utf-8")
            except OSError:
                errors += 1
                continue
            rel = md.relative_to(base)
            parts = rel.parts
            bucket: str | None = None
            if len(parts) > 1 and parts[0] in ALL_BUCKETS:
                bucket = parts[0]
            note = Note(
                id=_stable_id_from_path(rel),
                content=content,
                bucket=bucket,
                durability="durable",
                source="user",
                meta={"path": str(rel).replace("\\", "/")},
            )
            sess.merge(note)
            counts["Note"] += 1

    return {
        "counts": counts,
        "skipped": skipped,
        "errors": errors,
        "error_lines": [],
    }

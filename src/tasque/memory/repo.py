"""Typed facade over the memory layer. The only module outside ``tasque.memory``
that callers should touch."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select

from tasque.memory.db import get_session
from tasque.memory.entities import (
    Aim,
    Attachment,
    Base,
    ChainRun,
    ChainTemplate,
    FailedJob,
    Note,
    QueuedJob,
    Signal,
    WorkerPattern,
    utc_now_iso,
)

Entity = (
    Note
    | Aim
    | Signal
    | QueuedJob
    | WorkerPattern
    | FailedJob
    | ChainTemplate
    | ChainRun
    | Attachment
)

# All concrete entity classes — order matters for stable export and lookup.
_ALL_TYPES: tuple[type[Base], ...] = (
    Note,
    Aim,
    Signal,
    QueuedJob,
    WorkerPattern,
    FailedJob,
    ChainTemplate,
    ChainRun,
    Attachment,
)

# Types that have a single ``bucket`` column suitable for ``query_bucket``.
_BUCKETED_TYPES: tuple[type[Base], ...] = (
    Note,
    Aim,
    QueuedJob,
    WorkerPattern,
    FailedJob,
    ChainTemplate,
    ChainRun,
    Attachment,
)

_WORKER_PATTERN_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_./-]{2,}", re.IGNORECASE)
_WORKER_PATTERN_STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "bucket",
        "but",
        "can",
        "did",
        "directive",
        "done",
        "for",
        "from",
        "has",
        "have",
        "into",
        "its",
        "job",
        "now",
        "once",
        "only",
        "run",
        "step",
        "that",
        "the",
        "this",
        "through",
        "use",
        "using",
        "was",
        "when",
        "with",
        "worker",
        "you",
        "your",
    }
)


def write_entity[E: Base](entity: E) -> E:
    """Insert or update an entity (by primary key) and return the persisted row.

    For a fresh row (no ``id``), uses ``sess.add`` so column defaults like the
    UUID id land on the original instance after flush. For existing rows,
    uses ``sess.merge`` to upsert.
    """
    with get_session() as sess:
        if getattr(entity, "id", None) is None:
            sess.add(entity)
            sess.flush()
            sess.expunge(entity)
            return entity
        merged = sess.merge(entity)
        sess.flush()
        sess.expunge(merged)
        return merged


def get_entity(id: str) -> Entity | None:
    """Look up an entity by id across all tables. Returns the first match."""
    with get_session() as sess:
        for cls in _ALL_TYPES:
            obj = sess.get(cls, id)
            if obj is not None:
                sess.expunge(obj)
                return obj  # type: ignore[return-value]
    return None


def delete_entity(id: str) -> bool:
    """Delete an entity by id from whichever table holds it."""
    with get_session() as sess:
        for cls in _ALL_TYPES:
            obj = sess.get(cls, id)
            if obj is not None:
                sess.delete(obj)
                return True
    return False


def update_entity_status(id: str, status: str) -> bool:
    """Update status on a QueuedJob, Aim, or FailedJob.

    For FailedJob (which has ``resolved`` instead of ``status``), passing
    ``status="resolved"`` sets resolved=True; anything else sets it False.
    """
    with get_session() as sess:
        for cls in (QueuedJob, Aim):
            obj = sess.get(cls, id)
            if obj is not None:
                obj.status = status
                obj.updated_at = utc_now_iso()
                return True
        fj = sess.get(FailedJob, id)
        if fj is not None:
            fj.resolved = status == "resolved"
            fj.updated_at = utc_now_iso()
            return True
    return False


def archive(id: str) -> bool:
    """Set ``archived=True`` on a Note, Signal, Attachment, or WorkerPattern."""
    with get_session() as sess:
        for cls in (Note, Signal, Attachment, WorkerPattern):
            obj = sess.get(cls, id)
            if obj is not None:
                obj.archived = True
                obj.updated_at = utc_now_iso()
                return True
    return False


def query_bucket(
    bucket: str,
    *,
    types: list[type[Base]] | None = None,
    archived: bool = False,
    limit: int | None = None,
) -> list[Entity]:
    """Return entities filed under ``bucket``.

    ``types`` defaults to all entity types with a single ``bucket`` column
    (Signal is excluded — use ``query_signals_for`` instead).
    """
    use_types = types if types is not None else list(_BUCKETED_TYPES)
    results: list[Entity] = []
    with get_session() as sess:
        for cls in use_types:
            stmt = select(cls).where(cls.bucket == bucket)  # type: ignore[attr-defined]
            if hasattr(cls, "archived"):
                stmt = stmt.where(cls.archived == archived)  # type: ignore[attr-defined]
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = sess.execute(stmt).scalars().all()
            for row in rows:
                results.append(row)  # type: ignore[arg-type]
            if limit is not None and len(results) >= limit:
                break
        sess.expunge_all()
    return results[:limit] if limit is not None else results


def query_signals_for(bucket: str) -> list[Signal]:
    """Return active signals addressed to ``bucket`` (or to ``"all"``)."""
    with get_session() as sess:
        stmt = (
            select(Signal)
            .where((Signal.to_bucket == bucket) | (Signal.to_bucket == "all"))
            .where(Signal.archived.is_(False))
        )
        rows = list(sess.execute(stmt).scalars().all())
        sess.expunge_all()
        return rows


def query_pending_jobs(*, before_iso: str | None = None) -> list[QueuedJob]:
    """Return pending QueuedJobs. If ``before_iso`` is given, restrict to
    jobs with ``fire_at == 'now'`` or ``fire_at <= before_iso``."""
    with get_session() as sess:
        stmt = select(QueuedJob).where(QueuedJob.status == "pending")
        if before_iso is not None:
            stmt = stmt.where(
                (QueuedJob.fire_at == "now") | (QueuedJob.fire_at <= before_iso)
            )
        rows = list(sess.execute(stmt).scalars().all())
        sess.expunge_all()
        return rows


def query_unresolved_failures(limit: int = 20) -> list[FailedJob]:
    """Return up to ``limit`` unresolved FailedJobs, newest-first."""
    with get_session() as sess:
        stmt = (
            select(FailedJob)
            .where(FailedJob.resolved.is_(False))
            .order_by(FailedJob.created_at.desc())
            .limit(limit)
        )
        rows = list(sess.execute(stmt).scalars().all())
        sess.expunge_all()
        return rows


def worker_pattern_keywords(text: str, *, limit: int = 16) -> list[str]:
    """Extract stable keyword tokens for simple worker-pattern matching."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _WORKER_PATTERN_WORD_RE.finditer(text.lower()):
        token = match.group(0).strip("_./-")
        if (
            len(token) < 3
            or token in _WORKER_PATTERN_STOPWORDS
            or token.isdigit()
            or token in seen
        ):
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _json_search_text(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str).lower()
    except (TypeError, ValueError):
        return str(value).lower()


def _score_worker_pattern(row: WorkerPattern, tokens: Sequence[str]) -> int:
    key_text = (row.key or "").lower()
    content_text = (row.content or "").lower()
    tags_text = " ".join(str(t).lower() for t in (row.tags or []))
    meta_text = _json_search_text(row.meta or {})
    score = 0
    for token in tokens:
        if token in key_text:
            score += 4
        if token in tags_text:
            score += 3
        if token in content_text:
            score += 2
        if token in meta_text:
            score += 1
    return score


def search_worker_patterns(
    *,
    query: str,
    bucket: str = "",
    limit: int = 5,
    touch: bool = True,
) -> list[WorkerPattern]:
    """Keyword search over active WorkerPattern rows.

    ``bucket=""`` searches across buckets. When ``touch=True``, selected
    rows have ``last_used_at`` updated so pattern use is auditable.
    """
    if limit < 1 or not query.strip():
        return []
    tokens = worker_pattern_keywords(query, limit=24)
    if not tokens:
        return []
    bucket_value = bucket.strip()
    with get_session() as sess:
        stmt = select(WorkerPattern).where(WorkerPattern.archived.is_(False))
        if bucket_value:
            stmt = stmt.where(WorkerPattern.bucket == bucket_value)
        rows = list(
            sess.execute(
                stmt.order_by(WorkerPattern.updated_at.desc()).limit(250)
            )
            .scalars()
            .all()
        )
        scored: list[tuple[int, int, str, WorkerPattern]] = []
        for row in rows:
            score = _score_worker_pattern(row, tokens)
            if score > 0:
                scored.append((score, row.success_count, row.updated_at, row))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected = [item[3] for item in scored[:limit]]
        if touch and selected:
            now = utc_now_iso()
            for row in selected:
                row.last_used_at = now
                row.updated_at = now
            sess.flush()
        for row in selected:
            sess.expunge(row)
        return selected


def upsert_worker_pattern(
    *,
    bucket: str | None,
    source_kind: str,
    key: str,
    content: str,
    tags: Sequence[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> WorkerPattern:
    """Insert or update a compact successful-run WorkerPattern."""
    bucket_value = (bucket or "").strip()
    source_value = source_kind.strip()
    key_value = key.strip()
    content_value = content.strip()
    if not source_value:
        raise ValueError("source_kind must be non-empty")
    if not key_value:
        raise ValueError("key must be non-empty")
    if not content_value:
        raise ValueError("content must be non-empty")
    tags_value: list[str] = []
    seen: set[str] = set()
    for tag in tags or ():
        tag_value = str(tag).strip().lower()
        if not tag_value or tag_value in seen:
            continue
        seen.add(tag_value)
        tags_value.append(tag_value)

    with get_session() as sess:
        stmt = (
            select(WorkerPattern)
            .where(WorkerPattern.bucket == bucket_value)
            .where(WorkerPattern.source_kind == source_value)
            .where(WorkerPattern.key == key_value)
        )
        row = sess.execute(stmt).scalars().first()
        now = utc_now_iso()
        if row is None:
            row = WorkerPattern(
                bucket=bucket_value,
                source_kind=source_value,
                key=key_value,
                content=content_value,
                tags=tags_value,
                meta=meta or {},
                success_count=1,
            )
            sess.add(row)
        else:
            row.content = content_value
            row.tags = tags_value
            row.meta = meta or {}
            row.success_count += 1
            row.archived = False
            row.updated_at = now
        sess.flush()
        sess.expunge(row)
        return row


def bump_job_heartbeat(id: str, iso: str) -> bool:
    """Set ``last_heartbeat`` on a QueuedJob to ``iso``."""
    with get_session() as sess:
        job = sess.get(QueuedJob, id)
        if job is None:
            return False
        job.last_heartbeat = iso
        job.updated_at = utc_now_iso()
        return True


def supersede_note(old_id: str, new_text: str, *, durability: str) -> Note:
    """Create a new Note that supersedes ``old_id``.

    The old note's ``superseded_by`` is set to the new id; bucket and
    source carry over.
    """
    with get_session() as sess:
        old = sess.get(Note, old_id)
        if old is None:
            raise ValueError(f"no note with id={old_id}")
        new_note = Note(
            content=new_text,
            bucket=old.bucket,
            durability=durability,
            source=old.source,
            meta=dict(old.meta or {}),
        )
        sess.add(new_note)
        sess.flush()
        old.superseded_by = new_note.id
        old.updated_at = utc_now_iso()
        sess.flush()
        sess.expunge(new_note)
        return new_note


# Decay sweep — implementation lives in ``memory.decay`` so the broader
# nondurable-memory sweep (notes + signals + superseded chains + optional
# hard-delete) is a peer of this facade rather than buried in it. The
# legacy ``prune_decayed_notes`` is re-exported below for backward
# compatibility; new callers should use ``sweep_nondurable_memory``.
from tasque.memory.decay import (  # noqa: E402
    DecayReport,
    prune_decayed_notes,
    sweep_nondurable_memory,
)

__all__ = [
    "DecayReport",
    "Entity",
    "archive",
    "bump_job_heartbeat",
    "delete_entity",
    "get_entity",
    "prune_decayed_notes",
    "query_bucket",
    "query_pending_jobs",
    "query_signals_for",
    "query_unresolved_failures",
    "search_worker_patterns",
    "supersede_note",
    "sweep_nondurable_memory",
    "update_entity_status",
    "upsert_worker_pattern",
    "worker_pattern_keywords",
    "write_entity",
]

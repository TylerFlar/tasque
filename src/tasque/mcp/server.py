"""The tasque stdio MCP server.

Exposes the daemon's full write surface — notes, queued jobs, aims,
chain templates, chain runs, signals — as MCP tools that any upstream
invocation through the proxy can call mid-turn.
The reactive bucket coach, the chain worker, the planner, the
strategist, and the reply runtimes all share this catalog and act
through it directly: tool call → result observed in the same turn →
keep going.

Bucket scoping: every write that targets a bucket takes ``bucket`` as
an explicit argument, validated against :data:`tasque.buckets.ALL_BUCKETS`.
The agent's system prompt names its bucket; the LLM passes that value
through. Every action is auditable from the tool-call payload alone.

All tools return JSON-encoded strings. Mutations return ``{"ok": true,
...}`` envelopes on success and ``{"ok": false, "error": "..."}`` on
failure (including validation failures). Reads return the requested
object(s) directly.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import or_, select, text

from tasque.agents import result_inbox
from tasque.buckets import ALL_BUCKETS
from tasque.chains.crud import UNSET as _CRUD_UNSET
from tasque.chains.crud import (
    MirrorMismatch,
)
from tasque.chains.crud import (
    create_chain_template as _create_template,
)
from tasque.chains.crud import (
    delete_chain_template as _delete_template,
)
from tasque.chains.crud import (
    get_chain_template as _get_template,
)
from tasque.chains.crud import (
    list_chain_templates as _list_templates,
)
from tasque.chains.crud import (
    update_chain_template as _update_template,
)
from tasque.chains.manager import (
    get_chain_state as _get_chain_state,
)
from tasque.chains.manager import (
    pause_chain as _pause_chain,
)
from tasque.chains.manager import (
    resume_chain as _resume_chain,
)
from tasque.chains.manager import (
    stop_chain as _stop_chain,
)
from tasque.chains.scheduler import launch_chain_run
from tasque.chains.spec import SpecError, validate_spec
from tasque.jobs.cron import next_fire_at, to_iso, validate_cron
from tasque.llm.factory import ALL_TIERS
from tasque.mcp.condense import maybe_condense
from tasque.mcp.tool_triggers import dispatch_tool_event
from tasque.memory.db import get_session
from tasque.memory.entities import (
    Aim,
    ChainRun,
    FailedJob,
    Note,
    QueuedJob,
    Signal,
    WorkerPattern,
    utc_now_iso,
)
from tasque.memory.repo import (
    archive as _archive_entity,
)
from tasque.memory.repo import (
    search_worker_patterns,
    update_entity_status,
    write_entity,
)

# Cross-bucket "actor" identifiers permitted for Signal.from_bucket. The
# nine canonical buckets are valid; the strategist agent and ad-hoc
# system signals also need to be representable.
_ALL_SIGNAL_FROMS: frozenset[str] = frozenset({*ALL_BUCKETS, "strategist", "system"})
_AIM_CHAIN_DEDUP_SECONDS = 30 * 60
_AIM_CHAIN_ACTIVE_STATUSES = frozenset({"running", "paused", "awaiting_approval"})
_AIM_CHAIN_RECENT_TERMINAL_STATUSES = frozenset({"completed", "stopped"})
_MEMORY_KINDS = frozenset(
    {"fact", "preference", "policy", "working", "artifact", "summary", "question"}
)
_DEFAULT_TTLS_BY_KIND: dict[str, int] = {
    "artifact": 3,
    "working": 14,
    "question": 14,
}

if TYPE_CHECKING:  # pragma: no cover - imported only for type hints
    from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------- helpers


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, default=str)


def _err(message: str, **fields: Any) -> str:
    return json.dumps({"ok": False, "error": message, **fields}, default=str)


def _validate_bucket(bucket: str) -> str | None:
    if bucket in ALL_BUCKETS:
        return None
    return f"bucket must be one of {sorted(ALL_BUCKETS)!r}, got {bucket!r}"


def _validate_tier(tier: str) -> str | None:
    if tier in ALL_TIERS:
        return None
    return f"tier must be one of {sorted(ALL_TIERS)!r}, got {tier!r}"


def _validate_durability(durability: str) -> str | None:
    if durability in ("ephemeral", "durable", "behavioral"):
        return None
    return (
        "durability must be 'ephemeral', 'durable', or 'behavioral', "
        f"got {durability!r}"
    )


def _validate_memory_kind(memory_kind: str) -> str | None:
    if memory_kind in _MEMORY_KINDS:
        return None
    return (
        f"memory_kind must be one of {sorted(_MEMORY_KINDS)!r}, "
        f"got {memory_kind!r}"
    )


def _normalize_memory_kind(
    memory_kind: str | None,
    *,
    durability: str,
    source: str,
) -> str:
    if memory_kind is not None and memory_kind.strip():
        return memory_kind.strip().lower()
    source_value = source.strip().lower()
    if durability == "behavioral":
        return "policy"
    if source_value == "worker" or source_value.startswith("chain:"):
        return "artifact"
    if durability == "ephemeral":
        return "working"
    if durability == "durable" and source_value in {"user", "strategist"}:
        return "fact"
    return "working"


def _normalize_ttl_days(ttl_days: int | None, *, memory_kind: str) -> int | None:
    if ttl_days is not None:
        return ttl_days
    return _DEFAULT_TTLS_BY_KIND.get(memory_kind)


def _validate_ttl_days(ttl_days: int | None) -> str | None:
    if ttl_days is None:
        return None
    if ttl_days < 0:
        return "ttl_days must be >= 0 when provided"
    return None


def _normalize_canonical_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _parse_utc_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _aim_chain_dedup_window() -> timedelta:
    raw = os.environ.get("TASQUE_AIM_CHAIN_DEDUP_SECONDS")
    seconds = float(_AIM_CHAIN_DEDUP_SECONDS)
    if raw is not None:
        try:
            seconds = float(raw)
        except ValueError:
            seconds = float(_AIM_CHAIN_DEDUP_SECONDS)
    return timedelta(seconds=max(0.0, seconds))


def _chain_run_aim_id(row: ChainRun) -> str | None:
    try:
        initial = json.loads(row.initial_state_json or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(initial, dict):
        return None
    vars_raw = initial.get("vars")
    if not isinstance(vars_raw, dict):
        return None
    aim_id = vars_raw.get("aim_id")
    return aim_id if isinstance(aim_id, str) and aim_id else None


def _find_existing_aim_chain_run(spec: dict[str, Any]) -> ChainRun | None:
    """Return a recent/active chain for the same Aim-derived spec, if any."""
    chain_name = spec.get("chain_name")
    vars_raw = spec.get("vars")
    aim_id = vars_raw.get("aim_id") if isinstance(vars_raw, dict) else None
    if not isinstance(chain_name, str) or not chain_name:
        return None
    if not isinstance(aim_id, str) or not aim_id:
        return None

    cutoff = datetime.now(UTC) - _aim_chain_dedup_window()
    with get_session() as sess:
        stmt = (
            select(ChainRun)
            .where(ChainRun.chain_name == chain_name)
            .order_by(ChainRun.created_at.desc())
            .limit(50)
        )
        rows = list(sess.execute(stmt).scalars().all())
        for row in rows:
            if _chain_run_aim_id(row) != aim_id:
                continue
            if row.status in _AIM_CHAIN_ACTIVE_STATUSES:
                sess.expunge(row)
                return row
            if row.status in _AIM_CHAIN_RECENT_TERMINAL_STATUSES:
                reference = (
                    _parse_utc_iso(row.ended_at)
                    or _parse_utc_iso(row.updated_at)
                    or _parse_utc_iso(row.started_at)
                    or _parse_utc_iso(row.created_at)
                )
                if reference is not None and reference >= cutoff:
                    sess.expunge(row)
                    return row
        sess.expunge_all()
    return None


def _resolve_fire_at(fire_at: str, recurrence: str | None) -> str:
    """Return canonical ``fire_at`` for a queued job.

    For recurring jobs we compute the first scheduled firing from the
    cron expression so the row is consistent with what the scheduler
    would have written. For one-shot jobs we accept the user-supplied
    value (``"now"`` or an ISO-8601 timestamp) verbatim.
    """
    if recurrence is None:
        return fire_at
    nxt = next_fire_at(recurrence, after=datetime.now(UTC))
    return to_iso(nxt)


def _serialize_note(n: Note, *, truncate: int | None = None) -> dict[str, Any]:
    content = n.content
    if truncate is not None and len(content) > truncate:
        content = content[: truncate - 1].rstrip() + "…"
    return {
        "id": n.id,
        "content": content,
        "bucket": n.bucket,
        "durability": n.durability,
        "memory_kind": n.memory_kind,
        "ttl_days": n.ttl_days,
        "canonical_key": n.canonical_key,
        "source": n.source,
        "archived": n.archived,
        "superseded_by": n.superseded_by,
        "created_at": n.created_at,
        "updated_at": n.updated_at,
        "meta": n.meta,
    }


def _serialize_queued_job(j: QueuedJob) -> dict[str, Any]:
    return {
        "id": j.id,
        "bucket": j.bucket,
        "directive": j.directive,
        "reason": j.reason,
        "fire_at": j.fire_at,
        "status": j.status,
        "recurrence": j.recurrence,
        "visible": j.visible,
        "queued_by": j.queued_by,
        "tier": j.tier,
        "chain_id": j.chain_id,
        "chain_step_id": j.chain_step_id,
        "thread_id": j.thread_id,
        "created_at": j.created_at,
        "updated_at": j.updated_at,
        "last_summary": j.last_summary,
        "last_error": j.last_error,
    }


def _serialize_worker_pattern(p: WorkerPattern) -> dict[str, Any]:
    return {
        "id": p.id,
        "bucket": p.bucket,
        "source_kind": p.source_kind,
        "key": p.key,
        "content": p.content,
        "tags": p.tags,
        "meta": p.meta,
        "success_count": p.success_count,
        "last_used_at": p.last_used_at,
        "created_at": p.created_at,
        "updated_at": p.updated_at,
        "archived": p.archived,
    }


def _serialize_chain_run(r: ChainRun) -> dict[str, Any]:
    return {
        "chain_id": r.chain_id,
        "chain_name": r.chain_name,
        "bucket": r.bucket,
        "status": r.status,
        "template_id": r.template_id,
        "thread_id": r.thread_id,
        "started_at": r.started_at,
        "last_heartbeat": r.last_heartbeat,
        "lease_owner": r.lease_owner,
        "lease_expires_at": r.lease_expires_at,
        "ended_at": r.ended_at,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
    }


def _serialize_aim(a: Aim) -> dict[str, Any]:
    return {
        "id": a.id,
        "title": a.title,
        "bucket": a.bucket,
        "scope": a.scope,
        "target_date": a.target_date,
        "description": a.description,
        "status": a.status,
        "parent_id": a.parent_id,
        "source": a.source,
        "broken_down_at": dict(a.broken_down_at or {}),
        "created_at": a.created_at,
        "updated_at": a.updated_at,
    }


def _serialize_signal(s: Signal) -> dict[str, Any]:
    return {
        "id": s.id,
        "from_bucket": s.from_bucket,
        "to_bucket": s.to_bucket,
        "kind": s.kind,
        "urgency": s.urgency,
        "summary": s.summary,
        "body": s.body,
        "context": s.context,
        "expires_at": s.expires_at,
        "archived": s.archived,
        "created_at": s.created_at,
    }


# ------------------------------------------------------------------ server


def build_server() -> FastMCP:
    """Construct (but do not start) the FastMCP server.

    Imports ``mcp`` lazily so the rest of the codebase doesn't pay the
    import cost. The returned server is ready for ``run_stdio`` or for
    a test harness to invoke individual tools directly.
    """
    from mcp.server.fastmcp import FastMCP  # local import — heavy module

    mcp = FastMCP(
        "tasque",
        instructions=(
            "tasque — read and write the daemon's state: Notes, Aims, "
            "queued jobs, chain templates, chain runs, cross-bucket "
            "signals, and the cross-bucket summary. Tools that target a "
            "single bucket take a `bucket` argument (one of: "
            + ", ".join(sorted(ALL_BUCKETS)) + "). Use the bucket your "
            "system prompt names. Mutations return {\"ok\": true, ...} on "
            "success or {\"ok\": false, \"error\": \"...\"} on "
            "validation/runtime failure.\n\n"
            "Read tools (note_*, job_get/list, worker_pattern_search, chain_template_*, "
            "chain_run_get/list, signal_list, aim_get/list, "
            "bucket_summary) take a required `intent` argument: one short "
            "sentence describing what you want from this call (e.g. "
            "'durable health notes mentioning sleep', 'pending jobs in "
            "career'). When the raw result exceeds ~60KB it is condensed "
            "by a small-tier call using your intent as the filter and returned "
            "as {\"_condensed\": true, \"_intent\": ..., \"summary\": "
            "...}. Write a precise intent — vague intents waste the "
            "condense pass."
        ),
    )

    # ------------------------------------------------------------- notes

    @mcp.tool()
    def note_create(
        content: str,
        bucket: str,
        durability: str = "ephemeral",
        memory_kind: str | None = None,
        ttl_days: int | None = None,
        canonical_key: str | None = None,
        source: str = "mcp",
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Write a Note.

        ``durability`` remains the broad compatibility class:
        "ephemeral", "durable", or "behavioral". ``memory_kind`` is the
        lifecycle class: fact, preference, policy, working, artifact,
        summary, or question. Artifacts/working/questions decay
        aggressively by ``ttl_days`` (or their default TTL). Summaries may
        set ``canonical_key`` so older active summaries for the same key
        are archived by the memory sweep. Returns the new note id."""
        if not content.strip():
            return _err("content must be a non-empty string")
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        kind = _normalize_memory_kind(
            memory_kind,
            durability=durability,
            source=source,
        )
        if (msg := _validate_memory_kind(kind)) is not None:
            return _err(msg)
        if (msg := _validate_ttl_days(ttl_days)) is not None:
            return _err(msg)
        n = Note(
            content=content,
            bucket=bucket,
            durability=durability,
            memory_kind=kind,
            ttl_days=_normalize_ttl_days(ttl_days, memory_kind=kind),
            canonical_key=_normalize_canonical_key(canonical_key),
            source=source,
            meta=meta or {},
        )
        written = write_entity(n)
        dispatch_tool_event("note_create", bucket=bucket, source=source)
        return _ok(id=written.id)

    @mcp.tool()
    def note_update(
        note_id: str,
        content: str | None = None,
        bucket: str | None = None,
        durability: str | None = None,
        memory_kind: str | None = None,
        ttl_days: int | None = None,
        canonical_key: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Patch an existing Note. Only provide fields you intend to
        change. Use this for small corrections (typos, wrong bucket,
        metadata fixes). For replacing a stale durable/behavioral memory
        with a new fact, prefer ``note_supersede`` so the old version is
        preserved for audit but hidden from default reads."""
        if (
            content is None
            and bucket is None
            and durability is None
            and memory_kind is None
            and ttl_days is None
            and canonical_key is None
            and meta is None
        ):
            return _err("provide at least one field to update")
        if content is not None and not content.strip():
            return _err("content must be a non-empty string when provided")
        if bucket is not None and (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if durability is not None and (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        kind_update = memory_kind.strip().lower() if memory_kind is not None else None
        if kind_update is not None and (msg := _validate_memory_kind(kind_update)) is not None:
            return _err(msg)
        if (msg := _validate_ttl_days(ttl_days)) is not None:
            return _err(msg)
        changed_buckets: set[str] = set()
        with get_session() as sess:
            row = sess.get(Note, note_id)
            if row is None:
                return _err(f"no Note with id={note_id}")
            if isinstance(row.bucket, str) and row.bucket in ALL_BUCKETS:
                changed_buckets.add(row.bucket)
            if content is not None:
                row.content = content
            if bucket is not None:
                row.bucket = bucket
            if durability is not None:
                row.durability = durability
                if memory_kind is None:
                    row.memory_kind = _normalize_memory_kind(
                        None,
                        durability=row.durability,
                        source=row.source,
                    )
            if kind_update is not None:
                row.memory_kind = kind_update
                if ttl_days is None and row.ttl_days is None:
                    row.ttl_days = _normalize_ttl_days(
                        None,
                        memory_kind=kind_update,
                    )
            if ttl_days is not None:
                row.ttl_days = ttl_days
            if canonical_key is not None:
                row.canonical_key = _normalize_canonical_key(canonical_key)
            if meta is not None:
                row.meta = meta
            row.updated_at = utc_now_iso()
            sess.flush()
            if isinstance(row.bucket, str) and row.bucket in ALL_BUCKETS:
                changed_buckets.add(row.bucket)
            note = _serialize_note(row)

        for changed_bucket in sorted(changed_buckets):
            dispatch_tool_event("note_update", bucket=changed_bucket)
        return _ok(id=note_id, note=note)

    @mcp.tool()
    def note_supersede(
        note_id: str,
        content: str,
        durability: str | None = None,
        memory_kind: str | None = None,
        ttl_days: int | None = None,
        canonical_key: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Replace a Note with a new Note while preserving audit history.
        The old note gets ``superseded_by=<new id>`` and ``archived=True``,
        so default searches see the replacement, not both versions.

        Omit ``durability`` to carry over the old durability. Omit
        ``meta`` to carry over old metadata; provide it to replace the
        metadata on the new note."""
        if not content.strip():
            return _err("content must be a non-empty string")
        if durability is not None and (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        kind_update = memory_kind.strip().lower() if memory_kind is not None else None
        if kind_update is not None and (msg := _validate_memory_kind(kind_update)) is not None:
            return _err(msg)
        if (msg := _validate_ttl_days(ttl_days)) is not None:
            return _err(msg)
        changed_bucket: str | None = None
        with get_session() as sess:
            old = sess.get(Note, note_id)
            if old is None:
                return _err(f"no Note with id={note_id}")
            new_durability = durability or old.durability
            new_kind = kind_update or _normalize_memory_kind(
                old.memory_kind,
                durability=new_durability,
                source=old.source,
            )
            new_note = Note(
                content=content,
                bucket=old.bucket,
                durability=new_durability,
                memory_kind=new_kind,
                ttl_days=(
                    ttl_days
                    if ttl_days is not None
                    else old.ttl_days
                    if old.ttl_days is not None
                    else _normalize_ttl_days(None, memory_kind=new_kind)
                ),
                canonical_key=(
                    _normalize_canonical_key(canonical_key)
                    if canonical_key is not None
                    else old.canonical_key
                ),
                source=old.source,
                meta=dict(old.meta or {}) if meta is None else meta,
            )
            sess.add(new_note)
            sess.flush()
            old.superseded_by = new_note.id
            old.archived = True
            old.updated_at = utc_now_iso()
            sess.flush()
            if isinstance(old.bucket, str) and old.bucket in ALL_BUCKETS:
                changed_bucket = old.bucket
            note = _serialize_note(new_note)

        if changed_bucket is not None:
            dispatch_tool_event("note_supersede", bucket=changed_bucket)
        return _ok(id=note["id"], superseded_id=note_id, note=note)

    @mcp.tool()
    def note_get(intent: str, note_id: str) -> str:
        """Fetch one note by id (full untruncated content), or
        {"ok": false, "error": "missing"} if no such note.

        ``intent`` (required): one short sentence on why you're reading
        this note — used to condense the result if oversize."""
        with get_session() as sess:
            row = sess.get(Note, note_id)
            if row is None:
                return _err("missing")
            sess.expunge(row)
        return maybe_condense(
            json.dumps(_serialize_note(row)),
            intent=intent,
            tool_name="note_get",
        )

    @mcp.tool()
    def note_list(
        intent: str,
        bucket: str,
        durability: str | None = None,
        memory_kind: str | None = None,
        include_artifacts: bool = False,
        include_archived: bool = False,
        limit: int = 20,
    ) -> str:
        """List notes in ``bucket`` (newest first). Optional filters:
        ``durability`` ('ephemeral' / 'durable' / 'behavioral'),
        ``memory_kind`` (fact / preference / policy / working / artifact /
        summary / question), ``include_artifacts`` (default False), and
        ``include_archived`` (default False). Returns a JSON array.

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize."""
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if durability is not None and (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        kind_filter = memory_kind.strip().lower() if memory_kind is not None else None
        if kind_filter is not None and (msg := _validate_memory_kind(kind_filter)) is not None:
            return _err(msg)
        if limit < 1:
            return _err("limit must be >= 1")
        with get_session() as sess:
            stmt = select(Note).where(Note.bucket == bucket)
            if not include_archived:
                stmt = stmt.where(Note.archived.is_(False))
            if durability is not None:
                stmt = stmt.where(Note.durability == durability)
            if kind_filter is not None:
                stmt = stmt.where(Note.memory_kind == kind_filter)
            elif not include_artifacts:
                stmt = stmt.where(or_(Note.memory_kind.is_(None), Note.memory_kind != "artifact"))
            stmt = stmt.order_by(Note.updated_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return maybe_condense(
            json.dumps([_serialize_note(n) for n in rows]),
            intent=intent,
            tool_name="note_list",
        )

    @mcp.tool()
    def note_search(
        intent: str,
        bucket: str,
        query: str,
        durability: str | None = None,
        memory_kind: str | None = None,
        include_artifacts: bool = False,
        limit: int = 10,
    ) -> str:
        """Substring search (case-insensitive) over note content within
        ``bucket``. Returns up to ``limit`` matches newest-first; each
        note's content is truncated to ~400 chars in the result — call
        ``note_get`` for full content.

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize."""
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if durability is not None and (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        kind_filter = memory_kind.strip().lower() if memory_kind is not None else None
        if kind_filter is not None and (msg := _validate_memory_kind(kind_filter)) is not None:
            return _err(msg)
        if not query.strip():
            return json.dumps([])
        if limit < 1:
            return _err("limit must be >= 1")
        with get_session() as sess:
            stmt = (
                select(Note)
                .where(Note.bucket == bucket)
                .where(Note.archived.is_(False))
                .where(Note.content.ilike(f"%{query}%"))
            )
            if durability is not None:
                stmt = stmt.where(Note.durability == durability)
            if kind_filter is not None:
                stmt = stmt.where(Note.memory_kind == kind_filter)
            elif not include_artifacts:
                stmt = stmt.where(or_(Note.memory_kind.is_(None), Note.memory_kind != "artifact"))
            stmt = stmt.order_by(Note.updated_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return maybe_condense(
            json.dumps([_serialize_note(n, truncate=400) for n in rows]),
            intent=intent,
            tool_name="note_search",
        )

    @mcp.tool()
    def note_search_any(
        intent: str,
        bucket: str,
        keywords: list[str],
        durability: str | None = None,
        memory_kind: str | None = None,
        include_artifacts: bool = False,
        limit: int = 10,
    ) -> str:
        """Like ``note_search`` but matches any keyword (OR-style).

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize."""
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if durability is not None and (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        kind_filter = memory_kind.strip().lower() if memory_kind is not None else None
        if kind_filter is not None and (msg := _validate_memory_kind(kind_filter)) is not None:
            return _err(msg)
        words = [k.strip() for k in keywords if k.strip()]
        if not words:
            return json.dumps([])
        if limit < 1:
            return _err("limit must be >= 1")
        with get_session() as sess:
            stmt = (
                select(Note)
                .where(Note.bucket == bucket)
                .where(Note.archived.is_(False))
                .where(or_(*[Note.content.ilike(f"%{w}%") for w in words]))
            )
            if durability is not None:
                stmt = stmt.where(Note.durability == durability)
            if kind_filter is not None:
                stmt = stmt.where(Note.memory_kind == kind_filter)
            elif not include_artifacts:
                stmt = stmt.where(or_(Note.memory_kind.is_(None), Note.memory_kind != "artifact"))
            stmt = stmt.order_by(Note.updated_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return maybe_condense(
            json.dumps([_serialize_note(n, truncate=400) for n in rows]),
            intent=intent,
            tool_name="note_search_any",
        )

    @mcp.tool()
    def note_search_fts(
        intent: str,
        bucket: str,
        query: str,
        durability: str | None = None,
        memory_kind: str | None = None,
        include_artifacts: bool = False,
        limit: int = 10,
    ) -> str:
        """SQLite FTS5 search over note content (BM25-ranked, word-tokenized,
        stemmed). Prefer this when you want word-boundary matches and
        relevance-ordered results. Query syntax: bare words AND, ``OR``
        for alternatives, ``"phrase"`` exact, ``-word`` exclude,
        ``NEAR(x y, 5)`` proximity. Returns the literal string
        ``"fts5_unavailable"`` if the host SQLite lacks FTS5.

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize."""
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if durability is not None and (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        kind_filter = memory_kind.strip().lower() if memory_kind is not None else None
        if kind_filter is not None and (msg := _validate_memory_kind(kind_filter)) is not None:
            return _err(msg)
        if not query.strip():
            return json.dumps([])
        if limit < 1:
            return _err("limit must be >= 1")
        params: dict[str, Any] = {"bucket": bucket, "query": query, "limit": limit}
        sql = (
            "SELECT n.id, n.content, n.bucket, n.durability, n.source, "
            "  n.archived, n.created_at, n.updated_at, n.metadata, n.superseded_by, "
            "  n.memory_kind, n.ttl_days, n.canonical_key "
            "FROM notes_fts JOIN notes n ON n.id = notes_fts.note_id "
            "WHERE notes_fts.bucket = :bucket "
            "  AND notes_fts.archived = 0 "
            "  AND notes_fts MATCH :query "
        )
        if durability is not None:
            sql += "  AND notes_fts.durability = :durability "
            params["durability"] = durability
        if kind_filter is not None:
            sql += "  AND n.memory_kind = :memory_kind "
            params["memory_kind"] = kind_filter
        elif not include_artifacts:
            sql += "  AND (n.memory_kind IS NULL OR n.memory_kind != 'artifact') "
        sql += "ORDER BY bm25(notes_fts) LIMIT :limit"
        with get_session() as sess:
            try:
                rows = list(sess.execute(text(sql), params).all())
            except Exception as exc:
                msg = str(exc).lower()
                if "no such table: notes_fts" in msg or "fts5" in msg:
                    return "fts5_unavailable"
                raise
        out: list[dict[str, Any]] = []
        for row in rows:
            content = row[1]
            if len(content) > 400:
                content = content[:399].rstrip() + "…"
            try:
                meta = json.loads(row[8]) if row[8] else {}
            except (TypeError, json.JSONDecodeError):
                meta = {}
            out.append(
                {
                    "id": row[0],
                    "content": content,
                    "bucket": row[2],
                    "durability": row[3],
                    "source": row[4],
                    "archived": bool(row[5]),
                    "created_at": row[6],
                    "updated_at": row[7],
                    "meta": meta,
                    "superseded_by": row[9],
                    "memory_kind": row[10],
                    "ttl_days": row[11],
                    "canonical_key": row[12],
                }
            )
        return maybe_condense(
            json.dumps(out), intent=intent, tool_name="note_search_fts"
        )

    @mcp.tool()
    def note_archive(note_id: str) -> str:
        """Archive a Note (sets ``archived=True``). Archived notes are
        hidden from default queries but preserved for audit. Idempotent."""
        if _archive_entity(note_id):
            return _ok(id=note_id, archived=True)
        return _err(f"no Note with id={note_id}")

    # ------------------------------------------------------------- jobs

    @mcp.tool()
    def job_create(
        directive: str,
        bucket: str,
        tier: str,
        fire_at: str = "now",
        recurrence: str | None = None,
        reason: str = "",
        visible: bool = True,
        queued_by: str = "mcp",
    ) -> str:
        """Insert a pending QueuedJob. ``tier`` is REQUIRED — one of
        "large", "medium", "small". Pick small for trivial nudges, medium
        for multi-step tool / scrape / summarize work, large for agentic
        planning, code iteration, or deep creative generation.

        ``fire_at`` is "now" or an ISO-8601 UTC timestamp. ``recurrence``
        is None for one-shot jobs or a 5-field cron expression for
        recurring jobs (when set, ``fire_at`` is overwritten with the
        first scheduled firing). Day-of-week must use alias form
        (MON-FRI), not pure-numeric (1-5) — APScheduler's 0=Monday
        breaks naive numeric DOW.

        ``visible=False`` hides the job from coach views (use for
        internal bookkeeping). Returns the new job id."""
        if not directive.strip():
            return _err("directive must be a non-empty string")
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if (msg := _validate_tier(tier)) is not None:
            return _err(msg)
        if not fire_at:
            return _err("fire_at must be a non-empty string")
        if recurrence is not None:
            err = validate_cron(recurrence)
            if err is not None:
                return _err(f"recurrence: {err}")
        try:
            canonical_fire_at = _resolve_fire_at(fire_at, recurrence)
        except Exception as exc:
            return _err(f"could not resolve fire_at: {type(exc).__name__}: {exc}")
        qj = QueuedJob(
            kind="worker",
            bucket=bucket,
            directive=directive,
            reason=reason,
            fire_at=canonical_fire_at,
            status="pending",
            recurrence=recurrence,
            visible=visible,
            queued_by=queued_by,
            tier=tier,
        )
        written = write_entity(qj)
        return _ok(
            id=written.id,
            fire_at=written.fire_at,
            recurrence=written.recurrence,
            status=written.status,
        )

    @mcp.tool()
    def job_get(intent: str, job_id: str) -> str:
        """Fetch one QueuedJob by id, or {"ok": false, "error": "missing"}.

        ``intent`` (required): one short sentence on why you're reading
        this job — used to condense the result if oversize (e.g. when a
        chain step's ``last_summary`` is large)."""
        with get_session() as sess:
            row = sess.get(QueuedJob, job_id)
            if row is None:
                return _err("missing")
            sess.expunge(row)
        return maybe_condense(
            json.dumps(_serialize_queued_job(row)),
            intent=intent,
            tool_name="job_get",
        )

    @mcp.tool()
    def job_update(
        job_id: str,
        directive: str | None = None,
        reason: str | None = None,
        fire_at: str | None = None,
        recurrence: str | None = None,
        recurrence_clear: bool = False,
        visible: bool | None = None,
        tier: str | None = None,
    ) -> str:
        """Update a pending QueuedJob. Only pending jobs can be edited
        (claimed/completed/failed/stopped are terminal). Pass
        ``recurrence_clear=True`` to wipe the cron (otherwise omit
        ``recurrence`` to leave it untouched). Returns
        {"ok": true} or an error envelope."""
        if tier is not None and (msg := _validate_tier(tier)) is not None:
            return _err(msg)
        if recurrence is not None:
            err = validate_cron(recurrence)
            if err is not None:
                return _err(f"recurrence: {err}")
        with get_session() as sess:
            row = sess.get(QueuedJob, job_id)
            if row is None:
                return _err(f"no QueuedJob with id={job_id}")
            if row.status != "pending":
                return _err(
                    f"job {job_id} status is {row.status!r}; only pending jobs can be edited"
                )
            if directive is not None:
                if not directive.strip():
                    return _err("directive must be non-empty")
                row.directive = directive
            if reason is not None:
                row.reason = reason
            if recurrence_clear:
                row.recurrence = None
                if fire_at is not None:
                    row.fire_at = fire_at
            elif recurrence is not None:
                row.recurrence = recurrence
                row.fire_at = _resolve_fire_at(fire_at or "now", recurrence)
            elif fire_at is not None:
                row.fire_at = fire_at
            if visible is not None:
                row.visible = visible
            if tier is not None:
                row.tier = tier
            row.updated_at = utc_now_iso()
            sess.flush()
        return _ok(id=job_id)

    @mcp.tool()
    def job_cancel(job_id: str) -> str:
        """Mark a QueuedJob as ``stopped`` (terminal). Won't fire even
        if pending. Idempotent for already-stopped jobs."""
        with get_session() as sess:
            row = sess.get(QueuedJob, job_id)
            if row is None:
                return _err(f"no QueuedJob with id={job_id}")
            if row.status in ("completed", "failed", "stopped"):
                return _err(
                    f"job {job_id} is already in terminal status {row.status!r}"
                )
        update_entity_status(job_id, "stopped")
        return _ok(id=job_id, status="stopped")

    @mcp.tool()
    def job_list(
        intent: str,
        bucket: str | None = None,
        status: str | None = None,
        recurring_only: bool = False,
        limit: int = 30,
    ) -> str:
        """List QueuedJobs, newest first. All filters are optional —
        omit ``bucket`` to list across all buckets. ``status`` filters
        by lifecycle state (pending/claimed/completed/failed/stopped).
        ``recurring_only=True`` keeps only rows with a non-null
        recurrence cron.

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize."""
        if bucket is not None and (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if limit < 1:
            return _err("limit must be >= 1")
        with get_session() as sess:
            stmt = select(QueuedJob)
            if bucket is not None:
                stmt = stmt.where(QueuedJob.bucket == bucket)
            if status is not None:
                stmt = stmt.where(QueuedJob.status == status)
            if recurring_only:
                stmt = stmt.where(QueuedJob.recurrence.is_not(None))
            stmt = stmt.order_by(QueuedJob.created_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return maybe_condense(
            json.dumps([_serialize_queued_job(j) for j in rows]),
            intent=intent,
            tool_name="job_list",
        )

    # ---------------------------------------------------- worker patterns

    @mcp.tool()
    def worker_pattern_search(
        intent: str,
        query: str,
        bucket: str = "",
        limit: int = 5,
    ) -> str:
        """Search compact successful-run patterns for worker precedent.

        ``bucket`` defaults to ``""`` to search all buckets; pass the
        current run's bucket for bucket-scoped recall. Search is keyword
        based, not vector based. Returns active patterns ordered by simple
        relevance, success count, and recency.

        ``intent`` (required): one short sentence on what precedent you're
        looking for — used to condense the result if oversize."""
        bucket_value = bucket.strip()
        if bucket_value and (msg := _validate_bucket(bucket_value)) is not None:
            return _err(msg)
        if not query.strip():
            return json.dumps([])
        if limit < 1:
            return _err("limit must be >= 1")
        rows = search_worker_patterns(
            query=query,
            bucket=bucket_value,
            limit=limit,
        )
        return maybe_condense(
            json.dumps([_serialize_worker_pattern(p) for p in rows], default=str),
            intent=intent,
            tool_name="worker_pattern_search",
        )

    # ----------------------------------------------- agent result inbox

    @mcp.tool()
    def submit_worker_result(
        result_token: str,
        report: str,
        summary: str,
        produces: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> str:
        """Submit the structured result of THIS worker run.

        Call this exactly once near the end of your turn, with the
        ``result_token`` value from the run context section of your
        prompt. The token is unique to this run; tasque reads the
        payload back through it.

        - ``report``: the full markdown body the user reads in the
          thread. Long is fine; tasque chunks it.
        - ``summary``: 1-2 sentences shown in the small embed
          description. Do NOT duplicate the report here.
        - ``produces``: structured data (scalars, short lists, ids,
          status flags) downstream chain steps consume. Internal —
          never rendered to the user. Default ``{}``.
        - ``error``: short string identifying an application-level
          failure that should mark THIS chain step as failed. Pass
          this when your directive completed deterministically but
          the underlying action did not succeed (e.g. browser
          automation timed out, external API returned an error,
          gates failed). Leave as ``None`` (the default) for normal
          successful runs. The chain engine surfaces this on the
          step's ``failure_reason`` field, flips the step to
          ``failed``, and applies the step's ``on_failure`` policy
          (halt / replan); if any step in the run ends up failed,
          the overall ``ChainRun.status`` finalises as ``failed``
          even when every worker turn returned cleanly.

        Returns ``{"ok": true}`` on success or ``{"ok": false,
        "error": "..."}`` on validation failure (token must be a
        non-empty string, report/summary must be strings).
        """
        # The runtime ``isinstance`` checks below defend against malformed
        # MCP calls (the schema declares ``str`` but a buggy host could
        # still pass a number or null). Pyright sees the annotation and
        # flags the checks as unnecessary; the suppressions keep the
        # defensive guards in place. Same pattern in the other submit_*
        # tools below.
        if not isinstance(result_token, str) or not result_token.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("result_token must be a non-empty string")
        if not isinstance(report, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("report must be a string")
        if not isinstance(summary, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("summary must be a string")
        produces_dict: dict[str, Any] = produces or {}
        if not isinstance(produces_dict, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("produces must be an object (or omitted)")
        if error is not None and not isinstance(error, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("error must be a string or omitted")
        error_value = error.strip() if isinstance(error, str) else None  # pyright: ignore[reportUnnecessaryIsInstance]
        if error_value == "":
            error_value = None
        result_inbox.deposit(
            result_token=result_token,
            agent_kind="worker",
            payload={
                "report": report,
                "summary": summary,
                "produces": produces_dict,
                "error": error_value,
            },
        )
        return _ok()

    @mcp.tool()
    def submit_coach_result(
        result_token: str,
        thread_post: str | None = None,
    ) -> str:
        """Submit the structured result of THIS bucket-coach run.

        Call this exactly once near the end of your turn, with the
        ``result_token`` from the run context. Notes, jobs, signals,
        and chain fires happen via their own MCP tools mid-turn — this
        tool only carries the one declarative output: whether to post
        a markdown message to the bucket's Discord thread.

        - ``thread_post``: a non-empty markdown string when you want
          tasque to post to the bucket's thread, or ``None`` (the
          common case) for a quiet run.
        """
        if not isinstance(result_token, str) or not result_token.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("result_token must be a non-empty string")
        if thread_post is not None and not isinstance(thread_post, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("thread_post must be a string or null")
        result_inbox.deposit(
            result_token=result_token,
            agent_kind="coach",
            payload={"thread_post": thread_post},
        )
        return _ok()

    @mcp.tool()
    def submit_planner_result(
        result_token: str,
        mutations: list[dict[str, Any]] | None = None,
    ) -> str:
        """Submit the structured result of THIS chain-planner run.

        Call this exactly once with a list of plan mutations to apply
        atomically. Each mutation is a dict with an ``op`` field; the
        valid ops and their fields:

        - ``{"op": "add_step", "node": {<plan node dict>}}`` — add a
          new worker / approval node. ``origin`` is forced to
          ``"planner"``.
        - ``{"op": "remove_step", "id": "<step_id>"}``
        - ``{"op": "reorder_deps", "id": "<step_id>", "depends_on":
          [...]}``
        - ``{"op": "abort_chain"}`` — halt the chain.

        Empty list is the right answer when the failure is terminal
        and nothing remediates it. Pass an empty list, not null.
        """
        if not isinstance(result_token, str) or not result_token.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("result_token must be a non-empty string")
        muts = mutations if mutations is not None else []
        if not isinstance(muts, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("mutations must be a list (or omitted)")
        result_inbox.deposit(
            result_token=result_token,
            agent_kind="planner",
            payload={"mutations": muts},
        )
        return _ok()

    @mcp.tool()
    def submit_aim_chain_plan_result(
        result_token: str,
        plan: dict[str, Any] | None = None,
    ) -> str:
        """Submit the structured result of THIS Aim-to-chain planning run.

        Call this exactly once with a complete Tasque chain spec in
        ``plan``. The caller validates the spec with ``validate_spec`` and
        then either queues it as an ad-hoc chain or saves it as a template.
        """
        if not isinstance(result_token, str) or not result_token.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("result_token must be a non-empty string")
        if not isinstance(plan, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("plan must be an object")
        result_inbox.deposit(
            result_token=result_token,
            agent_kind="aim_chain_plan",
            payload={"plan": plan},
        )
        return _ok()

    @mcp.tool()
    def submit_strategist_result(
        result_token: str,
        summary: str,
        new_aims: list[dict[str, Any]] | None = None,
        signals: list[dict[str, Any]] | None = None,
        aim_status_changes: list[dict[str, Any]] | None = None,
    ) -> str:
        """Submit the structured result of THIS strategist monitoring run.

        Call this exactly once with the markdown summary and any
        Aims / Signals / status changes you want tasque to apply.

        - ``summary``: non-empty markdown posted verbatim to the
          strategist Discord thread.
        - ``new_aims``: list of Aim dicts to create. Each item:
          ``{"title", "scope": "long_term"|"bucket",
            "bucket": str|null, "target_date": "YYYY-MM-DD"|null,
            "description": str, "parent_id": str|null}``.
        - ``signals``: list of Signal dicts. Each item:
          ``{"to_bucket", "kind", "urgency", "summary", "body",
            "expires_at": str|null}``.
        - ``aim_status_changes``: list of status flips. Each item:
          ``{"aim_id", "status": "active"|"completed"|"dropped",
            "reason": str}``.

        Pass empty lists (not null) for sections you don't need.
        """
        if not isinstance(result_token, str) or not result_token.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("result_token must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("summary must be a non-empty string")
        new_aims_list = new_aims if new_aims is not None else []
        signals_list = signals if signals is not None else []
        status_changes_list = (
            aim_status_changes if aim_status_changes is not None else []
        )
        if not isinstance(new_aims_list, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("new_aims must be a list (or omitted)")
        if not isinstance(signals_list, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("signals must be a list (or omitted)")
        if not isinstance(status_changes_list, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            return _err("aim_status_changes must be a list (or omitted)")
        result_inbox.deposit(
            result_token=result_token,
            agent_kind="strategist",
            payload={
                "summary": summary,
                "new_aims": new_aims_list,
                "signals": signals_list,
                "aim_status_changes": status_changes_list,
            },
        )
        return _ok()

    # --------------------------------------------------- chain templates

    @mcp.tool()
    def chain_template_create(plan_json: str, recurrence: str | None = None) -> str:
        """Register a new ChainTemplate. ``plan_json`` is the JSON-encoded
        chain spec ({chain_name, bucket, plan, ...}). The spec's
        ``recurrence`` field and the ``recurrence`` argument must agree
        (both null for one-shot, both the same cron for recurring).

        See chain spec docs for the plan schema; common shape:
        ``{"chain_name": "...", "bucket": "...", "plan": [{"id": ...,
        "kind": "worker", "directive": "...", "tier": "..."}]}``.

        Returns the new chain_name on success."""
        try:
            plan = json.loads(plan_json)
        except json.JSONDecodeError as exc:
            return _err(f"plan_json did not parse: {exc}")
        if not isinstance(plan, dict):
            return _err("plan_json must encode a JSON object")
        if recurrence is not None:
            err = validate_cron(recurrence)
            if err is not None:
                return _err(f"recurrence: {err}")
        try:
            name = _create_template(cast(dict[str, Any], plan), recurrence=recurrence)
        except (SpecError, MirrorMismatch, ValueError) as exc:
            return _err(f"{type(exc).__name__}: {exc}")
        return _ok(chain_name=name)

    @mcp.tool()
    def chain_template_get(intent: str, name: str) -> str:
        """Fetch one ChainTemplate by ``chain_name`` (full plan dict
        included), or {"ok": false, "error": "missing"}.

        ``intent`` (required): one short sentence on why you're reading
        this template — used to condense the result if oversize (large
        plans with many steps can blow past the threshold)."""
        row = _get_template(name)
        if row is None:
            return _err("missing")
        return maybe_condense(
            json.dumps(row, default=str),
            intent=intent,
            tool_name="chain_template_get",
        )

    @mcp.tool()
    def chain_template_list(
        intent: str,
        bucket: str | None = None,
        enabled_only: bool = False,
    ) -> str:
        """List all ChainTemplates as a JSON array. Optionally filter by
        ``bucket`` and/or ``enabled_only=True``.

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize."""
        if bucket is not None and (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        rows = _list_templates(enabled_only=enabled_only)
        if bucket is not None:
            rows = [r for r in rows if r.get("bucket") == bucket]
        return maybe_condense(
            json.dumps(rows, default=str),
            intent=intent,
            tool_name="chain_template_list",
        )

    @mcp.tool()
    def chain_template_update(
        name: str,
        plan_json: str | None = None,
        recurrence: str | None = None,
        recurrence_clear: bool = False,
        enabled: bool | None = None,
    ) -> str:
        """Update a ChainTemplate. Pass ``recurrence_clear=True`` to wipe
        the cron (otherwise omit ``recurrence`` to leave it untouched).
        When updating ``plan_json``, the spec's ``recurrence`` must agree
        with the row's effective recurrence after this update."""
        plan: dict[str, Any] | None = None
        if plan_json is not None:
            try:
                parsed = json.loads(plan_json)
            except json.JSONDecodeError as exc:
                return _err(f"plan_json did not parse: {exc}")
            if not isinstance(parsed, dict):
                return _err("plan_json must encode a JSON object")
            plan = cast(dict[str, Any], parsed)
        if recurrence is not None:
            err = validate_cron(recurrence)
            if err is not None:
                return _err(f"recurrence: {err}")
        if recurrence_clear:
            effective_rec: Any = None
        elif recurrence is not None:
            effective_rec = recurrence
        else:
            effective_rec = _CRUD_UNSET
        try:
            ok = _update_template(
                name,
                plan_dict=plan,
                recurrence=effective_rec,
                enabled=enabled,
            )
        except (SpecError, MirrorMismatch, ValueError) as exc:
            return _err(f"{type(exc).__name__}: {exc}")
        if not ok:
            return _err(f"no chain template named {name!r}")
        return _ok(chain_name=name)

    @mcp.tool()
    def chain_template_delete(name: str) -> str:
        """Hard-delete a ChainTemplate. Existing ChainRuns referencing
        it have their ``template_id`` nulled (history preserved)."""
        if not _delete_template(name):
            return _err(f"no chain template named {name!r}")
        return _ok(chain_name=name, deleted=True)

    # -------------------------------------------------------- chain runs

    @mcp.tool()
    def chain_fire_template(
        name: str,
        thread_id: str | None = None,
        vars: dict[str, Any] | None = None,
    ) -> str:
        """One-shot launch a saved ChainTemplate by name.

        **Fire-and-forget**: returns the new ``chain_id`` in
        milliseconds. The daemon-owned chain runner claims and invokes
        the row via a DB lease. Use ``chain_run_get`` to poll status,
        or watch the chain status thread the daemon publishes.
        ``thread_id`` is normally None; the daemon's chain-status
        watcher creates the Discord thread.

        ``vars`` is an optional run-time override dict, merged over the
        template's static ``vars`` (caller wins on key collision).
        Workers see the merged dict in their prompt context, so
        directives can branch on it (e.g. ``{"force": true}`` to skip
        an age gate)."""
        row = _get_template(name)
        if row is None:
            return _err(f"no chain template named {name!r}")
        plan = row.get("plan")
        if not isinstance(plan, dict):
            return _err(f"chain template {name!r} has malformed plan_json")
        try:
            chain_id = launch_chain_run(
                cast(dict[str, Any], plan),
                template_id=row.get("id") if isinstance(row.get("id"), str) else None,
                thread_id=thread_id,
                vars=vars,
                wait=False,
            )
        except (SpecError, MirrorMismatch, ValueError) as exc:
            return _err(f"{type(exc).__name__}: {exc}")
        return _ok(chain_id=chain_id, chain_name=name)

    @mcp.tool()
    def chain_queue_adhoc(
        plan_json: str,
        thread_id: str | None = None,
        vars: dict[str, Any] | None = None,
    ) -> str:
        """One-shot launch an ad-hoc chain (no persisted template).
        ``plan_json`` is the JSON-encoded chain spec.

        **Fire-and-forget**: returns the new ``chain_id`` in
        milliseconds. The daemon-owned chain runner claims and invokes
        the row via a DB lease. Use this for one-off multi-step work;
        if you'll run the same shape repeatedly,
        ``chain_template_create`` it first instead.

        ``vars`` works the same as for ``chain_fire_template``: an
        optional run-time override dict merged over the spec's static
        ``vars`` and surfaced to every worker prompt."""
        try:
            plan = json.loads(plan_json)
        except json.JSONDecodeError as exc:
            return _err(f"plan_json did not parse: {exc}")
        if not isinstance(plan, dict):
            return _err("plan_json must encode a JSON object")
        try:
            validate_spec(cast(dict[str, Any], plan))
            chain_id = launch_chain_run(
                cast(dict[str, Any], plan),
                thread_id=thread_id,
                vars=vars,
                wait=False,
            )
        except (SpecError, MirrorMismatch, ValueError) as exc:
            return _err(f"{type(exc).__name__}: {exc}")
        return _ok(chain_id=chain_id)

    @mcp.tool()
    def chain_run_get(
        intent: str,
        chain_id: str,
        include_state: bool = False,
    ) -> str:
        """Fetch one ChainRun row by ``chain_id``. With
        ``include_state=True`` also includes the live checkpoint state
        (plan, completed, failures, history) under a ``state`` key.
        Returns {"ok": false, "error": "missing"} if no such run.

        ``intent`` (required): one short sentence on what you're reading
        the run for (e.g. 'is the resume step blocked', 'which fan-out
        children failed') — used to condense the result if oversize.
        ``include_state=True`` payloads regularly cross the threshold."""
        with get_session() as sess:
            stmt = select(ChainRun).where(ChainRun.chain_id == chain_id)
            row = sess.execute(stmt).scalars().first()
            if row is None:
                return _err("missing")
            sess.expunge(row)
        out: dict[str, Any] = _serialize_chain_run(row)
        if include_state:
            state = _get_chain_state(chain_id)
            out["state"] = state
        return maybe_condense(
            json.dumps(out, default=str),
            intent=intent,
            tool_name="chain_run_get",
        )

    @mcp.tool()
    def chain_run_list(
        intent: str,
        bucket: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> str:
        """List ChainRuns, newest first. Optional filters: ``bucket``,
        ``status`` (running/completed/failed/stopped/awaiting_approval/
        awaiting_user/paused).

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize."""
        if bucket is not None and (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if limit < 1:
            return _err("limit must be >= 1")
        with get_session() as sess:
            stmt = select(ChainRun)
            if bucket is not None:
                stmt = stmt.where(ChainRun.bucket == bucket)
            if status is not None:
                stmt = stmt.where(ChainRun.status == status)
            stmt = stmt.order_by(ChainRun.created_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return maybe_condense(
            json.dumps([_serialize_chain_run(r) for r in rows]),
            intent=intent,
            tool_name="chain_run_list",
        )

    @mcp.tool()
    def chain_run_pause(chain_id: str) -> str:
        """Mark a ChainRun ``paused`` (skipped on daemon restart).
        Active dispatches in flight will complete; no new step is
        claimed until the chain is resumed."""
        if not _pause_chain(chain_id):
            return _err(f"no chain run {chain_id!r}")
        return _ok(chain_id=chain_id, status="paused")

    @mcp.tool()
    def chain_run_resume(chain_id: str) -> str:
        """Flip a paused ChainRun back to ``running``."""
        if not _resume_chain(chain_id):
            return _err(f"no chain run {chain_id!r}")
        return _ok(chain_id=chain_id, status="running")

    @mcp.tool()
    def chain_run_stop(chain_id: str) -> str:
        """Stop a chain run immediately. Flips the row to ``stopped``
        and walks the checkpoint flipping every pending/running/
        awaiting_user plan node to ``stopped`` so the next supervisor
        pass terminates."""
        if not _stop_chain(chain_id):
            return _err(f"no chain run {chain_id!r}")
        return _ok(chain_id=chain_id, status="stopped")

    # ----------------------------------------------------------- signals

    @mcp.tool()
    def signal_create(
        from_bucket: str,
        to_bucket: str,
        kind: str,
        urgency: str,
        summary: str,
        body: str = "",
        context: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> str:
        """Send a cross-bucket signal. ``from_bucket`` is one of the nine
        canonical buckets, or ``"strategist"`` (when sent by the
        strategist agent), or ``"system"``. ``to_bucket`` is one of the
        nine buckets or ``"all"`` to broadcast. ``kind`` is a short
        label (e.g. "income-change", "deadline", "blocker", "aim_added",
        "rebalance"). ``urgency`` is one of "low", "normal", "high",
        "critical". ``expires_at`` is None or an ISO-8601 UTC timestamp
        after which the signal stops appearing in queries.

        Use this to nudge another bucket coach about an event in yours
        — the receiving coach sees signals in its next reactive run."""
        if from_bucket not in _ALL_SIGNAL_FROMS:
            return _err(
                f"from_bucket must be one of {sorted(_ALL_SIGNAL_FROMS)!r}, "
                f"got {from_bucket!r}"
            )
        if to_bucket != "all" and (msg := _validate_bucket(to_bucket)) is not None:
            return _err(f"to_bucket: {msg} (or 'all')")
        if not summary.strip():
            return _err("summary must be a non-empty string")
        if urgency not in ("low", "normal", "high", "critical"):
            return _err(
                f"urgency must be one of ('low', 'normal', 'high', 'critical'), got {urgency!r}"
            )
        s = Signal(
            from_bucket=from_bucket,
            to_bucket=to_bucket,
            kind=kind,
            urgency=urgency,
            summary=summary,
            body=body,
            context=context,
            expires_at=expires_at,
        )
        written = write_entity(s)
        dispatch_tool_event(
            "signal_create",
            from_bucket=from_bucket,
            to_bucket=to_bucket,
            kind=kind,
        )
        return _ok(id=written.id)

    @mcp.tool()
    def signal_list(
        intent: str,
        to_bucket: str | None = None,
        include_archived: bool = False,
        limit: int = 20,
    ) -> str:
        """List signals, newest first.

        With ``to_bucket`` set, returns active signals addressed to that
        bucket (and any broadcast to ``"all"``). With ``to_bucket=None``,
        returns the most recent signals across every from/to pairing —
        the cross-bucket view used by the strategist.

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize."""
        if (
            to_bucket is not None
            and to_bucket != "all"
            and (msg := _validate_bucket(to_bucket)) is not None
        ):
            return _err(msg)
        if limit < 1:
            return _err("limit must be >= 1")
        with get_session() as sess:
            stmt = select(Signal)
            if to_bucket is not None:
                stmt = stmt.where(
                    (Signal.to_bucket == to_bucket) | (Signal.to_bucket == "all")
                )
            if not include_archived:
                stmt = stmt.where(Signal.archived.is_(False))
            stmt = stmt.order_by(Signal.created_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return maybe_condense(
            json.dumps([_serialize_signal(s) for s in rows]),
            intent=intent,
            tool_name="signal_list",
        )

    @mcp.tool()
    def signal_archive(signal_id: str) -> str:
        """Archive a Signal (sets ``archived=True``). Idempotent."""
        if _archive_entity(signal_id):
            return _ok(id=signal_id, archived=True)
        return _err(f"no Signal with id={signal_id}")

    # -------------------------------------------------------------- aims

    @mcp.tool()
    def aim_create(
        title: str,
        scope: str,
        bucket: str | None = None,
        target_date: str | None = None,
        description: str = "",
        parent_id: str | None = None,
        source: str = "mcp",
    ) -> str:
        """Create a new Aim. ``scope`` is "long_term" or "bucket".

        For ``scope="long_term"``, ``bucket`` MUST be None — long-term
        Aims live above any single bucket. For ``scope="bucket"``,
        ``bucket`` MUST be one of the nine canonical buckets, and
        typically ``parent_id`` points at the long-term Aim being
        decomposed. ``target_date`` is "YYYY-MM-DD" or None.
        ``source`` is a label (defaults to "mcp"; the strategist
        passes ``source="strategist"``)."""
        if not title.strip():
            return _err("title must be a non-empty string")
        if scope not in ("long_term", "bucket"):
            return _err(f"scope must be 'long_term' or 'bucket', got {scope!r}")
        bucket_value: str | None = None
        if scope == "bucket":
            if bucket is None:
                return _err("scope='bucket' requires a bucket argument")
            if (msg := _validate_bucket(bucket)) is not None:
                return _err(msg)
            bucket_value = bucket
        elif bucket is not None:
            return _err("scope='long_term' must have bucket=None")
        a = Aim(
            title=title,
            bucket=bucket_value,
            scope=scope,
            target_date=target_date,
            description=description,
            status="active",
            parent_id=parent_id,
            source=source,
        )
        written = write_entity(a)
        dispatch_tool_event(
            "aim_create",
            aim_id=written.id,
            bucket=bucket_value,
            scope=scope,
            source=source,
        )
        return _ok(id=written.id)

    @mcp.tool()
    def aim_get(intent: str, aim_id: str) -> str:
        """Fetch one Aim by id, or {"ok": false, "error": "missing"}.

        ``intent`` (required): one short sentence on why you're reading
        this aim — used to condense the result if oversize (long-term
        aims with verbose descriptions occasionally cross the threshold)."""
        with get_session() as sess:
            row = sess.get(Aim, aim_id)
            if row is None:
                return _err("missing")
            sess.expunge(row)
        return maybe_condense(
            json.dumps(_serialize_aim(row)),
            intent=intent,
            tool_name="aim_get",
        )

    @mcp.tool()
    def aim_update(
        aim_id: str,
        status: str | None = None,
        description: str | None = None,
        target_date: str | None = None,
        title: str | None = None,
    ) -> str:
        """Update fields on an existing Aim. Only provide fields you
        intend to change. ``status`` is "active", "completed", or
        "dropped"."""
        if status is not None and status not in ("active", "completed", "dropped"):
            return _err(
                f"status must be 'active', 'completed', or 'dropped', got {status!r}"
            )
        with get_session() as sess:
            row = sess.get(Aim, aim_id)
            if row is None:
                return _err(f"no Aim with id={aim_id}")
            if status is not None:
                row.status = status
            if description is not None:
                row.description = description
            if target_date is not None:
                row.target_date = target_date
            if title is not None:
                if not title.strip():
                    return _err("title must be non-empty")
                row.title = title
            row.updated_at = utc_now_iso()
        return _ok(id=aim_id)

    @mcp.tool()
    def aim_list(
        intent: str,
        scope: str | None = None,
        bucket: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> str:
        """Return Aims as a JSON array (newest first). All filters
        optional. ``scope`` is "long_term" or "bucket". ``bucket`` is
        one of the nine canonical buckets. ``status`` is "active",
        "completed", or "dropped".

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize."""
        if scope is not None and scope not in ("long_term", "bucket"):
            return _err(f"scope must be 'long_term' or 'bucket', got {scope!r}")
        if bucket is not None and (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if status is not None and status not in ("active", "completed", "dropped"):
            return _err(
                f"status must be 'active', 'completed', or 'dropped', got {status!r}"
            )
        if limit < 1:
            return _err("limit must be >= 1")
        with get_session() as sess:
            stmt = select(Aim)
            if scope is not None:
                stmt = stmt.where(Aim.scope == scope)
            if bucket is not None:
                stmt = stmt.where(Aim.bucket == bucket)
            if status is not None:
                stmt = stmt.where(Aim.status == status)
            stmt = stmt.order_by(Aim.created_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return maybe_condense(
            json.dumps([_serialize_aim(a) for a in rows]),
            intent=intent,
            tool_name="aim_list",
        )

    @mcp.tool()
    def aim_plan_chain(
        aim_id: str,
        mode: str = "adhoc",
        enabled: bool = False,
        thread_id: str | None = None,
        planner_tier: str = "large",
    ) -> str:
        """Plan an Aim into a validated Tasque chain spec.

        ``mode="adhoc"`` queues the validated spec as a one-shot chain.
        ``mode="template"`` saves it as a ChainTemplate; templates are
        disabled by default here, and only created enabled when
        ``enabled=True`` is passed explicitly.
        """
        normalized_mode = mode.strip().lower()
        if normalized_mode not in ("adhoc", "template"):
            return _err("mode must be 'adhoc' or 'template'")
        if (msg := _validate_tier(planner_tier)) is not None:
            return _err(msg)

        from tasque.strategist.graph import plan_chain_for_aim

        planned = plan_chain_for_aim(
            aim_id,
            mode=normalized_mode,
            planner_tier=planner_tier,
        )
        if not planned.get("ok"):
            return json.dumps(planned, default=str)
        spec = cast(dict[str, Any], planned.get("spec"))

        if normalized_mode == "adhoc":
            existing = _find_existing_aim_chain_run(spec)
            if existing is not None:
                return _ok(
                    mode=normalized_mode,
                    chain_id=existing.chain_id,
                    chain_name=existing.chain_name,
                    spec=spec,
                    deduped=True,
                    existing_status=existing.status,
                )
            try:
                chain_id = launch_chain_run(
                    spec,
                    thread_id=thread_id,
                    wait=False,
                )
            except (SpecError, MirrorMismatch, ValueError) as exc:
                return _err(f"{type(exc).__name__}: {exc}", spec=spec)
            return _ok(
                mode=normalized_mode,
                chain_id=chain_id,
                chain_name=spec.get("chain_name"),
                spec=spec,
            )

        recurrence = cast(str | None, spec.get("recurrence"))
        try:
            name = _create_template(
                spec,
                recurrence=recurrence,
                enabled=enabled,
            )
        except (SpecError, MirrorMismatch, ValueError) as exc:
            return _err(f"{type(exc).__name__}: {exc}", spec=spec)
        return _ok(
            mode=normalized_mode,
            chain_name=name,
            enabled=enabled,
            template=_get_template(name),
            spec=spec,
        )

    # ------------------------------------------------- cross-bucket view

    @mcp.tool()
    def bucket_summary(intent: str) -> str:
        """One-line per-bucket snapshot across all nine buckets:
        active aim count, pending job count, unresolved failure count,
        and signal count (in or out, unarchived). Returns a JSON object
        keyed by bucket name.

        ``intent`` (required): one short sentence on what you're looking
        for — used to condense the result if oversize (the snapshot is
        almost always under the threshold but agents should still write
        the intent so the API stays uniform)."""
        out: dict[str, dict[str, int]] = {}
        with get_session() as sess:
            for b in ALL_BUCKETS:
                aims = sess.execute(
                    select(Aim).where(Aim.bucket == b).where(Aim.status == "active")
                ).scalars().all()
                jobs = sess.execute(
                    select(QueuedJob)
                    .where(QueuedJob.bucket == b)
                    .where(QueuedJob.status == "pending")
                ).scalars().all()
                failures = sess.execute(
                    select(FailedJob)
                    .where(FailedJob.bucket == b)
                    .where(FailedJob.resolved.is_(False))
                ).scalars().all()
                sigs = sess.execute(
                    select(Signal)
                    .where((Signal.to_bucket == b) | (Signal.from_bucket == b))
                    .where(Signal.archived.is_(False))
                ).scalars().all()
                out[b] = {
                    "active_aims": len(list(aims)),
                    "pending_jobs": len(list(jobs)),
                    "unresolved_failures": len(list(failures)),
                    "signals": len(list(sigs)),
                }
            sess.expunge_all()
        return maybe_condense(
            json.dumps(out), intent=intent, tool_name="bucket_summary"
        )

    # The @mcp.tool() decorator registers each function on ``mcp`` as a
    # side effect; pyright's reportUnusedFunction can't see that, so we
    # bind the inner names here to mark them deliberately registered.
    _ = (
        note_create, note_update, note_supersede,
        note_get, note_list, note_search, note_search_any,
        note_search_fts, note_archive,
        job_create, job_get, job_update, job_cancel, job_list,
        worker_pattern_search,
        submit_worker_result,
        submit_coach_result,
        submit_planner_result,
        submit_aim_chain_plan_result,
        submit_strategist_result,
        chain_template_create, chain_template_get, chain_template_list,
        chain_template_update, chain_template_delete,
        chain_fire_template, chain_queue_adhoc,
        chain_run_get, chain_run_list, chain_run_pause, chain_run_resume,
        chain_run_stop,
        signal_create, signal_list, signal_archive,
        aim_create, aim_get, aim_update, aim_list, aim_plan_chain,
        bucket_summary,
    )

    return mcp


def run_stdio() -> None:
    """Run the tasque MCP server over stdio. Blocks until the parent
    process closes the pipe (which is how stdio MCPs shut down).

    Intended to be invoked by a host that registers tasque in its MCP
    config, spawned anew per request and inherited by the proxy.
    """
    server = build_server()
    server.run("stdio")


__all__ = ["build_server", "run_stdio"]

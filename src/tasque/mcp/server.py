"""The tasque stdio MCP server.

Exposes the daemon's full write surface — notes, queued jobs, aims,
chain templates, chain runs, signals — as MCP tools that any
``claude --print`` invocation through the proxy can call mid-turn.
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
from datetime import UTC, datetime
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
from tasque.memory.db import get_session
from tasque.memory.entities import (
    Aim,
    ChainRun,
    FailedJob,
    Note,
    QueuedJob,
    Signal,
    utc_now_iso,
)
from tasque.memory.repo import (
    archive as _archive_entity,
)
from tasque.memory.repo import (
    update_entity_status,
    write_entity,
)

# Cross-bucket "actor" identifiers permitted for Signal.from_bucket. The
# nine canonical buckets are valid; the strategist agent and ad-hoc
# system signals also need to be representable.
_ALL_SIGNAL_FROMS: frozenset[str] = frozenset({*ALL_BUCKETS, "strategist", "system"})

if TYPE_CHECKING:  # pragma: no cover - imported only for type hints
    from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------- helpers


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, default=str)


def _err(message: str) -> str:
    return json.dumps({"ok": False, "error": message})


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
        "source": n.source,
        "archived": n.archived,
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


def _serialize_chain_run(r: ChainRun) -> dict[str, Any]:
    return {
        "chain_id": r.chain_id,
        "chain_name": r.chain_name,
        "bucket": r.bucket,
        "status": r.status,
        "template_id": r.template_id,
        "thread_id": r.thread_id,
        "started_at": r.started_at,
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
            "validation/runtime failure."
        ),
    )

    # ------------------------------------------------------------- notes

    @mcp.tool()
    def note_create(
        content: str,
        bucket: str,
        durability: str = "ephemeral",
        source: str = "mcp",
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Write a Note. ``durability`` is "ephemeral" (decays after ~30
        days), "durable" (long-lived fact), or "behavioral" (always-honor
        instruction). ``source`` is a short label identifying who wrote
        it (defaults to "mcp"). Returns the new note id."""
        if not content.strip():
            return _err("content must be a non-empty string")
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        n = Note(
            content=content,
            bucket=bucket,
            durability=durability,
            source=source,
            meta=meta or {},
        )
        written = write_entity(n)
        return _ok(id=written.id)

    @mcp.tool()
    def note_get(note_id: str) -> str:
        """Fetch one note by id (full untruncated content), or
        {"ok": false, "error": "missing"} if no such note."""
        with get_session() as sess:
            row = sess.get(Note, note_id)
            if row is None:
                return _err("missing")
            sess.expunge(row)
        return json.dumps(_serialize_note(row))

    @mcp.tool()
    def note_list(
        bucket: str,
        durability: str | None = None,
        include_archived: bool = False,
        limit: int = 20,
    ) -> str:
        """List notes in ``bucket`` (newest first). Optional filters:
        ``durability`` ('ephemeral' / 'durable' / 'behavioral') and
        ``include_archived`` (default False). Returns a JSON array."""
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if durability is not None and (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        if limit < 1:
            return _err("limit must be >= 1")
        with get_session() as sess:
            stmt = select(Note).where(Note.bucket == bucket)
            if not include_archived:
                stmt = stmt.where(Note.archived.is_(False))
            if durability is not None:
                stmt = stmt.where(Note.durability == durability)
            stmt = stmt.order_by(Note.updated_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return json.dumps([_serialize_note(n) for n in rows])

    @mcp.tool()
    def note_search(
        bucket: str,
        query: str,
        durability: str | None = None,
        limit: int = 10,
    ) -> str:
        """Substring search (case-insensitive) over note content within
        ``bucket``. Returns up to ``limit`` matches newest-first; each
        note's content is truncated to ~400 chars in the result — call
        ``note_get`` for full content."""
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if durability is not None and (msg := _validate_durability(durability)) is not None:
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
            stmt = stmt.order_by(Note.updated_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return json.dumps([_serialize_note(n, truncate=400) for n in rows])

    @mcp.tool()
    def note_search_any(
        bucket: str,
        keywords: list[str],
        durability: str | None = None,
        limit: int = 10,
    ) -> str:
        """Like ``note_search`` but matches any keyword (OR-style)."""
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if durability is not None and (msg := _validate_durability(durability)) is not None:
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
            stmt = stmt.order_by(Note.updated_at.desc()).limit(limit)
            rows = list(sess.execute(stmt).scalars().all())
            sess.expunge_all()
        return json.dumps([_serialize_note(n, truncate=400) for n in rows])

    @mcp.tool()
    def note_search_fts(
        bucket: str,
        query: str,
        durability: str | None = None,
        limit: int = 10,
    ) -> str:
        """SQLite FTS5 search over note content (BM25-ranked, word-tokenized,
        stemmed). Prefer this when you want word-boundary matches and
        relevance-ordered results. Query syntax: bare words AND, ``OR``
        for alternatives, ``"phrase"`` exact, ``-word`` exclude,
        ``NEAR(x y, 5)`` proximity. Returns the literal string
        ``"fts5_unavailable"`` if the host SQLite lacks FTS5."""
        if (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        if durability is not None and (msg := _validate_durability(durability)) is not None:
            return _err(msg)
        if not query.strip():
            return json.dumps([])
        if limit < 1:
            return _err("limit must be >= 1")
        params: dict[str, Any] = {"bucket": bucket, "query": query, "limit": limit}
        sql = (
            "SELECT n.id, n.content, n.bucket, n.durability, n.source, "
            "  n.archived, n.created_at, n.updated_at, n.metadata "
            "FROM notes_fts JOIN notes n ON n.id = notes_fts.note_id "
            "WHERE notes_fts.bucket = :bucket "
            "  AND notes_fts.archived = 0 "
            "  AND notes_fts MATCH :query "
        )
        if durability is not None:
            sql += "  AND notes_fts.durability = :durability "
            params["durability"] = durability
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
                }
            )
        return json.dumps(out)

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
        "opus", "sonnet", "haiku". Pick haiku for trivial nudges, sonnet
        for multi-step tool / scrape / summarize work, opus for agentic
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
    def job_get(job_id: str) -> str:
        """Fetch one QueuedJob by id, or {"ok": false, "error": "missing"}."""
        with get_session() as sess:
            row = sess.get(QueuedJob, job_id)
            if row is None:
                return _err("missing")
            sess.expunge(row)
        return json.dumps(_serialize_queued_job(row))

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
        bucket: str | None = None,
        status: str | None = None,
        recurring_only: bool = False,
        limit: int = 30,
    ) -> str:
        """List QueuedJobs, newest first. All filters are optional —
        omit ``bucket`` to list across all buckets. ``status`` filters
        by lifecycle state (pending/claimed/completed/failed/stopped).
        ``recurring_only=True`` keeps only rows with a non-null
        recurrence cron."""
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
        return json.dumps([_serialize_queued_job(j) for j in rows])

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
        if not isinstance(result_token, str) or not result_token.strip():
            return _err("result_token must be a non-empty string")
        if not isinstance(report, str):
            return _err("report must be a string")
        if not isinstance(summary, str):
            return _err("summary must be a string")
        produces_dict: dict[str, Any] = produces or {}
        if not isinstance(produces_dict, dict):
            return _err("produces must be an object (or omitted)")
        if error is not None and not isinstance(error, str):
            return _err("error must be a string or omitted")
        error_value = error.strip() if isinstance(error, str) else None
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
        if not isinstance(result_token, str) or not result_token.strip():
            return _err("result_token must be a non-empty string")
        if thread_post is not None and not isinstance(thread_post, str):
            return _err("thread_post must be a string or null")
        result_inbox.deposit(
            result_token=result_token,
            agent_kind="coach",
            payload={"thread_post": thread_post},
        )
        return _ok()

    @mcp.tool()
    def claim_idle_silence(seconds: int, reason: str) -> str:
        """Tell the tasque proxy you're about to be silent for ``seconds``.

        Call this BEFORE you start any tool call you expect to take
        more than ~2 minutes (model training, large download, long
        sleep, slow scrape). The proxy spawned this run inside a
        ``claude --print`` subprocess and runs a stall watchdog: if
        stdout produces zero bytes for longer than its threshold AND
        no idle-silence budget is in effect, it kills the subprocess
        as hung. Without this call, a legitimate 30-min training job
        looks identical to a hang and gets killed.

        With this call, the watchdog grants the silence — no kill
        until the budget expires, even with stdout completely silent.
        If you're still mid-work when the budget runs out, call
        ``claim_idle_silence`` again to extend.

        - ``seconds``: how long you expect to be silent. Must be > 0.
          Honest estimate is fine; over-budget always re-engages the
          watchdog so a hung tool past your estimate still gets caught.
        - ``reason``: short label ("training task1", "scraping
          leaderboard", "sleeping for cron") — surfaced in
          ``/status`` and in stall logs.

        Returns ``{"ok": true, "granted_until_iso": "...", "remaining_s":
        <float>}`` on success, or ``{"ok": false, "error": "..."}``
        when not running under the proxy (no ``TASQUE_PROXY_REQUEST_ID``
        in env — common for CLI-driven runs that don't go through the
        proxy at all) or on a transport error.

        Idempotent — extends rather than resets. Calling twice with
        ``seconds=600`` doesn't grant 1200s; the larger of the two
        ``now + seconds`` deadlines wins.
        """
        if not isinstance(seconds, int) or seconds <= 0:
            return _err("seconds must be a positive integer")
        if not isinstance(reason, str) or not reason.strip():
            return _err("reason must be a non-empty string")
        request_id = os.environ.get("TASQUE_PROXY_REQUEST_ID")
        base_url = os.environ.get("TASQUE_PROXY_INTERNAL_URL")
        if not request_id or not base_url:
            return _err(
                "no proxy context: TASQUE_PROXY_REQUEST_ID / "
                "TASQUE_PROXY_INTERNAL_URL not set in env. This run "
                "isn't going through tasque proxy, so there's no "
                "watchdog to extend. (Skip the call.)"
            )
        # Lazy import: keep ``mcp serve`` startup cost low for the
        # common case where this tool isn't called.
        import urllib.error
        import urllib.request

        url = f"{base_url.rstrip('/')}/v1/internal/idle_grant/{request_id}"
        body = json.dumps({"seconds": seconds, "reason": reason}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                err_body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                err_body = {"message": str(exc)}
            return _err(
                f"proxy returned HTTP {exc.code}: "
                f"{(err_body.get('error') or {}).get('message', err_body)}"
            )
        except Exception as exc:
            return _err(f"proxy unreachable: {type(exc).__name__}: {exc}")
        return _ok(
            granted_until_iso=payload.get("granted_until_iso"),
            remaining_s=payload.get("remaining_s"),
            reason=payload.get("reason"),
        )

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
        if not isinstance(result_token, str) or not result_token.strip():
            return _err("result_token must be a non-empty string")
        muts = mutations if mutations is not None else []
        if not isinstance(muts, list):
            return _err("mutations must be a list (or omitted)")
        result_inbox.deposit(
            result_token=result_token,
            agent_kind="planner",
            payload={"mutations": muts},
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
        if not isinstance(result_token, str) or not result_token.strip():
            return _err("result_token must be a non-empty string")
        if not isinstance(summary, str) or not summary.strip():
            return _err("summary must be a non-empty string")
        new_aims_list = new_aims if new_aims is not None else []
        signals_list = signals if signals is not None else []
        status_changes_list = (
            aim_status_changes if aim_status_changes is not None else []
        )
        if not isinstance(new_aims_list, list):
            return _err("new_aims must be a list (or omitted)")
        if not isinstance(signals_list, list):
            return _err("signals must be a list (or omitted)")
        if not isinstance(status_changes_list, list):
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
    def chain_template_get(name: str) -> str:
        """Fetch one ChainTemplate by ``chain_name`` (full plan dict
        included), or {"ok": false, "error": "missing"}."""
        row = _get_template(name)
        if row is None:
            return _err("missing")
        return json.dumps(row, default=str)

    @mcp.tool()
    def chain_template_list(
        bucket: str | None = None,
        enabled_only: bool = False,
    ) -> str:
        """List all ChainTemplates as a JSON array. Optionally filter by
        ``bucket`` and/or ``enabled_only=True``."""
        if bucket is not None and (msg := _validate_bucket(bucket)) is not None:
            return _err(msg)
        rows = _list_templates(enabled_only=enabled_only)
        if bucket is not None:
            rows = [r for r in rows if r.get("bucket") == bucket]
        return json.dumps(rows, default=str)

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
        milliseconds — the chain runs in the background. Use
        ``chain_run_get`` to poll status, or watch the chain status
        thread the daemon publishes. ``thread_id`` is normally None;
        the daemon's chain-status watcher creates the Discord thread.

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
        milliseconds — the chain runs in the background. Use this for
        one-off multi-step work; if you'll run the same shape
        repeatedly, ``chain_template_create`` it first instead.

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
    def chain_run_get(chain_id: str, include_state: bool = False) -> str:
        """Fetch one ChainRun row by ``chain_id``. With
        ``include_state=True`` also includes the live checkpoint state
        (plan, completed, failures, history) under a ``state`` key.
        Returns {"ok": false, "error": "missing"} if no such run."""
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
        return json.dumps(out, default=str)

    @mcp.tool()
    def chain_run_list(
        bucket: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> str:
        """List ChainRuns, newest first. Optional filters: ``bucket``,
        ``status`` (running/completed/failed/stopped/awaiting_approval/
        awaiting_user/paused)."""
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
        return json.dumps([_serialize_chain_run(r) for r in rows])

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
        return _ok(id=written.id)

    @mcp.tool()
    def signal_list(
        to_bucket: str | None = None,
        include_archived: bool = False,
        limit: int = 20,
    ) -> str:
        """List signals, newest first.

        With ``to_bucket`` set, returns active signals addressed to that
        bucket (and any broadcast to ``"all"``). With ``to_bucket=None``,
        returns the most recent signals across every from/to pairing —
        the cross-bucket view used by the strategist."""
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
        return json.dumps([_serialize_signal(s) for s in rows])

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
        return _ok(id=written.id)

    @mcp.tool()
    def aim_get(aim_id: str) -> str:
        """Fetch one Aim by id, or {"ok": false, "error": "missing"}."""
        with get_session() as sess:
            row = sess.get(Aim, aim_id)
            if row is None:
                return _err("missing")
            sess.expunge(row)
        return json.dumps(_serialize_aim(row))

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
        scope: str | None = None,
        bucket: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> str:
        """Return Aims as a JSON array (newest first). All filters
        optional. ``scope`` is "long_term" or "bucket". ``bucket`` is
        one of the nine canonical buckets. ``status`` is "active",
        "completed", or "dropped"."""
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
        return json.dumps([_serialize_aim(a) for a in rows])

    # ------------------------------------------------- cross-bucket view

    @mcp.tool()
    def bucket_summary() -> str:
        """One-line per-bucket snapshot across all nine buckets:
        active aim count, pending job count, unresolved failure count,
        and signal count (in or out, unarchived). Returns a JSON object
        keyed by bucket name."""
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
        return json.dumps(out)

    # The @mcp.tool() decorator registers each function on ``mcp`` as a
    # side effect; pyright's reportUnusedFunction can't see that, so we
    # bind the inner names here to mark them deliberately registered.
    _ = (
        note_create, note_get, note_list, note_search, note_search_any,
        note_search_fts, note_archive,
        job_create, job_get, job_update, job_cancel, job_list,
        submit_worker_result,
        submit_coach_result,
        claim_idle_silence,
        submit_planner_result,
        submit_strategist_result,
        chain_template_create, chain_template_get, chain_template_list,
        chain_template_update, chain_template_delete,
        chain_fire_template, chain_queue_adhoc,
        chain_run_get, chain_run_list, chain_run_pause, chain_run_resume,
        chain_run_stop,
        signal_create, signal_list, signal_archive,
        aim_create, aim_get, aim_update, aim_list,
        bucket_summary,
    )

    return mcp


def run_stdio() -> None:
    """Run the tasque MCP server over stdio. Blocks until the parent
    process closes the pipe (which is how stdio MCPs shut down).

    Intended to be invoked by a host that registers tasque in its MCP
    config — typically ``claude --print`` via the user's ``~/.claude.json``,
    spawned anew per request and inherited by the proxy.
    """
    server = build_server()
    server.run("stdio")


__all__ = ["build_server", "run_stdio"]

"""SQLAlchemy ORM definitions for the eight tasque entities.

Schema evolution policy: column additions happen automatically via
``_ensure_schema()`` at startup. Type changes or removals require an
export → wipe → import cycle. There is no Alembic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now_iso() -> str:
    """ISO-8601 timestamp in UTC, microseconds + Z suffix.

    Set in Python (not via SQL ``CURRENT_TIMESTAMP``) so the format
    survives JSONL round-trip across platforms.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_uuid() -> str:
    return uuid4().hex


def _empty_dict() -> dict[str, Any]:
    return {}


def _empty_str_dict() -> dict[str, str]:
    return {}


class Base(DeclarativeBase):
    pass


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    durability: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    superseded_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now_iso, nullable=False)
    updated_at: Mapped[str] = mapped_column(
        String(32), default=utc_now_iso, onupdate=utc_now_iso, nullable=False
    )
    # Python attribute is ``meta`` because ``metadata`` is reserved on
    # DeclarativeBase. The SQL column and JSONL key remain ``metadata``.
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=_empty_dict, nullable=False
    )


class Aim(Base):
    __tablename__ = "aims"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    target_date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    parent_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    broken_down_at: Mapped[dict[str, str]] = mapped_column(
        JSON, default=_empty_str_dict, nullable=False
    )
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now_iso, nullable=False)
    updated_at: Mapped[str] = mapped_column(
        String(32), default=utc_now_iso, onupdate=utc_now_iso, nullable=False
    )


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    from_bucket: Mapped[str] = mapped_column(String(64), nullable=False)
    to_bucket: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    urgency: Mapped[str] = mapped_column(String(16), nullable=False)
    summary: Mapped[str] = mapped_column(String(512), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    context: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    expires_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now_iso, nullable=False)
    updated_at: Mapped[str] = mapped_column(
        String(32), default=utc_now_iso, onupdate=utc_now_iso, nullable=False
    )


class QueuedJob(Base):
    __tablename__ = "queued_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    directive: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    fire_at: Mapped[str] = mapped_column(String(32), nullable=False, default="now")
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    recurrence: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claimed_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_heartbeat: Mapped[str | None] = mapped_column(String(32), nullable=True)
    queued_by: Mapped[str] = mapped_column(String(64), nullable=False)
    visible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_step_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timeout_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Model tier the worker should run this job at — "opus", "sonnet",
    # or "haiku". Resolved to a concrete model id at run time via
    # ``tasque.llm.factory.get_chat_model_for_tier``. Nullable for
    # forward compatibility with rows written before the column existed;
    # the runner falls back to the worker default when unset.
    tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # ISO timestamp set by ``tasque.discord.worker_run_watcher`` after a
    # successful worker-run notification has been posted to JOBS. Acts as
    # a persistent de-dup gate so a daemon restart between completion and
    # the watcher tick doesn't double-announce.
    notified_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Captured WorkerResult fields. Populated by the scheduler at the
    # moment the worker returns (success and failure paths) so the
    # asynchronous watcher can build the notify payload from DB state
    # alone — no cross-thread bridging needed.
    last_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_produces_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now_iso, nullable=False)
    updated_at: Mapped[str] = mapped_column(
        String(32), default=utc_now_iso, onupdate=utc_now_iso, nullable=False
    )


class FailedJob(Base):
    __tablename__ = "failed_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    job_id: Mapped[str] = mapped_column(String(32), nullable=False)
    agent_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_timestamp: Mapped[str] = mapped_column(String(32), nullable=False)
    error_type: Mapped[str] = mapped_column(String(128), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    traceback: Mapped[str] = mapped_column(Text, nullable=False, default="")
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    original_trigger: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    chain_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    plan_node_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Cached id of the per-failure Discord thread, if one is created
    # (current ``notify_failed_job`` doesn't anchor a thread, but the
    # column is kept for future use and historic rows that did).
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # ISO timestamp set by ``tasque.discord.dlq_watcher`` after a
    # successful DLQ embed has been posted. Acts as a persistent dedup
    # gate so a daemon restart between failure and the watcher tick
    # doesn't double-announce.
    notified_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now_iso, nullable=False)
    updated_at: Mapped[str] = mapped_column(
        String(32), default=utc_now_iso, onupdate=utc_now_iso, nullable=False
    )


class ChainTemplate(Base):
    __tablename__ = "chain_templates"
    __table_args__ = (UniqueConstraint("chain_name", name="uq_chain_templates_chain_name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    chain_name: Mapped[str] = mapped_column(String(128), nullable=False)
    bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recurrence: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_fired_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    plan_json: Mapped[str] = mapped_column(Text, nullable=False)
    seed_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # Hash of plan_json at last YAML seed; used by ``chain reload`` to
    # detect LLM-side edits and refuse to clobber them.
    seed_plan_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now_iso, nullable=False)
    updated_at: Mapped[str] = mapped_column(
        String(32), default=utc_now_iso, onupdate=utc_now_iso, nullable=False
    )


class ChainRun(Base):
    __tablename__ = "chain_runs"
    __table_args__ = (UniqueConstraint("chain_id", name="uq_chain_runs_chain_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    chain_id: Mapped[str] = mapped_column(String(64), nullable=False)
    chain_name: Mapped[str] = mapped_column(String(128), nullable=False)
    bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    template_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[str] = mapped_column(String(32), nullable=False)
    ended_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # ISO-8601 timestamp set the first time the chain status watcher
    # successfully posts the rich one-shot terminal embed for this run.
    # Persistent (vs. an in-memory set) so that a daemon restart between
    # observing the transition and posting doesn't cause a duplicate
    # iteration-summary post on the next boot.
    terminal_notified_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now_iso, nullable=False)
    updated_at: Mapped[str] = mapped_column(
        String(32), default=utc_now_iso, onupdate=utc_now_iso, nullable=False
    )


class CoachPending(Base):
    """A row in the coach trigger queue.

    One drainer claims rows in enqueue order, runs the bucket coach, then
    deletes the row on success (keeps it on failure for retry). At most one
    in-flight run per bucket at a time. ``dedup_key`` is used by
    ``coach.trigger.enqueue`` to suppress redundant fires — pending rows
    dedup permanently, recently-claimed rows dedup for a short window
    (default 5 min, see ``coach.trigger.dedup_window``).
    """

    __tablename__ = "coach_pending"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    bucket: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    dedup_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    enqueued_at: Mapped[str] = mapped_column(String(32), default=utc_now_iso, nullable=False)
    claimed_at: Mapped[str | None] = mapped_column(String(32), nullable=True)


class AgentResult(Base):
    """Transient inbox for an agent's structured LLM output.

    Each ``run_*`` call (worker, coach, planner, strategist) mints a
    fresh ``result_token`` and embeds it in the LLM prompt. The LLM is
    instructed to call the agent's ``submit_<kind>_result`` MCP tool,
    which writes a row here keyed by that token. The agent then reads
    and deletes the row to obtain its structured result — no post-hoc
    JSON parsing of the LLM's text output is involved.

    Rows are short-lived: the agent deletes its own row on read. Any
    leftover rows (process crash mid-run, LLM never called the tool)
    are swept by an idle reaper based on ``created_at`` age.
    """

    __tablename__ = "agent_results"

    result_token: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(
        String(32), default=utc_now_iso, nullable=False
    )


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    local_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    discord_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    discord_channel_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now_iso, nullable=False)
    updated_at: Mapped[str] = mapped_column(
        String(32), default=utc_now_iso, onupdate=utc_now_iso, nullable=False
    )


# Public registry of all entity types — used by repo, importer, exporter.
ENTITY_TYPES: tuple[type[Base], ...] = (
    Note,
    Aim,
    Signal,
    QueuedJob,
    FailedJob,
    ChainTemplate,
    ChainRun,
    Attachment,
)

ENTITY_BY_NAME: dict[str, type[Base]] = {cls.__name__: cls for cls in ENTITY_TYPES}

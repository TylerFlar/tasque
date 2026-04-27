"""Pure-function embed builders.

Each function returns a ``dict`` shaped like a Discord embed payload —
the bot's send adapter converts it to ``nextcord.Embed`` at the seam.
None of these touch I/O. Keep them deterministic so their output is
testable without mocking nextcord.

Worker / chain embeds carry only the agent-supplied ``summary`` plus
tasque-set metadata. The agent does not format embed fields; the full
``report`` body is posted separately inside the anchored thread by
:mod:`tasque.discord.notify`.
"""

from __future__ import annotations

from typing import Any

from tasque.chains.spec import PlanNode
from tasque.memory.entities import ChainRun, FailedJob, QueuedJob

# Discord caps embed field values at 1024 chars; description at 4096.
EMBED_DESC_LIMIT = 4096
EMBED_FIELD_LIMIT = 1024
EMBED_TITLE_LIMIT = 256

COLOR_GREEN = 0x2ECC71
COLOR_RED = 0xE74C3C
COLOR_YELLOW = 0xF1C40F
COLOR_BLURPLE = 0x5865F2


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


# ----------------------------------------------------------------- worker

def build_worker_embed(
    run: dict[str, Any],
) -> dict[str, Any]:
    """Build the embed for a completed worker run.

    Title is the agent-supplied ``summary`` (truncated to fit Discord's
    title limit), or ``"Worker run failed"`` on error. Description is a
    short pointer into the thread (the full ``report`` is posted as
    thread-body messages by :mod:`tasque.discord.notify`). Metadata
    fields are tasque-set.

    ``run`` is a dict shaped like::

        {
            "job_id": str,
            "bucket": str | None,
            "summary": str,
            "report": str,           # NOT rendered here; thread body
            "produces": dict,        # NOT rendered here
            "directive": str,        # NOT rendered here
            "error": str | None,
            "chain_id": str | None,
            "chain_step_id": str | None,
        }
    """
    error = run.get("error")
    if error:
        title = "Worker run failed"
        color = COLOR_RED
    else:
        summary = run.get("summary") or "Worker run"
        title = _truncate(summary, EMBED_TITLE_LIMIT)
        color = COLOR_GREEN

    if error:
        description = _truncate(f"**Error**: {error}", EMBED_DESC_LIMIT)
    else:
        description = "(see thread for full report)"

    fields: list[dict[str, Any]] = []
    job_id = run.get("job_id")
    if job_id:
        fields.append({"name": "job_id", "value": str(job_id), "inline": True})
    bucket = run.get("bucket")
    if bucket:
        fields.append({"name": "bucket", "value": str(bucket), "inline": True})
    chain_id = run.get("chain_id")
    if chain_id:
        fields.append({"name": "chain_id", "value": str(chain_id), "inline": True})
    chain_step_id = run.get("chain_step_id")
    if chain_step_id:
        fields.append(
            {"name": "step", "value": str(chain_step_id), "inline": True}
        )

    return {
        "title": title,
        "description": description,
        "color": color,
        "fields": fields,
    }


def worker_run_dict(
    job: QueuedJob,
    *,
    summary: str,
    report: str,
    produces: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    """Convert a (QueuedJob, WorkerResult) pair into the dict shape
    :func:`build_worker_embed` expects."""
    return {
        "job_id": job.id,
        "bucket": job.bucket,
        "directive": job.directive,
        "summary": summary,
        "report": report,
        "produces": produces or {},
        "error": error,
        "chain_id": job.chain_id,
        "chain_step_id": job.chain_step_id,
    }


# ----------------------------------------------------------------- chain

def build_chain_status_embed(
    chain_run: ChainRun,
    kind: str,
) -> dict[str, Any]:
    """Build the start / end embed for a chain run.

    ``kind`` is "started" / "completed" / "failed". This embed is
    posted *once* per kind — there is no in-flight editing.
    """
    if kind == "started":
        title = f"Chain started: {chain_run.chain_name}"
        color = COLOR_BLURPLE
    elif kind == "completed":
        title = f"Chain completed: {chain_run.chain_name}"
        color = COLOR_GREEN
    elif kind == "failed":
        title = f"Chain failed: {chain_run.chain_name}"
        color = COLOR_RED
    else:
        title = f"Chain {kind}: {chain_run.chain_name}"
        color = COLOR_YELLOW

    fields: list[dict[str, Any]] = [
        {"name": "chain_id", "value": chain_run.chain_id, "inline": True},
        {"name": "status", "value": chain_run.status, "inline": True},
    ]
    if chain_run.bucket:
        fields.append({"name": "bucket", "value": chain_run.bucket, "inline": True})
    if chain_run.started_at:
        fields.append(
            {"name": "started_at", "value": chain_run.started_at, "inline": True}
        )
    if chain_run.ended_at:
        fields.append(
            {"name": "ended_at", "value": chain_run.ended_at, "inline": True}
        )

    return {"title": title, "color": color, "fields": fields}


def _pick_final_completed_step(
    state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """Pick the most-recently-completed top-level (non-fan-out-child) step.

    Falls back to the last completed step of any kind if all completed
    steps are fan-out children. Returns ``None`` if no step has completed
    or no checkpoint state is available.
    """
    if not state:
        return None
    completed_raw = state.get("completed") or {}
    if not completed_raw:
        return None
    completed: dict[str, dict[str, Any]] = dict(completed_raw)
    non_fanout = [(k, v) for k, v in completed.items() if "[" not in k]
    if non_fanout:
        return non_fanout[-1]
    items = list(completed.items())
    return items[-1]


def build_chain_terminal_embed(
    chain_run: ChainRun,
    state: dict[str, Any] | None,
    kind: str,
) -> dict[str, Any]:
    """Build the terminal embed posted when a chain finishes.

    Description is the final completed step's agent-supplied ``summary``,
    truncated to fit. Metadata fields are tasque-set
    (``chain_id`` / ``bucket`` / ``started`` / ``ended``). The full
    ``report`` body is posted separately inside the anchored thread by
    :func:`tasque.discord.notify.notify_chain_terminal`. ``produces`` is
    NOT rendered to the user — it's chain-state-internal.
    """
    if kind == "completed":
        title = f"Chain completed: {chain_run.chain_name}"
        color = COLOR_GREEN
    elif kind == "failed":
        title = f"Chain failed: {chain_run.chain_name}"
        color = COLOR_RED
    elif kind == "stopped":
        title = f"Chain stopped: {chain_run.chain_name}"
        color = COLOR_YELLOW
    else:
        title = f"Chain {kind}: {chain_run.chain_name}"
        color = COLOR_BLURPLE

    fields: list[dict[str, Any]] = [
        {"name": "chain_id", "value": chain_run.chain_id, "inline": True},
    ]
    if chain_run.bucket:
        fields.append({"name": "bucket", "value": chain_run.bucket, "inline": True})
    if chain_run.started_at:
        fields.append({"name": "started", "value": chain_run.started_at, "inline": True})
    if chain_run.ended_at:
        fields.append({"name": "ended", "value": chain_run.ended_at, "inline": True})

    description = ""
    final = _pick_final_completed_step(state)
    if final is not None:
        _step_id, output = final
        # The agent emits a short summary alongside the long report; the
        # summary is what belongs in the description. Older WorkerResults
        # that didn't carry a summary fall through with empty description
        # and the user reads the thread.
        summary_text = (output.get("summary") or "").strip()
        if summary_text:
            description = _truncate(summary_text, EMBED_DESC_LIMIT)

    embed: dict[str, Any] = {
        "title": _truncate(title, EMBED_TITLE_LIMIT),
        "color": color,
        "fields": fields,
    }
    if description:
        embed["description"] = description
    return embed


# ----------------------------------------------------------------- DLQ

def build_failed_job_embed(failed_job: FailedJob) -> dict[str, Any]:
    """Build the embed for a DLQ entry. Buttons (Retry) are attached by
    the chain UI button layer at post time, not here."""
    fields: list[dict[str, Any]] = [
        {"name": "failed_job_id", "value": failed_job.id, "inline": True},
        {"name": "job_id", "value": failed_job.job_id, "inline": True},
        {"name": "agent_kind", "value": failed_job.agent_kind, "inline": True},
        {"name": "error_type", "value": failed_job.error_type, "inline": True},
        {
            "name": "retry_count",
            "value": str(failed_job.retry_count),
            "inline": True,
        },
    ]
    if failed_job.bucket:
        fields.append(
            {"name": "bucket", "value": failed_job.bucket, "inline": True}
        )
    if failed_job.chain_id:
        fields.append(
            {"name": "chain_id", "value": failed_job.chain_id, "inline": True}
        )
    if failed_job.plan_node_id:
        fields.append(
            {"name": "step", "value": failed_job.plan_node_id, "inline": True}
        )

    description = _truncate(failed_job.error_message or "", EMBED_FIELD_LIMIT)
    return {
        "title": f"Job failed: {failed_job.error_type}",
        "description": description,
        "color": COLOR_RED,
        "fields": fields,
    }


# ----------------------------------------------------------------- approval

def build_approval_embed(
    step: PlanNode,
    proposal: dict[str, Any] | None,
    *,
    chain_run: ChainRun | None = None,
) -> dict[str, Any]:
    """Build the interrupt embed for an approval step.

    The interrupt payload from the approval node carries
    ``{step_id, directive, consumes_payload}`` — the chain UI watcher
    passes ``proposal=consumes_payload`` so the user can see what the
    upstream worker handed off.
    """
    fields: list[dict[str, Any]] = [
        {"name": "step_id", "value": step["id"], "inline": True},
    ]
    if chain_run is not None:
        fields.append(
            {"name": "chain_id", "value": chain_run.chain_id, "inline": True}
        )
        fields.append(
            {"name": "chain_name", "value": chain_run.chain_name, "inline": True}
        )

    description_parts: list[str] = [
        f"**Directive**: {_truncate(step.get('directive') or '', EMBED_FIELD_LIMIT)}"
    ]
    if proposal:
        try:
            import json as _json

            blob = _json.dumps(proposal, indent=2, default=str)
        except Exception:
            blob = repr(proposal)
        description_parts.append(
            "**Proposal**:\n```json\n" + _truncate(blob, EMBED_DESC_LIMIT - 256) + "\n```"
        )

    return {
        "title": "Approval needed",
        "description": "\n\n".join(description_parts)[:EMBED_DESC_LIMIT],
        "color": COLOR_YELLOW,
        "fields": fields,
    }


def build_resolved_approval_embed(
    step: PlanNode,
    resolution: str,
    *,
    chain_run: ChainRun | None = None,
) -> dict[str, Any]:
    """Build the static embed an approval is replaced with after the user
    clicks. ``resolution`` is the literal value passed to ``Command(resume=...)``
    (e.g. "approved" / "declined" / freeform text)."""
    color = COLOR_GREEN if resolution.lower() in ("approved", "approve") else COLOR_RED
    fields: list[dict[str, Any]] = [
        {"name": "step_id", "value": step["id"], "inline": True},
        {"name": "resolution", "value": resolution, "inline": True},
    ]
    if chain_run is not None:
        fields.append(
            {"name": "chain_id", "value": chain_run.chain_id, "inline": True}
        )
    return {
        "title": "Approval resolved",
        "description": f"Resolved as: **{resolution}**",
        "color": color,
        "fields": fields,
    }


__all__ = [
    "build_approval_embed",
    "build_chain_status_embed",
    "build_chain_terminal_embed",
    "build_failed_job_embed",
    "build_resolved_approval_embed",
    "build_worker_embed",
    "worker_run_dict",
]

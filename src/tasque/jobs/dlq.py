"""Dead-letter queue: persist worker failures and retry them.

Failure mode:
    A worker raises, the LLM returns an error, or the JSON didn't parse
    → the scheduler calls :func:`record_failure` with the exception (if
    any) and the textual error message. We write a ``FailedJob`` row and
    flip the ``QueuedJob.status`` to ``"failed"``.

Retry mode:
    User runs ``tasque dlq retry <id>``. For a standalone job, we insert
    a fresh ``QueuedJob`` with the same directive + reason (status
    ``"pending"``, ``fire_at="now"``) and bump ``retry_count`` on the
    ``FailedJob``. For a chain-step failure (``chain_id`` set), we hand
    off to ``tasque.chains.dlq_retry`` — the chain runner owns step
    re-firing semantics.
"""

from __future__ import annotations

import traceback
from typing import Any

import structlog

from tasque.memory.entities import FailedJob, QueuedJob, utc_now_iso
from tasque.memory.repo import (
    get_entity,
    query_unresolved_failures,
    update_entity_status,
    write_entity,
)

log = structlog.get_logger(__name__)


def record_failure(
    job: QueuedJob,
    *,
    exc: BaseException | None = None,
    error_message: str | None = None,
    error_type: str | None = None,
    original_trigger: str = "scheduler",
) -> FailedJob:
    """Write a FailedJob row and flip the QueuedJob status to ``"failed"``.

    Either ``exc`` or both of ``error_message`` + ``error_type`` must be
    provided. ``exc`` is preferred because we capture its traceback.
    """
    if exc is not None:
        et = type(exc).__name__
        em = str(exc) or et
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    else:
        if error_message is None or error_type is None:
            raise ValueError(
                "record_failure requires either exc, or both error_message and error_type"
            )
        et = error_type
        em = error_message
        tb = ""

    fj = FailedJob(
        job_id=job.id,
        agent_kind=job.kind,
        bucket=job.bucket,
        failure_timestamp=utc_now_iso(),
        error_type=et,
        error_message=em,
        traceback=tb,
        retry_count=0,
        original_trigger=original_trigger,
        resolved=False,
        chain_id=job.chain_id,
        plan_node_id=job.chain_step_id,
    )
    written = write_entity(fj)
    update_entity_status(job.id, "failed")
    log.warning(
        "jobs.dlq.recorded",
        failed_job_id=written.id[:8],
        job_id=job.id[:8],
        bucket=job.bucket,
        error_type=et,
        error=em.splitlines()[0][:120] if em else "",
        chain_id=job.chain_id[:8] if job.chain_id else None,
    )
    return written


def list_unresolved(limit: int = 20) -> list[FailedJob]:
    """Return up to ``limit`` unresolved FailedJobs, newest first."""
    return query_unresolved_failures(limit=limit)


def mark_resolved(failed_job_id: str) -> bool:
    """Mark a FailedJob as resolved. Returns True if the row existed."""
    return update_entity_status(failed_job_id, "resolved")


def get_failure(failed_job_id: str) -> FailedJob | None:
    """Look up a FailedJob by id (returns None if not found or wrong type)."""
    obj = get_entity(failed_job_id)
    if isinstance(obj, FailedJob):
        return obj
    return None


def retry(failed_job_id: str) -> dict[str, Any]:
    """Re-fire a FailedJob.

    Standalone job → insert a fresh ``QueuedJob`` with the same directive
    + reason + bucket + recurrence (status ``"pending"``, ``fire_at="now"``,
    ``queued_by="dlq"``). Bump ``retry_count`` on the original ``FailedJob``.

    Chain-step job (``chain_id`` is set) → call
    ``tasque.chains.dlq_retry(chain_id, step_id)`` and bump
    ``retry_count``. The chain runner owns the actual re-firing.

    Returns a small report dict summarising what was done.
    """
    fj = get_failure(failed_job_id)
    if fj is None:
        raise ValueError(f"no FailedJob with id={failed_job_id}")

    # Look up the originating QueuedJob — needed for standalone retry to
    # carry directive/reason/bucket through unchanged. May be missing if
    # the user wiped it; we degrade gracefully for chain-step retry only.
    original = get_entity(fj.job_id)
    original_qj = original if isinstance(original, QueuedJob) else None

    if fj.chain_id is not None:
        # Defer to the chain engine. Imported here so the chain module
        # is optional for standalone-only deployments.
        from tasque import chains

        chains.dlq_retry(fj.chain_id, fj.plan_node_id or "")
        fj.retry_count += 1
        fj.updated_at = utc_now_iso()
        write_entity(fj)
        return {
            "kind": "chain-step",
            "chain_id": fj.chain_id,
            "plan_node_id": fj.plan_node_id,
            "retry_count": fj.retry_count,
        }

    if original_qj is None:
        raise ValueError(
            f"original QueuedJob {fj.job_id} missing — cannot reconstruct standalone retry"
        )

    new_job = QueuedJob(
        kind=original_qj.kind,
        bucket=original_qj.bucket,
        directive=original_qj.directive,
        reason=original_qj.reason,
        fire_at="now",
        status="pending",
        recurrence=original_qj.recurrence,
        queued_by="dlq",
        visible=original_qj.visible,
        thread_id=original_qj.thread_id,
        chain_id=None,
        chain_step_id=None,
        timeout_seconds=original_qj.timeout_seconds,
        tier=original_qj.tier,
    )
    written = write_entity(new_job)

    fj.retry_count += 1
    fj.updated_at = utc_now_iso()
    write_entity(fj)

    return {
        "kind": "standalone",
        "new_job_id": written.id,
        "retry_count": fj.retry_count,
    }

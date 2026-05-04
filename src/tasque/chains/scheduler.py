"""Chain launch + cron polling + restart-recovery.

Main entry points:

* :func:`launch_chain_run` — seed a fresh ChainRun row plus the
  serialized initial state the daemon runner needs. ``wait=False`` is a
  durable enqueue only; it never starts a graph in the caller process.
* :func:`fire_due_chain_templates` — APScheduler poll callback. Finds
  enabled templates whose cron is due, validates them, enqueues each.
* :func:`claim_and_run_ready_chains` — daemon-owned execution. Claims
  one runnable ChainRun with a DB lease, invokes LangGraph, and extends
  the lease while work is active.
* :func:`resume_interrupted_chains` — startup hook. Re-invokes the
  lease runner with failed checkpoint steps reset to ``pending`` so
  transient errors get retried on boot.
* :func:`resume_stale_chains` — compatibility wrapper for old call
  sites. Staleness no longer drives dispatch.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import structlog
from langchain_core.runnables import RunnableConfig
from sqlalchemy import or_, select, update

from tasque.chains.checkpointer import get_chain_checkpointer
from tasque.chains.graph import get_compiled_chain_graph
from tasque.chains.graph._common import _now_iso
from tasque.chains.spec import (
    HistoryEntry,
    PlanNode,
    resolve_planner_tier,
    validate_spec,
)
from tasque.jobs.cron import next_fire_at, validate_cron
from tasque.memory.db import get_session
from tasque.memory.entities import ChainRun, ChainTemplate, utc_now_iso

log = structlog.get_logger(__name__)


# In-process registry of chain ids whose graph.invoke is currently
# running on this process. The DB lease is the cross-process authority;
# this local set is just cheap introspection/test support while the
# heartbeat thread refreshes lease/heartbeat fields for active work.
_active_chain_invokes: set[str] = set()
_active_chain_heartbeats: dict[str, tuple[threading.Event, threading.Thread]] = {}
_active_chain_invokes_lock = threading.Lock()
DEFAULT_CHAIN_HEARTBEAT_INTERVAL_SECONDS = 30.0
DEFAULT_CHAIN_LEASE_SECONDS = 300.0


def _future_iso(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def _bump_chain_heartbeat(
    chain_id: str,
    iso: str | None = None,
    *,
    owner_id: str | None = None,
    lease_seconds: float = DEFAULT_CHAIN_LEASE_SECONDS,
) -> bool:
    """Persist a liveness mark for a currently-running graph invoke."""
    now_iso = iso or utc_now_iso()
    with get_session() as sess:
        stmt = select(ChainRun).where(ChainRun.chain_id == chain_id)
        if owner_id is not None:
            stmt = stmt.where(ChainRun.lease_owner == owner_id)
        row = sess.execute(stmt).scalars().first()
        if row is None:
            return False
        row.last_heartbeat = now_iso
        if owner_id is not None:
            row.lease_expires_at = _future_iso(lease_seconds)
        row.updated_at = now_iso
        return True


def _pump_chain_heartbeat(
    chain_id: str,
    stop: threading.Event,
    interval_seconds: float = DEFAULT_CHAIN_HEARTBEAT_INTERVAL_SECONDS,
    owner_id: str | None = None,
    lease_seconds: float = DEFAULT_CHAIN_LEASE_SECONDS,
) -> None:
    """Refresh ``ChainRun.last_heartbeat`` until the active invoke ends."""
    while not stop.is_set():
        try:
            if not _bump_chain_heartbeat(
                chain_id,
                owner_id=owner_id,
                lease_seconds=lease_seconds,
            ):
                return
        except Exception:
            log.exception(
                "chains.scheduler.heartbeat_failed",
                chain_id=chain_id[:8],
            )
        if stop.wait(timeout=interval_seconds):
            return


def _mark_invoke_active(
    chain_id: str,
    *,
    owner_id: str | None = None,
    lease_seconds: float = DEFAULT_CHAIN_LEASE_SECONDS,
) -> None:
    heartbeat: tuple[threading.Event, threading.Thread] | None = None
    with _active_chain_invokes_lock:
        _active_chain_invokes.add(chain_id)
        if chain_id not in _active_chain_heartbeats:
            stop = threading.Event()
            thread = threading.Thread(
                target=_pump_chain_heartbeat,
                args=(chain_id, stop),
                kwargs={"owner_id": owner_id, "lease_seconds": lease_seconds},
                name=f"tasque-chain-heartbeat-{chain_id[:8]}",
                daemon=True,
            )
            heartbeat = (stop, thread)
            _active_chain_heartbeats[chain_id] = heartbeat
    if heartbeat is not None:
        heartbeat[1].start()


def _mark_invoke_inactive(chain_id: str) -> None:
    heartbeat: tuple[threading.Event, threading.Thread] | None = None
    with _active_chain_invokes_lock:
        _active_chain_invokes.discard(chain_id)
        heartbeat = _active_chain_heartbeats.pop(chain_id, None)
    if heartbeat is not None:
        stop, thread = heartbeat
        stop.set()
        thread.join(timeout=2.0)


def _is_invoke_active(chain_id: str) -> bool:
    with _active_chain_invokes_lock:
        return chain_id in _active_chain_invokes


def is_chain_invoke_active(chain_id: str) -> bool:
    """Return True while this process is actively invoking ``chain_id``."""
    return _is_invoke_active(chain_id)


def _new_chain_id() -> str:
    return uuid4().hex


def _thread_config(chain_id: str) -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": chain_id})


def _initial_state(
    *,
    chain_id: str,
    chain_name: str,
    bucket: str,
    plan: list[PlanNode],
    thread_id: str | None,
    planner_tier: str,
    vars: dict[str, Any],
) -> dict[str, Any]:
    return {
        "chain_id": chain_id,
        "chain_name": chain_name,
        "bucket": bucket,
        "thread_id": thread_id,
        "plan": plan,
        "completed": {},
        "failures": {},
        "replan": False,
        "history": [],
        "approval_resume": None,
        "awaiting_posts": {},
        "planner_tier": planner_tier,
        "vars": vars,
    }


def _is_terminal_state(snapshot_state: dict[str, Any]) -> bool:
    plan: list[PlanNode] = list(snapshot_state.get("plan") or [])
    if not plan:
        return True
    return all(
        n["status"] in ("completed", "failed", "stopped")
        for n in plan
    )


def _set_run_terminal(chain_id: str, *, status: str) -> None:
    with get_session() as sess:
        stmt = select(ChainRun).where(ChainRun.chain_id == chain_id)
        row = sess.execute(stmt).scalars().first()
        if row is None:
            return
        if row.status == "stopped" and status != "stopped":
            row.lease_owner = None
            row.lease_expires_at = None
            row.updated_at = utc_now_iso()
            log.info(
                "chains.scheduler.terminal_preserved_stopped",
                chain_id=chain_id[:8],
                chain_name=row.chain_name,
                requested=status,
            )
            return
        row.status = status
        row.ended_at = utc_now_iso()
        row.lease_owner = None
        row.lease_expires_at = None
        row.updated_at = utc_now_iso()
        log.info(
            "chains.scheduler.terminal",
            chain_id=chain_id[:8],
            chain_name=row.chain_name,
            status=status,
        )


def _release_chain_lease(chain_id: str, owner_id: str) -> None:
    now_iso = utc_now_iso()
    with get_session() as sess:
        stmt = select(ChainRun).where(
            ChainRun.chain_id == chain_id,
            ChainRun.lease_owner == owner_id,
        )
        row = sess.execute(stmt).scalars().first()
        if row is None:
            return
        row.lease_owner = None
        row.lease_expires_at = None
        row.updated_at = now_iso


def _set_run_awaiting_approval(chain_id: str) -> None:
    now_iso = utc_now_iso()
    with get_session() as sess:
        stmt = select(ChainRun).where(ChainRun.chain_id == chain_id)
        row = sess.execute(stmt).scalars().first()
        if row is None or row.status != "running":
            return
        row.status = "awaiting_approval"
        row.lease_owner = None
        row.lease_expires_at = None
        row.updated_at = now_iso


def _state_has_waiting_approval(state: dict[str, Any] | None) -> bool:
    if state is None:
        return False
    plan: list[PlanNode] = list(state.get("plan") or [])
    return any(
        n.get("kind") == "approval"
        and n.get("status") in ("running", "awaiting_user")
        for n in plan
    )


def _checkpoint_state(chain_id: str) -> dict[str, Any] | None:
    saver = get_chain_checkpointer()
    snap = saver.get_tuple(_thread_config(chain_id))
    if snap is None:
        return None
    return snap.checkpoint.get("channel_values", {})


def _load_initial_state(row: ChainRun) -> dict[str, Any] | None:
    if not row.initial_state_json:
        return None
    try:
        raw = json.loads(row.initial_state_json)
    except json.JSONDecodeError:
        return None
    return cast(dict[str, Any], raw) if isinstance(raw, dict) else None


def _claim_next_runnable_chain(
    *,
    owner_id: str,
    lease_seconds: float = DEFAULT_CHAIN_LEASE_SECONDS,
    now: datetime | None = None,
) -> ChainRun | None:
    """Atomically claim one unowned or expired running chain for this daemon."""
    reference = now if now is not None else datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    now_iso = reference.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    lease_expires_at = (
        reference.astimezone(UTC) + timedelta(seconds=lease_seconds)
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    with get_session() as sess:
        candidate_stmt = (
            select(ChainRun)
            .where(ChainRun.status == "running")
            .where(
                or_(
                    ChainRun.lease_owner.is_(None),
                    ChainRun.lease_expires_at.is_(None),
                    ChainRun.lease_expires_at <= now_iso,
                )
            )
            .order_by(ChainRun.created_at.asc())
            .limit(10)
        )
        candidates = list(sess.execute(candidate_stmt).scalars().all())
        for candidate in candidates:
            if _is_invoke_active(candidate.chain_id):
                continue
            state = _checkpoint_state(candidate.chain_id)
            if _state_has_waiting_approval(state):
                candidate.status = "awaiting_approval"
                candidate.lease_owner = None
                candidate.lease_expires_at = None
                candidate.updated_at = now_iso
                continue

            claim_stmt = (
                update(ChainRun)
                .where(ChainRun.id == candidate.id)
                .where(ChainRun.status == "running")
                .where(
                    or_(
                        ChainRun.lease_owner.is_(None),
                        ChainRun.lease_expires_at.is_(None),
                        ChainRun.lease_expires_at <= now_iso,
                    )
                )
                .values(
                    lease_owner=owner_id,
                    lease_expires_at=lease_expires_at,
                    last_heartbeat=now_iso,
                    updated_at=now_iso,
                )
            )
            sess.execute(claim_stmt)
            row = sess.execute(
                select(ChainRun).where(
                    ChainRun.id == candidate.id,
                    ChainRun.lease_owner == owner_id,
                )
            ).scalars().first()
            if row is None:
                continue
            sess.expunge(row)
            return row
    return None


def _invoke_chain_graph(
    chain_id: str,
    initial: dict[str, Any] | None,
    *,
    owner_id: str | None = None,
    lease_seconds: float = DEFAULT_CHAIN_LEASE_SECONDS,
    raise_on_error: bool = False,
) -> None:
    """Run ``graph.invoke`` for ``chain_id`` and finalize status.

    Runtime failures are logged and the ChainRun row is flipped to
    ``failed`` so the daemon's chain-status watcher posts a terminal
    embed.
    """
    graph = get_compiled_chain_graph()
    cfg = _thread_config(chain_id)
    _mark_invoke_active(
        chain_id,
        owner_id=owner_id,
        lease_seconds=lease_seconds,
    )
    try:
        try:
            graph.invoke(cast(Any, initial), cfg)
        except Exception:
            log.exception(
                "chains.scheduler.launch_invoke_failed", chain_id=chain_id
            )
            _set_run_terminal(chain_id, status="failed")
            if raise_on_error:
                raise
            return
        maybe_finalize_status(chain_id)
        state = _checkpoint_state(chain_id)
        if _state_has_waiting_approval(state):
            _set_run_awaiting_approval(chain_id)
    finally:
        _mark_invoke_inactive(chain_id)


def launch_chain_run(
    spec: dict[str, Any],
    *,
    thread_id: str | None = None,
    template_id: str | None = None,
    vars: dict[str, Any] | None = None,
    wait: bool = True,
) -> str:
    """Start a new chain. Returns the new ``chain_id``.

    Validates the spec, creates a ``ChainRun`` row, and persists the
    initial state needed by the daemon-owned chain runner. The default
    ``wait=True`` still runs the graph synchronously for CLI/tests. Pass
    ``wait=False`` for durable enqueue semantics: the caller gets a
    ``chain_id`` immediately and the long-running daemon claims and
    invokes the chain via a DB lease.

    ``vars`` is an operator-supplied override that is merged over the
    spec's static ``vars`` (caller wins on key collision). The merged
    dict is frozen on the chain state and surfaced to every worker
    prompt — directives can branch on it (e.g. ``vars.force=true``
    skips an age gate). Pass ``None`` for the no-override case.
    """
    plan = validate_spec(spec)
    chain_name: str = spec["chain_name"]
    bucket: str = spec.get("bucket") or ""
    planner_tier = resolve_planner_tier(spec)

    spec_vars_raw = spec.get("vars") or {}
    if not isinstance(spec_vars_raw, dict):
        spec_vars_raw = {}
    merged_vars: dict[str, Any] = {**spec_vars_raw, **(vars or {})}

    chain_id = _new_chain_id()
    started = utc_now_iso()
    initial = _initial_state(
        chain_id=chain_id,
        chain_name=chain_name,
        bucket=bucket,
        plan=plan,
        thread_id=thread_id,
        planner_tier=planner_tier,
        vars=merged_vars,
    )
    with get_session() as sess:
        run = ChainRun(
            chain_id=chain_id,
            chain_name=chain_name,
            bucket=bucket or None,
            template_id=template_id,
            thread_id=thread_id,
            status="running",
            started_at=started,
            initial_state_json=json.dumps(initial, default=str),
            last_heartbeat=started,
        )
        sess.add(run)
        sess.flush()
    log.info(
        "chains.scheduler.launched",
        chain_id=chain_id[:8],
        chain_name=chain_name,
        bucket=bucket or None,
        steps=len(plan),
        vars_keys=sorted(merged_vars.keys()) or None,
        wait=wait,
    )

    if wait:
        # Original synchronous path: run the graph here and let the
        # caller see it through.
        _invoke_chain_graph(chain_id, initial, raise_on_error=True)
    else:
        log.info(
            "chains.scheduler.enqueued",
            chain_id=chain_id[:8],
            chain_name=chain_name,
        )

    return chain_id


def _invoke_claimed_chain(
    row: ChainRun,
    *,
    owner_id: str,
    lease_seconds: float,
    reset_failed: bool,
) -> None:
    state = _checkpoint_state(row.chain_id)
    if _state_has_waiting_approval(state):
        _set_run_awaiting_approval(row.chain_id)
        return

    initial: dict[str, Any] | None = None
    if state is None:
        initial = _load_initial_state(row)
        if initial is None:
            log.error(
                "chains.scheduler.missing_initial_state",
                chain_id=row.chain_id[:8],
                chain_name=row.chain_name,
            )
            _set_run_terminal(row.chain_id, status="failed")
            return
    elif reset_failed:
        reset_ids = _reset_failed_steps_for_resume(row.chain_id)
        if reset_ids:
            log.info(
                "chains.scheduler.resume_retrying_failed",
                chain_id=row.chain_id[:8],
                chain_name=row.chain_name,
                steps=reset_ids,
            )

    _invoke_chain_graph(
        row.chain_id,
        initial,
        owner_id=owner_id,
        lease_seconds=lease_seconds,
    )


def claim_and_run_ready_chains(
    *,
    owner_id: str | None = None,
    max_runs: int = 1,
    lease_seconds: float = DEFAULT_CHAIN_LEASE_SECONDS,
    reset_failed: bool = False,
) -> list[str]:
    """Claim and invoke runnable chains using a durable DB lease.

    This is the daemon-owned execution path. Enqueuers create
    ``running`` rows with ``initial_state_json`` and return; this runner
    is the only periodic component that invokes those rows. Leases
    replace checkpoint-age staleness guesses: a row is runnable only
    when no live owner holds it, or when the owner's lease has expired.
    """
    if max_runs <= 0:
        return []
    actual_owner = owner_id or f"daemon-{uuid4().hex[:12]}"
    ran: list[str] = []
    for _ in range(max_runs):
        row = _claim_next_runnable_chain(
            owner_id=actual_owner,
            lease_seconds=lease_seconds,
        )
        if row is None:
            break
        log.info(
            "chains.scheduler.claimed",
            chain_id=row.chain_id[:8],
            chain_name=row.chain_name,
            owner_id=actual_owner,
        )
        try:
            _invoke_claimed_chain(
                row,
                owner_id=actual_owner,
                lease_seconds=lease_seconds,
                reset_failed=reset_failed,
            )
        except Exception:
            log.exception(
                "chains.scheduler.claimed_invoke_failed",
                chain_id=row.chain_id,
                owner_id=actual_owner,
            )
            _set_run_terminal(row.chain_id, status="failed")
        finally:
            _release_chain_lease(row.chain_id, actual_owner)
        ran.append(row.chain_id)
    return ran


def maybe_finalize_status(chain_id: str) -> None:
    """If the chain's plan is fully terminal, flip the ChainRun to
    completed/failed/stopped accordingly."""
    from tasque.chains.manager import get_chain_state

    state = get_chain_state(chain_id)
    if state is None:
        return
    plan: list[PlanNode] = list(state.get("plan") or [])
    if not plan:
        return
    if not _is_terminal_state(state):
        return
    if any(n["status"] == "stopped" for n in plan):
        _set_run_terminal(chain_id, status="stopped")
    elif any(n["status"] == "failed" for n in plan):
        _set_run_terminal(chain_id, status="failed")
    else:
        _set_run_terminal(chain_id, status="completed")


def _due_templates(*, now: datetime | None = None) -> list[ChainTemplate]:
    """Return enabled templates whose cron is due relative to ``now``.

    The anchor for "next fire after" is ``last_fired_at`` if the template
    has fired before, else ``created_at`` — *not* ``now``. Anchoring an
    unfired template at ``now`` makes ``next_fire_at`` return a future
    time by construction, so the template never becomes due. Anchoring at
    ``created_at`` lets a freshly-seeded template catch up to the most
    recent missed fire on the next poll, then settle into normal cadence
    after ``last_fired_at`` is populated.
    """
    base = now if now is not None else datetime.now(UTC)
    due: list[ChainTemplate] = []
    with get_session() as sess:
        stmt = select(ChainTemplate).where(
            ChainTemplate.enabled.is_(True), ChainTemplate.recurrence.isnot(None)
        )
        rows = list(sess.execute(stmt).scalars().all())
        for row in rows:
            recurrence = row.recurrence
            if recurrence is None:
                continue
            if validate_cron(recurrence) is not None:
                continue
            anchor_iso = row.last_fired_at or row.created_at
            try:
                anchor = datetime.strptime(
                    anchor_iso, "%Y-%m-%dT%H:%M:%S.%fZ"
                ).replace(tzinfo=UTC)
            except (TypeError, ValueError):
                continue
            try:
                nxt = next_fire_at(recurrence, after=anchor)
            except ValueError:
                continue
            if nxt <= base:
                sess.expunge(row)
                due.append(row)
    return due


def fire_due_chain_templates(*, now: datetime | None = None) -> list[str]:
    """Fire each enabled template whose cron is due. Returns chain_ids fired."""
    base = now if now is not None else datetime.now(UTC)
    fired: list[str] = []
    for template in _due_templates(now=base):
        try:
            spec = json.loads(template.plan_json)
        except json.JSONDecodeError:
            log.error(
                "chains.scheduler.bad_plan_json",
                chain_name=template.chain_name,
            )
            continue
        log.info(
            "chains.scheduler.firing",
            chain_name=template.chain_name,
            bucket=template.bucket,
            recurrence=template.recurrence,
        )
        try:
            chain_id = launch_chain_run(spec, template_id=template.id, wait=False)
        except Exception:
            log.exception(
                "chains.scheduler.launch_failed",
                chain_name=template.chain_name,
            )
            continue
        with get_session() as sess:
            live = sess.get(ChainTemplate, template.id)
            if live is not None:
                live.last_fired_at = utc_now_iso()
                live.updated_at = utc_now_iso()
        fired.append(chain_id)
        log.info(
            "chains.scheduler.fired",
            chain_name=template.chain_name,
            chain_id=chain_id[:8],
        )
    return fired


def _reset_failed_steps_for_resume(chain_id: str) -> list[str]:
    """Flip every ``failed`` plan node back to ``pending`` and clear the
    matching ``failures[]`` entries in the chain checkpoint.

    Used by :func:`resume_interrupted_chains` so that bot restart retries
    failed chain steps. Without this, a chain whose fan-out children (or
    any ``on_failure="halt"`` step) hit a transient error like an
    ``APIConnectionError`` would wedge: the supervisor exits because no
    step is ``running``, but downstream nodes whose deps include the
    failed step stay ``pending`` forever, so the ChainRun row stays at
    ``running`` and resumes go nowhere.

    Returns the list of step ids reset (empty if nothing failed).
    """
    saver = get_chain_checkpointer()
    cfg = _thread_config(chain_id)
    snapshot = saver.get_tuple(cfg)
    if snapshot is None:
        return []

    state: dict[str, Any] = snapshot.checkpoint.get("channel_values", {})
    plan_in: list[PlanNode] = list(state.get("plan") or [])

    reset_ids: list[str] = []
    new_plan: list[PlanNode] = []
    for n in plan_in:
        if n["status"] == "failed":
            new_n: PlanNode = dict(n)  # type: ignore[assignment]
            new_n["status"] = "pending"
            new_n["failure_reason"] = None
            reset_ids.append(n["id"])
            new_plan.append(new_n)
        else:
            new_plan.append(n)

    if not reset_ids:
        return []

    history: list[HistoryEntry] = [
        {
            "timestamp": _now_iso(),
            "kind": "resume",
            "details": {"reset_failed_steps": reset_ids, "to": "pending"},
        }
    ]
    # ``{step_id: None}`` triggers the None-on-right reducer in
    # :func:`tasque.chains.graph._common._merge_failures` — required to
    # actually drop the prior failure rather than re-merging it.
    update: dict[str, Any] = {
        "plan": new_plan,
        "failures": cast(dict[str, str], {sid: None for sid in reset_ids}),
        "history": history,
        "replan": False,
    }
    graph = get_compiled_chain_graph()
    # Mark the update as if a worker emitted it, so the next routed node
    # is the supervisor — which then re-promotes the now-pending steps
    # and Send-dispatches them.
    from tasque.chains.graph.supervisor import WORKER_NODE

    graph.update_state(cfg, update, as_node=WORKER_NODE)
    return reset_ids


def resume_interrupted_chains() -> list[str]:
    """Boot-recovery wrapper around the lease runner.

    Before invoking, any ``failed`` plan nodes are flipped back to
    ``pending`` and their failures cleared, so a bot restart retries
    every failed chain step. This is the recovery path for transient
    failures (LLM API timeouts, network blips) that would otherwise
    wedge a chain forever — fan-out children with ``on_failure="halt"``
    fail, the supervisor returns ``END`` because nothing is running,
    but downstream consumers like a gather step stay ``pending`` and the
    ChainRun row never finalises.

    Rows are claimed through the same durable lease path as normal
    daemon execution.
    """
    return claim_and_run_ready_chains(
        owner_id=f"boot-{uuid4().hex[:12]}",
        max_runs=100,
        reset_failed=True,
    )


# Kept only for compatibility with callers that still pass a stale-resume
# threshold. Dispatch is lease-owned now, not checkpoint-age-owned.
DEFAULT_STALE_RESUME_THRESHOLD_SECONDS = 300.0


def _parse_utc_iso(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        if raw.endswith("Z"):
            ts = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
        else:
            ts = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _age_seconds(raw_iso: Any, *, now: datetime | None = None) -> float | None:
    ts = _parse_utc_iso(raw_iso)
    if ts is None:
        return None
    reference = now if now is not None else datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return (reference.astimezone(UTC) - ts).total_seconds()


def _checkpoint_age_seconds(  # pyright: ignore[reportUnusedFunction]
    chain_id: str,
    *,
    now: datetime | None = None,
) -> float | None:
    """Return seconds since the latest checkpoint write for ``chain_id``.

    ``None`` means there is no checkpoint at all (the chain row exists
    but no graph node has ever written state — treat as infinitely
    stale by callers that want to resume aggressively, or skip if the
    caller wants conservatism).
    """
    saver = get_chain_checkpointer()
    snap = saver.get_tuple(_thread_config(chain_id))
    if snap is None:
        return None
    raw_ts = snap.checkpoint.get("ts")
    if not raw_ts:
        return None
    return _age_seconds(raw_ts, now=now)


def resume_stale_chains(
    *,
    threshold_seconds: float = DEFAULT_STALE_RESUME_THRESHOLD_SECONDS,
    now: datetime | None = None,
) -> list[str]:
    """Compatibility wrapper for the old stale-resume entry point.

    Staleness is no longer used to decide whether a chain is safe to
    invoke. The daemon runner claims rows by durable lease instead.
    ``threshold_seconds`` and ``now`` are accepted so older call sites
    keep working, but they no longer affect dispatch.
    """
    _ = (threshold_seconds, now)
    return claim_and_run_ready_chains(max_runs=1, reset_failed=False)


__all__ = [
    "DEFAULT_CHAIN_LEASE_SECONDS",
    "DEFAULT_STALE_RESUME_THRESHOLD_SECONDS",
    "claim_and_run_ready_chains",
    "fire_due_chain_templates",
    "is_chain_invoke_active",
    "launch_chain_run",
    "maybe_finalize_status",
    "resume_interrupted_chains",
    "resume_stale_chains",
]

"""Chain launch + cron polling + restart-recovery.

Four entry points:

* :func:`launch_chain_run` — seed a fresh ChainRun row + initial
  checkpoint state and synchronously invoke the graph until it
  terminates or interrupts (the supervisor handles serial dispatch).
* :func:`fire_due_chain_templates` — APScheduler poll callback. Finds
  enabled templates whose cron is due, validates them, fires each.
* :func:`resume_interrupted_chains` — startup hook. Re-invokes the
  graph for every ChainRun left in ``running`` so an unclean shutdown
  doesn't leave a chain wedged. Also resets ``failed`` plan nodes back
  to ``pending`` so transient errors get retried on boot.
* :func:`resume_stale_chains` — daemon tick. Re-invokes the graph for
  any ``running`` ChainRun whose latest checkpoint is older than
  ``threshold_seconds``. Picks up chains whose runner thread died with
  the calling process (e.g. an MCP-fired chain whose ``claude --print``
  subprocess exited mid-invoke) without waiting for a daemon restart.
  Unlike :func:`resume_interrupted_chains`, this does NOT reset failed
  steps — running periodically with that behavior would erase legitimate
  failure state on every tick.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import structlog
from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

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
# running on this process. ``resume_stale_chains`` consults this to
# avoid double-dispatching a chain whose runner thread is alive in
# this process but happens to be sitting inside a slow LLM turn (no
# checkpoint update for the duration of the call). Cross-process
# invocations whose process died are NOT in this set, so they still
# get resumed correctly. The set is in-memory only — a daemon crash
# clears it, which is the right semantic: any chain that was active
# before the crash is no longer active after.
_active_chain_invokes: set[str] = set()
_active_chain_invokes_lock = threading.Lock()


def _mark_invoke_active(chain_id: str) -> None:
    with _active_chain_invokes_lock:
        _active_chain_invokes.add(chain_id)


def _mark_invoke_inactive(chain_id: str) -> None:
    with _active_chain_invokes_lock:
        _active_chain_invokes.discard(chain_id)


def _is_invoke_active(chain_id: str) -> bool:
    with _active_chain_invokes_lock:
        return chain_id in _active_chain_invokes


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
        row.status = status
        row.ended_at = utc_now_iso()
        row.updated_at = utc_now_iso()
        log.info(
            "chains.scheduler.terminal",
            chain_id=chain_id[:8],
            chain_name=row.chain_name,
            status=status,
        )


def _invoke_chain_graph(chain_id: str, initial: dict[str, Any]) -> None:
    """Run ``graph.invoke`` for ``chain_id`` and finalize status.

    Factored out so both the synchronous (``wait=True``) and
    fire-and-forget (``wait=False``) paths share one body. Runtime
    failures are logged and the ChainRun row is flipped to ``failed``
    so the daemon's chain-status watcher posts a terminal embed.
    """
    graph = get_compiled_chain_graph()
    cfg = _thread_config(chain_id)
    _mark_invoke_active(chain_id)
    try:
        try:
            graph.invoke(initial, cfg)
        except Exception:
            log.exception(
                "chains.scheduler.launch_invoke_failed", chain_id=chain_id
            )
            _set_run_terminal(chain_id, status="failed")
            return
        maybe_finalize_status(chain_id)
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

    Validates the spec, creates a ``ChainRun`` row, seeds initial state
    in the checkpointer, then invokes the graph. The default
    ``wait=True`` runs the graph synchronously and returns once it
    terminates or hits an interrupt — used by the CLI and the cron
    firing path so callers see the run through.

    Pass ``wait=False`` for fire-and-forget semantics: the graph is
    invoked on a background daemon thread, the ``chain_id`` is returned
    in milliseconds. Used by the MCP so a coach reply that fires a
    chain doesn't block on its completion. Caveat: the background
    thread runs in the *caller's* process — if that process exits
    before the chain finishes, the chain is left in ``running`` status
    and the daemon's ``resume_interrupted_chains`` startup hook picks
    it up on next boot.

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
    with get_session() as sess:
        run = ChainRun(
            chain_id=chain_id,
            chain_name=chain_name,
            bucket=bucket or None,
            template_id=template_id,
            thread_id=thread_id,
            status="running",
            started_at=started,
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

    initial = _initial_state(
        chain_id=chain_id,
        chain_name=chain_name,
        bucket=bucket,
        plan=plan,
        thread_id=thread_id,
        planner_tier=planner_tier,
        vars=merged_vars,
    )

    if wait:
        # Original synchronous path: run the graph here and let the
        # caller see it through.
        graph = get_compiled_chain_graph()
        cfg = _thread_config(chain_id)
        _mark_invoke_active(chain_id)
        try:
            try:
                graph.invoke(initial, cfg)
            except Exception:
                log.exception(
                    "chains.scheduler.launch_invoke_failed", chain_id=chain_id
                )
                _set_run_terminal(chain_id, status="failed")
                raise
            maybe_finalize_status(chain_id)
        finally:
            _mark_invoke_inactive(chain_id)
    else:
        # Fire-and-forget: spawn a daemon thread to run the graph and
        # return the chain_id immediately. The caller (typically the
        # MCP) can include chain_id in its response so the user knows
        # which run to watch.
        threading.Thread(
            target=_invoke_chain_graph,
            args=(chain_id, initial),
            name=f"tasque-chain-invoke-{chain_id[:8]}",
            daemon=True,
        ).start()

    return chain_id


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
            chain_id = launch_chain_run(spec, template_id=template.id)
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
    """Re-invoke the graph for every ``running`` ChainRun. Startup hook.

    Before invoking, any ``failed`` plan nodes are flipped back to
    ``pending`` and their failures cleared, so a bot restart retries
    every failed chain step. This is the recovery path for transient
    failures (LLM API timeouts, network blips) that would otherwise
    wedge a chain forever — fan-out children with ``on_failure="halt"``
    fail, the supervisor returns ``END`` because nothing is running,
    but downstream consumers like a gather step stay ``pending`` and the
    ChainRun row never finalises.

    Rows that are already in awaiting_user state simply pause again at
    the interrupt — that's fine; the resume comes via the reply runtime.
    """
    resumed: list[str] = []
    with get_session() as sess:
        stmt = select(ChainRun).where(ChainRun.status == "running")
        rows = list(sess.execute(stmt).scalars().all())
        for r in rows:
            sess.expunge(r)

    graph = get_compiled_chain_graph()
    for r in rows:
        cfg = _thread_config(r.chain_id)
        _mark_invoke_active(r.chain_id)
        try:
            try:
                reset_ids = _reset_failed_steps_for_resume(r.chain_id)
                if reset_ids:
                    log.info(
                        "chains.scheduler.resume_retrying_failed",
                        chain_id=r.chain_id[:8],
                        chain_name=r.chain_name,
                        steps=reset_ids,
                    )
                graph.invoke(cast(Any, None), cfg)
            except Exception:
                log.exception(
                    "chains.scheduler.resume_failed", chain_id=r.chain_id
                )
                continue
            maybe_finalize_status(r.chain_id)
        finally:
            _mark_invoke_inactive(r.chain_id)
        resumed.append(r.chain_id)
    return resumed


# Staleness threshold for ``resume_stale_chains``. The previous 30s value
# was shorter than a normal opus dispatch turn (60-90s of LLM time during
# which the chain's checkpoint isn't updated) and caused the resume tick
# to re-invoke the graph mid-step, double-dispatching workers and
# producing duplicate proposal rows. 300s is comfortably longer than any
# single LLM turn (the proxy's stall watchdog kills the subprocess after
# 5 min of stdout silence anyway, so a wedged in-process worker can't
# survive longer than this without surfacing as an exception). Combined
# with the ``_active_chain_invokes`` registry, which excludes chains
# whose runner thread is alive in this process from staleness-based
# resume entirely, this eliminates the mid-turn double-dispatch race.
DEFAULT_STALE_RESUME_THRESHOLD_SECONDS = 300.0


def _checkpoint_age_seconds(chain_id: str, *, now: datetime | None = None) -> float | None:
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
    try:
        ts = datetime.fromisoformat(raw_ts)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    reference = now if now is not None else datetime.now(UTC)
    return (reference - ts).total_seconds()


def resume_stale_chains(
    *,
    threshold_seconds: float = DEFAULT_STALE_RESUME_THRESHOLD_SECONDS,
    now: datetime | None = None,
) -> list[str]:
    """Re-invoke the graph for ``running`` chains whose checkpoint is stale.

    ``threshold_seconds`` is the minimum checkpoint age (vs. ``now``)
    before we step in. The daemon's tick uses this to leave actively
    progressing chains alone — a chain currently writing checkpoints
    has a fresh ``ts`` and is skipped.

    Two layers of guard against false-positive resumes:

    1. ``_active_chain_invokes`` — chains currently being run by this
       process are skipped outright, even if their checkpoint looks
       stale (a long LLM turn keeps the runner thread alive but doesn't
       update the checkpoint).
    2. ``threshold_seconds`` — for chains with no in-process owner, the
       checkpoint must be older than the threshold. Defaults to 5 min
       (longer than any single LLM turn the proxy permits).

    Recovery path for chains whose runner thread died with the calling
    process. Common case: the MCP server inside ``claude --print`` calls
    ``launch_chain_run(wait=False)`` which spawns a daemon thread; that
    thread is killed when ``claude --print`` finishes the user's turn,
    leaving the chain in ``running`` with a stale checkpoint. The
    periodic tick re-invokes the graph so the chain progresses without
    needing a daemon restart.

    Returns the list of resumed ``chain_id`` values. Failures during
    invoke are logged but do not abort the loop — one bad chain doesn't
    stop the others from progressing.
    """
    with get_session() as sess:
        stmt = select(ChainRun).where(ChainRun.status == "running")
        rows = list(sess.execute(stmt).scalars().all())
        for r in rows:
            sess.expunge(r)

    graph = get_compiled_chain_graph()
    resumed: list[str] = []
    for r in rows:
        # Skip chains whose runner thread is alive in this process — a
        # stale checkpoint here just means the worker is mid-turn, not
        # that the chain is wedged. Re-invoking would double-dispatch.
        if _is_invoke_active(r.chain_id):
            continue
        age = _checkpoint_age_seconds(r.chain_id, now=now)
        if age is None:
            # No checkpoint at all: the row exists but the calling
            # process died before the graph wrote even one checkpoint.
            # Treat as fully stale so we re-seed and run.
            age = float("inf")
        if age < threshold_seconds:
            continue
        cfg = _thread_config(r.chain_id)
        log.info(
            "chains.scheduler.resume_stale_invoking",
            chain_id=r.chain_id[:8],
            chain_name=r.chain_name,
            age_s=round(age, 1) if age != float("inf") else None,
        )
        _mark_invoke_active(r.chain_id)
        try:
            try:
                graph.invoke(cast(Any, None), cfg)
            except Exception:
                log.exception(
                    "chains.scheduler.resume_stale_failed", chain_id=r.chain_id
                )
                continue
            maybe_finalize_status(r.chain_id)
        finally:
            _mark_invoke_inactive(r.chain_id)
        resumed.append(r.chain_id)
    return resumed


__all__ = [
    "DEFAULT_STALE_RESUME_THRESHOLD_SECONDS",
    "fire_due_chain_templates",
    "launch_chain_run",
    "maybe_finalize_status",
    "resume_interrupted_chains",
    "resume_stale_chains",
]

"""``tasque serve`` — the long-running daemon entry point.

Brings up the Discord bot, the APScheduler-driven job scheduler, the
coach trigger drainer, and (implicitly via on_ready) the chain UI
watcher. All in a single asyncio event loop.

Shutdown is cooperative: SIGINT / SIGTERM cancels the bot task, which
cascades to the watcher; the scheduler shuts down via APScheduler's
synchronous ``shutdown()``; the drainer respects its ``stop`` event.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from typing import Annotated

import structlog
import typer

from tasque.chains.scheduler import (
    DEFAULT_STALE_RESUME_THRESHOLD_SECONDS,
    resume_interrupted_chains,
    resume_stale_chains,
)
from tasque.coach.trigger import run_drainer
from tasque.discord import poster
from tasque.discord.bot import build_bot, run_bot
from tasque.discord.chain_status_watcher import run_chain_status_watcher
from tasque.discord.dlq_watcher import run_dlq_watcher
from tasque.discord.worker_run_watcher import run_worker_run_watcher
from tasque.jobs.scheduler import start_scheduler

DEFAULT_BOT_READY_TIMEOUT_SECONDS = 30.0
# How often the daemon sweeps for chains whose runner thread died with
# the calling process (e.g. an MCP-fired chain in a finished
# ``claude --print`` subprocess). Independent of
# :data:`DEFAULT_STALE_RESUME_THRESHOLD_SECONDS` — the tick is the
# polling cadence (cheap; runs against the in-process active-invoke
# registry first), the threshold is the staleness gate that decides
# whether a stale-looking row is actually wedged.
DEFAULT_STALE_RESUME_TICK_SECONDS = 30.0


async def run_chain_resume_ticker(
    *,
    stop: asyncio.Event,
    interval_seconds: float = DEFAULT_STALE_RESUME_TICK_SECONDS,
    threshold_seconds: float = DEFAULT_STALE_RESUME_THRESHOLD_SECONDS,
) -> None:
    """Drive every chain-resume pass on this daemon.

    First iteration: :func:`resume_interrupted_chains` — full boot
    semantics, including resetting ``failed`` plan nodes back to
    ``pending`` so transient errors get retried after restart.

    Subsequent iterations (every ``interval_seconds``):
    :func:`resume_stale_chains` — staleness-filtered, no failure
    reset. Picks up chains whose runner thread died with the calling
    process (e.g. an MCP-fired chain whose ``claude --print``
    subprocess exited mid-invoke).

    Single source of truth: prior iterations of this codebase had a
    one-shot boot resume task running concurrently with the periodic
    ticker, which double-dispatched the same step on startup whenever
    the boot resume's checkpoint was already stale. Folding the boot
    pass into the ticker's first iteration eliminates the race.

    Each invocation runs in a worker thread so a long-running
    ``graph.invoke`` doesn't block the asyncio loop. Ticks are
    sequential — the next sleep begins after the previous invoke
    returns, so a tick that takes longer than ``interval_seconds``
    just delays the next one rather than overlapping.
    """
    log_local = structlog.get_logger(__name__)
    is_first_tick = True

    while not stop.is_set():
        try:
            if is_first_tick:
                resumed = await asyncio.to_thread(resume_interrupted_chains)
                if resumed:
                    log_local.info("serve.resumed_chains", count=len(resumed))
                is_first_tick = False
            else:
                resumed = await asyncio.to_thread(
                    resume_stale_chains, threshold_seconds=threshold_seconds
                )
                if resumed:
                    log_local.info("serve.chain_resume_tick", resumed=len(resumed))
        except Exception:
            log_local.exception("serve.chain_resume_tick_failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            return
        except TimeoutError:
            continue

serve_app = typer.Typer(
    help="Run the tasque daemon: bot + scheduler + coach drainer + chain UI watcher.",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)

log = structlog.get_logger(__name__)


async def _await_bot_ready(
    bot_task: asyncio.Task[None], *, timeout: float
) -> bool:
    """Block until the bot's ``on_ready`` has installed the poster client.

    Returns True iff the bot reached ready state inside ``timeout``. Returns
    False on timeout or if the bot task died first — the caller logs and
    proceeds anyway because the chain status watcher self-defers and the
    daemon should still come up rather than wedge on a network blip.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not poster.client_ready():
        if bot_task.done():
            exc = bot_task.exception()
            if exc is not None:
                log.error("serve.bot_died_before_ready", error=str(exc))
            return False
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.25)
    return True


async def _run_async(*, drainer_poll_seconds: float) -> None:
    """Boot all components in one event loop.

    Order matters: bring the Discord bot up *first* and wait for
    ``on_ready`` before starting any component that hits the LLM proxy.
    Otherwise ``resume_interrupted_chains``, the APScheduler ticks
    (``fire_due_chain_templates`` and ``claim_and_run_one``), and the
    coach drainer all start dispatching worker calls during the gateway
    handshake — chain status panels never get posted because the watcher
    is still gated on ``poster.client_ready()=False`` and the operator
    sees proxy traffic with no Discord feedback.

    The chain status watcher itself is started early so it's already
    looping (and self-deferring) by the time on_ready fires.
    """
    log.info("serve.starting")

    handle = build_bot()
    bot_task = asyncio.create_task(run_bot(handle), name="tasque-discord-bot")

    chain_status_stop = asyncio.Event()
    chain_status_task = asyncio.create_task(
        run_chain_status_watcher(stop=chain_status_stop),
        name="tasque-chain-status-watcher",
    )

    worker_run_stop = asyncio.Event()
    worker_run_task = asyncio.create_task(
        run_worker_run_watcher(stop=worker_run_stop),
        name="tasque-worker-run-watcher",
    )

    dlq_stop = asyncio.Event()
    dlq_task = asyncio.create_task(
        run_dlq_watcher(stop=dlq_stop),
        name="tasque-dlq-watcher",
    )

    ready = await _await_bot_ready(
        bot_task, timeout=DEFAULT_BOT_READY_TIMEOUT_SECONDS
    )
    if ready:
        log.info("serve.bot_ready")
    else:
        log.warning(
            "serve.bot_ready_timeout",
            timeout=DEFAULT_BOT_READY_TIMEOUT_SECONDS,
            note="proceeding anyway; chain status panels may be delayed",
        )

    scheduler = start_scheduler()
    drainer_stop = asyncio.Event()

    drainer_task = asyncio.create_task(
        run_drainer(stop=drainer_stop, poll_seconds=drainer_poll_seconds),
        name="tasque-coach-drainer",
    )

    # Single source of truth for chain resumes: the ticker's first
    # iteration runs the boot ``resume_interrupted_chains`` pass (with
    # failed-step reset), then it periodically runs the lighter
    # staleness-only ``resume_stale_chains``. Folding both into one
    # task prevents the boot/tick race that double-dispatched the same
    # step on startup.
    chain_resume_stop = asyncio.Event()
    chain_resume_task = asyncio.create_task(
        run_chain_resume_ticker(stop=chain_resume_stop),
        name="tasque-chain-resume-ticker",
    )

    loop = asyncio.get_running_loop()
    stop_received = asyncio.Event()

    def _request_stop() -> None:
        stop_received.set()

    for sig in _shutdown_signals():
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    try:
        done, _ = await asyncio.wait(
            {
                bot_task,
                drainer_task,
                chain_status_task,
                worker_run_task,
                dlq_task,
                chain_resume_task,
                asyncio.create_task(stop_received.wait(), name="tasque-stop-wait"),
            },
            return_when=asyncio.FIRST_COMPLETED,
        )
        for d in done:
            exc = d.exception()
            if exc is not None:
                log.error("serve.task_exited", task=d.get_name(), error=str(exc))
    finally:
        log.info("serve.shutting_down")
        drainer_stop.set()
        chain_status_stop.set()
        worker_run_stop.set()
        dlq_stop.set()
        chain_resume_stop.set()
        if handle.stop is not None:
            handle.stop.set()
        bot_task.cancel()
        drainer_task.cancel()
        chain_status_task.cancel()
        worker_run_task.cancel()
        dlq_task.cancel()
        chain_resume_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await bot_task
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await drainer_task
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await chain_status_task
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await worker_run_task
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await dlq_task
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await chain_resume_task
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            log.exception("serve.scheduler_shutdown_failed")
        log.info("serve.done")


def _shutdown_signals() -> list[signal.Signals]:
    sigs: list[signal.Signals] = []
    if hasattr(signal, "SIGINT"):
        sigs.append(signal.SIGINT)
    if hasattr(signal, "SIGTERM"):
        sigs.append(signal.SIGTERM)
    return sigs


@serve_app.callback()
def serve(
    drainer_poll_seconds: Annotated[
        float,
        typer.Option(
            "--drainer-poll-seconds",
            help="How often the coach drainer polls for new triggers.",
        ),
    ] = 1.0,
) -> None:
    """Run the tasque daemon."""
    # Already handled cooperatively above on POSIX; on Windows the
    # asyncio loop bails before our signal handler runs, so we land
    # here. Either way, exit cleanly.
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run_async(drainer_poll_seconds=drainer_poll_seconds))


__all__ = ["serve_app"]

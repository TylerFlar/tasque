"""Job runner + APScheduler poll loop.

Public surface:

- ``run_worker(job)``   — pure function: execute one QueuedJob via the
  worker LangGraph and return a ``WorkerResult``.
- ``start_scheduler()`` — start the BackgroundScheduler ticking every 5s,
  draining the QueuedJob queue serially.

Lower-level helpers (``claim_one``, ``sweep_stuck_jobs``, ``record_failure``)
are exported via their modules; CLI and tests import directly.
"""

from tasque.jobs.runner import WorkerResult, run_worker
from tasque.jobs.scheduler import claim_and_run_one, start_scheduler

__all__ = [
    "WorkerResult",
    "claim_and_run_one",
    "run_worker",
    "start_scheduler",
]

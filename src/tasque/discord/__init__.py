"""Discord transport.

One nextcord bot, one parameterised reply runtime, one chain UI, three
notification helpers. The CLI is the read interface; Discord is the
notify-and-reply interface.
"""

from __future__ import annotations

from tasque.discord.notify import (
    notify_chain_event,
    notify_chain_terminal,
    notify_failed_job,
    notify_worker_run,
)
from tasque.discord.threads import (
    PURPOSE_DLQ,
    PURPOSE_JOBS,
    bucket_purpose,
    ensure_threads,
    get_thread_id,
)

__all__ = [
    "PURPOSE_DLQ",
    "PURPOSE_JOBS",
    "bucket_purpose",
    "ensure_threads",
    "get_thread_id",
    "notify_chain_event",
    "notify_chain_terminal",
    "notify_failed_job",
    "notify_worker_run",
]

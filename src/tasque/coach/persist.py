"""Surface a parsed ``BucketCoachOutput`` to the runtime.

The bucket coach performs its writes — Notes, queued jobs, chain
fires, signals — in-turn via the tasque MCP. The structured JSON it
emits afterwards carries one field: ``thread_post``, an optional
markdown body the coach wants the Discord bot to publish to the
bucket's thread.

This function returns a small report dict the caller can act on.
"""

from __future__ import annotations

from typing import Any

from tasque.buckets import Bucket
from tasque.coach.output import BucketCoachOutput


def persist_results(bucket: Bucket, parsed: BucketCoachOutput) -> dict[str, Any]:
    """Return a small report surfacing ``thread_post`` to the caller.

    The caller (CLI, Discord runtime) decides whether to publish it.
    """
    return {
        "bucket": bucket,
        "thread_post": parsed.thread_post,
    }

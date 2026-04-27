"""Persist a parsed ``StrategistOutput`` and post the summary to Discord.

Writes:

- one ``Aim(source="strategist")`` per ``new_aims`` entry
- one ``Signal(from_bucket="strategist")`` per ``signals`` entry
- ``Aim.status`` flips for every ``aim_status_changes`` entry
- finally, posts the markdown ``summary`` to the strategist's
  Discord thread via :func:`tasque.discord.poster.post_long_message`

The Discord post is the strategist's only outbound side effect; if no
strategist thread is registered, the summary is logged and skipped (so
the run still records its DB writes).
"""

from __future__ import annotations

from typing import Any

import structlog

from tasque.discord import poster, threads
from tasque.memory.entities import Aim, Signal
from tasque.memory.repo import update_entity_status, write_entity
from tasque.strategist.output import StrategistOutput

log = structlog.get_logger(__name__)

STRATEGIST_FROM = "strategist"


def _persist_aim(item: Any) -> str:
    """Write one ``AimOutput`` as an Aim row and return its id."""
    bucket = item.normalised_bucket()
    aim = Aim(
        title=item.title,
        bucket=bucket,
        scope=item.scope,
        target_date=item.target_date,
        description=item.description,
        status="active",
        parent_id=item.parent_id,
        source=STRATEGIST_FROM,
    )
    written = write_entity(aim)
    return written.id


def _persist_signal(item: Any) -> str:
    """Write one ``SignalOutput`` as a Signal row and return its id."""
    to_bucket = item.normalised_to_bucket()
    sig = Signal(
        from_bucket=STRATEGIST_FROM,
        to_bucket=to_bucket,
        kind=item.kind,
        urgency=item.urgency,
        summary=item.summary,
        body=item.body,
        expires_at=item.expires_at,
    )
    written = write_entity(sig)
    return written.id


def persist_results(parsed: StrategistOutput) -> dict[str, Any]:
    """Persist Aim/Signal/status writes from a strategist monitoring run.

    Does NOT post to Discord — that's :func:`post_summary`. Split so
    callers (tests, CLI) can run the DB writes without a poster client
    installed.
    """
    new_aim_ids: list[str] = [_persist_aim(a) for a in parsed.new_aims]
    signal_ids: list[str] = [_persist_signal(s) for s in parsed.signals]
    flipped: list[str] = []
    for change in parsed.aim_status_changes:
        if update_entity_status(change.aim_id, change.status):
            flipped.append(change.aim_id)
        else:
            log.warning(
                "strategist.persist.aim_status_change_failed",
                aim_id=change.aim_id,
                status=change.status,
            )
    return {
        "new_aim_ids": new_aim_ids,
        "signal_ids": signal_ids,
        "flipped_aim_ids": flipped,
        "summary": parsed.summary,
    }


async def post_summary(summary: str) -> list[int]:
    """Post the strategist summary to its Discord thread.

    Returns the list of message ids written. If no strategist thread is
    registered (or no poster client is installed), logs and returns an
    empty list — the DB writes are independent of the Discord publish.
    """
    if not summary.strip():
        return []
    thread_id = threads.get_thread_id(threads.PURPOSE_STRATEGIST)
    if thread_id is None:
        log.info("strategist.persist.no_thread_registered")
        return []
    try:
        return await poster.post_long_message(thread_id, summary)
    except RuntimeError:
        log.info("strategist.persist.no_poster_client")
        return []


__all__ = ["STRATEGIST_FROM", "persist_results", "post_summary"]

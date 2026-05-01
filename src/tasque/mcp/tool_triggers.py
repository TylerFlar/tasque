"""Fire bucket-coach triggers when MCP write tools mutate state.

Until now coaches woke on three sources: scheduled (cron / explicit
``tasque coach wake``), worker completion, and Discord reply. None of
those covered "another Claude session, or the user via an MCP host,
just wrote a Note in this bucket" — that change sat in the database
until something else fired the coach.

This module plugs into the MCP write tools so that successful
mutations enqueue a coach trigger for the affected bucket. The
existing dedup window in :func:`tasque.coach.trigger.enqueue` (default
five minutes, post-claim) coalesces repeated writes — a coach run that
writes ten notes in one turn produces at most one follow-up trigger,
not ten. That dedup is also what prevents an infinite loop when the
follow-up coach run writes more notes itself.

Failure handling: dispatch is best-effort. A failure to enqueue MUST
NOT fail the underlying mutation — the user's note has already been
written; the trigger is gravy. Errors are logged and swallowed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from tasque.buckets import ALL_BUCKETS, Bucket
from tasque.coach.trigger import enqueue as _default_enqueue

log = structlog.get_logger(__name__)


# Indirection so tests can monkeypatch this module attribute and
# observe the calls without touching the real coach trigger queue.
EnqueueFn = Callable[..., str | None]
enqueue_fn: EnqueueFn = _default_enqueue


def _coerce_bucket(value: Any) -> Bucket | None:
    if isinstance(value, str) and value in ALL_BUCKETS:
        return value  # type: ignore[return-value]
    return None


def _wake_note_bucket(tool_name: str, bucket: Any) -> None:
    b = _coerce_bucket(bucket)
    if b is None:
        return
    enqueue_fn(
        b,
        reason=f"tool:{tool_name}:{b}",
        dedup_key=f"tool:{tool_name}:{b}",
    )


def _on_note_create(*, bucket: Any, **_: Any) -> None:
    _wake_note_bucket("note_create", bucket)


def _on_note_update(*, bucket: Any, **_: Any) -> None:
    _wake_note_bucket("note_update", bucket)


def _on_note_supersede(*, bucket: Any, **_: Any) -> None:
    _wake_note_bucket("note_supersede", bucket)


def _on_signal_create(*, to_bucket: Any, from_bucket: Any = None, **_: Any) -> None:
    # Broadcasts wake every bucket coach. Each gets its own dedup key so
    # a single broadcast can't mute another bucket's same-window trigger
    # from an unrelated source.
    if to_bucket == "all":
        for b in ALL_BUCKETS:
            enqueue_fn(
                b,
                reason="tool:signal_create:all",
                dedup_key=f"tool:signal_create:all:{b}",
            )
        return
    target = _coerce_bucket(to_bucket)
    if target is None:
        return
    enqueue_fn(
        target,
        reason=f"tool:signal_create:{target}",
        dedup_key=f"tool:signal_create:{target}",
    )


def _on_aim_create(*, bucket: Any, scope: Any, **_: Any) -> None:
    # Long-term Aims live above any single bucket and have no bucket
    # coach to wake. Bucket-scoped Aims wake the coach that owns them.
    if scope != "bucket":
        return
    b = _coerce_bucket(bucket)
    if b is None:
        return
    enqueue_fn(
        b,
        reason=f"tool:aim_create:{b}",
        dedup_key=f"tool:aim_create:{b}",
    )


_HANDLERS: dict[str, Callable[..., None]] = {
    "note_create": _on_note_create,
    "note_update": _on_note_update,
    "note_supersede": _on_note_supersede,
    "signal_create": _on_signal_create,
    "aim_create": _on_aim_create,
}


def dispatch_tool_event(tool_name: str, **fields: Any) -> None:
    """Invoke the registered handler for ``tool_name``, swallowing errors.

    Call this AFTER the underlying mutation has been written, never
    before — the coach trigger should only fire when the state change
    actually happened.
    """
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return
    try:
        handler(**fields)
    except Exception as exc:
        log.warning(
            "mcp.tool_trigger.failed",
            tool=tool_name,
            fields=sorted(fields.keys()),
            error=str(exc),
        )


__all__ = ["dispatch_tool_event", "enqueue_fn"]

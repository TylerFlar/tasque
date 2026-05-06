"""Fire bucket-coach triggers for explicit MCP work-starting mutations.

This dispatcher is intentionally narrow. Notes are memory edits, not
implicit requests for more work, and worker completion is reported
directly by the worker-run watcher. The only MCP writes that wake bucket
coaches are bucket-scoped Aim creation and Signals.

Keeping this surface small matters: a coach reply often writes a compact
Note before queueing a job, and a broad "every write wakes a coach" rule
can turn that bookkeeping into another full action pass.

Failure handling: dispatch is best-effort. A failure to enqueue MUST
NOT fail the underlying mutation. Errors are logged and swallowed.
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


def _signal_dedup_key(
    *,
    target: str,
    kind: Any,
    from_bucket: Any,
) -> str:
    # The strategist normally creates a bucket Aim and an aim_added Signal
    # in the same turn. Both should wake the coach, but not twice.
    if kind == "aim_added" and from_bucket == "strategist":
        return f"tool:aim_create:{target}"
    return f"tool:signal_create:{target}"


def _on_signal_create(
    *,
    to_bucket: Any,
    from_bucket: Any = None,
    kind: Any = None,
    **_: Any,
) -> None:
    # Broadcasts wake every bucket coach. Each gets its own dedup key so
    # a single broadcast can't mute another bucket's same-window trigger
    # from an unrelated source.
    if to_bucket == "all":
        for b in ALL_BUCKETS:
            enqueue_fn(
                b,
                reason="tool:signal_create:all",
                dedup_key=(
                    f"tool:aim_create:{b}"
                    if kind == "aim_added" and from_bucket == "strategist"
                    else f"tool:signal_create:all:{b}"
                ),
            )
        return
    target = _coerce_bucket(to_bucket)
    if target is None:
        return
    enqueue_fn(
        target,
        reason=f"tool:signal_create:{target}",
        dedup_key=_signal_dedup_key(
            target=target,
            kind=kind,
            from_bucket=from_bucket,
        ),
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
    "signal_create": _on_signal_create,
    "aim_create": _on_aim_create,
}


def dispatch_tool_event(tool_name: str, **fields: Any) -> None:
    """Invoke the registered handler for ``tool_name``, swallowing errors.

    Call this AFTER the underlying mutation has been written, never
    before. The coach trigger should only fire when the state change
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

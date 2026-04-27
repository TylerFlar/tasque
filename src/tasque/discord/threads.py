"""Persistent thread→purpose registry.

The Discord workspace has one thread per bucket (in the coach channel),
one DLQ thread, a strategist thread, and a fallback "jobs" channel for
cross-bucket worker notifications. Per-chain threads are created
on-demand by the chain UI.

The registry is persisted at ``data/discord_threads.json`` so a daemon
restart doesn't recreate threads. Per-chain threads are *not* tracked
here — the chain UI creates them on demand and stores their ids in the
``ChainState.thread_id`` checkpoint field.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from tasque.buckets import ALL_BUCKETS, Bucket
from tasque.config import get_settings

PURPOSE_DLQ = "dlq"
PURPOSE_JOBS = "jobs"  # fallback for cross-bucket / bucket-less worker runs
PURPOSE_STRATEGIST = "strategist"
# Stores the *message id* of the live-edited ops panel embed (NOT a
# thread or channel). The channel itself comes from the
# ``TASQUE_DISCORD_OPS_CHANNEL_ID`` env var; the watcher writes this
# entry once after the first post so subsequent ticks edit in place.
PURPOSE_OPS_PANEL = "ops_panel"


def bucket_purpose(bucket: Bucket) -> str:
    """Return the registry key for a bucket's coach thread."""
    if bucket not in ALL_BUCKETS:
        raise ValueError(f"unknown bucket: {bucket!r}")
    return f"bucket:{bucket}"


_lock = Lock()
_cache: dict[str, int] | None = None


def _registry_path() -> Path:
    raw = os.environ.get("TASQUE_DISCORD_THREAD_REGISTRY")
    if raw:
        return Path(raw)
    settings = get_settings()
    return settings.data_dir / "discord_threads.json"


def _load() -> dict[str, int]:
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is None:
            path = _registry_path()
            if not path.exists():
                _cache = {}
            else:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    _cache = {}
                else:
                    out: dict[str, int] = {}
                    for k, v in raw.items():
                        if isinstance(k, str) and isinstance(v, int):
                            out[k] = v
                    _cache = out
    return _cache


def _save() -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_load(), indent=2, sort_keys=True), encoding="utf-8")


def reset_cache() -> None:
    """Drop the in-memory cache so the next read reloads from disk.
    Tests use this to isolate runs from each other."""
    global _cache
    _cache = None


def get_thread_id(purpose: str) -> int | None:
    """Return the persisted thread id for ``purpose``, or None if missing."""
    return _load().get(purpose)


def set_thread_id(purpose: str, thread_id: int) -> None:
    """Record ``purpose → thread_id`` and flush to disk."""
    cache = _load()
    cache[purpose] = thread_id
    _save()


def all_thread_ids() -> dict[str, int]:
    """Return a copy of the full registry — for debugging and the bot's
    on_ready cross-check."""
    return dict(_load())


ThreadCreator = Any  # async Callable[[str, str], int] supplied by the bot


async def ensure_threads(
    *,
    create_missing: ThreadCreator | None = None,
) -> dict[str, int]:
    """Ensure each known purpose has a thread id. Returns the full map.

    For each purpose that's missing from the registry, ``create_missing``
    is called as ``await create_missing(purpose, label)`` and is
    expected to return the new thread id (or raise). If
    ``create_missing`` is None, missing threads are simply left
    unmapped — the daemon can run with a partial registry as long as
    the user fills it in via the bot's slash command (added later) or
    by editing the JSON file by hand.

    The known purposes:

    - one ``bucket:<bucket>`` per bucket
    - ``dlq``
    - ``jobs`` (fallback for bucketless runs)
    - ``strategist`` (cross-bucket conversational thread)
    """
    desired: list[tuple[str, str]] = []
    for b in ALL_BUCKETS:
        desired.append((bucket_purpose(b), f"coach-{b}"))
    desired.append((PURPOSE_DLQ, "dlq"))
    desired.append((PURPOSE_JOBS, "jobs"))
    desired.append((PURPOSE_STRATEGIST, "strategist"))

    cache = _load()
    for purpose, label in desired:
        if purpose in cache:
            continue
        if create_missing is None:
            continue
        new_id = await create_missing(purpose, label)
        if isinstance(new_id, int):
            # Persist after each success so a mid-loop failure (e.g. a
            # Discord 429 on the next iteration) doesn't strand the
            # threads we already created — without this, the next
            # startup would recreate them as zombies and hit the same
            # rate limit again.
            cache[purpose] = new_id
            _save()
    return dict(cache)


__all__ = [
    "PURPOSE_DLQ",
    "PURPOSE_JOBS",
    "PURPOSE_OPS_PANEL",
    "PURPOSE_STRATEGIST",
    "all_thread_ids",
    "bucket_purpose",
    "ensure_threads",
    "get_thread_id",
    "reset_cache",
    "set_thread_id",
]

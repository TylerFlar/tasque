"""Tests for the Discord thread registry — make sure bucket thread ids
get cached, persist to disk, and aren't recreated on the next startup."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tasque.buckets import ALL_BUCKETS
from tasque.discord import threads


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path: Path) -> Iterator[Path]:
    """Per-test registry path so each case is isolated."""
    registry = tmp_path / "discord_threads.json"
    old = os.environ.get("TASQUE_DISCORD_THREAD_REGISTRY")
    os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = registry.as_posix()
    threads.reset_cache()
    try:
        yield registry
    finally:
        threads.reset_cache()
        if old is None:
            os.environ.pop("TASQUE_DISCORD_THREAD_REGISTRY", None)
        else:
            os.environ["TASQUE_DISCORD_THREAD_REGISTRY"] = old


# ----------------------------------------------------------------- basics


def test_set_thread_id_persists_to_disk(isolated_registry: Path) -> None:
    threads.set_thread_id(threads.bucket_purpose("health"), 1234567890)
    # File on disk should hold the id.
    assert isolated_registry.exists()
    raw = json.loads(isolated_registry.read_text(encoding="utf-8"))
    assert raw == {"bucket:health": 1234567890}


def test_get_thread_id_reads_from_disk_after_cache_drop(
    isolated_registry: Path,
) -> None:
    threads.set_thread_id(threads.bucket_purpose("career"), 4242)
    # Drop the in-memory cache; the value should still come back from disk.
    threads.reset_cache()
    assert threads.get_thread_id(threads.bucket_purpose("career")) == 4242


def test_unknown_bucket_rejects() -> None:
    with pytest.raises(ValueError):
        threads.bucket_purpose("not-a-bucket")  # type: ignore[arg-type]


# ----------------------------------------------------------------- ensure_threads


@pytest.mark.asyncio
async def test_ensure_threads_creates_missing_bucket_ids(
    isolated_registry: Path,
) -> None:
    """First call should ask the creator for every purpose and record what
    it returns. Names land in the JSON file."""
    calls: list[tuple[str, str]] = []

    async def creator(purpose: str, label: str) -> int:
        calls.append((purpose, label))
        # Stable synthetic id keyed off the purpose so the test can
        # assert on the specific value.
        return abs(hash(purpose)) % 10_000_000

    result = await threads.ensure_threads(create_missing=creator)

    # One call per known purpose: 9 buckets + dlq + jobs + strategist.
    expected_purposes = (
        {threads.bucket_purpose(b) for b in ALL_BUCKETS}
        | {threads.PURPOSE_DLQ, threads.PURPOSE_JOBS, threads.PURPOSE_STRATEGIST}
    )
    assert {p for p, _ in calls} == expected_purposes
    assert set(result.keys()) == expected_purposes
    # Each bucket purpose got the creator's hashed id.
    for b in ALL_BUCKETS:
        purpose = threads.bucket_purpose(b)
        assert result[purpose] == abs(hash(purpose)) % 10_000_000

    # And the registry on disk matches.
    on_disk = json.loads(isolated_registry.read_text(encoding="utf-8"))
    assert on_disk == result


@pytest.mark.asyncio
async def test_ensure_threads_does_not_recreate_cached_buckets(
    isolated_registry: Path,
) -> None:
    """Second call should be a no-op for already-cached purposes, even
    after the in-memory cache is dropped (simulating a daemon restart)."""
    # Pre-populate one bucket and the dlq channel id.
    threads.set_thread_id(threads.bucket_purpose("health"), 111)
    threads.set_thread_id(threads.PURPOSE_DLQ, 222)
    # Simulate a process restart by dropping the in-memory cache.
    threads.reset_cache()

    calls: list[str] = []

    async def creator(purpose: str, _label: str) -> int:
        calls.append(purpose)
        return 9999

    await threads.ensure_threads(create_missing=creator)

    # The pre-populated entries must NOT have been re-created.
    assert "bucket:health" not in calls
    assert threads.PURPOSE_DLQ not in calls
    # Cached values survived.
    assert threads.get_thread_id(threads.bucket_purpose("health")) == 111
    assert threads.get_thread_id(threads.PURPOSE_DLQ) == 222
    # The remaining 8 buckets + jobs + strategist were created.
    assert len(calls) == len(ALL_BUCKETS) - 1 + 2


@pytest.mark.asyncio
async def test_ensure_threads_idempotent_on_second_run(
    isolated_registry: Path,
) -> None:
    """Calling ensure_threads twice in a row should not call the creator
    on the second run."""
    seen: list[str] = []

    async def creator(purpose: str, _label: str) -> int:
        seen.append(purpose)
        return 7000

    await threads.ensure_threads(create_missing=creator)
    first_count = len(seen)
    await threads.ensure_threads(create_missing=creator)
    assert len(seen) == first_count, (
        "second ensure_threads call should not re-invoke the creator"
    )


@pytest.mark.asyncio
async def test_ensure_threads_with_no_creator_leaves_missing_unset(
    isolated_registry: Path,
) -> None:
    """Without a creator, missing purposes are simply not populated —
    the daemon can run with a partial registry."""
    result = await threads.ensure_threads(create_missing=None)
    assert result == {}
    assert not isolated_registry.exists()


@pytest.mark.asyncio
async def test_ensure_threads_only_records_int_returns(
    isolated_registry: Path,
) -> None:
    """A creator that returns a non-int (e.g. None on failure) should not
    poison the registry with a junk value."""
    real_returns = {threads.bucket_purpose("health"): 11, threads.PURPOSE_JOBS: 99}

    async def creator(purpose: str, _label: str) -> Any:
        return real_returns.get(purpose)  # None for everything else

    await threads.ensure_threads(create_missing=creator)
    on_disk = json.loads(isolated_registry.read_text(encoding="utf-8"))
    # Only the two purposes that returned ints made it into the registry.
    assert on_disk == real_returns

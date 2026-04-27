"""Startup-time validation that every required Discord channel id is set.

Every chain/job/coach/ops/dlq channel id is required — there are no
fallbacks. ``require_channel_ids`` is the single point of truth and
``build_bot`` calls it before nextcord is even instantiated, so a
misconfigured deployment fails fast with a clear error listing every
missing/invalid variable in one shot.
"""

from __future__ import annotations

import pytest

from tasque.discord.bot import _REQUIRED_CHANNEL_ENV_VARS, require_channel_ids


def _set_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plant a known integer in every required channel-id env var."""
    for i, name in enumerate(_REQUIRED_CHANNEL_ENV_VARS, start=1):
        monkeypatch.setenv(name, str(1000 + i))


def test_returns_full_mapping_when_all_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_all(monkeypatch)
    out = require_channel_ids()
    assert set(out) == set(_REQUIRED_CHANNEL_ENV_VARS)
    for v in out.values():
        assert isinstance(v, int)


def test_raises_listing_every_missing_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """All five missing — the error should name all five so the user
    can fix the .env in one pass."""
    for name in _REQUIRED_CHANNEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        require_channel_ids()
    msg = str(excinfo.value)
    for name in _REQUIRED_CHANNEL_ENV_VARS:
        assert name in msg


def test_raises_when_one_var_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_all(monkeypatch)
    monkeypatch.delenv("TASQUE_DISCORD_JOBS_CHANNEL_ID", raising=False)
    with pytest.raises(RuntimeError, match="TASQUE_DISCORD_JOBS_CHANNEL_ID"):
        require_channel_ids()


def test_raises_when_one_var_is_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_all(monkeypatch)
    monkeypatch.setenv("TASQUE_DISCORD_DLQ_CHANNEL_ID", "not-an-int")
    with pytest.raises(RuntimeError, match="not an integer"):
        require_channel_ids()


def test_groups_missing_and_invalid_in_one_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mix of missing + invalid should be reported together so the
    operator sees both classes of problem in one error."""
    _set_all(monkeypatch)
    monkeypatch.delenv("TASQUE_DISCORD_JOBS_CHANNEL_ID", raising=False)
    monkeypatch.setenv("TASQUE_DISCORD_OPS_CHANNEL_ID", "garbage")
    with pytest.raises(RuntimeError) as excinfo:
        require_channel_ids()
    msg = str(excinfo.value)
    assert "TASQUE_DISCORD_JOBS_CHANNEL_ID" in msg
    assert "TASQUE_DISCORD_OPS_CHANNEL_ID" in msg
    assert "missing" in msg
    assert "not an integer" in msg

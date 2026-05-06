"""Settings for tasque. Loads from env vars and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".tasque")
    db_path: Path = Field(default=Path())
    tasque_timezone: str = "UTC"

    # --- nondurable-memory decay knobs ---------------------------------
    # Ephemeral Notes older than this are archived. More precise lifecycle
    # fields on Note (memory_kind / ttl_days / canonical_key) now drive most
    # cleanup; this is the compatibility fallback.
    decay_notes_cutoff_days: int = 7
    # Worker/chain artifacts are raw residue, not curated memory.
    decay_artifact_cutoff_days: int = 3
    # Working notes and unresolved questions should not linger by default.
    decay_working_cutoff_days: int = 14
    decay_question_cutoff_days: int = 14
    # Notes whose ``superseded_by`` was set more than this many days ago
    # get archived — once a chain is established, the older versions are
    # dead weight.
    decay_superseded_cutoff_days: int = 7
    # If set, rows that have been ``archived=True`` longer than this are
    # hard-deleted. ``None`` means never hard-delete (the default — keep
    # archived rows around for forensic recall and chain integrity).
    decay_hard_delete_cutoff_days: int | None = 30
    # How often the in-process scheduler runs the sweep.
    decay_sweep_interval_hours: int = 24

    @model_validator(mode="before")
    @classmethod
    def _resolve_db_path(cls, data: Any) -> Any:
        if isinstance(data, dict) and "db_path" not in data:
            data_dir = data.get("data_dir") or (Path.home() / ".tasque")
            data["db_path"] = Path(data_dir) / "tasque.db"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached Settings instance — useful in tests after env changes."""
    get_settings.cache_clear()

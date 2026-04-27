"""Loaders for the coach scaffold + per-bucket markdown prompts.

The scaffold lives at ``prompts/coach_scaffold.md`` (shared output spec,
time-block placeholder, "you are reactive" framing). Per-bucket files live
at ``coach_prompts/<bucket>.md`` — each contains only the bucket mindset.
At request time, scaffold + bucket file are concatenated.

Both files are cached by mtime so the daemon picks up edits on the next
coach run without restart.
"""

from __future__ import annotations

import os
from pathlib import Path

from tasque.buckets import ALL_BUCKETS, Bucket

# Resolve repo root via this file's location: src/tasque/coach/prompts.py
# → up three levels → repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCAFFOLD_PATH = _REPO_ROOT / "prompts" / "coach_scaffold.md"
DEFAULT_BUCKET_PROMPT_DIR = _REPO_ROOT / "coach_prompts"


def _scaffold_path() -> Path:
    raw = os.environ.get("TASQUE_COACH_SCAFFOLD")
    return Path(raw) if raw else DEFAULT_SCAFFOLD_PATH


def _bucket_prompt_dir() -> Path:
    raw = os.environ.get("TASQUE_COACH_PROMPT_DIR")
    return Path(raw) if raw else DEFAULT_BUCKET_PROMPT_DIR


# (path, mtime_ns) → cached text.
_cache: dict[tuple[str, int], str] = {}


def _read_cached(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    text = path.read_text(encoding="utf-8")
    _cache[key] = text
    # Drop any older mtime entries for the same path so the cache doesn't grow.
    for k in [k for k in _cache if k[0] == str(path) and k[1] != stat.st_mtime_ns]:
        del _cache[k]
    return text


def load_scaffold() -> str:
    """Return the scaffold prompt text, reading from disk on cache miss."""
    return _read_cached(_scaffold_path())


def load_bucket_prompt(bucket: Bucket) -> str:
    """Return the per-bucket mindset prompt text. Raises on missing file."""
    if bucket not in ALL_BUCKETS:
        raise ValueError(f"unknown bucket: {bucket!r}")
    path = _bucket_prompt_dir() / f"{bucket}.md"
    if not path.exists():
        raise FileNotFoundError(f"missing coach prompt for bucket {bucket!r}: {path}")
    return _read_cached(path)


def build_system_prompt(bucket: Bucket) -> str:
    """Return ``scaffold + "\\n\\n" + bucket-mindset`` ready for the LLM."""
    return f"{load_scaffold().rstrip()}\n\n{load_bucket_prompt(bucket).strip()}\n"


def reset_cache() -> None:
    """Clear the prompt cache. Tests use this between runs."""
    _cache.clear()

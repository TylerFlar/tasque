"""Discover ``chains/templates/*.yaml`` files and upsert into the DB.

YAML is seed-and-export format only — the DB row's ``plan_json`` is
canonical. ``reload_templates`` will *not* clobber a row that the LLM has
edited since the last seed; the user must call
:func:`export_template_to_yaml` to capture the runtime state first.

LLM-edit detection: every successful seed writes the canonical
``sha256(plan_json)`` into ``ChainTemplate.seed_plan_hash``. On reload,
that hash is compared to the live row; a mismatch means somebody (CRUD,
the planner) has touched the plan and we skip with a warning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

import structlog
import yaml
from sqlalchemy import select

from tasque.chains.crud import enforce_mirror
from tasque.chains.spec import SpecError, validate_spec
from tasque.memory.db import get_session
from tasque.memory.entities import ChainTemplate, utc_now_iso

log = structlog.get_logger(__name__)

DEFAULT_TEMPLATES_DIR: Path = Path("chains") / "templates"


class ReloadReport(TypedDict):
    """Summary of a single ``reload_templates`` call."""

    added: list[str]
    updated: list[str]
    skipped: list[str]
    errors: list[dict[str, str]]


def _hash_plan(plan_json: str) -> str:
    return hashlib.sha256(plan_json.encode("utf-8")).hexdigest()


def _normalize_for_hash(plan_dict: dict[str, Any]) -> str:
    """Canonical JSON form used for hashing — sorted keys, no whitespace."""
    return json.dumps(plan_dict, sort_keys=True, separators=(",", ":"))


def _load_yaml_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SpecError(f"YAML at {path} did not parse to a mapping")
    raw_d: dict[str, Any] = dict(raw)
    # The filename stem is authoritative for chain_name; warn but accept
    # if the YAML disagrees.
    stem = path.stem
    if "chain_name" not in raw_d:
        raw_d["chain_name"] = stem
    elif raw_d["chain_name"] != stem:
        raise SpecError(
            f"YAML at {path} has chain_name={raw_d['chain_name']!r} but the "
            f"filename stem is {stem!r}; they must match."
        )
    return raw_d


def _existing_by_name(name: str) -> ChainTemplate | None:
    with get_session() as sess:
        stmt = select(ChainTemplate).where(ChainTemplate.chain_name == name)
        row = sess.execute(stmt).scalars().first()
        if row is not None:
            sess.expunge(row)
        return row


def _seed_or_update(
    spec: dict[str, Any],
    *,
    seed_path: Path,
    report: ReloadReport,
) -> None:
    name = spec["chain_name"]
    plan_json_canonical = _normalize_for_hash(spec)
    new_hash = _hash_plan(plan_json_canonical)
    new_plan_json = json.dumps(spec)
    recurrence = spec.get("recurrence")
    bucket = spec.get("bucket")

    existing = _existing_by_name(name)
    if existing is None:
        with get_session() as sess:
            row = ChainTemplate(
                chain_name=name,
                bucket=bucket,
                recurrence=recurrence,
                enabled=True,
                plan_json=new_plan_json,
                seed_path=seed_path.as_posix(),
                seed_plan_hash=new_hash,
            )
            sess.add(row)
            sess.flush()
        report["added"].append(name)
        return

    # Existing row — check for LLM edits.
    current_hash = _hash_plan(_normalize_for_hash(json.loads(existing.plan_json)))
    seed_hash = existing.seed_plan_hash
    if seed_hash is not None and current_hash != seed_hash:
        log.warning(
            "chains.registry.skip_llm_edited",
            chain_name=name,
            seed_path=seed_path.as_posix(),
        )
        report["skipped"].append(name)
        return

    # If the YAML didn't change either, nothing to do (still count as
    # updated for visibility — the operation completed cleanly).
    if current_hash == new_hash:
        report["updated"].append(name)
        return

    with get_session() as sess:
        stmt = select(ChainTemplate).where(ChainTemplate.chain_name == name)
        row = sess.execute(stmt).scalars().first()
        if row is None:
            return
        row.plan_json = new_plan_json
        row.bucket = bucket
        row.recurrence = recurrence
        row.seed_path = seed_path.as_posix()
        row.seed_plan_hash = new_hash
        row.updated_at = utc_now_iso()
        sess.flush()
    report["updated"].append(name)


def reload_templates(
    *, templates_dir: Path | None = None
) -> ReloadReport:
    """Scan ``templates_dir`` for ``*.yaml`` and upsert each into the DB.

    Returns a report dict with ``added``, ``updated``, ``skipped``, and
    ``errors`` lists. Skipped rows have been LLM-edited since the last
    seed; the operator can capture them via :func:`export_template_to_yaml`.
    """
    base = templates_dir if templates_dir is not None else DEFAULT_TEMPLATES_DIR
    report: ReloadReport = {
        "added": [],
        "updated": [],
        "skipped": [],
        "errors": [],
    }
    if not base.exists():
        log.info("chains.registry.no_templates_dir", path=base.as_posix())
        return report

    for path in sorted(base.glob("*.yaml")):
        try:
            spec = _load_yaml_spec(path)
            validate_spec(spec)
            enforce_mirror(spec, recurrence=spec.get("recurrence"))
        except SpecError as exc:
            log.error(
                "chains.registry.invalid_yaml",
                path=path.as_posix(),
                error=str(exc),
            )
            report["errors"].append({"path": path.as_posix(), "error": str(exc)})
            continue
        except Exception as exc:
            log.exception("chains.registry.load_failed", path=path.as_posix())
            report["errors"].append({"path": path.as_posix(), "error": str(exc)})
            continue

        try:
            _seed_or_update(spec, seed_path=path, report=report)
        except Exception as exc:
            log.exception("chains.registry.seed_failed", path=path.as_posix())
            report["errors"].append({"path": path.as_posix(), "error": str(exc)})

    return report


def export_template_to_yaml(
    name: str, path: Path | None = None
) -> Path:
    """Write the row's current state to YAML.

    If ``path`` is omitted, the row's existing ``seed_path`` is used; if
    that is also missing, the file is written at
    ``chains/templates/<name>.yaml``. The new path is recorded as the
    row's ``seed_path`` and the hash is updated so a subsequent
    ``reload_templates`` treats the row as freshly seeded.
    """
    row = _existing_by_name(name)
    if row is None:
        raise ValueError(f"no chain template named {name!r}")

    target: Path
    if path is not None:
        target = path
    elif row.seed_path:
        target = Path(row.seed_path)
    else:
        target = DEFAULT_TEMPLATES_DIR / f"{name}.yaml"

    target.parent.mkdir(parents=True, exist_ok=True)
    spec_dict = json.loads(row.plan_json)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(spec_dict, f, sort_keys=False, allow_unicode=True)

    new_hash = _hash_plan(_normalize_for_hash(spec_dict))
    with get_session() as sess:
        live = sess.get(ChainTemplate, row.id)
        if live is not None:
            live.seed_path = target.as_posix()
            live.seed_plan_hash = new_hash
            live.updated_at = utc_now_iso()
    return target


__all__ = [
    "DEFAULT_TEMPLATES_DIR",
    "ReloadReport",
    "export_template_to_yaml",
    "reload_templates",
]

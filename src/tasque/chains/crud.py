"""Programmatic CRUD over ``ChainTemplate`` rows.

The reply-time LLM tools wrap these. ``plan_json`` is canonical; the
``recurrence``/``bucket`` mirror columns must agree on every write — a
mismatch raises :class:`MirrorMismatch` at the seam, which is exactly the
class of bug the recovery scripts existed to clean up.

All mutations route through :func:`tasque.chains.spec.validate_spec`.
"""

from __future__ import annotations

import json
from typing import Any, Final

from sqlalchemy import select

from tasque.chains.spec import SpecError, validate_spec
from tasque.memory.db import get_session
from tasque.memory.entities import ChainTemplate, utc_now_iso


class _Unset:
    """Sentinel for ``update_chain_template(recurrence=...)`` so callers can
    distinguish 'not provided' from 'explicit None' (clear the cron)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET: Final[_Unset] = _Unset()


class MirrorMismatch(ValueError):
    """Raised when ``recurrence`` (or ``bucket``) on the row would disagree
    with the value inside ``plan_json``."""


def enforce_mirror(plan_dict: dict[str, Any], *, recurrence: str | None) -> None:
    """Reject any pairing where the mirror column disagrees with plan_json."""
    inside = plan_dict.get("recurrence")
    if inside != recurrence:
        raise MirrorMismatch(
            f"recurrence mismatch: column={recurrence!r} but plan_json.recurrence={inside!r}. "
            "ChainTemplate.plan_json is canonical; the row's recurrence column must match."
        )


def _serialize_template(row: ChainTemplate) -> dict[str, Any]:
    try:
        plan_dict = json.loads(row.plan_json)
    except json.JSONDecodeError:
        plan_dict = None
    return {
        "id": row.id,
        "chain_name": row.chain_name,
        "bucket": row.bucket,
        "recurrence": row.recurrence,
        "enabled": row.enabled,
        "last_fired_at": row.last_fired_at,
        "plan": plan_dict,
        "plan_json": row.plan_json,
        "seed_path": row.seed_path,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _existing_by_name(name: str) -> ChainTemplate | None:
    with get_session() as sess:
        stmt = select(ChainTemplate).where(ChainTemplate.chain_name == name)
        row = sess.execute(stmt).scalars().first()
        if row is not None:
            sess.expunge(row)
        return row


def create_chain_template(
    plan_dict: dict[str, Any],
    *,
    recurrence: str | None = None,
    seed_path: str | None = None,
) -> str:
    """Insert a new ``ChainTemplate`` and return its ``chain_name``.

    Validates the spec, enforces the mirror-column contract, and rejects
    duplicate names. The ``recurrence`` keyword and ``plan_dict["recurrence"]``
    must match — pass ``None`` for both if there is no schedule.
    """
    validate_spec(plan_dict)
    enforce_mirror(plan_dict, recurrence=recurrence)

    name = plan_dict["chain_name"]
    if _existing_by_name(name) is not None:
        raise ValueError(f"a chain template named {name!r} already exists")

    with get_session() as sess:
        row = ChainTemplate(
            chain_name=name,
            bucket=plan_dict.get("bucket"),
            recurrence=recurrence,
            enabled=True,
            plan_json=json.dumps(plan_dict),
            seed_path=seed_path,
        )
        sess.add(row)
        sess.flush()
    return name


def update_chain_template(
    name: str,
    *,
    plan_dict: dict[str, Any] | None = None,
    recurrence: str | None | _Unset = UNSET,
    enabled: bool | None = None,
) -> bool:
    """Update an existing ``ChainTemplate``. Returns True if the row existed.

    ``recurrence`` defaults to a sentinel; pass ``None`` to clear the cron,
    a string to replace it, or omit it to leave it untouched. If
    ``plan_dict`` is provided, it is re-validated and the mirror-column
    rule is enforced against the *new* recurrence (current row value if
    ``recurrence`` was not provided).
    """
    recurrence_provided = not isinstance(recurrence, _Unset)
    new_recurrence_value: str | None = None if not recurrence_provided else recurrence  # type: ignore[assignment]

    with get_session() as sess:
        stmt = select(ChainTemplate).where(ChainTemplate.chain_name == name)
        row = sess.execute(stmt).scalars().first()
        if row is None:
            return False

        if plan_dict is not None:
            validate_spec(plan_dict)
            effective_rec = new_recurrence_value if recurrence_provided else row.recurrence
            enforce_mirror(plan_dict, recurrence=effective_rec)
            row.plan_json = json.dumps(plan_dict)
            row.bucket = plan_dict.get("bucket")
        elif recurrence_provided:
            try:
                current = json.loads(row.plan_json)
            except json.JSONDecodeError as exc:
                raise SpecError(
                    f"row {name!r} has malformed plan_json; cannot rotate recurrence"
                ) from exc
            current["recurrence"] = new_recurrence_value
            validate_spec(current)
            row.plan_json = json.dumps(current)

        if recurrence_provided:
            row.recurrence = new_recurrence_value

        if enabled is not None:
            row.enabled = enabled

        row.updated_at = utc_now_iso()
        sess.flush()
    return True


def delete_chain_template(name: str) -> bool:
    """Hard-delete a ``ChainTemplate``. Any ``ChainRun`` rows that reference
    it have their ``template_id`` nulled out (history preserved).

    Returns True if the row existed.
    """
    from tasque.memory.entities import ChainRun

    with get_session() as sess:
        stmt = select(ChainTemplate).where(ChainTemplate.chain_name == name)
        row = sess.execute(stmt).scalars().first()
        if row is None:
            return False
        template_id = row.id
        runs_stmt = select(ChainRun).where(ChainRun.template_id == template_id)
        for run in sess.execute(runs_stmt).scalars().all():
            run.template_id = None
            run.updated_at = utc_now_iso()
        sess.delete(row)
        sess.flush()
    return True


def get_chain_template(name: str) -> dict[str, Any] | None:
    """Return one template as a dict, or None if missing."""
    row = _existing_by_name(name)
    if row is None:
        return None
    return _serialize_template(row)


def list_chain_templates(*, enabled_only: bool = False) -> list[dict[str, Any]]:
    """Return all chain templates, newest first."""
    with get_session() as sess:
        stmt = select(ChainTemplate)
        if enabled_only:
            stmt = stmt.where(ChainTemplate.enabled.is_(True))
        stmt = stmt.order_by(ChainTemplate.created_at.desc())
        rows = list(sess.execute(stmt).scalars().all())
        sess.expunge_all()
    return [_serialize_template(r) for r in rows]


__all__ = [
    "MirrorMismatch",
    "create_chain_template",
    "delete_chain_template",
    "enforce_mirror",
    "get_chain_template",
    "list_chain_templates",
    "update_chain_template",
]

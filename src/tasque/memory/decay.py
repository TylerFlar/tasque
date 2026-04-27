"""Nondurable-memory decay sweep.

What counts as "nondurable" — and how it decays:

- **Ephemeral Notes** (``durability="ephemeral"``) older than
  ``decay_notes_cutoff_days`` are archived. Behavioral and durable Notes
  are never decayed by age — those are the long-lived memory.
- **Expired Signals** (``expires_at`` is set and in the past) are
  archived. Signals without an ``expires_at`` are kept indefinitely;
  callers opt into decay by setting one.
- **Superseded Notes** (``superseded_by`` is set, ``updated_at`` older
  than ``decay_superseded_cutoff_days``) are archived. Once a chain of
  supersedes is established, the older revisions are dead weight in
  bucket queries.
- **Already-archived rows** can optionally be hard-deleted past
  ``decay_hard_delete_cutoff_days``. Default is to never hard-delete —
  archived rows stay around for forensic recall.

The sweep is idempotent: re-running immediately reports zero archives
because the rows it would have touched are already ``archived=True``.

Settings live in ``tasque.config.Settings`` (``decay_*`` knobs). The
APScheduler in ``tasque.jobs.scheduler`` runs ``sweep_nondurable_memory``
on a configurable interval (default daily). The CLI exposes the same
sweep via ``tasque memory prune``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select

from tasque.memory.db import get_session
from tasque.memory.entities import Attachment, Note, Signal


@dataclass(frozen=True)
class DecayReport:
    """Counts produced by a single sweep. ``dry_run=True`` means the
    counts describe what *would* have been changed."""

    archived_ephemeral_notes: int = 0
    archived_expired_signals: int = 0
    archived_superseded_notes: int = 0
    hard_deleted_notes: int = 0
    hard_deleted_signals: int = 0
    hard_deleted_attachments: int = 0
    dry_run: bool = False

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)

    @property
    def total_archived(self) -> int:
        return (
            self.archived_ephemeral_notes
            + self.archived_expired_signals
            + self.archived_superseded_notes
        )

    @property
    def total_hard_deleted(self) -> int:
        return (
            self.hard_deleted_notes
            + self.hard_deleted_signals
            + self.hard_deleted_attachments
        )


def _iso_days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sweep_nondurable_memory(
    *,
    notes_cutoff_days: int | None = None,
    superseded_cutoff_days: int | None = None,
    hard_delete_cutoff_days: int | None = -1,
    dry_run: bool = False,
) -> DecayReport:
    """Run one decay sweep over all nondurable memory.

    Args:
        notes_cutoff_days: Override the ephemeral-Note age cutoff. ``None``
            uses ``Settings.decay_notes_cutoff_days``.
        superseded_cutoff_days: Override the superseded-Note age cutoff.
            ``None`` uses ``Settings.decay_superseded_cutoff_days``.
        hard_delete_cutoff_days: Override the hard-delete cutoff for
            already-archived rows. The sentinel ``-1`` means "use
            settings"; ``None`` means "never hard-delete"; an int means
            "hard-delete archived rows older than this many days."
        dry_run: If True, count what would be archived/deleted but don't
            mutate anything.
    """
    # Resolve effective cutoffs from settings if not overridden. Done
    # lazily so test fixtures that mutate the cached Settings still work.
    from tasque.config import get_settings

    s = get_settings()
    eff_notes = notes_cutoff_days if notes_cutoff_days is not None else s.decay_notes_cutoff_days
    eff_super = (
        superseded_cutoff_days
        if superseded_cutoff_days is not None
        else s.decay_superseded_cutoff_days
    )
    eff_hard: int | None
    if hard_delete_cutoff_days == -1:
        eff_hard = s.decay_hard_delete_cutoff_days
    else:
        eff_hard = hard_delete_cutoff_days

    notes_cutoff_iso = _iso_days_ago(eff_notes)
    superseded_cutoff_iso = _iso_days_ago(eff_super)
    now_iso = _now_iso()
    hard_cutoff_iso = _iso_days_ago(eff_hard) if eff_hard is not None else None

    n_ephemeral = 0
    n_expired = 0
    n_superseded = 0
    n_del_notes = 0
    n_del_signals = 0
    n_del_attach = 0

    with get_session() as sess:
        # 1. Archive ephemeral Notes past their cutoff.
        ephemeral_stmt = (
            select(Note)
            .where(Note.durability == "ephemeral")
            .where(Note.archived.is_(False))
            .where(Note.created_at < notes_cutoff_iso)
        )
        for row in sess.execute(ephemeral_stmt).scalars().all():
            n_ephemeral += 1
            if not dry_run:
                row.archived = True
                row.updated_at = now_iso

        # 2. Archive Signals past their expires_at.
        expired_stmt = (
            select(Signal)
            .where(Signal.archived.is_(False))
            .where(Signal.expires_at.is_not(None))
            .where(Signal.expires_at < now_iso)
        )
        for row in sess.execute(expired_stmt).scalars().all():
            n_expired += 1
            if not dry_run:
                row.archived = True
                row.updated_at = now_iso

        # 3. Archive superseded Notes whose updated_at is older than the
        #    superseded cutoff. The updated_at moves forward when the
        #    supersede happens (see ``supersede_note``), so this is the
        #    right clock — not created_at.
        superseded_stmt = (
            select(Note)
            .where(Note.archived.is_(False))
            .where(Note.superseded_by.is_not(None))
            .where(Note.updated_at < superseded_cutoff_iso)
        )
        for row in sess.execute(superseded_stmt).scalars().all():
            n_superseded += 1
            if not dry_run:
                row.archived = True
                row.updated_at = now_iso

        # 4. Optional hard-delete of already-archived rows (Note,
        #    Signal, Attachment — the three types with an ``archived``
        #    flag). Skipped entirely when no cutoff is configured.
        if hard_cutoff_iso is not None:
            for cls, counter_set in (
                (Note, "n_del_notes"),
                (Signal, "n_del_signals"),
                (Attachment, "n_del_attach"),
            ):
                count_stmt = (
                    select(cls.id)  # type: ignore[attr-defined]
                    .where(cls.archived.is_(True))  # type: ignore[attr-defined]
                    .where(cls.updated_at < hard_cutoff_iso)  # type: ignore[attr-defined]
                )
                ids = [r for r in sess.execute(count_stmt).scalars().all()]
                count = len(ids)
                if counter_set == "n_del_notes":
                    n_del_notes = count
                elif counter_set == "n_del_signals":
                    n_del_signals = count
                else:
                    n_del_attach = count
                if count and not dry_run:
                    sess.execute(delete(cls).where(cls.id.in_(ids)))  # type: ignore[attr-defined]

        if dry_run:
            sess.rollback()

    return DecayReport(
        archived_ephemeral_notes=n_ephemeral,
        archived_expired_signals=n_expired,
        archived_superseded_notes=n_superseded,
        hard_deleted_notes=n_del_notes,
        hard_deleted_signals=n_del_signals,
        hard_deleted_attachments=n_del_attach,
        dry_run=dry_run,
    )


def prune_decayed_notes(cutoff_days: int = 30) -> int:
    """Backwards-compatible thin wrapper: archive ephemeral Notes only.

    Prefer ``sweep_nondurable_memory`` for new callers — it covers
    Signals and superseded chains too.
    """
    report = sweep_nondurable_memory(
        notes_cutoff_days=cutoff_days,
        # ~270 years — effectively disabled, and well under the
        # ``timedelta`` magnitude cap (999,999,999 days).
        superseded_cutoff_days=100_000,
        hard_delete_cutoff_days=None,
    )
    return report.archived_ephemeral_notes

"""Tests for ``tasque.memory.decay`` — the nondurable-memory sweep."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tasque.config import get_settings, reset_settings_cache
from tasque.memory.db import get_session
from tasque.memory.decay import (
    DecayReport,
    prune_decayed_notes,
    sweep_nondurable_memory,
)
from tasque.memory.entities import Attachment, Note, Signal
from tasque.memory.repo import supersede_note, write_entity


def _iso_days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _iso_hours_ahead(hours: int) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _backdate(entity_cls: type, ids_and_columns: dict[str, dict[str, str]]) -> None:
    """Reach into the DB to set timestamp columns directly. Tests need this
    because ORM defaults always pin created_at to ``now``."""
    with get_session() as sess:
        for entity_id, updates in ids_and_columns.items():
            obj = sess.get(entity_cls, entity_id)
            assert obj is not None, f"{entity_cls.__name__} {entity_id} not found"
            for col, value in updates.items():
                setattr(obj, col, value)


# ---------------------------------------------------------- ephemeral notes

def test_sweep_archives_old_ephemeral_notes() -> None:
    old = Note(content="old", durability="ephemeral", source="user")
    fresh = Note(content="fresh", durability="ephemeral", source="user")
    write_entity(old)
    write_entity(fresh)
    _backdate(Note, {old.id: {"created_at": _iso_days_ago(60)}})

    report = sweep_nondurable_memory()
    assert report.archived_ephemeral_notes == 1
    assert report.archived_expired_signals == 0
    assert report.archived_superseded_notes == 0
    assert report.dry_run is False

    with get_session() as sess:
        assert sess.get(Note, old.id).archived is True  # type: ignore[union-attr]
        assert sess.get(Note, fresh.id).archived is False  # type: ignore[union-attr]


def test_sweep_leaves_durable_and_behavioral_notes_alone() -> None:
    durable = Note(content="durable", durability="durable", source="user")
    behavioral = Note(content="behavioral", durability="behavioral", source="user")
    write_entity(durable)
    write_entity(behavioral)
    _backdate(
        Note,
        {
            durable.id: {"created_at": _iso_days_ago(365)},
            behavioral.id: {"created_at": _iso_days_ago(365)},
        },
    )

    report = sweep_nondurable_memory()
    assert report.archived_ephemeral_notes == 0
    with get_session() as sess:
        assert sess.get(Note, durable.id).archived is False  # type: ignore[union-attr]
        assert sess.get(Note, behavioral.id).archived is False  # type: ignore[union-attr]


# ---------------------------------------------------------- lifecycle notes

def test_sweep_archives_lifecycle_artifacts_by_ttl() -> None:
    artifact = Note(
        content="worker residue",
        bucket="relationships",
        durability="ephemeral",
        memory_kind="artifact",
        ttl_days=3,
        source="worker",
    )
    fact = Note(
        content="real memory",
        bucket="relationships",
        durability="durable",
        memory_kind="fact",
        source="user",
    )
    write_entity(artifact)
    write_entity(fact)
    _backdate(
        Note,
        {
            artifact.id: {"created_at": _iso_days_ago(4)},
            fact.id: {"created_at": _iso_days_ago(365)},
        },
    )

    report = sweep_nondurable_memory()
    assert report.archived_lifecycle_notes == 1

    with get_session() as sess:
        assert sess.get(Note, artifact.id).archived is True  # type: ignore[union-attr]
        assert sess.get(Note, fact.id).archived is False  # type: ignore[union-attr]


def test_sweep_collapses_duplicate_canonical_summaries() -> None:
    older = Note(
        content="old dating summary",
        bucket="relationships",
        durability="durable",
        memory_kind="summary",
        canonical_key="relationships:dating",
        source="coach",
    )
    newer = Note(
        content="new dating summary",
        bucket="relationships",
        durability="durable",
        memory_kind="summary",
        canonical_key="relationships:dating",
        source="coach",
    )
    other = Note(
        content="other summary",
        bucket="relationships",
        durability="durable",
        memory_kind="summary",
        canonical_key="relationships:friends",
        source="coach",
    )
    write_entity(older)
    write_entity(newer)
    write_entity(other)
    _backdate(Note, {older.id: {"updated_at": _iso_days_ago(2)}})

    report = sweep_nondurable_memory()
    assert report.archived_duplicate_summaries == 1

    with get_session() as sess:
        assert sess.get(Note, older.id).archived is True  # type: ignore[union-attr]
        assert sess.get(Note, newer.id).archived is False  # type: ignore[union-attr]
        assert sess.get(Note, other.id).archived is False  # type: ignore[union-attr]


# ---------------------------------------------------------- expired signals

def test_sweep_archives_expired_signals() -> None:
    expired = Signal(
        from_bucket="health",
        to_bucket="all",
        kind="fyi",
        urgency="whenever",
        summary="stale",
        expires_at=_iso_days_ago(1),
    )
    not_yet_expired = Signal(
        from_bucket="health",
        to_bucket="all",
        kind="fyi",
        urgency="whenever",
        summary="future",
        expires_at=_iso_hours_ahead(24),
    )
    no_expiry = Signal(
        from_bucket="health",
        to_bucket="all",
        kind="fyi",
        urgency="whenever",
        summary="evergreen",
    )
    write_entity(expired)
    write_entity(not_yet_expired)
    write_entity(no_expiry)

    report = sweep_nondurable_memory()
    assert report.archived_expired_signals == 1

    with get_session() as sess:
        assert sess.get(Signal, expired.id).archived is True  # type: ignore[union-attr]
        assert sess.get(Signal, not_yet_expired.id).archived is False  # type: ignore[union-attr]
        assert sess.get(Signal, no_expiry.id).archived is False  # type: ignore[union-attr]


# -------------------------------------------------------- superseded chains

def test_sweep_archives_old_superseded_notes() -> None:
    original = Note(
        content="v1", bucket="creative", durability="durable", source="user"
    )
    write_entity(original)
    v2 = supersede_note(original.id, "v2", durability="durable")
    # supersede_note bumped original.updated_at to now; backdate it past
    # the superseded cutoff (default 14 days) so it's eligible.
    _backdate(Note, {original.id: {"updated_at": _iso_days_ago(30)}})

    report = sweep_nondurable_memory()
    assert report.archived_superseded_notes == 1
    assert report.archived_ephemeral_notes == 0  # durable, so not ephemeral-decayed

    with get_session() as sess:
        assert sess.get(Note, original.id).archived is True  # type: ignore[union-attr]
        assert sess.get(Note, v2.id).archived is False  # type: ignore[union-attr]


def test_sweep_leaves_recently_superseded_notes_alone() -> None:
    original = Note(
        content="v1", bucket="creative", durability="durable", source="user"
    )
    write_entity(original)
    supersede_note(original.id, "v2", durability="durable")
    # updated_at is now; default cutoff is 14 days ago.

    report = sweep_nondurable_memory()
    assert report.archived_superseded_notes == 0


# ------------------------------------------------------------ hard delete

def test_sweep_hard_deletes_archived_rows_by_default() -> None:
    archived = Note(
        content="x", durability="durable", source="user", archived=True
    )
    write_entity(archived)
    _backdate(Note, {archived.id: {"updated_at": _iso_days_ago(365)}})

    report = sweep_nondurable_memory()
    assert report.hard_deleted_notes == 1
    with get_session() as sess:
        assert sess.get(Note, archived.id) is None


def test_sweep_hard_deletes_when_cutoff_set() -> None:
    old_archived = Note(
        content="old", durability="durable", source="user", archived=True
    )
    young_archived = Note(
        content="young", durability="durable", source="user", archived=True
    )
    archived_signal = Signal(
        from_bucket="health",
        to_bucket="all",
        kind="fyi",
        urgency="whenever",
        summary="gone",
        archived=True,
    )
    archived_attach = Attachment(
        filename="x.txt",
        content_type="text/plain",
        size_bytes=1,
        source="user",
        local_path="/tmp/x.txt",
        sha256="deadbeef",
        archived=True,
    )
    write_entity(old_archived)
    write_entity(young_archived)
    write_entity(archived_signal)
    write_entity(archived_attach)
    # Backdate updated_at past the hard-delete cutoff for some.
    _backdate(
        Note,
        {
            old_archived.id: {"updated_at": _iso_days_ago(120)},
            young_archived.id: {"updated_at": _iso_days_ago(10)},
        },
    )
    _backdate(Signal, {archived_signal.id: {"updated_at": _iso_days_ago(120)}})
    _backdate(Attachment, {archived_attach.id: {"updated_at": _iso_days_ago(120)}})

    report = sweep_nondurable_memory(hard_delete_cutoff_days=90)
    assert report.hard_deleted_notes == 1
    assert report.hard_deleted_signals == 1
    assert report.hard_deleted_attachments == 1

    with get_session() as sess:
        assert sess.get(Note, old_archived.id) is None
        assert sess.get(Note, young_archived.id) is not None
        assert sess.get(Signal, archived_signal.id) is None
        assert sess.get(Attachment, archived_attach.id) is None


def test_sweep_hard_delete_can_be_disabled() -> None:
    archived = Note(
        content="x", durability="durable", source="user", archived=True
    )
    write_entity(archived)
    _backdate(Note, {archived.id: {"updated_at": _iso_days_ago(10**4)}})

    report = sweep_nondurable_memory(hard_delete_cutoff_days=None)
    assert report.hard_deleted_notes == 0
    with get_session() as sess:
        assert sess.get(Note, archived.id) is not None


# --------------------------------------------------------------- dry-run

def test_dry_run_reports_counts_without_mutating() -> None:
    old = Note(content="old", durability="ephemeral", source="user")
    write_entity(old)
    _backdate(Note, {old.id: {"created_at": _iso_days_ago(60)}})

    report = sweep_nondurable_memory(dry_run=True)
    assert report.dry_run is True
    assert report.archived_ephemeral_notes == 1

    with get_session() as sess:
        # Mutation rolled back — row is still un-archived.
        assert sess.get(Note, old.id).archived is False  # type: ignore[union-attr]


# ------------------------------------------------------------ idempotency

def test_sweep_is_idempotent() -> None:
    old = Note(content="old", durability="ephemeral", source="user")
    write_entity(old)
    _backdate(Note, {old.id: {"created_at": _iso_days_ago(60)}})

    first = sweep_nondurable_memory()
    second = sweep_nondurable_memory()
    assert first.archived_ephemeral_notes == 1
    assert second.archived_ephemeral_notes == 0


# ------------------------------------------------------- cutoff overrides

def test_explicit_cutoffs_override_settings() -> None:
    young = Note(content="young", durability="ephemeral", source="user")
    write_entity(young)
    _backdate(Note, {young.id: {"created_at": _iso_days_ago(5)}})

    # Default cutoff (30 days) → no archive.
    report = sweep_nondurable_memory()
    assert report.archived_ephemeral_notes == 0

    # Override to 1 day → it gets archived.
    report = sweep_nondurable_memory(notes_cutoff_days=1)
    assert report.archived_ephemeral_notes == 1


# -------------------------------------------- DecayReport conveniences

def test_decay_report_totals() -> None:
    r = DecayReport(
        archived_ephemeral_notes=3,
        archived_expired_signals=2,
        archived_superseded_notes=1,
        hard_deleted_notes=4,
        hard_deleted_signals=5,
        hard_deleted_attachments=6,
    )
    assert r.total_archived == 6
    assert r.total_hard_deleted == 15
    d = r.to_dict()
    assert d["archived_ephemeral_notes"] == 3
    assert d["dry_run"] is False


# ------------------------------------------------- legacy wrapper kept

def test_prune_decayed_notes_wrapper() -> None:
    old = Note(content="old", durability="ephemeral", source="user")
    write_entity(old)
    _backdate(Note, {old.id: {"created_at": _iso_days_ago(60)}})

    n = prune_decayed_notes(cutoff_days=30)
    assert n == 1


# --------------------------------------- settings default visible to sweep

@pytest.fixture
def short_decay_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the settings cutoffs low so a 5-day-old ephemeral note decays."""
    monkeypatch.setenv("DECAY_NOTES_CUTOFF_DAYS", "1")
    monkeypatch.setenv("DECAY_SUPERSEDED_CUTOFF_DAYS", "1")
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_sweep_honors_settings_cutoff(short_decay_settings: None) -> None:
    s = get_settings()
    assert s.decay_notes_cutoff_days == 1

    young = Note(content="young", durability="ephemeral", source="user")
    write_entity(young)
    _backdate(Note, {young.id: {"created_at": _iso_days_ago(5)}})

    report = sweep_nondurable_memory()
    assert report.archived_ephemeral_notes == 1

"""Tests for the typed repo facade."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tasque.memory.entities import (
    Aim,
    Attachment,
    FailedJob,
    Note,
    QueuedJob,
    Signal,
    WorkerPattern,
)
from tasque.memory.repo import (
    archive,
    bump_job_heartbeat,
    delete_entity,
    get_entity,
    prune_decayed_notes,
    query_bucket,
    query_pending_jobs,
    query_signals_for,
    query_unresolved_failures,
    search_worker_patterns,
    supersede_note,
    update_entity_status,
    upsert_worker_pattern,
    write_entity,
)


def _iso_days_ago(days: int) -> str:
    dt = datetime.now(UTC) - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def test_write_get_delete_note() -> None:
    n = Note(content="hello", bucket="health", durability="durable", source="user")
    written = write_entity(n)
    assert written.id

    fetched = get_entity(written.id)
    assert fetched is not None
    assert isinstance(fetched, Note)
    assert fetched.content == "hello"

    assert delete_entity(written.id) is True
    assert get_entity(written.id) is None


def test_update_entity_status_queued_job() -> None:
    job = QueuedJob(
        kind="worker",
        bucket="career",
        directive="do thing",
        reason="because",
        queued_by="test",
    )
    write_entity(job)
    assert update_entity_status(job.id, "completed") is True

    fetched = get_entity(job.id)
    assert isinstance(fetched, QueuedJob)
    assert fetched.status == "completed"


def test_update_entity_status_failed_job() -> None:
    fj = FailedJob(
        job_id="j1",
        agent_kind="worker",
        bucket="finance",
        failure_timestamp=_iso_days_ago(0),
        error_type="X",
        error_message="boom",
    )
    write_entity(fj)
    assert update_entity_status(fj.id, "resolved") is True

    fetched = get_entity(fj.id)
    assert isinstance(fetched, FailedJob)
    assert fetched.resolved is True


def test_archive_note() -> None:
    n = Note(content="x", bucket="home", durability="ephemeral", source="user")
    write_entity(n)
    assert archive(n.id) is True

    fetched = get_entity(n.id)
    assert isinstance(fetched, Note)
    assert fetched.archived is True


def test_query_bucket_filters_archived() -> None:
    a = Note(content="a", bucket="career", durability="durable", source="user")
    b = Note(content="b", bucket="career", durability="durable", source="user")
    write_entity(a)
    write_entity(b)
    archive(a.id)

    visible = query_bucket("career", types=[Note])
    assert len(visible) == 1
    assert visible[0].id == b.id

    hidden = query_bucket("career", types=[Note], archived=True)
    assert len(hidden) == 1
    assert hidden[0].id == a.id


def test_query_bucket_default_types_excludes_signals() -> None:
    Note_inst = Note(content="n", bucket="home", durability="durable", source="user")
    sig = Signal(
        from_bucket="home",
        to_bucket="home",
        kind="fyi",
        urgency="whenever",
        summary="hi",
    )
    write_entity(Note_inst)
    write_entity(sig)

    rows = query_bucket("home")
    assert all(not isinstance(r, Signal) for r in rows)


def test_query_signals_for_includes_all() -> None:
    direct = Signal(
        from_bucket="health",
        to_bucket="career",
        kind="request",
        urgency="now",
        summary="d",
    )
    broadcast = Signal(
        from_bucket="health",
        to_bucket="all",
        kind="fyi",
        urgency="whenever",
        summary="b",
    )
    other = Signal(
        from_bucket="health",
        to_bucket="finance",
        kind="fyi",
        urgency="whenever",
        summary="o",
    )
    write_entity(direct)
    write_entity(broadcast)
    write_entity(other)

    rows = query_signals_for("career")
    summaries = sorted(r.summary for r in rows)
    assert summaries == ["b", "d"]


def test_query_pending_jobs_respects_before_iso() -> None:
    now_job = QueuedJob(
        kind="worker", directive="now", reason="", queued_by="t", fire_at="now"
    )
    past_job = QueuedJob(
        kind="worker",
        directive="past",
        reason="",
        queued_by="t",
        fire_at="2026-04-01T00:00:00.000000Z",
    )
    future_job = QueuedJob(
        kind="worker",
        directive="future",
        reason="",
        queued_by="t",
        fire_at="2099-01-01T00:00:00.000000Z",
    )
    write_entity(now_job)
    write_entity(past_job)
    write_entity(future_job)

    cutoff = "2026-04-25T00:00:00.000000Z"
    rows = query_pending_jobs(before_iso=cutoff)
    directives = sorted(r.directive for r in rows)
    assert directives == ["now", "past"]


def test_query_unresolved_failures() -> None:
    a = FailedJob(
        job_id="j",
        agent_kind="w",
        failure_timestamp=_iso_days_ago(1),
        error_type="X",
        error_message="m",
    )
    b = FailedJob(
        job_id="j",
        agent_kind="w",
        failure_timestamp=_iso_days_ago(2),
        error_type="X",
        error_message="m",
    )
    write_entity(a)
    write_entity(b)
    update_entity_status(a.id, "resolved")

    rows = query_unresolved_failures(limit=10)
    assert len(rows) == 1
    assert rows[0].id == b.id


def test_bump_job_heartbeat() -> None:
    job = QueuedJob(kind="worker", directive="d", reason="", queued_by="t")
    write_entity(job)
    iso = "2026-04-25T12:00:00.000000Z"
    assert bump_job_heartbeat(job.id, iso) is True

    fetched = get_entity(job.id)
    assert isinstance(fetched, QueuedJob)
    assert fetched.last_heartbeat == iso


def test_supersede_note_chain() -> None:
    original = Note(
        content="v1",
        bucket="creative",
        durability="durable",
        source="user",
        meta={"k": 1},
    )
    write_entity(original)

    v2 = supersede_note(original.id, "v2", durability="durable")
    assert v2.bucket == "creative"
    assert v2.source == "user"
    assert v2.meta == {"k": 1}

    refreshed = get_entity(original.id)
    assert isinstance(refreshed, Note)
    assert refreshed.superseded_by == v2.id


def test_prune_decayed_notes_archives_only_old_ephemeral() -> None:
    old = Note(content="old", durability="ephemeral", source="user")
    fresh = Note(content="fresh", durability="ephemeral", source="user")
    durable_old = Note(content="durable-old", durability="durable", source="user")
    write_entity(old)
    write_entity(fresh)
    write_entity(durable_old)
    # Backdate the "old" notes by reaching directly through a session.
    from tasque.memory.db import get_session

    backdate = _iso_days_ago(60)
    with get_session() as sess:
        for nid in (old.id, durable_old.id):
            obj = sess.get(Note, nid)
            assert obj is not None
            obj.created_at = backdate

    archived_count = prune_decayed_notes(cutoff_days=30)
    assert archived_count == 1

    o = get_entity(old.id)
    f = get_entity(fresh.id)
    d = get_entity(durable_old.id)
    assert isinstance(o, Note) and o.archived is True
    assert isinstance(f, Note) and f.archived is False
    assert isinstance(d, Note) and d.archived is False


def test_archive_attachment() -> None:
    att = Attachment(
        filename="a.txt",
        content_type="text/plain",
        size_bytes=1,
        source="user",
        local_path="/tmp/a.txt",
        sha256="abc",
    )
    write_entity(att)
    assert archive(att.id) is True
    fetched = get_entity(att.id)
    assert isinstance(fetched, Attachment)
    assert fetched.archived is True


def test_worker_pattern_search_and_upsert() -> None:
    first = upsert_worker_pattern(
        bucket="finance",
        source_kind="worker",
        key="worker:rebalance-review",
        content="Directive: review rebalance gates\nSummary: Checked target weights.",
        tags=["rebalance", "weights"],
        meta={"produces_keys": ["proposal_id"]},
    )
    second = upsert_worker_pattern(
        bucket="finance",
        source_kind="worker",
        key="worker:rebalance-review",
        content="Directive: review rebalance gates\nSummary: Checked newer target weights.",
        tags=["rebalance", "weights"],
        meta={"produces_keys": ["proposal_id"]},
    )

    assert second.id == first.id
    assert second.success_count == 2

    rows = search_worker_patterns(
        query="rebalance target weights",
        bucket="finance",
        limit=5,
        touch=False,
    )
    assert [r.id for r in rows] == [first.id]
    assert rows[0].content.endswith("newer target weights.")

    assert search_worker_patterns(
        query="rebalance target weights",
        bucket="career",
        limit=5,
        touch=False,
    ) == []

    assert archive(first.id) is True
    fetched = get_entity(first.id)
    assert isinstance(fetched, WorkerPattern)
    assert fetched.archived is True
    assert search_worker_patterns(
        query="rebalance target weights",
        bucket="finance",
        limit=5,
        touch=False,
    ) == []


def test_aim_status_update() -> None:
    a = Aim(
        title="learn rust",
        bucket="education",
        scope="long_term",
        description="",
        source="user",
    )
    write_entity(a)
    assert update_entity_status(a.id, "completed") is True
    refreshed = get_entity(a.id)
    assert isinstance(refreshed, Aim)
    assert refreshed.status == "completed"

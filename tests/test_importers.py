"""Tests for JSONL and markdown importers / JSONL exporter round-trip."""

from __future__ import annotations

import json
from pathlib import Path

from tasque.memory.entities import Note
from tasque.memory.exporters import export_jsonl
from tasque.memory.importers import import_jsonl, import_markdown_dir
from tasque.memory.repo import get_entity

FIXTURES = Path(__file__).parent / "fixtures"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    """Demo fixture imports cleanly and exports back to a JSONL with the same rows."""
    report = import_jsonl(FIXTURES / "demo.jsonl")
    assert report["errors"] == 0
    assert report["skipped"] == 0
    assert sum(report["counts"].values()) == 8

    out = tmp_path / "out.jsonl"
    export_report = export_jsonl(out)
    assert sum(export_report["counts"].values()) == 8

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 8
    types = sorted(d["type"] for d in lines)
    assert types == sorted(
        [
            "Note",
            "Aim",
            "Signal",
            "QueuedJob",
            "FailedJob",
            "ChainTemplate",
            "ChainRun",
            "Attachment",
        ]
    )

    note_row = next(d for d in lines if d["type"] == "Note")
    assert "metadata" in note_row
    assert note_row["metadata"] == {"tag": "reminder"}
    assert "meta" not in note_row


def test_jsonl_import_is_idempotent_on_repeat() -> None:
    import_jsonl(FIXTURES / "demo.jsonl")
    first = get_entity("00000000000000000000000000000001")
    assert isinstance(first, Note)
    first_updated = first.updated_at

    # Re-import: same ids → merge, no duplication.
    import_jsonl(FIXTURES / "demo.jsonl")
    second = get_entity("00000000000000000000000000000001")
    assert isinstance(second, Note)
    assert second.id == first.id
    # updated_at refreshes on merge because we re-set fields.
    assert second.updated_at >= first_updated


def test_jsonl_import_updates_existing_row(tmp_path: Path) -> None:
    src = tmp_path / "data.jsonl"
    src.write_text(
        json.dumps(
            {
                "type": "Note",
                "id": "abc123",
                "content": "first",
                "durability": "durable",
                "source": "user",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import_jsonl(src)
    n1 = get_entity("abc123")
    assert isinstance(n1, Note)
    assert n1.content == "first"

    src.write_text(
        json.dumps(
            {
                "type": "Note",
                "id": "abc123",
                "content": "second",
                "durability": "durable",
                "source": "user",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import_jsonl(src)
    n2 = get_entity("abc123")
    assert isinstance(n2, Note)
    assert n2.content == "second"


def test_jsonl_import_skips_unknown_types(tmp_path: Path) -> None:
    src = tmp_path / "data.jsonl"
    src.write_text(
        '{"type": "Mystery", "foo": 1}\n'
        '{"type": "Note", "id": "n1", "content": "ok", "durability": "durable", "source": "user"}\n',
        encoding="utf-8",
    )
    report = import_jsonl(src)
    assert report["skipped"] == 1
    assert report["errors"] == 0
    assert report["counts"].get("Note") == 1


def test_jsonl_import_recovers_from_malformed_lines(tmp_path: Path) -> None:
    src = tmp_path / "data.jsonl"
    src.write_text(
        "this is not json\n"
        '{"type": "Note", "id": "n1", "content": "ok", "durability": "durable", "source": "user"}\n'
        "}{ also bad\n",
        encoding="utf-8",
    )
    report = import_jsonl(src)
    assert report["errors"] == 2
    assert report["counts"].get("Note") == 1
    assert get_entity("n1") is not None


def test_markdown_import_uses_bucket_subdir(tmp_path: Path) -> None:
    (tmp_path / "health").mkdir()
    (tmp_path / "health" / "habits.md").write_text("drink water", encoding="utf-8")
    (tmp_path / "stray.md").write_text("no bucket here", encoding="utf-8")

    report = import_markdown_dir(tmp_path)
    assert report["errors"] == 0
    assert report["counts"]["Note"] == 2

    from sqlalchemy import select

    from tasque.memory.db import get_session

    with get_session() as sess:
        rows = list(sess.execute(select(Note)).scalars().all())

    by_path = {n.meta["path"]: n for n in rows}
    assert by_path["health/habits.md"].bucket == "health"
    assert by_path["stray.md"].bucket is None


def test_markdown_import_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "creative").mkdir()
    (tmp_path / "creative" / "ideas.md").write_text("v1", encoding="utf-8")

    report1 = import_markdown_dir(tmp_path)
    assert report1["counts"]["Note"] == 1

    # Modify content; re-import should update, not duplicate.
    (tmp_path / "creative" / "ideas.md").write_text("v2", encoding="utf-8")
    report2 = import_markdown_dir(tmp_path)
    assert report2["counts"]["Note"] == 1

    from sqlalchemy import select

    from tasque.memory.db import get_session

    with get_session() as sess:
        rows = list(sess.execute(select(Note)).scalars().all())
    assert len(rows) == 1
    assert rows[0].content == "v2"

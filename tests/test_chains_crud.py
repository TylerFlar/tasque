"""Tests for ``tasque.chains.crud`` and the YAML registry round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tasque.chains.crud import (
    MirrorMismatch,
    create_chain_template,
    delete_chain_template,
    get_chain_template,
    list_chain_templates,
    update_chain_template,
)
from tasque.chains.registry import (
    export_template_to_yaml,
    reload_templates,
)


def _spec(**overrides: object) -> dict[str, object]:
    s: dict[str, object] = {
        "chain_name": "demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "large",
        "plan": [
            {"id": "a", "kind": "worker", "directive": "do A", "tier": "small"},
            {
                "id": "b",
                "kind": "approval",
                "directive": "approve",
                "depends_on": ["a"],
                "consumes": ["a"],
            },
        ],
    }
    s.update(overrides)
    return s


# ----------------------------------------------------------------- create

def test_create_chain_template_persists_row() -> None:
    name = create_chain_template(_spec())
    assert name == "demo"
    row = get_chain_template("demo")
    assert row is not None
    assert row["bucket"] == "personal"
    assert row["recurrence"] is None
    assert row["enabled"] is True


def test_create_chain_template_can_start_disabled() -> None:
    name = create_chain_template(_spec(), enabled=False)
    assert name == "demo"
    row = get_chain_template("demo")
    assert row is not None
    assert row["enabled"] is False


def test_create_chain_template_rejects_mirror_mismatch_for_recurrence() -> None:
    spec = _spec(recurrence="0 9 * * MON-FRI")
    with pytest.raises(MirrorMismatch):
        # Inside-spec recurrence is set, but column kwarg is None — mismatch.
        create_chain_template(spec, recurrence=None)


def test_create_chain_template_accepts_matching_recurrence() -> None:
    spec = _spec(recurrence="0 9 * * MON-FRI")
    name = create_chain_template(spec, recurrence="0 9 * * MON-FRI")
    assert name == "demo"
    row = get_chain_template("demo")
    assert row is not None
    assert row["recurrence"] == "0 9 * * MON-FRI"


def test_create_chain_template_rejects_duplicate_name() -> None:
    create_chain_template(_spec())
    with pytest.raises(ValueError, match="already exists"):
        create_chain_template(_spec())


# ----------------------------------------------------------------- update

def test_update_chain_template_replaces_plan() -> None:
    create_chain_template(_spec())
    new_spec = _spec(plan=[
        {"id": "x", "kind": "worker", "directive": "new", "tier": "small"},
    ])
    assert update_chain_template("demo", plan_dict=new_spec) is True
    row = get_chain_template("demo")
    assert row is not None
    assert row["plan"]["plan"][0]["id"] == "x"


def test_update_chain_template_rotates_recurrence() -> None:
    create_chain_template(_spec())
    spec_with_cron = _spec(recurrence="0 9 * * MON-FRI")
    assert update_chain_template(
        "demo", plan_dict=spec_with_cron, recurrence="0 9 * * MON-FRI"
    ) is True
    row = get_chain_template("demo")
    assert row is not None
    assert row["recurrence"] == "0 9 * * MON-FRI"


def test_update_chain_template_with_recurrence_only_keeps_mirror_in_sync() -> None:
    create_chain_template(_spec())
    # Update only the cron — the function rewrites plan_json.recurrence
    # to keep the mirror in sync.
    assert update_chain_template("demo", recurrence="0 9 * * SUN") is True
    row = get_chain_template("demo")
    assert row is not None
    assert row["recurrence"] == "0 9 * * SUN"
    assert row["plan"]["recurrence"] == "0 9 * * SUN"


def test_update_chain_template_rejects_plan_with_mismatched_recurrence() -> None:
    create_chain_template(_spec())
    bad_spec = _spec(recurrence="0 9 * * MON-FRI")  # implies cron
    with pytest.raises(MirrorMismatch):
        # We don't pass recurrence, so the row's None is used — mismatch.
        update_chain_template("demo", plan_dict=bad_spec)


def test_update_chain_template_disable_flag() -> None:
    create_chain_template(_spec())
    assert update_chain_template("demo", enabled=False) is True
    row = get_chain_template("demo")
    assert row is not None
    assert row["enabled"] is False


def test_update_unknown_returns_false() -> None:
    assert update_chain_template("ghost", enabled=False) is False


# ----------------------------------------------------------------- delete

def test_delete_chain_template_returns_true_then_false() -> None:
    create_chain_template(_spec())
    assert delete_chain_template("demo") is True
    assert delete_chain_template("demo") is False
    assert get_chain_template("demo") is None


def test_list_chain_templates_returns_rows() -> None:
    create_chain_template(_spec())
    rows = list_chain_templates()
    assert len(rows) == 1
    assert rows[0]["chain_name"] == "demo"


# ----------------------------------------------------------------- registry

def test_reload_templates_creates_rows_from_yaml(tmp_path: Path) -> None:
    spec = _spec()
    (tmp_path / "demo.yaml").write_text(yaml.safe_dump(spec))
    report = reload_templates(templates_dir=tmp_path)
    assert "demo" in report["added"]
    row = get_chain_template("demo")
    assert row is not None
    assert row["seed_path"] is not None


def test_reload_templates_skips_llm_edited_row(tmp_path: Path) -> None:
    spec = _spec()
    (tmp_path / "demo.yaml").write_text(yaml.safe_dump(spec))
    reload_templates(templates_dir=tmp_path)

    # Simulate an LLM-side edit through the CRUD path.
    new_spec = _spec(plan=[
        {"id": "x", "kind": "worker", "directive": "llm-edited", "tier": "small"},
    ])
    update_chain_template("demo", plan_dict=new_spec)

    # Re-running reload should detect the divergence and skip.
    report = reload_templates(templates_dir=tmp_path)
    assert "demo" in report["skipped"]
    row = get_chain_template("demo")
    assert row is not None
    assert row["plan"]["plan"][0]["id"] == "x"  # not clobbered


def test_export_template_to_yaml_round_trip(tmp_path: Path) -> None:
    create_chain_template(_spec())
    out = export_template_to_yaml("demo", tmp_path / "demo.yaml")
    assert out.exists()
    parsed = yaml.safe_load(out.read_text())
    assert parsed["chain_name"] == "demo"
    assert parsed["plan"][0]["id"] == "a"

    # After export, the row's seed_plan_hash matches — a reload should
    # treat it as freshly seeded.
    report = reload_templates(templates_dir=tmp_path)
    assert "demo" in report["updated"]


def test_reload_rejects_pure_numeric_dow_with_clear_error(tmp_path: Path) -> None:
    spec = _spec(recurrence="0 9 * * 1-5")
    (tmp_path / "demo.yaml").write_text(yaml.safe_dump(spec))
    report = reload_templates(templates_dir=tmp_path)
    assert report["added"] == []
    assert len(report["errors"]) == 1
    err_msg = report["errors"][0]["error"]
    assert "day-of-week" in err_msg


def test_reload_rejects_filename_chain_name_mismatch(tmp_path: Path) -> None:
    spec = _spec(chain_name="other-name")
    (tmp_path / "demo.yaml").write_text(yaml.safe_dump(spec))
    report = reload_templates(templates_dir=tmp_path)
    assert report["added"] == []
    assert len(report["errors"]) == 1
    assert "filename stem" in report["errors"][0]["error"]

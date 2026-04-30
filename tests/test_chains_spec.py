"""Tests for ``tasque.chains.spec.validate_spec``."""

from __future__ import annotations

import pytest

from tasque.chains.spec import SpecError, validate_spec


def _base_spec(**overrides: object) -> dict[str, object]:
    spec: dict[str, object] = {
        "chain_name": "demo",
        "bucket": "personal",
        "recurrence": None,
        "planner_tier": "opus",
        "plan": [
            {"id": "a", "kind": "worker", "directive": "do A", "tier": "haiku"},
            {
                "id": "b",
                "kind": "worker",
                "directive": "do B",
                "depends_on": ["a"],
                "consumes": ["a"],
                "tier": "haiku",
            },
        ],
    }
    spec.update(overrides)
    return spec


def test_valid_spec_returns_plan_with_defaults_filled() -> None:
    spec = _base_spec()
    plan = validate_spec(spec)
    assert [n["id"] for n in plan] == ["a", "b"]
    assert plan[0]["status"] == "pending"
    assert plan[0]["origin"] == "spec"
    assert plan[0]["on_failure"] == "halt"
    assert plan[0]["depends_on"] == []
    assert plan[0]["consumes"] == []
    assert plan[0]["fan_out_on"] is None
    assert plan[0]["tier"] == "haiku"


def test_unknown_top_level_key_rejected() -> None:
    spec = _base_spec(extra_field=True)
    with pytest.raises(SpecError, match="unknown top-level keys"):
        validate_spec(spec)


def test_vars_accepted_when_object() -> None:
    spec = _base_spec(vars={"force": True, "bucket_id": "bkt-1"})
    plan = validate_spec(spec)
    assert [n["id"] for n in plan] == ["a", "b"]


def test_vars_accepted_when_omitted() -> None:
    spec = _base_spec()
    assert "vars" not in spec
    plan = validate_spec(spec)
    assert [n["id"] for n in plan] == ["a", "b"]


def test_vars_accepted_when_null() -> None:
    spec = _base_spec(vars=None)
    plan = validate_spec(spec)
    assert [n["id"] for n in plan] == ["a", "b"]


def test_vars_rejected_when_not_object() -> None:
    spec = _base_spec(vars=["force"])
    with pytest.raises(SpecError, match="'vars' must be an object"):
        validate_spec(spec)


def test_unknown_node_key_rejected() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku", "typo_field": 1},
    ])
    with pytest.raises(SpecError, match="unknown keys"):
        validate_spec(spec)


def test_invalid_kind_rejected() -> None:
    spec = _base_spec(plan=[{"id": "a", "kind": "magic", "directive": "x"}])
    with pytest.raises(SpecError, match="invalid kind"):
        validate_spec(spec)


def test_consumes_must_be_subset_of_depends_on() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {
            "id": "b",
            "kind": "worker",
            "directive": "y",
            "depends_on": ["a"],
            "consumes": ["a", "ghost"],
            "tier": "haiku",
        },
    ])
    with pytest.raises(SpecError, match="consumes"):
        validate_spec(spec)


def test_fan_out_on_only_allowed_on_workers() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {
            "id": "b",
            "kind": "approval",
            "directive": "y",
            "depends_on": ["a"],
            "consumes": ["a"],
            "fan_out_on": "items",
        },
    ])
    with pytest.raises(SpecError, match="fan_out_on"):
        validate_spec(spec)


def test_dependency_cycle_detected() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "depends_on": ["b"], "tier": "haiku"},
        {"id": "b", "kind": "worker", "directive": "y", "depends_on": ["a"], "tier": "haiku"},
    ])
    with pytest.raises(SpecError, match="cycle"):
        validate_spec(spec)


def test_unknown_dep_id_rejected() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "depends_on": ["nope"], "tier": "haiku"},
    ])
    with pytest.raises(SpecError, match="unknown id"):
        validate_spec(spec)


def test_duplicate_step_ids_rejected() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {"id": "a", "kind": "worker", "directive": "y", "tier": "haiku"},
    ])
    with pytest.raises(SpecError, match="duplicate"):
        validate_spec(spec)


def test_pure_numeric_dow_recurrence_rejected() -> None:
    spec = _base_spec(recurrence="0 9 * * 1-5")
    with pytest.raises(SpecError, match="day-of-week"):
        validate_spec(spec)


def test_alias_dow_recurrence_accepted() -> None:
    spec = _base_spec(recurrence="0 9 * * MON-FRI")
    plan = validate_spec(spec)
    assert plan[0]["id"] == "a"


def test_missing_directive_rejected() -> None:
    spec = _base_spec(plan=[{"id": "a", "kind": "worker", "tier": "haiku"}])
    with pytest.raises(SpecError, match="directive"):
        validate_spec(spec)


def test_empty_plan_rejected() -> None:
    spec = _base_spec(plan=[])
    with pytest.raises(SpecError, match="non-empty"):
        validate_spec(spec)


def test_missing_chain_name_rejected() -> None:
    spec = _base_spec()
    del spec["chain_name"]
    with pytest.raises(SpecError, match="chain_name"):
        validate_spec(spec)


# ------------------------------------------------------------- tier validation


def test_missing_planner_tier_rejected() -> None:
    spec = _base_spec()
    del spec["planner_tier"]
    with pytest.raises(SpecError, match="planner_tier"):
        validate_spec(spec)


def test_invalid_planner_tier_rejected() -> None:
    spec = _base_spec(planner_tier="gpt4")
    with pytest.raises(SpecError, match="planner_tier"):
        validate_spec(spec)


def test_worker_step_missing_tier_rejected() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x"},
    ])
    with pytest.raises(SpecError, match="tier"):
        validate_spec(spec)


def test_worker_step_invalid_tier_rejected() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "gpt4"},
    ])
    with pytest.raises(SpecError, match="tier"):
        validate_spec(spec)


# ----------------------------------------------------- fan_out_concurrency

def test_fan_out_concurrency_defaults_to_none_when_omitted() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {
            "id": "b",
            "kind": "worker",
            "directive": "y",
            "depends_on": ["a"],
            "consumes": ["a"],
            "fan_out_on": "items",
            "tier": "haiku",
        },
    ])
    plan = validate_spec(spec)
    fanout = next(n for n in plan if n["id"] == "b")
    assert fanout["fan_out_concurrency"] is None


def test_fan_out_concurrency_accepts_positive_int() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {
            "id": "b",
            "kind": "worker",
            "directive": "y",
            "depends_on": ["a"],
            "consumes": ["a"],
            "fan_out_on": "items",
            "fan_out_concurrency": 1,
            "tier": "haiku",
        },
    ])
    plan = validate_spec(spec)
    fanout = next(n for n in plan if n["id"] == "b")
    assert fanout["fan_out_concurrency"] == 1


def test_fan_out_concurrency_zero_rejected() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {
            "id": "b",
            "kind": "worker",
            "directive": "y",
            "depends_on": ["a"],
            "consumes": ["a"],
            "fan_out_on": "items",
            "fan_out_concurrency": 0,
            "tier": "haiku",
        },
    ])
    with pytest.raises(SpecError, match="fan_out_concurrency"):
        validate_spec(spec)


def test_fan_out_concurrency_bool_rejected() -> None:
    """``True`` is an int subclass; the validator must not silently accept
    it as concurrency=1."""
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {
            "id": "b",
            "kind": "worker",
            "directive": "y",
            "depends_on": ["a"],
            "consumes": ["a"],
            "fan_out_on": "items",
            "fan_out_concurrency": True,
            "tier": "haiku",
        },
    ])
    with pytest.raises(SpecError, match="fan_out_concurrency"):
        validate_spec(spec)


def test_fan_out_concurrency_without_fan_out_on_rejected() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {
            "id": "b",
            "kind": "worker",
            "directive": "y",
            "depends_on": ["a"],
            "consumes": ["a"],
            "fan_out_concurrency": 1,
            "tier": "haiku",
        },
    ])
    with pytest.raises(SpecError, match="fan_out_concurrency"):
        validate_spec(spec)


def test_approval_step_with_tier_rejected() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {
            "id": "b",
            "kind": "approval",
            "directive": "approve",
            "depends_on": ["a"],
            "consumes": ["a"],
            "tier": "haiku",
        },
    ])
    with pytest.raises(SpecError, match="tier"):
        validate_spec(spec)


def test_approval_step_without_tier_accepted() -> None:
    spec = _base_spec(plan=[
        {"id": "a", "kind": "worker", "directive": "x", "tier": "haiku"},
        {
            "id": "b",
            "kind": "approval",
            "directive": "approve",
            "depends_on": ["a"],
            "consumes": ["a"],
        },
    ])
    plan = validate_spec(spec)
    assert plan[1]["kind"] == "approval"
    assert plan[1]["tier"] is None

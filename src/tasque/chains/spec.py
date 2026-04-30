"""Chain plan TypedDicts and ``validate_spec``.

A *spec* is the canonical JSON-serialisable description of a chain — the
shape stored in ``ChainTemplate.plan_json`` and shipped through the
reply-time LLM tools. ``validate_spec`` accepts a raw dict (the parsed
YAML or the LLM-emitted JSON), enforces the structural contract, and
returns a list of ``PlanNode`` dicts with defaults filled in.

The validator is the single guard that prevents typo'd keys, illegal
``kind`` values, fan-out on non-worker steps, dependency cycles, and the
classic ``consumes`` ⊄ ``depends_on`` mistake — every CRUD path runs it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict, cast

from tasque.jobs.cron import validate_cron
from tasque.llm.factory import ALL_TIERS, Tier

PlanNodeKind = Literal["worker", "approval"]
PlanNodeStatus = Literal[
    "pending", "running", "completed", "failed", "stopped", "awaiting_user"
]
PlanNodeOrigin = Literal["spec", "planner"]
OnFailure = Literal["halt", "replan"]

ALLOWED_NODE_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "kind",
        "directive",
        "depends_on",
        "consumes",
        "fan_out_on",
        "fan_out_concurrency",
        "status",
        "origin",
        "on_failure",
        "failure_reason",
        "fan_out_index",
        "fan_out_item",
        "tier",
        "produces_schema",
    }
)

ALLOWED_SPEC_KEYS: frozenset[str] = frozenset(
    {"chain_name", "bucket", "recurrence", "plan", "planner_tier", "vars"}
)


class ProducesListItemsSchema(TypedDict, total=False):
    """Per-list-item shape for ``ProducesSchema.list_items``."""

    required: list[str]
    min_count: int
    max_count: int


class ProducesSchema(TypedDict, total=False):
    """Optional shape declaration for a worker step's ``produces`` dict.

    Specs that omit ``produces_schema`` accept any shape (current
    behavior). When set, ``worker.validate_produces`` runs at step-end
    and any violation flips the step to ``failed`` with a clear reason —
    catches passthrough aggregators that silently drop fields when the
    LLM re-emits a list, which is the failure mode that motivated this
    schema in the first place.

    ``required`` lists top-level keys that must appear in produces.
    ``list_items[key]`` constrains a list-valued key: every item must be
    a dict and contain every key listed in ``required``; ``min_count`` /
    ``max_count`` bound list length.
    """

    required: list[str]
    list_items: dict[str, ProducesListItemsSchema]


class PlanNode(TypedDict):
    """One step in a chain plan.

    All fields are present after ``validate_spec`` fills defaults.

    ``fan_out_concurrency`` caps how many materialized children of a
    fan-out template may be in ``running`` state at the same time.
    ``None`` (the default) means unbounded — every materialized child
    starts in parallel. Set to a positive integer to serialise (1) or
    bound (>1) the fan-out, useful when the workers contend on a shared
    external resource (e.g. a single browser profile, a rate-limited
    API). Stored on the template; copied onto each child by
    ``_materialize_fan_out`` so the supervisor can read it during
    promotion without looking up the template separately.

    ``produces_schema`` (optional) declares the shape of ``produces`` so
    a passthrough aggregator that silently drops fields fails loudly
    instead of corrupting downstream data. None means no enforcement.
    """

    id: str
    kind: PlanNodeKind
    directive: str
    depends_on: list[str]
    consumes: list[str]
    fan_out_on: str | None
    fan_out_concurrency: int | None
    status: PlanNodeStatus
    origin: PlanNodeOrigin
    on_failure: OnFailure
    failure_reason: str | None
    fan_out_index: int | None
    fan_out_item: Any
    tier: Tier | None
    produces_schema: ProducesSchema | None


class CompletedOutput(TypedDict):
    """What a worker step leaves behind for downstream consumers."""

    report: str
    produces: dict[str, Any]


class HistoryEntryKind:
    STATUS = "status"
    MUTATION = "mutation"
    INTERRUPT = "interrupt"
    RESUME = "resume"
    PAUSE = "pause"
    RESUME_PAUSE = "resume_pause"


HistoryKindLiteral = Literal[
    "status", "mutation", "interrupt", "resume", "pause", "resume_pause"
]


class HistoryEntry(TypedDict):
    timestamp: str
    kind: HistoryKindLiteral
    details: dict[str, Any]


class ChainState(TypedDict, total=False):
    """The full state threaded through the chain LangGraph.

    Reducer semantics live in ``chains.graph._common`` (see
    ``CHAIN_STATE_SCHEMA``). ``failures`` supports None-on-right to clear
    a key — DLQ retry depends on this. ``history`` is append-only.
    ``vars`` is frozen at launch (operator override merged over the
    spec's static ``vars``) and surfaced to every worker prompt.
    """

    chain_id: str
    chain_name: str
    bucket: str
    thread_id: str | None
    plan: list[PlanNode]
    completed: dict[str, CompletedOutput]
    failures: dict[str, str]
    replan: bool
    history: list[HistoryEntry]
    approval_resume: NotRequired[str | None]
    awaiting_posts: dict[str, str]
    planner_tier: str
    vars: dict[str, Any]


class SpecError(ValueError):
    """Raised by ``validate_spec`` for any structural violation."""


def _validate_one_node(raw: Mapping[str, Any], *, idx: int) -> PlanNode:
    extras = set(raw.keys()) - ALLOWED_NODE_KEYS
    if extras:
        raise SpecError(
            f"plan[{idx}] has unknown keys {sorted(extras)!r}; allowed: "
            f"{sorted(ALLOWED_NODE_KEYS)!r}"
        )
    node_id_raw = raw.get("id")
    if not isinstance(node_id_raw, str) or not node_id_raw.strip():
        raise SpecError(f"plan[{idx}] missing non-empty string 'id'")
    node_id: str = node_id_raw

    kind_raw = raw.get("kind", "worker")
    if kind_raw not in ("worker", "approval"):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) has invalid kind {kind_raw!r}; "
            "expected 'worker' or 'approval'"
        )
    kind: PlanNodeKind = kind_raw

    directive_raw = raw.get("directive")
    if not isinstance(directive_raw, str) or not directive_raw.strip():
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) missing non-empty string 'directive'"
        )
    directive: str = directive_raw

    depends_on_raw = raw.get("depends_on", [])
    if not isinstance(depends_on_raw, list) or not all(
        isinstance(x, str) for x in cast(list[Any], depends_on_raw)
    ):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'depends_on' must be a list of strings"
        )
    depends_on: list[str] = list(cast(list[str], depends_on_raw))

    consumes_raw = raw.get("consumes", [])
    if not isinstance(consumes_raw, list) or not all(
        isinstance(x, str) for x in cast(list[Any], consumes_raw)
    ):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'consumes' must be a list of strings"
        )
    consumes: list[str] = list(cast(list[str], consumes_raw))

    fan_out_on_raw = raw.get("fan_out_on")
    if fan_out_on_raw is not None and not isinstance(fan_out_on_raw, str):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'fan_out_on' must be a string or null"
        )
    fan_out_on: str | None = fan_out_on_raw

    if fan_out_on is not None and kind != "worker":
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'fan_out_on' is only valid on worker steps; "
            f"got kind={kind!r}"
        )

    fan_out_concurrency_raw = raw.get("fan_out_concurrency")
    fan_out_concurrency: int | None
    if fan_out_concurrency_raw is None:
        fan_out_concurrency = None
    else:
        # Reject bool-as-int: ``True`` is an int subclass in Python and
        # would silently pass through, but the field is a count not a
        # flag.
        if isinstance(fan_out_concurrency_raw, bool) or not isinstance(
            fan_out_concurrency_raw, int
        ):
            raise SpecError(
                f"plan[{idx}] (id={node_id!r}) 'fan_out_concurrency' must be a "
                f"positive integer or null"
            )
        if fan_out_concurrency_raw < 1:
            raise SpecError(
                f"plan[{idx}] (id={node_id!r}) 'fan_out_concurrency' must be >= 1; "
                f"got {fan_out_concurrency_raw}"
            )
        fan_out_concurrency = fan_out_concurrency_raw

    if fan_out_concurrency is not None and fan_out_on is None:
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'fan_out_concurrency' is only valid "
            f"on a fan-out template (set 'fan_out_on' too)"
        )

    on_failure_raw = raw.get("on_failure", "halt")
    if on_failure_raw not in ("halt", "replan"):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'on_failure' must be 'halt' or 'replan'"
        )
    on_failure: OnFailure = on_failure_raw

    status_raw = raw.get("status", "pending")
    if status_raw not in (
        "pending",
        "running",
        "completed",
        "failed",
        "stopped",
        "awaiting_user",
    ):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) invalid 'status' {status_raw!r}"
        )
    status: PlanNodeStatus = status_raw

    origin_raw = raw.get("origin", "spec")
    if origin_raw not in ("spec", "planner"):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'origin' must be 'spec' or 'planner'"
        )
    origin: PlanNodeOrigin = origin_raw

    failure_reason_raw = raw.get("failure_reason")
    if failure_reason_raw is not None and not isinstance(failure_reason_raw, str):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'failure_reason' must be string or null"
        )

    fan_out_index_raw = raw.get("fan_out_index")
    if fan_out_index_raw is not None and not isinstance(fan_out_index_raw, int):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'fan_out_index' must be int or null"
        )

    tier_raw = raw.get("tier")
    tier_value: Tier | None
    if kind == "approval":
        # Approval steps don't call an LLM. Reject an explicit tier so
        # the spec author doesn't think it does anything.
        if tier_raw is not None:
            raise SpecError(
                f"plan[{idx}] (id={node_id!r}) 'tier' is only valid on "
                f"worker steps; approval steps don't call an LLM"
            )
        tier_value = None
    else:
        if tier_raw is None:
            raise SpecError(
                f"plan[{idx}] (id={node_id!r}) is missing required 'tier' "
                f"(must be one of {sorted(ALL_TIERS)!r})"
            )
        if not isinstance(tier_raw, str) or tier_raw not in ALL_TIERS:
            raise SpecError(
                f"plan[{idx}] (id={node_id!r}) 'tier' must be one of "
                f"{sorted(ALL_TIERS)!r}, got {tier_raw!r}"
            )
        tier_value = tier_raw  # type: ignore[assignment]

    consumes_extras = set(consumes) - set(depends_on)
    if consumes_extras:
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'consumes' contains ids not in "
            f"'depends_on': {sorted(consumes_extras)!r}"
        )

    produces_schema_raw = raw.get("produces_schema")
    produces_schema_value: ProducesSchema | None
    if produces_schema_raw is None:
        produces_schema_value = None
    else:
        if kind != "worker":
            raise SpecError(
                f"plan[{idx}] (id={node_id!r}) 'produces_schema' is only valid "
                f"on worker steps"
            )
        produces_schema_value = _validate_produces_schema(
            produces_schema_raw, node_id=node_id, idx=idx
        )

    return PlanNode(
        id=node_id,
        kind=kind,
        directive=directive,
        depends_on=depends_on,
        consumes=consumes,
        fan_out_on=fan_out_on,
        fan_out_concurrency=fan_out_concurrency,
        status=status,
        origin=origin,
        on_failure=on_failure,
        failure_reason=failure_reason_raw,
        fan_out_index=fan_out_index_raw,
        fan_out_item=raw.get("fan_out_item"),
        tier=tier_value,
        produces_schema=produces_schema_value,
    )


def _validate_produces_schema(
    raw: Any, *, node_id: str, idx: int
) -> ProducesSchema:
    """Validate the static ``produces_schema`` declaration on a plan node."""
    if not isinstance(raw, Mapping):
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'produces_schema' must be a mapping"
        )
    allowed = {"required", "list_items"}
    extras = set(raw.keys()) - allowed
    if extras:
        raise SpecError(
            f"plan[{idx}] (id={node_id!r}) 'produces_schema' has unknown keys "
            f"{sorted(extras)!r}; allowed: {sorted(allowed)!r}"
        )
    out: ProducesSchema = {}
    if "required" in raw:
        req_raw = raw["required"]
        if not isinstance(req_raw, list) or not all(
            isinstance(x, str) and x for x in cast(list[Any], req_raw)
        ):
            raise SpecError(
                f"plan[{idx}] (id={node_id!r}) 'produces_schema.required' "
                f"must be a list of non-empty strings"
            )
        out["required"] = list(cast(list[str], req_raw))
    if "list_items" in raw:
        li_raw = raw["list_items"]
        if not isinstance(li_raw, Mapping):
            raise SpecError(
                f"plan[{idx}] (id={node_id!r}) 'produces_schema.list_items' "
                f"must be a mapping"
            )
        list_items: dict[str, ProducesListItemsSchema] = {}
        for key, item_raw in li_raw.items():
            if not isinstance(key, str) or not key:
                raise SpecError(
                    f"plan[{idx}] (id={node_id!r}) 'produces_schema.list_items' "
                    f"has non-string key {key!r}"
                )
            if not isinstance(item_raw, Mapping):
                raise SpecError(
                    f"plan[{idx}] (id={node_id!r}) "
                    f"'produces_schema.list_items[{key!r}]' must be a mapping"
                )
            allowed_item = {"required", "min_count", "max_count"}
            extras_item = set(item_raw.keys()) - allowed_item
            if extras_item:
                raise SpecError(
                    f"plan[{idx}] (id={node_id!r}) "
                    f"'produces_schema.list_items[{key!r}]' has unknown keys "
                    f"{sorted(extras_item)!r}; allowed: {sorted(allowed_item)!r}"
                )
            item_out: ProducesListItemsSchema = {}
            if "required" in item_raw:
                ir = item_raw["required"]
                if not isinstance(ir, list) or not all(
                    isinstance(x, str) and x for x in cast(list[Any], ir)
                ):
                    raise SpecError(
                        f"plan[{idx}] (id={node_id!r}) "
                        f"'produces_schema.list_items[{key!r}].required' must "
                        f"be a list of non-empty strings"
                    )
                item_out["required"] = list(cast(list[str], ir))
            for bound in ("min_count", "max_count"):
                if bound in item_raw:
                    val = item_raw[bound]
                    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                        raise SpecError(
                            f"plan[{idx}] (id={node_id!r}) "
                            f"'produces_schema.list_items[{key!r}].{bound}' "
                            f"must be a non-negative integer"
                        )
                    item_out[bound] = val  # type: ignore[literal-required]
            list_items[key] = item_out
        out["list_items"] = list_items
    return out


def validate_produces(
    schema: ProducesSchema | None, produces: Mapping[str, Any]
) -> list[str]:
    """Check that ``produces`` matches ``schema``. Returns a list of error
    messages — empty list means it matched. ``schema=None`` always returns
    ``[]`` (no enforcement).
    """
    if schema is None:
        return []
    errors: list[str] = []
    required = schema.get("required") or []
    for key in required:
        if key not in produces:
            errors.append(f"missing required key {key!r}")
    list_items = schema.get("list_items") or {}
    for key, item_schema in list_items.items():
        value = produces.get(key)
        if value is None:
            # Missing-required already flagged above; if not required,
            # silently skip the list-item check.
            continue
        if not isinstance(value, list):
            errors.append(
                f"key {key!r} must be a list, got {type(value).__name__}"
            )
            continue
        items = cast(list[Any], value)
        min_count = item_schema.get("min_count")
        if isinstance(min_count, int) and len(items) < min_count:
            errors.append(
                f"key {key!r} has {len(items)} items, requires >= {min_count}"
            )
        max_count = item_schema.get("max_count")
        if isinstance(max_count, int) and len(items) > max_count:
            errors.append(
                f"key {key!r} has {len(items)} items, requires <= {max_count}"
            )
        item_required = item_schema.get("required") or []
        if not item_required:
            continue
        for i, item in enumerate(items):
            if not isinstance(item, Mapping):
                errors.append(
                    f"key {key!r}[{i}] must be a mapping, got {type(item).__name__}"
                )
                continue
            missing = [k for k in item_required if k not in item]
            if missing:
                errors.append(
                    f"key {key!r}[{i}] missing keys {missing!r}"
                )
    return errors


def _check_dag(plan: list[PlanNode]) -> None:
    by_id: dict[str, PlanNode] = {}
    for n in plan:
        if n["id"] in by_id:
            raise SpecError(f"duplicate plan node id: {n['id']!r}")
        by_id[n["id"]] = n

    for n in plan:
        for dep in n["depends_on"]:
            if dep not in by_id:
                raise SpecError(
                    f"plan node {n['id']!r} depends on unknown id {dep!r}"
                )

    # Iterative DFS cycle check using three colors.
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in by_id}
    for start in by_id:
        if color[start] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = GRAY
        while stack:
            nid, dep_idx = stack[-1]
            deps = by_id[nid]["depends_on"]
            if dep_idx >= len(deps):
                color[nid] = BLACK
                stack.pop()
                continue
            stack[-1] = (nid, dep_idx + 1)
            nxt = deps[dep_idx]
            if color[nxt] == GRAY:
                raise SpecError(
                    f"dependency cycle detected involving {nxt!r}"
                )
            if color[nxt] == WHITE:
                color[nxt] = GRAY
                stack.append((nxt, 0))


def validate_spec(raw: Mapping[str, Any]) -> list[PlanNode]:
    """Validate ``raw`` and return the plan as a list of ``PlanNode``.

    ``raw`` is the full chain spec dict (chain_name, bucket, recurrence,
    plan). Unknown top-level keys, missing/empty fields, illegal kinds,
    fan-out on approvals, ``consumes`` ⊄ ``depends_on``, dependency
    cycles, and bad cron expressions all raise :class:`SpecError`.
    """
    extras = set(raw.keys()) - ALLOWED_SPEC_KEYS
    if extras:
        raise SpecError(
            f"spec has unknown top-level keys {sorted(extras)!r}; allowed: "
            f"{sorted(ALLOWED_SPEC_KEYS)!r}"
        )

    chain_name_raw = raw.get("chain_name")
    if not isinstance(chain_name_raw, str) or not chain_name_raw.strip():
        raise SpecError("spec missing non-empty 'chain_name'")

    recurrence_raw = raw.get("recurrence")
    if recurrence_raw is not None:
        if not isinstance(recurrence_raw, str):
            raise SpecError("spec 'recurrence' must be a string or null")
        err = validate_cron(recurrence_raw)
        if err is not None:
            raise SpecError(f"spec 'recurrence' is invalid: {err}")

    bucket_raw = raw.get("bucket")
    if bucket_raw is not None and not isinstance(bucket_raw, str):
        raise SpecError("spec 'bucket' must be a string or null")

    planner_tier_raw = raw.get("planner_tier")
    if planner_tier_raw is None:
        raise SpecError(
            f"spec missing required 'planner_tier' "
            f"(must be one of {sorted(ALL_TIERS)!r})"
        )
    if not isinstance(planner_tier_raw, str) or planner_tier_raw not in ALL_TIERS:
        raise SpecError(
            f"spec 'planner_tier' must be one of {sorted(ALL_TIERS)!r}, "
            f"got {planner_tier_raw!r}"
        )

    vars_raw = raw.get("vars")
    if vars_raw is not None and not isinstance(vars_raw, Mapping):
        raise SpecError(
            f"spec 'vars' must be an object or null, got {type(vars_raw).__name__}"
        )

    plan_raw = raw.get("plan")
    if not isinstance(plan_raw, list) or not plan_raw:
        raise SpecError("spec 'plan' must be a non-empty list")

    plan: list[PlanNode] = []
    for idx, node_raw in enumerate(cast(list[Any], plan_raw)):
        if not isinstance(node_raw, Mapping):
            raise SpecError(
                f"plan[{idx}] must be a mapping, got {type(node_raw).__name__}"
            )
        plan.append(_validate_one_node(node_raw, idx=idx))

    _check_dag(plan)
    return plan


def spec_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return the (chain_name, bucket, recurrence, planner_tier) tuple
    from a validated spec.

    Callers should call :func:`validate_spec` first; this helper does
    not re-validate. Assumes ``planner_tier`` is present (validator
    enforces it).
    """
    return {
        "chain_name": raw.get("chain_name"),
        "bucket": raw.get("bucket"),
        "recurrence": raw.get("recurrence"),
        "planner_tier": raw.get("planner_tier"),
    }


def resolve_planner_tier(raw: Mapping[str, Any]) -> Tier:
    """Return the validated planner tier. Raises if missing or invalid.

    Validator-side checks already enforce presence; this helper is for
    runtime call sites that want a typed value without re-validating
    the whole spec.
    """
    value = raw.get("planner_tier")
    if not isinstance(value, str) or value not in ALL_TIERS:
        raise SpecError(
            f"spec 'planner_tier' must be one of {sorted(ALL_TIERS)!r}; "
            f"got {value!r}"
        )
    return value  # type: ignore[return-value]


__all__ = [
    "ALLOWED_NODE_KEYS",
    "ALLOWED_SPEC_KEYS",
    "ChainState",
    "CompletedOutput",
    "HistoryEntry",
    "HistoryKindLiteral",
    "OnFailure",
    "PlanNode",
    "PlanNodeKind",
    "PlanNodeOrigin",
    "PlanNodeStatus",
    "ProducesListItemsSchema",
    "ProducesSchema",
    "SpecError",
    "resolve_planner_tier",
    "spec_metadata",
    "validate_produces",
    "validate_spec",
]

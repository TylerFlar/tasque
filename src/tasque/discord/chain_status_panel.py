"""Live chain status — one embed per ChainRun, edited in place.

Pure-rendering side. The watcher in
:mod:`tasque.discord.chain_status_watcher` polls running chains, calls
:func:`build_chain_status_snapshot` against the LangGraph checkpoint, and
hands the embed to the poster. Keeping the renderer pure means tests can
exercise every fan-out / failure / terminal-status shape without
touching Discord or the checkpointer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict, cast
from zoneinfo import ZoneInfo

from tasque.chains.spec import PlanNode
from tasque.config import get_settings
from tasque.memory.entities import ChainRun

# Status icons — one cell wide so the tree alignment doesn't drift.
_ICON_PENDING = "·"
_ICON_RUNNING = "▶"
_ICON_AWAITING = "⏸"
_ICON_COMPLETED = "✓"
_ICON_FAILED = "✗"
_ICON_STOPPED = "■"

_ACTIVE_NODE_STATUSES = ("running", "awaiting_user")
_TERMINAL_RUN_STATUSES = ("completed", "failed", "stopped")
_TERMINAL_NODE_STATUSES = ("completed", "failed", "stopped")

_COLOR_BLURPLE = 0x5865F2
_COLOR_GREEN = 0x2ECC71
_COLOR_YELLOW = 0xF1C40F
_COLOR_RED = 0xE74C3C
_COLOR_GREY = 0x95A5A6

EMBED_DESC_LIMIT = 4096


def _icon(status: str) -> str:
    return {
        "pending": _ICON_PENDING,
        "running": _ICON_RUNNING,
        "awaiting_user": _ICON_AWAITING,
        "completed": _ICON_COMPLETED,
        "failed": _ICON_FAILED,
        "stopped": _ICON_STOPPED,
    }.get(status, "?")


# ----------------------------------------------------------------- snapshot


class ChainStatusSnapshot(TypedDict):
    """The shape :func:`build_chain_status_snapshot` returns.

    All fields are JSON-serialisable so the snapshot can be hashed for
    no-op-edit detection in the watcher.
    """

    chain_id: str
    chain_name: str
    bucket: str | None
    run_status: str
    started_at: str
    ended_at: str | None
    counts: dict[str, int]
    in_flight: list[str]
    failures: dict[str, str]
    tree_lines: list[str]


def _is_fanout_child(node_id: str) -> bool:
    return "[" in node_id and node_id.endswith("]")


def _fanout_template(node_id: str) -> str:
    return node_id.split("[", 1)[0]


def _build_tree_lines(plan: list[PlanNode]) -> list[str]:
    """Render the plan as an indented tree.

    Roots are nodes with no ``depends_on``. Children are indented one
    level per ``depends_on`` hop. Fan-out children (``id == "step[N]"``)
    are indented under their template parent regardless of their formal
    depends_on edge — that's how the user thinks about them.
    """
    children_by_template: dict[str, list[PlanNode]] = {}
    parents: dict[str, list[PlanNode]] = {}
    roots: list[PlanNode] = []

    for n in plan:
        nid = n["id"]
        if _is_fanout_child(nid):
            children_by_template.setdefault(_fanout_template(nid), []).append(n)
            continue
        if not n["depends_on"]:
            roots.append(n)
        else:
            for d in n["depends_on"]:
                parents.setdefault(d, []).append(n)

    # Stable child ordering: by id.
    for v in children_by_template.values():
        v.sort(key=lambda n: n["id"])
    for v in parents.values():
        v.sort(key=lambda n: n["id"])
    roots.sort(key=lambda n: n["id"])

    lines: list[str] = []
    seen: set[str] = set()

    def _format_one(node: PlanNode, depth: int, *, is_fanout_child: bool) -> str:
        indent = "  " * depth
        icon = _icon(node["status"])
        nid = node["id"]
        kind = node["kind"]
        suffix_parts: list[str] = []
        # Mark fan-out template steps so the user can tell the parent
        # apart from the children at a glance.
        if not is_fanout_child and node.get("fan_out_on"):
            template_children = children_by_template.get(nid, [])
            if template_children:
                suffix_parts.append(f"fan-out x{len(template_children)}")
            else:
                # Fan-out parent that hasn't materialised children yet.
                suffix_parts.append(f"fan-out on `{node['fan_out_on']}`")
        if node["status"] == "failed" and node.get("failure_reason"):
            suffix_parts.append(
                f"err: {(node['failure_reason'] or '').splitlines()[0][:60]}"
            )
        suffix = f"  ({', '.join(suffix_parts)})" if suffix_parts else ""
        return f"{indent}{icon} `{nid}` _{kind}_{suffix}"

    def _walk(node: PlanNode, depth: int) -> None:
        if node["id"] in seen:
            return
        seen.add(node["id"])
        lines.append(_format_one(node, depth, is_fanout_child=False))
        for child in children_by_template.get(node["id"], []):
            seen.add(child["id"])
            lines.append(_format_one(child, depth + 1, is_fanout_child=True))
        for nxt in parents.get(node["id"], []):
            _walk(nxt, depth + 1)

    for r in roots:
        _walk(r, 0)

    # Anything not reached (orphaned in the DAG, shouldn't normally
    # happen but be defensive) lands at the bottom flat.
    for n in plan:
        if n["id"] not in seen:
            lines.append(_format_one(n, 0, is_fanout_child=_is_fanout_child(n["id"])))

    return lines


def _count_statuses(plan: list[PlanNode]) -> dict[str, int]:
    counts: dict[str, int] = {
        "pending": 0,
        "running": 0,
        "awaiting_user": 0,
        "completed": 0,
        "failed": 0,
        "stopped": 0,
        "total": len(plan),
    }
    for n in plan:
        counts[n["status"]] = counts.get(n["status"], 0) + 1
    return counts


def build_chain_status_snapshot(
    chain_run: ChainRun,
    state: dict[str, Any] | None,
) -> ChainStatusSnapshot:
    """Build a JSON-serialisable snapshot of a chain's live state.

    ``state`` is the raw checkpoint state dict (as returned by
    :func:`tasque.chains.manager.get_chain_state`). When ``state`` is
    ``None`` (no checkpoint yet) the snapshot still renders — it just
    shows the run-row metadata and an empty plan.
    """
    plan_raw: list[Any] = []
    failures_raw: dict[str, Any] = {}
    if state is not None:
        plan_raw = list(state.get("plan") or [])
        failures_raw = dict(state.get("failures") or {})

    plan: list[PlanNode] = cast(list[PlanNode], plan_raw)
    counts = _count_statuses(plan)
    in_flight = [n["id"] for n in plan if n["status"] in _ACTIVE_NODE_STATUSES]
    in_flight.sort()
    tree = _build_tree_lines(plan)
    failures = {str(k): str(v) for k, v in failures_raw.items()}

    return {
        "chain_id": chain_run.chain_id,
        "chain_name": chain_run.chain_name,
        "bucket": chain_run.bucket,
        "run_status": chain_run.status,
        "started_at": chain_run.started_at,
        "ended_at": chain_run.ended_at,
        "counts": counts,
        "in_flight": in_flight,
        "failures": failures,
        "tree_lines": tree,
    }


# ----------------------------------------------------------------- embed


def _local_tz() -> Any:
    tz_name = get_settings().tasque_timezone or "UTC"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return UTC


def _format_local(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        return iso
    return dt.astimezone(_local_tz()).strftime("%a %m-%d %I:%M %p %Z").replace(" 0", " ")


def _color_for(snapshot: ChainStatusSnapshot) -> int:
    status = snapshot["run_status"]
    if status == "completed":
        return _COLOR_GREEN
    if status == "failed":
        return _COLOR_RED
    if status == "stopped":
        return _COLOR_GREY
    if status in ("awaiting_approval", "awaiting_user", "paused"):
        return _COLOR_YELLOW
    if snapshot["counts"].get("failed", 0) > 0:
        return _COLOR_YELLOW
    return _COLOR_BLURPLE  # running / unknown


def _summary_line(snapshot: ChainStatusSnapshot) -> str:
    c = snapshot["counts"]
    total = max(int(c.get("total", 0)), 0)
    done = int(c.get("completed", 0))
    in_flight = int(c.get("running", 0)) + int(c.get("awaiting_user", 0))
    pending = int(c.get("pending", 0))
    failed = int(c.get("failed", 0))
    stopped = int(c.get("stopped", 0))
    parts = [f"step **{done}/{total}**"]
    if in_flight:
        parts.append(f"in flight **{in_flight}**")
    if pending:
        parts.append(f"pending **{pending}**")
    if failed:
        parts.append(f"failed **{failed}**")
    if stopped:
        parts.append(f"stopped **{stopped}**")
    return "  •  ".join(parts)


def _truncate_lines(lines: list[str], limit: int) -> str:
    """Join ``lines`` with newlines, dropping the tail when the joined
    text would exceed ``limit``. Always leaves a clear truncation marker
    so the user knows there's more they aren't seeing."""
    out: list[str] = []
    used = 0
    for line in lines:
        # +1 for the newline.
        added = len(line) + 1
        if used + added > limit - 32:  # leave room for the marker
            remaining = len(lines) - len(out)
            out.append(f"… +{remaining} more steps")
            break
        out.append(line)
        used += added
    return "\n".join(out)


def build_chain_status_embed(snapshot: ChainStatusSnapshot) -> dict[str, Any]:
    """Render the snapshot as a Discord embed dict ready to post or edit."""
    title_status = snapshot["run_status"]
    title = f"Chain: {snapshot['chain_name']}  •  {title_status}"

    description_parts: list[str] = [_summary_line(snapshot)]
    if snapshot["in_flight"]:
        in_flight_str = ", ".join(f"`{nid}`" for nid in snapshot["in_flight"][:5])
        if len(snapshot["in_flight"]) > 5:
            in_flight_str += f" … +{len(snapshot['in_flight']) - 5}"
        description_parts.append(f"**Now:** {in_flight_str}")

    if snapshot["tree_lines"]:
        # Reserve ~600 chars for the header + footer text; the rest goes
        # to the tree. The truncator drops trailing rows on overflow.
        budget = max(EMBED_DESC_LIMIT - sum(len(p) + 2 for p in description_parts) - 256, 512)
        description_parts.append(_truncate_lines(snapshot["tree_lines"], budget))
    else:
        description_parts.append("_(no plan checkpoint yet — chain just started)_")

    if snapshot["failures"]:
        # Show first failure inline; details belong in DLQ entries.
        first = next(iter(snapshot["failures"].items()))
        description_parts.append(
            f"**Failure** on `{first[0]}`: {first[1].splitlines()[0][:200]}"
        )

    description = "\n\n".join(description_parts)[:EMBED_DESC_LIMIT]

    fields: list[dict[str, Any]] = [
        {"name": "chain_id", "value": snapshot["chain_id"], "inline": True},
    ]
    if snapshot["bucket"]:
        fields.append({"name": "bucket", "value": snapshot["bucket"], "inline": True})
    fields.append(
        {"name": "started", "value": _format_local(snapshot["started_at"]), "inline": True}
    )
    if snapshot["ended_at"]:
        fields.append(
            {"name": "ended", "value": _format_local(snapshot["ended_at"]), "inline": True}
        )

    return {
        "title": title[:256],
        "description": description,
        "color": _color_for(snapshot),
        "fields": fields,
    }


def is_terminal_run(snapshot: ChainStatusSnapshot) -> bool:
    """True when the chain's run-row status is terminal — the watcher
    will do one final edit and then stop touching the message."""
    return snapshot["run_status"] in _TERMINAL_RUN_STATUSES


__all__ = [
    "ChainStatusSnapshot",
    "build_chain_status_embed",
    "build_chain_status_snapshot",
    "is_terminal_run",
]

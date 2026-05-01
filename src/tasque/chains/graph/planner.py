"""The planner node.

Calls the large-tier ``planner`` LLM with the current plan, completed
outputs, and failures. The model emits a single fenced JSON block with a
list of mutations to apply atomically:

```json
{
  "mutations": [
    {"op": "add_step", "node": { ... PlanNode dict ... }},
    {"op": "remove_step", "id": "step-3"},
    {"op": "reorder_deps", "id": "step-4", "depends_on": ["step-1", "step-2"]},
    {"op": "abort_chain"}
  ]
}
```

The planner is opt-in (only invoked when ``state.replan == True``); it
clears ``replan`` before returning so the supervisor doesn't loop back
to it.
"""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from tasque.agents import result_inbox
from tasque.chains.graph._common import ChainStateSchema, _now_iso
from tasque.chains.spec import HistoryEntry, PlanNode, validate_spec
from tasque.llm.factory import ALL_TIERS, get_chat_model_for_tier

PLANNER_SYSTEM_PROMPT = """\
You are the tasque chain planner. A chain step has failed (or otherwise
asked for replanning). Inspect the current plan, completed outputs, and
failure reasons, and propose a small atomic patch.

When you've decided on the mutations, call ``submit_planner_result``
exactly once with the ``result_token`` from the user message:

    submit_planner_result(
      result_token="<value from run context>",
      mutations=[
        { "op": "add_step", "node": { "id": "...", "kind": "worker",
            "directive": "...", "depends_on": [...], "consumes": [...],
            "tier": "small" | "medium" | "large" } },
        { "op": "remove_step", "id": "..." },
        { "op": "reorder_deps", "id": "...", "depends_on": [...] },
        { "op": "abort_chain" }
      ]
    )

Rules:
- Only mutate; do not return a fully rewritten plan.
- ``add_step`` nodes are validated against the spec — same field rules
  as a YAML node. ``origin`` is forced to ``"planner"`` regardless.
  Worker nodes MUST include ``tier`` (one of "large" / "medium" /
  "small"); approval nodes must NOT include a tier. Pick small for
  trivial nudges, medium for multi-step tool / scrape / summarize work,
  large for agentic planning or deep generation.
- ``remove_step`` is a no-op if the id isn't present.
- ``abort_chain`` halts everything; use sparingly.
- Empty mutations list is fine — pass ``mutations=[]`` if the failure
  is truly terminal and nothing remediates it. Do not omit the call.
"""


def _format_plan(plan: list[PlanNode]) -> str:
    return json.dumps(plan, indent=2, default=str)


def _format_completed(completed: dict[str, Any]) -> str:
    return json.dumps(completed, indent=2, default=str)


def _format_failures(failures: dict[str, str]) -> str:
    return json.dumps(failures, indent=2)


def _apply_mutations(
    plan: list[PlanNode], mutations: list[dict[str, Any]]
) -> tuple[list[PlanNode], list[HistoryEntry], bool]:
    """Apply ``mutations`` to ``plan``. Returns (new_plan, history, aborted).

    ``aborted`` is True iff an ``abort_chain`` mutation was applied.
    """
    new_plan: list[PlanNode] = [dict(n) for n in plan]  # type: ignore[misc]
    history: list[HistoryEntry] = []
    aborted = False

    for m in mutations:
        op = m.get("op")
        if op == "add_step":
            raw = m.get("node")
            if not isinstance(raw, dict):
                history.append({
                    "timestamp": _now_iso(),
                    "kind": "mutation",
                    "details": {"op": "add_step", "error": "missing 'node'"},
                })
                continue
            raw_node = cast(dict[str, Any], dict(raw))
            raw_node["origin"] = "planner"
            # Validate via a single-node spec
            try:
                validated = validate_spec({
                    "chain_name": "planner-tmp",
                    "plan": [raw_node],
                })
            except Exception as exc:
                history.append({
                    "timestamp": _now_iso(),
                    "kind": "mutation",
                    "details": {
                        "op": "add_step",
                        "error": f"validation failed: {exc}",
                    },
                })
                continue
            new_node = validated[0]
            if any(n["id"] == new_node["id"] for n in new_plan):
                history.append({
                    "timestamp": _now_iso(),
                    "kind": "mutation",
                    "details": {
                        "op": "add_step",
                        "error": f"id {new_node['id']!r} already exists",
                    },
                })
                continue
            new_plan.append(new_node)
            history.append({
                "timestamp": _now_iso(),
                "kind": "mutation",
                "details": {"op": "add_step", "id": new_node["id"]},
            })

        elif op == "remove_step":
            target = m.get("id")
            if not isinstance(target, str):
                continue
            before = len(new_plan)
            new_plan = [n for n in new_plan if n["id"] != target]
            if len(new_plan) != before:
                history.append({
                    "timestamp": _now_iso(),
                    "kind": "mutation",
                    "details": {"op": "remove_step", "id": target},
                })

        elif op == "reorder_deps":
            target = m.get("id")
            new_deps = m.get("depends_on")
            if not isinstance(target, str) or not isinstance(new_deps, list):
                continue
            for n in new_plan:
                if n["id"] == target:
                    n["depends_on"] = list(cast(list[str], new_deps))
                    history.append({
                        "timestamp": _now_iso(),
                        "kind": "mutation",
                        "details": {
                            "op": "reorder_deps",
                            "id": target,
                            "depends_on": n["depends_on"],
                        },
                    })
                    break

        elif op == "abort_chain":
            aborted = True
            for n in new_plan:
                if n["status"] in ("pending", "running", "awaiting_user"):
                    n["status"] = "stopped"
            history.append({
                "timestamp": _now_iso(),
                "kind": "mutation",
                "details": {"op": "abort_chain"},
            })

    return new_plan, history, aborted


def planner(state: ChainStateSchema) -> dict[str, Any]:
    """Run one planning pass. Always clears ``replan`` so the supervisor
    doesn't loop back here next pass.
    """
    plan: list[PlanNode] = list(state.get("plan") or [])
    completed: dict[str, Any] = state.get("completed") or {}
    failures: dict[str, str] = state.get("failures") or {}
    llm = cast(BaseChatModel | None, cast(dict[str, Any], state).get("llm"))
    if llm is None:
        planner_tier = state.get("planner_tier")
        if not planner_tier or planner_tier not in ALL_TIERS:
            err_history: list[HistoryEntry] = [{
                "timestamp": _now_iso(),
                "kind": "mutation",
                "details": {
                    "op": "<config-error>",
                    "error": (
                        f"chain has no valid 'planner_tier' (got "
                        f"{planner_tier!r}); aborting replan"
                    ),
                },
            }]
            return {"replan": False, "history": err_history}
        llm = get_chat_model_for_tier(planner_tier)

    token = result_inbox.mint_token()
    user = (
        "## Run context\n"
        f"- result_token: {token}  "
        "(pass this to submit_planner_result)\n\n"
        "Current plan:\n"
        f"{_format_plan(plan)}\n\n"
        "Completed outputs:\n"
        f"{_format_completed(completed)}\n\n"
        "Failures (step_id → reason):\n"
        f"{_format_failures(failures)}\n\n"
        "Decide on the mutations and call submit_planner_result with the "
        "result_token above. Pass mutations=[] if nothing should change."
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user),
    ]

    history_appends: list[HistoryEntry] = []
    try:
        llm.invoke(messages)
    except Exception as exc:
        history_appends.append({
            "timestamp": _now_iso(),
            "kind": "mutation",
            "details": {"op": "<llm-error>", "error": f"{type(exc).__name__}: {exc}"},
        })
        return {"replan": False, "history": history_appends}

    payload = result_inbox.read_and_consume(token, agent_kind="planner")
    mutations: list[dict[str, Any]] = []
    if payload is None:
        history_appends.append({
            "timestamp": _now_iso(),
            "kind": "mutation",
            "details": {
                "op": "<missing-result>",
                "error": "planner did not call submit_planner_result",
            },
        })
    else:
        raw_muts = payload.get("mutations")
        if isinstance(raw_muts, list):
            mutations = [
                cast(dict[str, Any], m)
                for m in cast(list[Any], raw_muts)
                if isinstance(m, dict)
            ]

    new_plan, mut_history, _aborted = _apply_mutations(plan, mutations)
    history_appends.extend(mut_history)

    update: dict[str, Any] = {"plan": new_plan, "replan": False}
    if history_appends:
        update["history"] = history_appends
    # Clear failures so reconsidered steps can re-fire after add_step.
    if failures:
        update["failures"] = dict.fromkeys(failures.keys(), None)
    return update


__all__ = ["_apply_mutations", "planner"]

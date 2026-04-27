"""Compile and cache the supervisor → (worker | approval | planner | END)
chain graph.

The graph is built once per process; tests can override the cached
compiled graph via :func:`set_compiled_chain_graph` if they need a custom
compile (e.g., a different checkpointer).

Implementation note: we import the graph node modules as modules (not as
symbols) so that ``tasque.chains.graph.worker`` resolves to the submodule
and tests can monkeypatch ``run_worker`` inside it. Importing
``from .worker import worker`` would shadow the submodule with the
function and break ``monkeypatch.setattr(worker_mod, "run_worker", ...)``.
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any

from langgraph.graph import START, StateGraph

from tasque.chains.checkpointer import get_chain_checkpointer
from tasque.chains.graph import approval as _approval_mod
from tasque.chains.graph import planner as _planner_mod
from tasque.chains.graph import supervisor as _supervisor_mod
from tasque.chains.graph import worker as _worker_mod
from tasque.chains.graph._common import ChainStateSchema
from tasque.chains.graph.supervisor import (
    APPROVAL_NODE,
    PLANNER_NODE,
    SUPERVISOR_NODE,
    WORKER_NODE,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

_compiled: Any | None = None
_lock = Lock()


def _build() -> Any:
    sg: StateGraph[ChainStateSchema, Any, ChainStateSchema, ChainStateSchema] = StateGraph(
        ChainStateSchema
    )
    sg.add_node(SUPERVISOR_NODE, _supervisor_mod.supervisor)
    sg.add_node(WORKER_NODE, _worker_mod.worker)
    sg.add_node(APPROVAL_NODE, _approval_mod.approval)
    sg.add_node(PLANNER_NODE, _planner_mod.planner)

    sg.add_edge(START, SUPERVISOR_NODE)
    sg.add_conditional_edges(SUPERVISOR_NODE, _supervisor_mod._route_from_supervisor)
    sg.add_edge(WORKER_NODE, SUPERVISOR_NODE)
    sg.add_edge(APPROVAL_NODE, SUPERVISOR_NODE)
    sg.add_edge(PLANNER_NODE, SUPERVISOR_NODE)

    saver = get_chain_checkpointer()
    return sg.compile(checkpointer=saver)


def get_compiled_chain_graph() -> Any:
    """Return the singleton compiled chain graph."""
    global _compiled
    if _compiled is not None:
        return _compiled
    with _lock:
        if _compiled is None:
            _compiled = _build()
    return _compiled


def set_compiled_chain_graph(graph: Any) -> None:
    """Override the cached compiled graph — primarily for tests."""
    global _compiled
    _compiled = graph


def reset_compiled_chain_graph() -> None:
    """Drop the cached compiled graph so the next call rebuilds it."""
    global _compiled
    _compiled = None


__all__ = [
    "get_compiled_chain_graph",
    "reset_compiled_chain_graph",
    "set_compiled_chain_graph",
]


_ = CompiledStateGraph if TYPE_CHECKING else None

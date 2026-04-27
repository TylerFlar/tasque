"""Strategist agent — long-horizon goal decomposition + cross-bucket monitoring.

Two roles:

1. Decomposition: when the user adds a long-horizon Aim via the
   strategist's Discord thread, break it into per-bucket Aims and send
   Signals to the affected coaches. Runs through the parameterised
   reply runtime (see :mod:`tasque.reply.strategist`); writes are made
   via the tasque MCP tool catalog.

2. Monitoring: on a schedule (default weekly), survey the cross-bucket
   state, post a markdown summary to the strategist thread, and
   optionally emit new Aims / Signals or flip stale Aim statuses. Runs
   through :func:`run_monitoring` / :func:`run_monitoring_and_post`.
"""

from __future__ import annotations

from tasque.strategist.graph import (
    DEFAULT_HORIZON_DAYS,
    STRATEGIST_DIRECTIVE_SENTINEL,
    StrategistState,
    run_monitoring,
    run_monitoring_and_post,
)
from tasque.strategist.output import (
    AimOutput,
    AimStatusChange,
    SignalOutput,
    StrategistOutput,
)
from tasque.strategist.persist import persist_results, post_summary

__all__ = [
    "DEFAULT_HORIZON_DAYS",
    "STRATEGIST_DIRECTIVE_SENTINEL",
    "AimOutput",
    "AimStatusChange",
    "SignalOutput",
    "StrategistOutput",
    "StrategistState",
    "persist_results",
    "post_summary",
    "run_monitoring",
    "run_monitoring_and_post",
]

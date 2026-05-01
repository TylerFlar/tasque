"""tasque MCP server — full read/write access to the daemon's memory,
queued jobs, chain templates, and chain runs over stdio.

Registered under the user's host MCP config so every upstream call routed
through the tasque proxy inherits the tool catalog. That
puts the same write surface in front of every LLM the daemon spawns
(reactive coach, worker, chain planner, strategist, reply coach) — they
all get to fire chains, queue jobs, edit templates, and write notes
mid-turn instead of declaring intent in trailing JSON.
"""

from tasque.mcp.server import build_server, run_stdio

__all__ = ["build_server", "run_stdio"]

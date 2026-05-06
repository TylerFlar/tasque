"""Parameterised reply runtime: one ReAct loop, per-agent bindings.

Two bindings ship: the bucket coach reply and the strategist reply.
The runtime does not own any per-agent state; callers pass in
``system_prompt`` and ``tools``. Replies are synchronous; follow-up work
comes from tool calls made inside the reply turn.
"""

from __future__ import annotations

from tasque.reply.runtime import ReplyResult, run_reply

__all__ = ["ReplyResult", "run_reply"]

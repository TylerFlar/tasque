"""Parameterised reply runtime — one ReAct loop, per-agent bindings.

Two bindings ship: the bucket coach reply and the strategist reply.
The runtime does not own any per-agent state — callers pass in
``system_prompt``, ``tools``, and an optional ``post_reply_hook``.
The strategist binding has no post-reply hook.
"""

from __future__ import annotations

from tasque.reply.runtime import ReplyResult, run_reply

__all__ = ["ReplyResult", "run_reply"]

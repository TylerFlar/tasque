"""Auto-condense oversize MCP read-tool results via the haiku tier.

Read tools (`note_search`, `chain_run_get` with `include_state=True`,
`job_list`, etc.) can return tens of kilobytes of JSON. Loading that
straight into a coach / worker / strategist context wastes tokens —
most of the payload is irrelevant to the call. To fix it we borrow the
craft-agents pattern:

1. Every read tool takes a required ``intent`` parameter — one short
   sentence the calling LLM writes describing what it actually wants
   from the tool. ("which durable health notes mention sleep").
2. After the tool runs, if the JSON result exceeds a byte threshold,
   we route it plus the intent through a haiku call that returns a
   condensed JSON envelope.
3. The agent receives either the original result (under threshold) or
   ``{"_condensed": True, "_original_bytes": N, "_intent": "...",
   "_tool": "...", "summary": "..."}``.

Condensation is best-effort: if the haiku call fails (proxy down, env
mismatch, transport error) we return the original verbatim. Better an
oversize correct response than a lost one.

The threshold defaults to 60 KB (matching the craft-agents heuristic
and roughly aligning with a coach context budget) and can be tuned via
``TASQUE_MCP_CONDENSE_THRESHOLD`` (bytes, integer).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

DEFAULT_THRESHOLD_BYTES = 60_000


log = logging.getLogger(__name__)


def _threshold_bytes() -> int:
    raw = os.environ.get("TASQUE_MCP_CONDENSE_THRESHOLD")
    if not raw:
        return DEFAULT_THRESHOLD_BYTES
    try:
        v = int(raw)
    except ValueError:
        return DEFAULT_THRESHOLD_BYTES
    return v if v > 0 else DEFAULT_THRESHOLD_BYTES


_CONDENSE_SYSTEM = (
    "You condense a single tool-call result for a downstream LLM. "
    "Preserve the parts of the result that match the caller's stated "
    "intent. Drop irrelevant rows, repetitive metadata, and verbose "
    "payloads. Keep ids and timestamps that match the intent so the "
    "caller can drill in if it needs more. When the original is JSON, "
    "reply with valid JSON of the same shape (an array stays an array; "
    "an object stays an object) but with fewer / shorter entries. "
    "Reply with the condensed result only — no preamble, no markdown "
    "code fences, no commentary."
)


def _condense_via_haiku(result_json: str, *, intent: str, tool_name: str) -> str:
    """Invoke the haiku tier through the tasque proxy.

    Imports lazily so the MCP module's import cost stays bounded —
    ``langchain_openai`` is a heavy dependency and most MCP calls don't
    need it.
    """
    from tasque.llm.factory import get_chat_model_for_tier

    chat = get_chat_model_for_tier("haiku", temperature=0)
    msgs: list[tuple[str, str]] = [
        ("system", _CONDENSE_SYSTEM),
        (
            "user",
            f"Tool: {tool_name}\nCaller intent: {intent}\n\n"
            f"Result (JSON):\n{result_json}",
        ),
    ]
    resp = chat.invoke(msgs)
    content: Any = resp.content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text_part = block.get("text")
                if isinstance(text_part, str):
                    parts.append(text_part)
        text = "".join(parts)
    else:
        text = str(content)
    return text.strip()


def maybe_condense(result_json: str, *, intent: str, tool_name: str) -> str:
    """Return either ``result_json`` verbatim or a condensed envelope.

    Condensation only fires when:

    - the response is at least ``threshold`` bytes (UTF-8 encoded), AND
    - ``intent`` is a non-empty string (no intent → no useful filter).

    On any condensation error we return the original — never raise.
    """
    threshold = _threshold_bytes()
    encoded_size = len(result_json.encode("utf-8"))
    if encoded_size < threshold:
        return result_json
    if not intent.strip():
        return result_json
    try:
        summary = _condense_via_haiku(
            result_json, intent=intent.strip(), tool_name=tool_name
        )
    except Exception as exc:
        log.warning(
            "mcp.condense.failed tool=%s intent=%r size=%d err=%s",
            tool_name,
            intent,
            encoded_size,
            exc,
        )
        return result_json
    envelope: dict[str, Any] = {
        "_condensed": True,
        "_original_bytes": encoded_size,
        "_threshold_bytes": threshold,
        "_intent": intent.strip(),
        "_tool": tool_name,
        "summary": summary,
    }
    return json.dumps(envelope)


__all__ = ["DEFAULT_THRESHOLD_BYTES", "maybe_condense"]

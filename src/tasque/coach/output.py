"""Pydantic schema for the bucket coach's structured JSON output.

The bucket coach is a tool-using agent — note creation, job queueing,
signal sending, and chain firing happen mid-turn through the tasque
MCP. The structured JSON it emits afterwards carries one field:
``thread_post``, a runtime-controlled signal asking tasque's bot to
publish a message to the bucket's Discord thread on the coach's
behalf. The bot owns Discord posting, not the LLM, so this stays a
declarative output.

``extract_json_block`` peels the fence; ``BucketCoachOutput.model_validate_json``
validates the contents.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

# A fenced JSON block, optionally tagged ``json``. Non-greedy body capture.
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


class BucketCoachOutput(BaseModel):
    """The structured response from a single bucket-coach run.

    ``thread_post`` is a non-empty markdown string when the coach wants
    the runtime to publish a Discord post to the bucket's thread, or
    ``None`` for the routine "no announcement" case. Note / job / chain /
    signal writes happen in-turn via MCP tool calls and are not part
    of this schema.
    """

    model_config = ConfigDict(extra="forbid")

    thread_post: str | None = None


def extract_json_block(text: str) -> str | None:
    """Return the JSON object string from a fenced ```json``` block, or None.

    Falls back to the first ``{`` … matching ``}`` substring if no fenced
    block is present, which tolerates models that forget the fences.
    """
    match = _FENCED_JSON_RE.search(text)
    if match is not None:
        return match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

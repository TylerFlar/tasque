"""Pydantic schema for the bucket coach's structured tool-call output.

The bucket coach is a tool-using agent — note creation, job queueing,
signal sending, and chain firing happen mid-turn through the tasque
MCP. The structured payload it submits at the end of its turn carries
one field: ``thread_post``, a runtime-controlled signal asking
tasque's bot to publish a message to the bucket's Discord thread on
the coach's behalf. The bot owns Discord posting, not the LLM, so
this stays a declarative output.

The payload arrives via the ``submit_coach_result`` MCP tool and is
read back from the agent result inbox; this module only defines the
schema used to validate it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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

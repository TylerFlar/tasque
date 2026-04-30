"""Pydantic schema for the strategist's structured monitoring output.

The monitoring graph (``strategist.graph.run_monitoring``) calls the
LLM with the cross-bucket snapshot and expects the model to call
``submit_strategist_result`` with a payload matching
:class:`StrategistOutput`. The graph reads that payload back through
the agent result inbox. The decomposition path does *not* go through
this schema — that one runs through the parameterised reply runtime
and writes Aims/Signals via tools.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tasque.buckets import ALL_BUCKETS, Bucket

AimScope = Literal["long_term", "bucket"]
AimStatus = Literal["active", "completed", "dropped"]
SignalKind = Literal[
    "aim_added",
    "strategist_alert",
    "rebalance",
    "fyi",
]
SignalUrgency = Literal["low", "normal", "high"]


def _validate_bucket(value: str) -> Bucket:
    if value not in ALL_BUCKETS:
        raise ValueError(
            f"unknown bucket {value!r}; must be one of {sorted(ALL_BUCKETS)!r}"
        )
    return value  # type: ignore[return-value]


class AimOutput(BaseModel):
    """One Aim the strategist wants to create during a monitoring run."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    scope: AimScope
    bucket: str | None = None
    target_date: str | None = None
    description: str = ""
    parent_id: str | None = None

    def normalised_bucket(self) -> Bucket | None:
        if self.bucket is None:
            return None
        return _validate_bucket(self.bucket)


class SignalOutput(BaseModel):
    """One Signal the strategist wants to send during a monitoring run."""

    model_config = ConfigDict(extra="forbid")

    to_bucket: str
    kind: SignalKind = "strategist_alert"
    urgency: SignalUrgency = "normal"
    summary: str = Field(min_length=1)
    body: str = ""
    expires_at: str | None = None

    def normalised_to_bucket(self) -> Bucket:
        return _validate_bucket(self.to_bucket)


class AimStatusChange(BaseModel):
    """One in-place ``Aim.status`` flip the strategist wants to apply."""

    model_config = ConfigDict(extra="forbid")

    aim_id: str = Field(min_length=1)
    status: AimStatus
    reason: str = ""


class StrategistOutput(BaseModel):
    """The structured response from a single strategist monitoring run."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    new_aims: list[AimOutput] = Field(default_factory=list)
    signals: list[SignalOutput] = Field(default_factory=list)
    aim_status_changes: list[AimStatusChange] = Field(default_factory=list)


__all__ = [
    "AimOutput",
    "AimScope",
    "AimStatus",
    "AimStatusChange",
    "SignalKind",
    "SignalOutput",
    "SignalUrgency",
    "StrategistOutput",
]

"""ChatOpenAI factory pointed at the local tasque proxy.

Two resolution paths:

1. **Agent-kind path** — :func:`get_chat_model` for fixed-role agents
   (coach, strategist) whose tier is hardcoded by their role. Resolution
   priority: ``TASQUE_MODEL_<KIND>`` as a tier alias or concrete model →
   ``TASQUE_MODEL_<TIER>``.
2. **Tier path** — :func:`get_chat_model_for_tier` for the worker and
   planner, whose tier is chosen per-row (per ``QueuedJob`` and per chain
   plan node / chain spec). Resolution priority:
   ``TASQUE_MODEL_<TIER>``.

The proxy URL defaults to ``http://localhost:3456/v1`` and is overridable
via ``TASQUE_PROXY_BASE_URL``. The model ids may point at any upstream
selected by ``tasque proxy``. The OpenAI API key is a placeholder; the proxy
ignores it but ``ChatOpenAI`` requires *something* non-empty.
"""

from __future__ import annotations

import os
from typing import Any, Literal, get_args

from langchain_openai import ChatOpenAI

AgentKind = Literal["coach", "strategist", "worker", "planner"]
Tier = Literal["large", "medium", "small"]

ALL_AGENT_KINDS: tuple[AgentKind, ...] = get_args(AgentKind)
ALL_TIERS: tuple[Tier, ...] = get_args(Tier)

DEFAULT_PROXY_BASE_URL = "http://localhost:3456/v1"
DEFAULT_PROXY_API_KEY = "tasque-proxy-no-key"

# Built-in default tier per agent kind. Coach and strategist are fixed
# here; worker and planner are kept for backwards compatibility but the
# real per-row tier comes from QueuedJob.tier and PlanNode.tier.
DEFAULT_AGENT_TIER: dict[AgentKind, Tier] = {
    "coach": "large",
    "strategist": "large",
    "planner": "large",
    "worker": "small",
}


class UnknownTierError(ValueError):
    """Raised when a string isn't one of large/medium/small."""


class MissingModelEnvError(ValueError):
    """Raised when a tier has no concrete model id configured."""


def coerce_tier(value: str) -> Tier:
    """Validate ``value`` against :data:`ALL_TIERS` and return it typed."""
    if value not in ALL_TIERS:
        raise UnknownTierError(
            f"unknown tier {value!r}; must be one of {sorted(ALL_TIERS)!r}"
        )
    return value  # type: ignore[return-value]


def resolve_model_id_for_tier(tier: Tier) -> str:
    """Return the model id for ``tier`` after applying env overrides."""
    coerce_tier(tier)
    tier_env = f"TASQUE_MODEL_{tier.upper()}"
    override = os.environ.get(tier_env)
    if override:
        return override
    raise MissingModelEnvError(f"{tier_env} is required for tier {tier!r}")


def resolve_model_id(agent_kind: AgentKind) -> str:
    """Walk the env-override priority chain and return a model id string."""
    kind_env = f"TASQUE_MODEL_{agent_kind.upper()}"
    override = os.environ.get(kind_env)
    if override:
        if override in ALL_TIERS:
            return resolve_model_id_for_tier(override)
        return override

    tier = DEFAULT_AGENT_TIER[agent_kind]
    return resolve_model_id_for_tier(tier)


def _proxy_base_url() -> str:
    return os.environ.get("TASQUE_PROXY_BASE_URL", DEFAULT_PROXY_BASE_URL)


def _proxy_api_key() -> str:
    return os.environ.get("TASQUE_PROXY_API_KEY", DEFAULT_PROXY_API_KEY)


def _build_chat(
    model_id: str,
    *,
    disallowed_tools: list[str] | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    params: dict[str, Any] = {
        "model": model_id,
        "base_url": _proxy_base_url(),
        "api_key": _proxy_api_key(),
    }
    if disallowed_tools:
        # Forwarded to the tasque proxy as a vendor extension. The Claude
        # upstream turns it into ``--disallowedTools``; other upstreams get
        # an equivalent prompt-level instruction when no CLI flag exists.
        existing_extra = kwargs.pop("extra_body", None) or {}
        if not isinstance(existing_extra, dict):
            existing_extra = {}
        params["extra_body"] = {
            **existing_extra,
            "disallowed_tools": list(disallowed_tools),
        }
    params.update(kwargs)
    return ChatOpenAI(**params)


def get_chat_model(
    agent_kind: AgentKind,
    *,
    disallowed_tools: list[str] | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` configured against the local proxy.

    ``**kwargs`` are forwarded to ``ChatOpenAI`` for one-off tweaks
    (``temperature``, ``max_tokens``, etc.). Per-agent temperature/max-token
    env vars are intentionally not provided — set them at the call site.

    ``disallowed_tools`` is a per-call denylist forwarded to the tasque
    proxy and on to the selected upstream's tool-deny mechanism. Use it when a
    call site must not invoke certain MCP tools, such as a consolidation-only
    coach pass that should not start new user-visible work.
    """
    if agent_kind not in ALL_AGENT_KINDS:
        raise ValueError(f"unknown agent kind: {agent_kind!r}")
    return _build_chat(
        resolve_model_id(agent_kind),
        disallowed_tools=disallowed_tools,
        **kwargs,
    )


def get_chat_model_for_tier(
    tier: Tier,
    *,
    disallowed_tools: list[str] | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` for ``tier`` ("large" / "medium" / "small").

    Used by the worker and planner, whose tier is chosen per-row rather
    than fixed by agent kind. ``tier`` must be one of :data:`ALL_TIERS`;
    callers normally pass the value straight from ``QueuedJob.tier`` or a
    ``PlanNode["tier"]``.
    """
    coerce_tier(tier)
    return _build_chat(
        resolve_model_id_for_tier(tier),
        disallowed_tools=disallowed_tools,
        **kwargs,
    )

"""Models for the LLM routing layer (Task 2).

These types give the router's output a fixed, machine-checkable shape.
Pydantic AI validates the router agent's structured output against
``RouteDecision`` automatically, so the orchestrator never has to parse
free-form text to figure out where a request should go.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RouteType(str, Enum):
    """The destination a user request can be routed to.

    Mirrors the "Recommended Route" column from the challenge brief:
    RAG, MCP/live data, a RAG+MCP+reasoning combination, an MCP action,
    the data/analytics agent, or the general-purpose LLM fallback.
    """

    RAG = "rag"
    MCP_LIVE_DATA = "mcp_live_data"
    RAG_MCP_REASONING = "rag_mcp_reasoning"
    MCP_ACTION = "mcp_action"
    DATA_AGENT = "data_agent"
    GENERAL_LLM = "general_llm"
    CLARIFY = "clarify"
    """Used when the router cannot confidently classify the request and
    the orchestrator should ask the user a clarifying question instead
    of guessing (see confidence handling in orchestration/router.py)."""


class RouteDecision(BaseModel):
    """Structured output produced by the routing agent for one user turn.

    ``confidence`` is 0.0 (no confidence) to 1.0 (certain); ``reasoning``
    is a short justification kept for observability/eval, never shown to
    the user; ``is_multi_step`` flags requests likely needing more than
    one tool/agent call (e.g. "investigate and create a request if
    necessary").
    """

    route: RouteType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    requires_live_data: bool = False
    requires_knowledge_base: bool = False
    is_multi_step: bool = False


# Below this confidence threshold, the orchestrator treats the router's
# decision as unreliable and falls back to a clarifying question rather
# than acting on a guess. See orchestration/router.py for usage.
MIN_CONFIDENT_ROUTE_SCORE: float = 0.55

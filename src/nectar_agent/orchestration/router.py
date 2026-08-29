"""LLM-based intent router (Task 2).

Classifies each user turn into a ``RouteDecision`` using a small, fast
model (``settings.llm_model_router``) so that the more expensive
reasoning model is only invoked for the sub-agent(s) actually needed -
this is the primary cost/latency optimization called out in the brief's
evaluation questions.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from nectar_agent.config import get_settings
from nectar_agent.models.routing import MIN_CONFIDENT_ROUTE_SCORE, RouteDecision, RouteType
from nectar_agent.prompts.router_prompt import ROUTER_SYSTEM_PROMPT


@lru_cache
def _get_router_agent() -> Agent[None, RouteDecision]:
    """Build (once) and return the routing Pydantic AI agent.

    Built lazily rather than at import time, since ``Agent.__init__``
    eagerly constructs its model client and needs a valid API key to do
    so - so importing this module never requires credentials, only
    calling ``route()`` does.
    """
    settings = get_settings()
    return Agent(
        settings.llm_model_router,
        output_type=RouteDecision,
        system_prompt=ROUTER_SYSTEM_PROMPT,
    )


def enforce_confidence_threshold(decision: RouteDecision, threshold: float) -> RouteDecision:
    """Coerce a low-confidence route decision to CLARIFY.

    Factored out as a pure function (no LLM call) so the confidence
    policy itself is unit-testable in isolation from the routing model.

    Returns:
        ``decision`` unchanged if it meets ``threshold`` (or is already
        ``CLARIFY``), otherwise a copy with ``route`` forced to
        ``RouteType.CLARIFY`` and ``reasoning`` annotated to explain why.
    """
    if decision.confidence >= threshold or decision.route == RouteType.CLARIFY:
        return decision
    return decision.model_copy(
        update={
            "route": RouteType.CLARIFY,
            "reasoning": (
                f"Confidence {decision.confidence:.2f} below threshold "
                f"{threshold:.2f}; original guess was {decision.route.value}. "
                f"{decision.reasoning}"
            ),
        }
    )


async def route(query: str, conversation_history: str = "") -> RouteDecision:
    """Classify a user query into a route decision.

    Args:
        query: The current user turn's transcribed text.
        conversation_history: Recent transcript (see
            ``ConversationState.recent_history_text``), given as context
            so pronouns/follow-ups route correctly.

    Returns:
        The router's ``RouteDecision``, with confidence policy already
        enforced (see ``enforce_confidence_threshold``).
    """
    settings = get_settings()
    prompt = query
    if conversation_history:
        prompt = f"Conversation so far:\n{conversation_history}\n\nCurrent user request:\n{query}"

    result = await _get_router_agent().run(prompt)
    threshold = settings.router_min_confidence or MIN_CONFIDENT_ROUTE_SCORE
    return enforce_confidence_threshold(result.output, threshold)

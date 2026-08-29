"""General-purpose fallback agent for non-facility conversation.

Handles the "General conversation" -> "General LLM" route from the
brief's routing table: greetings, small talk, or anything that isn't
actually about facility operations. Deliberately has no tools - if the
router sends something here that turns out to need data or documents,
that is a routing miss to be caught by evaluation, not something this
agent should try to route around itself.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from nectar_agent.config import get_settings

GENERAL_SYSTEM_PROMPT = """\
You are the voice assistant for Nectar's Intelligent Facilities \
Platform. Respond briefly and naturally to general conversation or \
greetings. If the user asks something that sounds like it is actually \
about facility operations (assets, HVAC, energy, maintenance), say you \
did not quite catch what they needed and ask them to rephrase, rather \
than guessing at facility data you were not given.
"""


@lru_cache
def build_general_agent() -> Agent:
    """Build (once) and return the general-conversation fallback agent.

    Built lazily rather than at import time - see
    ``orchestration/router.py`` for why.
    """
    settings = get_settings()
    return Agent(settings.llm_model_reasoning, system_prompt=GENERAL_SYSTEM_PROMPT)


async def respond(query: str) -> str:
    """Produce a short general-conversation response."""
    result = await build_general_agent().run(query)
    return result.output

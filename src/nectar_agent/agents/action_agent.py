"""MCP-based investigation & action-proposal agent (Task 4).

This agent investigates facility issues using MCP *read* tools only -
its MCP client subprocess is launched with ``MCP_ALLOW_ACTIONS=false``
(see ``mcp_server/server.py``), so ``create_service_request`` and
``update_service_request`` are not even present in its tool list. It can
never call a write tool, regardless of what the LLM decides to do.

Instead of calling write tools, this agent's structured output is an
``ActionRecommendation``: whether a service request is warranted and,
if so, the proposed asset/summary. The orchestrator is responsible for
turning an affirmative recommendation into an actual confirmation
request (``orchestration/confirmation.py``), and only that module - via
a separate, actions-enabled MCP client - ever invokes the write tools.
This keeps "decide" and "execute" as two physically separate code paths.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset, StdioTransport

from nectar_agent.config import get_settings
from nectar_agent.prompts.action_agent_prompt import ACTION_AGENT_SYSTEM_PROMPT


class ActionRecommendation(BaseModel):
    """Structured output of the action agent's investigation.

    Attributes:
        findings: Natural-language summary of what was found while
            investigating (status, readings, alerts checked).
        action_warranted: Whether a maintenance service request should
            be proposed to the user.
        asset_id: Asset the proposed action concerns, if any.
        proposed_summary: Proposed service-request summary text, if
            ``action_warranted`` is true.
    """

    findings: str
    action_warranted: bool
    asset_id: str | None = None
    proposed_summary: str | None = None


def _read_only_mcp_toolset() -> MCPToolset:
    """Build a stdio MCP toolset whose server subprocess has no write tools."""
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "nectar_agent.mcp_server.server"],
        env={"MCP_ALLOW_ACTIONS": "false"},
    )
    # pydantic_ai's default 5s MCP handshake timeout can be too tight for
    # this subprocess's import + startup cost on a cold/loaded machine.
    return MCPToolset(transport, init_timeout=30)


@lru_cache
def build_action_agent() -> Agent[None, ActionRecommendation]:
    """Build (once) and return the investigation/action-recommendation agent.

    Built lazily rather than at import time, since ``Agent.__init__``
    eagerly constructs its model client and needs a valid API key to do
    so - see ``orchestration/router.py`` for the same pattern.
    """
    settings = get_settings()
    return Agent(
        settings.llm_model_reasoning,
        output_type=ActionRecommendation,
        system_prompt=ACTION_AGENT_SYSTEM_PROMPT,
        toolsets=[_read_only_mcp_toolset()],
    )


async def investigate(query: str) -> ActionRecommendation:
    """Investigate an issue and recommend whether action is warranted.

    Args:
        query: The user's natural-language request, e.g. "the office on
            the third floor feels very hot, investigate and let me know
            if we need maintenance."

    Returns:
        An ``ActionRecommendation`` describing findings and, if
        warranted, a proposed service request - never an executed one.
    """
    result = await build_action_agent().run(query)
    return result.output

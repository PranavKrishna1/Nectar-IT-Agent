"""Live facility data / analytics agent (Task 2 "Data Agent" route).

Handles requests that need live MCP data - either a single live value
("what is Chiller-01's current temperature") or a summary/aggregate over
live data ("summarize today's energy usage"). Connects to the FastMCP
server defined in ``mcp_server/server.py`` as an MCP client, so tool
calls genuinely go over the Model Context Protocol rather than calling
the Python functions directly - this is what makes the MCP layer a real
integration point rather than an implementation detail.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset, StdioTransport

from nectar_agent.config import get_settings

DATA_AGENT_SYSTEM_PROMPT = """\
You are the facility data agent for Nectar's Intelligent Facilities \
Platform. You answer questions using ONLY the live-data MCP tools \
available to you (asset details/status, sensor data, energy \
consumption, active alerts, asset relationships, location lookup).

Never invent a value. If a tool call returns an error or no data, say \
so plainly rather than guessing a plausible-sounding number. When asked \
to "summarize" or aggregate (e.g. today's energy usage), call the \
relevant tools for every asset/scope in question and combine the \
results into one concise, voice-friendly summary - no markdown, no \
bullet points, no raw JSON.
"""


def _mcp_toolset() -> MCPToolset:
    """Build the stdio MCP toolset pointed at the local FastMCP server.

    Launches ``nectar_agent.mcp_server.server`` as a subprocess speaking
    MCP over stdio - the standard way a Pydantic AI agent consumes an
    MCP tool server without a network hop.

    Returns:
        A configured ``MCPToolset`` instance.

    Raises:
        RuntimeError: If the toolset/transport cannot be constructed.
    """
    try:
        transport = StdioTransport(
            command=sys.executable, args=["-m", "nectar_agent.mcp_server.server"]
        )
        return MCPToolset(transport)
    except Exception as exc:
        raise RuntimeError(f"Failed to build the MCP toolset: {exc}") from exc


@lru_cache
def build_data_agent() -> Agent:
    """Build (once) and return the data agent wired to the facility MCP server.

    Built lazily and cached (rather than at import time) since
    ``Agent.__init__`` eagerly constructs its model client, which
    requires a valid API key to be present - see
    ``orchestration/router.py`` for the same pattern and rationale.

    Returns:
        A Pydantic AI ``Agent`` whose only tools are the MCP server's
        read tools.

    Raises:
        RuntimeError: If the agent cannot be constructed (e.g. missing
            or invalid API key for the configured model).
    """
    try:
        settings = get_settings()
        return Agent(
            settings.llm_model_reasoning,
            system_prompt=DATA_AGENT_SYSTEM_PROMPT,
            toolsets=[_mcp_toolset()],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to build the data agent: {exc}") from exc


async def answer_with_live_data(query: str) -> str:
    """Answer a live-data or data-summary question via MCP tools.

    Args:
        query: The user's natural-language question.

    Returns:
        A concise, voice-friendly answer grounded in live MCP tool
        results.

    Raises:
        RuntimeError: If building the agent or running the query fails
            (e.g. a network error or an LLM API failure).
    """
    try:
        agent = build_data_agent()
        result = await agent.run(query)
        return result.output
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Data agent failed to answer query: {exc}") from exc

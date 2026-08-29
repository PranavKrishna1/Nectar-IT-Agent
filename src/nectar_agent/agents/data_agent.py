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
    """
    transport = StdioTransport(
        command=sys.executable, args=["-m", "nectar_agent.mcp_server.server"]
    )
    # pydantic_ai's default 5s MCP handshake timeout can be too tight for
    # this subprocess's import + startup cost on a cold/loaded machine.
    return MCPToolset(transport, init_timeout=30)


@lru_cache
def build_data_agent() -> Agent:
    """Build (once) and return the data agent wired to the facility MCP server.

    Built lazily rather than at import time - see ``orchestration/router.py``
    for why (``Agent.__init__`` needs a valid API key up front).
    """
    settings = get_settings()
    return Agent(
        settings.llm_model_reasoning,
        system_prompt=DATA_AGENT_SYSTEM_PROMPT,
        toolsets=[_mcp_toolset()],
    )


async def answer_with_live_data(query: str) -> str:
    """Answer a live-data or data-summary question via MCP tools."""
    result = await build_data_agent().run(query)
    return result.output

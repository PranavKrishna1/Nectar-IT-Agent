"""FastMCP server exposing the facility tool layer (Task 4).

This module wires the plain functions in ``tools_read.py`` and
``tools_action.py`` onto a FastMCP application, so they become available
over the Model Context Protocol to any MCP-compatible client - including
the Pydantic AI agents in this project, which connect to it via
``pydantic_ai.mcp.MCPServerStreamableHTTP`` / stdio as configured in
``agents/action_agent.py`` and ``agents/data_agent.py``.

Run standalone with:
    python -m nectar_agent.mcp_server.server

which starts the server over stdio by default (suitable for local
Pydantic AI MCP clients), or set ``MCP_TRANSPORT=http`` to serve over
streamable HTTP on the host/port from ``config.py``.
"""

from __future__ import annotations

import os

from fastmcp import FastMCP

from nectar_agent.config import get_settings
from nectar_agent.mcp_server import tools_action, tools_read

# Whether this server process exposes the write/action tools at all.
# Read this ONCE at import time: since every MCP client in this project
# launches the server as its own stdio subprocess (see
# agents/data_agent.py and agents/action_agent.py), setting
# MCP_ALLOW_ACTIONS=false in the subprocess environment for a given
# client means that client's server instance never registers
# create_service_request/update_service_request at all - the tools are
# physically absent from that MCP session, not merely discouraged by a
# prompt. This backs the confirmation safety requirement with an
# enforcement mechanism instead of relying on the LLM to behave.
_ALLOW_ACTIONS = os.environ.get("MCP_ALLOW_ACTIONS", "true").lower() != "false"

mcp: FastMCP = FastMCP(
    name="nectar-facility-tools",
    instructions=(
        "Tools for reading live facility data (assets, sensors, alerts, "
        "energy) and creating/updating maintenance service requests for "
        "Nectar's Intelligent Facilities Platform. Read tools are always "
        "safe to call. Action tools (create_service_request, "
        "update_service_request) perform real writes and must only be "
        "called after the calling agent has confirmed the action with "
        "the user."
    ),
)

# --- Register read tools -----------------------------------------------
mcp.tool()(tools_read.get_asset_details)
mcp.tool()(tools_read.get_asset_status)
mcp.tool()(tools_read.get_sensor_data)
mcp.tool()(tools_read.get_energy_consumption)
mcp.tool()(tools_read.get_active_alerts)
mcp.tool()(tools_read.get_asset_relationships)
mcp.tool()(tools_read.find_assets_by_location)

# --- Register action (write) tools, unless explicitly disabled ---------
if _ALLOW_ACTIONS:
    mcp.tool()(tools_action.create_service_request)
    mcp.tool()(tools_action.update_service_request)


def main() -> None:
    """Entry point for running the MCP server as a standalone process.

    Reads the ``MCP_TRANSPORT`` environment variable to decide between
    stdio (default) and streamable-HTTP transport, and uses the
    host/port from application settings for the HTTP case.

    Raises:
        RuntimeError: If settings fail to load or the server fails to
            start, wrapping the underlying error with context about
            which transport was being started.
    """
    try:
        settings = get_settings()
        transport = os.environ.get("MCP_TRANSPORT", "stdio")
        if transport == "http":
            mcp.run(
                transport="streamable-http",
                host=settings.mcp_server_host,
                port=settings.mcp_server_port,
            )
        else:
            mcp.run(transport="stdio")
    except Exception as exc:
        raise RuntimeError(f"Failed to start the MCP server: {exc}") from exc


if __name__ == "__main__":
    main()

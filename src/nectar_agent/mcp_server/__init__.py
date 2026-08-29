"""FastMCP-based tool server exposing live facility data and actions.

Submodules:
    mock_facility_data: In-memory simulated facility dataset.
    tools_read: Read-only tool implementations (assets, sensors, alerts, energy).
    tools_action: Write/action tool implementations (service requests).
    server: FastMCP application wiring the above into an MCP server.
"""

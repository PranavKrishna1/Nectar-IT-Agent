"""Read-only MCP tool implementations (Task 4).

Each function here is a plain, independently-testable Python function.
``server.py`` registers them onto the FastMCP app with ``@mcp.tool()`` so
they are exposed over the MCP protocol; keeping the implementations here
(rather than defining them inline as decorated functions) means they can
also be unit-tested directly, without spinning up an MCP client/server
round-trip.

All functions return Pydantic models or plain JSON-serializable
structures so a calling LLM/agent always receives predictable, typed
data rather than ad-hoc strings.
"""

from __future__ import annotations

from nectar_agent.mcp_server import mock_facility_data as data
from nectar_agent.models.domain import AssetType


def get_asset_details(asset_id: str) -> dict:
    """Return static details (type, location, related assets) for one asset.

    Returns an ``error`` key instead if the asset ID isn't recognized or
    the lookup fails unexpectedly.
    """
    try:
        asset = data.get_asset(asset_id)
        if asset is None:
            return {"error": f"No asset found with id '{asset_id}'."}
        return asset.model_dump(mode="json")
    except Exception as exc:
        return {"error": f"Failed to get asset details for '{asset_id}': {exc}"}


def get_asset_status(asset_id: str) -> dict:
    """Return ``{asset_id, status}`` for one asset.

    ``status`` is one of "normal", "warning", "fault", "offline",
    "maintenance". Returns an ``error`` key if the asset is unknown or
    the lookup fails unexpectedly.
    """
    try:
        asset = data.get_asset(asset_id)
        if asset is None:
            return {"error": f"No asset found with id '{asset_id}'."}
        return {"asset_id": asset.asset_id, "status": asset.status.value}
    except Exception as exc:
        return {"error": f"Failed to get asset status for '{asset_id}': {exc}"}


def get_sensor_data(scope_id: str) -> dict:
    """Return the latest sensor readings for an asset or building.

    ``scope_id`` is an asset ID (e.g. "AHU-02") or building name (e.g.
    "Building A"). Returns ``{scope_id, readings}`` where ``readings``
    is a list of ``{sensor_id, metric, value, unit, timestamp}`` objects
    - empty if none are tracked for that scope. Returns an ``error`` key
    instead if the lookup fails unexpectedly.
    """
    try:
        readings = data.get_sensor_readings(scope_id)
        return {
            "scope_id": scope_id,
            "readings": [r.model_dump(mode="json") for r in readings],
        }
    except Exception as exc:
        return {"error": f"Failed to get sensor data for '{scope_id}': {exc}"}


def get_energy_consumption(scope_id: str) -> dict:
    """Return trailing 24-hour energy consumption for an asset or building.

    Includes consumption in kWh, the baseline, and the computed
    percentage over baseline. Returns an ``error`` key if no energy data
    is tracked for ``scope_id`` or the lookup fails unexpectedly.
    """
    try:
        energy = data.get_energy(scope_id)
        if energy is None:
            return {"error": f"No energy data tracked for '{scope_id}'."}
        return {
            "scope_id": energy.scope_id,
            "kwh": energy.kwh,
            "baseline_kwh": energy.baseline_kwh,
            "percent_over_baseline": energy.percent_over_baseline,
            "period_start": energy.period_start.isoformat(),
            "period_end": energy.period_end.isoformat(),
        }
    except Exception as exc:
        return {"error": f"Failed to get energy consumption for '{scope_id}': {exc}"}


def get_active_alerts(asset_id: str | None = None) -> dict:
    """Return currently active alerts, optionally scoped to one asset.

    If ``asset_id`` is omitted, all active alerts facility-wide are
    returned. Result is ``{alerts}``, a list of ``{alert_id, asset_id,
    severity, message, raised_at}`` objects.
    """
    try:
        alerts = data.get_alerts(asset_id=asset_id, active_only=True)
        return {"alerts": [a.model_dump(mode="json") for a in alerts]}
    except Exception as exc:
        return {"error": f"Failed to get active alerts for '{asset_id}': {exc}"}


def get_asset_relationships(asset_id: str) -> dict:
    """Return the IDs of assets directly related to the given asset.

    Used by the orchestrator to traverse from one asset to related ones
    during multi-step investigation, e.g. from a chiller to the AHUs it
    serves. Result is ``{asset_id, related_asset_ids}``.
    """
    try:
        asset = data.get_asset(asset_id)
        if asset is None:
            return {"error": f"No asset found with id '{asset_id}'."}
        return {"asset_id": asset.asset_id, "related_asset_ids": asset.related_asset_ids}
    except Exception as exc:
        return {"error": f"Failed to get asset relationships for '{asset_id}': {exc}"}


def find_assets_by_location(building: str, asset_type: str | None = None) -> dict:
    """Find assets by building (and optionally asset type).

    This supplements the six tools named explicitly in the brief and is
    what lets the orchestrator resolve a natural-language location
    ("the office on the third floor" / "Building A") into concrete
    asset IDs before calling the other tools. Returns ``{assets}``, or
    an ``error`` key if ``asset_type`` is unrecognized.
    """
    try:
        parsed_type: AssetType | None = None
        if asset_type:
            try:
                parsed_type = AssetType(asset_type.lower())
            except ValueError:
                return {"error": f"Unknown asset_type '{asset_type}'."}
        assets = data.find_assets(building=building, asset_type=parsed_type)
        return {"assets": [a.model_dump(mode="json") for a in assets]}
    except Exception as exc:
        return {"error": f"Failed to find assets in '{building}': {exc}"}

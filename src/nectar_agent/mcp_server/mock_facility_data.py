"""In-memory simulated facility dataset backing the MCP tools.

The challenge brief asks for tools that "query live facility data" - in a
take-home prototype without a real BMS/SCADA integration, this module
plays that role. It is intentionally isolated from the tool functions in
``tools_read.py``/``tools_action.py`` so that swapping this file for a
real facility-data client (e.g. a BACnet/Modbus gateway or a REST API)
later requires no changes to the MCP tool interface itself - only this
module's internals would change.

All "get" functions here are synchronous and side-effect free except for
``add_service_request``/``update_service_request_status``, which mutate
the in-memory store the same way a real write API would.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from nectar_agent.models.domain import (
    Alert,
    AlertSeverity,
    Asset,
    AssetStatus,
    AssetType,
    EnergyConsumption,
    SensorReading,
    ServiceRequest,
    ServiceRequestStatus,
)

_NOW = datetime.now(timezone.utc)

# --- Seed assets -----------------------------------------------------------
_ASSETS: dict[str, Asset] = {
    "CHILLER-01": Asset(
        asset_id="CHILLER-01",
        name="Chiller 01",
        asset_type=AssetType.CHILLER,
        building="Building A",
        floor="Rooftop",
        status=AssetStatus.WARNING,
        related_asset_ids=["AHU-02", "AHU-03", "PUMP-01"],
    ),
    "AHU-02": Asset(
        asset_id="AHU-02",
        name="Air Handling Unit 02",
        asset_type=AssetType.AHU,
        building="Building A",
        floor="3",
        status=AssetStatus.FAULT,
        related_asset_ids=["CHILLER-01"],
    ),
    "AHU-03": Asset(
        asset_id="AHU-03",
        name="Air Handling Unit 03",
        asset_type=AssetType.AHU,
        building="Building A",
        floor="4",
        status=AssetStatus.NORMAL,
        related_asset_ids=["CHILLER-01"],
    ),
    "PUMP-01": Asset(
        asset_id="PUMP-01",
        name="Chilled Water Pump 01",
        asset_type=AssetType.PUMP,
        building="Building A",
        floor="Rooftop",
        status=AssetStatus.NORMAL,
        related_asset_ids=["CHILLER-01"],
    ),
}

# --- Seed sensor readings ---------------------------------------------------
_SENSOR_READINGS: dict[str, list[SensorReading]] = {
    "CHILLER-01": [
        SensorReading(
            sensor_id="TEMP-CH01",
            asset_id="CHILLER-01",
            metric="supply_water_temp_c",
            value=7.8,
            unit="C",
            timestamp=_NOW - timedelta(minutes=5),
        ),
        SensorReading(
            sensor_id="POWER-CH01",
            asset_id="CHILLER-01",
            metric="power_draw_kw",
            value=142.0,
            unit="kW",
            timestamp=_NOW - timedelta(minutes=5),
        ),
    ],
    "AHU-02": [
        SensorReading(
            sensor_id="AIRFLOW-AHU02",
            asset_id="AHU-02",
            metric="airflow_cfm",
            value=410.0,
            unit="CFM",
            timestamp=_NOW - timedelta(minutes=3),
        ),
        SensorReading(
            sensor_id="TEMP-AHU02",
            asset_id="AHU-02",
            metric="zone_temp_c",
            value=27.4,
            unit="C",
            timestamp=_NOW - timedelta(minutes=3),
        ),
    ],
    "Building A": [
        SensorReading(
            sensor_id="TEMP-BLDGA-3F",
            asset_id="Building A",
            metric="zone_temp_c",
            value=27.1,
            unit="C",
            timestamp=_NOW - timedelta(minutes=2),
        ),
    ],
}

# --- Seed alerts -------------------------------------------------------------
_ALERTS: dict[str, list[Alert]] = {
    "CHILLER-01": [
        Alert(
            alert_id="ALRT-1001",
            asset_id="CHILLER-01",
            severity=AlertSeverity.WARNING,
            message="Power draw 18% above baseline for current load.",
            raised_at=_NOW - timedelta(hours=1, minutes=10),
            active=True,
        ),
    ],
    "AHU-02": [
        Alert(
            alert_id="ALRT-1002",
            asset_id="AHU-02",
            severity=AlertSeverity.CRITICAL,
            message="Low airflow detected: 410 CFM vs. expected 650 CFM minimum.",
            raised_at=_NOW - timedelta(minutes=42),
            active=True,
        ),
    ],
}

# --- Seed energy data --------------------------------------------------------
_ENERGY: dict[str, EnergyConsumption] = {
    "CHILLER-01": EnergyConsumption(
        scope_id="CHILLER-01",
        period_start=_NOW - timedelta(hours=24),
        period_end=_NOW,
        kwh=3120.0,
        baseline_kwh=2644.0,
    ),
    "Building A": EnergyConsumption(
        scope_id="Building A",
        period_start=_NOW - timedelta(hours=24),
        period_end=_NOW,
        kwh=18450.0,
        baseline_kwh=17200.0,
    ),
}

# --- Service requests (mutable store) ---------------------------------------
_SERVICE_REQUESTS: dict[str, ServiceRequest] = {}


def get_asset(asset_id: str) -> Asset | None:
    """Look up a single asset by ID.

    Args:
        asset_id: Asset identifier, e.g. "AHU-02".

    Returns:
        The matching ``Asset``, or ``None`` if no such asset exists.

    Raises:
        RuntimeError: If the lookup fails unexpectedly (e.g. ``asset_id``
            is not a string and has no ``.upper()``).
    """
    try:
        return _ASSETS.get(asset_id.upper())
    except Exception as exc:
        raise RuntimeError(f"Failed to look up asset '{asset_id}': {exc}") from exc


def find_assets(building: str | None = None, asset_type: AssetType | None = None) -> list[Asset]:
    """Search assets by building and/or type.

    Args:
        building: If given, only assets in this building are returned
            (case-insensitive substring match).
        asset_type: If given, only assets of this type are returned.

    Returns:
        List of matching assets, possibly empty.

    Raises:
        RuntimeError: If filtering fails unexpectedly (e.g. ``building``
            is not a string).
    """
    try:
        results = list(_ASSETS.values())
        if building:
            results = [a for a in results if building.lower() in a.building.lower()]
        if asset_type:
            results = [a for a in results if a.asset_type == asset_type]
        return results
    except Exception as exc:
        raise RuntimeError(f"Failed to search assets: {exc}") from exc


def get_sensor_readings(scope_id: str) -> list[SensorReading]:
    """Return the latest known sensor readings for an asset or building.

    Args:
        scope_id: Asset ID (e.g. "CHILLER-01") or building name
            (e.g. "Building A").

    Returns:
        List of sensor readings for that scope, possibly empty.

    Raises:
        RuntimeError: If the lookup fails unexpectedly.
    """
    try:
        return _SENSOR_READINGS.get(scope_id, [])
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch sensor readings for '{scope_id}': {exc}") from exc


def get_alerts(asset_id: str | None = None, active_only: bool = True) -> list[Alert]:
    """Return alerts, optionally scoped to one asset.

    Args:
        asset_id: If given, only alerts for this asset are returned.
        active_only: If true (default), only currently-open alerts are
            returned.

    Returns:
        List of matching alerts, possibly empty.

    Raises:
        RuntimeError: If the lookup/filtering fails unexpectedly.
    """
    try:
        if asset_id:
            alerts = _ALERTS.get(asset_id.upper(), [])
        else:
            alerts = [a for alerts in _ALERTS.values() for a in alerts]
        if active_only:
            alerts = [a for a in alerts if a.active]
        return alerts
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch alerts for '{asset_id}': {exc}") from exc


def get_energy(scope_id: str) -> EnergyConsumption | None:
    """Return the trailing-24h energy consumption for an asset or building.

    Args:
        scope_id: Asset ID or building name.

    Returns:
        The matching ``EnergyConsumption`` record, or ``None`` if no
        energy data is tracked for that scope.

    Raises:
        RuntimeError: If the lookup fails unexpectedly.
    """
    try:
        return _ENERGY.get(scope_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch energy data for '{scope_id}': {exc}") from exc


def add_service_request(asset_id: str, summary: str) -> ServiceRequest:
    """Create a new service request and persist it in the in-memory store.

    Args:
        asset_id: Asset the request concerns.
        summary: Short description of the issue/work needed.

    Returns:
        The newly created ``ServiceRequest``, including its generated ID.

    Raises:
        RuntimeError: If constructing or persisting the request fails
            unexpectedly (e.g. a validation error on the ``ServiceRequest``
            model).
    """
    try:
        request_id = f"SR-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc)
        request = ServiceRequest(
            request_id=request_id,
            asset_id=asset_id,
            summary=summary,
            status=ServiceRequestStatus.OPEN,
            created_at=now,
            updated_at=now,
        )
        _SERVICE_REQUESTS[request_id] = request
        return request
    except Exception as exc:
        raise RuntimeError(f"Failed to create service request for '{asset_id}': {exc}") from exc


def update_service_request_status(
    request_id: str, status: ServiceRequestStatus
) -> ServiceRequest | None:
    """Update the status of an existing service request.

    Args:
        request_id: ID of the request to update.
        status: New status to set.

    Returns:
        The updated ``ServiceRequest``, or ``None`` if ``request_id`` is
        not found.

    Raises:
        RuntimeError: If updating the request fails unexpectedly.
    """
    try:
        request = _SERVICE_REQUESTS.get(request_id)
        if request is None:
            return None
        request.status = status
        request.updated_at = datetime.now(timezone.utc)
        return request
    except Exception as exc:
        raise RuntimeError(f"Failed to update service request '{request_id}': {exc}") from exc

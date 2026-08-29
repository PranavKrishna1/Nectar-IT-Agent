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
    """Look up a single asset by ID, or ``None`` if it doesn't exist."""
    return _ASSETS.get(asset_id.upper())


def find_assets(building: str | None = None, asset_type: AssetType | None = None) -> list[Asset]:
    """Search assets by building (case-insensitive substring) and/or type."""
    results = list(_ASSETS.values())
    if building:
        results = [a for a in results if building.lower() in a.building.lower()]
    if asset_type:
        results = [a for a in results if a.asset_type == asset_type]
    return results


def get_sensor_readings(scope_id: str) -> list[SensorReading]:
    """Return the latest known sensor readings for an asset or building.

    ``scope_id`` is an asset ID (e.g. "CHILLER-01") or building name
    (e.g. "Building A").
    """
    return _SENSOR_READINGS.get(scope_id, [])


def get_alerts(asset_id: str | None = None, active_only: bool = True) -> list[Alert]:
    """Return alerts, optionally scoped to one asset.

    If ``asset_id`` is omitted, alerts for all assets are returned.
    ``active_only`` (default) restricts to currently-open alerts.
    """
    if asset_id:
        alerts = _ALERTS.get(asset_id.upper(), [])
    else:
        alerts = [a for alerts in _ALERTS.values() for a in alerts]
    if active_only:
        alerts = [a for a in alerts if a.active]
    return alerts


def get_energy(scope_id: str) -> EnergyConsumption | None:
    """Return the trailing-24h energy consumption for an asset or building."""
    return _ENERGY.get(scope_id)


def add_service_request(asset_id: str, summary: str) -> ServiceRequest:
    """Create a new service request and persist it in the in-memory store."""
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


def update_service_request_status(
    request_id: str, status: ServiceRequestStatus
) -> ServiceRequest | None:
    """Update an existing service request's status, or ``None`` if not found."""
    request = _SERVICE_REQUESTS.get(request_id)
    if request is None:
        return None
    request.status = status
    request.updated_at = datetime.now(timezone.utc)
    return request

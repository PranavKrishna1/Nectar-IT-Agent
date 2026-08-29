"""Domain models for facility entities.

These Pydantic models describe the "real world" objects the agent reasons
about: assets (chillers, AHUs, sensors), sensor readings, alerts, and
service requests. They are shared by the MCP tool layer (as return types),
the agents (as reasoning context), and the tests (as fixtures), so a single
definition here keeps every layer of the system in agreement about shape.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AssetType(str, Enum):
    """Category of a physical facility asset."""

    CHILLER = "chiller"
    AHU = "ahu"
    SENSOR = "sensor"
    PUMP = "pump"
    GENERATOR = "generator"


class AssetStatus(str, Enum):
    """Operational status of a facility asset."""

    NORMAL = "normal"
    WARNING = "warning"
    FAULT = "fault"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class Asset(BaseModel):
    """A physical facility asset (chiller, AHU, pump, etc.).

    Attributes:
        asset_id: Stable unique identifier, e.g. "CHILLER-01".
        name: Human-friendly display name.
        asset_type: Category of the asset.
        building: Building the asset is physically located in.
        floor: Floor the asset is located on, if applicable.
        status: Current operational status.
        related_asset_ids: IDs of assets directly related to this one
            (e.g. an AHU served by a chiller), used for relationship
            traversal during multi-step reasoning.
    """

    asset_id: str
    name: str
    asset_type: AssetType
    building: str
    floor: str | None = None
    status: AssetStatus
    related_asset_ids: list[str] = Field(default_factory=list)


class SensorReading(BaseModel):
    """A single point-in-time reading from a facility sensor."""

    sensor_id: str
    asset_id: str
    metric: str
    value: float
    unit: str
    timestamp: datetime


class AlertSeverity(str, Enum):
    """Severity level of a facility alert."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    """An active or historical fault/alert raised against an asset."""

    alert_id: str
    asset_id: str
    severity: AlertSeverity
    message: str
    raised_at: datetime
    active: bool = True


class EnergyConsumption(BaseModel):
    """Energy usage for an asset or building over a period.

    Attributes:
        scope_id: Asset ID or building name the reading applies to.
        kwh: Total energy consumed in kilowatt-hours.
        baseline_kwh: Expected/normal consumption for the same window,
            used to compute percentage-over-baseline figures.
    """

    scope_id: str
    period_start: datetime
    period_end: datetime
    kwh: float
    baseline_kwh: float | None = None

    @property
    def percent_over_baseline(self) -> float | None:
        """How far ``kwh`` is above ``baseline_kwh``, as a percentage.

        ``None`` if there's no baseline to compare against (including a
        zero baseline, which would otherwise divide by zero).
        """
        if not self.baseline_kwh:
            return None
        return round(((self.kwh - self.baseline_kwh) / self.baseline_kwh) * 100, 1)


class ServiceRequestStatus(str, Enum):
    """Lifecycle status of a maintenance service request."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class ServiceRequest(BaseModel):
    """A maintenance/service ticket created or updated by the action agent."""

    request_id: str
    asset_id: str
    summary: str
    status: ServiceRequestStatus = ServiceRequestStatus.OPEN
    created_at: datetime
    updated_at: datetime

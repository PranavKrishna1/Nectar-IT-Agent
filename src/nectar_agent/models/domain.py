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
    """A single point-in-time reading from a facility sensor.

    Attributes:
        sensor_id: Identifier of the sensor that produced the reading.
        asset_id: Asset the sensor is attached to or monitors.
        metric: Name of the measured quantity, e.g. "temperature_c".
        value: Numeric value of the reading.
        unit: Unit of measurement, e.g. "C", "%", "kW".
        timestamp: When the reading was captured.
    """

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
    """An active or historical fault/alert raised against an asset.

    Attributes:
        alert_id: Unique identifier for the alert.
        asset_id: Asset the alert was raised against.
        severity: How urgent the alert is.
        message: Human-readable description of the alert.
        raised_at: When the alert was first raised.
        active: Whether the alert is still open.
    """

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
        period_start: Start of the measurement window.
        period_end: End of the measurement window.
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
        """Return how far ``kwh`` is above ``baseline_kwh`` as a percentage.

        Returns:
            The percentage above baseline (e.g. 18.0 for 18% over), or
            ``None`` if no baseline is available to compare against, or
            if the computation fails unexpectedly.

        Raises:
            None: Errors are caught internally; this property degrades
                to ``None`` rather than raising, since callers treat a
                missing baseline and a failed computation the same way.
        """
        try:
            if not self.baseline_kwh:
                return None
            return round(((self.kwh - self.baseline_kwh) / self.baseline_kwh) * 100, 1)
        except (TypeError, ZeroDivisionError, ArithmeticError):
            return None


class ServiceRequestStatus(str, Enum):
    """Lifecycle status of a maintenance service request."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class ServiceRequest(BaseModel):
    """A maintenance/service ticket created or updated by the action agent.

    Attributes:
        request_id: Unique identifier assigned when the request is created.
        asset_id: Asset the request concerns.
        summary: Short description of the issue or work needed.
        status: Current lifecycle status.
        created_at: When the request was created.
        updated_at: When the request was last updated.
    """

    request_id: str
    asset_id: str
    summary: str
    status: ServiceRequestStatus = ServiceRequestStatus.OPEN
    created_at: datetime
    updated_at: datetime

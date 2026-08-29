"""Unit tests for the MCP read/action tool implementations.

These exercise the plain functions in ``mcp_server/tools_read.py`` and
``mcp_server/tools_action.py`` directly against the seeded in-memory
mock dataset - no LLM, MCP transport, or network access required, so
these run fast and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nectar_agent.mcp_server import tools_action, tools_read


def test_get_asset_details_known_asset() -> None:
    result = tools_read.get_asset_details("CHILLER-01")
    assert result["asset_id"] == "CHILLER-01"
    assert result["asset_type"] == "chiller"


def test_get_asset_details_unknown_asset_returns_error() -> None:
    result = tools_read.get_asset_details("DOES-NOT-EXIST")
    assert "error" in result


def test_get_asset_status_returns_expected_shape() -> None:
    result = tools_read.get_asset_status("AHU-02")
    assert result == {"asset_id": "AHU-02", "status": "fault"}


def test_get_sensor_data_returns_readings_list() -> None:
    result = tools_read.get_sensor_data("AHU-02")
    assert result["scope_id"] == "AHU-02"
    assert len(result["readings"]) >= 1
    assert result["readings"][0]["metric"] in {"airflow_cfm", "zone_temp_c"}


def test_get_energy_consumption_computes_percent_over_baseline() -> None:
    result = tools_read.get_energy_consumption("CHILLER-01")
    assert result["percent_over_baseline"] is not None
    assert result["percent_over_baseline"] > 0


def test_get_active_alerts_scoped_to_asset() -> None:
    result = tools_read.get_active_alerts(asset_id="AHU-02")
    assert len(result["alerts"]) == 1
    assert result["alerts"][0]["severity"] == "critical"


def test_get_asset_relationships() -> None:
    result = tools_read.get_asset_relationships("CHILLER-01")
    assert "AHU-02" in result["related_asset_ids"]


def test_find_assets_by_location_filters_by_building_and_type() -> None:
    result = tools_read.find_assets_by_location(building="Building A", asset_type="ahu")
    ids = {a["asset_id"] for a in result["assets"]}
    assert ids == {"AHU-02", "AHU-03"}


def test_create_service_request_then_update_it() -> None:
    created = tools_action.create_service_request(
        asset_id="AHU-02", summary="Low airflow fault investigation."
    )
    assert created["asset_id"] == "AHU-02"
    assert created["status"] == "open"

    updated = tools_action.update_service_request(
        request_id=created["request_id"], status="in_progress"
    )
    assert updated["status"] == "in_progress"


def test_create_service_request_unknown_asset_returns_error() -> None:
    result = tools_action.create_service_request(asset_id="NOPE-99", summary="test")
    assert "error" in result

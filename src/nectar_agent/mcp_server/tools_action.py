"""Write/action MCP tool implementations (Task 4).

These are the only tools in the system that mutate facility state
(creating or updating maintenance tickets). Per the brief's safety
requirement ("the agent must not blindly execute actions"), these
functions are never called directly by the orchestrator - every call is
required to pass through ``orchestration/confirmation.py`` first, which
gates execution behind an explicit user "yes". The functions themselves
stay simple and trust the caller to have already obtained that
confirmation; the safety policy is deliberately centralized in one place
rather than duplicated into every action tool.
"""

from __future__ import annotations

from nectar_agent.mcp_server import mock_facility_data as data
from nectar_agent.models.domain import ServiceRequestStatus


def create_service_request(asset_id: str, summary: str) -> dict:
    """Create a new maintenance service request for an asset.

    Safety: this is a write action. Callers in this codebase must only
    invoke it after the user has explicitly confirmed the action via the
    confirmation flow in ``orchestration/confirmation.py`` - never
    directly from routing or reasoning logic.

    Args:
        asset_id: Asset the request concerns, e.g. "AHU-02".
        summary: Short description of the issue or work needed, e.g.
            "Low airflow fault - possible blocked filter or belt slip."

    Returns:
        A dict describing the created request, including its generated
        ``request_id``, or an ``error`` key if the asset does not exist
        or creation fails unexpectedly.
    """
    try:
        asset = data.get_asset(asset_id)
        if asset is None:
            return {"error": f"Cannot create request: no asset found with id '{asset_id}'."}
        request = data.add_service_request(asset_id=asset_id, summary=summary)
        return request.model_dump(mode="json")
    except Exception as exc:
        return {"error": f"Failed to create service request for '{asset_id}': {exc}"}


def update_service_request(request_id: str, status: str) -> dict:
    """Update the status of an existing service request.

    Safety: this is a write action and, like ``create_service_request``,
    must only be invoked after explicit user confirmation.

    Args:
        request_id: ID of the request to update, e.g. "SR-AB12CD34".
        status: New status - one of "open", "in_progress", "resolved",
            "cancelled".

    Returns:
        A dict describing the updated request, or an ``error`` key if
        the request ID or status value is invalid, or the update fails
        unexpectedly.
    """
    try:
        try:
            parsed_status = ServiceRequestStatus(status.lower())
        except ValueError:
            return {"error": f"Invalid status '{status}'."}
        updated = data.update_service_request_status(request_id, parsed_status)
        if updated is None:
            return {"error": f"No service request found with id '{request_id}'."}
        return updated.model_dump(mode="json")
    except Exception as exc:
        return {"error": f"Failed to update service request '{request_id}': {exc}"}


# Tool names considered "destructive/operational" and therefore always
# subject to the confirmation gate, regardless of the
# `require_confirmation_for_actions` setting's default. Used by
# orchestration/confirmation.py to recognize which calls to intercept.
ACTION_TOOL_NAMES: frozenset[str] = frozenset(
    {"create_service_request", "update_service_request"}
)

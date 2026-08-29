"""Unit tests for the router's confidence-threshold policy.

Tests ``enforce_confidence_threshold`` directly (a pure function) rather
than the full ``route()`` coroutine, so this suite runs without an LLM
API key or network access - see router.py's docstring for why that
logic was factored out separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nectar_agent.models.routing import RouteDecision, RouteType
from nectar_agent.orchestration.router import enforce_confidence_threshold


def test_high_confidence_route_is_kept() -> None:
    decision = RouteDecision(
        route=RouteType.RAG, confidence=0.9, reasoning="Clear knowledge question."
    )
    result = enforce_confidence_threshold(decision, threshold=0.55)
    assert result.route == RouteType.RAG


def test_low_confidence_route_is_coerced_to_clarify() -> None:
    decision = RouteDecision(
        route=RouteType.MCP_ACTION, confidence=0.2, reasoning="Not sure which asset."
    )
    result = enforce_confidence_threshold(decision, threshold=0.55)
    assert result.route == RouteType.CLARIFY
    assert "0.20" in result.reasoning


def test_clarify_route_is_never_further_modified() -> None:
    decision = RouteDecision(
        route=RouteType.CLARIFY, confidence=0.1, reasoning="Ambiguous request."
    )
    result = enforce_confidence_threshold(decision, threshold=0.55)
    assert result is decision

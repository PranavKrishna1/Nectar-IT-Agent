"""Integration-style tests for the orchestrator's per-turn control flow.

Sub-agents and the router are monkeypatched so these tests exercise the
orchestrator's *logic* (routing dispatch, confirmation gating, session
bookkeeping) deterministically and without any LLM/API calls - real
end-to-end behaviour with live models is captured separately in
docs/sample_conversations.md.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nectar_agent.agents.action_agent import ActionRecommendation
from nectar_agent.agents.rag_agent import RagAnswer
from nectar_agent.models.routing import RouteDecision, RouteType
from nectar_agent.orchestration import orchestrator_agent
from nectar_agent.orchestration.session import session_store


@pytest.mark.asyncio
async def test_rag_route_returns_rag_agent_answer(monkeypatch) -> None:
    async def fake_route(query, conversation_history=""):
        return RouteDecision(route=RouteType.RAG, confidence=0.9, reasoning="kb question")

    async def fake_answer_question(query):
        return RagAnswer(answer="Chillers reject heat via a refrigeration cycle.",
                          found_in_knowledge_base=True, source_titles=["Chiller Manual"])

    monkeypatch.setattr(orchestrator_agent.router, "route", fake_route)
    monkeypatch.setattr(orchestrator_agent.rag_agent, "answer_question", fake_answer_question)

    session_id = str(uuid.uuid4())
    response = await orchestrator_agent.handle_turn(session_id, "What is a chiller?")
    assert "refrigeration cycle" in response


@pytest.mark.asyncio
async def test_action_route_proposes_confirmation_without_executing(monkeypatch) -> None:
    async def fake_route(query, conversation_history=""):
        return RouteDecision(route=RouteType.MCP_ACTION, confidence=0.85, reasoning="fault found")

    async def fake_investigate(query):
        return ActionRecommendation(
            findings="AHU-02 shows a low airflow fault.",
            action_warranted=True,
            asset_id="AHU-02",
            proposed_summary="Low airflow fault - check filters/belt.",
        )

    monkeypatch.setattr(orchestrator_agent.router, "route", fake_route)
    monkeypatch.setattr(orchestrator_agent.action_agent, "investigate", fake_investigate)

    session_id = str(uuid.uuid4())
    response = await orchestrator_agent.handle_turn(session_id, "Investigate AHU-02.")

    assert "Would you like me to create a maintenance request" in response
    state = session_store.get_or_create(session_id)
    assert state.pending_confirmation is not None
    assert state.pending_confirmation.action_name == "create_service_request"
    assert state.pending_confirmation.arguments["asset_id"] == "AHU-02"


@pytest.mark.asyncio
async def test_confirmation_yes_executes_the_held_action(monkeypatch) -> None:
    async def fake_route(query, conversation_history=""):
        return RouteDecision(route=RouteType.MCP_ACTION, confidence=0.85, reasoning="fault found")

    async def fake_investigate(query):
        return ActionRecommendation(
            findings="AHU-02 shows a low airflow fault.",
            action_warranted=True,
            asset_id="AHU-02",
            proposed_summary="Low airflow fault - check filters/belt.",
        )

    monkeypatch.setattr(orchestrator_agent.router, "route", fake_route)
    monkeypatch.setattr(orchestrator_agent.action_agent, "investigate", fake_investigate)

    session_id = str(uuid.uuid4())
    await orchestrator_agent.handle_turn(session_id, "Investigate AHU-02.")
    response = await orchestrator_agent.handle_turn(session_id, "Yes, please.")

    assert "Done" in response
    state = session_store.get_or_create(session_id)
    assert state.pending_confirmation is None


@pytest.mark.asyncio
async def test_confirmation_no_cancels_without_executing(monkeypatch) -> None:
    async def fake_route(query, conversation_history=""):
        return RouteDecision(route=RouteType.MCP_ACTION, confidence=0.85, reasoning="fault found")

    async def fake_investigate(query):
        return ActionRecommendation(
            findings="AHU-02 shows a low airflow fault.",
            action_warranted=True,
            asset_id="AHU-02",
            proposed_summary="Low airflow fault.",
        )

    monkeypatch.setattr(orchestrator_agent.router, "route", fake_route)
    monkeypatch.setattr(orchestrator_agent.action_agent, "investigate", fake_investigate)

    session_id = str(uuid.uuid4())
    await orchestrator_agent.handle_turn(session_id, "Investigate AHU-02.")
    response = await orchestrator_agent.handle_turn(session_id, "No, don't.")

    assert "won't take that action" in response
    state = session_store.get_or_create(session_id)
    assert state.pending_confirmation is None


@pytest.mark.asyncio
async def test_clarify_route_asks_instead_of_guessing(monkeypatch) -> None:
    async def fake_route(query, conversation_history=""):
        return RouteDecision(
            route=RouteType.CLARIFY, confidence=0.2, reasoning="ambiguous location"
        )

    monkeypatch.setattr(orchestrator_agent.router, "route", fake_route)

    session_id = str(uuid.uuid4())
    response = await orchestrator_agent.handle_turn(session_id, "It's too hot.")
    assert "which building or asset" in response

"""Top-level autonomous orchestrator (Task 2 + Task 5).

This is the "brain" of the system: for every user turn it decides what
is needed, delegates to the router and the relevant sub-agent(s),
chains additional steps when a result implies more investigation is
required, applies the confirm-before-action safety gate, and produces
one final voice-ready response. Everything else in ``agents/`` and
``mcp_server/`` exists to be called from here.

The autonomous multi-step behaviour is intentionally implemented as
explicit, readable Python control flow (route -> gather -> optionally
gather more -> synthesize) rather than an unconstrained agent loop that
freely re-plans forever. This is a deliberate design choice: it keeps
step count bounded and predictable (`settings.orchestrator_max_steps`
governs sub-agent calls), keeps the confirmation gate impossible to skip
by construction, and keeps the reasoning path auditable for the
evaluation report - at the cost of being less "creative" than a fully
open-ended planner. See docs/design_decisions.md for the trade-off
discussion.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from nectar_agent.agents import action_agent, data_agent, general_agent, rag_agent
from nectar_agent.config import get_settings
from nectar_agent.models.conversation import PendingConfirmation
from nectar_agent.models.routing import RouteDecision, RouteType
from nectar_agent.orchestration import confirmation, router
from nectar_agent.orchestration.session import session_store

_SYNTHESIS_SYSTEM_PROMPT = """\
You combine live facility data findings with retrieved documentation to \
answer a facility operator's question about *why* something is \
happening. Reason step by step internally, then give ONE concise, \
voice-friendly answer (no markdown, no bullet points) that states the \
likely cause and, if relevant, ends by asking whether they would like \
you to take any follow-up action. Do not repeat raw data verbatim - \
synthesize it into plain language.
"""


@lru_cache
def _get_synthesis_agent() -> Agent:
    """Build (once) and return the reasoning agent used to combine RAG + live data.

    Built lazily and cached (rather than at import time) since
    ``Agent.__init__`` eagerly constructs its model client, which
    requires a valid API key to be present - see
    ``orchestration/router.py`` for the same pattern and rationale.

    Returns:
        A tool-less Pydantic AI ``Agent`` used only for text synthesis
        over context already gathered by other agents.
    """
    settings = get_settings()
    return Agent(settings.llm_model_reasoning, system_prompt=_SYNTHESIS_SYSTEM_PROMPT)


async def _handle_pending_confirmation(session_id: str, user_text: str) -> str | None:
    """Check for and resolve a confirmation the previous turn is waiting on.

    Args:
        session_id: Current session ID.
        user_text: The user's current turn text, expected to be a
            yes/no reply to a previously asked confirmation question.

    Returns:
        The response text if a pending confirmation was resolved this
        turn, or ``None`` if there was nothing pending (in which case
        the caller should proceed with normal routing).
    """
    state = session_store.get_or_create(session_id)
    pending = state.pending_confirmation
    if pending is None:
        return None

    decision = confirmation.interpret_confirmation_reply(user_text)
    if decision is None:
        # Ambiguous reply: re-ask rather than guessing either way.
        return f"Sorry, just to confirm - {pending.description} Should I go ahead?"

    session_store.pop_pending_confirmation(session_id)
    if not decision:
        return "Understood, I won't take that action."

    result = confirmation.execute_confirmed_action(pending)
    if "error" in result:
        return f"I tried to take that action but it failed: {result['error']}"
    request_id = result.get("request_id", "")
    return f"Done - I've created service request {request_id} for {pending.description}"


async def _handle_route(
    route_decision: RouteDecision, query: str, conversation_history: str
) -> tuple[str, PendingConfirmation | None]:
    """Dispatch a routed query to the appropriate sub-agent(s).

    Args:
        route_decision: The router's decision for this query.
        query: The user's current query text.
        conversation_history: Recent transcript, for routes that benefit
            from conversational context.

    Returns:
        A ``(response_text, pending_confirmation)`` tuple. ``pending_
        confirmation`` is non-``None`` only when the action route
        determined a maintenance request should be proposed; the caller
        is responsible for attaching it to session state.
    """
    if route_decision.route == RouteType.CLARIFY:
        return (
            "I want to make sure I get this right - could you tell me which "
            "building or asset you mean, and what you'd like me to check?",
            None,
        )

    if route_decision.route == RouteType.RAG:
        answer = await rag_agent.answer_question(query)
        return answer.answer, None

    if route_decision.route in (RouteType.MCP_LIVE_DATA, RouteType.DATA_AGENT):
        return await data_agent.answer_with_live_data(query), None

    if route_decision.route == RouteType.RAG_MCP_REASONING:
        # Multi-step: gather live data AND documentation, then synthesize.
        live_data_findings = await data_agent.answer_with_live_data(query)
        kb_answer = await rag_agent.answer_question(query)
        synthesis_prompt = (
            f"User question: {query}\n\n"
            f"Live facility data findings:\n{live_data_findings}\n\n"
            f"Relevant documentation:\n{kb_answer.answer}\n\n"
            "Combine these into one grounded explanation."
        )
        result = await _get_synthesis_agent().run(synthesis_prompt)
        return result.output, None

    if route_decision.route == RouteType.MCP_ACTION:
        return await _handle_action_route(query)

    return await general_agent.respond(query), None


async def _handle_action_route(query: str) -> tuple[str, PendingConfirmation | None]:
    """Investigate a potential facility issue and propose action if needed.

    Args:
        query: The user's request, e.g. "create a maintenance request
            for AHU-02" or "investigate and let me know if we need
            maintenance."

    Returns:
        A ``(response_text, pending_confirmation)`` tuple. If no action
        is warranted, ``pending_confirmation`` is ``None`` and the text
        is a findings-only response. Otherwise the text is a
        confirmation question and ``pending_confirmation`` describes the
        exact tool call to run if the user says yes.
    """
    recommendation = await action_agent.investigate(query)
    if not recommendation.action_warranted or not recommendation.asset_id:
        return recommendation.findings, None

    pending = PendingConfirmation(
        action_name="create_service_request",
        arguments={
            "asset_id": recommendation.asset_id,
            "summary": recommendation.proposed_summary or recommendation.findings,
        },
        description=f"a maintenance request for {recommendation.asset_id}",
    )
    response = (
        f"{recommendation.findings} Would you like me to create a maintenance "
        f"request for {recommendation.asset_id}?"
    )
    return response, pending


async def handle_turn(session_id: str, user_text: str) -> str:
    """Process one user voice turn end-to-end and return the agent's reply.

    This is the single entry point ``main.py`` / ``scripts/run_demo.py``
    call per turn. It: (1) resolves any pending confirmation first,
    (2) otherwise routes the request and delegates to the right
    sub-agent(s), chaining RAG + live data when the route calls for it,
    (3) turns an action recommendation into a held confirmation rather
    than executing anything, and (4) records both turns to session
    history before returning.

    Args:
        session_id: Identifier for the ongoing voice conversation.
        user_text: The transcribed user utterance for this turn.

    Returns:
        The agent's natural-language response text, ready to be passed
        to ``voice.tts.synthesize``.
    """
    session_store.record_user_turn(session_id, user_text)
    state = session_store.get_or_create(session_id)

    confirmation_response = await _handle_pending_confirmation(session_id, user_text)
    if confirmation_response is not None:
        session_store.record_agent_turn(session_id, confirmation_response)
        return confirmation_response

    history_text = state.recent_history_text()
    route_decision = await router.route(user_text, conversation_history=history_text)
    response_text, pending = await _handle_route(route_decision, user_text, history_text)

    if pending is not None:
        session_store.set_pending_confirmation(session_id, pending)

    session_store.record_agent_turn(session_id, response_text)
    return response_text

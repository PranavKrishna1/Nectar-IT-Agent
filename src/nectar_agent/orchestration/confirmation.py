"""Confirm-before-action safety gate (Task 4 safety requirement).

This module is the ONLY place in the codebase authorized to invoke the
write tools in ``mcp_server/tools_action.py``. The action agent
(``agents/action_agent.py``) cannot reach them at all - its MCP
subprocess is launched without them registered. The orchestrator can
only trigger a write by going through ``execute_confirmed_action`` here,
after ``interpret_confirmation_reply`` has confirmed the user actually
said yes. Concentrating the decision in one small, reviewable function
is deliberate: it is much easier to audit "does this one function ever
get called without a prior affirmative" than to audit that constraint
across every prompt in the system.
"""

from __future__ import annotations

from nectar_agent.mcp_server import tools_action
from nectar_agent.models.conversation import PendingConfirmation

_AFFIRMATIVE_PHRASES = {
    "yes", "yes please", "yeah", "yep", "sure", "go ahead", "please do",
    "confirm", "confirmed", "do it", "correct", "affirmative", "ok", "okay",
}
_NEGATIVE_PHRASES = {
    "no", "nope", "don't", "do not", "cancel", "negative", "not now", "stop",
}

_ACTION_TOOLS = {
    "create_service_request": tools_action.create_service_request,
    "update_service_request": tools_action.update_service_request,
}


def interpret_confirmation_reply(user_text: str) -> bool | None:
    """Interpret a user's reply to a pending confirmation question.

    Returns:
        ``True``/``False`` for a clear yes/no, or ``None`` if the reply
        is ambiguous - the orchestrator should re-ask rather than guess.
    """
    normalized = user_text.strip().lower().rstrip(".!")
    if normalized in _AFFIRMATIVE_PHRASES:
        return True
    if normalized in _NEGATIVE_PHRASES:
        return False
    if any(normalized.startswith(p) for p in _AFFIRMATIVE_PHRASES):
        return True
    if any(normalized.startswith(p) for p in _NEGATIVE_PHRASES):
        return False
    return None


def execute_confirmed_action(confirmation: PendingConfirmation) -> dict:
    """Execute a previously-proposed action after the user has confirmed it.

    Raises:
        ValueError: If ``confirmation.action_name`` isn't a recognized
            action tool - unreachable in normal operation, so this is a
            loud signal of a bug upstream rather than something to hide.
    """
    action = _ACTION_TOOLS.get(confirmation.action_name)
    if action is None:
        raise ValueError(f"Unrecognized action tool: {confirmation.action_name}")
    return action(**confirmation.arguments)

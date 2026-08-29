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


def interpret_confirmation_reply(user_text: str) -> bool | None:
    """Interpret a user's reply to a pending confirmation question.

    Args:
        user_text: The user's transcribed reply.

    Returns:
        ``True`` if the reply reads as an affirmative confirmation,
        ``False`` if it reads as a decline, or ``None`` if the reply is
        ambiguous and the orchestrator should ask again rather than
        assume either way. Also returns ``None`` (treat as ambiguous)
        if interpretation fails unexpectedly, since re-asking is always
        the safe fallback for a safety gate like this one.

    Raises:
        None: Failures are caught internally and degrade to ``None``.
    """
    try:
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
    except Exception:
        return None


def execute_confirmed_action(confirmation: PendingConfirmation) -> dict:
    """Execute a previously-proposed action after the user has confirmed it.

    Args:
        confirmation: The ``PendingConfirmation`` that was held on the
            session, describing which action tool to call and with what
            arguments.

    Returns:
        The result dict returned by the underlying action tool.

    Raises:
        ValueError: If ``confirmation.action_name`` is not a recognized
            action tool - this should be unreachable in normal operation
            and indicates a bug upstream, so it is raised loudly rather
            than silently ignored.
        RuntimeError: If the underlying tool call itself fails
            unexpectedly for any other reason.
    """
    try:
        if confirmation.action_name == "create_service_request":
            return tools_action.create_service_request(**confirmation.arguments)
        if confirmation.action_name == "update_service_request":
            return tools_action.update_service_request(**confirmation.arguments)
        raise ValueError(f"Unrecognized action tool: {confirmation.action_name}")
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to execute confirmed action '{confirmation.action_name}': {exc}"
        ) from exc

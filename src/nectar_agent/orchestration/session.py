"""Conversation/session state management (Task 1 "maintaining context").

Keeps a per-session, in-memory ``ConversationState`` so multi-turn voice
conversations retain history, resolved entities, and any action pending
user confirmation. A process-local dict is sufficient for this
prototype; swapping in Redis/a database later only requires changing
``SessionStore``'s internals, not any caller.
"""

from __future__ import annotations

from nectar_agent.models.conversation import ConversationState, PendingConfirmation, Speaker, Turn


class SessionStore:
    """In-memory store of active conversation sessions, keyed by session ID."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str) -> ConversationState:
        """Fetch a session's state, creating a fresh one if it doesn't exist."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)
        return self._sessions[session_id]

    def record_user_turn(self, session_id: str, text: str) -> ConversationState:
        """Append a user turn to a session's history."""
        state = self.get_or_create(session_id)
        state.turns.append(Turn(speaker=Speaker.USER, text=text))
        return state

    def record_agent_turn(self, session_id: str, text: str) -> ConversationState:
        """Append an agent turn to a session's history."""
        state = self.get_or_create(session_id)
        state.turns.append(Turn(speaker=Speaker.AGENT, text=text))
        return state

    def set_pending_confirmation(
        self, session_id: str, confirmation: PendingConfirmation
    ) -> None:
        """Record an action awaiting the user's yes/no."""
        self.get_or_create(session_id).pending_confirmation = confirmation

    def pop_pending_confirmation(self, session_id: str) -> PendingConfirmation | None:
        """Retrieve and clear any pending confirmation for a session."""
        state = self.get_or_create(session_id)
        pending = state.pending_confirmation
        state.pending_confirmation = None
        return pending

    def update_entities(self, session_id: str, entities: dict[str, str]) -> None:
        """Merge newly-resolved entities into a session's active-entity slots."""
        self.get_or_create(session_id).active_entities.update(entities)


# Module-level singleton: one store shared across the process, mirroring
# how a single voice-gateway process would hold all active call sessions.
session_store = SessionStore()

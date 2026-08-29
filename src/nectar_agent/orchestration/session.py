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
    """In-memory store of active conversation sessions.

    Attributes:
        _sessions: Mapping of session ID to its ``ConversationState``.
    """

    def __init__(self) -> None:
        """Initialize an empty session store.

        Raises:
            RuntimeError: If the internal store cannot be initialized.
        """
        try:
            self._sessions: dict[str, ConversationState] = {}
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize the session store: {exc}") from exc

    def get_or_create(self, session_id: str) -> ConversationState:
        """Fetch a session's state, creating a fresh one if it doesn't exist.

        Args:
            session_id: Unique identifier for the conversation session
                (e.g. a device/call ID).

        Returns:
            The session's ``ConversationState``.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)
        return self._sessions[session_id]

    def record_user_turn(self, session_id: str, text: str) -> ConversationState:
        """Append a user turn to a session's history.

        Args:
            session_id: Session to update.
            text: Transcribed user utterance.

        Returns:
            The updated ``ConversationState``.
        """
        state = self.get_or_create(session_id)
        state.turns.append(Turn(speaker=Speaker.USER, text=text))
        return state

    def record_agent_turn(self, session_id: str, text: str) -> ConversationState:
        """Append an agent turn to a session's history.

        Args:
            session_id: Session to update.
            text: The agent's generated response text.

        Returns:
            The updated ``ConversationState``.
        """
        state = self.get_or_create(session_id)
        state.turns.append(Turn(speaker=Speaker.AGENT, text=text))
        return state

    def set_pending_confirmation(
        self, session_id: str, confirmation: PendingConfirmation
    ) -> None:
        """Record an action awaiting the user's yes/no.

        Args:
            session_id: Session to update.
            confirmation: The proposed action and its description.
        """
        self.get_or_create(session_id).pending_confirmation = confirmation

    def pop_pending_confirmation(self, session_id: str) -> PendingConfirmation | None:
        """Retrieve and clear any pending confirmation for a session.

        Args:
            session_id: Session to check.

        Returns:
            The ``PendingConfirmation`` that was pending, or ``None`` if
            there was none.
        """
        state = self.get_or_create(session_id)
        pending = state.pending_confirmation
        state.pending_confirmation = None
        return pending

    def update_entities(self, session_id: str, entities: dict[str, str]) -> None:
        """Merge newly-resolved entities into a session's active-entity slots.

        Args:
            session_id: Session to update.
            entities: New entity key/value pairs to remember, e.g.
                ``{"asset_id": "AHU-02"}``.
        """
        self.get_or_create(session_id).active_entities.update(entities)


# Module-level singleton: one store shared across the process, mirroring
# how a single voice-gateway process would hold all active call sessions.
session_store = SessionStore()

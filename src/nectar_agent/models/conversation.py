"""Models for conversation/session state management.

A voice conversation spans multiple turns ("the temperature in Building A
is high" -> "yes, please create a request"), so the agent needs a place to
keep short-term memory: prior turns, pending confirmations, and entities
already resolved (like which asset "it" refers to). These models define
that state; ``orchestration/session.py`` is what actually stores and
mutates it across turns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Speaker(str, Enum):
    """Who produced a given conversation turn."""

    USER = "user"
    AGENT = "agent"


class ToolCallRecord(BaseModel):
    """Record of a single tool or sub-agent invocation made while
    answering one user turn.

    Kept for transparency (used in the evaluation report and sample
    conversation transcripts) and so the orchestrator can inspect what
    has already been gathered before deciding whether another step is
    needed.

    Attributes:
        tool_name: Name of the MCP tool or sub-agent that was called.
        arguments: Arguments passed to the call.
        result_summary: Short summary of what the call returned.
    """

    tool_name: str
    arguments: dict = Field(default_factory=dict)
    result_summary: str


class Turn(BaseModel):
    """One utterance in the conversation, from either party.

    Attributes:
        speaker: Whether the user or the agent produced this turn.
        text: The transcribed/generated text content.
        timestamp: When the turn occurred.
        tool_calls: Tool/sub-agent calls made while producing this turn
            (empty for user turns).
    """

    speaker: Speaker
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class PendingConfirmation(BaseModel):
    """A proposed action awaiting explicit user confirmation.

    Set by ``orchestration/confirmation.py`` whenever the action agent
    wants to run a write tool (e.g. ``create_service_request``). The next
    user turn is checked against this before anything else: if it reads
    as an affirmative, the held action is executed; otherwise it is
    discarded and the turn is routed normally.

    Attributes:
        action_name: Name of the MCP action tool to run if confirmed.
        arguments: Arguments to call it with.
        description: Natural-language description of the action, used to
            phrase the confirmation question to the user.
    """

    action_name: str
    arguments: dict
    description: str


class ConversationState(BaseModel):
    """Full state of one voice conversation session.

    Attributes:
        session_id: Unique identifier for the session.
        turns: Ordered history of all turns so far, oldest first.
        pending_confirmation: An action awaiting the user's yes/no, or
            ``None`` if nothing is pending.
        active_entities: Lightweight slot-memory of recently discussed
            entities (e.g. {"asset_id": "AHU-02", "building": "Building A"})
            used to resolve pronouns/ellipsis like "check on it again".
    """

    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    pending_confirmation: PendingConfirmation | None = None
    active_entities: dict[str, str] = Field(default_factory=dict)

    def recent_history_text(self, max_turns: int = 6) -> str:
        """Render the most recent turns as plain text for LLM context.

        Args:
            max_turns: Maximum number of most-recent turns to include.

        Returns:
            A newline-separated transcript, e.g. "user: ...\\nagent: ...",
            suitable for dropping into a prompt as conversation history.
            Returns an empty string if rendering fails unexpectedly,
            rather than raising, so a history-formatting bug can never
            crash the turn that was trying to use it as context.

        Raises:
            None: Failures are caught and degrade to ``""``.
        """
        try:
            recent = self.turns[-max_turns:]
            return "\n".join(f"{t.speaker.value}: {t.text}" for t in recent)
        except Exception:
            return ""

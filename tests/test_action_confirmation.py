"""Unit tests for the confirm-before-action safety gate.

Verifies (1) the affirmative/negative reply interpretation used to
decide whether a held action should fire, and (2) that
``execute_confirmed_action`` correctly dispatches to the right
underlying tool - the only two things worth testing here in isolation,
since the "never call write tools except through this module" guarantee
is architectural (see confirmation.py's module docstring) rather than
something a unit test can prove.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nectar_agent.models.conversation import PendingConfirmation
from nectar_agent.orchestration import confirmation


def test_interpret_confirmation_reply_affirmative() -> None:
    assert confirmation.interpret_confirmation_reply("Yes, please go ahead.") is True
    assert confirmation.interpret_confirmation_reply("sure") is True
    assert confirmation.interpret_confirmation_reply("Okay") is True


def test_interpret_confirmation_reply_negative() -> None:
    assert confirmation.interpret_confirmation_reply("No, don't.") is False
    assert confirmation.interpret_confirmation_reply("cancel that") is False


def test_interpret_confirmation_reply_ambiguous_returns_none() -> None:
    assert confirmation.interpret_confirmation_reply("what time is it") is None


def test_execute_confirmed_action_creates_service_request() -> None:
    pending = PendingConfirmation(
        action_name="create_service_request",
        arguments={"asset_id": "AHU-02", "summary": "Low airflow investigation."},
        description="a maintenance request for AHU-02",
    )
    result = confirmation.execute_confirmed_action(pending)
    assert result["asset_id"] == "AHU-02"
    assert "request_id" in result


def test_execute_confirmed_action_rejects_unknown_action_name() -> None:
    pending = PendingConfirmation(
        action_name="delete_everything",
        arguments={},
        description="something unsafe",
    )
    try:
        confirmation.execute_confirmed_action(pending)
        assert False, "expected ValueError"
    except ValueError:
        pass

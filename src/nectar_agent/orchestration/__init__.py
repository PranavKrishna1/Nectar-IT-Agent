"""Routing, session state, confirmation gating, and top-level orchestration.

Submodules:
    router: LLM-based intent classification into a ``RouteDecision`` (Task 2).
    session: Per-session conversation state store (Task 1 context).
    confirmation: The sole authorized path to invoke write/action MCP tools.
    orchestrator_agent: Top-level autonomous per-turn handler (Task 5),
        the single entry point the voice loop calls.
"""

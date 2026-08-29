"""Specialized Pydantic AI agents, one per route destination.

Submodules:
    rag_agent: Answers from the documentation knowledge base (Task 3).
    data_agent: Answers/summarizes from live MCP data (Task 2 data route).
    action_agent: Investigates via read-only MCP tools and recommends
        (never executes) maintenance actions (Task 4).
    general_agent: Tool-less fallback for general conversation.

The top-level autonomous loop that decides which of these to call, in
what order, and how many times, lives in
``orchestration/orchestrator_agent.py`` - these modules are the
"workers," not the "brain."
"""

"""System prompt templates, one module per agent.

Kept as plain string constants in their own modules (rather than inline
in each agent file) so prompt text can be reviewed, diffed, and iterated
on independently of agent wiring/tool logic.
"""

from nectar_agent.prompts.action_agent_prompt import ACTION_AGENT_SYSTEM_PROMPT
from nectar_agent.prompts.orchestrator_prompt import ORCHESTRATOR_SYSTEM_PROMPT
from nectar_agent.prompts.rag_prompt import RAG_SYSTEM_PROMPT
from nectar_agent.prompts.router_prompt import ROUTER_SYSTEM_PROMPT

__all__ = [
    "ACTION_AGENT_SYSTEM_PROMPT",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "RAG_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
]

"""RAG-based facility knowledge agent (Task 3).

Wraps retrieval (``rag/retriever.py`` + ``rag/reranker.py``) and a
Pydantic AI ``Agent`` that is instructed to answer strictly from the
retrieved excerpts it is given as a tool result - never from its own
general knowledge - and to say so plainly when the knowledge base does
not cover the question.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from nectar_agent.config import get_settings
from nectar_agent.prompts.rag_prompt import RAG_SYSTEM_PROMPT
from nectar_agent.rag import reranker, retriever


@dataclass
class RagDeps:
    """Runtime dependencies injected into the RAG agent's tool calls.

    Attributes:
        query: The user's original question, used by the retrieval tool
            and for the lexical-overlap reranking step.
    """

    query: str


class RagAnswer(BaseModel):
    """Structured output of the RAG agent.

    Attributes:
        answer: The grounded, voice-friendly answer text.
        found_in_knowledge_base: Whether sufficient supporting context
            was found (false triggers the "not found" phrasing upstream).
        source_titles: Titles of the documents the answer was grounded in.
    """

    answer: str
    found_in_knowledge_base: bool
    source_titles: list[str] = []


@lru_cache
def _get_agent() -> Agent[RagDeps, RagAnswer]:
    """Build (once) and return the RAG agent with its retrieval tool.

    Built lazily rather than at import time - see
    ``orchestration/router.py`` for why.
    """
    settings = get_settings()
    agent: Agent[RagDeps, RagAnswer] = Agent(
        settings.llm_model_reasoning,
        deps_type=RagDeps,
        output_type=RagAnswer,
        system_prompt=RAG_SYSTEM_PROMPT,
    )

    @agent.tool
    def search_knowledge_base(ctx: RunContext[RagDeps]) -> str:
        """Retrieve and rerank knowledge-base excerpts relevant to the query.

        Returns an explicit "no relevant documents found" marker if
        nothing clears the similarity threshold, or if retrieval itself
        fails - treated the same way so the RAG agent can still answer
        honestly instead of the whole turn crashing.
        """
        try:
            matches = retriever.retrieve(ctx.deps.query)
            if not matches:
                return "NO_RELEVANT_DOCUMENTS_FOUND"
            top = reranker.rerank(ctx.deps.query, matches)
            blocks = [
                f"[Source: {chunk.title} ({chunk.source_type})]\n{chunk.text}" for chunk in top
            ]
            return "\n\n---\n\n".join(blocks)
        except Exception:
            return "NO_RELEVANT_DOCUMENTS_FOUND"

    return agent


async def answer_question(query: str) -> RagAnswer:
    """Answer a facility-knowledge question using the RAG pipeline.

    Returns:
        A ``RagAnswer`` grounded in retrieved documentation, or one with
        ``found_in_knowledge_base=False`` and an honest "not found"
        answer if the knowledge base doesn't cover the question.
    """
    result = await _get_agent().run(query, deps=RagDeps(query=query))
    return result.output

# Design Decisions & Assumptions

## Why the orchestrator is explicit control flow, not an open-ended planner

`orchestration/orchestrator_agent.py` implements the autonomous multi-step behaviour as readable Python (`route → gather → gather more if the route calls for it → synthesize → propose confirmation`) rather than a fully open-ended "let the LLM decide every next action in a loop until it decides to stop" planner.

**Trade-off, chosen deliberately:**

- *Pro:* step count is bounded and predictable, the confirmation gate cannot be skipped by a plan the LLM invents on its own, and the reasoning path is auditable turn-by-turn for the evaluation report — important for a facility-operations agent where an unreviewable action sequence is a real operational risk.
- *Con:* it is less "creative" than a fully agentic loop that could in principle discover investigation paths this code doesn't anticipate (e.g. a genuinely novel multi-hop chain the router's fixed route types don't cover).

Given the brief's explicit safety requirement ("the agent must not blindly execute actions") and its emphasis on autonomy over the *right things*, not autonomy over *everything*, bounded-but-multi-step was judged the better fit than unconstrained. `RouteDecision.is_multi_step` and `settings.orchestrator_max_steps` are the seams where a more open-ended planner could be substituted later without touching the sub-agents or the MCP/RAG layers.

## Why two knowledge-base sources (web-scraped + synthetic)

`https://www.nectarit.com/` (the company sponsoring this challenge) was scraped directly, as instructed. It turned out to be a marketing site: solutions pages, feature bullet points, headline stats, no chiller manuals, AHU troubleshooting steps, fault codes, or maintenance SOPs. A website-only knowledge base could not answer the brief's own worked examples ("why did Chiller-01 fail?", "what should I check if AHU airflow is low?").

Rather than either (a) ignoring the instruction to source from the website, or (b) building a knowledge base that can't satisfy the RAG task's stated requirements, both sources are used and tagged (`source_type: web_scrape` vs `synthetic_doc`) so:

- Platform/product questions are answered from what's actually published on nectarit.com, with the source URL retrievable.
- Facility-technical questions (the ones the demo scenario needs) are answered from authored documents written to the exact category list the brief specifies.
- The RAG agent can honestly say "I don't have documentation covering that" for anything outside both sets, rather than blending scraped marketing copy with invented technical claims.

## Why the action agent cannot call write tools at all (not just "is told not to")

An earlier design considered a single MCP server exposing both read and write tools to every agent, relying on the action agent's system prompt to never call the write tools directly. This was rejected: prompt instructions are not a security boundary, and an LLM being talked into a tool call it "shouldn't" make is a known failure mode.

Instead, `mcp_server/server.py` reads `MCP_ALLOW_ACTIONS` from its process environment **at import time** and only registers `create_service_request`/`update_service_request` when it's not explicitly disabled. `agents/action_agent.py` launches its MCP subprocess with `MCP_ALLOW_ACTIONS=false`, so those tools are structurally absent from that agent's tool list — there is no code path by which it could call them, regardless of what the model decides. The only code path to a write tool is `orchestration/confirmation.py:execute_confirmed_action`, invoked by the orchestrator exactly once, after an explicit affirmative reply.

## Known simplifications (first things to change for production)

- **In-memory session store** (`orchestration/session.py`) and **in-memory mock facility dataset** (`mcp_server/mock_facility_data.py`) reset on process restart and don't scale across multiple server instances. Both were built behind small interfaces specifically so they can be swapped (Redis/a database for sessions; a real BMS/SCADA/IoT gateway client for facility data) without changing any calling code.
- **Confirmation-reply interpretation** (`confirmation.interpret_confirmation_reply`) is a small keyword-based classifier, not an LLM call — chosen deliberately so the yes/no decision for a write action is deterministic and doesn't depend on model behavior; it errs toward `None` (re-ask) on anything it doesn't recognize rather than guessing.
- **Reranking** (`rag/reranker.py`) is a lightweight lexical-overlap boost on top of vector similarity, not a cross-encoder model — kept dependency-light for this prototype; the function signature is written so a real reranker model could be dropped in without touching callers.
- **STT/TTS providers** are wired but not exercised end-to-end with real audio in this submission's automated tests (they require live API keys and audio fixtures); `docs/sample_conversations.md` demonstrates the pipeline via the text-mode entry point (`process_text_turn`), which exercises every layer downstream of transcription identically to the voice path.

## Dependency pinning

`requirements.txt` pins `pydantic-ai==0.0.49` specifically because its MCP integration API (`MCPServerStdio`, `Agent(mcp_servers=...)`, `agent.run_mcp_servers()`, `result_type=`/`result.data`) is what this codebase is written against; later Pydantic AI releases redesigned MCP integration around a toolset/client abstraction with different names. `opentelemetry-api`/`-sdk` are pinned to `1.28.0` because `pydantic-ai==0.0.49` imports `opentelemetry._events`, a module removed from newer OpenTelemetry releases — an unpinned install resolves a newer OpenTelemetry version and fails at import time. Both were discovered and fixed by actually installing this project's `requirements.txt` into a clean virtual environment and running the test suite before submission, rather than assumed.

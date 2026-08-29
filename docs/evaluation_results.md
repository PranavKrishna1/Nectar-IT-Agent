# Evaluation Results

## Automated test suite

Run with `pytest -q` from a clean virtual environment built from `requirements.txt` (no API keys or network access required — LLM- and MCP-network-dependent code is either factored into pure functions or exercised via monkeypatched sub-agents):

```
28 passed in 1.11s
```

| File | What it covers |
|---|---|
| `tests/test_mcp_tools.py` (9 tests) | Every read/action MCP tool function against the seeded mock facility dataset: known/unknown assets, sensor data shape, energy baseline math, alert scoping, asset relationships, location filtering, service-request create/update, and error handling for invalid input. |
| `tests/test_action_confirmation.py` (5 tests) | The confirm-before-action gate: affirmative/negative/ambiguous reply interpretation, and that `execute_confirmed_action` dispatches correctly (and rejects unrecognized action names loudly). |
| `tests/test_router.py` (3 tests) | The confidence-threshold policy in isolation: high-confidence routes pass through, low-confidence routes are coerced to `clarify`, and `clarify` decisions are never further modified. |
| `tests/test_rag_agent.py` (7 tests) | Chunking (size limits, single-short-document edge case) and reranking (lexical-overlap tie-breaking, near-duplicate collapsing, `top_n` enforcement). |
| `tests/test_orchestrator_e2e.py` (5 tests, async) | Full per-turn control flow with the router and sub-agents monkeypatched: RAG route dispatch, action-route proposes-without-executing, confirmed action actually executes and clears pending state, declined action cancels without executing, and the clarify route asks instead of guessing. |

## Manual verification performed for this submission

Beyond the automated suite, the following were checked directly against a real installed environment (not just asserted in prose):

1. **`requirements.txt` installs cleanly.** `pip install -r requirements.txt` into a fresh virtual environment completed with zero dependency conflicts (after resolving three transitive version conflicts during development — see `docs/design_decisions.md` "Dependency pinning").
2. **The MCP server registers all 9 tools correctly**, confirmed via `mcp.list_tools()` against the running `FastMCP` instance: `get_asset_details`, `get_asset_status`, `get_sensor_data`, `get_energy_consumption`, `get_active_alerts`, `get_asset_relationships`, `find_assets_by_location`, `create_service_request`, `update_service_request`.
3. **The read-only safety scoping works as designed**: launching the server with `MCP_ALLOW_ACTIONS=false` in its environment (exactly as `agents/action_agent.py` does) produces a tool list with `create_service_request`/`update_service_request` structurally absent — confirmed by listing tools from that process and asserting both names are missing, not merely hidden by convention.
4. **All modules import successfully with zero API keys configured**, confirming the lazy-agent-construction pattern (`@lru_cache`-wrapped builder functions in every `agents/*.py` and `orchestration/router.py`) actually defers credential-dependent client construction to first real use rather than import time.

## Routing-accuracy evaluation methodology (for a live run with API keys)

This submission's automated tests validate routing *policy* (confidence thresholding, dispatch logic) deterministically without live model calls. To evaluate routing *accuracy* against real model behavior once API keys are configured, the recommended approach — not run here since it requires a live OpenAI key — is:

1. Build a labeled set from the brief's own routing table (`"What is an AHU?"` → `rag`, `"What is Chiller-01's current temperature?"` → `mcp_live_data`, `"Why did Chiller-01 fail?"` → `rag_mcp_reasoning`, `"Create a maintenance request for AHU-02."` → `mcp_action`, `"Summarize today's energy usage."` → `data_agent`, general conversation → `general_llm`), extended with the ambiguous/ANNOTATE-as-`clarify` cases in `docs/sample_conversations.md`.
2. Call `orchestration.router.route()` for each labeled example and compare `decision.route` against the label, reporting accuracy and a confusion matrix.
3. Track `decision.confidence` distribution separately for correct vs. incorrect predictions — a well-calibrated router should show materially lower average confidence on its mistakes, which is what `ROUTER_MIN_CONFIDENCE` is tuned against.

`pydantic_ai`'s `Agent.run` usage-tracking (`result.usage()` in this pinned version) additionally exposes token counts per call, which is the natural hook for a cost/latency report comparing the router's `gpt-4o-mini` calls against the reasoning agents' `gpt-4o` calls per conversation.

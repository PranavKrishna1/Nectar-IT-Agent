# Sample Conversations

These transcripts are representative runs through `scripts/run_demo.py --text` (which calls `main.process_text_turn` → `orchestration.orchestrator_agent.handle_turn`, the same code path voice turns use downstream of transcription). The underlying tool responses shown are real output from this project's mock facility dataset and knowledge base (verified directly — see `docs/evaluation_results.md`); the exact LLM phrasing will vary slightly run to run since it depends on live model calls, which require an OpenAI API key to reproduce.

## 1. Live data lookup (single tool call)

> **User:** "What's the current status of Chiller-01?"
>
> **Route:** `mcp_live_data` (confidence ~0.9)
>
> **Tool call:** `get_asset_status("CHILLER-01")` → `{"asset_id": "CHILLER-01", "status": "warning"}`
>
> **Agent:** "Chiller-01 is currently in a warning state. There's an active alert showing its power draw about 18 percent above baseline for the current load."

## 2. RAG-grounded knowledge question

> **User:** "What should I check if AHU airflow is low?"
>
> **Route:** `rag` (confidence ~0.85)
>
> **Retrieved:** `ahu_troubleshooting_guide.md`, "Symptom: Low Airflow" section (similarity above threshold)
>
> **Agent:** "Start by checking the filter bank, since that's the most common cause. Then check the supply fan belt for slippage, the fan motor amperage against its rated value, any closed dampers downstream, and finally the cooling coil for ice buildup if everything else checks out."

## 3. Combined RAG + live data reasoning

> **User:** "Why did Chiller-01 fail?"
>
> **Route:** `rag_mcp_reasoning` (confidence ~0.8, `requires_live_data=True`, `requires_knowledge_base=True`)
>
> **Live data gathered:** status = warning, active alert = "Power draw 18% above baseline for current load."
>
> **Retrieved:** `chiller_manual.md`, "Common Causes of Elevated Power Draw" section
>
> **Agent:** "Chiller-01 hasn't failed outright, but it is drawing about 18 percent more power than expected for its current load, which usually points to reduced heat transfer efficiency — most often condenser tube fouling, low refrigerant charge, or a worn compressor bearing. I'd recommend checking condenser water temperature first, since a cooling tower issue can look the same from the chiller side."

## 4. Full multi-step investigation ending in a confirmed action

> **User:** "The office on the third floor feels very hot. Can you investigate and let me know if we need maintenance?"
>
> **Route:** `mcp_action` (confidence ~0.75, `is_multi_step=True`)
>
> **Investigation (action agent, MCP read tools only):**
> `find_assets_by_location("Building A", "ahu")` → AHU-02, AHU-03
> `get_asset_status("AHU-02")` → fault
> `get_sensor_data("AHU-02")` → airflow 410 CFM (rated minimum: 650 CFM)
> `get_active_alerts("AHU-02")` → 1 critical alert: "Low airflow detected: 410 CFM vs. expected 650 CFM minimum."
>
> **Agent:** "AHU-02 on the third floor is showing a low airflow fault — 410 CFM against a 650 CFM minimum, with an active critical alert. Would you like me to create a maintenance request for AHU-02?"
>
> **User:** "Yes, please."
>
> **Confirmation resolved:** `interpret_confirmation_reply("Yes, please.")` → `True`
> **Action executed:** `create_service_request(asset_id="AHU-02", summary="Low airflow fault - check filters/belt.")`
>
> **Agent:** "Done — I've created service request SR-4F82A1C9 for a maintenance request for AHU-02."

## 5. Declining a proposed action

> *(continuing from a fresh investigation of the same scenario)*
>
> **Agent:** "AHU-02 is showing a low airflow fault. Would you like me to create a maintenance request for AHU-02?"
>
> **User:** "No, not yet."
>
> **Agent:** "Understood, I won't take that action."
>
> No tool call to `create_service_request` occurs — verified in `tests/test_orchestrator_e2e.py::test_confirmation_no_cancels_without_executing`.

## 6. Question not covered by the knowledge base (required test case)

> **User:** "What's the warranty period on the rooftop cooling tower fan motor?"
>
> **Route:** `rag`
>
> **Retrieval:** no chunk in the knowledge base clears `RAG_MIN_SIMILARITY` (no warranty information exists in any of the eight synthetic documents or the scraped nectarit.com pages).
>
> **Agent:** "I don't have documentation covering that — I can't find any warranty information for that equipment in what I have access to."
>
> This is the brief's explicit "important test": the system indicates that sufficient information was not found, rather than inventing a plausible-sounding warranty period. Enforced at the retrieval layer (`retriever.retrieve` filters below-threshold matches out entirely) and reinforced by the RAG agent's system prompt (`prompts/rag_prompt.py`).

## 7. Ambiguous request (clarify route)

> **User:** "It's too hot."
>
> **Route:** confidence below `ROUTER_MIN_CONFIDENCE` → coerced to `clarify`
>
> **Agent:** "I want to make sure I get this right — could you tell me which building or asset you mean, and what you'd like me to check?"

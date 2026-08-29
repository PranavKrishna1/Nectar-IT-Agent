# Architecture

## End-to-end diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              USER (Voice Input)                              │
└───────────────────────────────────┬────────────────────────────────────────┘
                                     ▼
                         ┌───────────────────────┐
                         │   voice/stt.py (STT)   │  Whisper / Deepgram
                         └───────────┬───────────┘
                                     ▼  (raw transcript)
                         ┌───────────────────────────┐
                         │ orchestration/session.py  │  loads conversation
                         │  (Conversation State)     │  history + context
                         └───────────┬───────────────┘
                                     ▼
                    ┌────────────────────────────────────┐
                    │   orchestration/router.py           │
                    │   Intent / Routing Layer            │
                    │   (LLM-based classifier, Pydantic AI│
                    │    structured output: RouteDecision)│
                    └───────────────┬──────────────────────┘
                                     │
        ┌────────────┬──────────────┼───────────────┬───────────────┐
        ▼            ▼              ▼               ▼               ▼
   ┌─────────┐  ┌──────────┐  ┌───────────┐   ┌────────────┐  ┌───────────┐
   │RAG Agent│  │Data Agent│  │RAG + MCP  │   │Action Agent│  │General LLM│
   │(agents/ │  │(agents/  │  │reasoning  │   │(agents/    │  │(agents/   │
   │ rag_    │  │ data_    │  │(combo of  │   │ action_    │  │ general_  │
   │ agent.py│  │ agent.py)│  │both below)│   │ agent.py)  │  │ agent.py) │
   └────┬────┘  └────┬─────┘  └─────┬─────┘   └─────┬──────┘  └─────┬─────┘
        ▼            ▼              ▼               ▼               ▼
  ┌───────────┐ ┌──────────┐  ┌───────────┐  ┌──────────────┐      │
  │ rag/      │ │ mcp_     │  │ rag/ +    │  │ orchestration/│      │
  │ retriever │ │ server   │  │ mcp_server│  │ confirmation. │      │
  │ (Pinecone)│ │ (read    │  │ (both)    │  │ py  (gate     │      │
  │           │ │  tools)  │  │           │  │  before write │      │
  └───────────┘ └──────────┘  └───────────┘  │  tools fire)  │      │
                                              │       ▼        │      │
                                              │ mcp_server/    │      │
                                              │ tools_action.py│      │
                                              │ (create/update │      │
                                              │  service req.) │      │
                                              └────────┬───────┘      │
                                     ▼                 ▼              ▼
                         ┌──────────────────────────────────────────────┐
                         │      orchestration/orchestrator_agent.py       │
                         │   Top-level per-turn handler — autonomous:     │
                         │   route → gather → gather more if implied →   │
                         │   synthesize → propose confirmation if needed  │
                         └───────────────────────┬────────────────────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  Response synthesis    │  final natural-language
                         │  (LLM, grounded on all │  answer, concise, cites
                         │  tool/RAG results)     │  sources if applicable
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  voice/tts.py (TTS)    │  ElevenLabs / AWS Polly
                         └───────────┬───────────┘
                                     ▼
                              USER (Voice Output)


 Supporting infrastructure:
 ┌─────────────────────────┐   ┌───────────────────────────────┐
 │ Pinecone Vector DB       │   │ FastMCP Server (mcp_server/)   │
 │ (rag/vector_store.py)    │   │  get_asset_details              │
 │  ← ingested via          │   │  get_asset_status                │
 │    rag/ingestion.py      │   │  get_sensor_data                  │
 │    rag/web_scraper.py    │   │  get_energy_consumption            │
 │    (nectarit.com +       │   │  get_active_alerts                  │
 │     synthetic docs)      │   │  get_asset_relationships             │
 │                          │   │  find_assets_by_location              │
 │                          │   │  create_service_request (guarded)      │
 │                          │   │  update_service_request (guarded)       │
 └─────────────────────────┘   └───────────────────────────────┘
```

## Walkthrough 1: single-tool live lookup

**User:** "What's the current status of Chiller-01?"

1. `router.route()` classifies this as `mcp_live_data` with high confidence (a single, specific live value; no documentation or reasoning needed).
2. `orchestrator_agent._handle_route` dispatches to `data_agent.answer_with_live_data`.
3. The data agent (a Pydantic AI agent with MCP tools) calls `get_asset_status("Chiller-01")` over MCP.
4. The MCP server resolves it against `mock_facility_data.py` and returns `{"asset_id": "CHILLER-01", "status": "warning"}`.
5. The data agent turns that into a natural sentence; the orchestrator returns it, TTS speaks it.

Matches the brief's diagram: `User → LLM → MCP Tool → get_asset_status("Chiller-01") → Live Facility Data → LLM Reasoning → Voice Response`.

## Walkthrough 2: full multi-step investigation

**User:** "The office on the third floor feels very hot. Can you investigate and let me know if we need maintenance?"

1. `router.route()` classifies this as `mcp_action` (the user is explicitly asking for an investigation that may end in a maintenance decision) with `is_multi_step=True`.
2. `orchestrator_agent._handle_action_route` calls `action_agent.investigate()`.
3. The action agent (MCP tools, read-only) autonomously chains:
   - `find_assets_by_location("Building A", "ahu")` → resolves "third floor" to AHU-02.
   - `get_asset_status("AHU-02")` → fault.
   - `get_sensor_data("AHU-02")` → 410 CFM (below the 650 CFM minimum from `equipment_specifications.md`).
   - `get_active_alerts("AHU-02")` → confirms a critical low-airflow alert.
   - `get_asset_relationships("AHU-02")` (optionally) → confirms it's fed by Chiller-01, so it can rule out an independent chiller-side cause before concluding the fault is local to the AHU.

   Note: the action agent's tools here are MCP read tools only (not RAG) — it investigates from live data. In the `rag_mcp_reasoning` route (a "why did X happen" question rather than an action request), the orchestrator additionally pulls the matching RAG document (e.g. the AHU troubleshooting guide's "what to check for low airflow" section) and synthesizes it together with the live findings; see `orchestrator_agent._handle_route`'s `RAG_MCP_REASONING` branch.
4. It returns an `ActionRecommendation`: `action_warranted=True`, `asset_id="AHU-02"`, with `findings` summarizing the low-airflow fault and `proposed_summary` for the ticket.
5. `orchestrator_agent` turns this into a `PendingConfirmation` attached to session state and asks: *"AHU-02 shows a low airflow fault at 410 CFM against a 650 CFM minimum, with an active critical alert. Would you like me to create a maintenance request for AHU-02?"*
6. On the user's next turn ("yes, please"), `confirmation.interpret_confirmation_reply` reads it as affirmative, and `confirmation.execute_confirmed_action` is the only call in the codebase that invokes `create_service_request`.
7. The orchestrator confirms the created ticket ID back to the user in speech.

This matches the brief's 11-step chain (identify building → temperature → HVAC assets → chiller status → AHU status → alerts → troubleshooting doc → reasoning → maintenance decision → create request → report result) while keeping the write action behind an explicit confirmation.

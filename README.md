# Nectar Autonomous Facility Operations Voice Agent

An autonomous voice AI agent for Nectar's Intelligent Facilities Platform. A facility operator speaks naturally; the agent transcribes the request, decides what it needs (documentation, live facility data, or both), reasons over multiple pieces of information, takes tool-mediated actions only after explicit confirmation, and replies in natural speech.

Built for the Nectar AI Engineer — Agentic AI Challenge (Tasks 1–5).

## Table of contents

1. [Technical stack](#technical-stack)
2. [Setup](#setup)
3. [Running it](#running-it)
4. [Architecture](#architecture)
5. [Folder structure](#folder-structure)
6. [Data sources](#data-sources)
7. [Agent workflow](#agent-workflow)
8. [LLM routing strategy](#llm-routing-strategy)
9. [RAG architecture](#rag-architecture)
10. [MCP architecture](#mcp-architecture)
11. [Safety: confirmation before action](#safety-confirmation-before-action)
12. [Design decisions & assumptions](#design-decisions--assumptions)
13. [Evaluation](#evaluation)

## Technical stack

| Layer | Choice | Why |
|---|---|---|
| Agent framework | **Pydantic AI** | Typed, structured-output agents (`Agent[DepsType, OutputType]`) mean the router's decision, the RAG agent's answer, and the action agent's recommendation are all validated Pydantic models, not parsed free text. Native async, native MCP client support, and tool-calling via plain typed Python functions with docstrings (which Pydantic AI turns into the tool's schema/description automatically). Also provider-agnostic, which is what makes the free/paid model swap below a one-line config change. |
| Tool server | **FastMCP** | The standard way to stand up an MCP-protocol tool server in Python with minimal boilerplate — a plain function plus `@mcp.tool()`/`mcp.tool()(func)` becomes a schema'd, discoverable MCP tool. Used to expose all facility read/action tools (Task 4). |
| Vector database | **Pinecone** (serverless Starter) | Managed, no infrastructure to run locally, index-per-project is trivial to provision (`rag/vector_store.py` creates it on first ingestion run), a genuinely free tier, and its hosted Inference embeddings mean the same key covers both storage *and* embedding generation. |
| LLM | **Google Gemini** (`gemini-2.0-flash`) by default; **OpenAI** (`gpt-4o-mini`/`gpt-4o`) as a swappable alternative | Gemini's free tier needs no credit card, which makes the project runnable by anyone cloning it. The router and reasoning models are separate config values specifically so a cheap model can handle routing and an expensive one the reasoning — see [LLM routing strategy](#llm-routing-strategy). Any Pydantic-AI-supported model string works. |
| Embeddings | **Pinecone Inference** (`multilingual-e5-large`, 1024-dim) by default; **OpenAI** (`text-embedding-3-small`, 1536-dim) as an alternative | Reuses the Pinecone key, so the free setup needs no third signup. Selected via `EMBEDDING_PROVIDER`; `rag/embeddings.py` hides the difference behind `embed_texts`/`embed_query`. |
| Speech-to-text | **faster-whisper** (local) by default; **Whisper API** and **Deepgram** as alternatives | The local backend runs Whisper on CPU via CTranslate2 with no key and no network, which keeps the free path fully self-contained. Whisper and Deepgram are both named in the brief and remain available via `STT_PROVIDER`. |
| Text-to-speech | **edge-tts** by default; **ElevenLabs** and **AWS Polly** as alternatives | edge-tts uses Microsoft Edge's neural voices with no key or signup. Same swap pattern via `TTS_PROVIDER` and `voice/tts.py`. |
| Web scraping | **httpx + BeautifulSoup** | Lightweight, no headless browser needed since the scraped pages (nectarit.com) are static marketing content. |
| Testing | **pytest + pytest-asyncio** | Standard for this ecosystem; `asyncio_mode = "auto"` in `pyproject.toml` lets async test functions run without extra decorators. |

### Why Pydantic AI's `pydantic-ai==0.0.49` specifically

Pydantic AI's public API for MCP integration has changed across releases. This project targets `0.0.49` deliberately because it has a simple, stable, well-documented MCP surface used throughout this codebase:

- `pydantic_ai.mcp.MCPServerStdio(command, args=[...], env={...})` — launches an MCP server as a subprocess.
- `Agent(model, mcp_servers=[...])` — wires an agent to one or more MCP servers.
- `async with agent.run_mcp_servers(): ...` — manages the MCP subprocess lifecycle around a run.
- `Agent(model, result_type=SomeModel)` and `result.data` — structured output (later Pydantic AI releases renamed these to `output_type` / `result.output`; if you upgrade, update `agents/*.py` and `orchestration/router.py` accordingly — every place is commented).

If you install a newer `pydantic-ai`, run `pip show pydantic-ai` and check `pydantic_ai.mcp` for `MCPServerStdio` vs. a toolset-based API before assuming this code runs unmodified.

## Setup

### Requirements

- Python 3.11+
- **Two free API keys** (neither requires a credit card) — see below

### Free-tier setup (the default configuration)

The project ships configured to run entirely on free tiers. You need exactly **two** keys:

| # | Key | Where to get it | Free tier | Card needed? |
|---|---|---|---|---|
| 1 | `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Generous daily request quota on Gemini Flash models | **No** |
| 2 | `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → Starter plan | 2 GB storage, 5 indexes, plus a monthly hosted-embedding allowance | **No** |

Everything else has a working free default and needs **no key at all**:

- **Embeddings** run on Pinecone's own hosted Inference models (`multilingual-e5-large`), reusing the Pinecone key you already have — so there's no third signup.
- **Speech-to-text** runs Whisper locally via `faster-whisper` — no key, no network after the first model download (~75 MB for the `base` model).
- **Text-to-speech** uses Microsoft Edge's neural voices via `edge-tts` — no key, no signup.

**Step by step:**

1. **Get the Gemini key.** Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey), sign in with a Google account, click *Create API key*. Copy it.
2. **Get the Pinecone key.** Sign up at [app.pinecone.io](https://app.pinecone.io), choose the free **Starter** plan, then go to *API Keys* → *Create API key*. Copy it. Leave `PINECONE_REGION=us-east-1` — the free tier only supports that region.
3. **Paste both into `.env`** (copy `.env.example` first). Only those two lines need filling in; every other value already has a working default.

To use paid providers instead, set `OPENAI_API_KEY`, change `LLM_MODEL_ROUTER`/`LLM_MODEL_REASONING` to `openai:gpt-4o-mini`/`openai:gpt-4o`, and set `EMBEDDING_PROVIDER=openai` with `EMBEDDING_MODEL=text-embedding-3-small` and `EMBEDDING_DIMENSIONS=1536`. Note that changing embedding provider changes vector dimensionality, so you must delete and re-create the Pinecone index and re-run ingestion.

### Install

```bash
git clone <this-repo>
cd nectar-voice-agent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and fill in OPENAI_API_KEY, PINECONE_API_KEY, etc.
```

`requirements.txt` is fully version-pinned and was verified to install cleanly (`pip install -r requirements.txt`, zero conflicts) into a clean virtual environment as part of this submission. The pins exist for two reasons worth knowing about if you touch them: (1) `pydantic-ai==0.0.49` needs `fastmcp>=2.3.4` and an `mcp` release `>=1.8.1,<2.0.0` — going outside that band changes the MCP API surface, and (2) `pydantic-ai==0.0.49` imports OpenTelemetry's now-removed experimental Events API, so `opentelemetry-api`/`-sdk` are pinned to `1.28.0` specifically (a version still exposing `opentelemetry._events`) — installing this project's dependencies without that pin will resolve a newer OpenTelemetry release and fail at import time with `ModuleNotFoundError: No module named 'opentelemetry._events'`.

### Check your setup first

Before ingesting anything, verify your credentials actually work:

```bash
python scripts/check_setup.py
```

This confirms `.env` is found, both keys are present and well-formed, Pinecone authenticates, and your Gemini model still has daily quota left — and prints a specific fix for whatever fails. It costs one Gemini request. Run it any time something breaks; it catches most problems in a couple of seconds rather than partway through an ingestion run.

### Troubleshooting

**`401 Unauthorized` from Pinecone**
The key is wrong, not the code. Almost always a copy-paste artifact — a duplicated leading character (`Ppcsk_...` instead of `pcsk_...`), wrapping quotes, or a trailing space. Pinecone keys start with `pcsk_`. `Settings` now validates this at load time and names the problem, so `python scripts/check_setup.py` will tell you directly. Re-copy the key from [app.pinecone.io](https://app.pinecone.io) → *API Keys*, paste the raw value with no quotes.

**`429` / daily quota exhausted from Gemini**
The key is valid; you've used up today's free requests. The usual cause is the model: **don't use `-latest` aliases** like `gemini-flash-latest`. They resolve to whatever is newest, and the newest preview models carry drastically lower free daily limits (as little as ~20 requests/day — one testing session). Pin an explicit stable Flash model instead:

```dotenv
LLM_MODEL_ROUTER=google-gla:gemini-2.0-flash
LLM_MODEL_REASONING=google-gla:gemini-2.0-flash
```

Stable Flash models allow far more per day. Your actual live per-model limits are at [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit). Other options: wait for the daily reset (midnight Pacific), or create a second free key under a different Google account.

**`Index dimension mismatch`**
Your Pinecone index was created with different dimensions than `EMBEDDING_DIMENSIONS`. This happens when switching embedding providers (Pinecone = 1024, OpenAI = 1536). Delete the index in the Pinecone console and re-run ingestion to recreate it at the right size. `check_setup.py` detects this before you hit it.

### Ingest the knowledge base

Before the RAG agent can answer anything, run the ingestion script once. This scrapes `nectarit.com`, chunks every document in both knowledge-base sources, embeds them, and upserts them into Pinecone (creating the index if it doesn't exist):

```bash
python scripts/ingest_knowledge_base.py
```

Re-run it (with `--skip-scrape` if you don't want to re-hit the website) whenever documents under `data/knowledge_base/` change.

## Running it

**Text demo** (fastest way to exercise the whole pipeline without audio):

```bash
python scripts/run_demo.py --text
```

**Single-turn voice demo** (pre-recorded file in, audio file out):

```bash
python scripts/run_demo.py --voice --input path/to/recording.wav --output response.mp3
```

**Live voice demo** (real microphone in, real speakers out — the fully spoken experience):

```bash
python scripts/run_demo.py --live
```

Press Enter, speak your request, press Enter again to stop recording; the agent transcribes it, runs the full routing/RAG/MCP/confirmation pipeline, and speaks its reply back through your default output device before prompting for the next turn. Ctrl+C exits. This mode needs two extra packages beyond the free-tier defaults — `sounddevice` (microphone capture and playback) and `miniaudio` (decodes the synthesized MP3 to raw audio in memory, no ffmpeg required) — both already in `requirements.txt`.

**Run the MCP server standalone** (useful for inspecting tools with an MCP client, or via `mcp dev`):

```bash
python -m nectar_agent.mcp_server.server
```

**Run tests** (all 28 tests are self-contained — no API keys or network required, since LLM/MCP-dependent code is either factored into pure functions or monkeypatched in `tests/test_orchestrator_e2e.py`):

```bash
pytest -q
```

## Architecture

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
   └────┬────┘  └────┬─────┘  │(combo)    │   └─────┬──────┘  └─────┬─────┘
        ▼            ▼        └─────┬─────┘         ▼               │
  ┌───────────┐ ┌──────────┐        ▼        ┌──────────────┐      │
  │ rag/      │ │ mcp_     │  RAG + MCP  │  orchestration/│      │
  │ retriever │ │ server   │  (both)     │  confirmation. │      │
  │ (Pinecone)│ │ (read    │             │  py (gate      │      │
  │           │ │  tools)  │             │  before write  │      │
  └───────────┘ └──────────┘             │  tools fire)   │      │
                                          │       ▼        │      │
                                          │ mcp_server/    │      │
                                          │ tools_action.py│      │
                                          └────────┬───────┘      │
                                     ▼              ▼              ▼
                         ┌──────────────────────────────────────────────┐
                         │  orchestration/orchestrator_agent.py           │
                         │  Top-level autonomous per-turn handler:        │
                         │  route → gather → gather more if needed →      │
                         │  synthesize → (confirm if action proposed)     │
                         └───────────────────────┬────────────────────────┘
                                     ▼
                         ┌───────────────────────┐
                         │  voice/tts.py (TTS)    │  ElevenLabs / AWS Polly
                         └───────────┬───────────┘
                                     ▼
                              USER (Voice Output)
```

See `docs/architecture.md` for the annotated version of this diagram plus a walkthrough of the two worked examples from the brief (the single-tool "what's Chiller-01's status" case and the full 11-step "investigate and create a request if necessary" case).

## Folder structure

```
nectar-voice-agent/
├── README.md
├── requirements.txt / pyproject.toml / .env.example
├── docs/                          architecture, design decisions, sample conversations, evaluation
├── data/knowledge_base/
│   ├── web_sourced/                scraped from nectarit.com
│   └── synthetic/                  authored HVAC/chiller/AHU/safety/etc. documents
├── src/nectar_agent/
│   ├── config.py                   central pydantic-settings configuration
│   ├── main.py                     voice-turn entry point
│   ├── voice/                      STT + TTS wrappers                         (Task 1)
│   ├── orchestration/              router, session, confirmation, orchestrator (Task 2 + 5)
│   ├── agents/                     rag_agent, data_agent, action_agent, general_agent
│   ├── rag/                        web_scraper, ingestion, embeddings, vector_store,
│   │                                retriever, reranker                        (Task 3)
│   ├── mcp_server/                 FastMCP server + read/action tools + mock data (Task 4)
│   ├── models/                     shared Pydantic domain/routing/conversation models
│   └── prompts/                    one system prompt module per agent
├── scripts/                        ingest_knowledge_base.py, run_demo.py
└── tests/                          28 tests across router, RAG, MCP tools, confirmation, e2e
```

Full rationale for this split is in `docs/design_decisions.md`.

## Data sources

Two knowledge-base sources, both ingested by the same pipeline but tagged by origin (`source_type` metadata on every Pinecone vector):

1. **`data/knowledge_base/web_sourced/`** — scraped directly from **https://www.nectarit.com/** by `rag/web_scraper.py` (homepage + Solutions pages + About). This covers platform/product/company questions ("what protocols does the platform support", "what does Connected Buildings monitor").
2. **`data/knowledge_base/synthetic/`** — eight authored documents matching the exact categories the brief specifies (HVAC operating procedures, chiller manual, AHU troubleshooting guide, maintenance procedures, safety instructions, equipment specifications, facility policies, troubleshooting FAQs). **This exists because nectarit.com is a marketing site, not a documentation portal** — it does not publish chiller fault codes, AHU airflow thresholds, or troubleshooting steps, so a website-only knowledge base cannot answer the brief's own example questions ("why did Chiller-01 fail?", "what should I check if AHU airflow is low?"). See `docs/design_decisions.md` for the full reasoning and what was found on the live site.

Run `python scripts/ingest_knowledge_base.py` to (re-)scrape and (re-)index both sources.

## Agent workflow

Per user turn, `orchestration/orchestrator_agent.handle_turn()`:

1. Records the turn to session history (`orchestration/session.py`).
2. **Checks for a pending confirmation first.** If the previous turn proposed an action, this turn's text is interpreted as yes/no/ambiguous before anything else happens.
3. Otherwise, **routes** the query (`orchestration/router.py`) using conversation history as context.
4. **Dispatches** to the right sub-agent(s) based on the route — a RAG-only answer, a live-data answer, a combined RAG+MCP+reasoning synthesis, an investigation that may propose an action, or a general-conversation reply.
5. If an action is warranted, the response becomes a confirmation question and the proposed action is attached to session state — **never executed**.
6. Records the agent's turn and returns the response text for TTS.

## LLM routing strategy

- A cheap/fast model (`gpt-4o-mini`) classifies intent into one of `rag`, `mcp_live_data`, `rag_mcp_reasoning`, `mcp_action`, `data_agent`, `general_llm`, or `clarify`, returning a structured `RouteDecision` (route, confidence, reasoning, flags) — never free text, so the orchestrator branches on it programmatically.
- **Ambiguous requests**: if confidence is below `ROUTER_MIN_CONFIDENCE` (default 0.55), the decision is coerced to `clarify` regardless of the model's guess (`orchestration/router.py:enforce_confidence_threshold`), and the orchestrator asks a clarifying question instead of acting on a low-confidence guess.
- **Incorrect routing** is caught two ways: (1) the confidence gate above prevents low-confidence misroutes from executing silently, and (2) sub-agents are scoped tightly enough (e.g. the RAG agent only has a retrieval tool, never live-data tools) that even a routing miss tends to fail loudly ("I don't have documentation covering that") rather than fabricating an answer from the wrong source.
- **Tool failure**: every MCP tool function returns a typed `{"error": "..."}` dict rather than raising, so a failed/unknown lookup surfaces to the LLM as data it can reason about and report honestly, instead of crashing the turn.
- **Cost/latency optimization**: routing uses the cheap model on every turn; the expensive reasoning model is only invoked for the specific sub-agent(s) the route actually calls for — a simple RAG or live-data lookup never pays for a `gpt-4o` call twice (once to route, once to answer) the way a single-model design would.

## RAG architecture

```
Documents (web_sourced/ + synthetic/)
    ↓
Document Processing (rag/ingestion.py: front-matter parsing, source tagging)
    ↓
Chunking (rag/ingestion.py: paragraph-aware, 800 chars / 120 overlap by default)
    ↓
Embeddings (rag/embeddings.py: OpenAI text-embedding-3-small)
    ↓
Vector Database (rag/vector_store.py: Pinecone serverless index)
    ↓
Retriever (rag/retriever.py: top-k similarity search + minimum-similarity cutoff)
    ↓
Reranker / Filtering (rag/reranker.py: lexical-overlap boost + dedup)
    ↓
LLM (agents/rag_agent.py: answers strictly from retrieved excerpts)
    ↓
Grounded Answer (with source titles; explicit "not found" when nothing clears the threshold)
```

The "ask a question not in the knowledge base" requirement is handled at two layers: `retriever.retrieve()` filters out anything below `RAG_MIN_SIMILARITY`, and the RAG agent's system prompt (`prompts/rag_prompt.py`) instructs it to say so explicitly rather than answer from weak or absent context — the agent's own tool call returns the literal string `"NO_RELEVANT_DOCUMENTS_FOUND"` when nothing qualifies, which the model is instructed to translate into a plain "I don't have documentation covering that" rather than paper over.

## MCP architecture

`mcp_server/server.py` is a FastMCP app exposing:

- **Read tools** (always available): `get_asset_details`, `get_asset_status`, `get_sensor_data`, `get_energy_consumption`, `get_active_alerts`, `get_asset_relationships`, plus `find_assets_by_location` (added to resolve natural-language locations like "the office on the third floor" into concrete asset IDs — needed for the Task 5 example scenario).
- **Action tools**: `create_service_request`, `update_service_request`.

Backed by an in-memory simulated dataset (`mock_facility_data.py`) standing in for a real BMS/SCADA integration — swappable without touching the tool interface.

Pydantic AI agents connect to this server as genuine MCP clients (`pydantic_ai.mcp.MCPServerStdio` launching `python -m nectar_agent.mcp_server.server` as a subprocess), so tool calls go over the actual Model Context Protocol, not direct Python calls.

## Safety: confirmation before action

This is enforced at three independent layers, not just a prompt instruction:

1. **The action agent physically cannot call write tools.** Its MCP subprocess is launched with `MCP_ALLOW_ACTIONS=false`; `mcp_server/server.py` reads that at import time and never registers `create_service_request`/`update_service_request` in that process at all — the tools are absent from its MCP session, not merely discouraged.
2. **Only one function in the whole codebase is allowed to call the write tools**: `orchestration/confirmation.py:execute_confirmed_action`. It is only ever invoked from `orchestrator_agent.py` after `interpret_confirmation_reply` has classified the user's reply as an explicit affirmative — an ambiguous reply re-asks rather than assuming either way.
3. **The action agent's own output type has no execution path** — `ActionRecommendation` is a proposal (`action_warranted`, `asset_id`, `proposed_summary`), never a completed action. Turning a recommendation into an actual tool call is the orchestrator's job, one turn later, gated on the user's next reply.

## Design decisions & assumptions

See `docs/design_decisions.md` for the full write-up, including: why the orchestrator is explicit Python control flow rather than an open-ended planning loop, the two-source knowledge-base strategy, dependency pinning rationale, and known simplifications (in-memory session store and mock facility data — both called out as the first things to swap for a production deployment).

## Evaluation

See `docs/evaluation_results.md` for the test suite summary and manual routing-accuracy spot checks, and `docs/sample_conversations.md` for transcripts of the brief's example scenarios (single-tool status lookup, RAG-grounded troubleshooting question, the full multi-step "investigate and create a request if necessary" flow, and an out-of-knowledge-base question to confirm the agent says so honestly).

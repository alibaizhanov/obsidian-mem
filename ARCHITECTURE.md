# Mengram — Architecture Overview

## Concept

You chat with any AI (Claude, GPT, or any LLM). The system **automatically** extracts
knowledge from conversations and builds structured memory with 3 types — like a human brain.

## Two Modes

### Cloud Mode (Production — mengram.io)

```
Any client (SDK, MCP, LangChain, CrewAI, OpenClaw)
      │
      ▼
┌─────────────────────────────────────────┐
│          FastAPI Cloud API              │
│   Auth · Rate limiting · Background jobs│
├─────────────────────────────────────────┤
│        Conversation Extractor           │
│   Claude Sonnet / GPT-4o-mini           │
│   → Entities, Facts, Relations          │
│   → Episodes (events)                   │
│   → Procedures (workflows)              │
├─────────────────────────────────────────┤
│        Evolution Engine (v2.7)          │
│   Auto-link episodes ↔ procedures       │
│   Failure → LLM analysis → evolve      │
│   3+ similar episodes → auto-create     │
│   Version history + evolution log       │
├─────────────────────────────────────────┤
│     Smart Triggers · Memory Agents      │
│   Reminders · Contradictions · Patterns │
│   Curator · Connector · Digest          │
├─────────────────────────────────────────┤
│        PostgreSQL + pgvector            │
│   HNSW index · BM25 hybrid search       │
│   LLM re-ranking · Matryoshka 1536D     │
└─────────────────────────────────────────┘
```

### Local Mode (Self-hosted)

```
Claude Desktop / Cursor (MCP)
      │
      ▼
┌─────────────────────────────────────────┐
│        MengramBrain (local engine)      │
│   Conversation Extractor → LLM          │
│   → .md files (Obsidian vault)          │
│   → SQLite vectors + knowledge graph    │
│   → Hybrid search (vector + graph)      │
└─────────────────────────────────────────┘
```

## Experience-Driven Procedures (v2.7)

Closed feedback loop between episodic and procedural memory:

```
Failure cycle:
  User says "deploy failed, forgot migrations"
    → Episode extracted (negative, auto-linked to "Deploy" procedure)
    → LLM analyzes: "missing migration step before push"
    → "Deploy" v1 → v2 (step added: run migrations)
    → Evolution logged in procedure_evolution table

Success cycle:
  3+ similar positive episodes without a procedure
    → Clustered by embedding similarity
    → LLM extracts common workflow
    → New procedure auto-created
    → Source episodes linked
```

## Project Structure

```
mengram/
├── cloud/                   # Cloud mode (production)
│   ├── api.py               # FastAPI server (all endpoints)
│   ├── store.py             # PostgreSQL + pgvector backend
│   ├── client.py            # Python SDK (CloudMemory)
│   ├── evolution.py         # Evolution Engine (v2.7)
│   ├── embedder.py          # OpenAI embeddings (API-based)
│   ├── schema.sql           # Full PostgreSQL schema
│   ├── dashboard.html       # Web dashboard
│   ├── landing.html         # Landing page
│   └── Dockerfile           # Docker deployment
├── engine/                  # Local mode (self-hosted)
│   ├── extractor/
│   │   ├── conversation_extractor.py  # LLM knowledge extraction
│   │   └── llm_client.py             # Claude / GPT / Ollama
│   ├── vault_manager/
│   │   └── vault_manager.py          # .md file manager
│   ├── graph/
│   │   └── knowledge_graph.py        # SQLite knowledge graph
│   ├── vector/
│   │   ├── embedder.py               # Local embeddings
│   │   └── vector_store.py           # SQLite vector search
│   └── retrieval/
│       └── hybrid_search.py          # Vector + graph retrieval
├── api/
│   ├── mcp_server.py        # MCP Server (local vault)
│   ├── cloud_mcp_server.py  # MCP Server (cloud API)
│   └── rest_server.py       # REST wrapper
├── integrations/
│   ├── langchain.py         # LangChain memory + retriever
│   ├── crewai.py            # CrewAI tools (5 tools)
│   └── openclaw/            # OpenClaw skill (bash scripts)
├── sdk/
│   └── js/                  # JavaScript / TypeScript SDK
│       ├── index.js
│       ├── index.d.ts
│       └── package.json
├── tests/
├── cli.py                   # CLI (`mengram` command)
├── pyproject.toml           # Python package config
└── README.md
```

## Key Data Flow

```
POST /v1/add (messages)
  ├── ConversationExtractor → LLM → entities, facts, relations, knowledge, episodes, procedures
  ├── Save entities + embeddings (pgvector HNSW)
  ├── Save episodes + embeddings
  │   └── Auto-link to existing procedures (embedding similarity ≥ 0.75)
  │       ├── Negative episode → evolve_on_failure() → new procedure version
  │       └── Positive episode → increment success_count
  ├── Save procedures + embeddings (ON CONFLICT upsert)
  ├── Detect patterns → auto-create procedures from 3+ similar episodes
  ├── Smart Triggers: reminders, contradictions, patterns
  └── Auto-reflection if threshold reached
```

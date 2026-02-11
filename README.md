# 🧠 Mengram

**AI memory as a typed knowledge graph in Obsidian.**

Every conversation with your AI builds a structured second brain — people, projects, technologies, companies — all as `.md` files with `[[wikilinks]]` you can browse in [Obsidian](https://obsidian.md).

Like [Mem0](https://github.com/mem0ai/mem0), but **you own your data** — and it actually saves your solutions with code, not just "user uses PostgreSQL".

---

## Why Mengram?

|  | **Mem0** | **Basic Memory** | **Mengram** |
|---|---|---|---|
| Storage | Cloud vectors | Flat markdown | **Typed knowledge graph in .md** |
| Entity types | ❌ Flat facts | ❌ One note per chat | ✅ Person, Project, Technology, Company |
| Relations | ❌ | ❌ | ✅ `works_at`, `uses`, `depends_on` |
| **Rich knowledge** | ❌ | ❌ | ✅ **Solutions, configs, formulas with code** |
| **Proactive context** | ❌ | ❌ | ✅ **Auto-injected — no manual recall** |
| Obsidian graph | ❌ | Partial | ✅ Full `[[wikilinks]]` + graph view |
| Semantic search | ✅ Cloud | ❌ | ✅ Local embeddings (384D) |
| Own your data | ❌ Cloud lock-in | ✅ | ✅ Plain `.md` files |
| LLM agnostic | ❌ | Partial | ✅ Claude / GPT / Ollama |
| Pricing | $24/mo+ | $14/mo | **Free & open source** |

### What it actually does

You chat with Claude (or any LLM). Mengram **automatically**:

1. **Extracts** entities, facts, relationships, and **rich knowledge** (solutions, commands, configs with code)
2. **Creates** typed `.md` files in your Obsidian vault
3. **Links** everything with `[[wikilinks]]` and YAML frontmatter
4. **Indexes** with local vector embeddings for semantic search
5. **Proactively injects** relevant context into every conversation — no manual recall needed

```
You: "We fixed the OOM with Redis cache. Config: hikari.pool-size=20"

         ┌─────────────────────────────────────┐
         │  vault/PostgreSQL.md                 │
         │  type: technology                    │
         │                                      │
         │  ## Facts                             │
         │  - Main database, version 15          │
         │                                      │
         │  ## Knowledge                         │
         │  **[solution] Connection pool fix**    │
         │  OOM at 200+ WebSocket → Redis cache  │
         │  ```yaml                              │
         │  spring.datasource.hikari.            │
         │    maximum-pool-size: 20              │
         │  ```                                  │
         └─────────────────────────────────────┘
```

Next time you ask "How did we fix the OOM?" → Claude **already knows**, with the config.

---

## Quick Start

### 1. Install

```bash
pip install mengram[all]
```

### 2. Setup (one command)

```bash
mengram init
```

This will:
- Ask for your LLM provider and API key
- Create `~/.mengram/config.yaml` and vault
- Auto-configure Claude Desktop MCP integration
- Tell you to restart Claude Desktop

That's it. **Talk to Claude — it remembers automatically and always has context.**

### Non-interactive:

```bash
mengram init --provider anthropic --api-key sk-ant-...
```

### Other commands:

```bash
mengram status    # Check setup
mengram stats     # Vault statistics
mengram server    # Start MCP server manually
```

---

## Proactive Context (v0.5.0)

The killer feature. Claude Desktop gets your knowledge profile **automatically** — no manual attach, no "recall", no "remember what I told you".

**How it works:**

```
Claude Desktop starts
  → MCP server reads vault
  → Generates compact knowledge index (scales to 1000+ notes)
  → Injects into Claude's instructions
  → Warms up semantic search model

You open any chat → Claude already knows:
  - Your tech stack, projects, team
  - Past solutions with code/configs
  - Entity relationships

You ask a question → Claude auto-calls recall()
  → Gets full details + code artifacts
  → Answers with context
```

---

## Rich Knowledge (v0.5.0)

Not just "user uses PostgreSQL" — but **solutions with code**, commands, formulas, configs.

The LLM **automatically** chooses the knowledge type based on context:

| Domain | Knowledge types | Example |
|---|---|---|
| Developer | `solution`, `command`, `config`, `debug` | HikariCP pool config with YAML |
| Doctor | `treatment`, `lab_result`, `diagnosis` | Metformin 500mg dosage |
| Scientist | `experiment`, `formula`, `hypothesis` | Protein denaturation at 60°C |
| Student | `formula`, `example`, `insight` | Bayes theorem with example |
| Chef | `recipe`, `tip`, `substitution` | Sourdough hydration ratio |

**No configuration needed.** The system adapts to any domain.

```markdown
## Knowledge

**[solution] Connection pool exhaustion fix** (2024-02-10)
OOM at 200+ WebSocket connections → Redis cache for UserService
​```yaml
spring.datasource.hikari.maximum-pool-size: 20
spring.datasource.hikari.idle-timeout: 30000
​```

**[command] Debug database connections** (2024-02-10)
Monitor active PostgreSQL connections
​```sql
SELECT count(*), state FROM pg_stat_activity GROUP BY state;
​```
```

---

## Python SDK (Mem0-compatible API)

```python
from mengram import Memory

m = Memory(
    vault_path="./my-brain",
    llm_provider="anthropic",
    api_key="sk-ant-..."
)

# Remember — extracts entities, facts, relations, AND knowledge
m.add("I work at Uzum Bank, backend on Spring Boot and PostgreSQL", user_id="ali")

# Semantic search (finds by MEANING, not just keywords)
results = m.search("database issues", user_id="ali")
for r in results:
    print(f"{r.memory.name} (score={r.score:.2f})")
    print(r.memory.facts)

# Get everything
all_memories = m.get_all(user_id="ali")

# Stats
print(m.stats(user_id="ali"))
```

### Auto-Memory Middleware

Drop-in wrapper that automatically remembers and recalls:

```python
from mengram import Memory
from mengram_middleware import AutoMemory

m = Memory(vault_path="./vault", llm_provider="anthropic", api_key="sk-ant-...")
auto = AutoMemory(memory=m, user_id="ali")

# Automatically: recall context → inject → LLM response → remember new knowledge
response = auto.chat("Help me fix the PostgreSQL connection pool issue")
```

---

## How It Works

```
Conversation → Extractor (LLM) → Entities + Facts + Relations + Knowledge
                                          ↓
                                   Vault Manager → .md files with [[wikilinks]]
                                          ↓
                                   Vector Index → local embeddings (SQLite)
                                          ↓
                                   MCP Server → instructions (compact index)
                                              → tools (recall, remember)
                                          ↓
                                   Claude Desktop → auto-context every chat
```

### Semantic Search (Hybrid)

3-level search strategy:

1. **Vector Search** — `all-MiniLM-L6-v2` (80MB, runs locally). Finds "database" when you search "PostgreSQL" — by meaning, not keywords.
2. **Graph Expansion** — follows `[[wikilinks]]` from top results. Found PostgreSQL? Also returns linked Project Alpha.
3. **Text Fallback** — substring match for edge cases.

### Entity Types

| Type | Examples |
|---|---|
| `person` | Team members, contacts |
| `project` | Services, repos, products |
| `technology` | PostgreSQL, Spring Boot, Kafka |
| `company` | Employers, clients, partners |
| `concept` | Patterns, strategies, ideas |

### File Format

```markdown
---
type: technology
created: 2024-02-10 15:30
updated: 2024-02-11 09:15
tags: [technology]
---

# PostgreSQL

## Facts

- Main database, version 15
- Connection pool issue in [[Project Alpha]]

## Relations

- ← uses [[Project Alpha]]: Main DB

## Knowledge

**[solution] Connection pool exhaustion fix** (2024-02-10)
OOM at 200+ WebSocket → Redis cache for UserService
​```yaml
spring.datasource.hikari.maximum-pool-size: 20
​```
```

---

## Configuration

```yaml
# config.yaml
vault_path: "./vault"

llm:
  provider: "anthropic"  # anthropic | openai | ollama | mock
  anthropic:
    api_key: "sk-ant-..."
    model: "claude-sonnet-4-20250514"

semantic_search:
  enabled: true
```

| Provider | Install | Cost |
|---|---|---|
| Anthropic (Claude) | `pip install mengram[anthropic]` | API pricing |
| OpenAI (GPT) | `pip install mengram[openai]` | API pricing |
| Ollama (local) | Install [ollama](https://ollama.ai) | Free |

---

## Roadmap

- [x] Typed entity extraction (person, project, technology, company)
- [x] Obsidian vault with `[[wikilinks]]` + YAML frontmatter
- [x] MCP Server for Claude Desktop
- [x] Semantic search with local embeddings
- [x] Hybrid retrieval (vector + graph)
- [x] Mem0-compatible Python SDK
- [x] Auto-memory middleware
- [x] **Rich knowledge extraction (solutions, configs, formulas with code)**
- [x] **Proactive context (auto-injected via MCP instructions)**
- [ ] Entity deduplication
- [ ] Obsidian plugin (TypeScript)
- [ ] Web dashboard
- [ ] REST API

## Contributing

```bash
git clone https://github.com/alibaizhanov/mengram
cd mengram
pip install -e ".[all,dev]"
pytest
```

## License

MIT

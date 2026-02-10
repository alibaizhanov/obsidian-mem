# 🧠 ObsidianMem

**AI memory as a typed knowledge graph in Obsidian.**

Every conversation with your AI builds a structured second brain — people, projects, technologies, companies — all as `.md` files with `[[wikilinks]]` you can browse in [Obsidian](https://obsidian.md).

Like [Mem0](https://github.com/mem0ai/mem0), but **you own your data**.

---

## Why ObsidianMem?

|  | **Mem0** | **Basic Memory** | **ObsidianMem** |
|---|---|---|---|
| Storage | Cloud vectors | Flat markdown | **Typed knowledge graph in .md** |
| Entity types | ❌ Flat facts | ❌ One note per chat | ✅ Person, Project, Technology, Company |
| Relations | ❌ | ❌ | ✅ `works_at`, `uses`, `depends_on` |
| Obsidian graph | ❌ | Partial | ✅ Full `[[wikilinks]]` + graph view |
| Semantic search | ✅ Cloud | ❌ | ✅ Local embeddings (384D) |
| Own your data | ❌ Cloud lock-in | ✅ | ✅ Plain `.md` files |
| LLM agnostic | ❌ | Partial | ✅ Claude / GPT / Ollama |
| Pricing | $24/mo+ | $14/mo | **Free & open source** |

### What it actually does

You chat with Claude (or any LLM). ObsidianMem **automatically**:

1. **Extracts** entities, facts, and relationships from your conversations
2. **Creates** typed `.md` files in your Obsidian vault
3. **Links** everything with `[[wikilinks]]` and YAML frontmatter
4. **Indexes** with local vector embeddings for semantic search
5. **Recalls** relevant context when you need it — by meaning, not just keywords

```
You: "I work at Uzum Bank, backend developer on Spring Boot"
                    ↓ LLM extracts knowledge
         ┌─────────────────────────────┐
         │  vault/Ali.md               │
         │  type: person               │
         │  - backend developer        │
         │  - → works_at [[Uzum Bank]] │
         │  - → uses [[Spring Boot]]   │
         └─────────────────────────────┘
         ┌──────────────────────────────┐
         │  vault/Uzum Bank.md          │
         │  type: company               │
         │  - ← works_at [[Ali]]        │
         └──────────────────────────────┘
```

Open in Obsidian → see your knowledge graph growing from every conversation.

---

## Quick Start

### 1. Install

```bash
pip install obsidian-mem[all]
```

### 2. Setup (one command)

```bash
obsidian-mem init
```

This will:
- Ask for your LLM provider and API key
- Create `~/.obsidian-mem/config.yaml` and vault
- Auto-configure Claude Desktop MCP integration
- Tell you to restart Claude Desktop

That's it. Talk to Claude — it remembers and recalls automatically.

### Non-interactive:

```bash
obsidian-mem init --provider anthropic --api-key sk-ant-...
```

### Other commands:

```bash
obsidian-mem status    # Check setup
obsidian-mem stats     # Vault statistics
obsidian-mem server    # Start MCP server manually
```

---

### Python SDK (Mem0-compatible API)

```python
from obsidian_mem import Memory

m = Memory(
    vault_path="./my-brain",
    llm_provider="anthropic",
    api_key="sk-ant-..."
)

# Remember
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

### MCP Server (Claude Desktop)

`obsidian-mem init` sets this up automatically. Manual setup:

```bash
obsidian-mem server --config ~/.obsidian-mem/config.yaml
```

### Auto-Memory Middleware

Drop-in wrapper that automatically remembers and recalls:

```python
from obsidian_mem import Memory
from obsidian_mem_middleware import AutoMemory

m = Memory(vault_path="./vault", llm_provider="anthropic", api_key="sk-ant-...")
auto = AutoMemory(memory=m, user_id="ali")

# Automatically: recall → inject context → LLM → remember
response = auto.chat("Help me fix the PostgreSQL connection pool issue")
```

---

## How It Works

```
Conversation → Extractor (LLM) → Entities + Facts + Relations
                                          ↓
                                   Vault Manager → .md files (Obsidian)
                                          ↓
                                   Vector Index → embeddings (SQLite)
                                          ↓
                                   Recall: Vector Search + Graph Expansion
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
- ← uses [[Ali]]: Primary expertise
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
| Anthropic (Claude) | `pip install obsidian-mem[anthropic]` | API pricing |
| OpenAI (GPT) | `pip install obsidian-mem[openai]` | API pricing |
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
- [ ] Entity deduplication
- [ ] Obsidian plugin (TypeScript)
- [ ] Web dashboard
- [ ] REST API

## Contributing

```bash
git clone https://github.com/alibaizhanov/obsidian-mem
cd obsidian-mem
pip install -e ".[all,dev]"
pytest
```

## License

MIT

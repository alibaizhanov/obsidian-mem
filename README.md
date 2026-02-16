# Mengram — AI Memory Layer with Autonomous Agents

**Open-source memory layer for AI apps.** Not just storage — Mengram has autonomous agents that clean, analyze, and find hidden patterns in your knowledge.

[![PyPI](https://img.shields.io/pypi/v/mengram-ai)](https://pypi.org/project/mengram-ai/)
[![npm](https://img.shields.io/npm/v/mengram-ai)](https://www.npmjs.com/package/mengram-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**[Website](https://mengram.io)** · **[Dashboard](https://mengram.io/dashboard)** · **[API Docs](https://mengram.io/docs)** · **[PyPI](https://pypi.org/project/mengram-ai/)** · **[npm](https://www.npmjs.com/package/mengram-ai)**

---

## Why Mengram?

|  | Mengram | Mem0 |
|---|---|---|
| Memory Storage | ✅ | ✅ |
| Semantic Search | ✅ | ✅ |
| Knowledge Graph | ✅ | ✅ |
| **Autonomous Agents** | ✅ Curator, Connector, Digest | ❌ |
| **Team Shared Memory** | ✅ Invite codes, privacy controls | ❌ |
| **AI Reflections** | ✅ Patterns, insights, behavioral analysis | ❌ |
| Webhooks | ✅ | ✅ |
| MCP Server | ✅ Claude Desktop, Cursor, Windsurf | ✅ |
| Python SDK | ✅ | ✅ |
| JavaScript SDK | ✅ | ✅ |
| Async Mode | ✅ Job tracking | ✅ |
| Memory Categories | ✅ user_id, agent_id, run_id, app_id | ✅ |
| Self-hostable | ✅ | ✅ |
| **Price** | **Free** | $19-249/mo |

## Quick Start (60 seconds)

### 1. Get API key
Sign up at [mengram.io](https://mengram.io) — free, no credit card.

### 2. Install
```bash
pip install mengram-ai    # Python
npm install mengram-ai    # JavaScript / TypeScript
```

### 3. Connect to Claude Desktop
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "mengram": {
      "command": "mengram",
      "args": ["server", "--cloud"],
      "env": {
        "MENGRAM_API_KEY": "your-key-here"
      }
    }
  }
}
```

Done. Claude now has persistent memory.

## Python SDK

```python
from mengram.cloud.client import CloudMemory

m = CloudMemory(api_key="mg-...")

# Add memories (async — returns immediately)
result = m.add([
    {"role": "user", "content": "I deployed Mengram on Railway with PostgreSQL 15"},
    {"role": "assistant", "content": "Great, noted the deployment setup."}
], user_id="ali", agent_id="devops-bot", app_id="infra")

# Track background job
job = m.wait_for_job(result["job_id"])

# Semantic search (filter by agent)
results = m.search("deployment setup", user_id="ali", agent_id="devops-bot")

# Run memory agents
m.run_agents(agent="all", auto_fix=True)

# Get AI insights
insights = m.insights()

# Team memory
team = m.create_team("Backend Team")
m.share_memory("Redis", team_id=team["id"])

# Webhooks
m.create_webhook(url="https://hooks.slack.com/...", name="Slack")

# Manage API keys
m.create_key(name="production")
keys = m.list_keys()
```

## JavaScript / TypeScript SDK

```javascript
const { MengramClient } = require('mengram-ai');

const m = new MengramClient('mg-...');

// Add memories with categories
const { job_id } = await m.add([
  { role: 'user', content: 'I prefer dark mode and use Vim' },
  { role: 'assistant', content: 'Noted your preferences!' }
], { userId: 'ali', agentId: 'support-bot', appId: 'helpdesk' });

// Wait for completion
const result = await m.waitForJob(job_id);

// Semantic search
const results = await m.search('editor preferences', { userId: 'ali' });

// Agents, teams, webhooks — all supported
await m.runAgents({ agent: 'curator', autoFix: true });
const teams = await m.listTeams();
```

Full TypeScript types included (`index.d.ts`).

## Memory Categories

Separate memory by user, agent, session, and application:

```python
# User's personal memory
m.add(messages, user_id="ali")

# Specific agent's memory
m.add(messages, user_id="ali", agent_id="support-bot")

# Session-scoped memory
m.add(messages, user_id="ali", run_id="session-123")

# Application-scoped memory
m.add(messages, user_id="ali", app_id="helpdesk")

# Search filters by category too
m.search("preferences", user_id="ali", agent_id="support-bot")
```

## Memory Agents

Three autonomous agents that analyze your memory:

**🧹 Curator** — Finds contradictions, stale facts, duplicates. Auto-cleans with `auto_fix=True`.

**🔗 Connector** — Discovers hidden connections, behavioral patterns, skill clusters.

**📰 Digest** — Weekly summary with headlines, trends, and recommendations.

## Team Shared Memory

Share knowledge across your team:

```bash
POST /v1/teams {"name": "Backend Team"}
POST /v1/teams/join {"invite_code": "xK9m2Qw5ab"}
POST /v1/teams/3/share {"entity": "Redis"}
```

Search automatically includes shared team knowledge.

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/add` | Add memories (async, returns job_id) |
| `GET /v1/jobs/{id}` | Check background job status |
| `POST /v1/search` | Semantic search |
| `POST /v1/agents/run` | Run memory agents |
| `GET /v1/insights` | AI-generated insights |
| `POST /v1/teams` | Create team |
| `POST /v1/webhooks` | Create webhook |
| `GET /v1/keys` | List API keys |
| `POST /v1/keys` | Create new API key |
| `GET /v1/graph` | Knowledge graph |
| `GET /v1/timeline` | Temporal search |
| `GET /v1/stats` | Usage statistics |

Full docs: **https://mengram.io/docs**

## Architecture

```
┌──────────────────────────────────────┐
│          Your AI Clients             │
│  Claude Desktop · Cursor · Windsurf  │
└──────────────┬───────────────────────┘
               │ MCP / REST API
┌──────────────▼───────────────────────┐
│        Mengram Cloud API             │
│  Extraction · Re-ranking · Search    │
├──────────────────────────────────────┤
│       Memory Agents Layer            │
│  🧹 Curator · 🔗 Connector · 📰 Digest│
├──────────────────────────────────────┤
│       Storage Layer                  │
│  PostgreSQL · pgvector · Teams       │
│  Webhooks · Reflections · Graph      │
└──────────────────────────────────────┘
```

## License

MIT

---

Built by **[Ali Baizhanov](https://github.com/alibaizhanov)**

# Mengram — Human-Like Memory for AI

**The only AI memory API with 3 memory types: semantic, episodic, and procedural.** Your AI remembers facts, events, and learned workflows — just like a human brain.

[![PyPI](https://img.shields.io/pypi/v/mengram-ai)](https://pypi.org/project/mengram-ai/)
[![npm](https://img.shields.io/npm/v/mengram-ai)](https://www.npmjs.com/package/mengram-ai)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**[Website](https://mengram.io)** · **[Dashboard](https://mengram.io/dashboard)** · **[API Docs](https://mengram.io/docs)** · **[PyPI](https://pypi.org/project/mengram-ai/)** · **[npm](https://www.npmjs.com/package/mengram-ai)**

---

## Why Mengram?

|  | Mengram | Mem0 | Supermemory |
|---|---|---|---|
| Semantic Memory (facts) | ✅ | ✅ | ✅ |
| **Episodic Memory (events)** | ✅ | ❌ | ❌ |
| **Procedural Memory (workflows)** | ✅ | ❌ | ❌ |
| **Cognitive Profile** | ✅ | ❌ | ❌ |
| **Smart Triggers** | ✅ | ❌ | ❌ |
| **Unified Search (all 3 types)** | ✅ | ❌ | ❌ |
| Knowledge Graph | ✅ | ✅ | ❌ |
| Autonomous Agents | ✅ Curator, Connector, Digest | ❌ | ❌ |
| Team Shared Memory | ✅ | ❌ | ✅ |
| AI Reflections | ✅ | ❌ | ❌ |
| Webhooks | ✅ | ✅ | ✅ |
| MCP Server | ✅ Claude Desktop, Cursor, Windsurf | ✅ | ❌ |
| **LangChain Integration** | ✅ | ❌ | ❌ |
| Python & JS SDK | ✅ | ✅ | ✅ |
| Self-hostable | ✅ | ✅ | ✅ |
| **Price** | **Free** | $19-249/mo | Enterprise |

## 3 Memory Types

Mengram automatically extracts all 3 types from a single `add()` call:

**🧠 Semantic** — Facts, preferences, skills: *"uses Python"*, *"prefers dark mode"*

**📝 Episodic** — Events, decisions, experiences: *"Debugged Railway deployment for 3 hours, fixed pgvector issue"*

**⚙️ Procedural** — Learned workflows, processes: *"Deploy: build → twine upload → npm publish → git push"*

```python
# One call extracts all 3 types automatically
m.add([
    {"role": "user", "content": "Fixed the auth bug today. Problem was API key cache TTL. My debug process: check Railway logs, reproduce locally, fix and deploy."},
])
# → Semantic: "API key caching caused auth bug"
# → Episodic: "Debugged auth bug, fixed cache TTL"  
# → Procedural: "Debug process: logs → reproduce → fix → deploy"
```

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

Done. Claude now has persistent memory with all 3 types.

## Authentication

All API calls are authenticated via your API key. The key identifies your account — all memories are automatically tied to it. No need to pass `user_id` separately.

```python
m = CloudMemory(api_key="om-...")  # Your key = your identity
m.add([...])                        # Memories saved to your account
m.search("query")                   # Searches your memories
```

## Python SDK

```python
from mengram.cloud.client import CloudMemory

m = CloudMemory(api_key="om-...")

# Add memories — auto-extracts facts, events, workflows
m.add([
    {"role": "user", "content": "I deployed Mengram on Railway with PostgreSQL 15"},
    {"role": "assistant", "content": "Great, noted the deployment setup."}
])

# Semantic search (classic)
results = m.search("deployment setup")

# Episodic search — what happened?
events = m.episodes(query="deployment")
# → [{summary: "Deployed on Railway", outcome: "Success", participants: [...]}]

# Procedural search — how to do it?
procs = m.procedures(query="deploy")
# → [{name: "Deploy Mengram", steps: [...], success_count: 5}]

# Unified search — all 3 types at once
all_results = m.search_all("deployment issues")
# → {semantic: [...], episodic: [...], procedural: [...]}

# Procedure feedback — AI learns what works
m.procedure_feedback(proc_id, success=True)

# Cognitive Profile — instant personalization
profile = m.get_profile()
# → {system_prompt: "You are talking to Ali, a developer in Almaty..."}

# Smart Triggers — proactive memory alerts
triggers = m.get_triggers()
# → [{"type": "reminder", "title": "Meeting with Anya at 3pm", ...}]

# Memory agents
m.run_agents(agent="all", auto_fix=True)

# Team memory
team = m.create_team("Backend Team")
m.share_memory("Redis", team_id=team["id"])
```

## JavaScript / TypeScript SDK

```javascript
const { MengramClient } = require('mengram-ai');

const m = new MengramClient('om-...');

// Add memories — extracts all 3 types
await m.add([
  { role: 'user', content: 'Fixed OOM with Redis cache' },
]);

// Episodic — what happened?
const events = await m.episodes({ query: 'OOM fix' });

// Procedural — how to do it?
const procs = await m.procedures({ query: 'cache setup' });

// Unified search — all 3 types
const all = await m.searchAll('database issues');
// → { semantic: [...], episodic: [...], procedural: [...] }

// Procedure feedback — AI learns
await m.procedureFeedback(procId, { success: true });

// Cognitive Profile
const profile = await m.getProfile();

// Smart Triggers
const triggers = await m.getTriggers();
```

Full TypeScript types included with `Episode`, `Procedure`, `SmartTrigger`, and `UnifiedSearchResult` interfaces.

## Cognitive Profile

One API call generates a ready-to-use system prompt from all 3 memory types:

```python
profile = m.get_profile()
print(profile["system_prompt"])
```

Output:
```
You are talking to Ali, a 22-year-old developer in Almaty building Mengram.
He uses Python, PostgreSQL, and Railway. Recently: debugged pgvector deployment,
researched competitors Mem0 and Supermemory, designed freemium pricing.
Workflows: deploys via build→twine→npm→git, prefers iterative shipping.
Communicate in Russian/English, direct style, focus on practical next steps.
```

Insert into any LLM's system prompt for instant personalization. Replace your RAG pipeline.

## Smart Triggers

Memory that proactively alerts you. Mengram automatically detects:

- **Reminders** — "meeting with Sarah at 3pm tomorrow" → triggers reminder 1h before
- **Contradictions** — "user is vegetarian" + "order steaks for dinner" → contradiction alert
- **Patterns** — "deploy on Friday" + 3/5 Friday deploys had bugs → risk warning

```python
triggers = m.get_triggers()
# [{"type": "reminder", "title": "Meeting with Sarah at 3pm", ...}]

m.process_triggers()       # Fire all pending triggers
m.dismiss_trigger(42)      # Dismiss a trigger
```

Triggers fire automatically via background cron (every 5 min) and send through your configured webhooks. Works with OpenClaw, Slack, Discord — any webhook endpoint.

## LangChain Integration

Drop-in replacement for LangChain's memory with all 3 memory types.

```bash
pip install mengram-ai[langchain]
```

**LCEL (recommended):**
```python
from mengram.integrations.langchain import MengramChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: MengramChatMessageHistory(
        api_key="om-...", session_id=session_id
    ),
    input_messages_key="input",
    history_messages_key="history",
)
```

**ConversationChain (legacy):**
```python
from mengram.integrations.langchain import MengramMemory

memory = MengramMemory(api_key="om-...")
# With Cognitive Profile:
memory = MengramMemory(api_key="om-...", use_profile=True)

chain = ConversationChain(llm=llm, memory=memory)
chain.predict(input="I deployed my app on Railway")
chain.predict(input="How did my last deployment go?")
# → Memory provides: facts about Railway, the deployment event, deploy workflow
```

**vs ConversationBufferMemory:**
| | ConversationBufferMemory | MengramMemory |
|---|---|---|
| Storage | RAM (lost on restart) | Persistent (PostgreSQL) |
| Context | Last N messages (raw) | Relevant knowledge (semantic search) |
| Memory types | 1 (messages) | 3 (semantic + episodic + procedural) |
| Cross-session | ❌ | ✅ |
| Personalization | ❌ | ✅ Cognitive Profile |

## CrewAI Integration

Give your CrewAI agents persistent memory with 3 types + procedural learning.

```bash
pip install mengram-ai[crewai]
```

```python
from crewai import Agent, Crew
from mengram.integrations.crewai import create_mengram_tools

tools = create_mengram_tools(api_key="om-...")

agent = Agent(
    role="Support Engineer",
    goal="Help users with technical issues",
    tools=tools,  # mengram_search, mengram_remember, mengram_profile,
                   # mengram_save_workflow, mengram_workflow_feedback
)

crew = Crew(agents=[agent], tasks=[...])
```

**Killer Feature — Procedural Learning:**

Agent completes a multi-step workflow → Mengram saves it as a procedure → Next similar task → agent finds the optimal path with success/failure tracking.

**vs CrewAI Default Memory:**
| | CrewAI Default | Mem0 + CrewAI | Mengram + CrewAI |
|---|---|---|---|
| Storage | Local files | Cloud | Cloud |
| Memory types | 3 (basic) | 1 (semantic) | 3 (semantic+episodic+procedural) |
| Cross-session | Partial | ✅ | ✅ |
| Workflow learning | ❌ | ❌ | ✅ Procedural memory |
| User profile | ❌ | ❌ | ✅ Cognitive Profile |
| Success tracking | ❌ | ❌ | ✅ per procedure |

## OpenClaw Skill

Give your OpenClaw agent long-term memory across WhatsApp, Telegram, Discord, Slack.

```bash
# Install from ClawHub
clawdhub install mengram-openclaw-skill

# Or copy manually
cp -r integrations/openclaw ~/.openclaw/skills/mengram-memory
```

Add to `~/.openclaw/openclaw.json`:
```json
{
  "skills": {
    "entries": {
      "mengram-memory": {
        "enabled": true,
        "env": {
          "MENGRAM_API_KEY": "om-your-key"
        }
      }
    }
  }
}
```

Your agent automatically searches memory before answering, saves new facts/events/workflows, and loads your Cognitive Profile at session start.

## Memory Categories

Separate memory by agent, session, and application:

```python
m.add(messages)                                    # Your memory
m.add(messages, agent_id="support-bot")            # Agent's memory
m.add(messages, run_id="session-123")              # Session-scoped
m.add(messages, app_id="helpdesk")                 # App-scoped
```

## Memory Agents

**🧹 Curator** — Finds contradictions, stale facts, duplicates. Auto-cleans with `auto_fix=True`.

**🔗 Connector** — Discovers hidden connections, behavioral patterns, skill clusters.

**📰 Digest** — Weekly summary with headlines, trends, and recommendations.

## API Endpoints

All endpoints require `Authorization: Bearer om-...` header. Your API key identifies you — no `user_id` needed.

| Endpoint | Description |
|---|---|
| `POST /v1/add` | Add memories (auto-extracts all 3 types) |
| `POST /v1/search` | Semantic search |
| `POST /v1/search/all` | **Unified search (semantic + episodic + procedural)** |
| `GET /v1/episodes` | List episodic memories |
| `GET /v1/episodes/search` | Search episodes by meaning |
| `GET /v1/procedures` | List procedural memories |
| `GET /v1/procedures/search` | Search procedures by trigger |
| `PATCH /v1/procedures/{id}/feedback` | Record success/failure |
| `GET /v1/profile` | **Cognitive Profile (system prompt)** |
| `GET /v1/triggers` | **Smart Triggers (reminders, contradictions, patterns)** |
| `POST /v1/triggers/process` | Fire all pending triggers |
| `DELETE /v1/triggers/{id}` | Dismiss a trigger |
| `POST /v1/agents/run` | Run memory agents |
| `GET /v1/insights` | AI-generated insights |
| `GET /v1/graph` | Knowledge graph |
| `GET /v1/timeline` | Temporal search |
| `POST /v1/teams` | Create team |
| `POST /v1/webhooks` | Create webhook |
| `GET /v1/keys` | List API keys |
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
│       3 Memory Types                 │
│  🧠 Semantic · 📝 Episodic · ⚙️ Proc │
├──────────────────────────────────────┤
│       Smart Triggers                 │
│  🔔 Reminders · ⚠️ Contradictions    │
│  📊 Patterns · Background Cron       │
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

Apache 2.0

---

Built by **[Ali Baizhanov](https://github.com/alibaizhanov)**

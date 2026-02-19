# mengram-ai

JavaScript / TypeScript SDK for [Mengram](https://mengram.io) — Human-like memory for AI with 3 memory types: semantic, episodic, and procedural.

## Install

```bash
npm install mengram-ai
```

## Quick Start

```javascript
const { MengramClient } = require('mengram-ai');

const m = new MengramClient('om-your-api-key');

// Add memories — auto-extracts facts, events, workflows
await m.add([
  { role: 'user', content: 'Fixed the auth bug. My process: check logs, reproduce locally, fix and deploy.' },
], { userId: 'ali' });

// Semantic search (classic)
const results = await m.search('auth issues', { userId: 'ali' });

// Episodic — what happened?
const events = await m.episodes({ query: 'auth bug' });
// → [{summary: "Fixed auth bug", outcome: "Resolved", participants: [...]}]

// Procedural — how to do it?
const procs = await m.procedures({ query: 'debug' });
// → [{name: "Debug process", steps: [...], success_count: 3}]

// Unified search — all 3 types at once
const all = await m.searchAll('deployment issues');
// → { semantic: [...], episodic: [...], procedural: [...] }

// Procedure feedback — AI learns what works
await m.procedureFeedback(procId, { success: true });

// Experience-driven evolution — procedure improves on failure
await m.procedureFeedback(procId, {
  success: false, context: 'OOM on step 3', failedAtStep: 3
});

// View procedure version history
const history = await m.procedureHistory(procId);
// → { versions: [v1, v2, v3], evolution_log: [...] }

// Cognitive Profile — instant personalization
const profile = await m.getProfile('ali');
// → { system_prompt: "You are talking to Ali, a developer..." }
```

## TypeScript

```typescript
import { MengramClient, SearchResult, Episode, Procedure, UnifiedSearchResult, ProcedureHistoryResult } from 'mengram-ai';

const m = new MengramClient('om-...');

const results: SearchResult[] = await m.search('preferences');
const events: Episode[] = await m.episodes({ query: 'deployment' });
const procs: Procedure[] = await m.procedures({ query: 'release' });
const all: UnifiedSearchResult = await m.searchAll('issues');
```

## API

| Method | Description |
|--------|-------------|
| `add(messages, options?)` | Add memories (extracts all 3 types) |
| `addText(text, options?)` | Add memories from plain text |
| `search(query, options?)` | Semantic search |
| `searchAll(query, options?)` | **Unified search (all 3 types)** |
| `episodes(options?)` | **Search/list episodic memories** |
| `procedures(options?)` | **Search/list procedural memories** |
| `procedureFeedback(id, options?)` | **Record success/failure (triggers evolution on failure)** |
| `procedureHistory(id)` | **Version history + evolution log** |
| `procedureEvolution(id)` | **Evolution log (what changed and why)** |
| `getProfile(userId?, options?)` | **Cognitive Profile** |
| `getAll(options?)` | List all memories |
| `get(name)` | Get specific entity |
| `delete(name)` | Delete entity |
| `runAgents(options?)` | Run memory agents |
| `insights()` | AI reflections |
| `createTeam(name)` | Create shared team |
| `joinTeam(code)` | Join team |
| `shareMemory(entity, teamId)` | Share with team |

## Links

- [Website](https://mengram.io)
- [Documentation](https://mengram.io/docs)
- [Python SDK](https://pypi.org/project/mengram-ai/)
- [GitHub](https://github.com/alibaizhanov/mengram)

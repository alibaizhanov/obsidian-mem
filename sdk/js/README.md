# mengram-ai

JavaScript / TypeScript SDK for [Mengram](https://mengram.io) — AI Memory Layer with Autonomous Agents.

## Install

```bash
npm install mengram-ai
```

## Quick Start

```javascript
const { MengramClient } = require('mengram-ai');

const m = new MengramClient('mg-your-api-key');

// Add memories from conversation
await m.add([
  { role: 'user', content: 'I work at Acme Corp as a senior engineer. I prefer dark mode.' },
  { role: 'assistant', content: 'Noted your preferences!' }
], { userId: 'ali' });

// Search memories
const results = await m.search('work preferences', { userId: 'ali' });
console.log(results);
// [{ entity: 'User', type: 'person', score: 0.92, facts: ['Works at Acme Corp', 'Prefers dark mode'] }]

// Multi-agent memory
await m.add(messages, {
  userId: 'ali',
  agentId: 'support-bot',
  appId: 'helpdesk'
});
```

## TypeScript

```typescript
import { MengramClient, SearchResult } from 'mengram-ai';

const m = new MengramClient('mg-...');
const results: SearchResult[] = await m.search('preferences');
```

## API

| Method | Description |
|--------|-------------|
| `add(messages, options?)` | Add memories from conversation |
| `addText(text, options?)` | Add memories from plain text |
| `search(query, options?)` | Semantic search |
| `getAll(options?)` | List all memories |
| `get(name)` | Get specific entity |
| `delete(name)` | Delete entity |
| `stats()` | Usage statistics |
| `graph()` | Knowledge graph |
| `runAgents(options?)` | Run memory agents |
| `insights()` | AI-generated reflections |
| `createTeam(name)` | Create shared team |
| `joinTeam(code)` | Join team with invite code |
| `shareMemory(entity, teamId)` | Share memory with team |
| `listKeys()` | List API keys |
| `createKey(name?)` | Create new API key |

All methods support `userId`, `agentId`, `runId`, `appId` options for multi-agent systems.

## Links

- [Documentation](https://mengram.io/docs)
- [Python SDK](https://pypi.org/project/mengram-ai/)
- [GitHub](https://github.com/alibaizhanov/mengram)

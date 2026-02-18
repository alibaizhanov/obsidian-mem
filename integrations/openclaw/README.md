# 🧠 Mengram Memory — OpenClaw Skill

**Give your OpenClaw agent human-like long-term memory with 3 types.**

Your agent remembers facts, events, and learned workflows across all sessions and channels. What it learns on WhatsApp is available on Discord.

## Why This Exists

OpenClaw's built-in memory is a flat text buffer. New session = mostly blank slate. Mengram adds structured, persistent memory that makes your agent genuinely personal over time.

| | Without Mengram | With Mengram |
|---|---|---|
| "Book the usual restaurant" | "Which restaurant? What time? How many?" | "Booking Kaganat at 7pm for 2. Vegan menu for Anya. Confirm?" |
| New session | Starts fresh | Knows your preferences, history, workflows |
| After 100 conversations | Same as day 1 | Deep understanding of who you are |

## Memory Types

- **Semantic** — facts: "Ali is a developer", "prefers oat milk lattes", "allergic to peanuts"
- **Episodic** — events: "deployed v2.6 on Feb 18, had 2 bugs", "meeting with Sarah went well"
- **Procedural** — workflows: "deploy process: test → build → push → update DNS" (with success/failure tracking)

## Install

```bash
# Copy skill folder to OpenClaw skills directory
cp -r mengram-memory ~/.openclaw/skills/
```

Or if published to ClawHub:
```bash
npx clawdhub@latest install mengram-memory
```

## Setup

1. Get a free API key at [mengram.io](https://mengram.io)

2. Add to `~/.openclaw/openclaw.json`:

```json
{
  "skills": {
    "entries": {
      "mengram-memory": {
        "enabled": true,
        "env": {
          "MENGRAM_API_KEY": "om-your-api-key-here"
        }
      }
    }
  }
}
```

3. Start a new OpenClaw session. The skill loads automatically.

4. Verify: ask your agent to run the setup check, or type `/mengram-memory` to invoke the skill.

## How It Works

The agent automatically:
1. **Searches memory** before answering personal questions
2. **Saves to memory** when you share new info
3. **Loads your profile** at the start of sessions
4. **Saves workflows** after completing multi-step tasks

You don't need to do anything special. Just talk naturally.

## Scripts

| Script | Purpose |
|---|---|
| `mengram-search.sh` | Search all 3 memory types |
| `mengram-add.sh` | Save text to memory (auto-extracts facts/events/procedures) |
| `mengram-profile.sh` | Get full Cognitive Profile |
| `mengram-workflow.sh` | Save completed workflow as reusable procedure |
| `mengram-setup.sh` | Verify connection and API key |

## Killer Feature: Procedural Learning

Your agent completes a task → Mengram saves the steps as a procedure → Next time a similar task comes up → agent finds the proven workflow with success/failure stats.

```
Day 1: You ask to deploy. Agent figures it out step by step.
Day 2: You ask to deploy. Agent already knows: test → build → push → update DNS (3 successes, 0 failures)
```

No other memory system does this.

## Security

- Only calls `mengram.io` API (open-source backend)
- Uses `curl` — no pip packages, no npm, no extra dependencies
- API key stays in env vars, never logged or exposed
- All scripts have security manifests
- Mengram is open-source: [github.com/AiBaizhanov/mengram](https://github.com/AiBaizhanov/mengram)

## Links

- **Mengram**: [mengram.io](https://mengram.io)
- **GitHub**: [github.com/AiBaizhanov/mengram](https://github.com/AiBaizhanov/mengram)
- **API Docs**: [mengram.io/docs](https://mengram.io/docs)

## License

Apache 2.0

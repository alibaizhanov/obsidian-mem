---
name: mengram-memory
description: Long-term memory with 3 types (facts, events, workflows). Remember user preferences, past conversations, and learned procedures across sessions. Use when recalling what the user said before, saving important info, getting user context, or tracking completed workflows.
homepage: https://github.com/AiBaizhanov/mengram
metadata: {"openclaw":{"emoji":"🧠","requires":{"env":["MENGRAM_API_KEY"],"bins":["curl","python3"]},"primaryEnv":"MENGRAM_API_KEY","files":["scripts/*"]}}
---

# Mengram Memory

Human-like long-term memory for your OpenClaw agent. Remembers facts, events, and workflows across all sessions and channels.

## What This Skill Does

Mengram gives you 3 types of memory that work together:

- **Semantic Memory** — facts about the user (preferences, relationships, work, habits)
- **Episodic Memory** — events with timestamps and outcomes (meetings, trips, decisions)
- **Procedural Memory** — learned workflows with success/failure tracking (how-to steps)

All memory persists across sessions and channels. What you learn on WhatsApp is available on Discord.

## Tools

This skill uses `Bash` to run scripts in `{baseDir}/scripts/`.

## When To Use

**ALWAYS search memory before answering questions** about the user, their preferences, their history, or anything personal. This is critical — check memory first, then respond.

**Save to memory after:**
- The user shares personal information (name, preferences, habits, relationships)
- A task or event is completed (booking, deployment, purchase)
- A multi-step workflow is finished successfully or fails
- The user corrects you or provides new context

**Get profile when:**
- Starting a new session or conversation
- The user asks "what do you know about me"
- You need comprehensive context about the user

## Scripts

### 1. Search Memory

Search all 3 memory types for relevant context. **Do this before answering personal questions.**

```bash
bash {baseDir}/scripts/mengram-search.sh "search query here"
```

Returns facts, past events, and known workflows matching the query. Use specific queries — "coffee preferences" not "stuff about user".

### 2. Save to Memory

Save conversation messages so Mengram can extract facts, events, and procedures automatically.

```bash
bash {baseDir}/scripts/mengram-add.sh "user said: I'm allergic to peanuts and my meeting with Sarah went well yesterday"
```

You can pass multiple messages. Mengram's AI automatically extracts:
- Facts → semantic memory ("user is allergic to peanuts")
- Events → episodic memory ("meeting with Sarah, yesterday, went well")
- Procedures → procedural memory (if workflow steps are described)

### 3. Get Cognitive Profile

Get a comprehensive portrait of the user — who they are, what they know, recent events, known workflows.

```bash
bash {baseDir}/scripts/mengram-profile.sh
```

Returns a full context block you can use to personalize responses.

### 4. Save Workflow

After completing a multi-step task, save it as a reusable procedure with success/failure tracking.

```bash
bash {baseDir}/scripts/mengram-workflow.sh "Resolved billing issue: 1) Checked subscription status 2) Found expired card 3) Sent renewal link 4) User confirmed payment"
```

Next time a similar task comes up, `mengram-search.sh` will return this workflow with its success rate.

### 5. Setup Check

Verify the Mengram connection is working:

```bash
bash {baseDir}/scripts/mengram-setup.sh
```

## Recommended Behavior

1. **Start of session:** Run `mengram-profile.sh` to load user context
2. **User asks something personal:** Run `mengram-search.sh "topic"` before answering
3. **User shares new info:** Run `mengram-add.sh` with the relevant messages
4. **After completing a task:** Run `mengram-workflow.sh` with the steps taken
5. **Periodically:** Run `mengram-add.sh` with recent conversation highlights to keep memory updated

## Examples

- "What's my favorite restaurant?" → `mengram-search.sh "favorite restaurant"`
- "Book the usual" → `mengram-search.sh "booking usual preferences"` to find what "the usual" means
- "I just switched to a new phone, Galaxy S26" → `mengram-add.sh "user switched to Samsung Galaxy S26"`
- "Remember that I'm vegetarian" → `mengram-add.sh "user is vegetarian"`
- User asks "what do you know about me?" → `mengram-profile.sh`

## Configuration

Set in `~/.openclaw/openclaw.json`:

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

Get your free API key at https://mengram.io

## Security & Privacy

- **External endpoint:** `https://mengram.io/v1/*` (only)
- **Data sent:** conversation text for memory extraction, search queries
- **Data stored:** extracted facts, events, and procedures on Mengram servers (PostgreSQL + pgvector)
- **Environment variables accessed:** `MENGRAM_API_KEY` (only)
- **Local files read/written:** none

**Trust Statement:** By using this skill, conversation data is sent to mengram.io for memory extraction and storage. Only install if you trust Mengram with your conversation data. Mengram is open-source: https://github.com/AiBaizhanov/mengram

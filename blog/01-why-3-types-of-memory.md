---
title: "Why AI Agents Need 3 Types of Memory (Not Just Vector Search)"
published: false
description: "Your agent remembers facts but forgets what happened yesterday and can't repeat what it's done 100 times. The fix is 50 years old — cognitive science solved this in 1972."
tags: ai, agents, python, machinelearning
cover_image:
canonical_url:
---

Your agent remembers that the database runs on PostgreSQL. It cannot tell you what happened during last Friday's deploy. And it has no idea how to perform a deployment it has executed a hundred times before.

It knows facts. It forgets events. It cannot learn skills.

This is the state of AI memory in 2026. Every major framework — Mem0, Zep, LangChain's memory modules — implements one type of memory: store text, embed it, retrieve by similarity. Vector search over facts. That's it.

Cognitive science figured out this was insufficient in 1972.

---

## Tulving's taxonomy: three types of memory

In 1972, Endel Tulving proposed that human long-term memory is not a single system. It divides into three distinct types:

**Semantic memory** — general knowledge and facts. "Paris is the capital of France." "PostgreSQL uses MVCC for concurrency." Context-free, timeless.

**Episodic memory** — personal experiences anchored in time and place. "Last Friday, I deployed the payments service and production crashed because migrations were missing." It has a when, a who, a what-happened, and an outcome.

**Procedural memory** — knowing how to do things. Riding a bicycle. Typing on a keyboard. "How do we deploy to production?" It is sequential, skill-based, and improves with practice.

Humans use all three constantly. When you ask a senior engineer "should we deploy today?", they draw on semantic memory (facts about the system), episodic memory (what happened last time), and procedural memory (the steps to do it). The answer depends on all three.

Current AI memory tools give your agent the first one and call it done.

---

## Semantic memory: the baseline everyone has

Semantic memory stores facts. This is the part every tool gets right.

You embed text, store vectors, search by cosine similarity. "What database does the payments service use?" returns "PostgreSQL 15 with pgvector extension." It is entity-centric, declarative, and context-free.

This covers a real use case. Agents need to recall facts about users, systems, and configurations. But facts alone are insufficient. Here is why.

Ask your agent: "What happened the last time we deployed on a Friday?" A fact store has no answer. It might find a fact that says "deployments happen via Kubernetes" — but that says nothing about what actually happened.

Ask your agent: "Walk me through our deployment process, step by step." A fact store might return fragments: "we use Docker," "the registry is ECR," "tests run in CI." These are facts about the deployment. They are not the deployment procedure.

Vector search over facts cannot answer temporal or procedural questions because those are structurally different types of knowledge.

---

## Episodic memory: what happened, when, and what was the outcome

Episodic memory stores events — things that happened in a specific context, at a specific time, with specific participants and outcomes.

"Deployed the payments service on Friday at 3pm. Build passed. Staging looked fine. Production crashed 4 minutes after rollout — the `payment_metadata` column didn't exist. Root cause: database migration was skipped. Rolled back at 3:22pm. Postmortem scheduled for Monday."

This is not a fact. It is an episode. It has:

- **Temporal anchoring** — when it happened
- **Participants** — who was involved
- **Causality** — what caused what
- **Outcome** — how it resolved

Why does this matter for agents? Because agents that learn from experience need to recall past experiences. When a similar situation arises, the agent should retrieve relevant episodes and adjust behavior accordingly.

"We're about to deploy on Friday afternoon" should trigger recall of the last Friday deploy that went wrong. A fact store cannot make this connection because it does not store events as events.

Episodic memory also enables something fact stores cannot: reasoning about patterns over time. "The last three deploys that included schema changes all had issues" is a temporal pattern across episodes, not a static fact.

---

## Procedural memory: how to do things

Procedural memory stores step-by-step workflows — ordered sequences of actions with success criteria.

"How do we deploy the payments service?" is answered by:

1. Run the test suite
2. Check for pending database migrations
3. Run migrations if needed
4. Build the Docker image
5. Push to the registry
6. Deploy to staging
7. Run smoke tests
8. Deploy to production
9. Monitor for 15 minutes
10. Rollback if issues detected

This is not a set of facts. It is an ordered procedure with dependencies between steps. Step 3 must come before step 5. Step 8 depends on step 7 passing. The structure matters.

Fact-based memory systems cannot represent this. If you embed each step as a separate fact, you lose the ordering. If you embed the entire procedure as one chunk, you cannot update individual steps when something changes.

Procedural memory is what separates an agent that knows things from an agent that can do things.

---

## The missing piece: procedures that learn

Here is where it gets interesting. Human procedural memory improves with practice. You get faster, you skip unnecessary steps, you add checks where you have been burned before. The procedure evolves through experience.

Static runbooks do not do this. A runbook written six months ago stays the same regardless of how many times it has failed. Someone might update it manually after an outage. Usually they do not.

Experience-driven procedures close the feedback loop:

```
Procedure v1: build → push → deploy
                                 ↓ FAILURE: forgot migrations, DB crashed
Procedure v2: build → run migrations → push → deploy
                                                 ↓ FAILURE: OOM in production
Procedure v3: build → run migrations → check memory limits → push → deploy
```

When a procedure fails, the system records the failure as an episode, links it to the procedure and the specific step that failed, analyzes the root cause, and generates an improved version. The procedure gets better every time something goes wrong.

This is not theoretical. It is how experienced teams work — except the knowledge lives in people's heads and gets lost when they leave. Encoding it in a memory system that evolves automatically means the knowledge compounds instead of decaying.

---

## Code: all three types from one conversation

Here is how this works in practice. One `add()` call sends a conversation. The system extracts all three memory types automatically.

### Setup

```python
from cloud.client import CloudMemory

m = CloudMemory(api_key="om-...")  # Free key → mengram.io/dashboard
```

### Add a conversation — all 3 types extracted automatically

```python
result = m.add([
    {"role": "user", "content": "How did the payments deploy go?"},
    {"role": "assistant", "content": (
        "Rough. Deployed the payments service to production at 3pm. "
        "Build passed, staging was green, but production crashed 4 "
        "minutes after rollout. Root cause: the payment_metadata "
        "migration wasn't applied. Containers kept restarting because "
        "the column didn't exist. Rolled back at 3:22pm.\n\n"
        "Going forward, the deploy process should be:\n"
        "1. Run pytest with --timeout=300\n"
        "2. Check for pending migrations\n"
        "3. Run database migrations\n"
        "4. Build and push Docker image\n"
        "5. Deploy to staging, run smoke tests\n"
        "6. Deploy to production\n"
        "7. Monitor dashboards for 15 minutes"
    )},
])

# Extraction is async
m.wait_for_job(result["job_id"])
```

From this single conversation, three things are extracted:

- **Semantic** — facts like "payments service uses PostgreSQL" and "payment_metadata column exists"
- **Episodic** — the deployment failure event with timeline, root cause, and resolution
- **Procedural** — the 7-step deployment procedure

### Search each type independently

```python
# Semantic: what do we know?
facts = m.search("payments service database")
for f in facts:
    print(f"{f['entity']}: {f['facts']}")
# → PaymentsService: ["Uses PostgreSQL", "Has payment_metadata column", ...]

# Episodic: what happened?
events = m.episodes(query="deployment failure")
for e in events:
    print(f"{e['summary']}  [{e['outcome']}]")
# → "Payments service production deploy failed due to missing migration"  [rolled_back]

# Procedural: how do we do it?
procs = m.procedures(query="deploy payments")
for p in procs:
    print(f"{p['name']} (v{p['version']})")
    for step in p["steps"]:
        print(f"  {step['step']}. {step['action']}")
# → Deploy Payments Service (v1)
#     1. Run test suite
#     2. Check for pending migrations
#     ...
```

### Unified search: all types at once

```python
results = m.search_all("deployment issues")

print(f"Facts:      {len(results['semantic'])} results")
print(f"Episodes:   {len(results['episodic'])} results")
print(f"Procedures: {len(results['procedural'])} results")
```

This is the point. One query, three structurally different types of knowledge. The agent asking "what should I know about deploying?" gets facts, relevant past events, and the current best-practice procedure in a single call.

### Report failure — procedure evolves automatically

```python
proc_id = procs[0]["id"]

# Two weeks later, another failure
m.procedure_feedback(
    proc_id,
    success=False,
    context="Health check passed but /api/payments returned 500. "
            "STRIPE_WEBHOOK_SECRET env var was missing from the "
            "ECS task definition.",
    failed_at_step=6,
)

# The procedure evolves to v2 with env var checks added
updated = m.procedures(query="deploy payments")
print(f"Version: {updated[0]['version']}")  # → 2
print(f"Steps:   {len(updated[0]['steps'])}")  # → 9 (was 7)
```

The procedure grew two steps: an environment variable check before deploy, and a smoke test on critical endpoints after deploy. No human had to update a wiki page.

---

## Three types, one system

The argument is simple: AI agents that only store facts are missing two-thirds of what makes human memory useful. Episodic memory provides learning from experience. Procedural memory provides repeatable skills. Both are structurally different from fact retrieval and require different storage and search mechanisms.

If you are building agents with LangChain, CrewAI, or any other framework, consider what happens when your agent needs to answer these three questions:

1. "What do we know about the payments service?" — **semantic**
2. "What happened the last time we deployed?" — **episodic**
3. "How do we deploy?" — **procedural**

If your memory layer can only answer the first one, your agent is operating with a fraction of the context it needs.

The psychology is 50 years old. The implementation does not have to be.

---

*[Mengram](https://mengram.io) implements all three memory types with automatic extraction and experience-driven procedure evolution. It is free, has Python and JavaScript SDKs, and integrates with LangChain and CrewAI. [GitHub](https://github.com/alibaizhanov/mengram) | [Docs](https://mengram.io/docs)*

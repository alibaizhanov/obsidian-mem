---
title: "Experience-Driven Procedures: How AI Agents Learn From Failures"
published: false
description: "Static runbooks rot. Experience-driven procedures evolve automatically when things go wrong. Here's how to build AI agents that actually learn from failures."
tags: ai, agents, devops, python
cover_image:
canonical_url:
---

Your DevOps runbook says `deploy: build -> push -> verify`. Last Friday the deploy crashed because someone forgot to run migrations. The runbook still says the same thing.

What if it could learn?

## The problem with static procedures

Every production team has runbooks. Every AI agent framework has workflow definitions. They all share the same flaw: **they're written once and they rot.**

A procedure that worked 50 times will fail on attempt 51 because the environment changed, a new edge case appeared, or someone added a dependency nobody documented. The failure gets discussed in Slack, maybe someone updates the doc, usually nobody does. The next person hits the same wall.

This is worse for AI agents. An agent following a static procedure will repeat the same failure indefinitely. It has no mechanism to incorporate what went wrong into what it does next.

Static procedures don't track outcomes. They don't know which steps fail. They don't evolve.

## What are experience-driven procedures

Experience-driven procedures are workflows that maintain a closed feedback loop with their execution history:

- **Every execution is tracked** as an episode (success or failure, with context)
- **Failures trigger automatic evolution** -- the system analyzes what went wrong and creates a new procedure version
- **Success patterns create new procedures** -- 3+ similar successful episodes auto-generate a reusable workflow
- **Full version history is preserved** -- every version, every change, every triggering episode

Think of it as git for workflows, where merges happen based on failure analysis instead of pull requests.

The feedback loop looks like this:

```
Procedure v1 executed
       |
   [failure with context]
       |
   Episode created (linked to procedure + failed step)
       |
   LLM analyzes: procedure steps + failure context + episode
       |
   Procedure v2 created (new steps, same lineage)
       |
   v2 executed...
```

## How it works technically

Four mechanisms drive the system:

**Extraction.** When you add a conversation that mentions a multi-step process, the extraction pipeline identifies it and creates a procedure automatically. No manual tagging required.

**Linking.** Episodes (events, interactions) are auto-linked to relevant procedures via embedding similarity. When an episode mentions "deploy" and a procedure covers deployment, they get connected.

**Evolution.** When you report a failure with context, the system: (1) creates a failure episode linked to the procedure and failed step, (2) passes the current procedure steps + failure context to an LLM, (3) the LLM produces improved steps with a structured diff, (4) a new procedure version is created with full lineage tracking.

**Pattern creation.** When 3+ similar positive episodes exist that aren't linked to any procedure, the system clusters them by embedding similarity and asks an LLM to extract a common workflow. New procedure, created from experience alone.

## Full walkthrough

### Setup

```python
from cloud.client import CloudMemory

m = CloudMemory(api_key="om-...")
```

### Step 1: Add a conversation that describes a deployment process

```python
result = m.add([
    {"role": "user", "content": "How do we deploy the backend service?"},
    {"role": "assistant", "content": """Here's our deployment process:
    1. Run the test suite: pytest tests/ --tb=short
    2. Build the Docker image: docker build -t api:latest .
    3. Push to ECR: docker push $ECR_REPO/api:latest
    4. Update the ECS service: aws ecs update-service --cluster prod --service api --force-new-deployment
    5. Verify health check: curl https://api.example.com/health"""},
])

# Processing is async -- wait for extraction
job = m.wait_for_job(result["job_id"])
```

The extraction pipeline identifies this as a procedure and creates it automatically.

### Step 2: Retrieve the extracted procedure

```python
procs = m.procedures(query="deploy backend")

proc = procs[0]
print(f"Name: {proc['name']}")
print(f"Version: {proc['version']}")
print(f"Steps: {len(proc['steps'])}")
print(f"Success: {proc['success_count']}  Fail: {proc['fail_count']}")
for step in proc["steps"]:
    print(f"  {step['step']}. {step['action']}")
```

```
Name: Backend Service Deployment
Version: 1
Steps: 5
Success: 0  Fail: 0
  1. Run test suite
  2. Build Docker image
  3. Push to ECR
  4. Update ECS service
  5. Verify health check
```

### Step 3: Report a failure -- trigger evolution to v2

Friday's deploy crashed. Migrations weren't run before the ECS update.

```python
m.procedure_feedback(
    proc["id"],
    success=False,
    context="Deploy failed: ECS containers started but crashed on boot. "
            "Root cause: new code expected a 'user_preferences' table that "
            "didn't exist. We forgot to run database migrations before "
            "updating the service.",
    failed_at_step=4,
)
```

Behind the scenes, the system:
1. Creates a failure episode linked to this procedure at step 4
2. Sends the procedure + failure context to the LLM
3. LLM analyzes the root cause and produces improved steps
4. New procedure version is created

```python
# Fetch the evolved procedure
import time; time.sleep(3)  # evolution runs in background
procs = m.procedures(query="deploy backend")
proc_v2 = procs[0]

print(f"Version: {proc_v2['version']}")
for step in proc_v2["steps"]:
    print(f"  {step['step']}. {step['action']} -- {step['detail']}")
```

```
Version: 2
  1. Run test suite -- pytest tests/ --tb=short
  2. Check for pending migrations -- python manage.py showmigrations | grep '\[ \]'
  3. Run database migrations -- python manage.py migrate --no-input
  4. Build Docker image -- docker build -t api:latest .
  5. Push to ECR -- docker push $ECR_REPO/api:latest
  6. Update ECS service -- aws ecs update-service --cluster prod --service api
  7. Wait for stabilization -- aws ecs wait services-stable --cluster prod --service api
  8. Verify health check -- curl https://api.example.com/health
```

The procedure grew from 5 to 8 steps. It added migration checks (steps 2-3) and a stabilization wait (step 7). This happened because a real failure provided the context for improvement.

### Step 4: Another failure -- evolve to v3

Two weeks later, another issue: the health check passed but the service was returning 500s on certain endpoints because environment variables weren't updated.

```python
m.procedure_feedback(
    proc_v2["id"],
    success=False,
    context="Health check passed (/ returned 200) but /api/users returned 500. "
            "New feature required STRIPE_WEBHOOK_SECRET env var that wasn't "
            "set in the ECS task definition. Need to verify env vars match "
            "what the code expects before deploying.",
    failed_at_step=8,
)
```

```python
time.sleep(3)
procs = m.procedures(query="deploy backend")
proc_v3 = procs[0]

print(f"Version: {proc_v3['version']}")
print(f"Success: {proc_v3['success_count']}  Fail: {proc_v3['fail_count']}")
for step in proc_v3["steps"]:
    print(f"  {step['step']}. {step['action']}")
```

```
Version: 3
Success: 0  Fail: 2
  1. Run test suite
  2. Check for pending migrations
  3. Run database migrations
  4. Diff environment variables against task definition
  5. Build Docker image
  6. Push to ECR
  7. Update ECS task definition (if env vars changed)
  8. Update ECS service
  9. Wait for stabilization
  10. Verify health check on /health
  11. Smoke test critical endpoints (/api/users, /api/billing)
```

Now the procedure checks env vars before deploy (step 4) and smoke tests critical endpoints instead of just hitting `/health` (step 11). Every failure made the procedure better.

### Step 5: Inspect the evolution history

```python
history = m.procedure_history(proc["id"])

print("=== Versions ===")
for v in history["versions"]:
    print(f"  v{v['version']} | {len(v['steps'])} steps | "
          f"created {v['created_at'][:10]}")

print("\n=== Evolution Log ===")
for entry in history["evolution_log"]:
    print(f"  v{entry['version_before']} -> v{entry['version_after']} "
          f"| {entry['change_type']}")
    if entry.get("diff"):
        for added in entry["diff"].get("added", []):
            print(f"    + {added}")
        for removed in entry["diff"].get("removed", []):
            print(f"    - {removed}")
```

```
=== Versions ===
  v1 | 5 steps | created 2026-02-01
  v2 | 8 steps | created 2026-02-14
  v3 | 11 steps | created 2026-02-28

=== Evolution Log ===
  v1 -> v2 | step_added
    + Check for pending database migrations
    + Run database migrations before deploy
    + Wait for ECS service stabilization
  v2 -> v3 | step_added
    + Diff environment variables against task definition
    + Update ECS task definition if env vars changed
    + Smoke test critical endpoints after deploy
```

Full lineage. Every change traced to a specific failure episode. Auditable.

## Real-world applications

**DevOps deployment pipelines.** Exactly what we walked through. Every outage makes the procedure smarter. New team members get the latest version with all lessons baked in.

**Customer support escalation.** An escalation procedure fails because a specific edge case wasn't handled. The procedure evolves to include a check for that case. Support agents always see the current best practice.

**Onboarding workflows.** New hire follows onboarding procedure, gets stuck because a tool requires admin approval nobody mentioned. Procedure evolves to include the approval step.

**Debugging procedures.** "Service is slow" triage procedure fails to catch a specific type of memory leak. Next version includes heap dump analysis as a step.

**Agent-to-agent handoffs.** Multi-agent systems where one agent's output feeds another. When the downstream agent fails, the upstream procedure evolves to produce better output.

## What this changes

Most AI memory systems store facts. Some store events. Almost none close the loop between **what an agent does** and **what happened when it did it**.

Experience-driven procedures are the difference between an agent that remembers and an agent that learns. The procedure is not a static document -- it is a living artifact that improves every time something goes wrong.

Three API calls: `procedures()` to retrieve, `procedure_feedback()` to report outcomes, `procedure_history()` to inspect evolution. The learning happens automatically.

Your runbook failed on Friday. By Monday, it already knows better.

---

*[Mengram](https://mengram.io) is an AI memory platform with semantic, episodic, and procedural memory. Experience-driven procedures ship in v2.7. [Get started](https://mengram.io) or check the [docs](https://mengram.io/docs).*

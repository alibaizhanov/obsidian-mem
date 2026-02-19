#!/usr/bin/env python3
"""
DevOps Agent — Experience-Driven Deployment Procedures

Demonstrates how Mengram learns deployment procedures from conversations
and evolves them automatically when failures are reported.

No LLM API key needed — Mengram handles all extraction server-side.
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from cloud.client import CloudMemory

# Terminal colors
CYAN, GREEN, YELLOW, RED = "\033[96m", "\033[92m", "\033[93m", "\033[91m"
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def header(text):
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}{RESET}\n")


def step(n, text):
    print(f"  {BOLD}{GREEN}[Step {n}]{RESET} {text}\n")


def main():
    api_key = os.environ.get("MENGRAM_API_KEY")
    if not api_key:
        print(f"{RED}Error: MENGRAM_API_KEY not set.{RESET}")
        print(f"  Get your key at https://mengram.io/dashboard")
        sys.exit(1)

    mem = CloudMemory(api_key=api_key)

    header("DevOps Agent — Experience-Driven Procedures")
    print(f"  This demo shows how Mengram learns deployment procedures")
    print(f"  from team conversations and evolves them from failure feedback.\n")

    # --- Step 1: Teach a deployment procedure ---
    step(1, "Teaching Mengram a deployment procedure...")

    conversation = [
        {"role": "user", "content": "How do we deploy the payments service to production?"},
        {"role": "assistant", "content": (
            "Here's our deployment procedure for the payments service:\n"
            "1. Run the test suite: pytest tests/ -x --timeout=300\n"
            "2. Build Docker image: docker build -t payments:latest .\n"
            "3. Push to registry: docker push registry.internal/payments:latest\n"
            "4. Deploy to staging: kubectl apply -f k8s/staging/\n"
            "5. Run smoke tests: ./scripts/smoke-test.sh staging\n"
            "6. Deploy to production: kubectl apply -f k8s/production/\n"
            "7. Monitor dashboards for 15 minutes\n"
            "8. If issues, rollback: kubectl rollout undo deployment/payments -n production"
        )},
    ]

    result = mem.add(conversation)
    job_id = result.get("job_id")
    print(f"  Sent conversation to Mengram (job: {job_id})")

    if job_id:
        print(f"  Waiting for extraction...", end="", flush=True)
        mem.wait_for_job(job_id, max_wait=120)
        time.sleep(3)
        print(f" {GREEN}done{RESET}")

    print(f"  {GREEN}Procedure extracted automatically from the conversation.{RESET}\n")

    # --- Step 2: Find the procedure ---
    step(2, "Searching for deployment procedures...")

    procedures = mem.procedures(query="deploy payments")

    if not procedures:
        print(f"  {DIM}Extraction still processing, retrying...{RESET}")
        time.sleep(5)
        procedures = mem.procedures(query="deploy payments")

    proc_id = None
    if procedures:
        proc = procedures[0]
        proc_id = str(proc.get("id") or proc.get("procedure_id"))
        print(f"  {BOLD}Found:{RESET} {proc.get('name', 'Unnamed procedure')}")
        steps = proc.get("steps", [])
        for i, s in enumerate(steps, 1):
            text = s if isinstance(s, str) else s.get("description", s.get("action", str(s)))
            print(f"    {i}. {text}")
        print(f"\n  {DIM}Reliability: {proc.get('success_count', 0)} successes, "
              f"{proc.get('fail_count', 0)} failures{RESET}\n")
    else:
        print(f"  {YELLOW}No procedures found yet — extraction may still be in progress.{RESET}\n")

    # --- Step 3: Report a failure ---
    step(3, "Reporting a deployment failure...")

    if proc_id:
        print(f"  {RED}Scenario: Production crashed after deploy — database migration")
        print(f"  was missing. The 'payment_metadata' column didn't exist.{RESET}\n")

        mem.procedure_feedback(
            procedure_id=proc_id,
            success=False,
            context=(
                "Deployment failed in production. Smoke tests passed on staging "
                "but the service crashed on production startup. Root cause: database "
                "migration was not applied before deploying the new version. The new "
                "code expected a 'payment_metadata' column that didn't exist. "
                "Fix: Always run pending database migrations before deploying."
            ),
            failed_at_step=6,
            user_id=user_id,
        )
        print(f"  {GREEN}Failure reported — Mengram is evolving the procedure...{RESET}")
        time.sleep(3)
    else:
        print(f"  {YELLOW}Skipping (no procedure ID available).{RESET}\n")
        return

    # --- Step 4: View the evolved procedure ---
    step(4, "Viewing the evolved procedure...")

    updated = mem.procedures(query="deploy payments")

    if updated:
        proc = updated[0]
        print(f"  {BOLD}Updated:{RESET} {proc.get('name', 'Unnamed procedure')}")
        steps = proc.get("steps", [])
        for i, s in enumerate(steps, 1):
            text = s if isinstance(s, str) else s.get("description", s.get("action", str(s)))
            print(f"    {i}. {text}")
        print(f"\n  {DIM}Reliability: {proc.get('success_count', 0)} successes, "
              f"{proc.get('fail_count', 0)} failures{RESET}\n")

    # --- Step 5: Version history ---
    step(5, "Viewing procedure version history...")

    history = mem.procedure_history(proc_id)

    if history and history.get("versions"):
        for v in history["versions"]:
            ver = v.get("version", "?")
            date = (v.get("created_at") or "")[:19]
            reason = v.get("evolution_reason", v.get("reason", "initial"))
            print(f"    v{ver}  {DIM}{date}{RESET}  {reason}")

    if history and history.get("evolution_log"):
        print(f"\n  {BOLD}Evolution log:{RESET}")
        for entry in history["evolution_log"]:
            print(f"    - {entry.get('summary', entry.get('reason', str(entry)))}")
    print()

    # --- Step 6: Unified search ---
    step(6, "Unified search across all memory types...")

    results = mem.search_all("deployment best practices")

    for mem_type in ["semantic", "episodic", "procedural"]:
        items = results.get(mem_type, [])
        print(f"  {BOLD}{mem_type.title()}:{RESET} {len(items)} results")
        for item in items[:2]:
            if mem_type == "semantic":
                facts = item.get("facts", item.get("knowledge", []))
                print(f"    {item.get('entity', '?')}: {facts[0] if facts else 'N/A'}")
            elif mem_type == "episodic":
                print(f"    {(item.get('summary', 'N/A'))[:80]}")
            elif mem_type == "procedural":
                print(f"    {item.get('name', '?')} "
                      f"(success: {item.get('success_count', 0)}, "
                      f"fail: {item.get('fail_count', 0)})")

    header("Demo Complete")
    print(f"  {BOLD}What you saw:{RESET}")
    print(f"  1. Mengram extracted a procedure from a conversation")
    print(f"  2. A failure report triggered automatic procedure evolution")
    print(f"  3. Version history tracks how procedures improve over time")
    print(f"  4. Unified search finds knowledge across all memory types\n")
    print(f"  {DIM}Learn more at https://mengram.io/docs{RESET}\n")


if __name__ == "__main__":
    main()

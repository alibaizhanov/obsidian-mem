"""
Mengram CrewAI Integration — persistent memory tools for CrewAI agents.

Give your CrewAI agents human-like memory that persists across sessions.
Agents automatically learn workflows as procedures and improve over time.

Usage:

    from integrations.crewai import create_mengram_tools

    tools = create_mengram_tools(api_key="om-...")

    agent = Agent(
        role="Support Engineer",
        goal="Help users with technical issues",
        tools=tools,
    )

    crew = Crew(agents=[agent], tasks=[...])

Killer Feature — Procedural Learning:

    When an agent completes a multi-step workflow, Mengram saves it as a procedure.
    Next time a similar task comes up, the agent already knows the optimal path
    with success/failure tracking. No other memory system does this.

Tools provided:
    - mengram_search: Search all 3 memory types (semantic, episodic, procedural)
    - mengram_remember: Save information to memory (auto-extracts all 3 types)
    - mengram_profile: Get full user context (Cognitive Profile)
    - mengram_save_workflow: Save a completed workflow as a procedure
    - mengram_workflow_feedback: Report success/failure of a workflow
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger("mengram.crewai")


def _get_client(api_key: str, base_url: str = None):
    try:
        from cloud.client import CloudMemory
    except ImportError:
        raise ImportError("Mengram SDK required. Install: pip install mengram-ai")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return CloudMemory(**kwargs)


def _get_base_tool():
    try:
        from crewai.tools import BaseTool
        return BaseTool
    except ImportError:
        raise ImportError("CrewAI required. Install: pip install crewai")


def create_mengram_tools(
    api_key: str,
    user_id: str = "default",
    base_url: str = None,
) -> list:
    """Create Mengram memory tools for CrewAI agents.

    Args:
        api_key: Mengram API key
        user_id: User identifier for memory scope
        base_url: Custom API URL (default: https://mengram.io)

    Returns:
        List of CrewAI tools: [search, remember, profile, save_workflow, workflow_feedback]
    """
    client = _get_client(api_key, base_url)
    BaseTool = _get_base_tool()

    # ---- Tool 1: Search Memory ----
    class MengramSearch(BaseTool):
        name: str = "mengram_search"
        description: str = (
            "Search user's memory for relevant information. Returns facts (semantic), "
            "past events (episodic), and known workflows (procedural). "
            "Use this before answering any question to check what you already know about the user. "
            "Input: a search query string."
        )

        def _run(self, query: str) -> str:
            try:
                results = client.search_all(query, user_id=user_id)
            except Exception as e:
                return f"Memory search failed: {e}"

            parts = []

            # Semantic
            semantic = results.get("semantic", [])
            if semantic:
                facts = []
                for r in semantic[:5]:
                    entity = r.get("entity", "")
                    for f in r.get("facts", [])[:5]:
                        facts.append(f"{entity}: {f}")
                if facts:
                    parts.append("KNOWN FACTS:\n" + "\n".join(f"- {f}" for f in facts))

            # Episodic
            episodic = results.get("episodic", [])
            if episodic:
                events = []
                for ep in episodic[:3]:
                    line = ep.get("summary", "")
                    if ep.get("outcome"):
                        line += f" → Outcome: {ep['outcome']}"
                    if ep.get("when"):
                        line += f" ({ep['when']})"
                    events.append(line)
                if events:
                    parts.append("PAST EVENTS:\n" + "\n".join(f"- {e}" for e in events))

            # Procedural
            procedural = results.get("procedural", [])
            if procedural:
                procs = []
                for pr in procedural[:3]:
                    name = pr.get("name", "")
                    steps = pr.get("steps", [])
                    steps_str = " → ".join(s.get("action", "") for s in steps[:10])
                    success = pr.get("success_count", 0)
                    fail = pr.get("fail_count", 0)
                    proc_id = pr.get("id", "")
                    procs.append(
                        f"{name} [id:{proc_id}]: {steps_str} "
                        f"(success: {success}, fail: {fail})"
                    )
                if procs:
                    parts.append(
                        "KNOWN WORKFLOWS:\n" + "\n".join(f"- {p}" for p in procs)
                    )

            if not parts:
                return "No relevant memories found."

            return "\n\n".join(parts)

    # ---- Tool 2: Remember ----
    class MengramRemember(BaseTool):
        name: str = "mengram_remember"
        description: str = (
            "Save important information to memory. Mengram automatically extracts "
            "facts (semantic), events (episodic), and workflows (procedural) from "
            "the conversation. Use this after learning something new about the user "
            "or after completing a task. Input: text describing what happened."
        )

        def _run(self, text: str) -> str:
            try:
                messages = [{"role": "user", "content": text}]
                result = client.add(messages, user_id=user_id)
                return f"Saved to memory. Job ID: {result.get('job_id', 'ok')}"
            except Exception as e:
                return f"Memory save failed: {e}"

    # ---- Tool 3: Cognitive Profile ----
    class MengramProfile(BaseTool):
        name: str = "mengram_profile"
        description: str = (
            "Get full context about the user — who they are, what they know, "
            "recent events, known workflows. Returns a Cognitive Profile that "
            "describes the user comprehensively. Use at the start of a task "
            "to understand who you're working with. No input required."
        )

        def _run(self, **kwargs) -> str:
            try:
                profile = client.get_profile(user_id, force=False)
                return profile.get("system_prompt", "No profile available yet.")
            except Exception as e:
                return f"Profile fetch failed: {e}"

    # ---- Tool 4: Save Workflow (Killer Feature) ----
    class MengramSaveWorkflow(BaseTool):
        name: str = "mengram_save_workflow"
        description: str = (
            "Save a completed multi-step workflow as a reusable procedure. "
            "Next time a similar task comes up, this workflow will appear in "
            "memory search results with success/failure tracking. "
            "Input: a description of what you just did, step by step. "
            "Example: 'Resolved billing issue: 1) Checked subscription status "
            "2) Found expired card 3) Sent renewal link 4) Confirmed payment'"
        )

        def _run(self, workflow_description: str) -> str:
            try:
                messages = [
                    {
                        "role": "assistant",
                        "content": f"I completed the following workflow: {workflow_description}",
                    }
                ]
                result = client.add(messages, user_id=user_id)
                return (
                    f"Workflow saved as procedure. "
                    f"Job ID: {result.get('job_id', 'ok')}. "
                    f"It will be available in future memory searches."
                )
            except Exception as e:
                return f"Workflow save failed: {e}"

    # ---- Tool 5: Workflow Feedback ----
    class MengramWorkflowFeedback(BaseTool):
        name: str = "mengram_workflow_feedback"
        description: str = (
            "Report whether a workflow from memory succeeded or failed. "
            "This helps Mengram learn which workflows are reliable. "
            "Input: procedure_id (from search results) and 'success' or 'failure'."
        )

        def _run(self, procedure_id: str, outcome: str = "success") -> str:
            success = outcome.lower().strip() in ("success", "true", "yes", "1")
            try:
                client.procedure_feedback(procedure_id, success=success)
                status = "successful" if success else "failed"
                return f"Recorded workflow as {status}. Future searches will reflect this."
            except Exception as e:
                return f"Feedback save failed: {e}"

    return [
        MengramSearch(),
        MengramRemember(),
        MengramProfile(),
        MengramSaveWorkflow(),
        MengramWorkflowFeedback(),
    ]

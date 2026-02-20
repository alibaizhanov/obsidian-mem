"""
Mengram MCP Server v3.0 — Proactive Context via Resources

Resources (auto-included in Claude's context when pinned):
  memory://profile        — Full user knowledge profile. PIN THIS!
  memory://recent         — Recent knowledge entries
  memory://entity/{name}  — Specific entity details

Tools:
  remember / remember_text — Save knowledge to vault
  recall / search          — Query memory
  recall_all / vault_stats — Overview
"""

import sys
import json
import asyncio
from pathlib import Path
from urllib.parse import unquote

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool, TextContent,
        Resource, ResourceTemplate,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from engine.brain import MengramBrain, create_brain, load_config


def _build_compact_profile(brain: MengramBrain) -> str:
    """
    Build compact vault summary for MCP instructions.
    NOT a full profile — just an "index" so Claude knows what exists.
    Scales to 1000+ notes.
    """
    try:
        vault = Path(brain.vault_path)
        files = list(vault.glob("*.md"))
        if not files:
            return "Memory vault is empty. Start conversations and use 'remember' to build knowledge."

        # Collect stats
        entities_by_type = {}
        all_knowledge_types = set()
        recent_entities = []

        # Sort by modification time
        files_sorted = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

        for f in files_sorted:
            data = brain._get_entity_data(f.stem)
            etype = data.get("type", "unknown")
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(data)
            for k in data.get("knowledge", []):
                all_knowledge_types.add(k["type"])

        # Top-10 most recent entities (one line each)
        for f in files_sorted[:10]:
            data = brain._get_entity_data(f.stem)
            facts_str = "; ".join(data.get("facts", [])[:2])
            k_count = len(data.get("knowledge", []))
            k_hint = f" ({k_count} knowledge)" if k_count else ""
            recent_entities.append(f"  - {f.stem} ({data.get('type', '?')}): {facts_str}{k_hint}")

        # Build summary
        lines = []

        # Stats
        total = len(files)
        type_counts = {t: len(es) for t, es in entities_by_type.items()}
        type_str = ", ".join(f"{count} {t}" for t, count in sorted(type_counts.items(), key=lambda x: -x[1]))
        lines.append(f"Vault: {total} entities ({type_str})")

        if all_knowledge_types:
            lines.append(f"Knowledge types: {', '.join(sorted(all_knowledge_types))}")

        # Recent/top entities
        lines.append("\nKey entities:")
        lines.extend(recent_entities)

        # All entity names grouped by type (just names, no details)
        for etype in ["person", "company", "project", "technology", "concept"]:
            entities = entities_by_type.get(etype, [])
            if not entities:
                continue
            names = [e["entity"] for e in entities]
            plural = {"person": "people", "company": "companies", "project": "projects",
                      "technology": "technologies", "concept": "concepts"}.get(etype, etype + "s")
            if len(names) <= 15:
                lines.append(f"\nAll {plural}: {', '.join(names)}")
            else:
                lines.append(f"\nAll {plural} ({len(names)}): {', '.join(names[:15])}...")

        return "\n".join(lines)

    except Exception as e:
        return f"Error loading profile: {e}"


def create_mcp_server(brain: MengramBrain) -> "Server":
    # Build dynamic instructions with compact profile
    compact_profile = _build_compact_profile(brain)
    instructions = (
        "YOU HAVE A PERSISTENT MEMORY SYSTEM (Mengram) with a knowledge graph.\n"
        "CRITICAL RULES:\n"
        "1. When the user asks about their work, projects, tech stack, past problems, "
        "or anything personal — ALWAYS use the 'recall' tool FIRST. Do NOT use built-in "
        "chat history search. Your recall tool has structured knowledge with code, configs, "
        "and solutions that chat history does not have.\n"
        "2. After meaningful conversations, call 'remember' to save new knowledge.\n"
        "3. The summary below is just an INDEX — use recall() to get full details.\n\n"
        f"{compact_profile}"
    )

    server = Server("mengram", instructions=instructions)

    # ==========================================
    # RESOURCES — Proactive Context
    # ==========================================

    @server.list_resources()
    async def list_resources():
        return [
            Resource(
                uri="memory://profile",
                name="User Knowledge Profile",
                description=(
                    "Complete user profile from memory vault — all entities, facts, "
                    "relations, and knowledge with code artifacts. PIN THIS for proactive context."
                ),
                mimeType="text/markdown",
            ),
            Resource(
                uri="memory://recent",
                name="Recent Knowledge",
                description="Most recent knowledge entries — solutions, insights, commands.",
                mimeType="text/markdown",
            ),
        ]

    @server.list_resource_templates()
    async def list_resource_templates():
        return [
            ResourceTemplate(
                uriTemplate="memory://entity/{name}",
                name="Entity Details",
                description="Full details for a specific entity (person, project, technology).",
                mimeType="text/markdown",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri):
        uri_str = str(uri)

        if uri_str == "memory://profile":
            return brain.get_profile()

        elif uri_str == "memory://recent":
            return brain.get_recent_knowledge(limit=10)

        elif uri_str.startswith("memory://entity/"):
            entity_name = unquote(uri_str.replace("memory://entity/", ""))
            data = brain._get_entity_data(entity_name)
            if not data["facts"] and not data["knowledge"]:
                return f"Entity '{entity_name}' not found in vault."

            lines = [f"# {entity_name} ({data['type']})\n"]
            if data["facts"]:
                lines.append("## Facts")
                for f in data["facts"]:
                    lines.append(f"- {f}")
            if data["relations"]:
                lines.append("\n## Relations")
                for r in data["relations"]:
                    arrow = "→" if r["direction"] == "outgoing" else "←"
                    lines.append(f"- {arrow} {r['type']}: {r['target']}")
            if data["knowledge"]:
                lines.append("\n## Knowledge")
                for k in data["knowledge"]:
                    lines.append(f"\n**[{k['type']}] {k['title']}**")
                    lines.append(k["content"])
                    if k.get("artifact"):
                        lines.append(f"```\n{k['artifact']}\n```")
            return "\n".join(lines)

        return f"Unknown resource: {uri_str}"

    # ==========================================
    # TOOLS
    # ==========================================

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="remember",
                description=(
                    "Save knowledge from conversation to memory vault. Call AFTER every "
                    "meaningful conversation. Extracts entities, facts, relations, and "
                    "rich knowledge (solutions, formulas, configs with code)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "conversation": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": ["user", "assistant"]},
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                            "description": "Conversation messages array",
                        },
                    },
                    "required": ["conversation"],
                },
            ),
            Tool(
                name="remember_text",
                description="Remember knowledge from text. E.g.: 'Remember that I work at Google on Kubernetes'.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to remember"},
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="recall",
                description=(
                    "Semantic search through memory. Finds by MEANING, not just keywords. "
                    "Returns context with facts, relations, and knowledge artifacts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to recall"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="search",
                description="Structured semantic search — returns JSON with scores, facts, knowledge, artifacts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Results count (1-10)", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="recall_all",
                description="Recall EVERYTHING from memory vault.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="vault_stats",
                description="Vault statistics.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            if name == "remember":
                result = brain.remember(arguments["conversation"])
                extraction = result.get("extraction")
                k_count = len(extraction.knowledge) if extraction and hasattr(extraction, "knowledge") else 0
                text = (
                    f"✅ Remembered!\n"
                    f"Created: {', '.join(result['entities_created']) or 'none'}\n"
                    f"Updated: {', '.join(result['entities_updated']) or 'none'}"
                )
                if k_count:
                    text += f"\nKnowledge entries: {k_count}"
                # Notify resource update
                try:
                    await server.request_context.session.send_resource_updated(uri="memory://profile")
                    await server.request_context.session.send_resource_updated(uri="memory://recent")
                except Exception:
                    pass
                return [TextContent(type="text", text=text)]

            elif name == "remember_text":
                result = brain.remember_text(arguments["text"])
                text = (
                    f"✅ Remembered!\n"
                    f"Created: {', '.join(result['entities_created']) or 'none'}\n"
                    f"Updated: {', '.join(result['entities_updated']) or 'none'}"
                )
                try:
                    await server.request_context.session.send_resource_updated(uri="memory://profile")
                    await server.request_context.session.send_resource_updated(uri="memory://recent")
                except Exception:
                    pass
                return [TextContent(type="text", text=text)]

            elif name == "recall":
                context = brain.recall(arguments["query"])
                return [TextContent(type="text", text=context)]

            elif name == "search":
                top_k = arguments.get("top_k", 5)
                results = brain.search(arguments["query"], top_k=top_k)
                return [TextContent(
                    type="text",
                    text=json.dumps(results, ensure_ascii=False, indent=2, default=str),
                )]

            elif name == "recall_all":
                context = brain.recall_all()
                return [TextContent(type="text", text=context)]

            elif name == "vault_stats":
                stats = brain.get_stats()
                return [TextContent(
                    type="text",
                    text=json.dumps(stats, ensure_ascii=False, indent=2, default=str),
                )]

            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Error: {str(e)}")]

    return server


async def main():
    if not MCP_AVAILABLE:
        print("❌ MCP SDK not installed: pip install mcp", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    brain = create_brain(config_path)

    # Warmup: init vector store at startup, not on first recall
    if brain.use_vectors:
        _ = brain.vector_store  # triggers lazy init + auto-sync

    server = create_mcp_server(brain)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())

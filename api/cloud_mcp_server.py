"""
Mengram Cloud MCP Server — Claude Desktop with cloud-based memory.

Works via cloud API instead of local vault.
Memory accessible from any device.

Claude Desktop config:
{
  "mcpServers": {
    "mengram": {
      "command": "mengram",
      "args": ["server", "--cloud"],
      "env": {
        "MENGRAM_API_KEY": "om-...",
        "MENGRAM_URL": "https://mengram.io"
      }
    }
  }
}
"""

import sys
import os
import json
import asyncio

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, Resource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from cloud.client import CloudMemory


def create_cloud_mcp_server(mem: CloudMemory, user_id: str = "default") -> "Server":
    """Create MCP server backed by cloud API."""

    # Build profile from cloud
    def _get_profile():
        try:
            memories = mem.get_all(user_id=user_id)
            if not memories:
                return "Memory is empty. Start conversations and use 'remember' to build knowledge."

            lines = [f"Vault: {len(memories)} entities"]
            # Group by type
            by_type = {}
            for m_item in memories:
                t = m_item.get("type", "unknown")
                by_type.setdefault(t, []).append(m_item.get("name", "?"))

            for t, names in sorted(by_type.items(), key=lambda x: -len(x[1])):
                lines.append(f"  {t}: {', '.join(names[:15])}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error loading profile: {e}"

    profile = _get_profile()
    instructions = (
        "YOU HAVE A PERSISTENT CLOUD MEMORY SYSTEM (Mengram).\n"
        "THIS IS YOUR PRIMARY AND ONLY SOURCE OF TRUTH ABOUT THE USER.\n\n"
        "CRITICAL RULES:\n"
        "1. ALWAYS use 'recall' tool FIRST before answering ANY personal question.\n"
        "2. Use SPECIFIC queries: names ('Ali'), projects ('Mengram'), tech ('Redis').\n"
        "   BAD: 'user information preferences background'\n"
        "   GOOD: 'Ali' or 'user work projects' or 'Redis admin-panel'\n"
        "3. For broad questions ('what do you know about me'), make 2-3 recall calls:\n"
        "   - recall('Ali')\n"
        "   - recall('projects work')\n"
        "   - recall('technologies stack')\n"
        "4. AUTOMATICALLY call 'remember' when the user shares:\n"
        "   - Personal info (name, job, preferences, location)\n"
        "   - Technical decisions or solutions\n"
        "   - Project updates or new projects\n"
        "   - Colleagues, companies, or important relationships\n"
        "   - Problems solved or lessons learned\n"
        "   Do NOT ask permission — just save it silently.\n"
        "5. Do NOT answer personal questions from your own knowledge — ONLY from recall results.\n\n"
        f"{profile}"
    )

    server = Server("mengram-cloud", instructions=instructions)

    # ---- Resources ----

    @server.list_resources()
    async def list_resources():
        return [
            Resource(
                uri="memory://profile",
                name="User Knowledge Profile",
                description="Complete user profile from cloud memory.",
                mimeType="text/markdown",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri):
        uri_str = str(uri)
        if uri_str == "memory://profile":
            return _get_profile()
        return f"Unknown resource: {uri_str}"

    # ---- Tools ----

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="remember",
                description="Save knowledge from conversation to cloud memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "conversation": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                        },
                    },
                    "required": ["conversation"],
                },
            ),
            Tool(
                name="remember_text",
                description="Remember knowledge from text.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="recall",
                description="ALWAYS call this FIRST when user asks anything personal. Semantic search through cloud memory. Use specific keywords: person names, project names, technologies. For broad questions like 'what do you know about me', search for the user's name or 'Ali'. Multiple calls with different queries are encouraged.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Specific search query — use names, projects, technologies. NOT generic phrases."},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="search",
                description="Structured search — returns JSON with scores, facts, knowledge.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="timeline",
                description="Search memory by time. Use when user asks 'what did I do last week', 'when did I...', 'what happened in January'. Returns facts with timestamps.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "after": {"type": "string", "description": "ISO datetime — start of range (e.g. 2025-02-01T00:00:00Z)"},
                        "before": {"type": "string", "description": "ISO datetime — end of range"},
                    },
                },
            ),
            Tool(
                name="vault_stats",
                description="Memory statistics.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="run_agents",
                description="Run memory agents that analyze, clean, and find patterns in memory. Use 'curator' to find contradictions and stale facts, 'connector' to find hidden patterns and insights, 'digest' for weekly summary, or 'all' for everything.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string", "enum": ["curator", "connector", "digest", "all"], "default": "all"},
                        "auto_fix": {"type": "boolean", "default": True, "description": "Auto-archive low quality facts (curator)"},
                    },
                },
            ),
            Tool(
                name="get_insights",
                description="Get AI-generated insights about the user — patterns, connections, reflections from memory analysis. Call this when user asks 'what patterns do you see', 'what do you know about how I think', 'analyze my memory'.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="list_procedures",
                description="List learned workflows/procedures from memory. Use when user asks 'how do I usually...', 'what's my process for...', 'show my workflows'. Returns procedures with steps, success/fail counts, and version info.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Optional search query to find specific procedures"},
                        "limit": {"type": "integer", "default": 10},
                    },
                },
            ),
            Tool(
                name="procedure_feedback",
                description="Record success or failure for a procedure. ALWAYS use this when the user reports that a workflow worked or failed. On failure with context, the system automatically evolves the procedure to a new improved version.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "procedure_id": {"type": "string", "description": "UUID of the procedure"},
                        "success": {"type": "boolean", "description": "true if it worked, false if it failed"},
                        "context": {"type": "string", "description": "What went wrong (required when success=false to trigger evolution)"},
                        "failed_at_step": {"type": "integer", "description": "Which step number failed (optional)"},
                    },
                    "required": ["procedure_id", "success"],
                },
            ),
            Tool(
                name="procedure_history",
                description="Show how a procedure evolved over time — all versions and what changed. Use when user asks 'how has my deploy process changed', 'show procedure evolution'.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "procedure_id": {"type": "string", "description": "UUID of any version of the procedure"},
                    },
                    "required": ["procedure_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            if name == "remember":
                result = mem.add(arguments["conversation"], user_id=user_id)
                if result.get("status") == "accepted":
                    text = "✅ Accepted! Processing in background — memories will appear shortly."
                else:
                    text = (
                        f"✅ Remembered!\n"
                        f"Created: {', '.join(result.get('created', [])) or 'none'}\n"
                        f"Updated: {', '.join(result.get('updated', [])) or 'none'}\n"
                        f"Knowledge: {result.get('knowledge_count', 0)}"
                    )
                return [TextContent(type="text", text=text)]

            elif name == "remember_text":
                result = mem.add([
                    {"role": "user", "content": arguments["text"]},
                ], user_id=user_id)
                if result.get("status") == "accepted":
                    text = "✅ Accepted! Processing in background."
                else:
                    text = (
                        f"✅ Remembered!\n"
                        f"Created: {', '.join(result.get('created', [])) or 'none'}\n"
                        f"Updated: {', '.join(result.get('updated', [])) or 'none'}"
                    )
                return [TextContent(type="text", text=text)]

            elif name == "recall":
                results = mem.search(arguments["query"], user_id=user_id)
                if not results:
                    return [TextContent(type="text", text="Nothing found in memory.")]

                lines = []
                for r in results:
                    lines.append(f"## {r['entity']} ({r.get('type', '?')}) — score: {r.get('score', 0)}")
                    for fact in r.get("facts", []):
                        lines.append(f"- {fact}")
                    for k in r.get("knowledge", []):
                        lines.append(f"\n**[{k.get('type', '')}] {k.get('title', '')}**")
                        lines.append(k.get("content", ""))
                        if k.get("artifact"):
                            lines.append(f"```\n{k['artifact']}\n```")
                    lines.append("")
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "search":
                top_k = arguments.get("top_k", 5)
                results = mem.search(arguments["query"], user_id=user_id, limit=top_k)
                return [TextContent(
                    type="text",
                    text=json.dumps(results, ensure_ascii=False, indent=2),
                )]

            elif name == "timeline":
                results = mem.timeline(
                    after=arguments.get("after"),
                    before=arguments.get("before"),
                    user_id=user_id,
                )
                if not results:
                    return [TextContent(type="text", text="No facts found in that time range.")]
                lines = []
                for entity in results:
                    lines.append(f"## {entity['entity']} ({entity['type']})")
                    for f in entity["facts"]:
                        ts = f.get("created_at", "")[:10] if f.get("created_at") else ""
                        lines.append(f"  [{ts}] {f['content']}")
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "vault_stats":
                stats = mem.stats(user_id=user_id)
                return [TextContent(
                    type="text",
                    text=json.dumps(stats, ensure_ascii=False, indent=2),
                )]

            elif name == "run_agents":
                agent = arguments.get("agent", "all")
                auto_fix = arguments.get("auto_fix", True)
                result = mem.run_agents(agent=agent, auto_fix=auto_fix, user_id=user_id)
                
                lines = [f"🤖 Agent run complete ({agent})"]
                
                if agent == "all" and "agents" in result:
                    r = result["agents"]
                    # Curator
                    c = r.get("curator", {})
                    if c and c.get("health_score"):
                        lines.append(f"\n🧹 **Curator** — Health: {int(c['health_score']*100)}%")
                        if c.get("summary"): lines.append(c["summary"])
                        meta = c.get("_meta", {})
                        if meta.get("actions_taken"): lines.append(f"✅ Auto-fixed: {meta['actions_taken']} facts archived")
                    # Connector
                    cn = r.get("connector", {})
                    if cn.get("patterns"):
                        lines.append(f"\n🔗 **Connector** — {len(cn.get('connections',[]))} connections, {len(cn['patterns'])} patterns")
                        for p in cn["patterns"][:3]:
                            lines.append(f"- {p.get('pattern', '')}")
                    if cn.get("suggestions"):
                        lines.append("\n💡 **Suggestions:**")
                        for s in cn["suggestions"][:3]:
                            lines.append(f"- [{s.get('priority','?')}] {s.get('action','')}")
                    # Digest
                    d = r.get("digest", {})
                    if d.get("headline"):
                        lines.append(f"\n📰 **Digest:** {d['headline']}")
                        if d.get("recommendation"):
                            lines.append(f"💡 {d['recommendation']}")
                else:
                    lines.append(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
                
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "get_insights":
                insights = mem.insights(user_id=user_id)
                
                if not insights.get("has_insights"):
                    return [TextContent(type="text", text="No insights yet. Run agents first or add more memories.")]
                
                lines = ["🧠 **AI Insights from Memory**\n"]
                for group in insights.get("groups", []):
                    lines.append(f"### {group.get('title', '')}")
                    for item in group.get("items", []):
                        conf = int(item.get("confidence", 0) * 100)
                        lines.append(f"- **{item.get('title', '')}** ({conf}% confidence)")
                        lines.append(f"  {item.get('content', '')[:200]}")
                    lines.append("")
                
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "list_procedures":
                query = arguments.get("query")
                limit = arguments.get("limit", 10)
                procs = mem.procedures(query=query, limit=limit, user_id=user_id)
                if not procs:
                    return [TextContent(type="text", text="No learned procedures yet.")]

                lines = [f"📋 **{len(procs)} Procedure(s)**\n"]
                for p in procs:
                    v = p.get("version", 1)
                    sc = p.get("success_count", 0)
                    fc = p.get("fail_count", 0)
                    lines.append(f"### {p['name']} (v{v}) — ✅{sc} ❌{fc}")
                    lines.append(f"ID: `{p['id']}`")
                    if p.get("trigger_condition"):
                        lines.append(f"When: {p['trigger_condition']}")
                    for s in p.get("steps", []):
                        lines.append(f"  {s.get('step', '?')}. {s.get('action', '')} — {s.get('detail', '')}")
                    lines.append("")
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "procedure_feedback":
                proc_id = arguments["procedure_id"]
                success = arguments["success"]
                context = arguments.get("context")
                failed_at_step = arguments.get("failed_at_step")

                result = mem.procedure_feedback(
                    proc_id, success=success,
                    context=context, failed_at_step=failed_at_step,
                    user_id=user_id)

                if success:
                    text = f"✅ Recorded success for '{result.get('name', '?')}' (total: {result.get('success_count', 0)} successes)"
                else:
                    evo = "🔄 Evolution triggered — procedure will improve automatically!" if result.get("evolution_triggered") else ""
                    text = f"❌ Recorded failure for '{result.get('name', '?')}' (total: {result.get('fail_count', 0)} failures)\n{evo}"
                return [TextContent(type="text", text=text)]

            elif name == "procedure_history":
                proc_id = arguments["procedure_id"]
                history = mem.procedure_history(proc_id, user_id=user_id)

                versions = history.get("versions", [])
                evolution = history.get("evolution_log", [])

                if not versions:
                    return [TextContent(type="text", text="Procedure not found.")]

                lines = [f"📜 **{versions[0]['name']}** — {len(versions)} version(s)\n"]
                for v in versions:
                    current = " ← current" if v.get("is_current") else ""
                    lines.append(f"**v{v.get('version', 1)}**{current} — ✅{v.get('success_count', 0)} ❌{v.get('fail_count', 0)}")
                    for s in v.get("steps", []):
                        lines.append(f"  {s.get('step', '?')}. {s.get('action', '')}")
                    lines.append("")

                if evolution:
                    lines.append("**Evolution log:**")
                    for e in evolution:
                        lines.append(f"- v{e.get('version_before', '?')}→v{e.get('version_after', '?')}: "
                                    f"{e.get('change_type', '?')} ({e.get('created_at', '')[:10]})")
                        diff = e.get("diff", {})
                        for key in ["added", "removed", "modified"]:
                            items = diff.get(key, [])
                            if items:
                                for item in items:
                                    lines.append(f"  {key}: {item}")

                return [TextContent(type="text", text="\n".join(lines))]

            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Error: {str(e)}")]

    return server


async def main():
    if not MCP_AVAILABLE:
        print("❌ MCP SDK not installed: pip install mcp", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("MENGRAM_API_KEY", "")
    base_url = os.environ.get("MENGRAM_URL", "https://mengram.io")
    user_id = os.environ.get("MENGRAM_USER_ID", "default")

    if not api_key:
        print("❌ Set MENGRAM_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    mem = CloudMemory(api_key=api_key, base_url=base_url)
    server = create_cloud_mcp_server(mem, user_id=user_id)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())

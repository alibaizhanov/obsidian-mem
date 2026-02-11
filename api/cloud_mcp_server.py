"""
Mengram Cloud MCP Server — Claude Desktop с облачной памятью.

Работает через cloud API вместо локального vault.
Память доступна с любого устройства.

Claude Desktop config:
{
  "mcpServers": {
    "mengram": {
      "command": "mengram",
      "args": ["mcp", "--cloud"],
      "env": {
        "MENGRAM_API_KEY": "om-abc123...",
        "MENGRAM_URL": "https://mengram-production.up.railway.app"
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
        "CRITICAL RULES:\n"
        "1. When the user asks about their work, projects, tech stack, past problems — "
        "ALWAYS use 'recall' tool FIRST.\n"
        "2. After meaningful conversations, call 'remember' to save new knowledge.\n"
        "3. Memory is synced across all devices via cloud.\n\n"
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
                description="Semantic search through cloud memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
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
                name="recall_all",
                description="Get all memories.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="vault_stats",
                description="Memory statistics.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            if name == "remember":
                result = mem.add(arguments["conversation"], user_id=user_id)
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

            elif name == "recall_all":
                memories = mem.get_all(user_id=user_id)
                if not memories:
                    return [TextContent(type="text", text="Memory is empty.")]

                lines = []
                for m_item in memories:
                    entity = mem.get(m_item["name"], user_id=user_id)
                    if entity:
                        lines.append(f"## {entity['entity']} ({entity.get('type', '?')})")
                        for f in entity.get("facts", []):
                            lines.append(f"- {f}")
                        for k in entity.get("knowledge", []):
                            lines.append(f"  [{k.get('type', '')}] {k.get('title', '')}")
                        lines.append("")
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "vault_stats":
                stats = mem.stats(user_id=user_id)
                return [TextContent(
                    type="text",
                    text=json.dumps(stats, ensure_ascii=False, indent=2),
                )]

            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            return [TextContent(type="text", text=f"❌ Error: {str(e)}")]

    return server


async def main():
    if not MCP_AVAILABLE:
        print("❌ MCP SDK not installed: pip install mcp", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("MENGRAM_API_KEY", "")
    base_url = os.environ.get("MENGRAM_URL", "https://mengram-production.up.railway.app")
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

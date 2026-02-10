"""
ObsidianMem MCP Server v2.1 — with Semantic Search

Tools:
1. remember — извлечь знания из разговора → vault
2. remember_text — запомнить текст
3. recall — семантический поиск по памяти (vector + graph)
4. recall_all — полный обзор vault
5. search — структурированный поиск (для приложений)
6. vault_stats — статистика
"""

import sys
import json
import asyncio
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    try:
        from mcp.server import Server
        from mcp.server.stdio import run_server as stdio_server
        from mcp.types import Tool, TextContent
        MCP_AVAILABLE = True
    except ImportError:
        MCP_AVAILABLE = False

from engine.brain import ObsidianMemBrain, create_brain, load_config


def create_mcp_server(brain: ObsidianMemBrain) -> "Server":
    server = Server("obsidian-mem")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="remember",
                description=(
                    "Запомнить знания из разговора. Вызывай ПОСЛЕ каждого содержательного "
                    "разговора. Извлекает сущности (люди, проекты, технологии), факты и связи, "
                    "сохраняет в Obsidian vault и индексирует для семантического поиска."
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
                            "description": "Массив сообщений разговора",
                        },
                    },
                    "required": ["conversation"],
                },
            ),
            Tool(
                name="remember_text",
                description="Запомнить знания из текста. Например: 'Запомни что я работаю в Uzum Bank'.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Текст для запоминания"},
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="recall",
                description=(
                    "Вспомнить информацию из памяти. Использует СЕМАНТИЧЕСКИЙ ПОИСК — "
                    "находит по смыслу, не только по словам. Вызывай ПЕРЕД ответом когда "
                    "нужен контекст о пользователе, проектах, технологиях."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Что вспомнить (тема, вопрос, имя)"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="search",
                description=(
                    "Семантический поиск — возвращает структурированные результаты с score. "
                    "Каждый результат содержит entity, type, score, facts, relations."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Поисковый запрос"},
                        "top_k": {"type": "integer", "description": "Кол-во результатов (1-10)", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="recall_all",
                description="Вспомнить ВСЁ что знаем о пользователе. Полный обзор vault.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="vault_stats",
                description="Статистика vault — заметки, типы, связи, vector index.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            if name == "remember":
                result = brain.remember(arguments["conversation"])
                return [TextContent(
                    type="text",
                    text=(
                        f"✅ Запомнил!\n"
                        f"Создано: {', '.join(result['entities_created']) or 'ничего'}\n"
                        f"Обновлено: {', '.join(result['entities_updated']) or 'ничего'}"
                    ),
                )]

            elif name == "remember_text":
                result = brain.remember_text(arguments["text"])
                return [TextContent(
                    type="text",
                    text=(
                        f"✅ Запомнил!\n"
                        f"Создано: {', '.join(result['entities_created']) or 'ничего'}\n"
                        f"Обновлено: {', '.join(result['entities_updated']) or 'ничего'}"
                    ),
                )]

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
            return [TextContent(type="text", text=f"❌ Ошибка: {str(e)}")]

    return server


async def main():
    if not MCP_AVAILABLE:
        print("❌ MCP SDK не установлен: pip install mcp", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    brain = create_brain(config_path)

    server = create_mcp_server(brain)

    # MCP SDK v1.x compatible
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())

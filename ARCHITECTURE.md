# Mengram v2 — Второй мозг из разговоров с AI

## Идея

Ты общаешься с Claude (или любой LLM). Система **автоматически** извлекает
знания из разговоров и строит Obsidian vault — твой второй мозг.

## Как это работает

```
                         ┌──────────────────────┐
                         │   Ты общаешься       │
                         │   с Claude / GPT /    │
                         │   любой LLM           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   CONVERSATION EXTRACTOR       │
                    │                               │
                    │ Анализирует разговор:          │
                    │ • Кто упомянут? (люди)         │
                    │ • Какие проекты?               │
                    │ • Какие технологии?            │
                    │ • Какие факты?                 │
                    │ • Какие связи между ними?      │
                    └───────────────┬───────────────┘
                                    │ extracted knowledge
                                    ▼
                    ┌───────────────────────────────┐
                    │     VAULT MANAGER              │
                    │                               │
                    │ Создаёт/обновляет .md файлы:  │
                    │ • Ali.md ← новые факты         │
                    │ • PostgreSQL.md ← обновление   │
                    │ • Проект Alpha.md ← создание   │
                    │ • [[links]] между файлами      │
                    └───────────────┬───────────────┘
                                    │ .md files
                                    ▼
                    ┌───────────────────────────────┐
                    │      OBSIDIAN VAULT            │
                    │                               │
                    │  📄 Ali.md                     │
                    │  📄 Uzum Bank.md               │
                    │  📄 Проект Alpha.md            │
                    │  📄 PostgreSQL.md              │
                    │  📄 Spring Boot.md             │
                    │                               │
                    │  Можно открыть в Obsidian!     │
                    │  → Graph View                  │
                    │  → Редактировать               │
                    │  → Добавлять заметки руками    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     MEMORY RETRIEVAL           │
                    │                               │
                    │ При следующем разговоре:       │
                    │ Claude спрашивает "что я знаю  │
                    │ об этом пользователе?"         │
                    │ → Ищет в vault                 │
                    │ → Возвращает контекст          │
                    │ → Claude отвечает умнее        │
                    └───────────────────────────────┘
```

## Структура проекта

```
mengram-v2/
├── engine/
│   ├── extractor/
│   │   ├── conversation_extractor.py  # Извлечение знаний из разговоров
│   │   └── llm_client.py             # Клиент для LLM (Claude/OpenAI/Ollama)
│   ├── vault_manager/
│   │   └── vault_manager.py          # Создание/обновление .md файлов
│   ├── graph/
│   │   └── knowledge_graph.py        # Индекс связей (SQLite кеш)
│   ├── vector/
│   │   ├── embedder.py               # Локальные embeddings
│   │   └── vector_store.py           # Семантический поиск
│   └── retrieval/
│       └── hybrid_search.py          # Поиск контекста для LLM
├── api/
│   └── mcp_server.py                 # MCP Server (Claude Desktop / Cursor)
├── vault/                            # Автоматически создаётся — Obsidian vault
├── tests/
├── config.yaml                       # Настройки (LLM provider, vault path, etc.)
├── setup.sh
└── README.md
```

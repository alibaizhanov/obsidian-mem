"""
Conversation Extractor v2 — извлекает RICH знания из разговоров.

Извлекает:
1. Entities (person, project, technology, company, concept)
2. Facts — короткие утверждения
3. Relations — связи между entities
4. Knowledge — решения, формулы, рецепты, конфиги, команды (с артефактами)

Knowledge — это killer feature. LLM сам определяет тип знания:
  [solution] — решение проблемы (код, конфиг)
  [formula] — формула, уравнение
  [treatment] — лечение, назначение
  [experiment] — результат эксперимента
  [recipe] — рецепт (кулинария, процесс)
  [decision] — принятое решение
  [command] — полезная команда / инструкция
  [reference] — ссылка, источник
  [insight] — наблюдение, инсайт
  [example] — пример, кейс
  ... любой другой тип который подходит по смыслу
"""

import sys
import json
from dataclasses import dataclass, field
from typing import Optional

from engine.extractor.llm_client import LLMClient


EXTRACTION_PROMPT = """You are a precision knowledge extraction system. Extract IMPORTANT, LASTING personal knowledge from the USER's messages.

Return ONLY valid JSON without markdown. Be VERY selective — quality over quantity.

WHO TO EXTRACT ABOUT:
- Extract knowledge revealed by the USER about themselves, their projects, their life
- DO NOT extract general knowledge the assistant explained (e.g. "nginx config is at /etc/nginx" — unless user confirms they used it)
- If assistant suggests something and user says "yes, I did that" or "that worked" — extract the user's action/result
- Focus on: user's identity, skills, preferences, projects, decisions, workflows, relationships

ENTITY RULES:
- ONLY named, specific entities with 2+ extractable facts
- If something is mentioned once in passing — make it a fact on the parent entity, NOT a separate entity
  GOOD: Entity "Mengram" → fact: "uses Redis as cache"
  BAD: Entity "Redis" → fact: "used as cache" (only 1 fact, make it a fact on the project instead)
- entity_type: person, project, technology, company, concept
- If user says "I"/"me"/"my" — resolve to their name if known, otherwise "User"
{existing_context}
ENTITY NAMING:
- EXACT casing from context: "Mengram" not "MENGRAM", "PostgreSQL" not "postgresql"
- FULL official name: "Ali Baizhanov" not "Ali", "Uzum Bank" not "uzum"
- If entity already exists above — use EXACT SAME NAME (do not create duplicates)

FACT RULES:
- Normalized format: subject + verb + object, present tense, under 10 words
  GOOD: "uses Python", "deployed on Railway", "prefers dark mode"
  BAD: "he has been using Python for a while now", "is currently in the process of deploying"
- ONLY facts that DIRECTLY describe the entity they're assigned to
- Keep project facts on projects, personal facts on the person — don't mix
- DO NOT extract: temporary actions ("asked about X"), session events ("sent a message"), assistant's explanations

FACT DEDUP — check existing facts above. Do NOT re-extract facts that already exist (even if worded slightly differently).
If user says "I use Python" and existing context already has "uses Python" → skip it.
If user says "I switched from React to Svelte" and existing has "uses React" → extract "switched to Svelte" (this is NEW info).

Response format (strict JSON, no ```):
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "person|project|technology|company|concept",
      "facts": ["fact 1", "fact 2"]
    }}
  ],
  "relations": [
    {{
      "from": "Entity 1",
      "to": "Entity 2",
      "type": "works_at|uses|member_of|related_to|depends_on|created_by",
      "description": "short description"
    }}
  ],
  "knowledge": [
    {{
      "entity": "Entity this knowledge belongs to",
      "type": "solution|formula|command|insight|decision|recipe|reference",
      "title": "Short descriptive title",
      "content": "Detailed explanation",
      "artifact": "code/config/formula/command (optional, null if none)"
    }}
  ]
}}

EXAMPLE:
Input conversation:
  User: "Я вчера задеплоил mengram на Railway, всё работает. Пришлось повозиться с pgvector"
  Assistant: "Отлично! Какая версия PostgreSQL?"
  User: "15-я, на Supabase хостится"

Output:
{{
  "entities": [
    {{"name": "Mengram", "type": "project", "facts": ["deployed on Railway", "uses pgvector extension"]}},
    {{"name": "Supabase", "type": "technology", "facts": ["hosts PostgreSQL 15 for Mengram"]}}
  ],
  "relations": [
    {{"from": "Mengram", "to": "Railway", "type": "depends_on", "description": "deployed on"}},
    {{"from": "Mengram", "to": "Supabase", "type": "depends_on", "description": "database hosting"}}
  ],
  "knowledge": []
}}

CONVERSATION:
{conversation}

Extract knowledge (return ONLY JSON):"""


EXISTING_CONTEXT_BLOCK = """
EXISTING ENTITIES FOR THIS USER (use same names, avoid duplicate facts):
{context}
"""


@dataclass
class ExtractedEntity:
    """Извлечённая сущность"""
    name: str
    entity_type: str  # person, project, technology, company, concept
    facts: list[str] = field(default_factory=list)

    def __repr__(self):
        return f"Entity({self.entity_type}: {self.name}, facts={len(self.facts)})"


@dataclass
class ExtractedRelation:
    """Извлечённая связь"""
    from_entity: str
    to_entity: str
    relation_type: str
    description: str = ""

    def __repr__(self):
        return f"Relation({self.from_entity} --{self.relation_type}--> {self.to_entity})"


@dataclass
class ExtractedKnowledge:
    """Извлечённое знание — solution, formula, command, etc."""
    entity: str           # к какой entity относится
    knowledge_type: str   # solution, formula, treatment, command, insight, ...
    title: str            # краткий заголовок
    content: str          # подробное описание
    artifact: Optional[str] = None  # код, конфиг, формула, команда

    def __repr__(self):
        has_artifact = "📎" if self.artifact else ""
        return f"Knowledge([{self.knowledge_type}] {self.title} → {self.entity} {has_artifact})"


@dataclass
class ExtractionResult:
    """Результат извлечения знаний из разговора"""
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    knowledge: list[ExtractedKnowledge] = field(default_factory=list)
    raw_response: str = ""

    def __repr__(self):
        return (
            f"ExtractionResult(entities={len(self.entities)}, "
            f"relations={len(self.relations)}, "
            f"knowledge={len(self.knowledge)})"
        )


class ConversationExtractor:
    """Извлекает структурированные знания из разговоров"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract(self, conversation: list[dict], existing_context: str = "") -> ExtractionResult:
        conv_text = self._format_conversation(conversation)

        # Build context block
        if existing_context:
            context_block = EXISTING_CONTEXT_BLOCK.format(context=existing_context)
        else:
            context_block = ""

        prompt = EXTRACTION_PROMPT.format(
            conversation=conv_text,
            existing_context=context_block
        )
        raw_response = self.llm.complete(prompt)
        return self._parse_response(raw_response)

    def extract_from_text(self, text: str) -> ExtractionResult:
        conversation = [{"role": "user", "content": text}]
        return self.extract(conversation)

    def _format_conversation(self, conversation: list[dict]) -> str:
        lines = []
        for msg in conversation:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n\n".join(lines)

    def _parse_response(self, raw: str) -> ExtractionResult:
        result = ExtractionResult(raw_response=raw)

        # Clean markdown
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    print(f"⚠️  Failed to parse JSON from LLM", file=sys.stderr)
                    return result
            else:
                print(f"⚠️  LLM returned no JSON", file=sys.stderr)
                return result

        # Entities
        for e in data.get("entities", []):
            result.entities.append(ExtractedEntity(
                name=e.get("name", "Unknown"),
                entity_type=e.get("type", "concept"),
                facts=e.get("facts", []),
            ))

        # Relations
        for r in data.get("relations", []):
            result.relations.append(ExtractedRelation(
                from_entity=r.get("from", ""),
                to_entity=r.get("to", ""),
                relation_type=r.get("type", "related_to"),
                description=r.get("description", ""),
            ))

        # Knowledge (NEW)
        for k in data.get("knowledge", []):
            result.knowledge.append(ExtractedKnowledge(
                entity=k.get("entity", ""),
                knowledge_type=k.get("type", "insight"),
                title=k.get("title", ""),
                content=k.get("content", ""),
                artifact=k.get("artifact"),
            ))

        return result


# --- Mock for testing ---

class MockLLMClient(LLMClient):
    """Mock LLM for testing without API"""

    def complete(self, prompt: str, system: str = "") -> str:
        return json.dumps({
            "entities": [
                {
                    "name": "User",
                    "type": "person",
                    "facts": [
                        "Работает backend разработчиком",
                        "Работает в Uzum Bank",
                        "Основной стек: Java, Spring Boot"
                    ]
                },
                {
                    "name": "Uzum Bank",
                    "type": "company",
                    "facts": ["Банк в Узбекистане", "Микросервисная архитектура"]
                },
                {
                    "name": "Проект Alpha",
                    "type": "project",
                    "facts": ["Backend сервис для платежей", "Проблема с connection pool"]
                },
                {
                    "name": "PostgreSQL",
                    "type": "technology",
                    "facts": ["Основная БД", "Версия 15"]
                },
                {
                    "name": "Spring Boot",
                    "type": "technology",
                    "facts": ["Основной фреймворк для микросервисов"]
                }
            ],
            "relations": [
                {"from": "User", "to": "Uzum Bank", "type": "works_at", "description": "Backend разработчик"},
                {"from": "User", "to": "Проект Alpha", "type": "member_of", "description": "Работает над проектом"},
                {"from": "Проект Alpha", "to": "PostgreSQL", "type": "uses", "description": "Основная БД"},
                {"from": "Проект Alpha", "to": "Spring Boot", "type": "uses", "description": "Backend фреймворк"},
                {"from": "Uzum Bank", "to": "Проект Alpha", "type": "related_to", "description": "Проект банка"}
            ],
            "knowledge": [
                {
                    "entity": "PostgreSQL",
                    "type": "solution",
                    "title": "Connection pool exhaustion fix",
                    "content": "OOM при 200+ WebSocket соединениях. Каждый WS держал отдельный connection. Решение: Redis кеш для UserService и BlockedAccountService.",
                    "artifact": "spring.datasource.hikari.maximum-pool-size: 20\nspring.datasource.hikari.idle-timeout: 30000\nspring.datasource.hikari.connection-timeout: 5000"
                },
                {
                    "entity": "PostgreSQL",
                    "type": "command",
                    "title": "Check active connections",
                    "content": "Мониторинг активных соединений PostgreSQL",
                    "artifact": "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
                }
            ]
        }, ensure_ascii=False)

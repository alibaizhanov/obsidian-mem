"""
Conversation Extractor — извлекает знания из разговоров с AI.

Берёт разговор (user + assistant сообщения) и с помощью LLM:
1. Извлекает сущности (люди, проекты, технологии, компании)
2. Извлекает факты о каждой сущности
3. Определяет связи между сущностями
4. Возвращает структурированные данные для записи в vault
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from engine.extractor.llm_client import LLMClient


EXTRACTION_PROMPT = """Ты — система извлечения знаний. Проанализируй разговор между пользователем и AI-ассистентом.

Извлеки ВСЕ сущности, факты и связи. Верни ТОЛЬКО валидный JSON без markdown.

Правила:
- entity_type: person, project, technology, company, concept
- Факты — короткие, конкретные утверждения
- Связи — как сущности связаны друг с другом
- Если пользователь говорит "я" — это сущность с именем "User" (type: person)
- Извлекай даже неявные факты (если говорит "мы используем Kafka" — значит есть проект где используется Kafka)

Формат ответа (строго JSON, без ```):
{{
  "entities": [
    {{
      "name": "Имя сущности",
      "type": "person|project|technology|company|concept",
      "facts": [
        "факт 1 об этой сущности",
        "факт 2 об этой сущности"
      ]
    }}
  ],
  "relations": [
    {{
      "from": "Имя сущности 1",
      "to": "Имя сущности 2",
      "type": "works_at|uses|member_of|related_to|depends_on|created_by",
      "description": "описание связи"
    }}
  ]
}}

РАЗГОВОР:
{conversation}

Извлеки знания (верни ТОЛЬКО JSON):"""


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
class ExtractionResult:
    """Результат извлечения знаний из разговора"""
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    raw_response: str = ""

    def __repr__(self):
        return f"ExtractionResult(entities={len(self.entities)}, relations={len(self.relations)})"


class ConversationExtractor:
    """Извлекает структурированные знания из разговоров"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract(self, conversation: list[dict]) -> ExtractionResult:
        """
        Извлекает знания из разговора.

        Args:
            conversation: список сообщений [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            ExtractionResult с entities и relations
        """
        # Форматируем разговор
        conv_text = self._format_conversation(conversation)

        # Вызываем LLM
        prompt = EXTRACTION_PROMPT.format(conversation=conv_text)
        raw_response = self.llm.complete(prompt)

        # Парсим JSON
        return self._parse_response(raw_response)

    def extract_from_text(self, text: str) -> ExtractionResult:
        """Извлекает знания из произвольного текста"""
        conversation = [{"role": "user", "content": text}]
        return self.extract(conversation)

    def _format_conversation(self, conversation: list[dict]) -> str:
        """Форматирует разговор в текст"""
        lines = []
        for msg in conversation:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            lines.append(f"{role}: {msg['content']}")
        return "\n\n".join(lines)

    def _parse_response(self, raw: str) -> ExtractionResult:
        """Парсит JSON ответ от LLM"""
        result = ExtractionResult(raw_response=raw)

        # Чистим ответ от markdown
        clean = raw.strip()
        if clean.startswith("```"):
            # Убираем ```json и ```
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Пробуем найти JSON в тексте
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    print(f"⚠️  Не удалось распарсить JSON от LLM")
                    return result
            else:
                print(f"⚠️  LLM не вернул JSON")
                return result

        # Извлекаем entities
        for e in data.get("entities", []):
            result.entities.append(ExtractedEntity(
                name=e.get("name", "Unknown"),
                entity_type=e.get("type", "concept"),
                facts=e.get("facts", []),
            ))

        # Извлекаем relations
        for r in data.get("relations", []):
            result.relations.append(ExtractedRelation(
                from_entity=r.get("from", ""),
                to_entity=r.get("to", ""),
                relation_type=r.get("type", "related_to"),
                description=r.get("description", ""),
            ))

        return result


# --- Для тестирования без LLM API ---

class MockLLMClient(LLMClient):
    """Мок для тестирования без реального LLM"""

    def complete(self, prompt: str, system: str = "") -> str:
        """Возвращает пример извлечённых данных"""
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
                    "facts": ["Банк в Узбекистане", "Использует микросервисную архитектуру"]
                },
                {
                    "name": "Проект Alpha",
                    "type": "project",
                    "facts": ["Backend сервис для платежей", "Проблема с connection pool"]
                },
                {
                    "name": "PostgreSQL",
                    "type": "technology",
                    "facts": ["Основная БД", "Версия 15", "Проблема с connection pool exhaustion"]
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
            ]
        }, ensure_ascii=False)


if __name__ == "__main__":
    # Тест с мок-клиентом (без реального API)
    print("🧪 Тест Conversation Extractor (mock LLM)\n")

    extractor = ConversationExtractor(MockLLMClient())

    conversation = [
        {"role": "user", "content": "Я работаю в Uzum Bank, backend разработчик. Делаю микросервисы на Spring Boot."},
        {"role": "assistant", "content": "Отлично! Какие базы данных используете?"},
        {"role": "user", "content": "PostgreSQL 15 как основная БД. Сейчас проблема с connection pool в Проект Alpha."},
    ]

    result = extractor.extract(conversation)
    print(f"📊 {result}\n")

    print("Entities:")
    for e in result.entities:
        print(f"  {e}")
        for fact in e.facts:
            print(f"    • {fact}")

    print(f"\nRelations:")
    for r in result.relations:
        print(f"  {r}")

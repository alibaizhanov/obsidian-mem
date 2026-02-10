"""
ObsidianMem SDK — Mem0-совместимый API с Knowledge Graph.

Использование:
    from obsidian_mem import Memory

    m = Memory(vault_path="./vault", llm_provider="anthropic", api_key="sk-ant-...")

    # Запомнить
    m.add("Я работаю в Uzum Bank, backend на Spring Boot", user_id="ali")

    # Найти
    results = m.search("где работает ali?", user_id="ali")

    # Всё что знаем
    all_memories = m.get_all(user_id="ali")

    # Удалить
    m.delete("PostgreSQL")

    # Статистика
    m.stats()

Отличие от Mem0:
    - Данные в .md файлах (можно открыть в Obsidian)
    - Knowledge Graph с типизированными связями
    - Полностью локальный
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from engine.brain import ObsidianMemBrain
from engine.extractor.llm_client import (
    LLMClient,
    AnthropicClient,
    OpenAIClient,
    OllamaClient,
    create_llm_client,
)
from engine.extractor.conversation_extractor import (
    ConversationExtractor,
    ExtractionResult,
    MockLLMClient,
)
from engine.vault_manager.vault_manager import VaultManager


@dataclass
class MemoryItem:
    """Один элемент памяти (entity + facts)"""
    id: str
    name: str
    entity_type: str
    facts: list[str]
    relations: list[dict]
    source_file: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Memory({self.entity_type}: {self.name}, facts={len(self.facts)})"


@dataclass
class SearchResult:
    """Результат поиска"""
    memory: MemoryItem
    score: float = 1.0
    context: str = ""

    def __repr__(self):
        return f"SearchResult({self.memory.name}, score={self.score:.2f})"


class Memory:
    """
    ObsidianMem — Mem0-совместимый API с Knowledge Graph.

    Каждый user_id получает свой vault (подпапку).
    Внутри vault — .md файлы с entities, facts, [[links]].
    """

    def __init__(
        self,
        vault_path: str = "./vault",
        llm_provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.base_vault_path = Path(vault_path)
        self.base_vault_path.mkdir(parents=True, exist_ok=True)

        # LLM client
        self.llm = self._create_llm(llm_provider, api_key, model, ollama_url)
        self.extractor = ConversationExtractor(self.llm)

        # Кеш brain-ов по user_id
        self._brains: dict[str, ObsidianMemBrain] = {}

    def _create_llm(
        self, provider: str, api_key: Optional[str],
        model: Optional[str], ollama_url: str
    ) -> LLMClient:
        """Создаёт LLM клиент"""
        if provider == "anthropic":
            key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            return AnthropicClient(api_key=key, model=model or "claude-sonnet-4-20250514")
        elif provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY", "")
            return OpenAIClient(api_key=key, model=model or "gpt-4o-mini")
        elif provider == "ollama":
            return OllamaClient(base_url=ollama_url, model=model or "llama3.2")
        elif provider == "mock":
            return MockLLMClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _get_brain(self, user_id: str = "default") -> ObsidianMemBrain:
        """Получает brain для конкретного user_id"""
        if user_id not in self._brains:
            user_vault = str(self.base_vault_path / user_id)
            self._brains[user_id] = ObsidianMemBrain(
                vault_path=user_vault,
                llm_client=self.llm,
            )
        return self._brains[user_id]

    # ==========================================
    # Основные методы (Mem0-совместимые)
    # ==========================================

    def add(
        self,
        messages: str | list[dict],
        user_id: str = "default",
    ) -> dict:
        """
        Добавить память из текста или разговора.

        Args:
            messages: Текст или [{"role": "user", "content": "..."}]
            user_id: ID пользователя

        Returns:
            {"entities_created": [...], "entities_updated": [...]}

        Примеры:
            m.add("Я работаю в Uzum Bank", user_id="ali")
            m.add([
                {"role": "user", "content": "Используем PostgreSQL 15"},
                {"role": "assistant", "content": "Хороший выбор!"},
            ], user_id="ali")
        """
        brain = self._get_brain(user_id)

        if isinstance(messages, str):
            return brain.remember_text(messages)
        else:
            return brain.remember(messages)

    def search(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Семантический поиск по памяти (vector + graph).

        Args:
            query: Запрос (ищет по смыслу, не только по словам)
            user_id: ID пользователя
            top_k: Максимум результатов

        Returns:
            [SearchResult(memory=..., score=..., context=...)]

        Примеры:
            results = m.search("проблемы с базой данных", user_id="ali")
            for r in results:
                print(f"{r.memory.name} (score={r.score:.2f})")
                print(r.memory.facts)
        """
        brain = self._get_brain(user_id)

        # Используем semantic search из brain
        raw_results = brain.search(query, top_k=top_k)
        context = brain.recall(query, top_k=top_k)

        results = []
        for data in raw_results:
            vault_path = str(self.base_vault_path / user_id)
            source_file = str(Path(vault_path) / f"{data['entity']}.md")

            memory = MemoryItem(
                id=data["entity"].lower().replace(" ", "_"),
                name=data["entity"],
                entity_type=data.get("type", "unknown"),
                facts=data.get("facts", []),
                relations=data.get("relations", []),
                source_file=source_file if Path(source_file).exists() else "",
                metadata={},
            )
            results.append(SearchResult(
                memory=memory,
                score=data.get("score", 0.5),
                context=context,
            ))

        return results

    def get_all(self, user_id: str = "default") -> list[MemoryItem]:
        """
        Получить все memories пользователя.

        Returns:
            [MemoryItem(...), ...]
        """
        brain = self._get_brain(user_id)
        graph = brain.graph
        entities = graph.all_entities()

        results = []
        for entity in entities:
            if entity.entity_type == "tag":
                continue

            facts = self._extract_facts_from_file(entity.source_file)
            neighbors = graph.get_neighbors(entity.id, depth=1)
            relations = [
                {
                    "type": n["relation_type"],
                    "target": n["entity"].name,
                    "target_type": n["entity"].entity_type,
                }
                for n in neighbors
                if n["entity"].entity_type != "tag"
            ]

            results.append(MemoryItem(
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                facts=facts,
                relations=relations,
                source_file=entity.source_file or "",
                metadata=entity.metadata or {},
            ))

        return results

    def get(self, entity_name: str, user_id: str = "default") -> Optional[MemoryItem]:
        """
        Получить конкретную entity по имени.

        Примеры:
            pg = m.get("PostgreSQL", user_id="ali")
            print(pg.facts, pg.relations)
        """
        brain = self._get_brain(user_id)
        entity = brain.graph.find_entity(entity_name)
        if not entity:
            return None

        facts = self._extract_facts_from_file(entity.source_file)
        neighbors = brain.graph.get_neighbors(entity.id, depth=1)
        relations = [
            {
                "type": n["relation_type"],
                "target": n["entity"].name,
                "target_type": n["entity"].entity_type,
            }
            for n in neighbors
            if n["entity"].entity_type != "tag"
        ]

        return MemoryItem(
            id=entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            facts=facts,
            relations=relations,
            source_file=entity.source_file or "",
            metadata=entity.metadata or {},
        )

    def delete(self, entity_name: str, user_id: str = "default") -> bool:
        """
        Удалить entity из vault (удаляет .md файл).

        Returns:
            True если удалено, False если не найдено
        """
        brain = self._get_brain(user_id)
        vault = brain.vault_manager
        file_path = vault._entity_file_path(entity_name)

        if file_path.exists():
            file_path.unlink()
            brain._graph = None  # Инвалидируем кеш
            return True
        return False

    def stats(self, user_id: str = "default") -> dict:
        """Статистика vault"""
        brain = self._get_brain(user_id)
        return brain.get_stats()

    def graph(self, entity_name: str, user_id: str = "default", depth: int = 2) -> dict:
        """
        Получить подграф вокруг entity.

        Returns:
            {"center": Entity, "nodes": [...], "edges": [...]}
        """
        brain = self._get_brain(user_id)
        entity = brain.graph.find_entity(entity_name)
        if not entity:
            return {"center": None, "nodes": [], "edges": []}
        return brain.graph.get_subgraph(entity.id, depth=depth)

    def _extract_facts_from_file(self, file_path: Optional[str]) -> list[str]:
        """Извлекает факты из .md файла"""
        if not file_path:
            return []
        path = Path(file_path)
        if not path.exists():
            return []

        content = path.read_text(encoding="utf-8")
        facts = []
        in_facts_section = False

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("## Факты") or line.startswith("## Обновления"):
                in_facts_section = True
                continue
            if line.startswith("## ") and in_facts_section:
                in_facts_section = False
                continue
            if in_facts_section and line.startswith("- "):
                # Убираем [[links]] для чистоты
                import re
                clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", line[2:])
                if clean and not clean.startswith("*"):  # Пропускаем даты
                    facts.append(clean)

        return facts


# ==========================================
# Удобная функция инициализации
# ==========================================

def init(
    vault_path: str = "./vault",
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Memory:
    """
    Быстрая инициализация.

    Примеры:
        import obsidian_mem
        m = obsidian_mem.init(provider="anthropic", api_key="sk-ant-...")
        m.add("Я люблю Python", user_id="ali")
    """
    return Memory(
        vault_path=vault_path,
        llm_provider=provider,
        api_key=api_key,
        model=model,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🧠 ObsidianMem SDK — Demo")
    print("=" * 60)

    # Mock LLM для теста
    m = Memory(vault_path="./demo_sdk_vault", llm_provider="mock")

    # 1. Add
    print("\n📝 m.add('Я работаю в Uzum Bank...')")
    result = m.add(
        "Я работаю в Uzum Bank, backend разработчик на Spring Boot. "
        "Проблема с PostgreSQL connection pool в Проект Alpha.",
        user_id="ali",
    )
    print(f"   Created: {result['entities_created']}")
    print(f"   Updated: {result['entities_updated']}")

    # 2. Get all
    print(f"\n📋 m.get_all():")
    all_memories = m.get_all(user_id="ali")
    for mem in all_memories:
        print(f"   {mem}")
        for fact in mem.facts:
            print(f"      • {fact}")

    # 3. Get specific
    print(f"\n🔍 m.get('PostgreSQL'):")
    pg = m.get("PostgreSQL", user_id="ali")
    if pg:
        print(f"   {pg}")
        print(f"   Facts: {pg.facts}")
        print(f"   Relations: {pg.relations}")

    # 4. Search
    print(f"\n🔍 m.search('база данных'):")
    results = m.search("база данных", user_id="ali")
    for r in results:
        print(f"   {r}")

    # 5. Stats
    print(f"\n📊 m.stats():")
    print(f"   {m.stats(user_id='ali')}")

    # 6. Graph
    print(f"\n🕸️ m.graph('User'):")
    g = m.graph("User", user_id="ali")
    print(f"   Nodes: {len(g.get('nodes', []))}")
    print(f"   Edges: {len(g.get('edges', []))}")

    # Сравнение с Mem0
    print(f"\n{'='*60}")
    print("📊 Сравнение API:")
    print(f"{'='*60}")
    print("""
    # Mem0:
    from mem0 import Memory
    m = Memory()
    m.add("Я работаю в Uzum Bank", user_id="ali")
    m.search("где работает ali?")
    
    # ObsidianMem (наш):
    from obsidian_mem import Memory
    m = Memory(vault_path="./vault", llm_provider="anthropic", api_key="...")
    m.add("Я работаю в Uzum Bank", user_id="ali")
    m.search("где работает ali?")
    
    # Разница: m.get("PostgreSQL") → типизированная entity с facts + relations + graph
    # У Mem0 такого нет
    """)

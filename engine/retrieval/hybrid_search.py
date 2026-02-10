"""
Hybrid Retrieval Engine — объединяет Vector Search + Graph Traversal.

Это главная фича, которая отличает нас от Mem0:
1. Vector search → находит семантически похожие чанки
2. Graph expansion → расширяет через связи на N уровней
3. Context assembly → собирает всё в структурированный ответ для AI
"""

from dataclasses import dataclass, field

from engine.graph.knowledge_graph import KnowledgeGraph, build_graph_from_vault
from engine.vector.vector_store import VectorStore, SearchResult, index_vault


@dataclass
class RetrievalResult:
    """Полный результат гибридного поиска"""
    query: str
    # Прямые совпадения из vector search
    direct_matches: list[SearchResult] = field(default_factory=list)
    # Связанные entities из graph expansion
    graph_context: list[dict] = field(default_factory=list)
    # Собранный контекст для AI-агента
    assembled_context: str = ""

    def __repr__(self):
        return (
            f"RetrievalResult(\n"
            f"  query='{self.query}'\n"
            f"  direct_matches={len(self.direct_matches)}\n"
            f"  graph_entities={len(self.graph_context)}\n"
            f"  context_length={len(self.assembled_context)} chars\n"
            f")"
        )


class HybridRetrieval:
    """
    Гибридный поиск: Vector + Graph.

    Workflow:
    1. Vector search по запросу → top-K чанков
    2. Извлекаем entity_id из найденных чанков
    3. Graph traversal от этих entities на depth уровней
    4. Собираем контекст: прямые совпадения + граф связей
    """

    def __init__(self, graph: KnowledgeGraph, vector_store: VectorStore):
        self.graph = graph
        self.vector_store = vector_store

    def query(self, text: str, top_k: int = 5, graph_depth: int = 1,
              min_score: float = 0.15) -> RetrievalResult:
        """
        Главный метод — гибридный поиск.

        Args:
            text: поисковый запрос
            top_k: сколько чанков из vector search
            graph_depth: глубина обхода графа
            min_score: минимальный score для vector search
        """
        result = RetrievalResult(query=text)

        # Step 1: Vector search
        result.direct_matches = self.vector_store.search(
            query=text, top_k=top_k, min_score=min_score
        )

        # Step 2: Graph expansion от найденных entities
        seen_entities = set()
        for match in result.direct_matches:
            entity_id = match.entity_id
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)

            # Получаем entity из графа
            entity = self.graph.get_entity(entity_id)
            if not entity:
                continue

            # Graph traversal
            neighbors = self.graph.get_neighbors(entity_id, depth=graph_depth)
            for neighbor in neighbors:
                n_id = neighbor["entity"].id
                if n_id not in seen_entities:
                    result.graph_context.append(neighbor)
                    seen_entities.add(n_id)

        # Step 3: Собираем контекст для AI
        result.assembled_context = self._assemble_context(result)

        return result

    def get_entity_context(self, entity_name: str, graph_depth: int = 2) -> RetrievalResult:
        """
        Получить полный контекст по конкретной entity.
        Для запросов типа "расскажи всё о Проект Alpha".
        """
        result = RetrievalResult(query=f"context:{entity_name}")

        entity = self.graph.find_entity(entity_name)
        if not entity:
            result.assembled_context = f"Entity '{entity_name}' не найдена в vault."
            return result

        # Все чанки этой entity
        chunks_data = self.vector_store.search_by_entity(entity.id)
        for chunk in chunks_data:
            result.direct_matches.append(SearchResult(
                chunk_id=chunk["id"],
                entity_id=chunk["entity_id"],
                entity_name=chunk["entity_name"],
                section=chunk["section"],
                content=chunk["content"],
                score=1.0,
            ))

        # Graph traversal
        neighbors = self.graph.get_neighbors(entity.id, depth=graph_depth)
        for neighbor in neighbors:
            result.graph_context.append(neighbor)

        result.assembled_context = self._assemble_context(result)
        return result

    def _assemble_context(self, result: RetrievalResult) -> str:
        """
        Собирает человекочитаемый контекст из результатов поиска.
        Этот текст пойдёт в промпт AI-агента.
        """
        parts = []

        # Прямые совпадения
        if result.direct_matches:
            parts.append("## Релевантные фрагменты из заметок\n")
            seen_content = set()
            for match in result.direct_matches:
                if match.content in seen_content:
                    continue
                seen_content.add(match.content)
                parts.append(
                    f"**{match.entity_name}** ({match.section}) "
                    f"[score: {match.score:.2f}]:\n"
                    f"{match.content}\n"
                )

        # Граф связей
        if result.graph_context:
            parts.append("\n## Связанные сущности (из графа знаний)\n")

            # Группируем по типу связи
            by_type: dict[str, list] = {}
            for ctx in result.graph_context:
                rel = ctx["relation_type"]
                entity = ctx["entity"]
                # Пропускаем теги для чистоты контекста
                if entity.entity_type == "tag":
                    continue
                by_type.setdefault(rel, []).append(entity)

            for rel_type, entities in by_type.items():
                names = ", ".join(e.name for e in entities)
                parts.append(f"- **{rel_type}**: {names}")

        return "\n".join(parts)


def build_retrieval_engine(vault_path: str) -> HybridRetrieval:
    """
    Создаёт полный retrieval engine из vault.
    Строит граф + индексирует вектора.
    """
    print("=" * 50)
    print("🏗️  Building ObsidianMem Retrieval Engine")
    print("=" * 50)

    # Step 1: Knowledge Graph
    print("\n📊 Step 1: Building Knowledge Graph...")
    graph = build_graph_from_vault(vault_path)
    stats = graph.stats()
    print(f"   ✅ {stats['total_entities']} entities, {stats['total_relations']} relations")

    # Step 2: Vector Store
    print("\n📐 Step 2: Indexing vectors...")
    vector_store = index_vault(vault_path)

    # Step 3: Hybrid Engine
    print("\n🔗 Step 3: Connecting hybrid retrieval...")
    engine = HybridRetrieval(graph, vector_store)

    print("\n✅ Engine ready!\n")
    return engine


if __name__ == "__main__":
    import sys

    vault_path = sys.argv[1] if len(sys.argv) > 1 else "./test_vault"
    engine = build_retrieval_engine(vault_path)

    # Тестовые запросы
    queries = [
        "проблема с производительностью базы данных",
        "кто работает над backend проектами",
        "какие технологии использует Ali",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: '{q}'")
        print(f"{'='*60}")

        result = engine.query(q, top_k=3, graph_depth=2)
        print(result)
        print(f"\n📋 Assembled context:\n{result.assembled_context}")

    # Тест: контекст по entity
    print(f"\n{'='*60}")
    print(f"🎯 Entity context: 'Проект Alpha'")
    print(f"{'='*60}")
    result = engine.get_entity_context("Проект Alpha")
    print(result)
    print(f"\n📋 Context:\n{result.assembled_context}")

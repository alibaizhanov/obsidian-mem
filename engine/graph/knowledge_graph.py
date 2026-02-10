"""
Knowledge Graph — SQLite-based хранилище графа знаний.

Строит граф из ParsedNote:
  - Entities (ноды): заметки, люди, проекты, технологии
  - Relations (рёбра): [[links]], tags, team membership
  - Позволяет traverse графа на N уровней глубины
"""

import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from engine.parser.markdown_parser import ParsedNote, parse_vault


@dataclass
class Entity:
    """Нода в графе знаний"""
    id: str
    name: str
    entity_type: str  # note, person, project, technology, tag
    source_file: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __repr__(self):
        return f"Entity({self.entity_type}: {self.name})"


@dataclass
class Relation:
    """Ребро в графе знаний"""
    source_id: str
    target_id: str
    relation_type: str  # links_to, tagged, uses, member_of, related_to
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Relation({self.source_id} --{self.relation_type}--> {self.target_id})"


class KnowledgeGraph:
    """
    SQLite-backed Knowledge Graph.

    Хранит entities (ноды) и relations (рёбра).
    Поддерживает multi-hop traversal для graph-aware retrieval.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source_file TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id),
                UNIQUE(source_id, target_id, relation_type)
            );

            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
            CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
        """)
        self.conn.commit()

    # ── CRUD Operations ──

    def add_entity(self, entity: Entity) -> Entity:
        """Добавить или обновить entity"""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """INSERT INTO entities (id, name, entity_type, source_file, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   name = excluded.name,
                   entity_type = excluded.entity_type,
                   source_file = excluded.source_file,
                   metadata = excluded.metadata,
                   updated_at = excluded.updated_at
            """,
            (entity.id, entity.name, entity.entity_type,
             entity.source_file, json.dumps(entity.metadata, ensure_ascii=False, default=str),
             entity.created_at or now, now),
        )
        self.conn.commit()
        return entity

    def add_relation(self, relation: Relation) -> Relation:
        """Добавить связь (пропускает дубликаты)"""
        now = datetime.utcnow().isoformat()
        try:
            self.conn.execute(
                """INSERT OR IGNORE INTO relations 
                   (source_id, target_id, relation_type, weight, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                """,
                (relation.source_id, relation.target_id, relation.relation_type,
                 relation.weight, json.dumps(relation.metadata, ensure_ascii=False),
                 now),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass  # связь уже существует
        return relation

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Получить entity по ID"""
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def find_entity(self, name: str) -> Optional[Entity]:
        """Найти entity по имени (case-insensitive)"""
        row = self.conn.execute(
            "SELECT * FROM entities WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def search_entities(self, query: str, entity_type: str = None) -> list[Entity]:
        """Поиск entities по подстроке в имени"""
        if entity_type:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? AND entity_type = ?",
                (f"%{query}%", entity_type),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE name LIKE ?",
                (f"%{query}%",),
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ── Graph Traversal ──

    def get_neighbors(self, entity_id: str, depth: int = 1, relation_type: str = None) -> list[dict]:
        """
        Получить соседей ноды на заданную глубину.
        Это ключевая функция — graph-aware retrieval.

        Возвращает список {entity, relation_type, distance}
        """
        visited = set()
        results = []
        self._traverse(entity_id, depth, 0, visited, results, relation_type)
        return results

    def _traverse(self, node_id: str, max_depth: int, current_depth: int,
                  visited: set, results: list, relation_type: str = None):
        """Рекурсивный обход графа"""
        if current_depth >= max_depth or node_id in visited:
            return

        visited.add(node_id)

        # Исходящие связи
        query = """
            SELECT r.target_id, r.relation_type, r.weight, e.*
            FROM relations r
            JOIN entities e ON e.id = r.target_id
            WHERE r.source_id = ?
        """
        params = [node_id]
        if relation_type:
            query += " AND r.relation_type = ?"
            params.append(relation_type)

        for row in self.conn.execute(query, params).fetchall():
            target = self._row_to_entity(row)
            results.append({
                "entity": target,
                "relation_type": row["relation_type"],
                "weight": row["weight"],
                "distance": current_depth + 1,
                "from": node_id,
            })
            self._traverse(target.id, max_depth, current_depth + 1, visited, results, relation_type)

        # Входящие связи (обратные)
        query = """
            SELECT r.source_id, r.relation_type, r.weight, e.*
            FROM relations r
            JOIN entities e ON e.id = r.source_id
            WHERE r.target_id = ?
        """
        params = [node_id]
        if relation_type:
            query += " AND r.relation_type = ?"
            params.append(relation_type)

        for row in self.conn.execute(query, params).fetchall():
            source = self._row_to_entity(row)
            if source.id not in visited:
                results.append({
                    "entity": source,
                    "relation_type": f"←{row['relation_type']}",
                    "weight": row["weight"],
                    "distance": current_depth + 1,
                    "from": node_id,
                })
                self._traverse(source.id, max_depth, current_depth + 1, visited, results, relation_type)

    def get_subgraph(self, entity_id: str, depth: int = 2) -> dict:
        """
        Получить подграф вокруг entity.
        Возвращает {center, nodes, edges} — всё что нужно для контекста.
        """
        center = self.get_entity(entity_id)
        if not center:
            return {"center": None, "nodes": [], "edges": []}

        neighbors = self.get_neighbors(entity_id, depth=depth)

        nodes = [center] + [n["entity"] for n in neighbors]
        edges = [
            {
                "from": n["from"],
                "to": n["entity"].id,
                "type": n["relation_type"],
                "distance": n["distance"],
            }
            for n in neighbors
        ]

        return {"center": center, "nodes": nodes, "edges": edges}

    # ── Stats ──

    def stats(self) -> dict:
        """Статистика графа"""
        entities = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relations = self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        types = self.conn.execute(
            "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type"
        ).fetchall()
        return {
            "total_entities": entities,
            "total_relations": relations,
            "by_type": {row[0]: row[1] for row in types},
        }

    def all_entities(self) -> list[Entity]:
        rows = self.conn.execute("SELECT * FROM entities ORDER BY name").fetchall()
        return [self._row_to_entity(r) for r in rows]

    def all_relations(self) -> list[Relation]:
        rows = self.conn.execute("SELECT * FROM relations").fetchall()
        return [
            Relation(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation_type=r["relation_type"],
                weight=r["weight"],
            )
            for r in rows
        ]

    # ── Helpers ──

    def _row_to_entity(self, row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            source_file=row["source_file"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )

    def _make_entity_id(self, name: str) -> str:
        """Генерирует ID из имени"""
        return name.lower().replace(" ", "_").replace("/", "_")

    def close(self):
        self.conn.close()


def _infer_entity_type(note: ParsedNote) -> str:
    """Определяет тип entity по тегам и frontmatter"""
    tags = set(note.tags)
    fm = note.frontmatter

    if "person" in tags or fm.get("role"):
        return "person"
    if "project" in tags or fm.get("status"):
        return "project"
    if "technology" in tags or fm.get("type") in ("RDBMS", "message-broker", "in-memory-store"):
        return "technology"
    return "note"


def build_graph_from_vault(vault_path: str, db_path: str = ":memory:") -> KnowledgeGraph:
    """
    Строит Knowledge Graph из Obsidian vault.

    1. Парсит все .md файлы
    2. Создаёт entities для каждой заметки
    3. Создаёт relations из [[wikilinks]]
    4. Создаёт tag relations
    5. Создаёт relations из frontmatter (team, etc.)
    """
    notes = parse_vault(vault_path)
    graph = KnowledgeGraph(db_path)

    # Pass 1: Создаём entities для каждой заметки
    for note in notes:
        entity_type = _infer_entity_type(note)
        entity = Entity(
            id=graph._make_entity_id(note.name),
            name=note.name,
            entity_type=entity_type,
            source_file=note.file_path,
            metadata=note.frontmatter,
        )
        graph.add_entity(entity)

    # Pass 2: Создаём «phantom» entities для ссылок на несуществующие заметки
    existing_ids = {e.id for e in graph.all_entities()}
    for note in notes:
        for link in note.wikilinks:
            link_id = graph._make_entity_id(link.target)
            if link_id not in existing_ids:
                phantom = Entity(
                    id=link_id,
                    name=link.target,
                    entity_type="reference",  # нет файла, только ссылка
                )
                graph.add_entity(phantom)
                existing_ids.add(link_id)

    # Pass 3: Relations из [[wikilinks]]
    for note in notes:
        source_id = graph._make_entity_id(note.name)
        for link in note.wikilinks:
            target_id = graph._make_entity_id(link.target)
            graph.add_relation(Relation(
                source_id=source_id,
                target_id=target_id,
                relation_type="links_to",
                metadata={"context": link.context},
            ))

    # Pass 4: Relations из frontmatter (team, etc.)
    for note in notes:
        source_id = graph._make_entity_id(note.name)
        fm = note.frontmatter

        # team: [Ali, Doston] → member_of relations
        team = fm.get("team", [])
        if isinstance(team, list):
            for member in team:
                member_id = graph._make_entity_id(member)
                graph.add_relation(Relation(
                    source_id=member_id,
                    target_id=source_id,
                    relation_type="member_of",
                ))

        # skills: [Java, Spring Boot] → has_skill relations
        skills = fm.get("skills", [])
        if isinstance(skills, list):
            for skill in skills:
                skill_id = graph._make_entity_id(skill)
                graph.add_relation(Relation(
                    source_id=source_id,
                    target_id=skill_id,
                    relation_type="has_skill",
                ))

    # Pass 5: Tag entities + relations
    for note in notes:
        source_id = graph._make_entity_id(note.name)
        for tag in note.tags:
            tag_id = f"tag:{tag}"
            if tag_id not in existing_ids:
                graph.add_entity(Entity(
                    id=tag_id,
                    name=f"#{tag}",
                    entity_type="tag",
                ))
                existing_ids.add(tag_id)
            graph.add_relation(Relation(
                source_id=source_id,
                target_id=tag_id,
                relation_type="tagged",
            ))

    return graph


# --- Entry point ---
if __name__ == "__main__":
    import sys

    vault_path = sys.argv[1] if len(sys.argv) > 1 else "./test_vault"
    graph = build_graph_from_vault(vault_path)

    stats = graph.stats()
    print(f"\n📊 Graph Stats:")
    print(f"   Entities: {stats['total_entities']}")
    print(f"   Relations: {stats['total_relations']}")
    print(f"   By type: {stats['by_type']}")

    # Тест: соседи Проект Alpha на 2 уровня
    print(f"\n🔍 Neighbors of 'проект_alpha' (depth=2):")
    neighbors = graph.get_neighbors("проект_alpha", depth=2)
    for n in neighbors:
        indent = "  " * n["distance"]
        print(f"  {indent}→ [{n['relation_type']}] {n['entity'].name} (d={n['distance']})")

    # Тест: подграф Ali
    print(f"\n🕸️  Subgraph around 'ali' (depth=2):")
    sg = graph.get_subgraph("ali", depth=2)
    print(f"   Center: {sg['center']}")
    print(f"   Nodes: {len(sg['nodes'])}")
    print(f"   Edges: {len(sg['edges'])}")
    for edge in sg["edges"][:10]:
        print(f"   {edge['from']} --{edge['type']}--> {edge['to']} (d={edge['distance']})")

"""
Mengram Cloud Storage — PostgreSQL backend.

Replaces VaultManager (local .md files) with PostgreSQL + pgvector.
Same interface, different storage.

Usage:
    store = CloudStore(database_url="postgresql://...")
    store.save_entity("PostgreSQL", "technology", facts=[...], relations=[...], knowledge=[...])
    results = store.search("database pool", user_id="...", top_k=5)
"""

import hashlib
import secrets
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


@dataclass
class CloudEntity:
    id: str
    name: str
    type: str
    facts: list[str]
    relations: list[dict]
    knowledge: list[dict]


class CloudStore:
    """
    PostgreSQL storage backend for Mengram Cloud.
    
    Sync API (psycopg2) for simplicity. 
    Can be swapped to async (asyncpg) for production.
    """

    def __init__(self, database_url: str):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("pip install psycopg2-binary")
        self.database_url = database_url
        self.conn = psycopg2.connect(database_url)
        self.conn.autocommit = True

    def close(self):
        if self.conn:
            self.conn.close()

    # ---- Auth ----

    def create_user(self, email: str) -> str:
        """Create user, return user_id."""
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email) VALUES (%s) RETURNING id",
                (email,)
            )
            return str(cur.fetchone()[0])

    def get_user_by_email(self, email: str) -> Optional[str]:
        """Get user_id by email."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
            return str(row[0]) if row else None

    def create_api_key(self, user_id: str, name: str = "default") -> str:
        """Generate API key, store hash, return raw key."""
        raw_key = f"mg-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:10]

        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO api_keys (user_id, key_hash, key_prefix, name)
                   VALUES (%s, %s, %s, %s)""",
                (user_id, key_hash, key_prefix, name)
            )
        return raw_key

    def verify_api_key(self, raw_key: str) -> Optional[str]:
        """Verify API key, return user_id or None."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT user_id FROM api_keys 
                   WHERE key_hash = %s AND is_active = TRUE""",
                (key_hash,)
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE api_keys SET last_used_at = NOW() WHERE key_hash = %s",
                    (key_hash,)
                )
                return str(row[0])
            return None

    def reset_api_key(self, user_id: str) -> str:
        """Deactivate all old keys and create a new one."""
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE api_keys SET is_active = FALSE WHERE user_id = %s",
                (user_id,)
            )
        return self.create_api_key(user_id, name="reset")

    # ---- Entities ----

    def save_entity(self, user_id: str, name: str, type: str,
                    facts: list[str] = None,
                    relations: list[dict] = None,
                    knowledge: list[dict] = None) -> str:
        """
        Create or update entity with facts, relations, knowledge.
        Returns entity_id.
        """
        with self.conn.cursor() as cur:
            # Upsert entity
            cur.execute(
                """INSERT INTO entities (user_id, name, type)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (user_id, name) 
                   DO UPDATE SET type = EXCLUDED.type, updated_at = NOW()
                   RETURNING id""",
                (user_id, name, type)
            )
            entity_id = str(cur.fetchone()[0])

            # Add facts (skip duplicates)
            for fact in (facts or []):
                cur.execute(
                    """INSERT INTO facts (entity_id, content)
                       VALUES (%s, %s)
                       ON CONFLICT (entity_id, content) DO NOTHING""",
                    (entity_id, fact)
                )

            # Add knowledge (skip duplicates)
            for k in (knowledge or []):
                cur.execute(
                    """INSERT INTO knowledge (entity_id, type, title, content, artifact)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (entity_id, title) DO NOTHING""",
                    (entity_id, k.get("type", "insight"), k.get("title", ""),
                     k.get("content", ""), k.get("artifact"))
                )

        # Relations (need target entity to exist)
        for rel in (relations or []):
            self._save_relation(user_id, entity_id, name, rel)

        return entity_id

    def _save_relation(self, user_id: str, source_entity_id: str,
                       source_name: str, rel: dict):
        """Save relation, creating target entity if needed."""
        target_name = rel.get("target", "")
        if not target_name:
            return

        with self.conn.cursor() as cur:
            # Ensure target entity exists
            cur.execute(
                """INSERT INTO entities (user_id, name, type)
                   VALUES (%s, %s, 'unknown')
                   ON CONFLICT (user_id, name) DO NOTHING""",
                (user_id, target_name)
            )
            cur.execute(
                "SELECT id FROM entities WHERE user_id = %s AND name = %s",
                (user_id, target_name)
            )
            target_id = str(cur.fetchone()[0])

            direction = rel.get("direction", "outgoing")
            if direction == "outgoing":
                src, tgt = source_entity_id, target_id
            else:
                src, tgt = target_id, source_entity_id

            cur.execute(
                """INSERT INTO relations (source_id, target_id, type, description)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (source_id, target_id, type) DO NOTHING""",
                (src, tgt, rel.get("type", "related_to"), rel.get("description", ""))
            )

    def get_entity(self, user_id: str, name: str) -> Optional[CloudEntity]:
        """Get entity with all data."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT id, name, type FROM entities WHERE user_id = %s AND name = %s",
                (user_id, name)
            )
            row = cur.fetchone()
            if not row:
                return None

            entity_id = str(row["id"])

            # Facts
            cur.execute("SELECT content FROM facts WHERE entity_id = %s", (entity_id,))
            facts = [r["content"] for r in cur.fetchall()]

            # Relations
            cur.execute(
                """SELECT r.type, 'outgoing' as direction, e.name as target, r.description
                   FROM relations r
                   JOIN entities e ON e.id = r.target_id
                   WHERE r.source_id = %s
                   UNION ALL
                   SELECT r.type, 'incoming' as direction, e.name as target, r.description
                   FROM relations r
                   JOIN entities e ON e.id = r.source_id
                   WHERE r.target_id = %s""",
                (entity_id, entity_id)
            )
            relations = [dict(r) for r in cur.fetchall()]

            # Knowledge
            cur.execute(
                "SELECT type, title, content, artifact FROM knowledge WHERE entity_id = %s",
                (entity_id,)
            )
            knowledge = [dict(r) for r in cur.fetchall()]

            return CloudEntity(
                id=entity_id,
                name=row["name"],
                type=row["type"],
                facts=facts,
                relations=relations,
                knowledge=knowledge,
            )

    def get_all_entities(self, user_id: str) -> list[dict]:
        """List all entities with counts."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """SELECT name, type, facts_count, knowledge_count, relations_count
                   FROM entity_overview WHERE user_id = %s
                   ORDER BY updated_at DESC""",
                (user_id,)
            )
            return [dict(r) for r in cur.fetchall()]

    def delete_entity(self, user_id: str, name: str) -> bool:
        """Delete entity and all related data."""
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM entities WHERE user_id = %s AND name = %s RETURNING id",
                (user_id, name)
            )
            return cur.fetchone() is not None

    # ---- Search ----

    def search_vector(self, user_id: str, embedding: list[float],
                      top_k: int = 5, min_score: float = 0.3) -> list[dict]:
        """
        Semantic search using pgvector cosine similarity.
        Returns [{"entity": name, "type": type, "score": float, ...}]
        """
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """SELECT DISTINCT ON (e.name)
                       e.name, e.type,
                       1 - (emb.embedding <=> %s::vector) AS score
                   FROM embeddings emb
                   JOIN entities e ON e.id = emb.entity_id
                   WHERE e.user_id = %s
                     AND 1 - (emb.embedding <=> %s::vector) > %s
                   ORDER BY e.name, score DESC
                   LIMIT %s""",
                (embedding_str, user_id, embedding_str, min_score, top_k)
            )
            results = []
            for row in cur.fetchall():
                entity = self.get_entity(user_id, row["name"])
                if entity:
                    results.append({
                        "entity": entity.name,
                        "type": entity.type,
                        "score": round(float(row["score"]), 3),
                        "facts": entity.facts,
                        "relations": [r for r in entity.relations],
                        "knowledge": [k for k in entity.knowledge],
                    })
            return results

    def search_text(self, user_id: str, query: str, top_k: int = 5) -> list[dict]:
        """Fallback text search (ILIKE)."""
        pattern = f"%{query}%"
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """SELECT DISTINCT e.name, e.type
                   FROM entities e
                   LEFT JOIN facts f ON f.entity_id = e.id
                   LEFT JOIN knowledge k ON k.entity_id = e.id
                   WHERE e.user_id = %s AND (
                       e.name ILIKE %s
                       OR f.content ILIKE %s
                       OR k.content ILIKE %s
                       OR k.title ILIKE %s
                   )
                   LIMIT %s""",
                (user_id, pattern, pattern, pattern, pattern, top_k)
            )
            results = []
            for row in cur.fetchall():
                entity = self.get_entity(user_id, row["name"])
                if entity:
                    results.append({
                        "entity": entity.name,
                        "type": entity.type,
                        "score": 0.5,
                        "facts": entity.facts,
                        "relations": [r for r in entity.relations],
                        "knowledge": [k for k in entity.knowledge],
                    })
            return results

    # ---- Embeddings ----

    def save_embedding(self, entity_id: str, chunk_text: str,
                       embedding: list[float]):
        """Store vector embedding for an entity chunk."""
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO embeddings (entity_id, chunk_text, embedding)
                   VALUES (%s, %s, %s::vector)""",
                (entity_id, chunk_text, embedding_str)
            )

    def delete_embeddings(self, entity_id: str):
        """Remove all embeddings for entity (before reindex)."""
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM embeddings WHERE entity_id = %s",
                (entity_id,)
            )

    # ---- Stats ----

    def get_stats(self, user_id: str) -> dict:
        """User's vault statistics."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT COUNT(*) FROM entities WHERE user_id = %s", (user_id,))
            entities = cur.fetchone()[0]

            cur.execute(
                """SELECT e.type, COUNT(*) as cnt
                   FROM entities e WHERE e.user_id = %s
                   GROUP BY e.type""",
                (user_id,)
            )
            by_type = {r["type"]: r["cnt"] for r in cur.fetchall()}

            cur.execute(
                """SELECT COUNT(*) FROM facts f
                   JOIN entities e ON e.id = f.entity_id
                   WHERE e.user_id = %s""",
                (user_id,)
            )
            facts = cur.fetchone()[0]

            cur.execute(
                """SELECT COUNT(*) FROM knowledge k
                   JOIN entities e ON e.id = k.entity_id
                   WHERE e.user_id = %s""",
                (user_id,)
            )
            knowledge = cur.fetchone()[0]

            cur.execute(
                """SELECT COUNT(*) FROM relations r
                   JOIN entities e ON e.id = r.source_id
                   WHERE e.user_id = %s""",
                (user_id,)
            )
            relations = cur.fetchone()[0]

            cur.execute(
                """SELECT COUNT(*) FROM embeddings emb
                   JOIN entities e ON e.id = emb.entity_id
                   WHERE e.user_id = %s""",
                (user_id,)
            )
            embeddings = cur.fetchone()[0]

            return {
                "entities": entities,
                "by_type": by_type,
                "facts": facts,
                "knowledge": knowledge,
                "relations": relations,
                "embeddings": embeddings,
            }

    # ---- Usage tracking ----

    def log_usage(self, user_id: str, action: str, tokens: int = 0):
        """Log API usage."""
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO usage_log (user_id, action, tokens_used)
                   VALUES (%s, %s, %s)""",
                (user_id, action, tokens)
            )

    # ---- Graph ----

    def get_graph(self, user_id: str) -> dict:
        """Get knowledge graph (nodes + edges) for visualization."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """SELECT name, type, facts_count, knowledge_count
                   FROM entity_overview WHERE user_id = %s""",
                (user_id,)
            )
            nodes = [dict(r) for r in cur.fetchall()]

            cur.execute(
                """SELECT es.name as source, et.name as target, r.type, r.description
                   FROM relations r
                   JOIN entities es ON es.id = r.source_id
                   JOIN entities et ON et.id = r.target_id
                   WHERE es.user_id = %s""",
                (user_id,)
            )
            edges = [dict(r) for r in cur.fetchall()]

            return {"nodes": nodes, "edges": edges}

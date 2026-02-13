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
import json
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
        self._migrate()

    def _migrate(self):
        """Auto-migrate: add new columns if missing."""
        with self.conn.cursor() as cur:
            # facts.created_at for temporal queries
            cur.execute("""
                ALTER TABLE facts ADD COLUMN IF NOT EXISTS created_at 
                TIMESTAMPTZ DEFAULT NOW()
            """)
            # facts.archived for conflict resolution
            cur.execute("""
                ALTER TABLE facts ADD COLUMN IF NOT EXISTS archived 
                BOOLEAN DEFAULT FALSE
            """)
            # facts.superseded_by for tracking what replaced it
            cur.execute("""
                ALTER TABLE facts ADD COLUMN IF NOT EXISTS superseded_by 
                TEXT DEFAULT NULL
            """)

            # --- v1.5 Hybrid search: tsvector on embeddings ---
            cur.execute("""
                ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS tsv tsvector
            """)
            # Populate tsvector for existing rows
            cur.execute("""
                UPDATE embeddings SET tsv = to_tsvector('english', chunk_text)
                WHERE tsv IS NULL AND chunk_text IS NOT NULL
            """)
            # GIN index for fast text search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_tsv 
                ON embeddings USING gin(tsv)
            """)

            # --- v1.5 HNSW index for vector search ---
            # Drop old index if wrong dimensions, recreate
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
                    ON embeddings USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
            except Exception:
                pass  # Index may already exist or dimensions mismatch

        print("✅ Migration complete (v1.5: HNSW + tsvector)", file=sys.stderr)

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
        return self.create_api_key(user_id)

    # ---- OAuth ----

    def save_email_code(self, email: str, code: str):
        """Save email verification code (expires in 10 min)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS email_codes (
                    email TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )"""
            )
            cur.execute(
                """INSERT INTO email_codes (email, code, created_at) 
                   VALUES (%s, %s, NOW())
                   ON CONFLICT (email) DO UPDATE SET code = %s, created_at = NOW()""",
                (email, code, code)
            )

    def verify_email_code(self, email: str, code: str) -> bool:
        """Verify email code (valid for 10 min)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT 1 FROM email_codes 
                   WHERE email = %s AND code = %s 
                   AND created_at > NOW() - INTERVAL '10 minutes'""",
                (email, code)
            )
            if cur.fetchone():
                cur.execute("DELETE FROM email_codes WHERE email = %s", (email,))
                return True
            return False

    def save_oauth_code(self, code: str, user_id: str, redirect_uri: str, state: str):
        """Save OAuth authorization code (expires in 5 min)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS oauth_codes (
                    code TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    redirect_uri TEXT,
                    state TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )"""
            )
            cur.execute(
                """INSERT INTO oauth_codes (code, user_id, redirect_uri, state)
                   VALUES (%s, %s, %s, %s)""",
                (code, user_id, redirect_uri, state)
            )

    def verify_oauth_code(self, code: str) -> Optional[dict]:
        """Verify and consume OAuth code. Returns {user_id, redirect_uri, state} or None."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """SELECT user_id, redirect_uri, state FROM oauth_codes
                   WHERE code = %s AND created_at > NOW() - INTERVAL '5 minutes'""",
                (code,)
            )
            row = cur.fetchone()
            if row:
                cur.execute("DELETE FROM oauth_codes WHERE code = %s", (code,))
                return {"user_id": str(row["user_id"]), "redirect_uri": row["redirect_uri"], "state": row["state"]}
            return None
        return self.create_api_key(user_id, name="reset")

    # ---- Entities ----

    def find_duplicate(self, user_id: str, name: str) -> Optional[tuple]:
        """Find existing entity that matches this name.
        Only matches if: same type context AND one name is a complete word prefix/suffix of the other.
        Returns (entity_id, canonical_name) or None."""
        name_lower = name.strip().lower()
        if not name_lower or len(name_lower) < 3:
            return None

        with self.conn.cursor() as cur:
            # Find entities where one name starts with the other + space
            # e.g. "Ali" matches "Ali Baizhanov" but "Rust" does NOT match "Rustem"
            cur.execute(
                """SELECT id, name, type FROM entities 
                   WHERE user_id = %s AND name != %s
                   AND (
                       LOWER(name) LIKE %s || ' %%'
                       OR %s LIKE LOWER(name) || ' %%'
                       OR LOWER(name) = %s
                   )""",
                (user_id, name, name_lower, name_lower, name_lower)
            )
            matches = cur.fetchall()
            if not matches:
                return None

            # Pick the longest name as canonical
            best = max(matches, key=lambda m: len(m[1]))
            canonical_name = best[1] if len(best[1]) >= len(name) else name
            return (str(best[0]), canonical_name)

    def merge_entities(self, user_id: str, source_id: str, target_id: str,
                       target_name: str):
        """Merge source entity into target. Moves facts, relations, knowledge, embeddings."""
        with self.conn.cursor() as cur:
            # Move facts (skip duplicates)
            cur.execute(
                """INSERT INTO facts (entity_id, content)
                   SELECT %s, content FROM facts WHERE entity_id = %s
                   ON CONFLICT (entity_id, content) DO NOTHING""",
                (target_id, source_id)
            )

            # Move knowledge (skip duplicates)
            cur.execute(
                """INSERT INTO knowledge (entity_id, type, title, content, artifact)
                   SELECT %s, type, title, content, artifact FROM knowledge WHERE entity_id = %s
                   ON CONFLICT (entity_id, title) DO NOTHING""",
                (target_id, source_id)
            )

            # Move relations — update source_id references
            cur.execute(
                """UPDATE relations SET source_id = %s 
                   WHERE source_id = %s 
                   AND NOT EXISTS (
                       SELECT 1 FROM relations r2 
                       WHERE r2.source_id = %s AND r2.target_id = relations.target_id AND r2.type = relations.type
                   )""",
                (target_id, source_id, target_id)
            )
            cur.execute(
                """UPDATE relations SET target_id = %s 
                   WHERE target_id = %s 
                   AND NOT EXISTS (
                       SELECT 1 FROM relations r2 
                       WHERE r2.source_id = relations.source_id AND r2.target_id = %s AND r2.type = relations.type
                   )""",
                (target_id, source_id, target_id)
            )

            # Move embeddings
            cur.execute(
                "UPDATE embeddings SET entity_id = %s WHERE entity_id = %s",
                (target_id, source_id)
            )

            # Delete leftover relations and source entity
            cur.execute("DELETE FROM relations WHERE source_id = %s OR target_id = %s", (source_id, source_id))
            cur.execute("DELETE FROM facts WHERE entity_id = %s", (source_id,))
            cur.execute("DELETE FROM knowledge WHERE entity_id = %s", (source_id,))
            cur.execute("DELETE FROM entities WHERE id = %s", (source_id,))

        print(f"🔀 Merged entity {source_id} into {target_id} ({target_name})", file=sys.stderr)

    def save_entity(self, user_id: str, name: str, type: str,
                    facts: list[str] = None,
                    relations: list[dict] = None,
                    knowledge: list[dict] = None) -> str:
        """
        Create or update entity with facts, relations, knowledge.
        Auto-deduplicates: merges if similar entity exists.
        Returns entity_id.
        """
        # Normalize: if name is ALL CAPS and >3 chars, title-case it
        if name == name.upper() and len(name) > 3 and ' ' not in name:
            name = name.capitalize()

        # Check for case-insensitive exact match first
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, name FROM entities WHERE user_id = %s AND LOWER(name) = LOWER(%s)",
                (user_id, name)
            )
            exact = cur.fetchone()
            if exact:
                entity_id = str(exact[0])
                existing_name = exact[1]
                # Keep the more "normal" casing (not all-caps)
                if existing_name != name:
                    better_name = name if name != name.upper() else existing_name
                    if better_name != existing_name:
                        cur.execute(
                            "UPDATE entities SET name = %s, updated_at = NOW() WHERE id = %s",
                            (better_name, entity_id)
                        )
                else:
                    cur.execute("UPDATE entities SET updated_at = NOW() WHERE id = %s", (entity_id,))

                # Add facts, knowledge, relations below
                self._add_facts_knowledge_relations(entity_id, user_id, name, facts, relations, knowledge)
                return entity_id

        # Check for duplicate entity (word-boundary match)
        duplicate = self.find_duplicate(user_id, name)
        if duplicate:
            existing_id, canonical_name = duplicate
            if len(name) > len(canonical_name):
                canonical_name = name
                with self.conn.cursor() as cur:
                    cur.execute(
                        "UPDATE entities SET name = %s, type = %s, updated_at = NOW() WHERE id = %s",
                        (canonical_name, type, existing_id)
                    )
            else:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "UPDATE entities SET type = %s, updated_at = NOW() WHERE id = %s",
                        (type, existing_id)
                    )
            entity_id = existing_id
            print(f"🔀 Dedup: '{name}' → '{canonical_name}' (id: {entity_id})", file=sys.stderr)
        else:
            with self.conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO entities (user_id, name, type)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (user_id, name) 
                       DO UPDATE SET type = EXCLUDED.type, updated_at = NOW()
                       RETURNING id""",
                    (user_id, name, type)
                )
                entity_id = str(cur.fetchone()[0])

        self._add_facts_knowledge_relations(entity_id, user_id, name, facts, relations, knowledge)
        return entity_id

    def _add_facts_knowledge_relations(self, entity_id: str, user_id: str, name: str,
                                        facts: list[str] = None,
                                        relations: list[dict] = None,
                                        knowledge: list[dict] = None):
        """Add facts, knowledge, and relations to an entity."""
        with self.conn.cursor() as cur:
            for fact in (facts or []):
                cur.execute(
                    """INSERT INTO facts (entity_id, content)
                       VALUES (%s, %s)
                       ON CONFLICT (entity_id, content) DO NOTHING""",
                    (entity_id, fact)
                )
            for k in (knowledge or []):
                cur.execute(
                    """INSERT INTO knowledge (entity_id, type, title, content, artifact)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (entity_id, title) DO NOTHING""",
                    (entity_id, k.get("type", "insight"), k.get("title", ""),
                     k.get("content", ""), k.get("artifact"))
                )

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

    def get_entity_id(self, user_id: str, name: str) -> Optional[str]:
        """Get entity ID by name."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM entities WHERE user_id = %s AND name = %s",
                (user_id, name)
            )
            row = cur.fetchone()
            return str(row[0]) if row else None

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

            # Facts (exclude archived)
            cur.execute("SELECT content FROM facts WHERE entity_id = %s AND archived = FALSE", (entity_id,))
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

    def get_all_entities_full(self, user_id: str) -> list[dict]:
        """Get ALL entities with full facts, relations, knowledge in 4 queries total."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # 1. Get all entities
            cur.execute(
                "SELECT id, name, type FROM entities WHERE user_id = %s ORDER BY updated_at DESC",
                (user_id,)
            )
            entities = cur.fetchall()
            if not entities:
                return []

            entity_ids = [str(e["id"]) for e in entities]
            entity_map = {str(e["id"]): {
                "entity": e["name"],
                "type": e["type"],
                "facts": [],
                "relations": [],
                "knowledge": [],
            } for e in entities}

            # 2. Batch all facts (exclude archived)
            cur.execute(
                "SELECT entity_id, content FROM facts WHERE entity_id = ANY(%s::uuid[]) AND archived = FALSE",
                (entity_ids,)
            )
            for row in cur.fetchall():
                eid = str(row["entity_id"])
                if eid in entity_map:
                    entity_map[eid]["facts"].append(row["content"])

            # 3. Batch all relations
            cur.execute(
                """SELECT r.source_id, r.target_id, r.type, r.description,
                          se.name as source_name, te.name as target_name
                   FROM relations r
                   JOIN entities se ON se.id = r.source_id
                   JOIN entities te ON te.id = r.target_id
                   WHERE r.source_id = ANY(%s::uuid[]) OR r.target_id = ANY(%s::uuid[])""",
                (entity_ids, entity_ids)
            )
            for row in cur.fetchall():
                src_id = str(row["source_id"])
                tgt_id = str(row["target_id"])
                rel = {"type": row["type"], "detail": row["description"] or ""}
                if src_id in entity_map:
                    entity_map[src_id]["relations"].append(
                        {**rel, "direction": "outgoing", "target": row["target_name"]})
                if tgt_id in entity_map and tgt_id != src_id:
                    entity_map[tgt_id]["relations"].append(
                        {**rel, "direction": "incoming", "target": row["source_name"]})

            # 4. Batch all knowledge
            cur.execute(
                "SELECT entity_id, type, title, content, artifact FROM knowledge WHERE entity_id = ANY(%s::uuid[])",
                (entity_ids,)
            )
            for row in cur.fetchall():
                eid = str(row["entity_id"])
                if eid in entity_map:
                    entity_map[eid]["knowledge"].append({
                        "type": row["type"],
                        "title": row["title"],
                        "content": row["content"],
                        "artifact": row["artifact"],
                    })

            return [entity_map[str(e["id"])] for e in entities]

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
                      top_k: int = 5, min_score: float = 0.3,
                      query_text: str = "") -> list[dict]:
        """
        Hybrid search: vector + BM25 text + graph expansion.
        
        Pipeline:
        1. Vector search (semantic similarity via pgvector)
        2. BM25 text search (exact keyword match via tsvector)
        3. Reciprocal Rank Fusion to merge results
        4. Graph expansion: follow relations to find connected entities
        5. Recency boost + dedup + limit
        """
        import datetime
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:

            # ========== STAGE 1: Vector search ==========
            cur.execute(
                """SELECT DISTINCT ON (e.id)
                       e.id, e.name, e.type,
                       1 - (emb.embedding <=> %s::vector) AS score,
                       e.updated_at
                   FROM embeddings emb
                   JOIN entities e ON e.id = emb.entity_id
                   WHERE e.user_id = %s
                     AND 1 - (emb.embedding <=> %s::vector) > %s
                   ORDER BY e.id, score DESC""",
                (embedding_str, user_id, embedding_str, min_score)
            )
            vector_rows = cur.fetchall()
            # Rank by score
            vector_rows.sort(key=lambda r: float(r["score"]), reverse=True)
            vector_ranked = {str(r["id"]): (i + 1, r) for i, r in enumerate(vector_rows[:20])}

            # ========== STAGE 2: BM25 text search ==========
            bm25_ranked = {}
            if query_text:
                # Build tsquery: split words, join with &
                words = [w.strip() for w in query_text.split() if len(w.strip()) >= 2]
                if words:
                    # Use plainto_tsquery for robustness (handles any language)
                    cur.execute(
                        """SELECT DISTINCT ON (e.id)
                               e.id, e.name, e.type,
                               ts_rank(emb.tsv, plainto_tsquery('english', %s)) AS rank,
                               e.updated_at
                           FROM embeddings emb
                           JOIN entities e ON e.id = emb.entity_id
                           WHERE e.user_id = %s
                             AND emb.tsv @@ plainto_tsquery('english', %s)
                           ORDER BY e.id, rank DESC""",
                        (query_text, user_id, query_text)
                    )
                    bm25_rows = cur.fetchall()
                    bm25_rows.sort(key=lambda r: float(r["rank"]), reverse=True)
                    bm25_ranked = {str(r["id"]): (i + 1, r) for i, r in enumerate(bm25_rows[:20])}

                    # Also search entity names directly (ILIKE)
                    cur.execute(
                        """SELECT id, name, type, updated_at
                           FROM entities
                           WHERE user_id = %s AND (
                               name ILIKE %s OR name ILIKE %s
                           )""",
                        (user_id, f"%{query_text}%", f"%{'%'.join(words)}%")
                    )
                    for i, row in enumerate(cur.fetchall()):
                        eid = str(row["id"])
                        if eid not in bm25_ranked:
                            bm25_ranked[eid] = (i + 1, row)

            # ========== STAGE 3: Reciprocal Rank Fusion ==========
            k = 60  # RRF constant
            all_entity_ids = set(vector_ranked.keys()) | set(bm25_ranked.keys())
            
            rrf_scores = {}
            entity_info = {}  # id -> (name, type, updated_at)
            
            for eid in all_entity_ids:
                score = 0.0
                if eid in vector_ranked:
                    rank, row = vector_ranked[eid]
                    score += 1.0 / (k + rank)
                    entity_info[eid] = (row["name"], row["type"], row.get("updated_at"))
                if eid in bm25_ranked:
                    rank, row = bm25_ranked[eid]
                    score += 1.0 / (k + rank)
                    if eid not in entity_info:
                        entity_info[eid] = (row["name"], row["type"], row.get("updated_at"))
                rrf_scores[eid] = score

            # ========== STAGE 4: Graph expansion ==========
            # Take top entities from RRF, then expand via relations
            sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            seed_ids = [eid for eid, _ in sorted_rrf[:8]]

            graph_entities = {}
            if seed_ids:
                cur.execute(
                    """SELECT 
                           CASE WHEN r.source_id = ANY(%s::uuid[]) THEN r.target_id ELSE r.source_id END AS related_id,
                           CASE WHEN r.source_id = ANY(%s::uuid[]) THEN te.name ELSE se.name END AS related_name,
                           CASE WHEN r.source_id = ANY(%s::uuid[]) THEN te.type ELSE se.type END AS related_type,
                           CASE WHEN r.source_id = ANY(%s::uuid[]) THEN te.updated_at ELSE se.updated_at END AS related_updated,
                           r.type AS rel_type
                       FROM relations r
                       JOIN entities se ON se.id = r.source_id
                       JOIN entities te ON te.id = r.target_id
                       WHERE (r.source_id = ANY(%s::uuid[]) OR r.target_id = ANY(%s::uuid[]))
                         AND se.user_id = %s""",
                    (seed_ids, seed_ids, seed_ids, seed_ids, seed_ids, seed_ids, user_id)
                )
                for row in cur.fetchall():
                    rid = str(row["related_id"])
                    if rid not in rrf_scores and rid not in graph_entities:
                        # Graph-expanded entity gets a lower base score
                        graph_entities[rid] = {
                            "name": row["related_name"],
                            "type": row["related_type"],
                            "updated_at": row["related_updated"],
                            "via_relation": row["rel_type"],
                        }

            # Add graph entities with discounted score
            max_rrf = max(rrf_scores.values()) if rrf_scores else 0.01
            for eid, info in graph_entities.items():
                rrf_scores[eid] = max_rrf * 0.5  # 50% of best direct match
                entity_info[eid] = (info["name"], info["type"], info.get("updated_at"))

            # ========== STAGE 5: Recency boost + build results ==========
            now = datetime.datetime.now(datetime.timezone.utc)
            final_scores = {}
            for eid, base_score in rrf_scores.items():
                score = base_score
                if eid in entity_info:
                    updated_at = entity_info[eid][2]
                    if updated_at:
                        try:
                            age_days = (now - updated_at.replace(tzinfo=datetime.timezone.utc)).days
                            if age_days <= 7:
                                score *= 1.15  # 15% boost for recent
                            elif age_days <= 30:
                                score *= 1.05  # 5% boost
                        except Exception:
                            pass
                final_scores[eid] = score

            # Sort and limit
            sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            top_entities = sorted_final[:top_k]

            if not top_entities:
                return []

            # Touch accessed entities
            accessed_ids = [eid for eid, _ in top_entities]
            cur.execute(
                "UPDATE entities SET updated_at = NOW() WHERE id = ANY(%s::uuid[])",
                (accessed_ids,)
            )

            # ========== STAGE 6: Batch load details ==========
            entity_ids = [eid for eid, _ in top_entities]
            entity_map = {}
            for eid, score in top_entities:
                name, etype, _ = entity_info.get(eid, ("?", "?", None))
                entity_map[eid] = {
                    "entity": name,
                    "type": etype,
                    "score": round(score, 4),
                    "facts": [],
                    "relations": [],
                    "knowledge": [],
                }

            # Batch facts (exclude archived)
            cur.execute(
                "SELECT entity_id, content FROM facts WHERE entity_id = ANY(%s::uuid[]) AND archived = FALSE",
                (entity_ids,)
            )
            for row in cur.fetchall():
                eid = str(row["entity_id"])
                if eid in entity_map:
                    entity_map[eid]["facts"].append(row["content"])

            # Batch relations
            cur.execute(
                """SELECT r.source_id, r.target_id, r.type, r.description,
                          se.name as source_name, te.name as target_name
                   FROM relations r
                   JOIN entities se ON se.id = r.source_id
                   JOIN entities te ON te.id = r.target_id
                   WHERE r.source_id = ANY(%s::uuid[]) OR r.target_id = ANY(%s::uuid[])""",
                (entity_ids, entity_ids)
            )
            for row in cur.fetchall():
                src_id = str(row["source_id"])
                tgt_id = str(row["target_id"])
                rel = {
                    "type": row["type"],
                    "detail": row["description"] or "",
                }
                if src_id in entity_map:
                    rel_out = {**rel, "direction": "outgoing", "target": row["target_name"]}
                    entity_map[src_id]["relations"].append(rel_out)
                if tgt_id in entity_map and tgt_id != src_id:
                    rel_in = {**rel, "direction": "incoming", "target": row["source_name"]}
                    entity_map[tgt_id]["relations"].append(rel_in)

            # Batch knowledge
            cur.execute(
                "SELECT entity_id, type, title, content, artifact FROM knowledge WHERE entity_id = ANY(%s::uuid[])",
                (entity_ids,)
            )
            for row in cur.fetchall():
                eid = str(row["entity_id"])
                if eid in entity_map:
                    entity_map[eid]["knowledge"].append({
                        "type": row["type"],
                        "title": row["title"],
                        "content": row["content"],
                        "artifact": row["artifact"],
                    })

            # Return in score order
            return [entity_map[eid] for eid, _ in top_entities if eid in entity_map]

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

    def search_temporal(self, user_id: str, after: str = None, before: str = None,
                        top_k: int = 20) -> list[dict]:
        """Search facts by time range. Returns entities with facts created in the window."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            conditions = ["e.user_id = %s", "f.archived = FALSE"]
            params = [user_id]

            if after:
                conditions.append("f.created_at >= %s")
                params.append(after)
            if before:
                conditions.append("f.created_at <= %s")
                params.append(before)

            where = " AND ".join(conditions)
            cur.execute(
                f"""SELECT e.name, e.type, f.content, f.created_at
                    FROM facts f
                    JOIN entities e ON e.id = f.entity_id
                    WHERE {where}
                    ORDER BY f.created_at DESC
                    LIMIT %s""",
                (*params, top_k)
            )

            # Group by entity
            entity_map = {}
            for row in cur.fetchall():
                name = row["name"]
                if name not in entity_map:
                    entity_map[name] = {
                        "entity": name,
                        "type": row["type"],
                        "facts": [],
                    }
                entity_map[name]["facts"].append({
                    "content": row["content"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                })

            return list(entity_map.values())

    def archive_contradicted_facts(self, entity_id: str, new_facts: list[str],
                                    llm_client) -> list[str]:
        """Use LLM to find old facts contradicted by new ones. Archive them.
        Returns list of archived fact contents."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT content FROM facts WHERE entity_id = %s AND archived = FALSE",
                (entity_id,)
            )
            old_facts = [r["content"] for r in cur.fetchall()]

        if not old_facts or not new_facts:
            return []

        # Ask LLM which old facts are contradicted
        prompt = f"""Given these EXISTING facts about an entity:
{json.dumps(old_facts)}

And these NEW facts being added:
{json.dumps(new_facts)}

Which existing facts should be REMOVED? Remove facts that are:
1. CONTRADICTED by new facts (e.g. old: "lives in Almaty" vs new: "relocated to Dubai")
2. OBVIOUSLY WRONG given the new context (e.g. "is a programming language" on a person entity that has fact "is a colleague")
3. DUPLICATED or redundant with new facts

Return ONLY a JSON array of the old fact strings to remove, or empty array [] if none.
No markdown, no explanation."""

        try:
            response = llm_client.complete(prompt)
            clean = response.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1])
            contradicted = json.loads(clean)
            if not isinstance(contradicted, list):
                return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Conflict resolution failed: {e}", file=sys.stderr)
            return []

        # Archive contradicted facts
        archived = []
        for old_fact in contradicted:
            if old_fact in old_facts:
                with self.conn.cursor() as cur:
                    # Find matching new fact for superseded_by
                    superseded_by = new_facts[0] if new_facts else None
                    cur.execute(
                        """UPDATE facts SET archived = TRUE, superseded_by = %s
                           WHERE entity_id = %s AND content = %s AND archived = FALSE""",
                        (superseded_by, entity_id, old_fact)
                    )
                archived.append(old_fact)
                print(f"📦 Archived: '{old_fact}' → superseded by '{superseded_by}'", file=sys.stderr)

        return archived

    # ---- Embeddings ----

    def save_embedding(self, entity_id: str, chunk_text: str,
                       embedding: list[float]):
        """Store vector embedding for an entity chunk."""
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO embeddings (entity_id, chunk_text, embedding, tsv)
                   VALUES (%s, %s, %s::vector, to_tsvector('english', %s))""",
                (entity_id, chunk_text, embedding_str, chunk_text)
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

"""
Mengram Cloud Storage — PostgreSQL backend.

Replaces VaultManager (local .md files) with PostgreSQL + pgvector.
Same interface, different storage.

Usage:
    store = CloudStore(database_url="postgresql://...")
    store.save_entity("PostgreSQL", "technology", facts=[...], relations=[...], knowledge=[...])
    results = store.search("database pool", user_id="...", top_k=5)
"""

import datetime
import hashlib
import json
import math
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

        # --- v1.6 Importance scoring ---
        with self.conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE facts ADD COLUMN IF NOT EXISTS importance 
                FLOAT DEFAULT 0.5
            """)
            cur.execute("""
                ALTER TABLE facts ADD COLUMN IF NOT EXISTS access_count 
                INTEGER DEFAULT 0
            """)
            cur.execute("""
                ALTER TABLE facts ADD COLUMN IF NOT EXISTS last_accessed 
                TIMESTAMPTZ DEFAULT NULL
            """)
        print("✅ Migration complete (v1.6: importance scoring)", file=sys.stderr)

        # --- v1.7 Reflection system ---
        with self.conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS scope 
                VARCHAR(20) DEFAULT 'insight'
            """)
            cur.execute("""
                ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS confidence 
                FLOAT DEFAULT 1.0
            """)
            cur.execute("""
                ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS based_on_facts 
                TEXT[] DEFAULT '{}'
            """)
            cur.execute("""
                ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS refreshed_at 
                TIMESTAMPTZ DEFAULT NOW()
            """)
            cur.execute("""
                ALTER TABLE knowledge ADD COLUMN IF NOT EXISTS user_id 
                VARCHAR(255) DEFAULT NULL
            """)
            # Index for efficient reflection queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_scope 
                ON knowledge (scope) WHERE scope IN ('entity', 'cross', 'temporal')
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_user 
                ON knowledge (user_id) WHERE user_id IS NOT NULL
            """)
        print("✅ Migration complete (v1.7: reflection system)", file=sys.stderr)

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

    def _find_primary_person(self, user_id: str) -> Optional[tuple]:
        """Find the primary person entity for this user.
        Prefers: full name (has space) > most facts > most recent."""
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT e.id, e.name, COUNT(f.id) as fact_count,
                          CASE WHEN e.name LIKE '%% %%' THEN 1 ELSE 0 END as has_full_name
                   FROM entities e
                   LEFT JOIN facts f ON f.entity_id = e.id AND f.archived = FALSE
                   WHERE e.user_id = %s AND e.type = 'person' AND LOWER(e.name) != 'user'
                   GROUP BY e.id, e.name
                   ORDER BY has_full_name DESC, fact_count DESC, e.updated_at DESC
                   LIMIT 1""",
                (user_id,)
            )
            row = cur.fetchone()
            if row:
                return (str(row[0]), row[1])
            return None

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

        # ---- "User" resolution: merge into primary person entity ----
        if name.lower() == "user" and type == "person":
            primary = self._find_primary_person(user_id)
            if primary:
                entity_id, canonical_name = primary
                print(f"🔀 User → '{canonical_name}' (id: {entity_id})", file=sys.stderr)
                with self.conn.cursor() as cur:
                    cur.execute("UPDATE entities SET updated_at = NOW() WHERE id = %s", (entity_id,))
                self._add_facts_knowledge_relations(entity_id, user_id, canonical_name, facts, relations, knowledge)
                return entity_id

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

    @staticmethod
    def estimate_importance(fact: str) -> float:
        """Estimate fact importance 0.0-1.0 based on content patterns."""
        f = fact.lower().strip()

        # Identity / role — highest
        if any(p in f for p in [
            'is a ', 'works as', 'works at', 'ceo of', 'founder of',
            'created by', 'built by', 'lives in', 'born in', 'age ',
            'studies at', 'graduated from', 'native language',
            'citizenship', 'nationality'
        ]):
            return 0.9

        # Skills / tech stack — high
        if any(p in f for p in [
            'uses ', 'primary language', 'tech stack', 'proficient in',
            'expert in', 'main database', 'built with', 'powered by',
            'written in', 'developed in', 'architecture'
        ]):
            return 0.8

        # Long-term preferences — medium-high
        if any(p in f for p in [
            'prefers ', 'always ', 'never ', 'favorite', 'hates',
            'allergic', 'dietary', 'philosophy', 'likes ', 'loves ',
            'enjoys ', 'dislikes ', 'avoids '
        ]):
            return 0.7

        # Goals / plans — medium
        if any(p in f for p in [
            'wants to', 'plans to', 'goal', 'learning', 'interested in',
            'considering', 'thinking about', 'exploring'
        ]):
            return 0.6

        # Current state — medium-low
        if any(p in f for p in [
            'currently', 'right now', 'working on', 'building',
            'deployed', 'version', 'released'
        ]):
            return 0.5

        # Default
        return 0.5

    def _add_facts_knowledge_relations(self, entity_id: str, user_id: str, name: str,
                                        facts: list[str] = None,
                                        relations: list[dict] = None,
                                        knowledge: list[dict] = None):
        """Add facts, knowledge, and relations to an entity."""
        with self.conn.cursor() as cur:
            for fact in (facts or []):
                importance = self.estimate_importance(fact)
                cur.execute(
                    """INSERT INTO facts (entity_id, content, importance)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (entity_id, content) DO NOTHING""",
                    (entity_id, fact, importance)
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
        """List all entities with counts (excludes internal entities)."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """SELECT name, type, facts_count, knowledge_count, relations_count
                   FROM entity_overview WHERE user_id = %s AND name NOT LIKE '\\_%%'
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

    def get_existing_context(self, user_id: str, max_entities: int = 20, max_facts_per: int = 5) -> str:
        """Get compact summary of existing entities for extraction context.
        Resolves 'User' to primary person name.
        Returns a string like:
          The user's name is Ali Baizhanov. Always use this name instead of "User".
          - Ali Baizhanov (person): works as developer, uses Python, lives in Almaty
          - Mengram (project): AI memory protocol, built with FastAPI
        """
        # Find primary person name
        primary = self._find_primary_person(user_id)
        primary_name = primary[1] if primary else None

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Get top entities by recent activity
            cur.execute(
                """SELECT e.id, e.name, e.type 
                   FROM entities e
                   WHERE e.user_id = %s
                   ORDER BY e.updated_at DESC NULLS LAST
                   LIMIT %s""",
                (user_id, max_entities)
            )
            entities = cur.fetchall()
            if not entities:
                if primary_name:
                    return f'The user\'s name is "{primary_name}". Always use this name instead of "User".'
                return ""

            entity_ids = [str(e["id"]) for e in entities]

            # Get top facts per entity (by importance)
            cur.execute(
                """SELECT DISTINCT ON (entity_id, content) entity_id, content, importance
                   FROM facts 
                   WHERE entity_id = ANY(%s::uuid[]) AND archived = FALSE
                   ORDER BY entity_id, content, importance DESC""",
                (entity_ids,)
            )
            facts_by_entity = {}
            for row in cur.fetchall():
                eid = str(row["entity_id"])
                if eid not in facts_by_entity:
                    facts_by_entity[eid] = []
                facts_by_entity[eid].append((row["content"], float(row["importance"] or 0.5)))

            # Sort each entity's facts by importance, take top N
            for eid in facts_by_entity:
                facts_by_entity[eid].sort(key=lambda x: x[1], reverse=True)
                facts_by_entity[eid] = facts_by_entity[eid][:max_facts_per]

            lines = []
            # Add name hint if known
            if primary_name:
                lines.append(f'The user\'s name is "{primary_name}". Always use "{primary_name}" instead of "User".')

            for e in entities:
                eid = str(e["id"])
                name = e["name"]
                # Skip "User" and "_reflections" from context
                if name.lower() in ("user", "_reflections"):
                    continue
                facts = facts_by_entity.get(eid, [])
                if facts:
                    fact_strs = ", ".join(f[0] for f in facts)
                    lines.append(f"- {name} ({e['type']}): {fact_strs}")
                else:
                    lines.append(f"- {name} ({e['type']})")

            # Add top reflections for richer context
            reflections = self.get_reflections(user_id)
            if reflections:
                top_refs = [r for r in reflections if r["confidence"] >= 0.7][:3]
                if top_refs:
                    lines.append("\nAI-generated insights (use for context, don't re-extract):")
                    for r in top_refs:
                        lines.append(f"  [{r['scope']}] {r['content'][:200]}")

            return "\n".join(lines)

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

            # Batch facts (exclude archived) — sorted by importance
            cur.execute(
                """SELECT id, entity_id, content, importance, access_count, last_accessed
                   FROM facts WHERE entity_id = ANY(%s::uuid[]) AND archived = FALSE
                   ORDER BY importance DESC""",
                (entity_ids,)
            )
            fact_ids_accessed = []
            for row in cur.fetchall():
                eid = str(row["entity_id"])
                if eid in entity_map:
                    # Apply Ebbinghaus decay: importance * e^(-0.03 * days_since_access)
                    base_imp = float(row["importance"] or 0.5)
                    if row["last_accessed"]:
                        try:
                            days_since = (now - row["last_accessed"].replace(
                                tzinfo=datetime.timezone.utc)).days
                            decay = math.exp(-0.03 * days_since)
                        except Exception:
                            decay = 1.0
                    else:
                        decay = 0.8  # never accessed = slight penalty
                    # Access frequency boost: log(1 + access_count) * 0.05
                    access_boost = math.log1p(row["access_count"] or 0) * 0.05
                    effective_imp = min(base_imp * decay + access_boost, 1.0)

                    entity_map[eid]["facts"].append({
                        "content": row["content"],
                        "importance": round(effective_imp, 3)
                    })
                    fact_ids_accessed.append(str(row["id"]))

            # Sort facts by effective importance, return as strings
            for eid in entity_map:
                sorted_facts = sorted(
                    entity_map[eid]["facts"],
                    key=lambda f: f["importance"], reverse=True
                )
                entity_map[eid]["facts"] = [f["content"] for f in sorted_facts]

            # Track fact access — update access_count and last_accessed
            if fact_ids_accessed:
                cur.execute(
                    """UPDATE facts 
                       SET access_count = access_count + 1, last_accessed = NOW()
                       WHERE id = ANY(%s::uuid[])""",
                    (fact_ids_accessed,)
                )

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

    def dedup_entity_facts(self, entity_id: str, entity_name: str, llm_client) -> dict:
        """Use LLM to deduplicate facts on an entity. 
        Groups similar facts, keeps the best one, archives the rest.
        Returns {kept: [...], archived: [...]}"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT content FROM facts WHERE entity_id = %s AND archived = FALSE ORDER BY importance DESC, created_at DESC",
                (entity_id,)
            )
            facts = [r["content"] for r in cur.fetchall()]

        if len(facts) < 3:
            return {"kept": facts, "archived": []}

        prompt = f"""You are a fact deduplication system.

Entity: "{entity_name}"
Facts:
{json.dumps(facts, ensure_ascii=False)}

Many of these facts say the SAME thing in different words. Your job:
1. Group duplicate/redundant facts together
2. For each group, pick the SINGLE BEST version (most concise, accurate, normalized)
3. Return JSON with facts to KEEP and facts to ARCHIVE

Rules for picking the best:
- Shorter and more specific beats longer and vague
- "specializes in Java/Spring Boot" beats "specializes in Java" + "specializes in Spring Boot" (combined is better)
- If one fact is strictly more informative, keep that one
- "works in Almaty, Kazakhstan" beats "works in Almaty" (more context)
- Remove truly obsolete facts only if a newer one clearly replaces it

Return ONLY this JSON (no markdown):
{{
  "keep": ["fact1", "fact2", ...],
  "archive": ["redundant1", "redundant2", ...]
}}"""

        try:
            response = llm_client.complete(prompt)
            clean = response.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1])
            result = json.loads(clean)
            if not isinstance(result, dict) or "archive" not in result:
                return {"kept": facts, "archived": []}
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Dedup failed: {e}", file=sys.stderr)
            return {"kept": facts, "archived": []}

        archived = []
        to_archive = result.get("archive", [])
        for fact in to_archive:
            if fact in facts:
                with self.conn.cursor() as cur:
                    cur.execute(
                        """UPDATE facts SET archived = TRUE, superseded_by = 'dedup'
                           WHERE entity_id = %s AND content = %s AND archived = FALSE""",
                        (entity_id, fact)
                    )
                    if cur.rowcount > 0:
                        archived.append(fact)

        kept = result.get("keep", [])
        print(f"🧹 Dedup '{entity_name}': {len(facts)} → {len(facts)-len(archived)} facts ({len(archived)} archived)", file=sys.stderr)
        return {"kept": kept, "archived": archived}

    # ---- Reflection Engine ----

    REFLECTION_PROMPT = """You are a cognitive memory system that synthesizes insights from raw facts.

ENTITIES AND FACTS:
{facts_text}

EXISTING REFLECTIONS (update if stale):
{prev_reflections}

Generate reflections in 3 categories:

1. ENTITY REFLECTIONS — for entities with 3+ facts, write a 2-3 sentence summary.
   Focus: what/who it is, relation to the user, current status.

2. CROSS-ENTITY PATTERNS — patterns across multiple entities.
   Focus: career direction, tech preferences, behavioral patterns, relationships.

3. TEMPORAL — what changed recently based on fact timestamps.
   Focus: new interests, shifting priorities, recent activity.

Rate confidence 0.0-1.0 based on how well-supported by facts.

Return ONLY JSON (no markdown):
{{
  "entity_reflections": [
    {{"entity": "EntityName", "title": "short title", "reflection": "2-3 sentences", "confidence": 0.9, "key_facts": ["fact1", "fact2"]}}
  ],
  "cross_entity": [
    {{"entities": ["E1", "E2"], "title": "short title", "reflection": "2-3 sentences", "confidence": 0.85}}
  ],
  "temporal": [
    {{"period": "recent", "title": "short title", "reflection": "2-3 sentences", "confidence": 0.8}}
  ]
}}"""

    def get_reflection_stats(self, user_id: str) -> dict:
        """Get stats to decide if reflection is needed."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Count new facts since last reflection
            cur.execute(
                """SELECT MAX(refreshed_at) as last_reflection
                   FROM knowledge 
                   WHERE user_id = %s AND scope IN ('entity', 'cross', 'temporal')""",
                (user_id,)
            )
            row = cur.fetchone()
            last_reflection = row["last_reflection"] if row and row["last_reflection"] else None

            if last_reflection:
                cur.execute(
                    """SELECT COUNT(*) as cnt FROM facts f
                       JOIN entities e ON e.id = f.entity_id
                       WHERE e.user_id = %s AND f.archived = FALSE 
                       AND f.created_at > %s""",
                    (user_id, last_reflection)
                )
                new_facts = cur.fetchone()["cnt"]
                hours_since = (datetime.datetime.now(datetime.timezone.utc) - 
                              last_reflection.replace(tzinfo=datetime.timezone.utc)).total_seconds() / 3600
            else:
                # Never reflected — count all facts
                cur.execute(
                    """SELECT COUNT(*) as cnt FROM facts f
                       JOIN entities e ON e.id = f.entity_id
                       WHERE e.user_id = %s AND f.archived = FALSE""",
                    (user_id,)
                )
                new_facts = cur.fetchone()["cnt"]
                hours_since = 999

            return {
                "new_facts_since_last": new_facts,
                "hours_since_last": round(hours_since, 1),
                "last_reflection": last_reflection.isoformat() if last_reflection else None,
            }

    def should_reflect(self, user_id: str) -> bool:
        """Check if reflection is needed based on triggers."""
        stats = self.get_reflection_stats(user_id)
        # Trigger 1: 10+ new facts since last reflection
        if stats["new_facts_since_last"] >= 10:
            return True
        # Trigger 2: 24h+ since last reflection AND has new facts
        if stats["hours_since_last"] >= 24 and stats["new_facts_since_last"] >= 3:
            return True
        return False

    def generate_reflections(self, user_id: str, llm_client) -> dict:
        """Generate all 3 types of reflections using LLM."""
        # Gather facts grouped by entity
        entities = self.get_all_entities_full(user_id)
        if not entities:
            return {"entity_reflections": [], "cross_entity": [], "temporal": []}

        # Build facts text
        facts_lines = []
        for e in entities:
            if not e["facts"]:
                continue
            facts_str = ", ".join(e["facts"][:15])  # cap at 15 per entity
            facts_lines.append(f"- {e['entity']} ({e['type']}): {facts_str}")
        facts_text = "\n".join(facts_lines)

        # Get previous reflections
        prev = self.get_reflections(user_id)
        prev_text = ""
        if prev:
            prev_lines = []
            for r in prev[:10]:
                prev_lines.append(f"- [{r['scope']}] {r['title']}: {r['content'][:200]}")
            prev_text = "\n".join(prev_lines)
        if not prev_text:
            prev_text = "(none yet)"

        prompt = self.REFLECTION_PROMPT.format(
            facts_text=facts_text,
            prev_reflections=prev_text
        )

        try:
            response = llm_client.complete(prompt)
            clean = response.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1])
            result = json.loads(clean)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Reflection generation failed: {e}", file=sys.stderr)
            return {"entity_reflections": [], "cross_entity": [], "temporal": []}

        # Save reflections
        saved = {"entity_reflections": 0, "cross_entity": 0, "temporal": 0}

        for r in result.get("entity_reflections", []):
            entity_name = r.get("entity", "")
            entity_id = self.get_entity_id(user_id, entity_name) if entity_name else None
            self._save_reflection(
                user_id=user_id,
                entity_id=entity_id,
                scope="entity",
                title=r.get("title", f"{entity_name} profile"),
                content=r.get("reflection", ""),
                confidence=r.get("confidence", 0.8),
                based_on=r.get("key_facts", [])
            )
            saved["entity_reflections"] += 1

        for r in result.get("cross_entity", []):
            self._save_reflection(
                user_id=user_id,
                entity_id=None,
                scope="cross",
                title=r.get("title", "Cross-entity pattern"),
                content=r.get("reflection", ""),
                confidence=r.get("confidence", 0.8),
                based_on=[]
            )
            saved["cross_entity"] += 1

        for r in result.get("temporal", []):
            self._save_reflection(
                user_id=user_id,
                entity_id=None,
                scope="temporal",
                title=r.get("title", "Recent changes"),
                content=r.get("reflection", ""),
                confidence=r.get("confidence", 0.8),
                based_on=[]
            )
            saved["temporal"] += 1

        print(f"🧠 Reflections generated for {user_id}: {saved}", file=sys.stderr)
        return result

    def _save_reflection(self, user_id: str, entity_id: Optional[str],
                         scope: str, title: str, content: str,
                         confidence: float = 0.8, based_on: list = None):
        """Save or update a reflection."""
        with self.conn.cursor() as cur:
            if entity_id:
                # Entity reflection — upsert by entity + scope
                cur.execute(
                    """INSERT INTO knowledge (entity_id, user_id, type, title, content, scope, confidence, based_on_facts, refreshed_at)
                       VALUES (%s, %s, 'reflection', %s, %s, %s, %s, %s, NOW())
                       ON CONFLICT (entity_id, title) 
                       DO UPDATE SET content = EXCLUDED.content, 
                                     confidence = EXCLUDED.confidence,
                                     based_on_facts = EXCLUDED.based_on_facts,
                                     refreshed_at = NOW()""",
                    (entity_id, user_id, title, content, scope, confidence, based_on or [])
                )
            else:
                # Cross/temporal reflection — need a "global" entity
                global_id = self._get_or_create_global_entity(user_id)
                cur.execute(
                    """INSERT INTO knowledge (entity_id, user_id, type, title, content, scope, confidence, based_on_facts, refreshed_at)
                       VALUES (%s, %s, 'reflection', %s, %s, %s, %s, %s, NOW())
                       ON CONFLICT (entity_id, title)
                       DO UPDATE SET content = EXCLUDED.content,
                                     confidence = EXCLUDED.confidence,
                                     based_on_facts = EXCLUDED.based_on_facts,
                                     refreshed_at = NOW()""",
                    (global_id, user_id, title, content, scope, confidence, based_on or [])
                )

    def _get_or_create_global_entity(self, user_id: str) -> str:
        """Get or create a special _reflections entity for cross/temporal reflections."""
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO entities (user_id, name, type)
                   VALUES (%s, '_reflections', 'concept')
                   ON CONFLICT (user_id, name) DO UPDATE SET updated_at = NOW()
                   RETURNING id""",
                (user_id,)
            )
            return str(cur.fetchone()[0])

    def get_reflections(self, user_id: str, scope: str = None) -> list[dict]:
        """Get all reflections for a user, optionally filtered by scope."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if scope:
                cur.execute(
                    """SELECT k.title, k.content, k.scope, k.confidence, k.refreshed_at,
                              e.name as entity_name
                       FROM knowledge k
                       JOIN entities e ON e.id = k.entity_id
                       WHERE k.user_id = %s AND k.scope = %s AND k.type = 'reflection'
                       ORDER BY k.confidence DESC, k.refreshed_at DESC""",
                    (user_id, scope)
                )
            else:
                cur.execute(
                    """SELECT k.title, k.content, k.scope, k.confidence, k.refreshed_at,
                              e.name as entity_name
                       FROM knowledge k
                       JOIN entities e ON e.id = k.entity_id
                       WHERE k.user_id = %s AND k.type = 'reflection'
                       ORDER BY k.scope, k.confidence DESC, k.refreshed_at DESC""",
                    (user_id,)
                )
            return [{
                "title": r["title"],
                "content": r["content"],
                "scope": r["scope"],
                "confidence": float(r["confidence"] or 0.8),
                "entity": r["entity_name"],
                "refreshed_at": r["refreshed_at"].isoformat() if r["refreshed_at"] else None,
            } for r in cur.fetchall()]

    def get_insights(self, user_id: str) -> dict:
        """Get formatted insights for dashboard — profile, weekly, network, patterns."""
        reflections = self.get_reflections(user_id)
        if not reflections:
            return {"has_insights": False, "profile": None, "weekly": None, "network": None, "patterns": None}

        profile = next((r for r in reflections if r["scope"] == "entity" and "profile" in r["title"].lower()), None)
        # Fallback: first entity reflection for primary person
        if not profile:
            primary = self._find_primary_person(user_id)
            if primary:
                profile = next((r for r in reflections if r["scope"] == "entity" and r["entity"] == primary[1]), None)

        weekly = next((r for r in reflections if r["scope"] == "temporal"), None)
        network = next((r for r in reflections if r["scope"] == "cross" and 
                        any(w in r["title"].lower() for w in ["network", "colleague", "team"])), None)
        patterns = next((r for r in reflections if r["scope"] == "cross" and r != network), None)

        return {
            "has_insights": True,
            "profile": profile,
            "weekly": weekly,
            "network": network,
            "patterns": patterns,
            "all_reflections": reflections,
        }

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

    def get_feed(self, user_id: str, limit: int = 50) -> dict:
        """Get recent facts with entity info for Memory Feed."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """SELECT f.id, f.content, f.created_at, f.archived,
                          f.importance, f.access_count,
                          e.name as entity_name, e.type as entity_type
                   FROM facts f
                   JOIN entities e ON e.id = f.entity_id
                   WHERE e.user_id = %s AND f.archived = FALSE
                   ORDER BY f.created_at DESC
                   LIMIT %s""",
                (user_id, limit)
            )
            items = []
            for row in cur.fetchall():
                items.append({
                    "id": str(row["id"]),
                    "fact": row["content"],
                    "entity": row["entity_name"],
                    "entity_type": row["entity_type"],
                    "importance": round(float(row["importance"] or 0.5), 2),
                    "access_count": row["access_count"] or 0,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                })
            return {"feed": items, "total": len(items)}

    # =====================================================
    # MEMORY AGENTS v2.0
    # =====================================================

    AGENT_CURATOR_PROMPT = """You are a Memory Curator Agent. Analyze this user's memory for quality issues.

ALL FACTS (grouped by entity):
{facts_text}

Find these issues:
1. CONTRADICTIONS — facts that conflict with each other (e.g., "lives in Almaty" vs "relocated to USA")
2. STALE FACTS — facts that are likely outdated based on context (old job titles, old plans, completed tasks)
3. LOW QUALITY — vague, trivial, or non-useful facts (e.g., "asked a question", "mentioned something")
4. DUPLICATES — facts that say the same thing differently across entities

Return JSON:
{{
  "contradictions": [
    {{"fact_a": "...", "fact_b": "...", "entity_a": "...", "entity_b": "...", "suggestion": "keep A / keep B / ask user"}}
  ],
  "stale": [
    {{"fact": "...", "entity": "...", "reason": "why it seems outdated", "confidence": 0.0-1.0}}
  ],
  "low_quality": [
    {{"fact": "...", "entity": "...", "reason": "why it's low quality"}}
  ],
  "duplicates": [
    {{"facts": ["fact1", "fact2"], "entities": ["entity1", "entity2"], "keep": "best version"}}
  ],
  "health_score": 0.0-1.0,
  "summary": "One paragraph overview of memory health"
}}

Be thorough. Real problems only, not nitpicking. No markdown, just JSON."""

    AGENT_CONNECTOR_PROMPT = """You are a Memory Connector Agent. Your job is to find NON-OBVIOUS connections and patterns in this user's memory that they might not see themselves.

ALL FACTS (grouped by entity):
{facts_text}

EXISTING REFLECTIONS:
{reflections_text}

Find:
1. HIDDEN CONNECTIONS — entities that are related in ways not explicitly stated
2. BEHAVIORAL PATTERNS — recurring decision-making or work patterns
3. SKILL CLUSTERS — groups of related skills/knowledge that form expertise areas
4. STRATEGIC INSIGHTS — observations about trajectory, growth areas, blind spots
5. ACTIONABLE SUGGESTIONS — concrete things the user could do based on their memory

Return JSON:
{{
  "connections": [
    {{"entities": ["A", "B"], "connection": "how they're related", "strength": 0.0-1.0, "insight": "why this matters"}}
  ],
  "patterns": [
    {{"pattern": "description", "evidence": ["fact1", "fact2", "..."], "implication": "what this means"}}
  ],
  "skill_clusters": [
    {{"name": "cluster name", "skills": ["skill1", "skill2"], "level": "beginner/intermediate/expert", "growth_direction": "where this is heading"}}
  ],
  "strategic_insights": [
    {{"insight": "observation", "confidence": 0.0-1.0, "category": "career/technical/personal/project"}}
  ],
  "suggestions": [
    {{"action": "what to do", "reason": "why", "priority": "high/medium/low"}}
  ]
}}

Be insightful, not generic. Find things the user wouldn't notice themselves. No markdown, just JSON."""

    AGENT_DIGEST_PROMPT = """You are a Memory Digest Agent. Create a concise activity digest.

RECENT FACTS (last 7 days):
{recent_facts}

ALL-TIME STATS:
- Total entities: {total_entities}
- Total facts: {total_facts}
- Memory health score: {health_score}

RECENT AGENT FINDINGS:
{agent_findings}

Create a digest:
{{
  "headline": "One-line summary of this week's memory activity",
  "highlights": ["3-5 key things that happened in memory this week"],
  "trends": ["2-3 trends you notice"],
  "memory_grew": {{"entities_added": N, "facts_added": N, "facts_archived": N}},
  "focus_areas": ["what the user has been thinking about most"],
  "recommendation": "One actionable recommendation for the user"
}}

Be specific and personal, not generic. No markdown, just JSON."""

    def ensure_agents_table(self):
        """Create agent_runs table if not exists."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) DEFAULT 'completed',
                    result JSONB,
                    issues_found INTEGER DEFAULT 0,
                    actions_taken INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_runs_user 
                ON agent_runs(user_id, agent_type, created_at DESC)
            """)
            self.conn.commit()

    def run_curator_agent(self, user_id: str, llm_client, auto_fix: bool = False) -> dict:
        """Curator Agent — finds contradictions, stale facts, duplicates, low quality."""
        self.ensure_agents_table()

        entities = self.get_all_entities_full(user_id)
        if not entities:
            return {"status": "empty", "message": "No memories to curate"}

        facts_lines = []
        total_facts = 0
        for e in entities:
            if not e["facts"]:
                continue
            total_facts += len(e["facts"])
            facts_str = ", ".join(e["facts"][:20])
            facts_lines.append(f"- {e['entity']} ({e['type']}): {facts_str}")
        facts_text = "\n".join(facts_lines)

        prompt = self.AGENT_CURATOR_PROMPT.format(facts_text=facts_text)

        try:
            response = llm_client.complete(prompt)
            clean = response.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1])
            result = json.loads(clean)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Curator agent failed: {e}", file=sys.stderr)
            return {"status": "error", "message": str(e)}

        issues_found = (
            len(result.get("contradictions", [])) +
            len(result.get("stale", [])) +
            len(result.get("low_quality", [])) +
            len(result.get("duplicates", []))
        )

        actions_taken = 0
        # Auto-fix: archive low-quality facts with high confidence
        if auto_fix:
            for item in result.get("low_quality", []):
                entity_name = item.get("entity", "")
                fact = item.get("fact", "")
                entity_id = self.get_entity_id(user_id, entity_name)
                if entity_id and fact:
                    with self.conn.cursor() as cur:
                        cur.execute(
                            "UPDATE facts SET archived = TRUE, superseded_by = 'curator: low quality' WHERE entity_id = %s AND content = %s AND archived = FALSE",
                            (entity_id, fact)
                        )
                        if cur.rowcount > 0:
                            actions_taken += 1

            # Auto-fix: archive stale facts with high confidence
            for item in result.get("stale", []):
                if item.get("confidence", 0) >= 0.85:
                    entity_name = item.get("entity", "")
                    fact = item.get("fact", "")
                    entity_id = self.get_entity_id(user_id, entity_name)
                    if entity_id and fact:
                        with self.conn.cursor() as cur:
                            cur.execute(
                                "UPDATE facts SET archived = TRUE, superseded_by = 'curator: stale' WHERE entity_id = %s AND content = %s AND archived = FALSE",
                                (entity_id, fact)
                            )
                            if cur.rowcount > 0:
                                actions_taken += 1

            self.conn.commit()

        # Save run
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_runs (user_id, agent_type, result, issues_found, actions_taken) VALUES (%s, %s, %s, %s, %s)",
                (user_id, "curator", json.dumps(result), issues_found, actions_taken)
            )
            self.conn.commit()

        result["_meta"] = {
            "issues_found": issues_found,
            "actions_taken": actions_taken,
            "total_facts_scanned": total_facts,
            "auto_fix": auto_fix
        }

        print(f"🧹 Curator agent: {issues_found} issues, {actions_taken} auto-fixed for {user_id}", file=sys.stderr)
        return result

    def run_connector_agent(self, user_id: str, llm_client) -> dict:
        """Connector Agent — finds hidden connections, patterns, insights."""
        self.ensure_agents_table()

        entities = self.get_all_entities_full(user_id)
        if not entities:
            return {"status": "empty", "message": "No memories to analyze"}

        facts_lines = []
        for e in entities:
            if not e["facts"]:
                continue
            facts_str = ", ".join(e["facts"][:15])
            facts_lines.append(f"- {e['entity']} ({e['type']}): {facts_str}")
        facts_text = "\n".join(facts_lines)

        # Get existing reflections
        prev = self.get_reflections(user_id)
        reflections_text = "(none)"
        if prev:
            r_lines = [f"- [{r['scope']}] {r['title']}: {r['content'][:150]}" for r in prev[:8]]
            reflections_text = "\n".join(r_lines)

        prompt = self.AGENT_CONNECTOR_PROMPT.format(
            facts_text=facts_text,
            reflections_text=reflections_text
        )

        try:
            response = llm_client.complete(prompt)
            clean = response.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1])
            result = json.loads(clean)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Connector agent failed: {e}", file=sys.stderr)
            return {"status": "error", "message": str(e)}

        issues_found = (
            len(result.get("connections", [])) +
            len(result.get("patterns", [])) +
            len(result.get("strategic_insights", [])) +
            len(result.get("suggestions", []))
        )

        # Save run
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_runs (user_id, agent_type, result, issues_found) VALUES (%s, %s, %s, %s)",
                (user_id, "connector", json.dumps(result), issues_found)
            )
            self.conn.commit()

        print(f"🔗 Connector agent: {issues_found} insights for {user_id}", file=sys.stderr)
        return result

    def run_digest_agent(self, user_id: str, llm_client) -> dict:
        """Digest Agent — generates weekly activity summary."""
        self.ensure_agents_table()

        # Recent facts (last 7 days)
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT f.content, e.name as entity_name, f.created_at
                FROM facts f
                JOIN entities e ON e.id = f.entity_id
                WHERE e.user_id = %s AND f.created_at > NOW() - INTERVAL '7 days'
                AND f.archived = FALSE
                ORDER BY f.created_at DESC LIMIT 50
            """, (user_id,))
            recent = cur.fetchall()

        recent_facts = "(no recent activity)"
        if recent:
            lines = [f"- [{r['entity_name']}] {r['content']} ({r['created_at'].strftime('%m/%d')})" for r in recent]
            recent_facts = "\n".join(lines)

        # Stats
        stats = self.get_stats(user_id)

        # Last curator/connector results
        agent_findings = "(none)"
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT agent_type, result, issues_found, created_at
                FROM agent_runs
                WHERE user_id = %s AND created_at > NOW() - INTERVAL '7 days'
                ORDER BY created_at DESC LIMIT 3
            """, (user_id,))
            runs = cur.fetchall()
            if runs:
                lines = []
                for r in runs:
                    res = r["result"] if isinstance(r["result"], dict) else json.loads(r["result"])
                    summary = res.get("summary", res.get("headline", f"{r['issues_found']} findings"))
                    lines.append(f"- {r['agent_type']}: {summary}")
                agent_findings = "\n".join(lines)

        # Get health score from last curator run
        health_score = "N/A"
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT result FROM agent_runs
                WHERE user_id = %s AND agent_type = 'curator'
                ORDER BY created_at DESC LIMIT 1
            """, (user_id,))
            row = cur.fetchone()
            if row:
                res = row["result"] if isinstance(row["result"], dict) else json.loads(row["result"])
                health_score = str(res.get("health_score", "N/A"))

        prompt = self.AGENT_DIGEST_PROMPT.format(
            recent_facts=recent_facts,
            total_entities=stats.get("total_entities", 0),
            total_facts=stats.get("total_facts", 0),
            health_score=health_score,
            agent_findings=agent_findings
        )

        try:
            response = llm_client.complete(prompt)
            clean = response.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1])
            result = json.loads(clean)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Digest agent failed: {e}", file=sys.stderr)
            return {"status": "error", "message": str(e)}

        # Save run
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_runs (user_id, agent_type, result, issues_found) VALUES (%s, %s, %s, %s)",
                (user_id, "digest", json.dumps(result), len(result.get("highlights", [])))
            )
            self.conn.commit()

        print(f"📰 Digest agent completed for {user_id}", file=sys.stderr)
        return result

    def run_all_agents(self, user_id: str, llm_client, auto_fix: bool = False) -> dict:
        """Run all three agents in sequence."""
        results = {}

        print(f"🤖 Running all agents for {user_id}...", file=sys.stderr)

        # 1. Curator first (clean up)
        results["curator"] = self.run_curator_agent(user_id, llm_client, auto_fix=auto_fix)

        # 2. Connector (find patterns in clean data)
        results["connector"] = self.run_connector_agent(user_id, llm_client)

        # 3. Digest (summarize everything)
        results["digest"] = self.run_digest_agent(user_id, llm_client)

        print(f"✅ All agents completed for {user_id}", file=sys.stderr)
        return results

    def get_agent_history(self, user_id: str, agent_type: str = None, limit: int = 10) -> list:
        """Get history of agent runs."""
        self.ensure_agents_table()
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if agent_type:
                cur.execute("""
                    SELECT agent_type, status, result, issues_found, actions_taken, created_at
                    FROM agent_runs WHERE user_id = %s AND agent_type = %s
                    ORDER BY created_at DESC LIMIT %s
                """, (user_id, agent_type, limit))
            else:
                cur.execute("""
                    SELECT agent_type, status, result, issues_found, actions_taken, created_at
                    FROM agent_runs WHERE user_id = %s
                    ORDER BY created_at DESC LIMIT %s
                """, (user_id, limit))
            rows = cur.fetchall()
            return [{
                "agent_type": r["agent_type"],
                "status": r["status"],
                "result": r["result"] if isinstance(r["result"], dict) else json.loads(r["result"]) if r["result"] else {},
                "issues_found": r["issues_found"],
                "actions_taken": r["actions_taken"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None
            } for r in rows]

    def should_run_agents(self, user_id: str) -> dict:
        """Check if agents should run. Returns which agents are due."""
        self.ensure_agents_table()
        due = {}
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            for agent in ["curator", "connector", "digest"]:
                cur.execute("""
                    SELECT created_at FROM agent_runs
                    WHERE user_id = %s AND agent_type = %s
                    ORDER BY created_at DESC LIMIT 1
                """, (user_id, agent))
                row = cur.fetchone()
                if not row:
                    due[agent] = True
                else:
                    hours_since = (datetime.datetime.now(datetime.timezone.utc) - row["created_at"]).total_seconds() / 3600
                    # Curator: every 24h, Connector: every 48h, Digest: every 7 days
                    thresholds = {"curator": 24, "connector": 48, "digest": 168}
                    due[agent] = hours_since >= thresholds.get(agent, 24)
        return due

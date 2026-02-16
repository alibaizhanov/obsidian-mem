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
import logging
import math
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger("mengram")

# ---- Simple TTL Cache ----
class TTLCache:
    """Thread-safe in-memory cache with TTL."""
    def __init__(self, default_ttl: int = 60):
        self._store = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl

    def get(self, key: str):
        with self._lock:
            item = self._store.get(key)
            if item and item["expires"] > time.time():
                return item["value"]
            if item:
                del self._store[key]
            return None

    def set(self, key: str, value, ttl: int = None):
        with self._lock:
            self._store[key] = {
                "value": value,
                "expires": time.time() + (ttl or self.default_ttl)
            }

    def invalidate(self, prefix: str = ""):
        with self._lock:
            if not prefix:
                self._store.clear()
            else:
                keys = [k for k in self._store if k.startswith(prefix)]
                for k in keys:
                    del self._store[k]

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            alive = sum(1 for v in self._store.values() if v["expires"] > now)
            return {"total_keys": len(self._store), "alive": alive}


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
    
    Features:
    - Connection pooling (ThreadedConnectionPool) for concurrent requests
    - TTL cache for frequent reads (stats, entities, insights)
    - Auto-reconnect on connection failures
    - Proper logging
    """

    def __init__(self, database_url: str, pool_min: int = 2, pool_max: int = 10):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("pip install psycopg2-binary")
        self.database_url = database_url
        self.cache = TTLCache(default_ttl=30)

        # Connection pool
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                pool_min, pool_max, database_url
            )
            logger.info(f"Connection pool created ({pool_min}-{pool_max})")
        except Exception as e:
            logger.warning(f"Pool creation failed, falling back to single connection: {e}")
            self._pool = None

        # Fallback single connection (also used for migrations)
        self.conn = psycopg2.connect(database_url)
        self.conn.autocommit = True
        self._migrate()

    @contextmanager
    def _get_conn(self):
        """Get a connection from pool (or fallback to self.conn).
        Auto-returns to pool on exit. Auto-reconnects on failure."""
        conn = None
        from_pool = False
        try:
            if self._pool:
                conn = self._pool.getconn()
                conn.autocommit = True
                from_pool = True
            else:
                conn = self.conn
            yield conn
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection error: {e}")
            if from_pool and self._pool:
                try:
                    self._pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = self._pool.getconn()
                conn.autocommit = True
                yield conn
            else:
                # Reconnect single connection
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = psycopg2.connect(self.database_url)
                self.conn.autocommit = True
                raise
        finally:
            if from_pool and self._pool and conn:
                try:
                    self._pool.putconn(conn)
                except Exception:
                    pass

    @contextmanager
    def _cursor(self, dict_cursor=False):
        """Get a cursor from a pooled connection. THIS is the primary DB access method.
        All methods should use: with self._cursor() as cur: ...
        This ensures connection pooling is actually used."""
        factory = psycopg2.extras.DictCursor if dict_cursor else None
        with self._get_conn() as conn:
            cur = conn.cursor(cursor_factory=factory)
            try:
                yield cur
            finally:
                cur.close()

    def _migrate(self):
        """Auto-migrate: add new columns if missing."""
        with self._cursor() as cur:
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

        logger.info("✅ Migration complete (v1.5: HNSW + tsvector)")

        # --- v1.6 Importance scoring ---
        with self._cursor() as cur:
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
        logger.info("✅ Migration complete (v1.6: importance scoring)")

        # --- v1.7 Reflection system ---
        with self._cursor() as cur:
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
        logger.info("✅ Migration complete (v1.7: reflection system)")

    def close(self):
        if self._pool:
            self._pool.closeall()
            logger.info("Connection pool closed")
        if self.conn:
            self.conn.close()

    # ---- Auth ----

    def create_user(self, email: str) -> str:
        """Create user, return user_id."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO users (email) VALUES (%s) RETURNING id",
                (email,)
            )
            return str(cur.fetchone()[0])

    def get_user_by_email(self, email: str) -> Optional[str]:
        """Get user_id by email."""
        with self._cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
            return str(row[0]) if row else None

    def create_api_key(self, user_id: str, name: str = "default") -> str:
        """Generate API key, store hash, return raw key."""
        raw_key = f"mg-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:10]

        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO api_keys (user_id, key_hash, key_prefix, name)
                   VALUES (%s, %s, %s, %s)""",
                (user_id, key_hash, key_prefix, name)
            )
        return raw_key

    def verify_api_key(self, raw_key: str) -> Optional[str]:
        """Verify API key, return user_id or None."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        with self._cursor() as cur:
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
        with self._cursor() as cur:
            cur.execute(
                "UPDATE api_keys SET is_active = FALSE WHERE user_id = %s",
                (user_id,)
            )
        return self.create_api_key(user_id)

    # ---- OAuth ----

    def save_email_code(self, email: str, code: str):
        """Save email verification code (expires in 10 min)."""
        with self._cursor() as cur:
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
        with self._cursor() as cur:
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
        with self._cursor() as cur:
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
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor() as cur:
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

        with self._cursor() as cur:
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
        with self._cursor() as cur:
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

        logger.info(f"🔀 Merged entity {source_id} into {target_id} ({target_name})")

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
                logger.info(f"🔀 User → '{canonical_name}' (id: {entity_id})")
                with self._cursor() as cur:
                    cur.execute("UPDATE entities SET updated_at = NOW() WHERE id = %s", (entity_id,))
                self._add_facts_knowledge_relations(entity_id, user_id, canonical_name, facts, relations, knowledge)
                return entity_id

        # Check for case-insensitive exact match first
        with self._cursor() as cur:
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
                with self._cursor() as cur:
                    cur.execute(
                        "UPDATE entities SET name = %s, type = %s, updated_at = NOW() WHERE id = %s",
                        (canonical_name, type, existing_id)
                    )
            else:
                with self._cursor() as cur:
                    cur.execute(
                        "UPDATE entities SET type = %s, updated_at = NOW() WHERE id = %s",
                        (type, existing_id)
                    )
            entity_id = existing_id
            logger.info(f"🔀 Dedup: '{name}' → '{canonical_name}' (id: {entity_id})")
        else:
            with self._cursor() as cur:
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
        added_facts = []
        with self._cursor() as cur:
            for fact in (facts or []):
                importance = self.estimate_importance(fact)
                cur.execute(
                    """INSERT INTO facts (entity_id, content, importance)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (entity_id, content) DO NOTHING
                       RETURNING content""",
                    (entity_id, fact, importance)
                )
                row = cur.fetchone()
                if row:
                    added_facts.append(fact)
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

        # Fire webhooks for new facts
        if added_facts:
            self.fire_webhooks(user_id, "memory_add", {
                "entity": name,
                "facts": added_facts,
                "count": len(added_facts)
            })

        return entity_id

    def _save_relation(self, user_id: str, source_entity_id: str,
                       source_name: str, rel: dict):
        """Save relation, creating target entity if needed."""
        target_name = rel.get("target", "")
        if not target_name:
            return

        with self._cursor() as cur:
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

        # Invalidate caches after write
        self.cache.invalidate(f"stats:{user_id}")

    def get_entity_id(self, user_id: str, name: str) -> Optional[str]:
        """Get entity ID by name."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT id FROM entities WHERE user_id = %s AND name = %s",
                (user_id, name)
            )
            row = cur.fetchone()
            return str(row[0]) if row else None

    def get_entity(self, user_id: str, name: str) -> Optional[CloudEntity]:
        """Get entity with all data."""
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor(dict_cursor=True) as cur:
            cur.execute(
                """SELECT name, type, facts_count, knowledge_count, relations_count
                   FROM entity_overview WHERE user_id = %s AND name NOT LIKE '\\_%%'
                   ORDER BY updated_at DESC""",
                (user_id,)
            )
            return [dict(r) for r in cur.fetchall()]

    def get_all_entities_full(self, user_id: str) -> list[dict]:
        """Get ALL entities with full facts, relations, knowledge in 4 queries total."""
        with self._cursor(dict_cursor=True) as cur:
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

        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM entities WHERE user_id = %s AND name = %s RETURNING id",
                (user_id, name)
            )
            deleted = cur.fetchone() is not None
        if deleted:
            self.cache.invalidate(f"stats:{user_id}")
        return deleted

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

        with self._cursor(dict_cursor=True) as cur:

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
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor(dict_cursor=True) as cur:
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
            logger.error(f"⚠️ Conflict resolution failed: {e}")
            return []

        # Archive contradicted facts
        archived = []
        for old_fact in contradicted:
            if old_fact in old_facts:
                with self._cursor() as cur:
                    # Find matching new fact for superseded_by
                    superseded_by = new_facts[0] if new_facts else None
                    cur.execute(
                        """UPDATE facts SET archived = TRUE, superseded_by = %s
                           WHERE entity_id = %s AND content = %s AND archived = FALSE""",
                        (superseded_by, entity_id, old_fact)
                    )
                archived.append(old_fact)
                logger.info(f"📦 Archived: '{old_fact}' → superseded by '{superseded_by}'")

        return archived

    def dedup_entity_facts(self, entity_id: str, entity_name: str, llm_client) -> dict:
        """Use LLM to deduplicate facts on an entity. 
        Groups similar facts, keeps the best one, archives the rest.
        Returns {kept: [...], archived: [...]}"""
        with self._cursor(dict_cursor=True) as cur:
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
            logger.error(f"⚠️ Dedup failed: {e}")
            return {"kept": facts, "archived": []}

        archived = []
        to_archive = result.get("archive", [])
        for fact in to_archive:
            if fact in facts:
                with self._cursor() as cur:
                    cur.execute(
                        """UPDATE facts SET archived = TRUE, superseded_by = 'dedup'
                           WHERE entity_id = %s AND content = %s AND archived = FALSE""",
                        (entity_id, fact)
                    )
                    if cur.rowcount > 0:
                        archived.append(fact)

        kept = result.get("keep", [])
        logger.info(f"🧹 Dedup '{entity_name}': {len(facts)} → {len(facts)-len(archived)} facts ({len(archived)} archived)")
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
        with self._cursor(dict_cursor=True) as cur:
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
            logger.error(f"⚠️ Reflection generation failed: {e}")
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

        logger.info(f"🧠 Reflections generated for {user_id}: {saved}")
        return result

    def _save_reflection(self, user_id: str, entity_id: Optional[str],
                         scope: str, title: str, content: str,
                         confidence: float = 0.8, based_on: list = None):
        """Save or update a reflection."""
        with self._cursor() as cur:
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
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO entities (user_id, name, type)
                   VALUES (%s, '_reflections', 'concept')
                   ON CONFLICT (user_id, name) DO UPDATE SET updated_at = NOW()
                   RETURNING id""",
                (user_id,)
            )
            return str(cur.fetchone()[0])

    def get_reflections(self, user_id: str, scope: str = None) -> list[dict]:
        """Get all reflections for a user. Cached 120s."""
        cache_key = f"reflections:{user_id}:{scope or 'all'}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._get_reflections_uncached(user_id, scope)
        self.cache.set(cache_key, result, ttl=120)
        return result

    def _get_reflections_uncached(self, user_id: str, scope: str = None) -> list[dict]:
        """Get all reflections for a user (uncached)."""
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO embeddings (entity_id, chunk_text, embedding, tsv)
                   VALUES (%s, %s, %s::vector, to_tsvector('english', %s))""",
                (entity_id, chunk_text, embedding_str, chunk_text)
            )

    def delete_embeddings(self, entity_id: str):
        """Remove all embeddings for entity (before reindex)."""
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM embeddings WHERE entity_id = %s",
                (entity_id,)
            )

    # ---- Stats ----

    def get_stats(self, user_id: str) -> dict:
        """User's vault statistics. Cached for 30s."""
        cache_key = f"stats:{user_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        result = self._get_stats_uncached(user_id)
        self.cache.set(cache_key, result, ttl=30)
        return result

    def _get_stats_uncached(self, user_id: str) -> dict:
        """User's vault statistics (uncached)."""
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO usage_log (user_id, action, tokens_used)
                   VALUES (%s, %s, %s)""",
                (user_id, action, tokens)
            )

    # ---- Graph ----

    def get_graph(self, user_id: str) -> dict:
        """Get knowledge graph (nodes + edges) for visualization."""
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor() as cur:
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

    def run_curator_agent(self, user_id: str, llm_client, auto_fix: bool = False) -> dict:
        """Curator Agent — finds contradictions, stale facts, duplicates, low quality."""
        self.ensure_agents_table()

        entities = self.get_all_entities_full(user_id)
        if not entities:
            return {"status": "empty", "message": "No memories to curate"}

        # Cap data to prevent LLM token overflow
        facts_lines = []
        total_facts = 0
        for e in entities[:50]:  # max 50 entities
            if not e["facts"]:
                continue
            total_facts += len(e["facts"])
            facts_str = ", ".join(e["facts"][:15])  # max 15 facts per entity
            facts_lines.append(f"- {e['entity']} ({e['type']}): {facts_str}")
        facts_text = "\n".join(facts_lines)
        # Hard cap on text size (~8K chars ≈ 2K tokens)
        if len(facts_text) > 8000:
            facts_text = facts_text[:8000] + "\n... (truncated)"
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
            logger.error(f"⚠️ Curator agent failed: {e}")
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
                    with self._cursor() as cur:
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
                        with self._cursor() as cur:
                            cur.execute(
                                "UPDATE facts SET archived = TRUE, superseded_by = 'curator: stale' WHERE entity_id = %s AND content = %s AND archived = FALSE",
                                (entity_id, fact)
                            )
                            if cur.rowcount > 0:
                                actions_taken += 1


        # Save run
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO agent_runs (user_id, agent_type, result, issues_found, actions_taken) VALUES (%s, %s, %s, %s, %s)",
                (user_id, "curator", json.dumps(result), issues_found, actions_taken)
            )

        result["_meta"] = {
            "issues_found": issues_found,
            "actions_taken": actions_taken,
            "total_facts_scanned": total_facts,
            "auto_fix": auto_fix
        }

        logger.info(f"🧹 Curator agent: {issues_found} issues, {actions_taken} auto-fixed for {user_id}")
        return result

    def run_connector_agent(self, user_id: str, llm_client) -> dict:
        """Connector Agent — finds hidden connections, patterns, insights."""
        self.ensure_agents_table()

        entities = self.get_all_entities_full(user_id)
        if not entities:
            return {"status": "empty", "message": "No memories to analyze"}

        facts_lines = []
        for e in entities[:50]:  # max 50 entities
            if not e["facts"]:
                continue
            facts_str = ", ".join(e["facts"][:15])
            facts_lines.append(f"- {e['entity']} ({e['type']}): {facts_str}")
        facts_text = "\n".join(facts_lines)
        if len(facts_text) > 8000:
            facts_text = facts_text[:8000] + "\n... (truncated)"

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
            logger.error(f"⚠️ Connector agent failed: {e}")
            return {"status": "error", "message": str(e)}

        issues_found = (
            len(result.get("connections", [])) +
            len(result.get("patterns", [])) +
            len(result.get("strategic_insights", [])) +
            len(result.get("suggestions", []))
        )

        # Save run
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO agent_runs (user_id, agent_type, result, issues_found) VALUES (%s, %s, %s, %s)",
                (user_id, "connector", json.dumps(result), issues_found)
            )

        logger.info(f"🔗 Connector agent: {issues_found} insights for {user_id}")
        return result

    def run_digest_agent(self, user_id: str, llm_client) -> dict:
        """Digest Agent — generates weekly activity summary."""
        self.ensure_agents_table()

        # Recent facts (last 7 days)
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor(dict_cursor=True) as cur:
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
            logger.error(f"⚠️ Digest agent failed: {e}")
            return {"status": "error", "message": str(e)}

        # Save run
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO agent_runs (user_id, agent_type, result, issues_found) VALUES (%s, %s, %s, %s)",
                (user_id, "digest", json.dumps(result), len(result.get("highlights", [])))
            )

        logger.info(f"📰 Digest agent completed for {user_id}")
        return result

    def run_all_agents(self, user_id: str, llm_client, auto_fix: bool = False) -> dict:
        """Run all three agents in sequence."""
        results = {}

        logger.info(f"🤖 Running all agents for {user_id}...")

        # 1. Curator first (clean up)
        results["curator"] = self.run_curator_agent(user_id, llm_client, auto_fix=auto_fix)

        # 2. Connector (find patterns in clean data)
        results["connector"] = self.run_connector_agent(user_id, llm_client)

        # 3. Digest (summarize everything)
        results["digest"] = self.run_digest_agent(user_id, llm_client)

        logger.info(f"✅ All agents completed for {user_id}")
        return results

    def get_agent_history(self, user_id: str, agent_type: str = None, limit: int = 10) -> list:
        """Get history of agent runs."""
        self.ensure_agents_table()
        with self._cursor(dict_cursor=True) as cur:
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
        with self._cursor(dict_cursor=True) as cur:
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

    # =====================================================
    # WEBHOOKS
    # =====================================================

    def ensure_webhooks_table(self):
        """Create webhooks table if not exists."""
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS webhooks (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    url TEXT NOT NULL,
                    name VARCHAR(255) DEFAULT '',
                    event_types JSONB DEFAULT '["memory_add","memory_update","memory_delete"]',
                    secret VARCHAR(255) DEFAULT '',
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_triggered TIMESTAMPTZ,
                    trigger_count INTEGER DEFAULT 0,
                    last_error TEXT
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_webhooks_user
                ON webhooks(user_id, active)
            """)

    def create_webhook(self, user_id: str, url: str, name: str = "",
                       event_types: list = None, secret: str = "") -> dict:
        """Create a new webhook."""
        self.ensure_webhooks_table()
        if not event_types:
            event_types = ["memory_add", "memory_update", "memory_delete"]

        # Validate event types
        valid = {"memory_add", "memory_update", "memory_delete"}
        for et in event_types:
            if et not in valid:
                raise ValueError(f"Invalid event type: {et}. Valid: {', '.join(valid)}")

        with self._cursor(dict_cursor=True) as cur:
            cur.execute("""
                INSERT INTO webhooks (user_id, url, name, event_types, secret)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, user_id, url, name, event_types, secret, active, created_at
            """, (user_id, url, name, json.dumps(event_types), secret))
            row = cur.fetchone()
            return {
                "id": row["id"],
                "url": row["url"],
                "name": row["name"],
                "event_types": row["event_types"],
                "active": row["active"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None
            }

    def get_webhooks(self, user_id: str) -> list:
        """Get all webhooks for a user."""
        self.ensure_webhooks_table()
        with self._cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT id, url, name, event_types, active, created_at,
                       last_triggered, trigger_count, last_error
                FROM webhooks WHERE user_id = %s ORDER BY created_at DESC
            """, (user_id,))
            return [{
                "id": r["id"],
                "url": r["url"],
                "name": r["name"],
                "event_types": r["event_types"] if isinstance(r["event_types"], list) else json.loads(r["event_types"]),
                "active": r["active"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "last_triggered": r["last_triggered"].isoformat() if r["last_triggered"] else None,
                "trigger_count": r["trigger_count"],
                "last_error": r["last_error"]
            } for r in cur.fetchall()]

    def update_webhook(self, user_id: str, webhook_id: int,
                       url: str = None, name: str = None,
                       event_types: list = None, active: bool = None) -> dict:
        """Update a webhook."""
        self.ensure_webhooks_table()
        updates = []
        params = []
        if url is not None:
            updates.append("url = %s")
            params.append(url)
        if name is not None:
            updates.append("name = %s")
            params.append(name)
        if event_types is not None:
            updates.append("event_types = %s")
            params.append(json.dumps(event_types))
        if active is not None:
            updates.append("active = %s")
            params.append(active)

        if not updates:
            return {"status": "no changes"}

        params.extend([webhook_id, user_id])
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE webhooks SET {', '.join(updates)} WHERE id = %s AND user_id = %s",
                params
            )
            return {"status": "updated", "id": webhook_id}

    def delete_webhook(self, user_id: str, webhook_id: int) -> bool:
        """Delete a webhook."""
        self.ensure_webhooks_table()
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM webhooks WHERE id = %s AND user_id = %s",
                (webhook_id, user_id)
            )
            return cur.rowcount > 0

    def fire_webhooks(self, user_id: str, event_type: str, payload: dict):
        """Fire all active webhooks for this event type. Non-blocking."""
        self.ensure_webhooks_table()
        import threading
        import urllib.request

        with self._cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT id, url, secret FROM webhooks
                WHERE user_id = %s AND active = TRUE
                AND event_types ? %s
            """, (user_id, event_type))
            hooks = cur.fetchall()

        if not hooks:
            return

        data = json.dumps({
            "event": event_type,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data": payload
        }).encode("utf-8")

        def _send(hook_id, url, secret):
            try:
                req = urllib.request.Request(
                    url, data=data,
                    headers={"Content-Type": "application/json"}
                )
                if secret:
                    import hmac as _hmac
                    sig = _hmac.new(secret.encode(), data, hashlib.sha256).hexdigest()
                    req.add_header("X-Mengram-Signature", sig)

                urllib.request.urlopen(req, timeout=10)

                with self._cursor() as cur2:
                    cur2.execute("""
                        UPDATE webhooks SET last_triggered = NOW(),
                        trigger_count = trigger_count + 1, last_error = NULL
                        WHERE id = %s
                    """, (hook_id,))
            except Exception as e:
                logger.error(f"⚠️ Webhook {hook_id} failed: {e}")
                try:
                    with self._cursor() as cur2:
                        cur2.execute("""
                            UPDATE webhooks SET last_error = %s WHERE id = %s
                        """, (str(e)[:500], hook_id))
                except:
                    pass

        for hook in hooks:
            t = threading.Thread(
                target=_send,
                args=(hook["id"], hook["url"], hook["secret"] or ""),
                daemon=True
            )
            t.start()

        logger.info(f"🔔 Fired {len(hooks)} webhooks for {event_type} ({user_id})")

    # =====================================================
    # SHARED MEMORY — TEAMS
    # =====================================================

    def ensure_teams_table(self):
        """Create teams infrastructure."""
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT DEFAULT '',
                    invite_code VARCHAR(20) UNIQUE NOT NULL,
                    created_by VARCHAR(255) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS team_members (
                    id SERIAL PRIMARY KEY,
                    team_id INTEGER REFERENCES teams(id) ON DELETE CASCADE,
                    user_id VARCHAR(255) NOT NULL,
                    role VARCHAR(20) DEFAULT 'member',
                    joined_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(team_id, user_id)
                )
            """)
            # Add team_id column to entities if not exists
            cur.execute("""
                DO $$ BEGIN
                    ALTER TABLE entities ADD COLUMN team_id INTEGER REFERENCES teams(id);
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_team
                ON entities(team_id) WHERE team_id IS NOT NULL
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_team_members_user
                ON team_members(user_id)
            """)

    def create_team(self, user_id: str, name: str, description: str = "") -> dict:
        """Create a new team. Creator becomes owner."""
        self.ensure_teams_table()
        invite_code = secrets.token_urlsafe(8)[:10]

        with self._cursor(dict_cursor=True) as cur:
            cur.execute("""
                INSERT INTO teams (name, description, invite_code, created_by)
                VALUES (%s, %s, %s, %s)
                RETURNING id, name, description, invite_code, created_at
            """, (name, description, invite_code, user_id))
            team = cur.fetchone()
            team_id = team["id"]

            # Creator is owner
            cur.execute("""
                INSERT INTO team_members (team_id, user_id, role)
                VALUES (%s, %s, 'owner')
            """, (team_id, user_id))

            self.cache.invalidate(f"teams:{user_id}")
            return {
                "id": team_id,
                "name": team["name"],
                "description": team["description"],
                "invite_code": team["invite_code"],
                "role": "owner",
                "created_at": team["created_at"].isoformat() if team["created_at"] else None
            }

    def join_team(self, user_id: str, invite_code: str) -> dict:
        """Join a team via invite code."""
        self.ensure_teams_table()
        with self._cursor(dict_cursor=True) as cur:
            cur.execute("SELECT id, name FROM teams WHERE invite_code = %s", (invite_code,))
            team = cur.fetchone()
            if not team:
                raise ValueError("Invalid invite code")

            try:
                cur.execute("""
                    INSERT INTO team_members (team_id, user_id, role)
                    VALUES (%s, %s, 'member')
                """, (team["id"], user_id))
            except Exception:
                pass  # autocommit mode
                raise ValueError("Already a member of this team")

            self.cache.invalidate(f"teams:{user_id}")
            return {"team_id": team["id"], "team_name": team["name"], "role": "member"}

    def get_user_teams(self, user_id: str) -> list:
        """Get all teams user belongs to."""
        self.ensure_teams_table()
        with self._cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT t.id, t.name, t.description, t.invite_code,
                       tm.role, t.created_by, t.created_at,
                       (SELECT COUNT(*) FROM team_members WHERE team_id = t.id) as member_count,
                       (SELECT COUNT(*) FROM entities WHERE team_id = t.id) as shared_memories
                FROM teams t
                JOIN team_members tm ON tm.team_id = t.id
                WHERE tm.user_id = %s
                ORDER BY t.created_at DESC
            """, (user_id,))
            return [{
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "invite_code": r["invite_code"] if r["role"] == "owner" else None,
                "role": r["role"],
                "member_count": r["member_count"],
                "shared_memories": r["shared_memories"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None
            } for r in cur.fetchall()]

    def get_team_members(self, user_id: str, team_id: int) -> list:
        """Get members of a team (must be a member)."""
        self.ensure_teams_table()
        with self._cursor(dict_cursor=True) as cur:
            # Check membership
            cur.execute(
                "SELECT role FROM team_members WHERE team_id = %s AND user_id = %s",
                (team_id, user_id)
            )
            if not cur.fetchone():
                raise ValueError("Not a member of this team")

            cur.execute("""
                SELECT user_id, role, joined_at
                FROM team_members WHERE team_id = %s
                ORDER BY joined_at
            """, (team_id,))
            return [{
                "user_id": r["user_id"],
                "role": r["role"],
                "joined_at": r["joined_at"].isoformat() if r["joined_at"] else None
            } for r in cur.fetchall()]

    def leave_team(self, user_id: str, team_id: int) -> bool:
        """Leave a team."""
        self.ensure_teams_table()
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM team_members WHERE team_id = %s AND user_id = %s AND role != 'owner'",
                (team_id, user_id)
            )
            left = cur.rowcount > 0
        if left:
            self.cache.invalidate(f"teams:{user_id}")
        return left

    def delete_team(self, user_id: str, team_id: int) -> bool:
        """Delete a team (owner only). Shared entities become personal to their creators."""
        self.ensure_teams_table()
        with self._cursor() as cur:
            cur.execute(
                "SELECT role FROM team_members WHERE team_id = %s AND user_id = %s",
                (team_id, user_id)
            )
            row = cur.fetchone()
            if not row or row[0] != "owner":
                raise ValueError("Only the owner can delete a team")

            # Unshare all entities (they become personal again)
            cur.execute("UPDATE entities SET team_id = NULL WHERE team_id = %s", (team_id,))
            cur.execute("DELETE FROM teams WHERE id = %s", (team_id,))
            self.cache.invalidate(f"teams:{user_id}")
            return True

    def share_entity(self, user_id: str, entity_name: str, team_id: int) -> dict:
        """Share a personal entity with a team."""
        self.ensure_teams_table()
        # Verify membership
        with self._cursor(dict_cursor=True) as cur:
            cur.execute(
                "SELECT 1 FROM team_members WHERE team_id = %s AND user_id = %s",
                (team_id, user_id)
            )
            if not cur.fetchone():
                raise ValueError("Not a member of this team")

            cur.execute(
                "UPDATE entities SET team_id = %s WHERE user_id = %s AND LOWER(name) = LOWER(%s)",
                (team_id, user_id, entity_name)
            )
            if cur.rowcount == 0:
                raise ValueError(f"Entity '{entity_name}' not found")
            return {"entity": entity_name, "team_id": team_id, "status": "shared"}

    def unshare_entity(self, user_id: str, entity_name: str) -> dict:
        """Make a shared entity personal again."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE entities SET team_id = NULL WHERE user_id = %s AND LOWER(name) = LOWER(%s)",
                (user_id, entity_name)
            )
            return {"entity": entity_name, "status": "personal"}

    def get_user_team_ids(self, user_id: str) -> list:
        """Get list of team IDs user belongs to. Cached 60s."""
        cache_key = f"teams:{user_id}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        self.ensure_teams_table()
        with self._cursor() as cur:
            cur.execute(
                "SELECT team_id FROM team_members WHERE user_id = %s", (user_id,)
            )
            result = [r[0] for r in cur.fetchall()]
        self.cache.set(cache_key, result, ttl=60)
        return result

    def search_vector_with_teams(self, user_id: str, embedding: list[float],
                                  top_k: int = 5, min_score: float = 0.3,
                                  query_text: str = "") -> list[dict]:
        """
        Same as search_vector but includes shared team memories.
        Results from team entities are marked with team_shared=True.
        """
        team_ids = self.get_user_team_ids(user_id)

        if not team_ids:
            # No teams — use normal search
            return self.search_vector(user_id, embedding, top_k, min_score, query_text)

        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        with self._cursor(dict_cursor=True) as cur:
            # Vector search: personal + team entities
            cur.execute(
                """SELECT DISTINCT ON (e.id)
                       e.id, e.name, e.type, e.user_id, e.team_id,
                       1 - (emb.embedding <=> %s::vector) AS score,
                       e.updated_at
                   FROM embeddings emb
                   JOIN entities e ON e.id = emb.entity_id
                   WHERE (e.user_id = %s OR e.team_id = ANY(%s))
                     AND 1 - (emb.embedding <=> %s::vector) > %s
                   ORDER BY e.id, score DESC""",
                (embedding_str, user_id, team_ids, embedding_str, min_score)
            )
            vector_rows = cur.fetchall()
            vector_rows.sort(key=lambda r: float(r["score"]), reverse=True)
            vector_ranked = {str(r["id"]): (i + 1, r) for i, r in enumerate(vector_rows[:20])}

            # BM25 text search
            bm25_ranked = {}
            if query_text:
                words = [w.strip() for w in query_text.split() if len(w.strip()) >= 2]
                if words:
                    cur.execute(
                        """SELECT DISTINCT ON (e.id)
                               e.id, e.name, e.type, e.user_id, e.team_id,
                               ts_rank(emb.tsv, plainto_tsquery('english', %s)) AS rank,
                               e.updated_at
                           FROM embeddings emb
                           JOIN entities e ON e.id = emb.entity_id
                           WHERE (e.user_id = %s OR e.team_id = ANY(%s))
                             AND emb.tsv @@ plainto_tsquery('english', %s)
                           ORDER BY e.id, rank DESC""",
                        (query_text, user_id, team_ids, query_text)
                    )
                    bm25_rows = cur.fetchall()
                    bm25_rows.sort(key=lambda r: float(r["rank"]), reverse=True)
                    bm25_ranked = {str(r["id"]): (i + 1, r) for i, r in enumerate(bm25_rows[:20])}

            # RRF merge
            k = 60
            all_entity_ids = set(vector_ranked.keys()) | set(bm25_ranked.keys())
            rrf_scores = {}
            entity_info = {}

            for eid in all_entity_ids:
                score = 0.0
                if eid in vector_ranked:
                    rank, row = vector_ranked[eid]
                    score += 1.0 / (k + rank)
                    entity_info[eid] = {
                        "name": row["name"], "type": row["type"],
                        "updated_at": row.get("updated_at"),
                        "team_shared": row["team_id"] is not None and row["user_id"] != user_id
                    }
                if eid in bm25_ranked:
                    rank, row = bm25_ranked[eid]
                    score += 1.0 / (k + rank)
                    if eid not in entity_info:
                        entity_info[eid] = {
                            "name": row["name"], "type": row["type"],
                            "updated_at": row.get("updated_at"),
                            "team_shared": row["team_id"] is not None and row["user_id"] != user_id
                        }
                rrf_scores[eid] = score

            # Sort and build results
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results = []
            for eid, score in sorted_results:
                info = entity_info[eid]
                # Fetch facts
                cur.execute(
                    """SELECT content, importance FROM facts
                       WHERE entity_id = %s AND archived = FALSE
                       ORDER BY importance DESC, created_at DESC LIMIT 15""",
                    (eid,)
                )
                facts = [r["content"] for r in cur.fetchall()]

                # Fetch knowledge
                cur.execute(
                    "SELECT type, title, content, artifact FROM knowledge WHERE entity_id = %s LIMIT 5",
                    (eid,)
                )
                knowledge = [dict(r) for r in cur.fetchall()]

                results.append({
                    "entity": info["name"],
                    "type": info["type"],
                    "score": round(score, 4),
                    "facts": facts,
                    "knowledge": knowledge,
                    "team_shared": info.get("team_shared", False),
                })

            return results

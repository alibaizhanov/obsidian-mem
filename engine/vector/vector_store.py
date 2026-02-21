"""
Vector Store — embeddings storage in SQLite.

Stores text chunks + their vectors.
Search via cosine similarity (numpy, no external dependencies).

Sufficient for vaults up to ~10K notes.
At scale, can be replaced with FAISS/pgvector.
"""

import json
import sqlite3
import numpy as np
from dataclasses import dataclass
from typing import Optional

from engine.vector.embedder import Embedder
from engine.parser.markdown_parser import ParsedNote, parse_vault


@dataclass
class SearchResult:
    """Search result"""
    chunk_id: str
    entity_id: str
    entity_name: str
    section: str
    content: str
    score: float

    def __repr__(self):
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"Result({self.score:.3f} | {self.entity_name}/{self.section}: {preview})"


class VectorStore:
    """SQLite-based Vector Store с cosine similarity search"""

    def __init__(self, db_path: str = ":memory:", embedder: Optional[Embedder] = None):
        self.db_path = db_path
        self.embedder = embedder or Embedder()
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

        # In-memory cache for fast search
        self._vectors: Optional[np.ndarray] = None
        self._chunk_ids: list[str] = []

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                position INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_entity ON chunks(entity_id);
        """)
        self.conn.commit()

    def add_chunk(self, chunk_id: str, entity_id: str, entity_name: str,
                  section: str, content: str, position: int = 0):
        """Add single chunk with automatic embedding generation"""
        vector = self.embedder.embed(content)
        self.conn.execute(
            """INSERT OR REPLACE INTO chunks 
               (id, entity_id, entity_name, section, content, embedding, position)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, entity_id, entity_name, section, content,
             vector.tobytes(), position),
        )
        self.conn.commit()
        self._invalidate_cache()

    def add_chunks_batch(self, chunks: list[dict]):
        """
        Batch-add chunks (faster for bulk indexing).
        chunks: [{chunk_id, entity_id, entity_name, section, content, position}]
        """
        if not chunks:
            return

        # Generate all embeddings in one batch
        texts = [c["content"] for c in chunks]
        vectors = self.embedder.embed_batch(texts)

        # Write to database
        rows = [
            (c["chunk_id"], c["entity_id"], c["entity_name"],
             c["section"], c["content"], vectors[i].tobytes(), c.get("position", 0))
            for i, c in enumerate(chunks)
        ]
        self.conn.executemany(
            """INSERT OR REPLACE INTO chunks 
               (id, entity_id, entity_name, section, content, embedding, position)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()
        self._invalidate_cache()
        print(f"   ✅ Indexed {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[SearchResult]:
        """
        Semantic search by query.
        Returns top_k most relevant chunks.
        """
        self._ensure_cache()

        if len(self._chunk_ids) == 0:
            return []

        # Embed query
        query_vec = self.embedder.embed(query)

        # Cosine similarity (vectors already normalized)
        scores = np.dot(self._vectors, query_vec)

        # Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break

            chunk_id = self._chunk_ids[idx]
            row = self.conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()

            if row:
                results.append(SearchResult(
                    chunk_id=row["id"],
                    entity_id=row["entity_id"],
                    entity_name=row["entity_name"],
                    section=row["section"],
                    content=row["content"],
                    score=score,
                ))

        return results

    def search_by_entity(self, entity_id: str) -> list[dict]:
        """Get all chunks for specific entity"""
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE entity_id = ? ORDER BY position",
            (entity_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        entities = self.conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM chunks"
        ).fetchone()[0]
        return {"total_chunks": total, "total_entities": entities}

    def _ensure_cache(self):
        """Loads all vectors into RAM for fast search"""
        if self._vectors is not None:
            return

        rows = self.conn.execute("SELECT id, embedding FROM chunks").fetchall()
        if not rows:
            self._vectors = np.array([])
            self._chunk_ids = []
            return

        self._chunk_ids = [r["id"] for r in rows]
        self._vectors = np.array([
            np.frombuffer(r["embedding"], dtype=np.float32)
            for r in rows
        ])

    def _invalidate_cache(self):
        """Resets cache when data changes"""
        self._vectors = None
        self._chunk_ids = []

    def close(self):
        self.conn.close()


def index_vault(vault_path: str, db_path: str = ":memory:") -> VectorStore:
    """
    Indexes entire Obsidian vault into Vector Store.
    """
    notes = parse_vault(vault_path)
    store = VectorStore(db_path)

    print(f"📝 Indexing {len(notes)} notes...")

    # Collect all chunks
    all_chunks = []
    for note in notes:
        entity_id = note.name.lower().replace(" ", "_")
        for chunk in note.chunks:
            all_chunks.append({
                "chunk_id": f"{entity_id}:{chunk.position}",
                "entity_id": entity_id,
                "entity_name": note.name,
                "section": chunk.section,
                "content": chunk.content,
                "position": chunk.position,
            })

    # Batch indexing
    store.add_chunks_batch(all_chunks)

    stats = store.stats()
    print(f"✅ Done: {stats['total_chunks']} chunks from {stats['total_entities']} notes")

    return store


if __name__ == "__main__":
    import sys

    vault_path = sys.argv[1] if len(sys.argv) > 1 else "./test_vault"
    store = index_vault(vault_path)

    # Test queries
    queries = [
        "database performance issue",
        "who works on backend",
        "caching and Redis",
        "microservices architecture",
    ]

    for q in queries:
        print(f"\n🔍 Query: '{q}'")
        results = store.search(q, top_k=3)
        for r in results:
            print(f"   {r.score:.3f} | {r.entity_name}/{r.section}")
            print(f"           {r.content[:80]}...")

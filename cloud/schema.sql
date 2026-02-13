-- ObsidianMem Cloud — PostgreSQL Schema
-- Replaces .md files + SQLite vectors with single PostgreSQL + pgvector

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. Users & API Keys
-- ============================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 of "om-..."
    key_prefix VARCHAR(10) NOT NULL,       -- "om-abc..." for display
    name VARCHAR(100) DEFAULT 'default',
    created_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- ============================================
-- 2. Entities (replaces .md files)
-- ============================================

CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL DEFAULT 'concept',  -- person, project, technology, company, concept
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(user_id, name)
);

CREATE INDEX idx_entities_user ON entities(user_id);
CREATE INDEX idx_entities_type ON entities(user_id, type);
CREATE INDEX idx_entities_name ON entities(user_id, name);

-- ============================================
-- 3. Facts (replaces ## Facts section in .md)
-- ============================================

CREATE TABLE facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(entity_id, content)
);

CREATE INDEX idx_facts_entity ON facts(entity_id);

-- ============================================
-- 4. Relations (replaces ## Relations section)
-- ============================================

CREATE TABLE relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    type VARCHAR(100) NOT NULL,          -- uses, works_at, depends_on, etc
    description TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(source_id, target_id, type)
);

CREATE INDEX idx_relations_source ON relations(source_id);
CREATE INDEX idx_relations_target ON relations(target_id);

-- ============================================
-- 5. Knowledge (replaces ## Knowledge section)
-- ============================================

CREATE TABLE knowledge (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,           -- solution, config, command, debug, formula, etc
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    artifact TEXT,                        -- code snippet, YAML config, SQL query, etc
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(entity_id, title)
);

CREATE INDEX idx_knowledge_entity ON knowledge(entity_id);
CREATE INDEX idx_knowledge_type ON knowledge(entity_id, type);

-- ============================================
-- 6. Vector Embeddings (replaces SQLite vectors)
-- ============================================

CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1536),               -- OpenAI text-embedding-3-small = 1536 dimensions
    tsv tsvector,                         -- BM25 text search (hybrid search)
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_embeddings_entity ON embeddings(entity_id);

-- HNSW index for fast approximate nearest neighbor search (O(log n) vs O(n))
CREATE INDEX idx_embeddings_hnsw ON embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for fast BM25 text search
CREATE INDEX idx_embeddings_tsv ON embeddings USING gin(tsv);

-- ============================================
-- 7. Usage tracking (for dashboard / billing)
-- ============================================

CREATE TABLE usage_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL,         -- remember, recall, search, chat
    tokens_used INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_usage_user_date ON usage_log(user_id, created_at);

-- ============================================
-- Helper views
-- ============================================

-- Entity overview with counts
CREATE VIEW entity_overview AS
SELECT
    e.id,
    e.user_id,
    e.name,
    e.type,
    e.created_at,
    e.updated_at,
    COUNT(DISTINCT f.id) AS facts_count,
    COUNT(DISTINCT k.id) AS knowledge_count,
    COUNT(DISTINCT r1.id) + COUNT(DISTINCT r2.id) AS relations_count
FROM entities e
LEFT JOIN facts f ON f.entity_id = e.id
LEFT JOIN knowledge k ON k.entity_id = e.id
LEFT JOIN relations r1 ON r1.source_id = e.id
LEFT JOIN relations r2 ON r2.target_id = e.id
GROUP BY e.id;

-- ============================================
-- Example: semantic search query
-- ============================================
-- SELECT e.name, e.type, 1 - (emb.embedding <=> $1::vector) AS score
-- FROM embeddings emb
-- JOIN entities e ON e.id = emb.entity_id
-- WHERE e.user_id = $2
-- ORDER BY emb.embedding <=> $1::vector
-- LIMIT 5;

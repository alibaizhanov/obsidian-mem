-- Mengram Cloud — PostgreSQL Schema
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
    embedding vector(1536),               -- OpenAI text-embedding-3-large @ 1536 dimensions (Matryoshka)
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
-- 7. Episodic Memory (v2.5 — event/interaction memory)
-- ============================================

CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    context TEXT,
    outcome TEXT,
    participants TEXT[] DEFAULT '{}',
    emotional_valence VARCHAR(20) DEFAULT 'neutral',
    importance FLOAT DEFAULT 0.5,
    linked_procedure_id UUID,             -- v2.7: link to procedure that was followed/failed
    failed_at_step INT,                   -- v2.7: which step failed (NULL = not a procedure failure)
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX idx_episodes_user ON episodes(user_id, created_at DESC);
CREATE INDEX idx_episodes_participants ON episodes USING gin(participants);
CREATE INDEX idx_episodes_expires ON episodes(expires_at) WHERE expires_at IS NOT NULL;

CREATE TABLE episode_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1536),
    tsv tsvector,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ep_emb_episode ON episode_embeddings(episode_id);
CREATE INDEX idx_ep_emb_hnsw ON episode_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_ep_emb_tsv ON episode_embeddings USING gin(tsv);

-- ============================================
-- 8. Procedural Memory (v2.5 — workflow/skill memory)
-- ============================================

CREATE TABLE procedures (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    name VARCHAR(255) NOT NULL,
    trigger_condition TEXT,
    steps JSONB NOT NULL DEFAULT '[]',
    source_episode_ids UUID[] DEFAULT '{}',
    entity_names TEXT[] DEFAULT '{}',
    success_count INT DEFAULT 0,
    fail_count INT DEFAULT 0,
    last_used TIMESTAMPTZ,
    version INT DEFAULT 1,                          -- v2.7: procedure version number
    parent_version_id UUID REFERENCES procedures(id),  -- v2.7: previous version
    evolved_from_episode UUID,                      -- v2.7: episode that triggered evolution
    is_current BOOLEAN DEFAULT TRUE,                -- v2.7: only latest version is current
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    UNIQUE(user_id, name, version)
);

CREATE INDEX idx_procedures_user ON procedures(user_id, updated_at DESC);
CREATE INDEX idx_procedures_entities ON procedures USING gin(entity_names);

CREATE TABLE procedure_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    procedure_id UUID REFERENCES procedures(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1536),
    tsv tsvector,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_proc_emb_procedure ON procedure_embeddings(procedure_id);
CREATE INDEX idx_proc_emb_hnsw ON procedure_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_proc_emb_tsv ON procedure_embeddings USING gin(tsv);

-- Add FK for episodes → procedures (deferred because procedures table is created after episodes)
ALTER TABLE episodes ADD CONSTRAINT fk_episodes_linked_procedure
    FOREIGN KEY (linked_procedure_id) REFERENCES procedures(id) ON DELETE SET NULL;

-- v2.7: filter only current versions
CREATE INDEX idx_procedures_current ON procedures(user_id, is_current) WHERE is_current = TRUE;

-- ============================================
-- 8b. Procedure Evolution Log (v2.7 — experience-driven procedures)
-- ============================================

CREATE TABLE procedure_evolution (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,
    episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
    change_type VARCHAR(30) NOT NULL,   -- step_added, step_removed, step_modified, step_reordered, auto_created
    diff JSONB DEFAULT '{}',            -- {added: [...], removed: [...], modified: [...]}
    version_before INT,
    version_after INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_proc_evolution_proc ON procedure_evolution(procedure_id, created_at DESC);

-- ============================================
-- 9. Usage tracking (for dashboard / billing)
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
-- 10. Smart Memory Triggers (v2.6 — proactive memory)
-- ============================================

CREATE TABLE memory_triggers (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    trigger_type VARCHAR(30) NOT NULL,  -- 'reminder', 'contradiction', 'pattern'
    title TEXT NOT NULL,                -- human-readable summary
    detail TEXT,                        -- full context
    source_type VARCHAR(30),            -- 'episode', 'fact', 'procedure'
    source_id UUID,                     -- ID of source memory
    fire_at TIMESTAMPTZ,               -- when to fire (NULL = fire immediately)
    fired BOOLEAN DEFAULT FALSE,
    fired_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_triggers_pending ON memory_triggers(user_id, fired, fire_at)
    WHERE fired = FALSE;
CREATE INDEX idx_triggers_user ON memory_triggers(user_id, created_at DESC);

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

-- Memory Bus tables — spec §10.3
-- Runs against the 'hololoom' database on first Docker boot.

\connect hololoom;

-- Facts — flat fast lookup
CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    neo4j_node_id TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ,
    domain TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Plans — state machine
CREATE TABLE IF NOT EXISTS plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    neo4j_node_id TEXT NOT NULL,
    name TEXT NOT NULL,
    state TEXT CHECK (state IN ('active', 'paused', 'completed', 'abandoned')),
    progress FLOAT DEFAULT 0.0,
    loom_id TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Configs — key-value store
CREATE TABLE IF NOT EXISTS configs (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    loom_id TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit log — append-only, never deleted
CREATE TABLE IF NOT EXISTS memory_audit (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    loop_id TEXT,
    loom_id TEXT,
    action TEXT,
    resolution_path TEXT,
    tokens_used INT,
    pressure_tier INT,
    details JSONB
);

-- Entity aliases — supports resolve_entity
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    loom_id TEXT,
    confidence FLOAT DEFAULT 1.0,
    PRIMARY KEY (alias, entity_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_alias_lower ON entity_aliases (LOWER(alias));
CREATE INDEX IF NOT EXISTS idx_alias_entity ON entity_aliases (entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON memory_audit (action);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON memory_audit (timestamp);
CREATE INDEX IF NOT EXISTS idx_facts_domain ON facts (domain);
CREATE INDEX IF NOT EXISTS idx_plans_state ON plans (state);

-- HoloLoom PostgreSQL Schema
-- ===========================
-- Initial schema for workflow state persistence and caching
-- Version: 1.0.0
-- Date: November 17, 2025

-- ===========================================================================
-- Extensions
-- ===========================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For composite indexes

-- ===========================================================================
-- Workflow State Tables
-- ===========================================================================

-- Workflow definitions
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    definition JSONB NOT NULL,  -- Stores workflow nodes and connections
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    CONSTRAINT workflows_name_version_unique UNIQUE (name, version)
);

CREATE INDEX idx_workflows_name ON workflows(name);
CREATE INDEX idx_workflows_status ON workflows(status);
CREATE INDEX idx_workflows_created_at ON workflows(created_at DESC);

-- Workflow executions
CREATE TABLE workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_ms INTEGER,
    CONSTRAINT workflow_executions_workflow_fk FOREIGN KEY (workflow_id) REFERENCES workflows(id)
);

CREATE INDEX idx_workflow_executions_workflow ON workflow_executions(workflow_id);
CREATE INDEX idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX idx_workflow_executions_created_at ON workflow_executions(created_at DESC);

-- Workflow execution steps
CREATE TABLE workflow_execution_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    node_id VARCHAR(255) NOT NULL,
    node_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    retry_count INTEGER DEFAULT 0,
    CONSTRAINT workflow_execution_steps_execution_fk FOREIGN KEY (execution_id) REFERENCES workflow_executions(id)
);

CREATE INDEX idx_workflow_execution_steps_execution ON workflow_execution_steps(execution_id);
CREATE INDEX idx_workflow_execution_steps_node ON workflow_execution_steps(node_id);
CREATE INDEX idx_workflow_execution_steps_status ON workflow_execution_steps(status);

-- ===========================================================================
-- Compositional Cache Tables
-- ===========================================================================

-- Parse cache (Universal Grammar chunking)
CREATE TABLE parse_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 of input text
    text_content TEXT NOT NULL,
    parsed_structure JSONB NOT NULL,  -- X-bar theory structure
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);

CREATE INDEX idx_parse_cache_hash ON parse_cache(text_hash);
CREATE INDEX idx_parse_cache_last_accessed ON parse_cache(last_accessed_at DESC);

-- Merge cache (compositional phrase reuse)
CREATE TABLE merge_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phrase_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 of phrase
    phrase_content TEXT NOT NULL,
    merged_result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);

CREATE INDEX idx_merge_cache_hash ON merge_cache(phrase_hash);
CREATE INDEX idx_merge_cache_last_accessed ON merge_cache(last_accessed_at DESC);

-- Semantic cache (228D projections)
CREATE TABLE semantic_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 of query
    query_content TEXT NOT NULL,
    semantic_projection VECTOR(228),  -- Requires pgvector extension
    embedding_384d VECTOR(384),  -- Full embedding
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count INTEGER DEFAULT 1
);

-- Note: VECTOR type requires pgvector extension
-- If pgvector not available, use JSONB instead:
-- ALTER TABLE semantic_cache ALTER COLUMN semantic_projection TYPE JSONB;

CREATE INDEX idx_semantic_cache_hash ON semantic_cache(query_hash);
CREATE INDEX idx_semantic_cache_last_accessed ON semantic_cache(last_accessed_at DESC);

-- ===========================================================================
-- Query History Tables
-- ===========================================================================

-- Query history for analytics
CREATE TABLE query_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_mode VARCHAR(50) NOT NULL,  -- direct, verify, research, plan_execute
    confidence FLOAT NOT NULL,
    latency_ms INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB  -- Additional context
);

CREATE INDEX idx_query_history_created_at ON query_history(created_at DESC);
CREATE INDEX idx_query_history_mode ON query_history(query_mode);
CREATE INDEX idx_query_history_user ON query_history(user_id);
CREATE INDEX idx_query_history_session ON query_history(session_id);

-- ===========================================================================
-- Learning System Tables
-- ===========================================================================

-- Hot patterns (recursive learning)
CREATE TABLE hot_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_text TEXT NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,  -- motif, tool, refinement_strategy
    heat_score FLOAT NOT NULL DEFAULT 0.0,
    access_count INTEGER DEFAULT 1,
    success_rate FLOAT,
    avg_confidence FLOAT,
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_hot_patterns_heat ON hot_patterns(heat_score DESC);
CREATE INDEX idx_hot_patterns_type ON hot_patterns(pattern_type);
CREATE INDEX idx_hot_patterns_last_accessed ON hot_patterns(last_accessed_at DESC);

-- Thompson Sampling priors
CREATE TABLE thompson_priors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tool_name VARCHAR(255) NOT NULL UNIQUE,
    alpha FLOAT NOT NULL DEFAULT 1.0,  -- Success parameter
    beta FLOAT NOT NULL DEFAULT 1.0,   -- Failure parameter
    total_uses INTEGER DEFAULT 0,
    last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_thompson_priors_tool ON thompson_priors(tool_name);

-- ===========================================================================
-- Alignment & Safety Tables
-- ===========================================================================

-- Audit trail
CREATE TABLE audit_trail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    decision_id VARCHAR(255) NOT NULL,
    decision_type VARCHAR(100) NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    reason TEXT,
    confidence FLOAT,
    safety_score FLOAT,
    context JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_trail_timestamp ON audit_trail(timestamp DESC);
CREATE INDEX idx_audit_trail_decision_id ON audit_trail(decision_id);
CREATE INDEX idx_audit_trail_decision_type ON audit_trail(decision_type);

-- Deception detection alerts
CREATE TABLE deception_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    evidence JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE,
    resolution_note TEXT
);

CREATE INDEX idx_deception_alerts_timestamp ON deception_alerts(timestamp DESC);
CREATE INDEX idx_deception_alerts_resolved ON deception_alerts(resolved);
CREATE INDEX idx_deception_alerts_severity ON deception_alerts(severity);

-- ===========================================================================
-- Triggers (Auto-update timestamps)
-- ===========================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_workflows_updated_at
    BEFORE UPDATE ON workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===========================================================================
-- Views (Analytics)
-- ===========================================================================

-- Workflow execution summary
CREATE VIEW workflow_execution_summary AS
SELECT
    w.id AS workflow_id,
    w.name AS workflow_name,
    w.version,
    COUNT(DISTINCT we.id) AS total_executions,
    COUNT(DISTINCT CASE WHEN we.status = 'completed' THEN we.id END) AS completed_executions,
    COUNT(DISTINCT CASE WHEN we.status = 'failed' THEN we.id END) AS failed_executions,
    AVG(we.duration_ms) AS avg_duration_ms,
    MAX(we.duration_ms) AS max_duration_ms,
    MIN(we.duration_ms) AS min_duration_ms
FROM workflows w
LEFT JOIN workflow_executions we ON w.id = we.workflow_id
GROUP BY w.id, w.name, w.version;

-- Query performance summary
CREATE VIEW query_performance_summary AS
SELECT
    query_mode,
    COUNT(*) AS total_queries,
    COUNT(*) FILTER (WHERE success = TRUE) AS successful_queries,
    COUNT(*) FILTER (WHERE success = FALSE) AS failed_queries,
    AVG(latency_ms) AS avg_latency_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms,
    AVG(confidence) AS avg_confidence
FROM query_history
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY query_mode;

-- Cache effectiveness
CREATE VIEW cache_effectiveness AS
SELECT
    'parse_cache' AS cache_type,
    COUNT(*) AS total_entries,
    SUM(access_count) AS total_accesses,
    AVG(access_count) AS avg_accesses_per_entry
FROM parse_cache
UNION ALL
SELECT
    'merge_cache' AS cache_type,
    COUNT(*) AS total_entries,
    SUM(access_count) AS total_accesses,
    AVG(access_count) AS avg_accesses_per_entry
FROM merge_cache
UNION ALL
SELECT
    'semantic_cache' AS cache_type,
    COUNT(*) AS total_entries,
    SUM(access_count) AS total_accesses,
    AVG(access_count) AS avg_accesses_per_entry
FROM semantic_cache;

-- ===========================================================================
-- Initial Data
-- ===========================================================================

-- Insert default Thompson Sampling priors for common tools
INSERT INTO thompson_priors (tool_name, alpha, beta) VALUES
    ('answer', 1.0, 1.0),
    ('search', 1.0, 1.0),
    ('synthesis', 1.0, 1.0),
    ('verification', 1.0, 1.0)
ON CONFLICT (tool_name) DO NOTHING;

-- ===========================================================================
-- Cleanup Function (for old cache entries)
-- ===========================================================================

CREATE OR REPLACE FUNCTION cleanup_old_cache_entries()
RETURNS void AS $$
BEGIN
    -- Delete parse cache entries older than 30 days with low access
    DELETE FROM parse_cache
    WHERE last_accessed_at < NOW() - INTERVAL '30 days'
      AND access_count < 5;

    -- Delete merge cache entries older than 30 days with low access
    DELETE FROM merge_cache
    WHERE last_accessed_at < NOW() - INTERVAL '30 days'
      AND access_count < 5;

    -- Delete semantic cache entries older than 30 days with low access
    DELETE FROM semantic_cache
    WHERE last_accessed_at < NOW() - INTERVAL '30 days'
      AND access_count < 5;

    -- Delete old query history (keep 90 days)
    DELETE FROM query_history
    WHERE created_at < NOW() - INTERVAL '90 days';

    -- Delete old audit trail (keep 1 year)
    DELETE FROM audit_trail
    WHERE timestamp < NOW() - INTERVAL '1 year';
END;
$$ LANGUAGE plpgsql;

-- Schedule cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup_old_cache', '0 2 * * *', 'SELECT cleanup_old_cache_entries();');

-- HoloLoom Promptly Workflow Versioning Schema
-- =============================================
-- SQLite extension for Promptly prompt management system
-- Provides workflow versioning, evaluation tracking, and prompt linking
--
-- Design Principles:
-- - Compatible with existing Promptly schema (prompts table)
-- - Supports branching and versioning workflows
-- - Complete evaluation metrics tracking
-- - Idempotent (can be run multiple times safely)
-- - Foreign key constraints for referential integrity

-- ============================================================================
-- WORKFLOWS TABLE
-- ============================================================================
-- Stores workflow definitions with version history and branching support
--
-- Key Fields:
--   - id: Unique workflow identifier (auto-increment)
--   - name: Human-readable workflow name (e.g., "research_loop", "code_review")
--   - workflow_json: Complete workflow definition in JSON format
--     Structure: {"nodes": [...], "connections": [...], "config": {...}}
--   - branch: Git-like branching (default: "main")
--   - version: Version number within branch (starts at 1)
--   - parent_id: Previous version ID (enables history traversal)
--   - commit_hash: Unique SHA-256 hash (for deduplication, audit trails)
--   - created_at: Creation timestamp (UTC)
--   - metadata: JSON with structural info: nodes, connections, performance
--     Structure: {"node_count": int, "connection_count": int, "avg_latency_ms": float}

CREATE TABLE IF NOT EXISTS workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    workflow_json TEXT NOT NULL,
    branch TEXT DEFAULT 'main',
    version INTEGER NOT NULL DEFAULT 1,
    parent_id INTEGER,
    commit_hash TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,  -- JSON: {nodes, connections, performance}

    FOREIGN KEY (parent_id) REFERENCES workflows(id),
    CONSTRAINT workflow_version_uniqueness UNIQUE (name, branch, version)
);

-- Index for fast workflow lookups by name and branch
CREATE INDEX IF NOT EXISTS idx_workflow_name_branch
    ON workflows(name, branch);

-- Index for commit hash lookups (deduplication)
CREATE INDEX IF NOT EXISTS idx_workflow_commit_hash
    ON workflows(commit_hash);

-- Index for parent tracking (history traversal)
CREATE INDEX IF NOT EXISTS idx_workflow_parent
    ON workflows(parent_id);

-- Index for timestamp range queries (e.g., "recent changes")
CREATE INDEX IF NOT EXISTS idx_workflow_created_at
    ON workflows(created_at DESC);

-- ============================================================================
-- WORKFLOW_PROMPTS TABLE
-- ============================================================================
-- Links workflows to prompts (many-to-many relationship)
-- Enables tracking which prompts are used by which workflows and in which nodes
--
-- Key Fields:
--   - workflow_id: Reference to workflows table
--   - prompt_id: Reference to existing prompts table (Promptly schema)
--   - node_id: Identifies which node in the workflow uses this prompt
--     Format: "node_<type>_<index>" (e.g., "node_analysis_1", "node_retrieve_0")

CREATE TABLE IF NOT EXISTS workflow_prompts (
    workflow_id INTEGER NOT NULL,
    prompt_id INTEGER NOT NULL,
    node_id TEXT NOT NULL,  -- Which node uses this prompt

    PRIMARY KEY (workflow_id, prompt_id, node_id),
    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE,
    FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
);

-- Index for finding all prompts used by a workflow
CREATE INDEX IF NOT EXISTS idx_workflow_prompts_workflow
    ON workflow_prompts(workflow_id);

-- Index for finding all workflows using a prompt (reverse lookup)
CREATE INDEX IF NOT EXISTS idx_workflow_prompts_prompt
    ON workflow_prompts(prompt_id);

-- Index for node-based lookups
CREATE INDEX IF NOT EXISTS idx_workflow_prompts_node
    ON workflow_prompts(node_id);

-- ============================================================================
-- WORKFLOW_EVALUATIONS TABLE
-- ============================================================================
-- Tracks evaluation metrics for workflow versions
-- Used for A/B testing, performance monitoring, and quality assessment
--
-- Key Fields:
--   - id: Unique evaluation identifier
--   - workflow_name: Which workflow was evaluated (matches workflows.name)
--   - commit_hash: Which version was evaluated (matches workflows.commit_hash)
--   - test_case: Specific test case input (can be null for aggregate evaluations)
--   - metrics: JSON with detailed performance metrics
--     Structure: {
--       "latency_ms": float,
--       "confidence": float (0.0-1.0),
--       "quality": float (0.0-1.0),
--       "token_usage": int,
--       "cache_hit": bool,
--       "tool_used": string,
--       "error": string (null if no error)
--     }
--   - created_at: Evaluation timestamp (UTC)

CREATE TABLE IF NOT EXISTS workflow_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_name TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    test_case TEXT,  -- Optional specific test case
    metrics TEXT NOT NULL,  -- JSON: latency, confidence, quality, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (commit_hash) REFERENCES workflows(commit_hash)
);

-- Index for workflow performance queries
CREATE INDEX IF NOT EXISTS idx_eval_workflow_commit
    ON workflow_evaluations(workflow_name, commit_hash);

-- Index for time-based queries (e.g., "evaluations from last 24 hours")
CREATE INDEX IF NOT EXISTS idx_eval_created_at
    ON workflow_evaluations(created_at DESC);

-- Index for test case specific evaluations
CREATE INDEX IF NOT EXISTS idx_eval_test_case
    ON workflow_evaluations(test_case) WHERE test_case IS NOT NULL;

-- ============================================================================
-- VIEWS (for common queries)
-- ============================================================================

-- Latest version of each workflow on each branch
CREATE VIEW IF NOT EXISTS workflow_latest_by_branch AS
SELECT DISTINCT ON (name, branch)
    id, name, branch, version, commit_hash, created_at
FROM workflows
ORDER BY name, branch, version DESC;

-- Workflow evaluation summary (aggregate metrics per commit)
CREATE VIEW IF NOT EXISTS workflow_eval_summary AS
SELECT
    workflow_name,
    commit_hash,
    COUNT(*) as eval_count,
    AVG(CAST(json_extract(metrics, '$.latency_ms') AS FLOAT)) as avg_latency_ms,
    AVG(CAST(json_extract(metrics, '$.confidence') AS FLOAT)) as avg_confidence,
    AVG(CAST(json_extract(metrics, '$.quality') AS FLOAT)) as avg_quality,
    MIN(created_at) as first_eval,
    MAX(created_at) as last_eval
FROM workflow_evaluations
GROUP BY workflow_name, commit_hash;

-- Workflows using specific prompt
CREATE VIEW IF NOT EXISTS prompt_usage_analysis AS
SELECT
    p.id as prompt_id,
    p.name as prompt_name,
    COUNT(DISTINCT wp.workflow_id) as workflow_count,
    GROUP_CONCAT(DISTINCT w.name) as workflows_using_prompt
FROM workflow_prompts wp
JOIN prompts p ON wp.prompt_id = p.id
JOIN workflows w ON wp.workflow_id = w.id
GROUP BY p.id, p.name;

-- ============================================================================
-- COMPATIBILITY NOTES
-- ============================================================================
-- This schema is designed to be fully compatible with existing Promptly
-- database. It assumes:
--
-- 1. A "prompts" table exists with at least:
--    - id (PRIMARY KEY)
--    - name (TEXT NOT NULL)
--    - version (optional)
--
-- 2. Optional integration with workflow automation:
--    - Workflows reference prompts via workflow_prompts table
--    - Evaluations track quality metrics for optimization
--
-- 3. If Promptly schema changes, only foreign keys may need updating
--    (workflow_prompts.prompt_id reference)
--
-- All tables use CASCADE delete for orphan cleanup and UNIQUE constraints
-- to prevent duplicate entries.

-- HoloLoom Hybrid Query Routing - SQL Schema
-- Part 2: Foundation Infrastructure (Days 1-5)
-- Validated in Demo 3 (1ms p95 latency)

-- ============================================================================
-- Precision Data Tables (4 tables)
-- ============================================================================

-- 1. Policy Rules (ground truth policies)
CREATE TABLE IF NOT EXISTS policy_rules (
    rule_id TEXT PRIMARY KEY,
    rule_name TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    rule_logic TEXT NOT NULL,  -- JSON serialized
    confidence REAL DEFAULT 1.0,
    domain TEXT DEFAULT 'beekeeping',
    neo4j_node_id TEXT,  -- Link to Neo4j graph
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_policy_rules_name ON policy_rules(rule_name);
CREATE INDEX IF NOT EXISTS idx_policy_rules_type ON policy_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_policy_rules_domain ON policy_rules(domain);


-- 2. Transaction Logs (precision data)
CREATE TABLE IF NOT EXISTS transaction_logs (
    transaction_id TEXT PRIMARY KEY,
    transaction_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action_data TEXT NOT NULL,  -- JSON serialized
    neo4j_node_id TEXT,  -- Link to Neo4j graph
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transaction_logs_entity ON transaction_logs(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_transaction_logs_user ON transaction_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_transaction_logs_timestamp ON transaction_logs(timestamp);


-- 3. Audit Trails (compliance tracking)
CREATE TABLE IF NOT EXISTS audit_trails (
    audit_id TEXT PRIMARY KEY,
    audit_type TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    before_state TEXT,  -- JSON snapshot
    after_state TEXT,   -- JSON snapshot
    compliance_flag BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_trails_resource ON audit_trails(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_trails_compliance ON audit_trails(compliance_flag);
CREATE INDEX IF NOT EXISTS idx_audit_trails_timestamp ON audit_trails(timestamp);


-- 4. User Permissions (access control)
CREATE TABLE IF NOT EXISTS user_permissions (
    permission_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    permission_level TEXT NOT NULL,  -- read, write, admin
    neo4j_user_node TEXT,  -- Link to Neo4j user node
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_permissions_user ON user_permissions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_permissions_resource ON user_permissions(resource_type);
CREATE INDEX IF NOT EXISTS idx_user_permissions_expires ON user_permissions(expires_at);


-- ============================================================================
-- Schema Metadata (for migration tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id TEXT PRIMARY KEY,
    migration_name TEXT NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT NOT NULL
);

-- Record initial schema creation
INSERT INTO schema_migrations (migration_id, migration_name, checksum)
VALUES
    ('001', 'initial_schema', 'demo3_validated')
ON CONFLICT (migration_id) DO NOTHING;


-- ============================================================================
-- Performance Analysis View
-- ============================================================================

-- View for monitoring query performance
CREATE VIEW IF NOT EXISTS v_policy_summary AS
SELECT
    domain,
    rule_type,
    COUNT(*) as rule_count,
    AVG(confidence) as avg_confidence,
    MIN(created_at) as oldest_rule,
    MAX(created_at) as newest_rule
FROM policy_rules
GROUP BY domain, rule_type;

CREATE VIEW IF NOT EXISTS v_transaction_summary AS
SELECT
    entity_type,
    transaction_type,
    COUNT(*) as transaction_count,
    MIN(timestamp) as first_transaction,
    MAX(timestamp) as last_transaction
FROM transaction_logs
GROUP BY entity_type, transaction_type;

CREATE VIEW IF NOT EXISTS v_audit_summary AS
SELECT
    resource_type,
    audit_type,
    COUNT(*) as audit_count,
    SUM(CASE WHEN compliance_flag THEN 1 ELSE 0 END) as compliance_issues,
    MIN(timestamp) as first_audit,
    MAX(timestamp) as last_audit
FROM audit_trails
GROUP BY resource_type, audit_type;


-- ============================================================================
-- Schema Version
-- ============================================================================

-- Track schema version for compatibility checks
CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    description TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO schema_version (version, description)
VALUES ('1.0.0', 'Initial schema from Part 1 validation (Demo 3)')
ON CONFLICT (version) DO NOTHING;

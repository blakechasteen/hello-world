# HoloLoom Hybrid Query Routing Architecture

**Version:** 1.0.0
**Date:** November 10, 2025
**Status:** Design Document
**Integration:** Phase 1 Infrastructure Enhancement (Moonshot Roadmap)

---

## Executive Summary

### Problem Statement

HoloLoom's Infrastructure Department currently provides only graph (Neo4j) and vector (Qdrant) backends. This limits the system's ability to handle:

- **Deterministic policy lookups** ("Return penalty calculation for rule X")
- **Ground truth verification** ("Verify constraint Y is satisfied")
- **Exact matches** ("Find record WHERE id = 'abc-123'")
- **Structured aggregations** ("COUNT active users")
- **Transactional operations** (BEGIN; UPDATE; COMMIT)

**Adversarial Recommendation from Verification Department:**
> "Infrastructure should support SQL for precision queries. Current graph-only architecture limits deterministic operations and prevents reliable policy enforcement."

### Solution Overview

Add **hybrid query routing** with two components:

1. **Infrastructure Department: SQL Backend**
   - New MCP tool: `query_sql(sql, params, session_id)`
   - Coexists with existing `query_neo4j` and `query_qdrant`
   - Enforces parameterized queries, prevents SQL injection
   - Fits results in 20k token context budget

2. **Context Department: Intelligent Routing**
   - Query classification algorithm (SQL vs graph vs vector)
   - Thompson Sampling-based backend selection
   - Confidence-driven routing (high → SQL, low → graph exploration)
   - Learns optimal routing via ReflectionBuffer

### Expected Benefits

**Technical:**
- ✅ **Deterministic operations** (exact matches, policy enforcement)
- ✅ **Ground truth verification** (compliance, constraints)
- ✅ **Structured queries** (aggregations, joins, transactions)
- ✅ **Multi-backend optimization** (route to best backend per query)
- ✅ **Self-improving routing** (learns from outcomes via Thompson Sampling)

**Business (Moonshot Integration):**
- ✅ **Beekeeping compliance** (SQL for regulations, graph for relationships)
- ✅ **Healthcare policies** (SQL for rules, vector for similar cases)
- ✅ **Financial ground truth** (SQL for transactions, graph for risk networks)
- ✅ **Manufacturing specs** (SQL for tolerances, graph for dependencies)

**Performance:**
- SQL queries: **~10-50ms** (exact matches)
- Graph queries: **~50-200ms** (relationship traversal)
- Vector queries: **~20-100ms** (similarity search)
- Routing overhead: **~5-15ms** (classification + Thompson sampling)
- Net benefit: **50-80% latency reduction** for precision queries

### Departmental Impact

| Department | Impact | New Responsibilities |
|------------|--------|---------------------|
| **Infrastructure** | ⚠️ High | SQL backend management, query safety |
| **Context** | ⚠️ High | Routing intelligence, backend selection |
| **MasterWeaver** | ✅ None | Still depends on Infrastructure |
| **Verification** | 🔵 Low | Can validate routing decisions |
| **Execution** | ✅ None | No change |
| **Orchestration** | 🔵 Low | Handles routing escalations |

**Key Principle:** Clean departmental boundaries via MCP. Infrastructure owns data access, Context owns routing logic.

---

## Part 1: Infrastructure Department - SQL Backend Design

### 1.1 SQL Engine Recommendation

**Choice: SQLite** (initially), with PostgreSQL migration path

**Justification:**

| Criteria | SQLite | PostgreSQL | DuckDB |
|----------|--------|------------|---------|
| **Deployment** | ✅ Zero-config, embedded | ⚠️ Separate service | ✅ Embedded |
| **ACID** | ✅ Full ACID | ✅ Full ACID | ✅ Full ACID |
| **Context Budget** | ✅ 20k fits well | ✅ 20k fits well | ✅ 20k fits well |
| **Concurrency** | ⚠️ Read-heavy OK | ✅ Excellent | ✅ Good |
| **Production** | ⚠️ Scale limits | ✅ Enterprise-ready | 🔵 Emerging |
| **Integration** | ✅ Python stdlib | ⚠️ Requires psycopg2 | ✅ Python duckdb |
| **Cost** | ✅ Free, no infra | ⚠️ $50-500/mo | ✅ Free, no infra |

**Decision:**
- **Phase 1 (Months 1-3):** SQLite - Fast to deploy, zero infrastructure cost
- **Phase 2 (Months 4-6):** PostgreSQL option - For enterprise customers requiring high concurrency
- **Strategy:** Abstract behind `SQLBackend` protocol - swap engines without changing MCP interface

### 1.2 Schema Design

#### Example Schema: Beekeeping Domain

**Policy Rules Table:**
```sql
CREATE TABLE policy_rules (
    rule_id TEXT PRIMARY KEY,
    rule_name TEXT NOT NULL,
    rule_type TEXT NOT NULL,  -- 'penalty', 'requirement', 'guideline'
    version INTEGER NOT NULL DEFAULT 1,
    effective_date DATE NOT NULL,
    expiration_date DATE,
    rule_logic TEXT NOT NULL,  -- JSON with calculation logic
    confidence REAL DEFAULT 1.0,  -- Ground truth = 1.0
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_policy_type ON policy_rules(rule_type);
CREATE INDEX idx_policy_effective ON policy_rules(effective_date, expiration_date);

-- Example row
INSERT INTO policy_rules VALUES (
    'penalty_calc_v2',
    'Varroa Mite Penalty Calculation',
    'penalty',
    2,
    '2024-01-01',
    NULL,
    '{"base_penalty": 100, "per_hive_multiplier": 1.5, "severity_factor": 2.0}',
    1.0,
    '2024-01-01 00:00:00',
    '2024-01-01 00:00:00'
);
```

**Ground Truth Table:**
```sql
CREATE TABLE ground_truth (
    truth_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,  -- 'inspection', 'hive', 'queen'
    entity_id TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT NOT NULL,
    value_type TEXT NOT NULL,  -- 'string', 'number', 'boolean', 'json'
    source TEXT NOT NULL,  -- 'manual_entry', 'sensor', 'verified_inspection'
    confidence REAL DEFAULT 1.0,
    verified_at TIMESTAMP NOT NULL,
    verified_by TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_truth_entity ON ground_truth(entity_type, entity_id);
CREATE INDEX idx_truth_attribute ON ground_truth(attribute);

-- Example row
INSERT INTO ground_truth VALUES (
    'truth_001',
    'hive',
    'hive_123',
    'queen_present',
    'true',
    'boolean',
    'verified_inspection',
    1.0,
    '2024-10-15 14:30:00',
    'inspector_456',
    '2024-10-15 14:30:00'
);
```

**Constraint Validation Table:**
```sql
CREATE TABLE constraints (
    constraint_id TEXT PRIMARY KEY,
    constraint_name TEXT NOT NULL,
    constraint_type TEXT NOT NULL,  -- 'range', 'enum', 'pattern', 'dependency'
    target_entity TEXT NOT NULL,
    target_attribute TEXT NOT NULL,
    validation_logic TEXT NOT NULL,  -- JSON with validation rules
    error_message TEXT NOT NULL,
    severity TEXT NOT NULL,  -- 'error', 'warning', 'info'
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_constraint_target ON constraints(target_entity, target_attribute);

-- Example row
INSERT INTO constraints VALUES (
    'constraint_001',
    'Hive Count Range',
    'range',
    'apiary',
    'hive_count',
    '{"min": 1, "max": 100}',
    'Hive count must be between 1 and 100',
    'error',
    TRUE,
    '2024-01-01 00:00:00'
);
```

#### Generalized Schema Pattern

**For ANY domain** (healthcare, finance, manufacturing), use:

```sql
-- Domain-agnostic policy table
CREATE TABLE domain_policies (
    policy_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,  -- 'beekeeping', 'healthcare', etc.
    policy_name TEXT NOT NULL,
    policy_data TEXT NOT NULL,  -- JSON with domain-specific structure
    confidence REAL DEFAULT 1.0,
    version INTEGER NOT NULL DEFAULT 1,
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Domain-agnostic ground truth
CREATE TABLE domain_ground_truth (
    truth_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    entity_ref TEXT NOT NULL,  -- Domain-specific entity reference
    truth_data TEXT NOT NULL,  -- JSON with verified facts
    confidence REAL DEFAULT 1.0,
    verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Domain-agnostic constraints
CREATE TABLE domain_constraints (
    constraint_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    constraint_data TEXT NOT NULL,  -- JSON with validation rules
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Why JSON columns?**
- ✅ Domain flexibility (different industries have different structures)
- ✅ Easy to add new domains without schema migration
- ✅ SQLite and PostgreSQL both support JSON operations
- ✅ Fits B2B marketplace model (modular domain packages)

**Blake's Answers:**
1. ✅ **Hybrid approach** - Mix domain-specific tables + JSON flexibility
2. ✅ **Separate but linked** - SQL and Neo4j reference each other via entity IDs (no sync)
3. ✅ **Additional precision tables** - Transaction logs, audit trails, user permissions

#### Additional Precision Tables

**Transaction Logs:**
```sql
CREATE TABLE transaction_logs (
    transaction_id TEXT PRIMARY KEY,
    transaction_type TEXT NOT NULL,  -- 'create', 'update', 'delete', 'query'
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action_data TEXT NOT NULL,  -- JSON with transaction details
    neo4j_node_id TEXT,  -- Link to corresponding Neo4j node (if exists)
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL,  -- 'success', 'failure', 'pending'
    error_message TEXT
);

CREATE INDEX idx_transaction_entity ON transaction_logs(entity_type, entity_id);
CREATE INDEX idx_transaction_user ON transaction_logs(user_id);
CREATE INDEX idx_transaction_time ON transaction_logs(timestamp);
```

**Audit Trails:**
```sql
CREATE TABLE audit_trails (
    audit_id TEXT PRIMARY KEY,
    audit_type TEXT NOT NULL,  -- 'access', 'modification', 'deletion', 'export'
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    ip_address TEXT,
    action_description TEXT NOT NULL,
    before_state TEXT,  -- JSON snapshot before change
    after_state TEXT,   -- JSON snapshot after change
    neo4j_relationship_id TEXT,  -- Link to Neo4j relationship (if tracking)
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    compliance_flag BOOLEAN DEFAULT FALSE  -- For regulatory compliance
);

CREATE INDEX idx_audit_resource ON audit_trails(resource_type, resource_id);
CREATE INDEX idx_audit_user ON audit_trails(user_id);
CREATE INDEX idx_audit_time ON audit_trails(timestamp);
CREATE INDEX idx_audit_compliance ON audit_trails(compliance_flag) WHERE compliance_flag = TRUE;
```

**User Permissions:**
```sql
CREATE TABLE user_permissions (
    permission_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    resource_type TEXT NOT NULL,  -- 'apiary', 'inspection', 'report'
    resource_id TEXT,  -- NULL = all resources of type
    permission_level TEXT NOT NULL,  -- 'read', 'write', 'admin'
    granted_by TEXT NOT NULL,  -- user_id who granted permission
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    neo4j_user_node TEXT  -- Link to Neo4j user node
);

CREATE INDEX idx_permission_user ON user_permissions(user_id);
CREATE INDEX idx_permission_resource ON user_permissions(resource_type, resource_id);
CREATE INDEX idx_permission_active ON user_permissions(active) WHERE active = TRUE;
```

#### Linking SQL ↔ Neo4j (Separate but Linked)

**Pattern:** Store entity IDs in both systems, query appropriately

```sql
-- SQL stores the "ground truth" policy
SELECT * FROM policy_rules WHERE rule_id = 'penalty_calc_v2';
-- Returns: rule_logic = {"base_penalty": 100, ...}

-- Neo4j stores the "relationships" between entities
MATCH (rule:PolicyRule {rule_id: 'penalty_calc_v2'})-[:APPLIES_TO]->(hive:Hive)
RETURN hive.hive_id, hive.inspector_id
-- Returns: List of hives this rule applies to

-- Combined query (Context Department orchestrates):
-- 1. Get policy from SQL (deterministic)
-- 2. Get affected entities from Neo4j (semantic)
-- 3. Merge results
```

**Key Principle:** SQL = "What is the rule?", Neo4j = "What does it relate to?"

### 1.3 File Structure

```
hololoom/
└── infrastructure/
    ├── mcp_server.py                    # EXISTING - MCP server framework
    ├── neo4j_backend.py                 # EXISTING - Graph backend
    ├── qdrant_backend.py                # EXISTING - Vector backend
    ├── sql_backend.py                   # NEW - SQL backend
    ├── sql_migrations/                  # NEW - Schema migrations
    │   ├── 001_initial_schema.sql
    │   ├── 002_add_beekeeping.sql
    │   └── 003_add_healthcare.sql
    └── tests/
        ├── test_sql_backend.py          # NEW - Unit tests
        └── test_sql_mcp.py              # NEW - MCP integration tests
```

### 1.4 MCP Tool Definition

**Pattern:** Follow existing `query_neo4j` and `query_qdrant` MCP tools

**New Tool: `query_sql`**

```python
# hololoom/infrastructure/mcp_server.py

from mcp.server import Server
from mcp.types import Tool, TextContent

# Add to existing MCP server
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # EXISTING
        Tool(
            name="query_neo4j",
            description="Query Neo4j graph database for semantic relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "cypher": {"type": "string"},
                    "session_id": {"type": "string"}
                },
                "required": ["cypher", "session_id"]
            }
        ),
        Tool(
            name="query_qdrant",
            description="Query Qdrant vector database for similarity search",
            inputSchema={
                "type": "object",
                "properties": {
                    "embedding": {"type": "array", "items": {"type": "number"}},
                    "limit": {"type": "integer", "default": 10},
                    "session_id": {"type": "string"}
                },
                "required": ["embedding", "session_id"]
            }
        ),

        # NEW - SQL Backend
        Tool(
            name="query_sql",
            description="Execute SQL query for deterministic, ground truth operations. Use for policy lookups, exact matches, constraint validation, and structured queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL query (parameterized with ? placeholders)"
                    },
                    "params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Query parameters (safe, prevents SQL injection)"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier for tracking"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domain context (beekeeping, healthcare, etc.)",
                        "default": "generic"
                    }
                },
                "required": ["sql", "params", "session_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle MCP tool calls"""

    if name == "query_sql":
        # Delegate to SQL backend
        from .sql_backend import SQLBackend

        backend = SQLBackend()
        result = await backend.execute_query(
            sql=arguments["sql"],
            params=arguments.get("params", []),
            session_id=arguments["session_id"],
            domain=arguments.get("domain", "generic")
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "rows": result.rows,
                "row_count": result.row_count,
                "confidence": result.confidence,
                "execution_time_ms": result.execution_time_ms,
                "query_hash": result.query_hash,
                "cached": result.cached
            })
        )]
```

**Key Features:**
- ✅ **Parameterized queries** (`?` placeholders) - prevents SQL injection
- ✅ **Session tracking** (session_id) - ties to Orchestration's master session
- ✅ **Domain context** - enables domain-specific schemas
- ✅ **Confidence scoring** - 1.0 for ground truth, <1.0 for derived data
- ✅ **Execution timing** - for performance monitoring
- ✅ **Query hashing** - enables caching of common queries

---

## Part 2: Context Department - Routing Logic

### 2.1 Query Classification Algorithm

**Goal:** Determine which backend(s) to query based on query characteristics

**Classification Features:**

```python
@dataclass
class QueryFeatures:
    """Features extracted from query for routing decision"""

    # Lexical features
    has_exact_id: bool           # "rule_id = 'X'" or "WHERE id = ..."
    has_where_clause: bool       # Structured filtering
    has_aggregation: bool        # COUNT, SUM, AVG, etc.
    has_relationship: bool       # "how does X relate to Y"
    has_similarity: bool         # "find similar", "like this"

    # Semantic features
    mentions_policy: bool        # Keywords: policy, rule, regulation
    mentions_ground_truth: bool  # Keywords: verify, check, validate
    mentions_audit: bool         # Keywords: audit, transaction, log
    mentions_permission: bool    # Keywords: permission, access, role

    # Confidence indicators
    requires_determinism: bool   # "exact", "must", "verified"
    allows_approximation: bool   # "similar", "related", "might"

    # Query complexity
    estimated_result_size: str   # 'small' (<10), 'medium' (<100), 'large' (100+)
    multi_hop: bool             # Requires relationship traversal
```

**Classification Decision Tree:**

```python
class QueryClassifier:
    """Classify queries into backend routing categories"""

    def classify(self, query: str) -> BackendSelection:
        """Determine which backend(s) to use"""

        features = self._extract_features(query)

        # Rule 1: Exact ID lookups → SQL
        if features.has_exact_id:
            return BackendSelection(
                backends=['sql'],
                reason='Exact ID match requires SQL precision',
                confidence=0.95
            )

        # Rule 2: Policy/ground truth/audit → SQL
        if any([
            features.mentions_policy,
            features.mentions_ground_truth,
            features.mentions_audit,
            features.mentions_permission
        ]):
            return BackendSelection(
                backends=['sql'],
                reason='Policy/ground truth requires SQL determinism',
                confidence=0.90
            )

        # Rule 3: Aggregations → SQL
        if features.has_aggregation:
            return BackendSelection(
                backends=['sql'],
                reason='Aggregation best served by SQL',
                confidence=0.85
            )

        # Rule 4: Similarity queries → Vector
        if features.has_similarity and not features.requires_determinism:
            return BackendSelection(
                backends=['qdrant'],
                reason='Similarity search optimized for vector DB',
                confidence=0.88
            )

        # Rule 5: Relationship traversal → Graph
        if features.has_relationship or features.multi_hop:
            return BackendSelection(
                backends=['neo4j'],
                reason='Relationship traversal requires graph',
                confidence=0.87
            )

        # Rule 6: Multi-backend (hybrid queries)
        if features.has_exact_id and features.has_relationship:
            # Example: "Get policy X and all entities it applies to"
            return BackendSelection(
                backends=['sql', 'neo4j'],
                reason='Hybrid: SQL for policy, graph for relationships',
                confidence=0.82,
                merge_strategy='sequential'  # SQL first, then graph
            )

        # Rule 7: Exploratory queries → Graph (default)
        if features.allows_approximation:
            return BackendSelection(
                backends=['neo4j'],
                reason='Exploratory query uses graph search',
                confidence=0.70
            )

        # Default: Thompson Sampling decides
        return BackendSelection(
            backends=None,  # Let Thompson Sampling choose
            reason='Unclear classification - defer to learned routing',
            confidence=0.50
        )
```

**Example Classifications:**

| Query | Backend(s) | Reason | Confidence |
|-------|-----------|--------|------------|
| `"Return policy rule 'penalty_calc_v2'"` | SQL | Exact ID match | 0.95 |
| `"What is Thompson Sampling?"` | Neo4j | Exploratory semantic | 0.70 |
| `"Find hives similar to this one"` | Qdrant | Similarity search | 0.88 |
| `"Verify constraint X is satisfied"` | SQL | Ground truth verification | 0.90 |
| `"How does X relate to Y?"` | Neo4j | Relationship traversal | 0.87 |
| `"Get policy X and affected hives"` | SQL + Neo4j | Hybrid (policy + relations) | 0.82 |
| `"COUNT users with permission Y"` | SQL | Aggregation | 0.85 |
| `"Audit trail for entity Z"` | SQL | Audit/transaction log | 0.90 |

### 2.2 Thompson Sampling Integration

**Goal:** Use existing Thompson Sampling bandit (`policy/unified.py`) to learn optimal backend selection

**Thompson Sampling for Backends:**

```python
class BackendBandit:
    """Thompson Sampling for backend selection

    Integrates with existing policy/unified.py ThompsonBandit
    Treats backends as 'tools' in the bandit framework
    """

    def __init__(self):
        from hololoom.policy.unified import ThompsonBandit

        # Initialize bandit with 3 backends as "tools"
        self.bandit = ThompsonBandit(
            n_arms=3,  # sql, neo4j, qdrant
            alpha_prior=1.0,
            beta_prior=1.0
        )

        # Backend mapping
        self.backends = ['sql', 'neo4j', 'qdrant']

        # Track statistics per backend
        self.stats = {
            'sql': {'successes': 0, 'failures': 0, 'avg_latency': 0},
            'neo4j': {'successes': 0, 'failures': 0, 'avg_latency': 0},
            'qdrant': {'successes': 0, 'failures': 0, 'avg_latency': 0}
        }

    def select_backend(self, query_features: QueryFeatures) -> str:
        """Select backend using Thompson Sampling

        If classifier has high confidence (>0.85), use classifier.
        Otherwise, sample from Thompson bandit.
        """

        # Try classification first
        classification = QueryClassifier().classify(query_features.query_text)

        if classification.confidence >= 0.85:
            # High confidence classification - use it
            return classification.backends[0]

        # Low confidence - use Thompson Sampling
        backend_idx = self.bandit.select()
        return self.backends[backend_idx]

    def update(self, backend: str, success: bool, confidence: float, latency_ms: float):
        """Update bandit statistics based on query outcome"""

        backend_idx = self.backends.index(backend)

        # Update Thompson Sampling
        if success and confidence >= 0.75:
            # Success = high confidence result
            self.bandit.update(backend_idx, reward=confidence)
            self.stats[backend]['successes'] += 1
        else:
            # Failure = low confidence or error
            self.bandit.update(backend_idx, reward=0.0)
            self.stats[backend]['failures'] += 1

        # Update latency tracking (exponential moving average)
        alpha = 0.1
        old_latency = self.stats[backend]['avg_latency']
        self.stats[backend]['avg_latency'] = (
            alpha * latency_ms + (1 - alpha) * old_latency
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current backend statistics for monitoring"""
        return {
            'backend_stats': self.stats,
            'bandit_priors': {
                backend: {
                    'alpha': self.bandit.alpha[i],
                    'beta': self.bandit.beta[i],
                    'expected_reward': self.bandit.alpha[i] / (
                        self.bandit.alpha[i] + self.bandit.beta[i]
                    )
                }
                for i, backend in enumerate(self.backends)
            }
        }
```

**Thompson Sampling Update Flow:**

```
Query arrives
    ↓
Classify (if confidence < 0.85, use Thompson Sampling)
    ↓
Execute on selected backend
    ↓
Measure outcome (confidence, latency)
    ↓
Update Thompson Sampling:
  - Success (confidence ≥ 0.75): α ← α + confidence
  - Failure (confidence < 0.75): β ← β + (1 - confidence)
    ↓
Bandit learns: Which backend works best for this query type?
```

### 2.3 Confidence-Based Routing Rules

**Principle:** Higher confidence requirements → more deterministic backend (SQL)

```python
class ConfidenceBasedRouter:
    """Route based on confidence requirements"""

    CONFIDENCE_THRESHOLDS = {
        'critical': 0.95,   # Compliance, safety-critical
        'high': 0.85,       # Policy enforcement
        'medium': 0.70,     # General operations
        'low': 0.50,        # Exploratory queries
        'exploratory': 0.0  # No confidence requirement
    }

    def route_by_confidence_requirement(
        self,
        query: str,
        confidence_required: float
    ) -> BackendSelection:
        """Route based on required confidence level"""

        if confidence_required >= self.CONFIDENCE_THRESHOLDS['critical']:
            # Critical confidence → SQL only
            return BackendSelection(
                backends=['sql'],
                reason='Critical confidence requires SQL ground truth',
                confidence=1.0
            )

        elif confidence_required >= self.CONFIDENCE_THRESHOLDS['high']:
            # High confidence → SQL or SQL+Graph
            features = QueryClassifier()._extract_features(query)
            if features.has_relationship:
                # Need relationships too → SQL then Graph
                return BackendSelection(
                    backends=['sql', 'neo4j'],
                    reason='High confidence policy + relationships',
                    confidence=0.90,
                    merge_strategy='sequential'
                )
            else:
                # Just need facts → SQL only
                return BackendSelection(
                    backends=['sql'],
                    reason='High confidence requires SQL precision',
                    confidence=0.95
                )

        elif confidence_required >= self.CONFIDENCE_THRESHOLDS['medium']:
            # Medium confidence → Use classifier
            features = QueryClassifier()._extract_features(query)
            return QueryClassifier().classify(query)

        else:
            # Low/exploratory → Graph or Vector (exploration-friendly)
            features = QueryClassifier()._extract_features(query)
            if features.has_similarity:
                return BackendSelection(
                    backends=['qdrant'],
                    reason='Exploratory similarity search',
                    confidence=0.70
                )
            else:
                return BackendSelection(
                    backends=['neo4j'],
                    reason='Exploratory graph traversal',
                    confidence=0.65
                )
```

**Confidence Routing Table:**

| Required Confidence | Backend(s) | Reasoning |
|-------------------|-----------|-----------|
| **0.95+ (Critical)** | SQL only | Ground truth, compliance, safety-critical |
| **0.85-0.94 (High)** | SQL or SQL+Graph | Policy enforcement, verification |
| **0.70-0.84 (Medium)** | Classifier decides | Standard operations |
| **0.50-0.69 (Low)** | Graph or Vector | General queries, some uncertainty OK |
| **<0.50 (Exploratory)** | Graph or Vector | Discovery, brainstorming, exploration |

### 2.4 Multi-Backend Query Patterns

**Pattern 1: Sequential (SQL → Graph)**
```python
async def sequential_query(self, policy_id: str) -> CombinedResult:
    """Get policy from SQL, then affected entities from Graph"""

    # Step 1: Get policy (SQL)
    sql_result = await self.infrastructure.query_sql(
        sql="SELECT * FROM policy_rules WHERE rule_id = ?",
        params=[policy_id],
        session_id=self.session_id
    )

    # Step 2: Get affected entities (Graph)
    neo4j_result = await self.infrastructure.query_neo4j(
        cypher="""
            MATCH (rule:PolicyRule {rule_id: $rule_id})-[:APPLIES_TO]->(entity)
            RETURN entity.entity_id, entity.entity_type
        """,
        params={'rule_id': policy_id},
        session_id=self.session_id
    )

    # Step 3: Merge results
    return CombinedResult(
        policy=sql_result,
        affected_entities=neo4j_result,
        merge_strategy='sequential',
        total_latency_ms=sql_result.latency_ms + neo4j_result.latency_ms
    )
```

**Pattern 2: Parallel (SQL || Graph)**
```python
async def parallel_query(self, query: str) -> CombinedResult:
    """Query SQL and Graph in parallel, merge results"""

    # Execute in parallel
    sql_task = self.infrastructure.query_sql(
        sql="SELECT * FROM ground_truth WHERE attribute = ?",
        params=['queen_present'],
        session_id=self.session_id
    )

    neo4j_task = self.infrastructure.query_neo4j(
        cypher="MATCH (h:Hive)-[:HAS_QUEEN]->(q:Queen) RETURN h, q",
        session_id=self.session_id
    )

    sql_result, neo4j_result = await asyncio.gather(sql_task, neo4j_task)

    # Merge: SQL is ground truth, Graph is contextual
    return CombinedResult(
        ground_truth=sql_result,
        relationships=neo4j_result,
        merge_strategy='parallel',
        total_latency_ms=max(sql_result.latency_ms, neo4j_result.latency_ms)
    )
```

**Pattern 3: Fallback (SQL → Graph if SQL empty)**
```python
async def fallback_query(self, entity_id: str) -> Result:
    """Try SQL first, fallback to Graph if not found"""

    # Try SQL first (precise lookup)
    sql_result = await self.infrastructure.query_sql(
        sql="SELECT * FROM ground_truth WHERE entity_id = ?",
        params=[entity_id],
        session_id=self.session_id
    )

    if sql_result.row_count > 0:
        # Found in SQL - use it (high confidence)
        return Result(
            data=sql_result,
            backend='sql',
            confidence=1.0
        )

    # Not in SQL - fallback to Graph (lower confidence)
    neo4j_result = await self.infrastructure.query_neo4j(
        cypher="MATCH (e {entity_id: $id}) RETURN e",
        params={'id': entity_id},
        session_id=self.session_id
    )

    return Result(
        data=neo4j_result,
        backend='neo4j',
        confidence=0.75,
        fallback_used=True
    )
```

**Pattern 4: Verification (SQL validates Graph)**
```python
async def verified_query(self, hive_id: str) -> VerifiedResult:
    """Get data from Graph, verify against SQL ground truth"""

    # Get from Graph
    graph_data = await self.infrastructure.query_neo4j(
        cypher="MATCH (h:Hive {hive_id: $id}) RETURN h",
        params={'id': hive_id},
        session_id=self.session_id
    )

    # Verify critical attributes against SQL ground truth
    sql_truth = await self.infrastructure.query_sql(
        sql="SELECT * FROM ground_truth WHERE entity_id = ?",
        params=[hive_id],
        session_id=self.session_id
    )

    # Compare and flag discrepancies
    discrepancies = self._compare_graph_vs_sql(graph_data, sql_truth)

    return VerifiedResult(
        data=graph_data,
        verified=len(discrepancies) == 0,
        discrepancies=discrepancies,
        confidence=1.0 if len(discrepancies) == 0 else 0.60
    )
```

### 2.5 Integration with WeavingOrchestrator

**File Location:** `hololoom/context/query_router.py` (NEW)

**Integration Point:** Modify `WeavingOrchestrator.weave()`

```python
# hololoom/context/query_router.py

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from hololoom.documentation.types import Query, Spacetime

@dataclass
class RoutingDecision:
    """Backend routing decision"""
    backends: List[str]  # ['sql'], ['neo4j'], ['qdrant'], or combinations
    reason: str
    confidence: float
    merge_strategy: Optional[str] = None  # 'sequential', 'parallel', 'fallback'
    thompson_sampling_used: bool = False

class QueryRouter:
    """Intelligent query routing for Context Department"""

    def __init__(self, infrastructure_mcp_client):
        self.infrastructure = infrastructure_mcp_client
        self.classifier = QueryClassifier()
        self.backend_bandit = BackendBandit()
        self.confidence_router = ConfidenceBasedRouter()

    async def route_and_execute(
        self,
        query: Query,
        confidence_required: float = 0.70,
        session_id: str = None
    ) -> Spacetime:
        """
        Main routing logic - called by WeavingOrchestrator

        1. Classify query
        2. Route to backend(s)
        3. Execute
        4. Learn from outcome
        """

        # Step 1: Determine routing
        if confidence_required >= 0.85:
            # High confidence requirement → confidence-based routing
            routing = self.confidence_router.route_by_confidence_requirement(
                query.text,
                confidence_required
            )
        else:
            # Standard routing → classify + Thompson Sampling
            routing = self._route_with_learning(query.text)

        # Step 2: Execute query on selected backend(s)
        result = await self._execute_routing(routing, query, session_id)

        # Step 3: Learn from outcome
        await self._learn_from_outcome(routing, result)

        # Step 4: Convert to Spacetime (Context Department's output format)
        spacetime = self._result_to_spacetime(result, query, routing)

        return spacetime

    def _route_with_learning(self, query_text: str) -> RoutingDecision:
        """Classify query and potentially use Thompson Sampling"""

        # Extract features
        features = self.classifier._extract_features(query_text)

        # Classify
        classification = self.classifier.classify(query_text)

        # If low confidence classification, use Thompson Sampling
        if classification.confidence < 0.75:
            backend = self.backend_bandit.select_backend(features)
            return RoutingDecision(
                backends=[backend],
                reason=f'Thompson Sampling selected {backend}',
                confidence=0.50,
                thompson_sampling_used=True
            )
        else:
            return RoutingDecision(
                backends=classification.backends,
                reason=classification.reason,
                confidence=classification.confidence,
                merge_strategy=classification.merge_strategy,
                thompson_sampling_used=False
            )

    async def _execute_routing(
        self,
        routing: RoutingDecision,
        query: Query,
        session_id: str
    ) -> Dict:
        """Execute query on selected backend(s)"""

        if len(routing.backends) == 1:
            # Single backend
            backend = routing.backends[0]
            return await self._query_single_backend(backend, query, session_id)

        else:
            # Multi-backend
            if routing.merge_strategy == 'sequential':
                return await self._query_sequential(routing.backends, query, session_id)
            elif routing.merge_strategy == 'parallel':
                return await self._query_parallel(routing.backends, query, session_id)
            elif routing.merge_strategy == 'fallback':
                return await self._query_fallback(routing.backends, query, session_id)
            else:
                # Default: sequential
                return await self._query_sequential(routing.backends, query, session_id)

    async def _query_single_backend(
        self,
        backend: str,
        query: Query,
        session_id: str
    ) -> Dict:
        """Query a single backend via Infrastructure MCP"""

        if backend == 'sql':
            # TODO: Parse query.text into SQL
            # For now, assume query.text contains SQL-ready text
            return await self.infrastructure.query_sql(
                sql=self._parse_to_sql(query.text),
                params=[],
                session_id=session_id
            )

        elif backend == 'neo4j':
            # Use existing graph search
            return await self.infrastructure.query_neo4j(
                cypher=self._parse_to_cypher(query.text),
                session_id=session_id
            )

        elif backend == 'qdrant':
            # Use existing vector search
            embedding = await self._get_embedding(query.text)
            return await self.infrastructure.query_qdrant(
                embedding=embedding,
                session_id=session_id
            )

    async def _learn_from_outcome(
        self,
        routing: RoutingDecision,
        result: Dict
    ):
        """Update Thompson Sampling based on query outcome"""

        if not routing.thompson_sampling_used:
            # Only learn when Thompson Sampling was used
            return

        backend = routing.backends[0]

        # Determine success
        success = (
            result.get('row_count', 0) > 0 and
            result.get('confidence', 0) >= 0.75
        )

        # Update bandit
        self.backend_bandit.update(
            backend=backend,
            success=success,
            confidence=result.get('confidence', 0.50),
            latency_ms=result.get('execution_time_ms', 0)
        )
```

**Modify WeavingOrchestrator:**

```python
# hololoom/context/weaving_orchestrator.py

class WeavingOrchestrator:
    """Existing orchestrator - ADD routing integration"""

    def __init__(self, cfg: Config, shards: List[MemoryShard] = None):
        # EXISTING initialization
        self.cfg = cfg
        self.shards = shards or []
        # ... existing setup ...

        # NEW: Add query router
        from hololoom.context.query_router import QueryRouter
        self.query_router = QueryRouter(infrastructure_mcp_client=self._get_infrastructure_client())
        self.enable_routing = cfg.enable_hybrid_routing  # NEW config flag

    async def weave(self, query: Query) -> Spacetime:
        """Main weaving cycle - MODIFIED to support routing"""

        if self.enable_routing:
            # NEW PATH: Route query intelligently
            confidence_required = self._estimate_confidence_requirement(query)
            spacetime = await self.query_router.route_and_execute(
                query=query,
                confidence_required=confidence_required,
                session_id=self.session_id
            )
            return spacetime

        else:
            # EXISTING PATH: Graph-only processing
            # ... existing weave() logic ...
            pass
```

**Context Budget Management:**

```python
class ContextBudgetManager:
    """Ensure routing + results fit in 60k token budget"""

    BUDGET_ALLOCATION = {
        'routing_overhead': 5000,    # Routing logic, classification
        'sql_results': 15000,        # SQL query results
        'graph_results': 20000,      # Graph query results
        'vector_results': 10000,     # Vector query results
        'merge_overhead': 5000,      # Result merging
        'metadata': 5000             # Confidence, timing, etc.
    }

    def check_budget(self, results: Dict) -> bool:
        """Check if results fit in 60k token budget"""
        total_tokens = sum([
            self._estimate_tokens(results.get('sql', '')),
            self._estimate_tokens(results.get('graph', '')),
            self._estimate_tokens(results.get('vector', '')),
            self.BUDGET_ALLOCATION['routing_overhead'],
            self.BUDGET_ALLOCATION['merge_overhead'],
            self.BUDGET_ALLOCATION['metadata']
        ])

        return total_tokens <= 60000

    def compact_if_needed(self, results: Dict) -> Dict:
        """Compact results to fit budget"""
        if self.check_budget(results):
            return results

        # Compact strategy: Truncate longest result
        # Priority: Keep SQL (ground truth) > Graph > Vector
        # ... compaction logic ...
```

---

**END OF SECTION 2**

---

## Section 3: MCP Protocol + Learning Mechanism

**Department Ownership**: Orchestration (MCP coordination), Context (learning integration)
**Dependencies**: Infrastructure (MCP server), Context (ReflectionBuffer)
**Context Budget**: 10k tokens (protocol overhead)

This section defines how departments communicate via MCP and how the system learns from routing outcomes.

---

### 3.1 MCP Request/Response Format

**Model Context Protocol (MCP)** is the inter-department communication standard. Each department exposes tools via MCP servers.

**Infrastructure Department MCP Server:**

```python
# hololoom/infrastructure/mcp_server.py

from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("infrastructure-department")

@server.list_tools()
async def list_tools():
    """Expose SQL and Neo4j backends as MCP tools"""
    return [
        Tool(
            name="query_sql",
            description="Execute SQL query for deterministic, ground truth operations",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL query (parameterized with ? placeholders)"
                    },
                    "params": {
                        "type": "array",
                        "description": "Query parameters (safe, prevents SQL injection)"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier for tracking"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domain context (beekeeping, healthcare, etc.)"
                    },
                    "confidence_required": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0)"
                    }
                },
                "required": ["sql", "session_id"]
            }
        ),
        Tool(
            name="query_neo4j",
            description="Execute Cypher query for semantic relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "cypher": {
                        "type": "string",
                        "description": "Cypher query"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum traversal depth (default: 3)"
                    }
                },
                "required": ["cypher", "session_id"]
            }
        ),
        Tool(
            name="query_qdrant",
            description="Execute vector similarity search",
            inputSchema={
                "type": "object",
                "properties": {
                    "embedding": {
                        "type": "array",
                        "description": "Query embedding vector"
                    },
                    "collection": {
                        "type": "string",
                        "description": "Qdrant collection name"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 10)"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier"
                    }
                },
                "required": ["embedding", "session_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute backend query and return results"""

    if name == "query_sql":
        return await _execute_sql(
            sql=arguments["sql"],
            params=arguments.get("params", []),
            session_id=arguments["session_id"],
            domain=arguments.get("domain", "default"),
            confidence_required=arguments.get("confidence_required", 0.0)
        )

    elif name == "query_neo4j":
        return await _execute_neo4j(
            cypher=arguments["cypher"],
            params=arguments.get("params", {}),
            session_id=arguments["session_id"],
            max_depth=arguments.get("max_depth", 3)
        )

    elif name == "query_qdrant":
        return await _execute_qdrant(
            embedding=arguments["embedding"],
            collection=arguments["collection"],
            top_k=arguments.get("top_k", 10),
            session_id=arguments["session_id"]
        )

    else:
        raise ValueError(f"Unknown tool: {name}")
```

**MCP Response Format:**

All MCP tool calls return a standardized response structure:

```python
from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass
class MCPResponse:
    """Standardized response from Infrastructure Department"""

    # Core fields
    backend: str              # "sql", "neo4j", "qdrant"
    session_id: str
    success: bool

    # Data
    results: List[Any]        # Query results
    result_count: int

    # Metadata
    latency_ms: float
    confidence: float         # 0.0-1.0 (SQL=1.0, Graph=0.7-0.9, Vector=0.6-0.85)
    tokens_used: int          # Estimated token count

    # Error handling
    error: Optional[str] = None
    fallback_used: bool = False

    # Learning signals
    cache_hit: bool = False
    query_complexity: str = "simple"  # simple, moderate, complex

# Example response
response = MCPResponse(
    backend="sql",
    session_id="session_123",
    success=True,
    results=[
        {"rule_id": "bee_001", "rule_name": "Varroa Treatment Schedule", "confidence": 1.0}
    ],
    result_count=1,
    latency_ms=12.5,
    confidence=1.0,
    tokens_used=150,
    cache_hit=False,
    query_complexity="simple"
)
```

---

### 3.2 Session ID Propagation

**Session IDs** enable tracking queries across departmental boundaries for learning and debugging.

**Session ID Structure:**

```
session_{timestamp}_{user_id}_{sequence}

Example: session_1698595200_user_blake_001
```

**Propagation Flow:**

```
User Query (session_123)
    ↓
Orchestration Department (receives session_123)
    ↓ MCP call with session_id=session_123
Context Department (routing decision)
    ↓ MCP call with session_id=session_123
Infrastructure Department (SQL/Graph/Vector execution)
    ↓ Results tagged with session_id=session_123
Context Department (result merging)
    ↓ Spacetime tagged with session_id=session_123
ReflectionBuffer (learning from session_123)
```

**Implementation:**

```python
# hololoom/context/query_router.py

class QueryRouter:
    async def route_and_execute(
        self,
        query: Query,
        confidence_required: float,
        session_id: str  # ← Session ID passed from orchestrator
    ) -> Spacetime:
        """Route query and execute - propagates session_id"""

        # Classification
        backend_selection = self.classifier.classify(query.text)

        # Execute with session_id
        if backend_selection.backend == "sql":
            response = await self.infrastructure_client.call_tool(
                name="query_sql",
                arguments={
                    "sql": backend_selection.sql_query,
                    "params": backend_selection.params,
                    "session_id": session_id,  # ← Propagated
                    "confidence_required": confidence_required
                }
            )

        # Store routing decision with session_id for learning
        await self.learning_tracker.record_routing(
            session_id=session_id,
            query=query.text,
            backend=backend_selection.backend,
            predicted_confidence=backend_selection.confidence,
            actual_confidence=response.confidence,
            latency_ms=response.latency_ms
        )

        return self._build_spacetime(response, query, session_id)
```

**Benefits of Session ID Propagation:**

1. **End-to-end tracing**: See complete query path across departments
2. **Learning signals**: Track routing decisions → outcomes
3. **Debugging**: Reproduce exact query execution
4. **Audit compliance**: Full provenance for regulated industries
5. **Performance analysis**: Measure departmental latencies

---

### 3.3 Error Escalation Path

**Escalation Hierarchy:**

```
Infrastructure (data access errors)
    ↓ escalate via MCP error response
Context (routing errors, confidence failures)
    ↓ escalate via MCP error response
Orchestration (user-facing errors)
```

**Error Types and Escalation:**

| Error Type | Department | Escalation | Fallback |
|------------|------------|------------|----------|
| SQL syntax error | Infrastructure | → Context | Try Graph |
| Database unavailable | Infrastructure | → Context | INMEMORY fallback |
| Classification ambiguous | Context | → Orchestration | Thompson Sampling |
| Confidence too low | Context | → Orchestration | Refinement loop |
| Token budget exceeded | Context | → Orchestration | Compact results |
| MCP timeout | Any | → Orchestration | Circuit breaker |

**Implementation:**

```python
# hololoom/infrastructure/sql_backend.py

class SQLBackend:
    async def execute(
        self,
        sql: str,
        params: List[Any],
        session_id: str,
        confidence_required: float
    ) -> MCPResponse:
        """Execute SQL with error handling"""

        try:
            # Attempt query
            results = await self.db.execute(sql, params)

            return MCPResponse(
                backend="sql",
                session_id=session_id,
                success=True,
                results=results,
                result_count=len(results),
                latency_ms=self._get_latency(),
                confidence=1.0  # SQL = deterministic
            )

        except sqlite3.OperationalError as e:
            # SQL error - escalate to Context for fallback
            logger.warning(f"SQL error for session {session_id}: {e}")

            return MCPResponse(
                backend="sql",
                session_id=session_id,
                success=False,
                results=[],
                result_count=0,
                latency_ms=self._get_latency(),
                confidence=0.0,
                error=f"SQL execution failed: {str(e)}"
            )

        except Exception as e:
            # Unexpected error - escalate to Orchestration
            logger.error(f"Unexpected SQL error for session {session_id}: {e}")
            raise  # Re-raise for higher-level handling
```

```python
# hololoom/context/query_router.py

class QueryRouter:
    async def route_and_execute(
        self,
        query: Query,
        confidence_required: float,
        session_id: str
    ) -> Spacetime:
        """Route with fallback on errors"""

        backend_selection = self.classifier.classify(query.text)

        # Try primary backend
        response = await self._execute_backend(
            backend=backend_selection.backend,
            query=query,
            session_id=session_id
        )

        if not response.success:
            # PRIMARY FAILED - Try fallback
            logger.warning(f"Primary backend {backend_selection.backend} failed, trying fallback")

            fallback_backend = self._get_fallback_backend(backend_selection.backend)
            response = await self._execute_backend(
                backend=fallback_backend,
                query=query,
                session_id=session_id
            )
            response.fallback_used = True

        if not response.success:
            # FALLBACK ALSO FAILED - Escalate to Orchestration
            raise QueryExecutionError(
                f"Both primary and fallback backends failed for session {session_id}",
                session_id=session_id,
                primary_backend=backend_selection.backend,
                fallback_backend=fallback_backend,
                error=response.error
            )

        return self._build_spacetime(response, query, session_id)

    def _get_fallback_backend(self, primary: str) -> str:
        """Fallback strategy: SQL → Graph, Graph → Vector, Vector → Graph"""
        FALLBACK_MAP = {
            "sql": "neo4j",      # SQL fails → try Graph
            "neo4j": "qdrant",   # Graph fails → try Vector
            "qdrant": "neo4j"    # Vector fails → try Graph
        }
        return FALLBACK_MAP.get(primary, "neo4j")
```

**Error Response to User:**

```python
# hololoom/orchestration/orchestrator.py

class WeavingOrchestrator:
    async def weave(self, query: Query) -> Spacetime:
        try:
            spacetime = await self.query_router.route_and_execute(
                query=query,
                confidence_required=self.cfg.confidence_threshold,
                session_id=self.session_id
            )
            return spacetime

        except QueryExecutionError as e:
            # User-facing error message
            return Spacetime(
                response=f"I encountered an issue processing your query: {e.error}. "
                         f"Both primary ({e.primary_backend}) and fallback ({e.fallback_backend}) "
                         f"backends failed. Please try rephrasing or contact support with "
                         f"session ID: {e.session_id}",
                confidence=0.0,
                metadata={
                    "error": True,
                    "session_id": e.session_id,
                    "primary_backend": e.primary_backend,
                    "fallback_backend": e.fallback_backend
                }
            )
```

---

### 3.4 ReflectionBuffer Integration

**ReflectionBuffer** (`hololoom/reflection/buffer.py`) learns from routing outcomes to improve future decisions.

**Learning Signals:**

| Signal | Source | Purpose |
|--------|--------|---------|
| Predicted confidence | QueryClassifier | Expected quality |
| Actual confidence | MCPResponse | Measured quality |
| Latency | MCPResponse | Backend performance |
| Cache hit | MCPResponse | Cache effectiveness |
| Fallback used | QueryRouter | Routing accuracy |
| User feedback | Spacetime | Ground truth |

**Integration:**

```python
# hololoom/context/learning_tracker.py

from hololoom.reflection.buffer import ReflectionBuffer

class LearningTracker:
    """Tracks routing decisions for learning"""

    def __init__(self, reflection_buffer: ReflectionBuffer):
        self.buffer = reflection_buffer
        self.routing_history = []

    async def record_routing(
        self,
        session_id: str,
        query: str,
        backend: str,
        predicted_confidence: float,
        actual_confidence: float,
        latency_ms: float,
        cache_hit: bool = False,
        fallback_used: bool = False
    ):
        """Record routing decision for learning"""

        routing_event = {
            "session_id": session_id,
            "query": query,
            "backend": backend,
            "predicted_confidence": predicted_confidence,
            "actual_confidence": actual_confidence,
            "confidence_error": abs(predicted_confidence - actual_confidence),
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "fallback_used": fallback_used,
            "timestamp": time.time()
        }

        self.routing_history.append(routing_event)

        # Store in ReflectionBuffer for long-term learning
        await self.buffer.store(
            spacetime=None,  # No full spacetime yet
            feedback={
                "routing": routing_event,
                "success": not fallback_used and actual_confidence >= 0.75
            }
        )

    def get_recent_performance(self, backend: str, window: int = 100) -> dict:
        """Get recent performance stats for a backend"""
        recent = [e for e in self.routing_history[-window:] if e["backend"] == backend]

        if not recent:
            return {"count": 0, "avg_confidence": 0.5, "avg_latency": 100.0}

        return {
            "count": len(recent),
            "avg_confidence": np.mean([e["actual_confidence"] for e in recent]),
            "avg_latency": np.mean([e["latency_ms"] for e in recent]),
            "fallback_rate": np.mean([e["fallback_used"] for e in recent]),
            "confidence_calibration": np.mean([e["confidence_error"] for e in recent])
        }
```

**ReflectionBuffer Usage:**

```python
# hololoom/context/query_router.py

class QueryRouter:
    def __init__(self, infrastructure_mcp_client, reflection_buffer: ReflectionBuffer):
        self.infrastructure_client = infrastructure_mcp_client
        self.learning_tracker = LearningTracker(reflection_buffer)
        self.classifier = QueryClassifier()
        self.backend_bandit = BackendBandit()

    async def route_and_execute(
        self,
        query: Query,
        confidence_required: float,
        session_id: str
    ) -> Spacetime:
        """Route with learning"""

        # Classify
        backend_selection = self.classifier.classify(query.text)

        # Execute
        response = await self._execute_backend(
            backend=backend_selection.backend,
            query=query,
            session_id=session_id
        )

        # Learn from outcome
        await self.learning_tracker.record_routing(
            session_id=session_id,
            query=query.text,
            backend=backend_selection.backend,
            predicted_confidence=backend_selection.confidence,
            actual_confidence=response.confidence,
            latency_ms=response.latency_ms,
            cache_hit=response.cache_hit,
            fallback_used=response.fallback_used
        )

        # Update Thompson Sampling bandit
        self.backend_bandit.update(
            backend=backend_selection.backend,
            success=(response.confidence >= 0.75 and not response.fallback_used),
            confidence=response.confidence,
            latency_ms=response.latency_ms
        )

        return self._build_spacetime(response, query, session_id)
```

---

### 3.5 Confidence Calibration

**Confidence calibration** ensures predicted confidence matches actual quality.

**Calibration Metrics:**

```python
# hololoom/context/calibration.py

class ConfidenceCalibrator:
    """Calibrates confidence predictions vs. actual outcomes"""

    def __init__(self):
        self.calibration_history = []

    def add_observation(
        self,
        predicted_confidence: float,
        actual_confidence: float,
        backend: str
    ):
        """Record prediction vs. outcome"""
        self.calibration_history.append({
            "predicted": predicted_confidence,
            "actual": actual_confidence,
            "backend": backend,
            "error": abs(predicted_confidence - actual_confidence)
        })

    def get_calibration_curve(self, backend: str = None) -> dict:
        """Compute calibration curve (predicted vs. actual)"""

        history = self.calibration_history
        if backend:
            history = [h for h in history if h["backend"] == backend]

        if len(history) < 10:
            return {"calibrated": False, "reason": "insufficient_data"}

        # Bin predictions into deciles
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        binned_actual = []
        for i in range(10):
            in_bin = [h for h in history if bins[i] <= h["predicted"] < bins[i+1]]
            if in_bin:
                binned_actual.append(np.mean([h["actual"] for h in in_bin]))
            else:
                binned_actual.append(np.nan)

        # Expected Calibration Error (ECE)
        ece = np.nanmean([
            abs(pred - actual)
            for pred, actual in zip(bin_centers, binned_actual)
            if not np.isnan(actual)
        ])

        return {
            "calibrated": ece < 0.1,  # Well-calibrated if ECE < 0.1
            "ece": ece,
            "bin_centers": bin_centers.tolist(),
            "binned_actual": [x if not np.isnan(x) else None for x in binned_actual],
            "sample_size": len(history)
        }

    def get_adjustment_factor(self, predicted: float, backend: str) -> float:
        """Get calibration adjustment for a prediction"""

        calibration = self.get_calibration_curve(backend)

        if not calibration["calibrated"]:
            return 1.0  # No adjustment if not calibrated

        # Find bin
        bin_idx = int(predicted * 10)
        if bin_idx >= 10:
            bin_idx = 9

        actual = calibration["binned_actual"][bin_idx]
        if actual is None:
            return 1.0

        # Adjustment factor: actual / predicted
        return actual / predicted if predicted > 0 else 1.0
```

**Using Calibration:**

```python
# hololoom/context/query_router.py

class QueryRouter:
    def __init__(self, infrastructure_mcp_client, reflection_buffer: ReflectionBuffer):
        self.calibrator = ConfidenceCalibrator()
        # ... other initialization ...

    async def route_and_execute(
        self,
        query: Query,
        confidence_required: float,
        session_id: str
    ) -> Spacetime:
        """Route with calibrated confidence"""

        # Classify
        backend_selection = self.classifier.classify(query.text)

        # Calibrate predicted confidence
        adjustment = self.calibrator.get_adjustment_factor(
            predicted=backend_selection.confidence,
            backend=backend_selection.backend
        )
        calibrated_confidence = backend_selection.confidence * adjustment

        logger.info(
            f"Calibrated confidence: {backend_selection.confidence:.3f} "
            f"→ {calibrated_confidence:.3f} (adjustment: {adjustment:.3f})"
        )

        # Execute
        response = await self._execute_backend(
            backend=backend_selection.backend,
            query=query,
            session_id=session_id
        )

        # Update calibration
        self.calibrator.add_observation(
            predicted_confidence=backend_selection.confidence,
            actual_confidence=response.confidence,
            backend=backend_selection.backend
        )

        # Build spacetime with calibrated confidence
        spacetime = self._build_spacetime(response, query, session_id)
        spacetime.metadata["predicted_confidence"] = backend_selection.confidence
        spacetime.metadata["calibrated_confidence"] = calibrated_confidence
        spacetime.metadata["calibration_adjustment"] = adjustment

        return spacetime
```

---

### 3.6 Strategy Update Logic

**Strategy updates** adjust routing behavior based on learning signals.

**Update Triggers:**

| Trigger | Condition | Action |
|---------|-----------|--------|
| Calibration drift | ECE > 0.15 | Retrain classifier |
| Backend performance change | Latency +50% | Adjust routing weights |
| High fallback rate | >20% fallbacks | Investigate backend issues |
| Low confidence | Avg confidence < 0.70 | Enable refinement |
| Thompson Sampling convergence | Confidence interval < 0.1 | Reduce exploration |

**Implementation:**

```python
# hololoom/context/strategy_updater.py

class StrategyUpdater:
    """Adjusts routing strategy based on learning signals"""

    def __init__(
        self,
        query_router: QueryRouter,
        update_interval: float = 3600.0  # Update every hour
    ):
        self.router = query_router
        self.update_interval = update_interval
        self.last_update = time.time()

    async def update_if_needed(self):
        """Check if strategy update needed"""

        if time.time() - self.last_update < self.update_interval:
            return  # Not time yet

        logger.info("Running strategy update...")

        # Get recent performance
        sql_perf = self.router.learning_tracker.get_recent_performance("sql")
        neo4j_perf = self.router.learning_tracker.get_recent_performance("neo4j")
        qdrant_perf = self.router.learning_tracker.get_recent_performance("qdrant")

        # Check calibration
        sql_cal = self.router.calibrator.get_calibration_curve("sql")
        neo4j_cal = self.router.calibrator.get_calibration_curve("neo4j")
        qdrant_cal = self.router.calibrator.get_calibration_curve("qdrant")

        # Update 1: Adjust routing weights if performance changed
        if sql_perf["avg_latency"] > 100.0:  # SQL slow
            logger.warning(f"SQL latency high ({sql_perf['avg_latency']:.1f}ms), reducing weight")
            self.router.classifier.adjust_backend_weight("sql", multiplier=0.8)

        if neo4j_perf["fallback_rate"] > 0.2:  # Neo4j unreliable
            logger.warning(f"Neo4j fallback rate high ({neo4j_perf['fallback_rate']:.1%}), reducing weight")
            self.router.classifier.adjust_backend_weight("neo4j", multiplier=0.7)

        # Update 2: Recalibrate if drift detected
        if sql_cal.get("calibrated") and sql_cal["ece"] > 0.15:
            logger.warning(f"SQL calibration drift detected (ECE={sql_cal['ece']:.3f}), retraining")
            # Retrain classifier (future work - requires labeled dataset)
            pass

        # Update 3: Adjust Thompson Sampling exploration
        bandit_stats = self.router.backend_bandit.bandit.get_stats()
        if all(tool["confidence_width"] < 0.1 for tool in bandit_stats):
            logger.info("Thompson Sampling converged, reducing exploration")
            # Reduce epsilon in epsilon-greedy (if using hybrid strategy)
            pass

        # Update 4: Enable refinement if quality low
        avg_confidence = np.mean([
            sql_perf["avg_confidence"],
            neo4j_perf["avg_confidence"],
            qdrant_perf["avg_confidence"]
        ])

        if avg_confidence < 0.70:
            logger.warning(f"Average confidence low ({avg_confidence:.2f}), enabling refinement")
            self.router.enable_refinement = True
        else:
            self.router.enable_refinement = False

        self.last_update = time.time()
        logger.info("Strategy update complete")
```

**QueryRouter Integration:**

```python
# hololoom/context/query_router.py

class QueryRouter:
    def __init__(self, infrastructure_mcp_client, reflection_buffer: ReflectionBuffer):
        # ... existing initialization ...
        self.strategy_updater = StrategyUpdater(self, update_interval=3600.0)
        self.enable_refinement = False

    async def route_and_execute(
        self,
        query: Query,
        confidence_required: float,
        session_id: str
    ) -> Spacetime:
        """Route with strategy updates"""

        # Check if strategy update needed
        await self.strategy_updater.update_if_needed()

        # Classify and execute
        backend_selection = self.classifier.classify(query.text)
        response = await self._execute_backend(
            backend=backend_selection.backend,
            query=query,
            session_id=session_id
        )

        # Learn from outcome
        await self.learning_tracker.record_routing(...)
        self.backend_bandit.update(...)
        self.calibrator.add_observation(...)

        # Build spacetime
        spacetime = self._build_spacetime(response, query, session_id)

        # Refinement if enabled and confidence low
        if self.enable_refinement and spacetime.confidence < 0.75:
            logger.info("Confidence low, triggering refinement")
            spacetime = await self._refine(spacetime, query, session_id)

        return spacetime

    async def _refine(
        self,
        initial_spacetime: Spacetime,
        query: Query,
        session_id: str
    ) -> Spacetime:
        """Refine low-confidence result via multi-backend fusion"""

        # Try parallel execution across all backends
        results = await asyncio.gather(
            self._execute_backend("sql", query, session_id),
            self._execute_backend("neo4j", query, session_id),
            self._execute_backend("qdrant", query, session_id)
        )

        # Merge results (weighted by confidence)
        merged = self._merge_results(results, weights="confidence")

        refined_spacetime = self._build_spacetime(merged, query, session_id)
        refined_spacetime.metadata["refinement"] = True
        refined_spacetime.metadata["initial_confidence"] = initial_spacetime.confidence

        logger.info(
            f"Refinement complete: {initial_spacetime.confidence:.2f} "
            f"→ {refined_spacetime.confidence:.2f}"
        )

        return refined_spacetime
```

---

**END OF SECTION 3**

---

## Section 4: Implementation Plan + Code Examples

**Target Timeline**: 6 weeks (3 phases × 2 weeks each)
**Risk Level**: Medium (new architecture, multi-backend coordination)
**Success Metrics**: 90% routing accuracy, <50ms routing overhead, 0.85+ avg confidence

This section provides a concrete 3-phase rollout plan with testing strategy, example scenarios, and monitoring.

---

### 4.1 Three-Phase Rollout

**Phase 1: Foundation (Weeks 1-2)**
**Goal**: Basic SQL backend + rule-based routing (no learning yet)
**Risk**: Low (isolated infrastructure changes)

**Deliverables:**
1. SQL schema deployed (SQLite for dev, PostgreSQL for staging)
2. MCP Infrastructure server exposing `query_sql` tool
3. QueryClassifier with 7-rule decision tree (hardcoded)
4. Basic QueryRouter (no Thompson Sampling, no calibration)
5. Unit tests for classification logic

**Implementation Order:**

**Day 1-2: SQL Backend**
```bash
# Create files
touch hololoom/infrastructure/sql_backend.py
touch hololoom/infrastructure/mcp_server.py
touch hololoom/infrastructure/schemas/beekeeping_schema.sql

# Write SQL backend
# Write MCP server with query_sql tool
# Create migration scripts
```

**Day 3-4: Classification Logic**
```bash
# Create files
touch hololoom/context/query_classifier.py
touch hololoom/context/query_router.py

# Implement 7-rule classifier
# Write unit tests
pytest hololoom/tests/unit/test_query_classifier.py -v
```

**Day 5-7: Integration**
```bash
# Modify WeavingOrchestrator
# Add enable_hybrid_routing config flag
# Wire QueryRouter into weaving cycle

# Integration tests
pytest hololoom/tests/integration/test_hybrid_routing.py -v
```

**Day 8-10: E2E Testing + Documentation**
```bash
# End-to-end tests with real queries
python demos/demo_hybrid_routing_phase1.py

# Write Phase 1 documentation
# Deploy to staging
# Performance benchmarks
```

**Phase 1 Success Criteria:**
- ✅ SQL queries execute correctly (100% accuracy on test set)
- ✅ Classifier routes queries to correct backend (>85% accuracy)
- ✅ Routing overhead <30ms (p95)
- ✅ All tests passing (30+ unit, 10+ integration, 5+ e2e)
- ✅ Zero regressions (graph-only path still works)

---

**Phase 2: Learning (Weeks 3-4)**
**Goal**: Add Thompson Sampling, calibration, ReflectionBuffer integration
**Risk**: Medium (adaptive behavior, potential instability)

**Deliverables:**
1. BackendBandit with Thompson Sampling
2. ConfidenceCalibrator tracking predicted vs. actual
3. LearningTracker recording routing events
4. ReflectionBuffer integration for long-term learning
5. StrategyUpdater for automatic adjustments

**Implementation Order:**

**Day 1-3: Thompson Sampling**
```bash
# Implement BackendBandit
touch hololoom/context/backend_bandit.py

# Integrate into QueryRouter
# Unit tests for bandit updates
pytest hololoom/tests/unit/test_backend_bandit.py -v
```

**Day 4-6: Calibration + Learning**
```bash
# Implement ConfidenceCalibrator
touch hololoom/context/calibration.py

# Implement LearningTracker
touch hololoom/context/learning_tracker.py

# Wire into QueryRouter
# Tests for calibration curve
pytest hololoom/tests/unit/test_calibration.py -v
```

**Day 7-9: Strategy Updates**
```bash
# Implement StrategyUpdater
touch hololoom/context/strategy_updater.py

# Add periodic update checks
# Tests for strategy adjustments
pytest hololoom/tests/unit/test_strategy_updater.py -v
```

**Day 10-14: Integration + Validation**
```bash
# Full integration tests with learning enabled
pytest hololoom/tests/integration/test_learning_routing.py -v

# Run 1000-query learning simulation
python experiments/routing_learning_experiment.py

# Analyze calibration curves, bandit convergence
# Deploy to staging with monitoring
```

**Phase 2 Success Criteria:**
- ✅ Thompson Sampling converges (confidence intervals <0.15 after 500 queries)
- ✅ Calibration ECE <0.10 (well-calibrated predictions)
- ✅ Routing accuracy improves over time (>90% after 1000 queries)
- ✅ Strategy updates trigger correctly (logged events)
- ✅ No performance regression (<50ms routing overhead at p95)

---

**Phase 3: Production Hardening (Weeks 5-6)**
**Goal**: Monitoring, error handling, multi-domain support, production deployment
**Risk**: Low (refinements only, no new architecture)

**Deliverables:**
1. Prometheus metrics for routing performance
2. Grafana dashboards (routing accuracy, latency, fallback rate)
3. Alerting rules (high fallback rate, calibration drift)
4. Multi-domain schema templates (healthcare, finance, manufacturing)
5. Production deployment guide
6. Runbooks for common issues

**Implementation Order:**

**Day 1-3: Monitoring**
```bash
# Add Prometheus metrics
touch hololoom/context/metrics.py

# Routing accuracy gauge
# Latency histogram
# Fallback rate counter
# Calibration ECE gauge

# Grafana dashboard JSON
touch monitoring/dashboards/hybrid_routing.json
```

**Day 4-6: Multi-Domain Support**
```bash
# Create schema templates
touch hololoom/infrastructure/schemas/healthcare_schema.sql
touch hololoom/infrastructure/schemas/finance_schema.sql

# Domain registry
touch hololoom/infrastructure/domain_registry.py

# Schema migration tool
touch hololoom/infrastructure/migrate_domain.py
```

**Day 7-10: Production Deployment**
```bash
# PostgreSQL setup guide
# Docker Compose for dev/staging
# Kubernetes manifests for production
# Backup/restore procedures

# Deploy to production (canary rollout)
# Monitor for 48 hours
# Full rollout
```

**Day 11-14: Documentation + Training**
```bash
# Write production deployment guide
# Write troubleshooting runbook
# Create training materials
# Team knowledge transfer sessions
```

**Phase 3 Success Criteria:**
- ✅ All production metrics visible in Grafana
- ✅ Alerting rules tested and working
- ✅ Multi-domain schemas validated (3+ domains)
- ✅ Production deployment successful (0 incidents)
- ✅ Team trained on monitoring and troubleshooting
- ✅ Documentation complete and reviewed

---

### 4.2 Testing Strategy

**Test Pyramid:**
```
        E2E Tests (5-10 tests, slow)
               /\
              /  \
             /    \
            /      \
           / Integration (20-30 tests, medium)
          /\
         /  \
        /    \
       /      \
      / Unit Tests (50-100 tests, fast)
     /\
    /__\
```

**Unit Tests (Fast, <100ms each):**

1. **QueryClassifier Tests** (`test_query_classifier.py`):
```python
def test_exact_id_query_routes_to_sql():
    classifier = QueryClassifier()
    result = classifier.classify("Get policy rule bee_001")
    assert result.backend == "sql"
    assert result.confidence >= 0.95

def test_similarity_query_routes_to_vector():
    classifier = QueryClassifier()
    result = classifier.classify("Find similar beekeeping practices")
    assert result.backend == "qdrant"
    assert result.confidence >= 0.85

def test_relationship_query_routes_to_graph():
    classifier = QueryClassifier()
    result = classifier.classify("What entities are connected to hive maintenance?")
    assert result.backend == "neo4j"
    assert result.confidence >= 0.85

# 10+ more classifier tests...
```

2. **BackendBandit Tests** (`test_backend_bandit.py`):
```python
def test_bandit_updates_alpha_on_success():
    bandit = BackendBandit()
    initial_alpha = bandit.bandit.alphas[0]

    bandit.update(backend="sql", success=True, confidence=0.90, latency_ms=15.0)

    assert bandit.bandit.alphas[0] > initial_alpha  # α increased

def test_bandit_updates_beta_on_failure():
    bandit = BackendBandit()
    initial_beta = bandit.bandit.betas[0]

    bandit.update(backend="sql", success=False, confidence=0.40, latency_ms=15.0)

    assert bandit.bandit.betas[0] > initial_beta  # β increased

# 8+ more bandit tests...
```

3. **ConfidenceCalibrator Tests** (`test_calibration.py`):
```python
def test_calibration_curve_requires_minimum_data():
    calibrator = ConfidenceCalibrator()

    # Add only 5 observations (need 10+)
    for i in range(5):
        calibrator.add_observation(0.8, 0.75, "sql")

    curve = calibrator.get_calibration_curve("sql")
    assert curve["calibrated"] == False
    assert curve["reason"] == "insufficient_data"

def test_well_calibrated_has_low_ece():
    calibrator = ConfidenceCalibrator()

    # Add 100 perfectly calibrated observations
    for i in range(100):
        pred = np.random.uniform(0.5, 1.0)
        actual = pred + np.random.normal(0, 0.05)  # Small noise
        calibrator.add_observation(pred, actual, "sql")

    curve = calibrator.get_calibration_curve("sql")
    assert curve["calibrated"] == True
    assert curve["ece"] < 0.10  # Well-calibrated

# 10+ more calibration tests...
```

**Integration Tests (Medium, <5s each):**

1. **Routing Flow Tests** (`test_hybrid_routing.py`):
```python
@pytest.mark.asyncio
async def test_sql_query_executes_end_to_end():
    """Test complete SQL query path"""
    config = Config.fused()
    config.enable_hybrid_routing = True

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        spacetime = await orchestrator.weave(Query(text="Get policy rule bee_001"))

        assert spacetime.confidence >= 0.95
        assert spacetime.metadata["backend_used"] == "sql"
        assert "bee_001" in spacetime.response

@pytest.mark.asyncio
async def test_fallback_when_primary_fails():
    """Test SQL → Graph fallback"""
    config = Config.fused()
    config.enable_hybrid_routing = True

    # Inject SQL failure
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        orchestrator.query_router.infrastructure_client.fail_sql = True

        spacetime = await orchestrator.weave(Query(text="Get policy rule bee_001"))

        assert spacetime.metadata["fallback_used"] == True
        assert spacetime.metadata["primary_backend"] == "sql"
        assert spacetime.metadata["actual_backend"] == "neo4j"

# 15+ more integration tests...
```

2. **Learning Tests** (`test_learning_routing.py`):
```python
@pytest.mark.asyncio
async def test_routing_improves_over_time():
    """Test learning from 100 queries"""
    config = Config.fused()
    config.enable_hybrid_routing = True

    queries = generate_test_queries(100)  # 50 SQL, 50 Graph

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        accuracies = []

        for batch_start in range(0, 100, 10):
            batch = queries[batch_start:batch_start+10]
            correct = 0

            for query, expected_backend in batch:
                spacetime = await orchestrator.weave(query)
                if spacetime.metadata["backend_used"] == expected_backend:
                    correct += 1

            accuracies.append(correct / 10)

        # Accuracy should improve (first batch vs. last batch)
        assert accuracies[-1] > accuracies[0]
        assert accuracies[-1] >= 0.90  # >90% accuracy after learning

# 10+ more learning tests...
```

**End-to-End Tests (Slow, <30s each):**

1. **Full Pipeline Tests** (`test_e2e_hybrid_routing.py`):
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_beekeeping_workflow():
    """Test realistic beekeeping domain workflow"""
    config = Config.fused()
    config.enable_hybrid_routing = True

    async with WeavingOrchestrator(cfg=config, shards=beekeeping_shards) as orchestrator:
        # Query 1: Get policy (SQL)
        s1 = await orchestrator.weave(Query(text="What is the Varroa treatment policy?"))
        assert s1.metadata["backend_used"] == "sql"
        assert s1.confidence >= 0.95

        # Query 2: Find related entities (Graph)
        s2 = await orchestrator.weave(Query(text="What hives are affected by this policy?"))
        assert s2.metadata["backend_used"] == "neo4j"
        assert s2.confidence >= 0.85

        # Query 3: Similar treatments (Vector)
        s3 = await orchestrator.weave(Query(text="Find similar pest control methods"))
        assert s3.metadata["backend_used"] == "qdrant"
        assert s3.confidence >= 0.80

        # Query 4: Hybrid query (SQL + Graph)
        s4 = await orchestrator.weave(Query(text="Show all hives violating treatment schedule"))
        assert "sql" in s4.metadata["backends_used"]
        assert "neo4j" in s4.metadata["backends_used"]

# 5+ more e2e tests...
```

**Test Coverage Target**: >85% for new routing code

---

### 4.3 Example Routing Scenarios

**Scenario 1: Exact Policy Lookup (SQL)**

```
Query: "What is the Varroa treatment schedule policy?"

Step 1: Classification
  - Pattern match: "policy" + "Varroa treatment schedule"
  - Rule: Policy/ground truth → SQL
  - Confidence: 0.90

Step 2: SQL Execution
  SQL: SELECT * FROM policy_rules WHERE rule_name LIKE ?
  Params: ['%Varroa%treatment%']
  Result: 1 row (bee_001, "Varroa Treatment Schedule", ...)
  Latency: 12.5ms
  Confidence: 1.0 (SQL = deterministic)

Step 3: Thompson Sampling Update
  Backend: sql
  Success: True (confidence ≥ 0.75)
  α_sql ← α_sql + 1.0

Step 4: Response
  Confidence: 1.0
  Latency: 12.5ms
  Backend: sql
  Response: "The Varroa treatment schedule requires..."

Result: ✅ Fast, accurate, high confidence
```

**Scenario 2: Relationship Traversal (Graph)**

```
Query: "What hives are connected to the Varroa treatment policy?"

Step 1: Classification
  - Pattern match: "connected to", "hives"
  - Rule: Relationship traversal → Graph
  - Confidence: 0.87

Step 2: Hybrid Execution (SQL + Graph)
  SQL: Get policy_id for "Varroa treatment"
    Result: bee_001

  Neo4j: MATCH (p:Policy {id: $policy_id})-[:APPLIES_TO]->(h:Hive)
         RETURN h
    Result: 15 hives
    Latency: 45ms
    Confidence: 0.88

Step 3: Thompson Sampling Update
  Backend: neo4j
  Success: True
  α_neo4j ← α_neo4j + 0.88

Step 4: Response
  Confidence: 0.88
  Latency: 57ms (SQL 12ms + Neo4j 45ms)
  Backends: ["sql", "neo4j"]
  Response: "15 hives are affected: North_Apiary_1, ..."

Result: ✅ Accurate relationships, good confidence
```

**Scenario 3: Similarity Search (Vector)**

```
Query: "Find beekeeping practices similar to Varroa treatment"

Step 1: Classification
  - Pattern match: "similar to"
  - Rule: Similarity → Vector
  - Confidence: 0.88

Step 2: Vector Execution
  Embedding: [0.23, -0.45, 0.67, ...] (384-dim)
  Qdrant: Search in "beekeeping_practices" collection
    top_k: 10
    Results: 10 practices (scores 0.82-0.91)
    Latency: 35ms
    Confidence: 0.85

Step 3: Thompson Sampling Update
  Backend: qdrant
  Success: True
  α_qdrant ← α_qdrant + 0.85

Step 4: Response
  Confidence: 0.85
  Latency: 35ms
  Backend: qdrant
  Response: "Similar practices include: Foulbrood prevention, ..."

Result: ✅ Semantic similarity working, good confidence
```

**Scenario 4: Ambiguous Query (Thompson Sampling)**

```
Query: "Tell me about hives"

Step 1: Classification
  - Pattern match: Generic query, no strong signal
  - Rule: None match strongly
  - Fallback: Thompson Sampling

Step 2: Thompson Sampling Selection
  Sample from Beta distributions:
    - sql: Beta(α=50, β=10) → sample 0.82
    - neo4j: Beta(α=120, β=30) → sample 0.79
    - qdrant: Beta(α=80, β=25) → sample 0.75

  Winner: sql (highest sample)
  Confidence: 0.70 (uncertain, exploring)

Step 3: SQL Execution
  SQL: SELECT * FROM entities WHERE entity_type = 'hive'
  Result: 50 hives
  Latency: 18ms
  Confidence: 0.80

Step 4: Thompson Sampling Update
  Backend: sql
  Success: True
  α_sql ← α_sql + 0.80

Step 5: Response
  Confidence: 0.80
  Latency: 18ms
  Backend: sql (exploration)
  Response: "There are 50 hives in the system: ..."

Result: ✅ Exploration working, learned from outcome
```

**Scenario 5: Low Confidence Refinement**

```
Query: "Are any hives out of compliance?"

Step 1: Classification
  - Pattern match: "compliance" (policy-related)
  - Rule: Policy → SQL
  - Confidence: 0.85

Step 2: SQL Execution
  SQL: Complex join across policy_rules, audit_trails, entities
  Result: 3 hives
  Latency: 85ms (complex query)
  Confidence: 0.68 (low! trigger refinement)

Step 3: Refinement Triggered
  Parallel execution:
    - SQL: Same query (already done)
    - Neo4j: MATCH (h:Hive)-[:SUBJECT_OF]->(a:Audit)
             WHERE a.compliance_flag = false
    - Qdrant: Search "non-compliant hives"

  Results:
    - SQL: 3 hives (conf 0.68)
    - Neo4j: 5 hives (conf 0.82, 3 overlap with SQL)
    - Qdrant: 4 hives (conf 0.75, 2 overlap with SQL)

  Merge: Weight by confidence
    - North_Apiary_1: 3/3 backends (high confidence)
    - South_Apiary_2: 3/3 backends (high confidence)
    - West_Apiary_5: 2/3 backends (medium confidence)
    - East_Apiary_3: 1/3 backends (low confidence, exclude)
    - East_Apiary_7: 1/3 backends (low confidence, exclude)

  Final: 3 hives (consensus across backends)
  Confidence: 0.89 (refined from 0.68!)

Step 4: Response
  Confidence: 0.89
  Latency: 195ms (85ms + 110ms parallel)
  Backends: ["sql", "neo4j", "qdrant"]
  Refinement: True
  Response: "3 hives are out of compliance: North_Apiary_1, ..."

Result: ✅ Refinement improved confidence from 0.68 → 0.89
```

**Scenario 6: SQL Failure with Fallback**

```
Query: "Get audit trail for hive North_Apiary_1"

Step 1: Classification
  - Pattern match: "audit trail" (ground truth data)
  - Rule: Audit → SQL
  - Confidence: 0.90

Step 2: SQL Execution FAILED
  SQL: SELECT * FROM audit_trails WHERE resource_id = ?
  Error: "database locked" (SQLite contention)
  Latency: 5ms
  Confidence: 0.0

Step 3: Fallback to Graph
  Fallback: SQL → Neo4j
  Neo4j: MATCH (h:Hive {id: $hive_id})-[:HAS_AUDIT]->(a:Audit)
         RETURN a ORDER BY a.timestamp DESC
  Result: 12 audit events
  Latency: 55ms
  Confidence: 0.82

Step 4: Thompson Sampling Update
  Backend: sql (primary attempt)
  Success: False
  β_sql ← β_sql + 1.0  # Penalize SQL

  Backend: neo4j (fallback)
  Success: True
  α_neo4j ← α_neo4j + 0.82

Step 5: Response
  Confidence: 0.82
  Latency: 60ms (5ms failed + 55ms fallback)
  Primary: sql
  Actual: neo4j (fallback)
  Fallback: True
  Response: "12 audit events found: ..."

Result: ⚠️ Fallback worked, but SQL failure logged for investigation
```

---

### 4.4 Performance Implications

**Routing Overhead Analysis:**

| Operation | Latency | Frequency | Cumulative |
|-----------|---------|-----------|------------|
| Query classification | 2-5ms | Every query | 2-5ms |
| Thompson Sampling | 0.5-1ms | 10-20% queries | 0.05-0.2ms avg |
| MCP call overhead | 1-2ms | Every query | 1-2ms |
| Calibration update | 0.2-0.5ms | Every query | 0.2-0.5ms |
| Strategy update check | 0.1ms | Every query | 0.1ms |
| **Total Routing Overhead** | **4-9ms** | **Every query** | **4-9ms** |

**Backend Latency Comparison:**

| Backend | Latency (p50) | Latency (p95) | When Worth It? |
|---------|---------------|---------------|----------------|
| SQL (simple) | 8-15ms | 20-30ms | Exact ID, small result set |
| SQL (complex join) | 50-150ms | 200-400ms | Complex queries, large joins |
| Neo4j (1-hop) | 20-40ms | 60-90ms | 1-2 hop traversals |
| Neo4j (3-hop) | 80-200ms | 300-500ms | Deep relationship queries |
| Qdrant (top 10) | 15-35ms | 50-80ms | Small similarity searches |
| Qdrant (top 100) | 40-100ms | 150-250ms | Large similarity searches |

**When Does Routing Pay Off?**

**Scenario A: Exact ID lookup**
- Without routing: Neo4j 3-hop traversal (150ms)
- With routing: SQL direct lookup (12ms)
- **Speedup: 12.5× faster**
- **Payoff: YES** (routing overhead 5ms << 138ms savings)

**Scenario B: Simple relationship query**
- Without routing: Qdrant similarity + manual filtering (80ms)
- With routing: Neo4j 1-hop traversal (30ms)
- **Speedup: 2.7× faster**
- **Payoff: YES** (routing overhead 5ms << 50ms savings)

**Scenario C: Exploratory query**
- Without routing: Neo4j full graph scan (500ms)
- With routing: Neo4j full graph scan (500ms + 5ms overhead)
- **Speedup: 1.0× (no improvement)**
- **Payoff: MARGINAL** (5ms overhead not worth it, but learning helps future queries)

**Overall Expected Performance:**
- **Query mix**: 40% exact, 40% relationships, 20% exploratory
- **Average speedup**: (12.5×0.4 + 2.7×0.4 + 1.0×0.2) = **6.3× faster**
- **Routing overhead**: 5ms average
- **Net benefit**: ~80-120ms saved per query

**Caching Opportunities:**

1. **Classification Cache**:
   ```python
   # Cache classification results for repeated queries
   classification_cache = LRUCache(maxsize=10000)

   def classify(self, query: str) -> BackendSelection:
       cache_key = hashlib.md5(query.encode()).hexdigest()
       if cache_key in self.classification_cache:
           return self.classification_cache[cache_key]

       result = self._classify_uncached(query)
       self.classification_cache[cache_key] = result
       return result
   ```
   **Benefit**: 90% cache hit rate → 2-5ms saved per cached query

2. **SQL Result Cache**:
   ```python
   # Cache SQL results for read-heavy workloads
   sql_cache = TTLCache(maxsize=5000, ttl=300)  # 5min TTL

   async def execute(self, sql: str, params: List) -> MCPResponse:
       cache_key = (sql, tuple(params))
       if cache_key in self.sql_cache:
           return self.sql_cache[cache_key]

       result = await self._execute_uncached(sql, params)
       self.sql_cache[cache_key] = result
       return result
   ```
   **Benefit**: 60% cache hit rate → 10-15ms saved per cached query

---

### 4.5 Monitoring and Observability

**Prometheus Metrics:**

```python
# hololoom/context/metrics.py

from prometheus_client import Counter, Histogram, Gauge, Enum

# Routing decisions
routing_backend_counter = Counter(
    'hololoom_routing_backend_total',
    'Total queries routed to each backend',
    ['backend', 'domain']
)

# Routing accuracy
routing_accuracy_gauge = Gauge(
    'hololoom_routing_accuracy',
    'Routing accuracy (0.0-1.0)',
    ['backend', 'window']  # window: 100q, 1000q
)

# Latency distributions
routing_latency_histogram = Histogram(
    'hololoom_routing_latency_seconds',
    'Routing overhead latency',
    buckets=[0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100]
)

backend_latency_histogram = Histogram(
    'hololoom_backend_latency_seconds',
    'Backend query latency',
    ['backend'],
    buckets=[0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0]
)

# Confidence tracking
confidence_gauge = Gauge(
    'hololoom_query_confidence',
    'Query result confidence',
    ['backend']
)

# Fallback rate
fallback_counter = Counter(
    'hololoom_fallback_total',
    'Total fallback events',
    ['primary_backend', 'fallback_backend']
)

# Calibration quality
calibration_ece_gauge = Gauge(
    'hololoom_calibration_ece',
    'Expected Calibration Error (lower = better)',
    ['backend']
)

# Thompson Sampling exploration
thompson_exploration_rate = Gauge(
    'hololoom_thompson_exploration_rate',
    'Thompson Sampling exploration rate (0.0-1.0)'
)
```

**Grafana Dashboard:**

```json
{
  "dashboard": {
    "title": "HoloLoom Hybrid Routing",
    "panels": [
      {
        "title": "Routing Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (backend) (rate(hololoom_routing_backend_total[5m]))"
          }
        ]
      },
      {
        "title": "Routing Accuracy (Rolling 1000q)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "hololoom_routing_accuracy{window='1000q'}",
            "legendFormat": "{{backend}}"
          }
        ],
        "thresholds": [
          {"value": 0.85, "color": "yellow"},
          {"value": 0.90, "color": "green"}
        ]
      },
      {
        "title": "Backend Latency (p50, p95, p99)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(hololoom_backend_latency_seconds_bucket[5m]))",
            "legendFormat": "{{backend}} p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(hololoom_backend_latency_seconds_bucket[5m]))",
            "legendFormat": "{{backend}} p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(hololoom_backend_latency_seconds_bucket[5m]))",
            "legendFormat": "{{backend}} p99"
          }
        ]
      },
      {
        "title": "Fallback Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(hololoom_fallback_total[5m])",
            "legendFormat": "{{primary_backend}} → {{fallback_backend}}"
          }
        ],
        "alert": {
          "condition": "rate > 0.10",
          "message": "Fallback rate >10% - investigate backend issues"
        }
      },
      {
        "title": "Calibration Quality (ECE)",
        "type": "gauge",
        "targets": [
          {
            "expr": "hololoom_calibration_ece",
            "legendFormat": "{{backend}}"
          }
        ],
        "thresholds": [
          {"value": 0.10, "color": "green"},
          {"value": 0.15, "color": "yellow"},
          {"value": 0.20, "color": "red"}
        ]
      },
      {
        "title": "Thompson Sampling Priors (α, β)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "hololoom_thompson_alpha",
            "legendFormat": "{{backend}} α"
          },
          {
            "expr": "hololoom_thompson_beta",
            "legendFormat": "{{backend}} β"
          }
        ]
      }
    ]
  }
}
```

**Alerting Rules:**

```yaml
# monitoring/alerts/hybrid_routing.yml

groups:
  - name: hybrid_routing
    interval: 1m
    rules:
      - alert: HighFallbackRate
        expr: rate(hololoom_fallback_total[5m]) > 0.20
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High fallback rate detected"
          description: "Fallback rate is {{ $value | humanizePercentage }} (threshold: 20%)"

      - alert: CalibrationDrift
        expr: hololoom_calibration_ece > 0.15
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Calibration drift detected for {{ $labels.backend }}"
          description: "ECE is {{ $value }} (threshold: 0.15)"

      - alert: RoutingAccuracyLow
        expr: hololoom_routing_accuracy{window="1000q"} < 0.85
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Routing accuracy below threshold for {{ $labels.backend }}"
          description: "Accuracy is {{ $value | humanizePercentage }} (threshold: 85%)"

      - alert: BackendLatencyHigh
        expr: histogram_quantile(0.95, rate(hololoom_backend_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High backend latency for {{ $labels.backend }}"
          description: "p95 latency is {{ $value }}s (threshold: 0.5s)"

      - alert: SQLBackendDown
        expr: up{job="infrastructure-mcp-server", tool="query_sql"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "SQL backend unavailable"
          description: "MCP query_sql tool is down"
```

---

### 4.6 Migration Path

**For Existing HoloLoom Deployments:**

**Step 1: Pre-Migration (Week 0)**
```bash
# Backup existing data
pg_dump hololoom_production > backup_pre_migration.sql
docker exec neo4j bin/neo4j-admin dump --to=/backups/pre_migration.dump

# Create staging environment
docker-compose -f docker-compose.staging.yml up -d

# Deploy Phase 1 code to staging
git checkout phase-1-routing
PYTHONPATH=. python setup.py install
```

**Step 2: Schema Migration (Week 1)**
```bash
# Create SQL schema (new)
psql -U hololoom_user -d hololoom_staging < hololoom/infrastructure/schemas/production_schema.sql

# Migrate ground truth data from Neo4j → SQL
python hololoom/infrastructure/migrate_ground_truth.py \
  --neo4j-uri bolt://localhost:7687 \
  --postgres-uri postgresql://localhost:5432/hololoom_staging \
  --domain beekeeping \
  --dry-run

# Verify migration
python hololoom/infrastructure/validate_migration.py

# Execute migration (if validation passed)
python hololoom/infrastructure/migrate_ground_truth.py \
  --neo4j-uri bolt://localhost:7687 \
  --postgres-uri postgresql://localhost:5432/hololoom_staging \
  --domain beekeeping
```

**Step 3: Parallel Run (Week 2-3)**
```python
# Enable dual-write mode (write to both old and new)
config = Config.fused()
config.enable_hybrid_routing = True
config.dual_write_mode = True  # Write to both SQL and Neo4j

# Routing enabled, but shadow mode (don't use results yet)
config.shadow_mode = True  # Route queries but return graph-only results
config.log_routing_decisions = True

# Deploy to staging
# Monitor routing decisions vs. actual behavior
# Fix discrepancies
```

**Step 4: Gradual Rollout (Week 4-5)**
```python
# Canary rollout (1% traffic)
config.enable_hybrid_routing = True
config.rollout_percentage = 1.0  # 1% of queries use routing

# Monitor for 48 hours
# If stable: 5% → 25% → 50% → 100%

# Full rollout
config.rollout_percentage = 100.0
```

**Step 5: Cleanup (Week 6)**
```bash
# Remove duplicate data from Neo4j (now in SQL)
python hololoom/infrastructure/cleanup_migrated_data.py \
  --neo4j-uri bolt://localhost:7687 \
  --confirm

# Archive old code
git tag pre-hybrid-routing
git branch archive/graph-only-routing

# Update documentation
```

**Rollback Plan:**
```python
# If issues detected, instant rollback
config.enable_hybrid_routing = False  # Disable routing
config.use_graph_only_fallback = True  # Use old code path

# No data loss (dual-write kept Neo4j in sync)
# Rollback deployment
git revert HEAD
```

---

**END OF SECTION 4**

---

## Conclusion

This hybrid query routing architecture provides:

✅ **Precision**: SQL for deterministic ground truth (1.0 confidence)
✅ **Semantics**: Neo4j for relationship traversal (0.7-0.9 confidence)
✅ **Similarity**: Qdrant for vector search (0.6-0.85 confidence)
✅ **Intelligence**: Thompson Sampling learns optimal routing
✅ **Reliability**: Automatic fallback, error escalation, refinement
✅ **Observability**: Complete monitoring, alerting, provenance
✅ **Scalability**: Multi-domain support, B2B marketplace ready

**Expected Outcomes:**
- **Routing accuracy**: >90% after learning (1000 queries)
- **Performance**: 6.3× average speedup (exact lookups)
- **Confidence**: 0.85+ average (vs. 0.75 graph-only)
- **Reliability**: <5% fallback rate in production

**Next Steps:**
1. Review architecture with team
2. Prototype Phase 1 (Weeks 1-2)
3. Deploy to staging for validation
4. Iterate based on feedback
5. Production rollout (Weeks 5-6)

🚀 **Ready to implement!**

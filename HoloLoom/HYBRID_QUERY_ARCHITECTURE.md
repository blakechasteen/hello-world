# HoloLoom Hybrid Query Architecture
## SQL + Graph with Agentic Routing

**Status**: Design Document (Phase 1)
**Owner**: Infrastructure Department
**Generated**: November 10, 2025

---

## 📊 Executive Summary

This architecture adds a **SQL precision layer** alongside the existing graph semantic layer, with an **agentic routing engine** that learns optimal query path selection.

**Key Innovation**: Zero-copy architecture with importance-based duplication tracking.

```
┌─────────────────────────────────────────────────────────────┐
│                    QUERY ORCHESTRATOR                       │
│              (Agentic Routing Decision Layer)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│   SQL Layer     │       │  Graph Layer    │
│  (Precision)    │       │  (Semantic)     │
├─────────────────┤       ├─────────────────┤
│• Policy metrics │       │• Entities       │
│• Ground truth   │       │• Relationships  │
│• Audit trail    │       │• Traversal      │
│• Statistics     │       │• Exploration    │
└─────────────────┘       └─────────────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
              ┌──────────────┐
              │ Zero-Copy    │
              │ Coordinator  │
              └──────────────┘
```

---

## 1. SQL Schema Design

### 1.1 Core Tables

#### `policy_statistics` - Thompson Sampling & Tool Performance
```sql
CREATE TABLE policy_statistics (
    id                  SERIAL PRIMARY KEY,
    tool_name           VARCHAR(100) NOT NULL,
    adapter_mode        VARCHAR(50),      -- BARE/FAST/FUSED

    -- Thompson Sampling (Beta distribution)
    alpha               FLOAT NOT NULL DEFAULT 1.0,
    beta                FLOAT NOT NULL DEFAULT 1.0,

    -- Performance metrics
    total_uses          INTEGER DEFAULT 0,
    successful_uses     INTEGER DEFAULT 0,
    avg_confidence      FLOAT,
    avg_latency_ms      FLOAT,

    -- Temporal tracking
    last_updated        TIMESTAMP DEFAULT NOW(),
    created_at          TIMESTAMP DEFAULT NOW(),

    -- Indexing
    UNIQUE(tool_name, adapter_mode)
);

CREATE INDEX idx_policy_tool ON policy_statistics(tool_name);
CREATE INDEX idx_policy_updated ON policy_statistics(last_updated DESC);
```

#### `ground_truth` - Verified Facts & User Feedback
```sql
CREATE TABLE ground_truth (
    id                  SERIAL PRIMARY KEY,

    -- Source
    source_type         VARCHAR(50) NOT NULL,  -- user_feedback, verified_fact, alignment_eval
    source_id           VARCHAR(255),          -- reference to memory/query ID

    -- Content
    statement           TEXT NOT NULL,
    verification_status VARCHAR(50),           -- verified, disputed, pending
    confidence          FLOAT,

    -- Metadata
    verified_by         VARCHAR(100),          -- user_id or system
    verified_at         TIMESTAMP DEFAULT NOW(),
    tags                JSONB,                 -- flexible tagging

    -- Links to graph (foreign key to entity/memory)
    graph_entity_id     VARCHAR(255),
    graph_memory_id     VARCHAR(255),

    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_gt_source ON ground_truth(source_type, source_id);
CREATE INDEX idx_gt_status ON ground_truth(verification_status);
CREATE INDEX idx_gt_entity ON ground_truth(graph_entity_id);
CREATE INDEX idx_gt_tags ON ground_truth USING GIN(tags);
```

#### `audit_trail` - Alignment & Safety Decisions
```sql
CREATE TABLE audit_trail (
    id                  SERIAL PRIMARY KEY,

    -- Decision context
    decision_id         VARCHAR(255) NOT NULL UNIQUE,
    decision_type       VARCHAR(50) NOT NULL,  -- safety_gate, deception_check, etc.
    outcome             VARCHAR(50) NOT NULL,  -- approved, rejected, escalated

    -- Content
    query_text          TEXT,
    action_description  TEXT,
    reason              TEXT NOT NULL,
    risk_level          VARCHAR(50),
    confidence          FLOAT,

    -- Provenance
    reasoning_chain     JSONB,                 -- List of reasoning steps
    data_sources        JSONB,                 -- List of source IDs
    metadata            JSONB,

    -- Temporal
    timestamp           TIMESTAMP DEFAULT NOW(),

    -- Department tracking (for agent swarm)
    department          VARCHAR(100),
    session_id          VARCHAR(255)
);

CREATE INDEX idx_audit_decision ON audit_trail(decision_id);
CREATE INDEX idx_audit_type ON audit_trail(decision_type);
CREATE INDEX idx_audit_timestamp ON audit_trail(timestamp DESC);
CREATE INDEX idx_audit_risk ON audit_trail(risk_level);
CREATE INDEX idx_audit_session ON audit_trail(session_id);
```

#### `query_routing_history` - Learning Dataset
```sql
CREATE TABLE query_routing_history (
    id                  SERIAL PRIMARY KEY,

    -- Query characteristics
    query_text          TEXT NOT NULL,
    query_type          VARCHAR(50),           -- factual, procedural, analytical
    query_complexity    VARCHAR(50),           -- simple, medium, complex

    -- Routing decision
    route_chosen        VARCHAR(50) NOT NULL,  -- sql, graph, hybrid
    decision_confidence FLOAT,
    decision_reason     TEXT,

    -- Outcome metrics
    latency_ms          FLOAT,
    result_count        INTEGER,
    result_quality      FLOAT,                 -- 0-1 score
    user_satisfaction   FLOAT,                 -- 0-1 if available

    -- Learning signals
    was_optimal         BOOLEAN,               -- Learned later if this was best route
    alternative_routes  JSONB,                 -- Other routes tried + their metrics

    -- Temporal
    timestamp           TIMESTAMP DEFAULT NOW(),

    -- Department
    department          VARCHAR(100)
);

CREATE INDEX idx_routing_type ON query_routing_history(query_type);
CREATE INDEX idx_routing_route ON query_routing_history(route_chosen);
CREATE INDEX idx_routing_optimal ON query_routing_history(was_optimal);
CREATE INDEX idx_routing_timestamp ON query_routing_history(timestamp DESC);
```

#### `metric_aggregates` - Precomputed Statistics
```sql
CREATE TABLE metric_aggregates (
    id                  SERIAL PRIMARY KEY,

    -- Metric identity
    metric_name         VARCHAR(100) NOT NULL,
    dimension           VARCHAR(100),          -- tool, department, session, etc.
    dimension_value     VARCHAR(255),

    -- Time bucket
    time_bucket         VARCHAR(50),           -- hourly, daily, weekly
    bucket_start        TIMESTAMP NOT NULL,
    bucket_end          TIMESTAMP NOT NULL,

    -- Aggregated values
    count               INTEGER,
    sum_value           FLOAT,
    avg_value           FLOAT,
    min_value           FLOAT,
    max_value           FLOAT,
    stddev_value        FLOAT,

    -- Additional statistics
    percentile_50       FLOAT,
    percentile_90       FLOAT,
    percentile_99       FLOAT,

    created_at          TIMESTAMP DEFAULT NOW(),

    UNIQUE(metric_name, dimension, dimension_value, time_bucket, bucket_start)
);

CREATE INDEX idx_metric_name ON metric_aggregates(metric_name);
CREATE INDEX idx_metric_bucket ON metric_aggregates(time_bucket, bucket_start DESC);
```

#### `importance_tracker` - Zero-Copy Duplication Tracking
```sql
CREATE TABLE importance_tracker (
    id                  SERIAL PRIMARY KEY,

    -- Entity/data reference
    entity_id           VARCHAR(255) NOT NULL,
    entity_type         VARCHAR(50) NOT NULL,  -- memory, entity, relationship

    -- Duplication tracking
    graph_location      VARCHAR(255),          -- Neo4j node ID
    sql_locations       JSONB,                 -- List of SQL table references
    duplication_count   INTEGER DEFAULT 0,

    -- Importance signals
    access_frequency    INTEGER DEFAULT 0,
    last_access         TIMESTAMP,
    avg_confidence      FLOAT,
    user_feedback_score FLOAT,

    -- Computed importance
    importance_score    FLOAT,                 -- Composite score
    importance_tier     VARCHAR(50),           -- critical, high, medium, low

    -- Temporal
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW(),

    UNIQUE(entity_id, entity_type)
);

CREATE INDEX idx_importance_entity ON importance_tracker(entity_id, entity_type);
CREATE INDEX idx_importance_score ON importance_tracker(importance_score DESC);
CREATE INDEX idx_importance_tier ON importance_tracker(importance_tier);
CREATE INDEX idx_importance_access ON importance_tracker(last_access DESC);
```

---

## 2. Routing Criteria

### 2.1 Decision Matrix

| Query Characteristic | SQL | Graph | Hybrid | Confidence |
|---------------------|-----|-------|--------|-----------|
| **Exact match filter** (tool=X, risk=HIGH) | ✓ | | | 0.95 |
| **Aggregation** (avg, count, sum) | ✓ | | | 0.95 |
| **Time-series** (last 7 days, trend) | ✓ | | | 0.90 |
| **Semantic exploration** (related to X) | | ✓ | | 0.90 |
| **Multi-hop traversal** (X → Y → Z) | | ✓ | | 0.95 |
| **Entity relationships** (what uses X?) | | ✓ | | 0.90 |
| **Precision + semantics** (high-risk + related) | | | ✓ | 0.85 |
| **Complex analytics** (correlation, regression) | ✓ | | | 0.80 |
| **Recent + connected** (last week + neighbors) | | | ✓ | 0.85 |

### 2.2 Query Classification Rules

#### SQL-Optimal Queries
```python
SQL_PATTERNS = [
    # Exact filters
    r'tool(?:_name)?\s*=\s*["\']?\w+',
    r'risk_level\s*=\s*["\']?(HIGH|MEDIUM|LOW)',
    r'confidence\s*[<>]=?\s*\d+\.?\d*',

    # Aggregations
    r'\b(count|sum|avg|average|mean|min|max|stddev)\b',
    r'\bhow many\b',
    r'\btotal\b.*\b(uses|queries|decisions)\b',

    # Time-based
    r'\blast\s+\d+\s+(days?|weeks?|months?)',
    r'\bsince\s+\d{4}-\d{2}-\d{2}',
    r'\btrend\b|\btimeseries\b',

    # Comparisons
    r'\b(greater|less|more|fewer|higher|lower)\s+than\b',
    r'\b(top|bottom)\s+\d+\b',
]
```

#### Graph-Optimal Queries
```python
GRAPH_PATTERNS = [
    # Semantic
    r'\brelated to\b',
    r'\bsimilar to\b',
    r'\bconnected to\b',

    # Traversal
    r'\bwhat (uses|mentions|leads to)\b',
    r'\bpath (from|between)\b',
    r'\bneighbors? of\b',

    # Exploration
    r'\bexplore\b',
    r'\bfind all\b.*\b(entities|relationships)\b',
    r'\bsubgraph\b',
]
```

#### Hybrid Queries
```python
HYBRID_PATTERNS = [
    # Combined filters
    r'(HIGH|MEDIUM|LOW)\s+risk.*\b(related|connected)\b',
    r'\b(recent|last)\b.*\b(and|with)\b.*\b(related|connected)\b',

    # Semantic + precision
    r'\bconfidence\s*>\s*\d+.*\brelated to\b',
    r'\btool\s*=.*\band\b.*\bneighbors\b',
]
```

### 2.3 Complexity-Based Latency Gates

```python
class LatencyGate(Enum):
    INSTANT = "instant"      # <10ms - Simple SQL lookups
    FAST = "fast"            # <50ms - Indexed SQL queries
    MODERATE = "moderate"    # <200ms - Graph traversal, SQL joins
    COMPLEX = "complex"      # <1s - Hybrid queries, multi-hop
    RESEARCH = "research"    # >1s - Deep analysis, unrestricted
```

**Assignment Logic**:
```python
def assign_latency_gate(query_complexity: str, route: str) -> LatencyGate:
    if query_complexity == "simple" and route == "sql":
        return LatencyGate.INSTANT
    elif query_complexity == "simple" and route == "graph":
        return LatencyGate.FAST
    elif query_complexity == "medium":
        return LatencyGate.MODERATE
    elif query_complexity == "complex" or route == "hybrid":
        return LatencyGate.COMPLEX
    else:
        return LatencyGate.RESEARCH
```

---

## 3. Agentic Decision Layer

### 3.1 Query Router Architecture

```python
# HoloLoom/infrastructure/query_router.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple
import re
import asyncio

class QueryRoute(Enum):
    SQL = "sql"
    GRAPH = "graph"
    HYBRID = "hybrid"

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class QueryAnalysis:
    """Analysis of query characteristics."""
    text: str
    query_type: str              # factual, procedural, analytical
    complexity: QueryComplexity

    # Feature extraction
    has_exact_filters: bool
    has_aggregations: bool
    has_time_constraints: bool
    has_semantic_intent: bool
    has_traversal_intent: bool

    # Entity mentions
    entities_mentioned: List[str]
    tools_mentioned: List[str]

    # Confidence in classification
    classification_confidence: float

@dataclass
class RoutingDecision:
    """Routing decision with reasoning."""
    route: QueryRoute
    confidence: float
    reason: str

    # Alternative routes (for learning)
    alternatives: List[Tuple[QueryRoute, float]]

    # Latency expectation
    latency_gate: LatencyGate
    expected_latency_ms: float

    # Metadata for learning
    features: Dict[str, any]

class AgenticQueryRouter:
    """
    Agentic query router with learning capability.

    Analyzes queries and routes to optimal backend (SQL/graph/hybrid).
    Learns from outcomes to improve future routing decisions.
    """

    def __init__(self, sql_engine, graph_store, learning_enabled=True):
        self.sql = sql_engine
        self.graph = graph_store
        self.learning_enabled = learning_enabled

        # Thompson Sampling for route selection
        self.route_bandits = {
            QueryRoute.SQL: TSBandit(alpha=1.0, beta=1.0),
            QueryRoute.GRAPH: TSBandit(alpha=1.0, beta=1.0),
            QueryRoute.HYBRID: TSBandit(alpha=1.0, beta=1.0),
        }

        # Pattern matchers
        self.sql_patterns = [re.compile(p) for p in SQL_PATTERNS]
        self.graph_patterns = [re.compile(p) for p in GRAPH_PATTERNS]
        self.hybrid_patterns = [re.compile(p) for p in HYBRID_PATTERNS]

    async def analyze_query(self, query_text: str) -> QueryAnalysis:
        """
        Analyze query to extract characteristics.

        Steps:
        1. Pattern matching (SQL/graph/hybrid)
        2. Entity extraction (mentions of tools, entities)
        3. Complexity assessment
        4. Confidence scoring
        """
        text_lower = query_text.lower()

        # Pattern matching
        sql_matches = sum(1 for p in self.sql_patterns if p.search(text_lower))
        graph_matches = sum(1 for p in self.graph_patterns if p.search(text_lower))
        hybrid_matches = sum(1 for p in self.hybrid_patterns if p.search(text_lower))

        # Feature flags
        has_exact_filters = any(p.search(text_lower) for p in self.sql_patterns[:5])
        has_aggregations = any(p.search(text_lower) for p in self.sql_patterns[5:8])
        has_time_constraints = any(p.search(text_lower) for p in self.sql_patterns[8:11])
        has_semantic_intent = any(p.search(text_lower) for p in self.graph_patterns[:3])
        has_traversal_intent = any(p.search(text_lower) for p in self.graph_patterns[3:6])

        # Complexity scoring
        feature_count = sum([
            has_exact_filters,
            has_aggregations,
            has_time_constraints,
            has_semantic_intent,
            has_traversal_intent,
            hybrid_matches > 0
        ])

        if feature_count <= 1:
            complexity = QueryComplexity.SIMPLE
        elif feature_count == 2:
            complexity = QueryComplexity.MEDIUM
        else:
            complexity = QueryComplexity.COMPLEX

        # Query type classification
        if has_aggregations or has_time_constraints:
            query_type = "analytical"
        elif has_traversal_intent or has_semantic_intent:
            query_type = "exploratory"
        else:
            query_type = "factual"

        # Entity extraction (simple heuristic - enhance with NER later)
        entities = self._extract_entities(query_text)
        tools = self._extract_tools(query_text)

        # Confidence scoring
        total_matches = sql_matches + graph_matches + hybrid_matches
        if total_matches > 0:
            classification_confidence = max(
                sql_matches, graph_matches, hybrid_matches
            ) / total_matches
        else:
            classification_confidence = 0.5  # Uncertain

        return QueryAnalysis(
            text=query_text,
            query_type=query_type,
            complexity=complexity,
            has_exact_filters=has_exact_filters,
            has_aggregations=has_aggregations,
            has_time_constraints=has_time_constraints,
            has_semantic_intent=has_semantic_intent,
            has_traversal_intent=has_traversal_intent,
            entities_mentioned=entities,
            tools_mentioned=tools,
            classification_confidence=classification_confidence
        )

    async def decide_route(
        self,
        analysis: QueryAnalysis,
        use_learning: bool = True
    ) -> RoutingDecision:
        """
        Decide optimal route based on query analysis.

        Strategy:
        1. Rule-based initial routing (high confidence patterns)
        2. Thompson Sampling for uncertain cases
        3. Latency gate assignment
        """
        # Rule-based routing (high confidence)
        if analysis.classification_confidence > 0.8:
            route = self._rule_based_route(analysis)
            confidence = analysis.classification_confidence
            reason = f"High-confidence pattern match ({analysis.query_type})"

            # Calculate alternatives (Thompson Sampling scores)
            alternatives = [
                (r, self.route_bandits[r].sample())
                for r in QueryRoute
                if r != route
            ]
            alternatives.sort(key=lambda x: x[1], reverse=True)

        # Thompson Sampling (uncertain cases)
        else:
            if use_learning and self.learning_enabled:
                # Sample from bandits
                scores = {
                    r: self.route_bandits[r].sample()
                    for r in QueryRoute
                }
                route = max(scores, key=scores.get)
                confidence = scores[route]
                reason = f"Thompson Sampling selection (uncertainty)"

                alternatives = [
                    (r, scores[r]) for r in QueryRoute if r != route
                ]
                alternatives.sort(key=lambda x: x[1], reverse=True)
            else:
                # Fallback to graph (default)
                route = QueryRoute.GRAPH
                confidence = 0.5
                reason = "Default fallback (learning disabled)"
                alternatives = []

        # Assign latency gate
        latency_gate = assign_latency_gate(
            analysis.complexity.value,
            route.value
        )
        expected_latency = self._estimate_latency(route, analysis.complexity)

        return RoutingDecision(
            route=route,
            confidence=confidence,
            reason=reason,
            alternatives=alternatives,
            latency_gate=latency_gate,
            expected_latency_ms=expected_latency,
            features={
                'complexity': analysis.complexity.value,
                'has_exact_filters': analysis.has_exact_filters,
                'has_aggregations': analysis.has_aggregations,
                'has_semantic_intent': analysis.has_semantic_intent,
            }
        )

    def _rule_based_route(self, analysis: QueryAnalysis) -> QueryRoute:
        """Apply deterministic routing rules."""
        # SQL-optimal
        if (analysis.has_exact_filters or analysis.has_aggregations) and \
           not analysis.has_semantic_intent:
            return QueryRoute.SQL

        # Graph-optimal
        if (analysis.has_semantic_intent or analysis.has_traversal_intent) and \
           not (analysis.has_exact_filters or analysis.has_aggregations):
            return QueryRoute.GRAPH

        # Hybrid (both SQL + graph features)
        if (analysis.has_exact_filters or analysis.has_aggregations) and \
           (analysis.has_semantic_intent or analysis.has_traversal_intent):
            return QueryRoute.HYBRID

        # Default to graph
        return QueryRoute.GRAPH

    def _estimate_latency(
        self,
        route: QueryRoute,
        complexity: QueryComplexity
    ) -> float:
        """Estimate query latency in milliseconds."""
        base_latency = {
            QueryRoute.SQL: 10.0,
            QueryRoute.GRAPH: 30.0,
            QueryRoute.HYBRID: 50.0,
        }

        complexity_multiplier = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MEDIUM: 3.0,
            QueryComplexity.COMPLEX: 10.0,
        }

        return base_latency[route] * complexity_multiplier[complexity]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity mentions (simple heuristic)."""
        # TODO: Replace with spaCy NER or similar
        words = text.split()
        entities = [
            w.strip('.,!?;:"()[]{}')
            for w in words
            if w and w[0].isupper() and len(w) > 1
        ]
        return entities[:10]  # Limit

    def _extract_tools(self, text: str) -> List[str]:
        """Extract tool mentions."""
        # Known tools from policy
        known_tools = [
            "answer", "search", "verify", "refine", "synthesize"
        ]
        text_lower = text.lower()
        return [t for t in known_tools if t in text_lower]

    async def execute_query(
        self,
        query_text: str,
        route: Optional[QueryRoute] = None
    ) -> Tuple[any, float, Dict]:
        """
        Execute query on chosen route.

        Returns:
            (result, latency_ms, metadata)
        """
        import time

        # Analyze if route not specified
        if route is None:
            analysis = await self.analyze_query(query_text)
            decision = await self.decide_route(analysis)
            route = decision.route
        else:
            analysis = None
            decision = None

        # Execute
        start = time.time()

        if route == QueryRoute.SQL:
            result = await self._execute_sql(query_text)
        elif route == QueryRoute.GRAPH:
            result = await self._execute_graph(query_text)
        elif route == QueryRoute.HYBRID:
            result = await self._execute_hybrid(query_text)
        else:
            raise ValueError(f"Unknown route: {route}")

        latency_ms = (time.time() - start) * 1000

        # Metadata
        metadata = {
            'route': route.value,
            'latency_ms': latency_ms,
            'analysis': analysis,
            'decision': decision,
        }

        return result, latency_ms, metadata

    async def _execute_sql(self, query_text: str):
        """Execute SQL query (simplified - actual impl would parse + construct SQL)."""
        # TODO: Implement natural language → SQL translation
        # For now, assume parameterized queries
        raise NotImplementedError("SQL execution not yet implemented")

    async def _execute_graph(self, query_text: str):
        """Execute graph query."""
        from HoloLoom.memory.protocol import MemoryQuery

        query = MemoryQuery(text=query_text)
        result = await self.graph.recall(query, limit=10)
        return result

    async def _execute_hybrid(self, query_text: str):
        """Execute hybrid query (SQL + graph fusion)."""
        # Execute both in parallel
        sql_task = asyncio.create_task(self._execute_sql(query_text))
        graph_task = asyncio.create_task(self._execute_graph(query_text))

        sql_result, graph_result = await asyncio.gather(
            sql_task, graph_task, return_exceptions=True
        )

        # Fuse results (simple merge for now)
        # TODO: Implement intelligent fusion based on query type
        return {
            'sql': sql_result if not isinstance(sql_result, Exception) else None,
            'graph': graph_result if not isinstance(graph_result, Exception) else None,
        }
```

---

## 4. Learning Mechanism

### 4.1 Outcome Tracking

```python
# HoloLoom/infrastructure/routing_learner.py

@dataclass
class QueryOutcome:
    """Outcome of a query execution."""
    query_text: str
    route_used: QueryRoute

    # Performance metrics
    latency_ms: float
    result_count: int

    # Quality metrics (if available)
    result_quality: Optional[float] = None      # 0-1
    user_satisfaction: Optional[float] = None   # 0-1

    # Learning signal
    was_optimal: Optional[bool] = None

    timestamp: datetime = field(default_factory=datetime.now)

class RoutingLearner:
    """
    Learns from query outcomes to improve routing decisions.

    Features:
    1. Thompson Sampling updates (route-level)
    2. Quality prediction (ML model)
    3. Optimal route identification
    """

    def __init__(self, router: AgenticQueryRouter, sql_engine):
        self.router = router
        self.sql = sql_engine

    async def record_outcome(self, outcome: QueryOutcome) -> None:
        """
        Record query outcome and update learning models.

        Process:
        1. Persist to SQL (query_routing_history table)
        2. Update Thompson Sampling bandits
        3. Update quality prediction model
        """
        # 1. Persist to SQL
        await self._persist_outcome(outcome)

        # 2. Update Thompson Sampling
        await self._update_bandit(outcome)

        # 3. Identify if route was optimal (if we have comparison data)
        if outcome.was_optimal is None:
            outcome.was_optimal = await self._evaluate_optimality(outcome)

    async def _persist_outcome(self, outcome: QueryOutcome) -> None:
        """Persist outcome to query_routing_history table."""
        query = """
            INSERT INTO query_routing_history (
                query_text, route_chosen, latency_ms, result_count,
                result_quality, user_satisfaction, was_optimal, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        await self.sql.execute(
            query,
            outcome.query_text,
            outcome.route_used.value,
            outcome.latency_ms,
            outcome.result_count,
            outcome.result_quality,
            outcome.user_satisfaction,
            outcome.was_optimal,
            outcome.timestamp
        )

    async def _update_bandit(self, outcome: QueryOutcome) -> None:
        """
        Update Thompson Sampling bandit for route selection.

        Success criteria:
        - Latency within expected range
        - Result count > 0
        - Quality > threshold (if available)
        """
        route = outcome.route_used
        bandit = self.router.route_bandits[route]

        # Define success
        success = (
            outcome.latency_ms < 1000 and  # Not too slow
            outcome.result_count > 0 and    # Got results
            (outcome.result_quality is None or outcome.result_quality > 0.6)
        )

        # Update bandit
        if success:
            bandit.update(success=True, reward=outcome.result_quality or 1.0)
        else:
            bandit.update(success=False, reward=0.0)

    async def _evaluate_optimality(self, outcome: QueryOutcome) -> bool:
        """
        Evaluate if route was optimal by comparing to alternatives.

        Strategy:
        - Run same query on alternative routes (async, cached)
        - Compare latency + quality
        - Mark route as optimal if best

        NOTE: Expensive - only do this periodically or for important queries
        """
        # TODO: Implement alternative route testing
        # For now, assume optimal if successful
        return (
            outcome.latency_ms < 500 and
            outcome.result_count > 0 and
            (outcome.result_quality is None or outcome.result_quality > 0.7)
        )

    async def get_routing_statistics(self) -> Dict:
        """Get routing performance statistics from SQL."""
        query = """
            SELECT
                route_chosen,
                COUNT(*) as total_queries,
                AVG(latency_ms) as avg_latency,
                AVG(result_quality) as avg_quality,
                SUM(CASE WHEN was_optimal THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as optimality_rate
            FROM query_routing_history
            WHERE timestamp > NOW() - INTERVAL '7 days'
            GROUP BY route_chosen
        """

        rows = await self.sql.fetch(query)

        return {
            row['route_chosen']: {
                'total_queries': row['total_queries'],
                'avg_latency_ms': row['avg_latency'],
                'avg_quality': row['avg_quality'],
                'optimality_rate': row['optimality_rate'],
            }
            for row in rows
        }
```

---

## 5. Integration Architecture

### 5.1 Zero-Copy Coordinator

```python
# HoloLoom/infrastructure/zero_copy_coordinator.py

class ZeroCopyCoordinator:
    """
    Coordinates SQL + graph without unnecessary data duplication.

    Philosophy:
    - Data lives in ONE authoritative source
    - Other sources have REFERENCES only
    - Importance-based duplication for critical data
    """

    def __init__(self, sql_engine, graph_store):
        self.sql = sql_engine
        self.graph = graph_store

    async def store_memory(
        self,
        memory: Memory,
        importance_score: Optional[float] = None
    ) -> str:
        """
        Store memory with intelligent placement.

        Strategy:
        1. Always store in graph (authoritative)
        2. Track importance in SQL
        3. If important, duplicate to SQL for fast access
        """
        # 1. Store in graph (authoritative)
        memory_id = await self.graph.store(memory)

        # 2. Track importance
        if importance_score is None:
            importance_score = self._calculate_importance(memory)

        await self._track_importance(
            entity_id=memory_id,
            entity_type='memory',
            graph_location=memory_id,
            importance_score=importance_score
        )

        # 3. Duplicate if important (>0.7)
        if importance_score > 0.7:
            await self._duplicate_to_sql(memory, memory_id)

        return memory_id

    async def _track_importance(
        self,
        entity_id: str,
        entity_type: str,
        graph_location: str,
        importance_score: float
    ) -> None:
        """Track entity importance in importance_tracker table."""
        query = """
            INSERT INTO importance_tracker (
                entity_id, entity_type, graph_location,
                importance_score, importance_tier,
                access_frequency, last_access, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, 1, NOW(), NOW(), NOW())
            ON CONFLICT (entity_id, entity_type)
            DO UPDATE SET
                importance_score = $4,
                importance_tier = $5,
                access_frequency = importance_tracker.access_frequency + 1,
                last_access = NOW(),
                updated_at = NOW()
        """

        # Tier assignment
        if importance_score > 0.9:
            tier = 'critical'
        elif importance_score > 0.7:
            tier = 'high'
        elif importance_score > 0.5:
            tier = 'medium'
        else:
            tier = 'low'

        await self.sql.execute(
            query,
            entity_id,
            entity_type,
            graph_location,
            importance_score,
            tier
        )

    async def _duplicate_to_sql(self, memory: Memory, memory_id: str) -> None:
        """
        Duplicate important memory to SQL for fast access.

        NOTE: This is intentional duplication for performance.
        importance_tracker.sql_locations tracks where duplicates live.
        """
        # Example: Create fast-access table
        query = """
            INSERT INTO memory_cache (
                memory_id, text, timestamp, entities, confidence
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (memory_id) DO UPDATE SET
                text = $2,
                timestamp = $3,
                entities = $4,
                confidence = $5
        """

        entities_json = json.dumps(memory.context.get('entities', []))

        await self.sql.execute(
            query,
            memory_id,
            memory.text,
            memory.timestamp,
            entities_json,
            memory.metadata.get('confidence', 0.5)
        )

        # Update importance_tracker to track duplication
        update_query = """
            UPDATE importance_tracker
            SET
                sql_locations = COALESCE(sql_locations, '[]'::jsonb) || '["memory_cache"]'::jsonb,
                duplication_count = duplication_count + 1
            WHERE entity_id = $1 AND entity_type = 'memory'
        """

        await self.sql.execute(update_query, memory_id)

    def _calculate_importance(self, memory: Memory) -> float:
        """
        Calculate importance score (0-1).

        Factors:
        - Confidence (30%)
        - Entity count (20%)
        - User feedback (30%)
        - Access frequency (20%)
        """
        confidence = memory.metadata.get('confidence', 0.5)
        entity_count = len(memory.context.get('entities', []))
        feedback = memory.metadata.get('user_feedback', 0.5)

        # Normalize entity count (assume max 10)
        entity_score = min(entity_count / 10.0, 1.0)

        # Weighted average
        importance = (
            0.3 * confidence +
            0.2 * entity_score +
            0.3 * feedback +
            0.2 * 0.5  # Access frequency (unknown for new memories)
        )

        return importance
```

### 5.2 Backend Factory Integration

```python
# HoloLoom/memory/backend_factory.py (additions)

async def create_hybrid_with_sql(
    config: Config,
    guardrails: Optional[SafetyGuardrails] = None,
) -> HybridMemoryStore:
    """
    Create hybrid backend with SQL + graph.

    Extensions:
    - Adds SQL engine (SQLAlchemy)
    - Initializes query router
    - Configures zero-copy coordinator
    """
    # Create SQL engine
    sql_engine = await create_sql_engine(config)

    # Create graph backends (existing logic)
    neo4j, qdrant, fallback = await _initialize_backends(config)

    # Create query router
    graph_store = neo4j or fallback
    router = AgenticQueryRouter(
        sql_engine=sql_engine,
        graph_store=graph_store,
        learning_enabled=config.enable_routing_learning
    )

    # Create zero-copy coordinator
    coordinator = ZeroCopyCoordinator(
        sql_engine=sql_engine,
        graph_store=graph_store
    )

    # Create hybrid store with extensions
    hybrid = HybridMemoryStore(
        neo4j=neo4j,
        qdrant=qdrant,
        fallback=fallback,
        guardrails=guardrails,
        sql_engine=sql_engine,        # NEW
        query_router=router,           # NEW
        coordinator=coordinator         # NEW
    )

    return hybrid

async def create_sql_engine(config: Config):
    """
    Create SQL engine (SQLAlchemy + asyncpg).

    Supports:
    - PostgreSQL (production)
    - SQLite (development)
    """
    if config.sql_backend == "postgresql":
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(
            f"postgresql+asyncpg://{config.postgres_user}:{config.postgres_password}"
            f"@{config.postgres_host}:{config.postgres_port}/{config.postgres_db}",
            echo=config.sql_echo_queries,
            pool_size=config.sql_pool_size,
        )
    elif config.sql_backend == "sqlite":
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(
            f"sqlite+aiosqlite:///{config.sqlite_path}",
            echo=config.sql_echo_queries,
        )
    else:
        raise ValueError(f"Unknown SQL backend: {config.sql_backend}")

    # Run migrations
    await _run_sql_migrations(engine)

    return engine

async def _run_sql_migrations(engine):
    """Run SQL migrations to create tables."""
    # Import schemas
    from HoloLoom.infrastructure.sql_schema import Base

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

---

## 6. Department Integration

### 6.1 Infrastructure Department Tools

Add to `mcp_department_registry.py`:

```python
# Infrastructure Department (updated)
{
    "name": "Infrastructure",
    "charter": "Data systems, performance optimization, HYBRID QUERY ROUTING",
    "tools": [
        # Existing tools
        "query_neo4j",
        "query_qdrant",
        "diagnose_performance",

        # NEW: Hybrid query tools
        "route_query",               # Agentic routing decision
        "query_sql_precision",       # Direct SQL execution
        "query_graph_semantic",      # Direct graph execution
        "query_hybrid_fusion",       # Hybrid SQL+graph
        "get_routing_statistics",    # Learning metrics
        "update_importance",         # Update importance tracker
    ],
    "permissions": {
        "read": True,
        "write": True,     # Can update metrics, importance
        "execute": False,
        "deploy": False,
        "admin": True,     # System administration
    }
}
```

### 6.2 MCP Server Implementation

```python
# HoloLoom/infrastructure/mcp_server.py

from mcp import MCPServer

server = MCPServer("infrastructure")

@server.tool("route_query")
async def route_query(query_text: str, prefer_route: Optional[str] = None):
    """
    Route query to optimal backend (SQL/graph/hybrid).

    Args:
        query_text: Natural language query
        prefer_route: Optional route preference (sql, graph, hybrid)

    Returns:
        {
            "route": "sql" | "graph" | "hybrid",
            "confidence": 0.85,
            "reason": "High-confidence pattern match",
            "expected_latency_ms": 45.2
        }
    """
    router = get_router()  # Global router instance

    analysis = await router.analyze_query(query_text)
    decision = await router.decide_route(analysis)

    # Override if preference specified
    if prefer_route:
        decision.route = QueryRoute(prefer_route)
        decision.reason = f"User preference: {prefer_route}"

    return {
        "route": decision.route.value,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "expected_latency_ms": decision.expected_latency_ms,
        "analysis": {
            "complexity": analysis.complexity.value,
            "has_exact_filters": analysis.has_exact_filters,
            "has_semantic_intent": analysis.has_semantic_intent,
        }
    }

@server.tool("query_sql_precision")
async def query_sql_precision(query_text: str, parameters: Optional[Dict] = None):
    """
    Execute precision SQL query.

    Args:
        query_text: SQL query or natural language (will be translated)
        parameters: Optional query parameters

    Returns:
        {
            "results": [...],
            "row_count": 10,
            "latency_ms": 23.4
        }
    """
    # TODO: Implement natural language → SQL translation
    # For now, assume parameterized SQL
    router = get_router()
    result, latency_ms, metadata = await router._execute_sql(query_text)

    return {
        "results": result,
        "row_count": len(result) if isinstance(result, list) else 1,
        "latency_ms": latency_ms,
    }

@server.tool("get_routing_statistics")
async def get_routing_statistics(time_window: str = "7d"):
    """
    Get routing performance statistics.

    Args:
        time_window: Time window (7d, 30d, 90d)

    Returns:
        {
            "sql": {"total_queries": 1234, "avg_latency_ms": 45.2, ...},
            "graph": {...},
            "hybrid": {...}
        }
    """
    learner = get_learner()  # Global learner instance
    stats = await learner.get_routing_statistics()
    return stats
```

---

## 7. Deployment Plan

### Phase 1: Foundation (Week 1)
- [ ] SQL schema creation (`sql_schema.py`)
- [ ] Migration scripts (Alembic)
- [ ] SQLAlchemy models
- [ ] Basic SQL engine integration

### Phase 2: Router (Week 2)
- [ ] Query analysis (`query_router.py`)
- [ ] Pattern matching rules
- [ ] Thompson Sampling bandits
- [ ] Routing decision logic

### Phase 3: Learning (Week 3)
- [ ] Outcome tracking (`routing_learner.py`)
- [ ] Bandit updates
- [ ] Optimality evaluation
- [ ] Statistics dashboard

### Phase 4: Zero-Copy (Week 4)
- [ ] Coordinator implementation (`zero_copy_coordinator.py`)
- [ ] Importance tracking
- [ ] Intelligent duplication
- [ ] Reference management

### Phase 5: Integration (Week 5)
- [ ] Backend factory updates
- [ ] MCP server tools
- [ ] Department charter updates
- [ ] End-to-end testing

### Phase 6: Production (Week 6)
- [ ] Performance benchmarking
- [ ] Load testing
- [ ] Monitoring/alerting
- [ ] Documentation

---

## 8. Example Queries & Routing

### SQL-Routed Queries

**Q1**: "Show me all HIGH_RISK safety decisions from last week"
```
Route: SQL (confidence: 0.95)
Reason: Exact filter (risk_level=HIGH) + time constraint (last week)
Expected: <50ms

SQL:
SELECT * FROM audit_trail
WHERE risk_level = 'HIGH'
  AND timestamp > NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC;
```

**Q2**: "What's the average confidence for the verify tool?"
```
Route: SQL (confidence: 0.95)
Reason: Aggregation (avg) + exact filter (tool=verify)
Expected: <20ms

SQL:
SELECT AVG(avg_confidence) as avg_conf
FROM policy_statistics
WHERE tool_name = 'verify';
```

### Graph-Routed Queries

**Q3**: "What entities are related to Thompson Sampling?"
```
Route: GRAPH (confidence: 0.90)
Reason: Semantic exploration (related to)
Expected: <100ms

Graph:
entities = graph.get_neighbors("Thompson Sampling", direction="both", max_hops=2)
```

**Q4**: "Find the path from 'attention' to 'neural_network'"
```
Route: GRAPH (confidence: 0.95)
Reason: Multi-hop traversal (path finding)
Expected: <150ms

Graph:
paths = graph.get_paths("attention", "neural_network", max_length=3)
```

### Hybrid-Routed Queries

**Q5**: "Show high-confidence memories related to Bayesian methods"
```
Route: HYBRID (confidence: 0.85)
Reason: Precision filter (confidence>0.8) + semantic intent (related to)
Expected: <300ms

Hybrid:
1. SQL: SELECT memory_id FROM memory_cache WHERE confidence > 0.8
2. Graph: graph.get_neighbors("Bayesian methods")
3. Fusion: Intersect SQL results with graph neighborhood
```

**Q6**: "Recent HIGH_RISK decisions and their connected entities"
```
Route: HYBRID (confidence: 0.85)
Reason: Time + risk filter (SQL) + entity expansion (graph)
Expected: <400ms

Hybrid:
1. SQL: SELECT * FROM audit_trail WHERE risk_level='HIGH' AND timestamp > ...
2. For each decision: Graph: graph.get_neighbors(decision.entities)
3. Fusion: Merge decision + entity subgraph
```

---

## 9. Performance Targets

| Operation | Latency Target | Backend | Notes |
|-----------|---------------|---------|-------|
| Simple SQL lookup | <10ms | SQL | Indexed columns |
| Complex SQL aggregation | <50ms | SQL | With joins |
| Graph 1-hop traversal | <30ms | Graph | Direct neighbors |
| Graph multi-hop | <150ms | Graph | 2-3 hops |
| Hybrid query | <300ms | Both | Sequential execution |
| Learning update | <5ms | SQL | Async, non-blocking |

---

## 10. Monitoring & Observability

### Key Metrics

1. **Routing Accuracy**
   - Optimality rate (% queries routed correctly)
   - Thompson Sampling convergence
   - Route distribution (SQL/graph/hybrid %)

2. **Performance**
   - Latency by route (p50, p90, p99)
   - Query success rate
   - Cache hit rate (importance tracker)

3. **Learning**
   - Bandit α/β parameters over time
   - Routing confidence trends
   - Alternative route comparisons

### Dashboard Queries

```sql
-- Routing accuracy (last 7 days)
SELECT
    route_chosen,
    COUNT(*) as total,
    AVG(CASE WHEN was_optimal THEN 1.0 ELSE 0.0 END) as optimality_rate
FROM query_routing_history
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY route_chosen;

-- Performance by route
SELECT
    route_chosen,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50_latency,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY latency_ms) as p90_latency,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency
FROM query_routing_history
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY route_chosen;

-- Thompson Sampling convergence
SELECT
    tool_name,
    alpha,
    beta,
    alpha / (alpha + beta) as expected_success_rate
FROM policy_statistics
ORDER BY tool_name;
```

---

## 11. Future Enhancements

### Phase 2 (Q1 2026)
- Natural language → SQL translation (LLM-based)
- Learned query embeddings for similarity-based routing
- Multi-modal routing (text + images)

### Phase 3 (Q2 2026)
- Federated queries across department boundaries
- Query optimization (query plan caching)
- Real-time streaming queries (change data capture)

### Phase 4 (Q3 2026)
- Graph analytics in SQL (recursive CTEs)
- Vector search in SQL (pgvector integration)
- Cross-database joins (FDW for PostgreSQL)

---

## 12. References

- **Zero-Copy Architecture**: Minimize data duplication, track importance
- **Thompson Sampling**: Bayesian bandit for exploration/exploitation
- **Conway's Law**: Architecture mirrors organizational structure (departmental agents)
- **Latency Gating**: Open levels based on query complexity
- **Importance Tracking**: Duplication indicates critical data

**Generated**: November 10, 2025
**Department**: Infrastructure
**Status**: Design Complete → Ready for Implementation

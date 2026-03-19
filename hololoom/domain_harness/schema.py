"""
Domain Harness Schema - Neo4j Node and Relationship Types

Defines the graph structure for persistent domain memory:
- Feature: Backlog items with status and dependencies
- ProgressEntry: Worker run history  
- DomainState: Current project state
- TestResult: Test execution outcomes

Weaving Metaphor:
These are the "structural threads" that define the shape of work.
Features form the warp, progress entries weave the weft.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ============================================================================
# Enums
# ============================================================================

class FeatureStatus(str, Enum):
    """Status of a feature in the backlog."""
    PENDING = "pending"           # Not started
    IN_PROGRESS = "in_progress"   # Worker currently active
    PASSING = "passing"           # Tests pass
    FAILING = "failing"           # Tests fail
    BLOCKED = "blocked"           # Waiting on dependency
    SKIPPED = "skipped"           # Explicitly skipped


class ProgressAction(str, Enum):
    """Type of progress entry."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REVERTED = "reverted"


# ============================================================================
# Node Types
# ============================================================================

@dataclass
class Feature:
    """
    A unit of work in the domain backlog.
    
    Maps to flat-file features.json but with graph capabilities:
    - Explicit DEPENDS_ON relationships
    - Rich metadata
    - Temporal queries
    """
    name: str
    description: str
    status: FeatureStatus = FeatureStatus.PENDING
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Test configuration
    test_command: str | None = None
    acceptance_criteria: list[str] = field(default_factory=list)

    # Metadata
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Graph relationships (populated when loaded)
    depends_on: list[str] = field(default_factory=list)  # Feature IDs
    blocks: list[str] = field(default_factory=list)       # Feature IDs

    def to_cypher_properties(self) -> dict[str, Any]:
        """Convert to Neo4j property dict."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "test_command": self.test_command,
            "acceptance_criteria": self.acceptance_criteria,
            "labels": self.labels,
            "metadata": str(self.metadata)  # JSON string for Neo4j
        }

    @classmethod
    def from_cypher_record(cls, record: dict[str, Any]) -> 'Feature':
        """Create from Neo4j query result."""
        return cls(
            id=record["id"],
            name=record["name"],
            description=record["description"],
            status=FeatureStatus(record["status"]),
            priority=record.get("priority", 0),
            created_at=datetime.fromisoformat(record["created_at"]),
            updated_at=datetime.fromisoformat(record["updated_at"]),
            test_command=record.get("test_command"),
            acceptance_criteria=record.get("acceptance_criteria", []),
            labels=record.get("labels", []),
            metadata=eval(record.get("metadata", "{}"))
        )

    def is_actionable(self) -> bool:
        """Can a worker act on this feature?"""
        return self.status in (FeatureStatus.PENDING, FeatureStatus.FAILING)

    def is_complete(self) -> bool:
        """Has this feature been successfully completed?"""
        return self.status == FeatureStatus.PASSING


@dataclass
class ProgressEntry:
    """
    Record of a worker run.
    
    Maps to flat-file progress.log but with:
    - Semantic embedding for search
    - Graph links to features
    - Rich metadata
    """
    feature_id: str
    action: ProgressAction
    summary: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    # Git/code tracking
    diff_hash: str | None = None
    files_changed: list[str] = field(default_factory=list)

    # Quality metrics
    confidence: float = 0.0
    duration_ms: int = 0

    # Worker identity (for multi-agent)
    worker_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_cypher_properties(self) -> dict[str, Any]:
        """Convert to Neo4j property dict."""
        return {
            "id": self.id,
            "feature_id": self.feature_id,
            "action": self.action.value,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "diff_hash": self.diff_hash,
            "files_changed": self.files_changed,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "worker_id": self.worker_id,
            "metadata": str(self.metadata)
        }

    @classmethod
    def from_cypher_record(cls, record: dict[str, Any]) -> 'ProgressEntry':
        """Create from Neo4j query result."""
        return cls(
            id=record["id"],
            feature_id=record["feature_id"],
            action=ProgressAction(record["action"]),
            summary=record["summary"],
            timestamp=datetime.fromisoformat(record["timestamp"]),
            diff_hash=record.get("diff_hash"),
            files_changed=record.get("files_changed", []),
            confidence=record.get("confidence", 0.0),
            duration_ms=record.get("duration_ms", 0),
            worker_id=record.get("worker_id"),
            metadata=eval(record.get("metadata", "{}"))
        )

    def to_embedding_text(self) -> str:
        """Text for semantic embedding."""
        return f"{self.action.value}: {self.summary}"


@dataclass
class DomainState:
    """
    Singleton node representing current project state.
    
    Maps to flat-file state.json but with:
    - Direct graph links
    - Temporal tracking
    """
    project_name: str
    id: str = "singleton"

    # Current work
    current_feature_id: str | None = None

    # Environment
    environment: dict[str, str] = field(default_factory=dict)

    # Constraints
    constraints: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_worker_run: datetime | None = None
    last_test_run: datetime | None = None

    # Metrics
    total_features: int = 0
    passing_features: int = 0
    failing_features: int = 0

    def to_cypher_properties(self) -> dict[str, Any]:
        """Convert to Neo4j property dict."""
        return {
            "id": self.id,
            "project_name": self.project_name,
            "current_feature_id": self.current_feature_id,
            "environment": str(self.environment),
            "constraints": self.constraints,
            "created_at": self.created_at.isoformat(),
            "last_worker_run": self.last_worker_run.isoformat() if self.last_worker_run else None,
            "last_test_run": self.last_test_run.isoformat() if self.last_test_run else None,
            "total_features": self.total_features,
            "passing_features": self.passing_features,
            "failing_features": self.failing_features
        }

    @classmethod
    def from_cypher_record(cls, record: dict[str, Any]) -> 'DomainState':
        """Create from Neo4j query result."""
        return cls(
            id=record["id"],
            project_name=record["project_name"],
            current_feature_id=record.get("current_feature_id"),
            environment=eval(record.get("environment", "{}")),
            constraints=record.get("constraints", []),
            created_at=datetime.fromisoformat(record["created_at"]),
            last_worker_run=datetime.fromisoformat(record["last_worker_run"]) if record.get("last_worker_run") else None,
            last_test_run=datetime.fromisoformat(record["last_test_run"]) if record.get("last_test_run") else None,
            total_features=record.get("total_features", 0),
            passing_features=record.get("passing_features", 0),
            failing_features=record.get("failing_features", 0)
        )

    def completion_rate(self) -> float:
        """Percentage of features passing."""
        if self.total_features == 0:
            return 0.0
        return self.passing_features / self.total_features


@dataclass
class TestResult:
    """
    Record of a test execution.
    
    Linked to Feature and ProgressEntry.
    """
    feature_id: str
    passed: bool
    output: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0

    # Test details
    test_command: str | None = None
    exit_code: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_cypher_properties(self) -> dict[str, Any]:
        """Convert to Neo4j property dict."""
        return {
            "id": self.id,
            "feature_id": self.feature_id,
            "passed": self.passed,
            "output": self.output[:5000],  # Truncate long output
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "test_command": self.test_command,
            "exit_code": self.exit_code,
            "metadata": str(self.metadata)
        }

    @classmethod
    def from_cypher_record(cls, record: dict[str, Any]) -> 'TestResult':
        """Create from Neo4j query result."""
        return cls(
            id=record["id"],
            feature_id=record["feature_id"],
            passed=record["passed"],
            output=record["output"],
            timestamp=datetime.fromisoformat(record["timestamp"]),
            duration_ms=record.get("duration_ms", 0),
            test_command=record.get("test_command"),
            exit_code=record.get("exit_code", 0),
            metadata=eval(record.get("metadata", "{}"))
        )


# ============================================================================
# Relationship Types
# ============================================================================

@dataclass
class DependsOn:
    """Feature dependency relationship."""
    from_feature_id: str
    to_feature_id: str
    optional: bool = False
    description: str | None = None

    def to_cypher(self) -> str:
        """Cypher MERGE statement."""
        props = f"optional: {str(self.optional).lower()}"
        if self.description:
            props += f", description: '{self.description}'"
        return f"""
        MATCH (a:Feature {{id: '{self.from_feature_id}'}})
        MATCH (b:Feature {{id: '{self.to_feature_id}'}})
        MERGE (a)-[:DEPENDS_ON {{{props}}}]->(b)
        """


@dataclass
class RecordsProgress:
    """Link from ProgressEntry to Feature."""
    progress_id: str
    feature_id: str

    def to_cypher(self) -> str:
        """Cypher MERGE statement."""
        return f"""
        MATCH (p:ProgressEntry {{id: '{self.progress_id}'}})
        MATCH (f:Feature {{id: '{self.feature_id}'}})
        MERGE (p)-[:RECORDS_PROGRESS_ON]->(f)
        """


# ============================================================================
# Context Types (for protocol)
# ============================================================================

@dataclass
class DomainContext:
    """
    Context loaded during boot ritual.
    
    Everything a worker needs to act.
    """
    state: DomainState
    target: Feature | None
    pending_features: list[Feature]
    recent_progress: list[ProgressEntry]

    # Computed
    blocked_features: list[str] = field(default_factory=list)

    def has_work(self) -> bool:
        """Is there actionable work?"""
        return self.target is not None


@dataclass
class ActionResult:
    """
    Result of worker action ritual.
    """
    feature_id: str
    success: bool
    summary: str

    # Code changes
    files_changed: list[str] = field(default_factory=list)
    diff_hash: str | None = None

    # Test result
    test_result: TestResult | None = None

    # Timing
    duration_ms: int = 0

    # Worker identity
    worker_id: str | None = None


# ============================================================================
# Cypher Queries
# ============================================================================

class DomainCypher:
    """Cypher query templates for domain memory operations."""

    # Schema setup
    CREATE_CONSTRAINTS = """
    CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (f:Feature) REQUIRE f.id IS UNIQUE;
    CREATE CONSTRAINT progress_id IF NOT EXISTS FOR (p:ProgressEntry) REQUIRE p.id IS UNIQUE;
    CREATE CONSTRAINT state_id IF NOT EXISTS FOR (s:DomainState) REQUIRE s.id IS UNIQUE;
    CREATE CONSTRAINT test_id IF NOT EXISTS FOR (t:TestResult) REQUIRE t.id IS UNIQUE;
    """

    CREATE_INDEXES = """
    CREATE INDEX feature_status IF NOT EXISTS FOR (f:Feature) ON (f.status);
    CREATE INDEX feature_priority IF NOT EXISTS FOR (f:Feature) ON (f.priority);
    CREATE INDEX progress_timestamp IF NOT EXISTS FOR (p:ProgressEntry) ON (p.timestamp);
    CREATE INDEX progress_feature IF NOT EXISTS FOR (p:ProgressEntry) ON (p.feature_id);
    """

    # Feature queries
    GET_ACTIONABLE_FEATURES = """
    MATCH (f:Feature)
    WHERE f.status IN ['pending', 'failing']
    OPTIONAL MATCH (f)-[:DEPENDS_ON]->(dep:Feature)
    WHERE dep.status <> 'passing'
    WITH f, collect(dep.id) as blocking_deps
    WHERE size(blocking_deps) = 0
    RETURN f
    ORDER BY f.priority DESC, f.created_at ASC
    """

    GET_FEATURE_BY_ID = """
    MATCH (f:Feature {id: $feature_id})
    OPTIONAL MATCH (f)-[:DEPENDS_ON]->(dep:Feature)
    OPTIONAL MATCH (blocked:Feature)-[:DEPENDS_ON]->(f)
    RETURN f, collect(DISTINCT dep.id) as depends_on, collect(DISTINCT blocked.id) as blocks
    """

    UPDATE_FEATURE_STATUS = """
    MATCH (f:Feature {id: $feature_id})
    SET f.status = $status, f.updated_at = datetime()
    RETURN f
    """

    # Progress queries
    GET_RECENT_PROGRESS = """
    MATCH (p:ProgressEntry)
    RETURN p
    ORDER BY p.timestamp DESC
    LIMIT $limit
    """

    GET_FEATURE_PROGRESS = """
    MATCH (p:ProgressEntry {feature_id: $feature_id})
    RETURN p
    ORDER BY p.timestamp DESC
    """

    # State queries
    GET_OR_CREATE_STATE = """
    MERGE (s:DomainState {id: 'singleton'})
    ON CREATE SET s.project_name = $project_name, s.created_at = datetime()
    RETURN s
    """

    UPDATE_STATE = """
    MATCH (s:DomainState {id: 'singleton'})
    SET s += $properties
    RETURN s
    """

    # Metrics
    GET_FEATURE_COUNTS = """
    MATCH (f:Feature)
    RETURN f.status as status, count(f) as count
    """


# ============================================================================
# Factory
# ============================================================================

def create_feature(
    name: str,
    description: str,
    depends_on: list[str] | None = None,
    **kwargs
) -> Feature:
    """Factory for creating features with defaults."""
    feature = Feature(name=name, description=description, **kwargs)
    if depends_on:
        feature.depends_on = depends_on
    return feature


def create_progress_entry(
    feature: Feature,
    action: ProgressAction,
    summary: str,
    **kwargs
) -> ProgressEntry:
    """Factory for creating progress entries."""
    return ProgressEntry(
        feature_id=feature.id,
        action=action,
        summary=summary,
        **kwargs
    )

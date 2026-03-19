"""
Memory Bus Configuration — spec §3, §6.2, §9.2, §10.
"""

from dataclasses import dataclass
from enum import Enum


class PressureTier(Enum):
    """Memory pressure tiers from spec §3.2."""
    NOMINAL = 0     # utilization < 50%
    ELEVATED = 1    # 50-75%
    HIGH = 2        # 75-90%
    CRITICAL = 3    # > 90%


class ResolutionPath(Enum):
    """Query resolution levels from spec §4.2."""
    EXACT = "exact"           # Key-value lookup by entity ID
    STRUCTURED = "structured" # Cypher/SQL deterministic query
    GRAPH = "graph"           # Neo4j traversal, relationship-aware
    SEMANTIC = "semantic"     # Qdrant similarity (DEFERRED)


class MemoryType(Enum):
    """Memory taxonomy from spec §1.1."""
    EPISODIC = "episodic"
    ENTITY = "entity"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    PLAN = "plan"
    CONFIG = "config"
    ARTIFACT = "artifact"


class AuditAction(Enum):
    """Audit log action types from spec §10.3."""
    QUERY = "query"
    PROMOTE = "promote"
    DECAY = "decay"
    ARCHIVE = "archive"
    CONFLICT = "conflict"
    INCONSISTENCY = "inconsistency"


@dataclass
class MemoryBusConfig:
    """Configuration for the Memory Bus.

    Defaults match docker-compose.yml service configuration.
    """
    # Neo4j connection
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_auth: tuple[str, str] = ("neo4j", "hololoom123")
    neo4j_database: str = "neo4j"

    # Postgres connection (separate DB from mem0)
    postgres_dsn: str = "postgresql://hololoom:hololoom123@localhost:5432/hololoom"

    # Token budgets (spec §6.2)
    token_budget_pct: float = 0.15       # 15% of context window
    token_budget_absolute: int = 8000    # fallback absolute cap
    max_tokens_per_result: int = 200     # per-item cap at CRITICAL tier

    # Pressure thresholds (spec §3.2)
    pressure_tier_1: float = 0.50        # ELEVATED
    pressure_tier_2: float = 0.75        # HIGH
    pressure_tier_3: float = 0.90        # CRITICAL

    # Query quality target (spec §9.2)
    max_semantic_fallback_pct: float = 0.20  # alert if >20% semantic

    # Promotion (spec §2.2)
    novelty_threshold: float = 0.3
    significance_threshold: float = 0.4

    # Query defaults
    default_max_results: int = 5
    default_min_importance: float = 0.1

    def pressure_tier_for(self, utilization: float) -> PressureTier:
        """Determine pressure tier from context utilization."""
        if utilization >= self.pressure_tier_3:
            return PressureTier.CRITICAL
        elif utilization >= self.pressure_tier_2:
            return PressureTier.HIGH
        elif utilization >= self.pressure_tier_1:
            return PressureTier.ELEVATED
        return PressureTier.NOMINAL

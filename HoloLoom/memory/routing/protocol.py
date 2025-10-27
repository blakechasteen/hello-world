"""
Routing Strategy Protocol - Interface for Backend Selection
===========================================================

Protocol-based design following HoloLoom philosophy.
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class BackendType(Enum):
    """Available backend types."""
    NEO4J = "neo4j"          # Graph relationships
    QDRANT = "qdrant"        # Vector similarity
    MEM0 = "mem0"           # Intelligent extraction
    INMEMORY = "inmemory"   # Fast cache
    HYBRID = "hybrid"       # Fusion of multiple


class QueryType(Enum):
    """Types of queries for routing."""
    RELATIONSHIP = "relationship"  # Who/what/where/when
    SIMILARITY = "similarity"      # Find similar content
    PERSONAL = "personal"          # User-specific
    TEMPORAL = "temporal"          # Recent/time-based
    PATTERN = "pattern"            # Mathematical patterns
    UNKNOWN = "unknown"            # Needs classification


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    backend_type: BackendType
    confidence: float                    # [0, 1] confidence in this decision
    query_type: QueryType                # Classified query type
    reasoning: str                       # Why this backend was chosen
    alternatives: List[BackendType]      # Other viable backends
    metadata: Dict[str, Any]             # Additional context


@dataclass
class RoutingOutcome:
    """Feedback from routing decision (for learning)."""
    decision: RoutingDecision
    query: str
    result_count: int
    avg_relevance: float                 # [0, 1] average relevance score
    latency_ms: float
    user_feedback: Optional[float] = None  # Optional user satisfaction
    timestamp: Optional[str] = None


class RoutingStrategy(Protocol):
    """
    Protocol for backend routing strategies.

    Any routing strategy (rule-based, ML-based, hybrid) must implement this.
    """

    def select_backend(
        self,
        query: str,
        available_backends: List[BackendType],
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Select optimal backend for query.

        Args:
            query: Query text
            available_backends: Which backends are available
            context: Optional additional context (user_id, session, etc.)

        Returns:
            RoutingDecision with backend choice and reasoning
        """
        ...

    def record_outcome(self, outcome: RoutingOutcome):
        """
        Record outcome of routing decision (for learning).

        Args:
            outcome: Feedback from query execution
        """
        ...

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get routing performance statistics.

        Returns:
            Dict with accuracy, latency, backend distribution, etc.
        """
        ...


class LearnableStrategy(Protocol):
    """
    Extended protocol for strategies that can learn/optimize.
    """

    def train(self, outcomes: List[RoutingOutcome]):
        """
        Train/update strategy from historical outcomes.

        Args:
            outcomes: Historical routing outcomes
        """
        ...

    def save(self, path: str):
        """Save learned parameters."""
        ...

    def load(self, path: str):
        """Load learned parameters."""
        ...


class ExperimentalStrategy(Protocol):
    """
    Protocol for A/B testing multiple strategies.
    """

    def add_strategy(
        self,
        name: str,
        strategy: RoutingStrategy,
        weight: float = 1.0
    ):
        """
        Add strategy variant to experiment.

        Args:
            name: Strategy identifier
            strategy: RoutingStrategy implementation
            weight: Probability weight for selection
        """
        ...

    def get_winner(self) -> str:
        """
        Get best-performing strategy based on outcomes.

        Returns:
            Name of winning strategy
        """
        ...
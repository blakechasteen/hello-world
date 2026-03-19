"""
Memory Symphony Protocol Definitions
=====================================

Protocol definitions for unified memory coordination.

Coordinates across 7 memory systems:
1. Knowledge Graph (Yarn Graph) - Discrete symbolic memory
2. Vector Memory - Semantic embeddings
3. Query Cache - 100x speedup for repeated queries
4. Hot Pattern Feedback - Usage-based adaptation
5. Awareness Graph - Activation tracking
6. Spring Dynamics - Physics-based connectivity
7. Multi-Wave Engine - Temporal propagation

Author: Claude Code
Date: 2025-11-22
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class MemoryStrategy(str, Enum):
    """Memory access strategies."""
    FAST = "fast"                    # Cache-first, minimal graph traversal
    BALANCED = "balanced"            # Hybrid: cache + vector + graph
    DEEP = "deep"                    # Full graph traversal + spreading activation
    RESEARCH = "research"            # Maximum exploration, all systems
    AUTO = "auto"                    # Automatic strategy selection


class MemorySystem(str, Enum):
    """Individual memory systems."""
    QUERY_CACHE = "query_cache"
    VECTOR_MEMORY = "vector_memory"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    HOT_PATTERNS = "hot_patterns"
    AWARENESS_GRAPH = "awareness_graph"
    SPRING_DYNAMICS = "spring_dynamics"
    MULTI_WAVE = "multi_wave"


@dataclass
class MemoryQuery:
    """Unified memory query."""
    text: str
    k: int = 10                      # Number of results to retrieve
    strategy: MemoryStrategy = MemoryStrategy.AUTO
    min_relevance: float = 0.0       # Minimum relevance threshold
    include_metadata: bool = True
    enable_spreading: bool = True    # Enable activation spreading
    max_hops: int = 3                # For graph traversal
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryResult:
    """Result from memory retrieval."""
    node_id: str
    content: str
    relevance: float                 # 0.0-1.0
    source_system: MemorySystem      # Which system provided this
    activation: float = 0.0          # Activation level (if applicable)
    heat: float = 0.0                # Heat score (if applicable)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCoordinationResult:
    """Result from coordinated memory access."""
    results: list[MemoryResult]
    strategy_used: MemoryStrategy
    systems_accessed: list[MemorySystem]
    total_latency_ms: float
    cache_hit: bool
    coordination_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryPerformanceMetrics:
    """Performance metrics for memory symphony."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    avg_results_per_query: float = 0.0
    strategy_usage: dict[MemoryStrategy, int] = field(default_factory=dict)
    system_usage: dict[MemorySystem, int] = field(default_factory=dict)


@runtime_checkable
class MemoryConductorProtocol(Protocol):
    """Protocol for memory conductor (main orchestrator)."""

    async def recall(
        self,
        query: MemoryQuery
    ) -> MemoryCoordinationResult:
        """
        Unified memory recall across all systems.

        Automatically selects optimal strategy and coordinates
        across multiple memory systems.
        """
        ...

    def select_strategy(
        self,
        query: MemoryQuery
    ) -> MemoryStrategy:
        """
        Select optimal memory access strategy based on query characteristics.

        Considers:
        - Query complexity
        - Cache availability
        - Required depth
        - Performance requirements
        """
        ...

    def get_performance_metrics(self) -> MemoryPerformanceMetrics:
        """Get performance metrics across all systems."""
        ...


@runtime_checkable
class MemorySystemProtocol(Protocol):
    """Protocol for individual memory systems."""

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        **kwargs
    ) -> list[MemoryResult]:
        """Retrieve memories from this system."""
        ...

    def get_latency_estimate(self) -> float:
        """Estimate latency for this system (ms)."""
        ...

    def get_coverage_score(self, query: str) -> float:
        """Estimate coverage/relevance for this query (0.0-1.0)."""
        ...


@dataclass
class StrategySelectionCriteria:
    """Criteria for automatic strategy selection."""
    query_length: int
    is_cached: bool
    requires_deep_traversal: bool    # Complex/research queries
    performance_critical: bool       # Latency-sensitive
    exploration_mode: bool           # Research vs focused retrieval


@dataclass
class CoordinationPlan:
    """Plan for coordinating multiple memory systems."""
    strategy: MemoryStrategy
    systems_to_query: list[MemorySystem]
    parallel_execution: bool         # Execute in parallel vs sequential
    fallback_systems: list[MemorySystem]
    estimated_latency_ms: float
    expected_coverage: float         # 0.0-1.0


# Convenience type aliases
MemoryQueryResult = MemoryResult
SystemAccessOrder = list[MemorySystem]

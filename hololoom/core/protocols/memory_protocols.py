"""
Memory Protocols - Protocol definitions for memory systems
===========================================================

All memory-related protocol definitions (MemoryStore, MemoryNavigator, PatternDetector)

This module contains the protocol interfaces for memory systems.
Data types are in memory_types.py to avoid circular imports.

Philosophy:
- Define WHAT, not HOW (protocols define interface, not implementation)
- Dependency injection (components don't know concrete implementations)
- Protocol-based design (type-safe duck typing)
- Async-first (non-blocking operations)

Author: HoloLoom Architecture Team
Date: 2025-01-12 (Phase 0, Task 7: Protocol Consolidation)
"""

from typing import Any, Protocol, runtime_checkable

# Import types from memory_types (no circular dependency)
from .memory_types import Memory, MemoryQuery, MemoryRetrievalResult, Strategy

# ============================================================================
# Core Storage Protocol
# ============================================================================

@runtime_checkable
class MemoryStore(Protocol):
    """
    Protocol for memory storage backends.

    All memory stores (Mem0, Neo4j, Qdrant, HoloLoom) implement this.
    The orchestrator doesn't know which implementation it's using.

    Examples of implementations:
    - Mem0MemoryStore: User-specific, LLM-extracted memories
    - Neo4jMemoryStore: Graph-based, thread-model storage
    - QdrantMemoryStore: Vector-based similarity search
    - HoloLoomMemoryStore: Multi-scale, domain-aware retrieval
    - HybridMemoryStore: Fusion of multiple stores

    Design Principles:
    - Async-first for non-blocking I/O
    - Simple interface (store, recall, forget)
    - Result metadata for observability
    - Graceful failure handling
    """

    async def store(self, memory: Memory, user_id: str = "default") -> str:
        """
        Store a memory.

        Args:
            memory: Memory object to store
            user_id: User identifier

        Returns:
            memory_id: Unique identifier for stored memory
        """
        ...

    async def store_many(self, memories: list[Memory], user_id: str = "default") -> list[str]:
        """
        Store multiple memories efficiently.

        Args:
            memories: List of Memory objects
            user_id: User identifier

        Returns:
            List of memory_ids in same order
        """
        ...

    async def get_by_id(self, memory_id: str) -> Memory | None:
        """
        Retrieve memory by ID.

        Args:
            memory_id: Unique identifier

        Returns:
            Memory object if found, None otherwise
        """
        ...

    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> MemoryRetrievalResult:
        """
        Retrieve memories matching query.

        Args:
            query: MemoryQuery with search criteria
            strategy: Retrieval strategy to use

        Returns:
            MemoryRetrievalResult with matches and metadata
        """
        ...

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Unique identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    async def health_check(self) -> dict[str, Any]:
        """
        Check backend health and return diagnostics.

        Returns:
            Dict with health status and metrics
        """
        ...


# ============================================================================
# Navigation Protocol
# ============================================================================

@runtime_checkable
class MemoryNavigator(Protocol):
    """
    Protocol for spatial/relational memory navigation.

    Enables graph-like traversal and spatial operations:
    - Follow relationships (forward/backward)
    - Find paths between concepts
    - Discover neighborhoods
    - Compute similarity/distance

    Examples:
    - Neo4j: Native graph traversal
    - Qdrant: Vector space navigation
    - Mem0: Entity relationship graphs

    Inspiration:
    - Hofstadter's "strange loops" (recursive self-reference)
    - Semantic networks and spreading activation
    - Graph neural networks (message passing)
    """

    async def navigate_forward(
        self,
        memory_id: str,
        relation_type: str | None = None
    ) -> list[str]:
        """
        Navigate forward from memory (outgoing edges).

        Args:
            memory_id: Starting memory
            relation_type: Optional relation filter (e.g., "causes", "implies")

        Returns:
            List of connected memory IDs
        """
        ...

    async def navigate_backward(
        self,
        memory_id: str,
        relation_type: str | None = None
    ) -> list[str]:
        """
        Navigate backward to memory (incoming edges).

        Args:
            memory_id: Target memory
            relation_type: Optional relation filter

        Returns:
            List of memory IDs pointing to this one
        """
        ...

    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> list[str] | None:
        """
        Find shortest path between two memories.

        Args:
            start_id: Starting memory
            end_id: Target memory
            max_depth: Maximum path length

        Returns:
            List of memory IDs forming path, or None if no path exists
        """
        ...

    async def get_neighborhood(
        self,
        memory_id: str,
        radius: int = 1
    ) -> list[str]:
        """
        Get neighborhood of memory within radius.

        Args:
            memory_id: Center memory
            radius: Number of hops (1 = immediate neighbors)

        Returns:
            List of memory IDs in neighborhood
        """
        ...

    async def compute_similarity(
        self,
        memory_id1: str,
        memory_id2: str
    ) -> float:
        """
        Compute similarity between two memories.

        Args:
            memory_id1: First memory
            memory_id2: Second memory

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        ...


# ============================================================================
# Pattern Detection Protocol
# ============================================================================

@runtime_checkable
class PatternDetector(Protocol):
    """
    Protocol for discovering patterns and structure in memories.

    Enables:
    - Clustering similar memories
    - Extracting common themes
    - Detecting temporal patterns
    - Finding anomalies

    Examples:
    - SpectraDetector: Graph spectral clustering
    - LLMPatternMiner: LLM-based theme extraction
    - TemporalPatternFinder: Time-series analysis

    Inspiration:
    - Unsupervised learning (clustering, dimensionality reduction)
    - Topic modeling (LDA, NMF)
    - Anomaly detection
    """

    async def discover_patterns(
        self,
        memory_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Discover patterns across memories.

        Args:
            memory_ids: Optional subset to analyze (None = all memories)

        Returns:
            List of discovered patterns with metadata:
            [
                {
                    "type": "cluster",
                    "name": "beekeeping_winter",
                    "member_ids": ["mem_1", "mem_2", ...],
                    "strength": 0.85,
                    "description": "Winter beekeeping preparation"
                },
                ...
            ]
        """
        ...

    async def find_anomalies(
        self,
        memory_ids: list[str] | None = None,
        threshold: float = 0.95
    ) -> list[str]:
        """
        Find memories that don't fit established patterns.

        Args:
            memory_ids: Optional subset to analyze
            threshold: Anomaly threshold (higher = more unusual required)

        Returns:
            List of anomalous memory IDs
        """
        ...

    async def extract_themes(
        self,
        memory_ids: list[str] | None = None,
        n_themes: int = 5
    ) -> list[dict[str, Any]]:
        """
        Extract common themes across memories.

        Args:
            memory_ids: Optional subset to analyze
            n_themes: Number of themes to extract

        Returns:
            List of themes with representative memories:
            [
                {
                    "theme": "apiary_management",
                    "keywords": ["hive", "frames", "inspection"],
                    "exemplar_ids": ["mem_5", "mem_12"],
                    "coverage": 0.35  # fraction of memories
                },
                ...
            ]
        """
        ...


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'MemoryStore',
    'MemoryNavigator',
    'PatternDetector',
]

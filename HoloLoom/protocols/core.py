"""
HoloLoom Canonical Protocols
=============================
Central protocol definitions for the entire HoloLoom system.

This is the SINGLE SOURCE OF TRUTH for all protocol definitions.
All modules should import from here, not define their own.

Philosophy:
- Define WHAT, not HOW (protocols define interface, not implementation)
- Dependency injection (components don't know concrete implementations)
- Protocol-based design (type-safe duck typing)
- Async-first (non-blocking operations)

Protocols Defined:
- MemoryStore: Storage and retrieval operations
- MemoryNavigator: Spatial/relational navigation
- PatternDetector: Structure discovery and mining

Usage:
    from HoloLoom.protocols import MemoryStore, MemoryNavigator, PatternDetector

    class MyMemoryBackend(MemoryStore):
        async def store(self, memory: Memory) -> str:
            ...
        async def recall(self, query: MemoryQuery) -> RetrievalResult:
            ...

Author: HoloLoom Architecture Team
Date: 2025-10-27
"""

from typing import List, Dict, Optional, Protocol, runtime_checkable, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


# Import core data types (these stay in memory.protocol for now)
# This avoids circular dependencies
try:
    from HoloLoom.memory.protocol import Memory, MemoryQuery, RetrievalResult
except ImportError:
    # Fallback for when protocol.py tries to import from us
    Memory = Any
    MemoryQuery = Any
    RetrievalResult = Any


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

    async def store(self, memory) -> str:
        """
        Store a memory.

        Args:
            memory: Memory object to store

        Returns:
            memory_id: Unique identifier for stored memory
        """
        ...

    async def store_many(self, memories: List) -> List[str]:
        """
        Store multiple memories efficiently.

        Args:
            memories: List of Memory objects

        Returns:
            List of memory_ids in same order
        """
        ...

    async def recall(self, query) -> Any:
        """
        Retrieve memories matching query.

        Args:
            query: MemoryQuery with search criteria

        Returns:
            RetrievalResult with matches and metadata
        """
        ...

    async def forget(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Unique identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update memory fields.

        Args:
            memory_id: Memory to update
            updates: Dict of field updates

        Returns:
            True if updated, False if not found
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

    async def navigate_forward(self, memory_id: str, relation_type: Optional[str] = None) -> List[str]:
        """
        Navigate forward from memory (outgoing edges).

        Args:
            memory_id: Starting memory
            relation_type: Optional relation filter (e.g., "causes", "implies")

        Returns:
            List of connected memory IDs
        """
        ...

    async def navigate_backward(self, memory_id: str, relation_type: Optional[str] = None) -> List[str]:
        """
        Navigate backward to memory (incoming edges).

        Args:
            memory_id: Target memory
            relation_type: Optional relation filter

        Returns:
            List of memory IDs pointing to this one
        """
        ...

    async def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> Optional[List[str]]:
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

    async def get_neighborhood(self, memory_id: str, radius: int = 1) -> List[str]:
        """
        Get neighborhood of memory within radius.

        Args:
            memory_id: Center memory
            radius: Number of hops (1 = immediate neighbors)

        Returns:
            List of memory IDs in neighborhood
        """
        ...

    async def compute_similarity(self, memory_id1: str, memory_id2: str) -> float:
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

    async def discover_patterns(self, memory_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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

    async def find_anomalies(self, memory_ids: Optional[List[str]] = None, threshold: float = 0.95) -> List[str]:
        """
        Find memories that don't fit established patterns.

        Args:
            memory_ids: Optional subset to analyze
            threshold: Anomaly threshold (higher = more unusual required)

        Returns:
            List of anomalous memory IDs
        """
        ...

    async def extract_themes(self, memory_ids: Optional[List[str]] = None, n_themes: int = 5) -> List[Dict[str, Any]]:
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

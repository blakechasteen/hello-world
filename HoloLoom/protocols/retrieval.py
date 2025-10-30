"""
HoloLoom Retrieval Protocols
=============================
Protocol definitions for memory retrieval strategies.

This defines HOW memories are retrieved, not WHERE they're stored.
Different strategies can be swapped without changing the orchestrator.

Strategies:
- StaticRetrieval: Traditional BM25 + cosine similarity
- SpringActivation: Physics-based spreading activation
- HybridRetrieval: Combine embedding seeds + spring expansion

Philosophy:
- Strategy pattern for retrieval
- Modular, swappable implementations
- A/B testable
- Graceful degradation

Author: HoloLoom Architecture Team
Date: 2025-10-29
"""

from typing import List, Protocol, runtime_checkable
from dataclasses import dataclass


# Import core types
try:
    from HoloLoom.documentation.types import Query, MemoryShard
except ImportError:
    # Fallback for circular imports
    from typing import Any
    Query = Any
    MemoryShard = Any


# ============================================================================
# Core Retrieval Protocol
# ============================================================================

@runtime_checkable
class RetrievalStrategy(Protocol):
    """
    Protocol for memory retrieval strategies.

    All retrieval implementations (static, spring, hybrid) implement this.
    The orchestrator doesn't know which strategy it's using.

    Examples of implementations:
    - StaticRetrieval: BM25 + cosine similarity (default)
    - SpringActivation: Energy-based spreading activation
    - HybridRetrieval: Embedding seeds + graph expansion
    - SemanticZoom: Multi-scale retrieval at different granularities

    Design Principles:
    - Async-first for non-blocking retrieval
    - Simple interface (retrieve + metadata)
    - Confidence scores for result quality
    - Graceful failure handling
    """

    async def retrieve(
        self,
        query: Query,
        k: int = 5,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Retrieve memories relevant to query.

        Args:
            query: Query object with text and optional metadata
            k: Number of results to retrieve
            **kwargs: Strategy-specific parameters

        Returns:
            List of MemoryShard objects (up to k items)

        Notes:
            - Results should be ordered by relevance (most relevant first)
            - May return fewer than k results if not enough matches
            - Each shard should have a confidence/score if possible
        """
        ...

    async def retrieve_with_metadata(
        self,
        query: Query,
        k: int = 5,
        **kwargs
    ) -> 'RetrievalResult':
        """
        Retrieve memories with detailed metadata.

        Args:
            query: Query object
            k: Number of results
            **kwargs: Strategy-specific parameters

        Returns:
            RetrievalResult with shards + metadata

        Metadata includes:
            - Retrieval time (ms)
            - Strategy used
            - Confidence scores
            - Cache hit/miss info
            - Debug information
        """
        ...


# ============================================================================
# Result Data Structure
# ============================================================================

@dataclass
class RetrievalResult:
    """
    Rich retrieval result with metadata.

    Provides observability into retrieval process:
    - What was retrieved
    - How confident we are
    - How long it took
    - What strategy was used
    """

    # Core results
    shards: List[MemoryShard]

    # Metadata
    strategy: str                    # "static", "spring", "hybrid"
    query_text: str                 # Original query
    k_requested: int                # How many requested
    k_returned: int                 # How many actually returned

    # Timing
    retrieval_time_ms: float        # How long retrieval took

    # Confidence
    avg_confidence: float           # Average confidence across results
    min_confidence: float           # Lowest confidence result
    max_confidence: float           # Highest confidence result

    # Strategy-specific metadata
    metadata: dict                  # Additional strategy-specific info

    # Examples of metadata:
    # Static: {"cache_hit": True, "bm25_scores": [...]}
    # Spring: {"iterations": 47, "energy": 0.23, "activated_nodes": 12}
    # Hybrid: {"seed_count": 3, "expansion_depth": 2}


# ============================================================================
# Spring-Specific Types
# ============================================================================

@dataclass
class SpringActivationMetadata:
    """
    Metadata specific to spring activation retrieval.

    Provides insights into physics simulation:
    - Convergence behavior
    - Energy landscape
    - Activated node count
    """

    # Physics convergence
    iterations: int                 # How many steps to converge
    converged: bool                # Did it reach equilibrium?
    final_energy: float            # Final energy state

    # Activation spread
    seed_nodes: List[str]          # Initial activated nodes
    activated_count: int           # Total nodes activated above threshold
    activation_threshold: float    # Min activation to retrieve

    # Node activations
    node_activations: dict         # {node_id: activation_level}

    # Timing breakdown
    embedding_time_ms: float       # Time to find seed nodes
    propagation_time_ms: float     # Time for spring simulation
    shard_retrieval_time_ms: float # Time to get shards from activated nodes


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'RetrievalStrategy',
    'RetrievalResult',
    'SpringActivationMetadata',
]

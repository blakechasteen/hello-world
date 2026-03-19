"""
Awareness System Protocol Definitions
======================================

Core protocols and types for the Context Packing awareness system.

Author: Claude Code
Date: 2025-11-22
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class ImportanceSignal(str, Enum):
    """Types of importance signals for context ranking."""
    RECENCY = "recency"              # How recently accessed
    RELEVANCE = "relevance"          # Semantic similarity to query
    CENTRALITY = "centrality"        # Graph centrality (PageRank-style)
    ACCESS_FREQUENCY = "frequency"   # How often accessed
    CONFIDENCE = "confidence"        # Historical confidence scores
    HEAT = "heat"                    # Hot pattern feedback score
    INFORMATION_CONTENT = "information"  # Mutual information I(Node; Query) - Phase 5


@dataclass
class ActivationState:
    """
    Current activation state of a memory node.

    Attributes:
        node_id: Unique identifier for the memory node
        activation: Current activation level (0.0-1.0)
        last_accessed: Timestamp of last access
        access_count: Total number of accesses
        heat: Hot pattern heat score
        importance_scores: Dict of signal_type -> score
    """
    node_id: str
    activation: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    heat: float = 0.0
    importance_scores: dict[ImportanceSignal, float] = None

    def __post_init__(self):
        if self.importance_scores is None:
            self.importance_scores = {}


@dataclass
class CompressionResult:
    """
    Result of context compression operation.

    Attributes:
        compressed_nodes: List of node IDs kept after compression
        original_count: Number of nodes before compression
        compressed_count: Number of nodes after compression
        compression_ratio: Ratio of reduction (e.g., 0.6 = 40% savings)
        token_savings: Estimated token count saved
        importance_threshold: Threshold used for compression
    """
    compressed_nodes: list[str]
    original_count: int
    compressed_count: int
    compression_ratio: float
    token_savings: int
    importance_threshold: float


@dataclass
class BetaWave:
    """
    Beta wave activation spreading parameters.

    Beta waves (12-30 Hz in neuroscience) represent alert, focused attention.
    In our system, they spread activation across the knowledge graph.

    Attributes:
        frequency: Wave frequency (Hz)
        amplitude: Initial wave amplitude (0.0-1.0)
        decay_rate: Exponential decay per hop (0.0-1.0)
        max_hops: Maximum propagation distance
        source_nodes: List of initial activation sources
    """
    frequency: float = 20.0  # Beta range: 12-30 Hz
    amplitude: float = 1.0
    decay_rate: float = 0.7  # 30% loss per hop
    max_hops: int = 3
    source_nodes: list[str] = None

    def __post_init__(self):
        if self.source_nodes is None:
            self.source_nodes = []


class ActivationSpreaderProtocol(Protocol):
    """Protocol for beta wave activation spreading."""

    def spread_activation(
        self,
        source_nodes: list[str],
        graph: Any,  # KG or NetworkX MultiDiGraph
        initial_activation: float = 1.0,
        max_hops: int = 3,
        decay_rate: float = 0.7
    ) -> dict[str, float]:
        """
        Spread activation from source nodes across graph.

        Args:
            source_nodes: Initial activation sources
            graph: Knowledge graph to propagate across
            initial_activation: Starting activation level
            max_hops: Maximum propagation distance
            decay_rate: Decay per hop (0.0-1.0)

        Returns:
            Dict mapping node_id -> final_activation
        """
        ...

    def get_activation_wave(
        self,
        wave: BetaWave,
        graph: Any
    ) -> dict[str, float]:
        """
        Generate activation pattern from beta wave.

        Args:
            wave: Beta wave parameters
            graph: Knowledge graph

        Returns:
            Dict mapping node_id -> activation
        """
        ...


class ImportanceScorerProtocol(Protocol):
    """Protocol for multi-signal importance scoring."""

    def score_importance(
        self,
        node_id: str,
        query: str,
        graph: Any,
        activation_state: ActivationState | None = None
    ) -> dict[ImportanceSignal, float]:
        """
        Compute multi-signal importance scores for a node.

        Args:
            node_id: Node to score
            query: Current query text
            graph: Knowledge graph
            activation_state: Optional current activation state

        Returns:
            Dict mapping ImportanceSignal -> score (0.0-1.0)
        """
        ...

    def aggregate_scores(
        self,
        scores: dict[ImportanceSignal, float],
        weights: dict[ImportanceSignal, float] | None = None
    ) -> float:
        """
        Aggregate multi-signal scores into single importance value.

        Args:
            scores: Dict of signal -> score
            weights: Optional custom weights per signal

        Returns:
            Aggregated importance score (0.0-1.0)
        """
        ...


class ContextCompressorProtocol(Protocol):
    """Protocol for Matryoshka-aware context compression."""

    def compress(
        self,
        nodes: list[str],
        importance_scores: dict[str, float],
        target_ratio: float = 0.5,
        preserve_top_k: int | None = None
    ) -> CompressionResult:
        """
        Compress context by selecting most important nodes.

        Args:
            nodes: List of node IDs to compress
            importance_scores: Dict mapping node_id -> importance
            target_ratio: Target compression ratio (0.5 = 50% reduction)
            preserve_top_k: Optional override to keep top K nodes

        Returns:
            CompressionResult with compressed nodes and metrics
        """
        ...

    def matryoshka_compress(
        self,
        nodes: list[str],
        embeddings: dict[str, Any],  # node_id -> embedding tensor
        scales: list[int] = [128, 256, 384]
    ) -> tuple[list[str], dict[str, int]]:
        """
        Multi-scale compression using Matryoshka embeddings.

        Different nodes compressed to different scales based on importance:
        - High importance: 384D (full detail)
        - Medium importance: 256D (moderate detail)
        - Low importance: 128D (minimal detail) or dropped

        Args:
            nodes: List of node IDs
            embeddings: Full embeddings for each node
            scales: Available scales in descending order

        Returns:
            Tuple of (kept_nodes, node_id -> assigned_scale)
        """
        ...


class ContextPackerProtocol(Protocol):
    """Protocol for main Context Packer orchestrator."""

    def pack(
        self,
        query: str,
        candidate_nodes: list[str],
        graph: Any,
        target_tokens: int = 2000
    ) -> CompressionResult:
        """
        Pack context to fit within token budget.

        Combines:
        1. Beta wave activation spreading
        2. Multi-signal importance scoring
        3. Matryoshka-aware compression

        Args:
            query: Current query text
            candidate_nodes: Initial set of candidate memory nodes
            graph: Knowledge graph
            target_tokens: Maximum token budget

        Returns:
            CompressionResult with optimally packed context
        """
        ...

    def adaptive_pack(
        self,
        query: str,
        candidate_nodes: list[str],
        graph: Any,
        min_tokens: int = 500,
        max_tokens: int = 4000
    ) -> CompressionResult:
        """
        Adaptively pack context within range.

        Finds optimal compression that maximizes information density
        while staying within token budget.

        Args:
            query: Current query text
            candidate_nodes: Initial candidates
            graph: Knowledge graph
            min_tokens: Minimum acceptable tokens
            max_tokens: Maximum acceptable tokens

        Returns:
            CompressionResult with adaptively packed context
        """
        ...


# Type aliases for convenience
ActivationMap = dict[str, float]
ImportanceMap = dict[str, float]
NodeEmbeddings = dict[str, Any]
ScaleAssignments = dict[str, int]

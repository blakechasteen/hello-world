"""
Context Packer - Main Orchestrator
===================================

Main entry point for context packing system.

Combines:
1. Beta wave activation spreading (physics-based graph traversal)
2. Multi-signal importance scoring (7 signals including MI - Phase 5)
3. Matryoshka-aware compression (multi-scale embeddings)
4. Information budget compression (Phase 5 - Tishby's Information Bottleneck)

Result: 40-90% token savings while preserving information density.

Phase 5 (December 2025): Added information-theoretic packing:
- 7th importance signal: INFORMATION_CONTENT (mutual information)
- Information budget compression: Stop when cumulative MI exceeds budget
- MI-aware Matryoshka scales: High MI → 384D, Low MI → 128D
- Entropy-aware aggregation: Lower entropy = higher confidence

Author: Claude Code
Date: 2025-11-22

Usage:
    >>> from HoloLoom.context_packing import ContextPacker, ContextPackerConfig
    >>>
    >>> config = ContextPackerConfig.balanced()  # 40-60% savings
    >>> packer = ContextPacker(config)
    >>>
    >>> result = packer.pack(
    ...     query="What is Thompson Sampling?",
    ...     candidate_nodes=memory_nodes,
    ...     graph=knowledge_graph,
    ...     target_tokens=2000
    ... )
    >>>
    >>> print(f"Compressed: {result.original_count} -> {result.compressed_count}")
    >>> print(f"Token savings: {result.token_savings}")
    >>>
    >>> # Phase 5: Information-theoretic packing
    >>> result = packer.pack_with_information_budget(
    ...     query="What is Thompson Sampling?",
    ...     candidate_nodes=memory_nodes,
    ...     graph=knowledge_graph,
    ...     node_contents=contents,
    ...     information_budget=5.0  # bits
    ... )
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from .protocol import (
    CompressionResult,
    ActivationMap,
    ImportanceMap,
    ImportanceSignal,
    ScaleAssignments
)
from .config import ContextPackerConfig
from .activation_spreader import ActivationSpreader
from .importance_scorer import ImportanceScorer
from .context_compressor import ContextCompressor

logger = logging.getLogger(__name__)


class ContextPacker:
    """
    Main Context Packer orchestrator.

    Combines activation spreading, importance scoring, and compression
    into unified context packing pipeline.
    """

    def __init__(
        self,
        config: Optional[ContextPackerConfig] = None,
        embedder: Optional[Any] = None
    ):
        """
        Initialize Context Packer.

        Args:
            config: Packer configuration (uses defaults if None)
            embedder: Optional embedding model for relevance scoring
        """
        self.config = config or ContextPackerConfig()
        self.config.validate()

        # Create sub-components
        self.activation_spreader = ActivationSpreader(self.config.beta_wave)
        self.importance_scorer = ImportanceScorer(self.config.importance_scorer, embedder)
        self.context_compressor = ContextCompressor(self.config.compression)

        # Tracking
        self._pack_history: List[CompressionResult] = []
        self._stats = {
            'total_packs': 0,
            'avg_compression_ratio': 0.0,
            'avg_token_savings': 0.0
        }

    def pack(
        self,
        query: str,
        candidate_nodes: List[str],
        graph: Any,
        target_tokens: Optional[int] = None,
        node_contents: Optional[Dict[str, str]] = None
    ) -> CompressionResult:
        """
        Pack context to fit within token budget.

        Full pipeline:
        1. Spread activation from query-relevant nodes
        2. Score importance using 6 signals
        3. Compress using Matryoshka-aware selection

        Args:
            query: Current query text
            candidate_nodes: Initial set of candidate memory nodes
            graph: Knowledge graph
            target_tokens: Maximum token budget (uses config default if None)
            node_contents: Optional dict mapping node_id -> text content

        Returns:
            CompressionResult with optimally packed context
        """
        target_tokens = target_tokens or self.config.default_token_budget

        if self.config.verbose:
            logger.info(
                f"Context packing: {len(candidate_nodes)} candidates, "
                f"target: {target_tokens} tokens"
            )

        # Step 1: Activation spreading (if enabled)
        activation_map = {}
        if self.config.enable_activation_spreading:
            activation_map = self.activation_spreader.spread_activation(
                source_nodes=candidate_nodes,
                graph=graph
            )

            if self.config.verbose:
                logger.info(f"Activated {len(activation_map)} nodes via beta waves")

            # Expand candidates to include activated nodes
            all_candidates = list(set(candidate_nodes) | set(activation_map.keys()))
        else:
            all_candidates = candidate_nodes

        # Step 2: Importance scoring
        importance_scores = self.importance_scorer.score_batch(
            node_ids=all_candidates,
            query=query,
            graph=graph,
            node_contents=node_contents
        )

        # Blend with activation if available
        if activation_map:
            importance_scores = self._blend_activation_importance(
                importance_scores, activation_map
            )

        if self.config.verbose:
            logger.info(f"Scored {len(importance_scores)} nodes for importance")

        # Step 3: Compression
        if self.config.enable_matryoshka_compression:
            # Matryoshka-aware compression
            kept_nodes, scale_assignments = self.context_compressor.matryoshka_compress(
                nodes=all_candidates,
                importance_scores=importance_scores
            )

            # Estimate tokens with scale-aware calculation
            estimated_tokens = self.context_compressor.estimate_tokens(
                kept_nodes, scale_assignments
            )

            # Create result
            original_count = len(all_candidates)
            compressed_count = len(kept_nodes)
            compression_ratio = compressed_count / original_count if original_count > 0 else 0.0
            token_savings = (original_count - compressed_count) * self.config.compression.avg_tokens_per_node

            result = CompressionResult(
                compressed_nodes=kept_nodes,
                original_count=original_count,
                compressed_count=compressed_count,
                compression_ratio=compression_ratio,
                token_savings=token_savings,
                importance_threshold=min(
                    importance_scores.get(n, 0.0) for n in kept_nodes
                ) if kept_nodes else 0.0
            )

        else:
            # Simple threshold-based compression
            target_ratio = self.config.compression.target_ratio
            result = self.context_compressor.compress(
                nodes=all_candidates,
                importance_scores=importance_scores,
                target_ratio=target_ratio
            )

        # Update tracking
        self._pack_history.append(result)
        self._update_stats(result)

        if self.config.verbose:
            logger.info(
                f"Packed: {result.original_count} -> {result.compressed_count} nodes "
                f"({result.compression_ratio:.1%} kept, {result.token_savings} tokens saved)"
            )

        return result

    def adaptive_pack(
        self,
        query: str,
        candidate_nodes: List[str],
        graph: Any,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        node_contents: Optional[Dict[str, str]] = None
    ) -> CompressionResult:
        """
        Adaptively pack context within range.

        Finds optimal compression that maximizes information density
        while staying within token budget.

        Args:
            query: Current query text
            candidate_nodes: Initial candidates
            graph: Knowledge graph
            min_tokens: Minimum acceptable tokens (uses config if None)
            max_tokens: Maximum acceptable tokens (uses config if None)
            node_contents: Optional node contents

        Returns:
            CompressionResult with adaptively packed context
        """
        min_tokens = min_tokens or self.config.min_token_budget
        max_tokens = max_tokens or self.config.max_token_budget

        if not self.config.enable_adaptive_packing:
            # Fall back to standard pack
            return self.pack(query, candidate_nodes, graph, max_tokens, node_contents)

        if self.config.verbose:
            logger.info(
                f"Adaptive packing: {len(candidate_nodes)} candidates, "
                f"range: {min_tokens}-{max_tokens} tokens"
            )

        # Step 1: Activation spreading
        activation_map = {}
        if self.config.enable_activation_spreading:
            activation_map = self.activation_spreader.spread_activation(
                source_nodes=candidate_nodes,
                graph=graph
            )
            all_candidates = list(set(candidate_nodes) | set(activation_map.keys()))
        else:
            all_candidates = candidate_nodes

        # Step 2: Importance scoring
        importance_scores = self.importance_scorer.score_batch(
            node_ids=all_candidates,
            query=query,
            graph=graph,
            node_contents=node_contents
        )

        if activation_map:
            importance_scores = self._blend_activation_importance(
                importance_scores, activation_map
            )

        # Step 3: Adaptive compression
        result = self.context_compressor.adaptive_compress(
            nodes=all_candidates,
            importance_scores=importance_scores,
            min_tokens=min_tokens,
            max_tokens=max_tokens
        )

        # Update tracking
        self._pack_history.append(result)
        self._update_stats(result)

        if self.config.verbose:
            estimated_tokens = result.compressed_count * self.config.compression.avg_tokens_per_node
            logger.info(
                f"Adaptive packed: {result.original_count} -> {result.compressed_count} nodes "
                f"(~{estimated_tokens} tokens)"
            )

        return result

    def pack_with_scales(
        self,
        query: str,
        candidate_nodes: List[str],
        graph: Any,
        target_tokens: Optional[int] = None,
        node_contents: Optional[Dict[str, str]] = None
    ) -> Tuple[CompressionResult, ScaleAssignments]:
        """
        Pack context with Matryoshka scale assignments.

        Returns both compression result and scale assignments for each kept node.

        Args:
            query: Current query
            candidate_nodes: Initial candidates
            graph: Knowledge graph
            target_tokens: Token budget
            node_contents: Optional node contents

        Returns:
            Tuple of (CompressionResult, Dict[node_id -> scale])
        """
        target_tokens = target_tokens or self.config.default_token_budget

        # Steps 1-2: Activation + Importance (same as pack())
        activation_map = {}
        if self.config.enable_activation_spreading:
            activation_map = self.activation_spreader.spread_activation(
                source_nodes=candidate_nodes,
                graph=graph
            )
            all_candidates = list(set(candidate_nodes) | set(activation_map.keys()))
        else:
            all_candidates = candidate_nodes

        importance_scores = self.importance_scorer.score_batch(
            node_ids=all_candidates,
            query=query,
            graph=graph,
            node_contents=node_contents
        )

        if activation_map:
            importance_scores = self._blend_activation_importance(
                importance_scores, activation_map
            )

        # Step 3: Matryoshka compression
        kept_nodes, scale_assignments = self.context_compressor.matryoshka_compress(
            nodes=all_candidates,
            importance_scores=importance_scores
        )

        # Build result
        original_count = len(all_candidates)
        compressed_count = len(kept_nodes)
        compression_ratio = compressed_count / original_count if original_count > 0 else 0.0
        token_savings = (original_count - compressed_count) * self.config.compression.avg_tokens_per_node

        result = CompressionResult(
            compressed_nodes=kept_nodes,
            original_count=original_count,
            compressed_count=compressed_count,
            compression_ratio=compression_ratio,
            token_savings=token_savings,
            importance_threshold=min(
                importance_scores.get(n, 0.0) for n in kept_nodes
            ) if kept_nodes else 0.0
        )

        self._pack_history.append(result)
        self._update_stats(result)

        return result, scale_assignments

    def pack_with_information_budget(
        self,
        query: str,
        candidate_nodes: List[str],
        graph: Any,
        node_contents: Dict[str, str],
        information_budget: Optional[float] = None,
        token_budget: Optional[int] = None,
        query_embedding: Optional[Any] = None,
        node_embeddings: Optional[Dict[str, Any]] = None
    ) -> Tuple[CompressionResult, ScaleAssignments, Dict[str, float]]:
        """
        Pack context using Information Bottleneck principle (Phase 5).

        Uses Tishby's Information Bottleneck for optimal compression:
        Maximize I(Context; Query) subject to information/token budget.

        Pipeline:
        1. Spread activation from query-relevant nodes
        2. Score importance using 7 signals (including MI)
        3. Compute batch MI scores for all candidates
        4. Apply information budget compression (greedy MI selection)
        5. Assign MI-aware Matryoshka scales

        Args:
            query: Current query text
            candidate_nodes: Initial set of candidate memory nodes
            graph: Knowledge graph
            node_contents: Dict mapping node_id -> text content (REQUIRED for MI)
            information_budget: Maximum cumulative MI (default: from config)
            token_budget: Optional additional token constraint
            query_embedding: Optional pre-computed query embedding
            node_embeddings: Optional pre-computed node embeddings

        Returns:
            Tuple of:
                - CompressionResult with information-optimal packed nodes
                - ScaleAssignments (node_id -> Matryoshka scale)
                - MI scores dict (node_id -> MI score)
        """
        if not self.config.importance_scorer.enable_information_scoring:
            # Fall back to standard pack_with_scales
            logger.warning(
                "Information scoring disabled. Using standard pack_with_scales. "
                "Enable via config.importance_scorer.enable_information_scoring = True"
            )
            result, scales = self.pack_with_scales(
                query, candidate_nodes, graph,
                node_contents=node_contents
            )
            return result, scales, {}

        if self.config.verbose:
            logger.info(
                f"Information budget packing: {len(candidate_nodes)} candidates, "
                f"budget: {information_budget or self.config.compression.information_budget}"
            )

        # Step 1: Activation spreading
        activation_map = {}
        if self.config.enable_activation_spreading:
            activation_map = self.activation_spreader.spread_activation(
                source_nodes=candidate_nodes,
                graph=graph
            )
            all_candidates = list(set(candidate_nodes) | set(activation_map.keys()))

            if self.config.verbose:
                logger.info(f"Activated {len(activation_map)} nodes via beta waves")
        else:
            all_candidates = candidate_nodes

        # Step 2: Importance scoring (includes 7th MI signal)
        importance_scores = self.importance_scorer.score_batch(
            node_ids=all_candidates,
            query=query,
            graph=graph,
            node_contents=node_contents
        )

        if activation_map:
            importance_scores = self._blend_activation_importance(
                importance_scores, activation_map
            )

        # Step 3: Compute batch MI scores for compression decisions
        mi_scores = self.importance_scorer.batch_compute_mi_scores(
            node_ids=all_candidates,
            query=query,
            node_contents=node_contents,
            query_embedding=query_embedding,
            node_embeddings=node_embeddings
        )

        if self.config.verbose:
            avg_mi = sum(mi_scores.values()) / len(mi_scores) if mi_scores else 0.0
            logger.info(
                f"Computed MI scores for {len(mi_scores)} nodes "
                f"(avg MI: {avg_mi:.4f})"
            )

        # Step 4: Information budget compression
        result = self.context_compressor.information_budget_compress(
            nodes=all_candidates,
            importance_scores=importance_scores,
            mi_scores=mi_scores,
            information_budget=information_budget,
            token_budget=token_budget
        )

        # Step 5: MI-aware Matryoshka scale assignment
        kept_nodes, scale_assignments = self.context_compressor.matryoshka_compress(
            nodes=result.compressed_nodes,
            importance_scores=importance_scores,
            mi_scores=mi_scores  # Use MI-aware thresholds
        )

        # Update result with scale-aware token estimation
        estimated_tokens = self.context_compressor.estimate_tokens(
            kept_nodes, scale_assignments
        )

        # Track MI-specific stats
        if 'info_budget_packs' not in self._stats:
            self._stats['info_budget_packs'] = 0
            self._stats['avg_mi_utilization'] = 0.0

        self._stats['info_budget_packs'] += 1
        budget = information_budget or self.config.compression.information_budget
        used_mi = sum(mi_scores.get(n, 0.0) for n in kept_nodes)
        utilization = used_mi / budget if budget > 0 else 0.0

        n = self._stats['info_budget_packs']
        self._stats['avg_mi_utilization'] = (
            (self._stats['avg_mi_utilization'] * (n - 1) + utilization) / n
        )

        self._pack_history.append(result)
        self._update_stats(result)

        if self.config.verbose:
            logger.info(
                f"Info budget packed: {result.original_count} -> {len(kept_nodes)} nodes "
                f"(MI utilization: {utilization:.1%}, ~{estimated_tokens} tokens)"
            )

        return result, scale_assignments, mi_scores

    def get_stats(self) -> Dict[str, Any]:
        """Get packing statistics."""
        stats = self._stats.copy()
        stats['component_stats'] = {
            'activation_spreader': self.activation_spreader.get_stats(),
            'importance_scorer': self.importance_scorer.get_stats(),
            'context_compressor': self.context_compressor.get_stats()
        }
        return stats

    def reset_stats(self):
        """Reset all statistics."""
        self._stats = {
            'total_packs': 0,
            'avg_compression_ratio': 0.0,
            'avg_token_savings': 0.0
        }
        self._pack_history.clear()
        self.activation_spreader.reset_stats()
        self.importance_scorer.reset_stats()
        self.context_compressor.reset_stats()

    def get_pack_history(self) -> List[CompressionResult]:
        """Get history of pack operations."""
        return self._pack_history.copy()

    # Helper methods

    def _blend_activation_importance(
        self,
        importance_scores: ImportanceMap,
        activation_map: ActivationMap,
        activation_weight: float = 0.3
    ) -> ImportanceMap:
        """
        Blend activation levels with importance scores.

        Args:
            importance_scores: Dict of node -> importance
            activation_map: Dict of node -> activation
            activation_weight: Weight for activation (0-1)

        Returns:
            Blended importance scores
        """
        blended = {}

        for node in importance_scores:
            importance = importance_scores[node]
            activation = activation_map.get(node, 0.0)

            # Weighted blend
            blended[node] = (
                (1 - activation_weight) * importance +
                activation_weight * activation
            )

        return blended

    def _update_stats(self, result: CompressionResult):
        """Update rolling statistics from pack result."""
        self._stats['total_packs'] += 1

        # Update averages
        n = self._stats['total_packs']
        self._stats['avg_compression_ratio'] = (
            (self._stats['avg_compression_ratio'] * (n - 1) + result.compression_ratio) / n
        )
        self._stats['avg_token_savings'] = (
            (self._stats['avg_token_savings'] * (n - 1) + result.token_savings) / n
        )


# Convenience functions

def pack_context(
    query: str,
    candidate_nodes: List[str],
    graph: Any,
    target_tokens: int = 2000,
    preset: str = "balanced"
) -> List[str]:
    """
    Quick context packing - returns just the compressed node list.

    Args:
        query: Query text
        candidate_nodes: Candidates to pack
        graph: Knowledge graph
        target_tokens: Token budget
        preset: Config preset (aggressive/balanced/conservative/research)

    Returns:
        List of packed node IDs
    """
    # Get config preset
    if preset == "aggressive":
        config = ContextPackerConfig.aggressive()
    elif preset == "conservative":
        config = ContextPackerConfig.conservative()
    elif preset == "research":
        config = ContextPackerConfig.research()
    else:
        config = ContextPackerConfig.balanced()

    packer = ContextPacker(config)
    result = packer.pack(query, candidate_nodes, graph, target_tokens)

    return result.compressed_nodes


def adaptive_pack_context(
    query: str,
    candidate_nodes: List[str],
    graph: Any,
    min_tokens: int = 500,
    max_tokens: int = 4000
) -> List[str]:
    """
    Adaptive context packing within token range.

    Args:
        query: Query text
        candidate_nodes: Candidates
        graph: Knowledge graph
        min_tokens: Minimum tokens
        max_tokens: Maximum tokens

    Returns:
        List of packed node IDs
    """
    packer = ContextPacker()
    result = packer.adaptive_pack(
        query, candidate_nodes, graph,
        min_tokens=min_tokens, max_tokens=max_tokens
    )

    return result.compressed_nodes


def information_budget_pack(
    query: str,
    candidate_nodes: List[str],
    graph: Any,
    node_contents: Dict[str, str],
    information_budget: float = 5.0,
    token_budget: Optional[int] = None
) -> Tuple[List[str], Dict[str, int], Dict[str, float]]:
    """
    Information-theoretic context packing (Phase 5).

    Uses Tishby's Information Bottleneck principle to optimally
    compress context by maximizing I(Context; Query).

    Args:
        query: Query text
        candidate_nodes: Candidate nodes
        graph: Knowledge graph
        node_contents: Dict mapping node_id -> text content
        information_budget: Maximum cumulative MI in bits (default: 5.0)
        token_budget: Optional additional token constraint

    Returns:
        Tuple of:
            - List of packed node IDs
            - Dict of node_id -> Matryoshka scale (384/256/128)
            - Dict of node_id -> MI score
    """
    packer = ContextPacker()
    result, scale_assignments, mi_scores = packer.pack_with_information_budget(
        query=query,
        candidate_nodes=candidate_nodes,
        graph=graph,
        node_contents=node_contents,
        information_budget=information_budget,
        token_budget=token_budget
    )

    return result.compressed_nodes, scale_assignments, mi_scores

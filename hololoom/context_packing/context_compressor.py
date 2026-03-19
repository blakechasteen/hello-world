"""
Matryoshka-Aware Context Compression
=====================================

Compresses context by:
1. Selecting most important nodes (threshold-based)
2. Assigning different Matryoshka scales based on importance
3. Estimating token savings from scale assignment

Key Innovation: Not all nodes need full 384D embeddings!
- High importance: 384D (full detail)
- Medium importance: 256D (moderate detail)
- Low importance: 128D (minimal detail) or dropped

This enables 40-90% token savings while preserving information density.

Phase 5 (December 2025): Added information-theoretic compression using
Tishby's Information Bottleneck principle:
- information_budget_compress(): Greedy MI-based selection until budget exhausted
- MI-aware Matryoshka scales: High MI → 384D, Low MI → 128D

Author: Claude Code
Date: 2025-11-22
"""

import logging
from collections import defaultdict
from typing import Any

from .config import CompressionConfig
from .protocol import CompressionResult, ImportanceMap, NodeEmbeddings, ScaleAssignments

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    Matryoshka-aware context compression engine.

    Selects most important nodes and assigns optimal embedding scales
    to maximize information density within token budget.
    """

    def __init__(self, config: CompressionConfig | None = None):
        """
        Initialize context compressor.

        Args:
            config: Compression configuration (uses defaults if None)
        """
        self.config = config or CompressionConfig()
        self.config.validate()

        # Stats
        self._stats = {
            'total_compressions': 0,
            'avg_compression_ratio': 0.0,
            'avg_token_savings': 0.0
        }

    def compress(
        self,
        nodes: list[str],
        importance_scores: ImportanceMap,
        target_ratio: float | None = None,
        preserve_top_k: int | None = None
    ) -> CompressionResult:
        """
        Compress context by selecting most important nodes.

        Args:
            nodes: List of node IDs to compress
            importance_scores: Dict mapping node_id -> importance (0-1)
            target_ratio: Override config target_ratio
            preserve_top_k: Override config preserve_top_k

        Returns:
            CompressionResult with compressed nodes and metrics
        """
        target_ratio = target_ratio or self.config.target_ratio
        preserve_top_k = preserve_top_k or self.config.preserve_top_k

        original_count = len(nodes)

        if original_count == 0:
            return CompressionResult(
                compressed_nodes=[],
                original_count=0,
                compressed_count=0,
                compression_ratio=0.0,
                token_savings=0,
                importance_threshold=0.0
            )

        # Sort nodes by importance
        sorted_nodes = sorted(
            nodes,
            key=lambda n: importance_scores.get(n, 0.0),
            reverse=True
        )

        # Calculate target count
        target_count = max(
            preserve_top_k,
            int(original_count * target_ratio)
        )

        # Select top nodes
        compressed_nodes = sorted_nodes[:target_count]

        # Determine threshold (importance of last kept node)
        if compressed_nodes:
            importance_threshold = importance_scores.get(
                compressed_nodes[-1], 0.0
            )
        else:
            importance_threshold = 0.0

        # Calculate compression ratio
        compressed_count = len(compressed_nodes)
        compression_ratio = compressed_count / original_count if original_count > 0 else 0.0

        # Estimate token savings
        dropped_count = original_count - compressed_count
        token_savings = dropped_count * self.config.avg_tokens_per_node

        # Update stats
        self._stats['total_compressions'] += 1
        self._stats['avg_compression_ratio'] = (
            (self._stats['avg_compression_ratio'] * (self._stats['total_compressions'] - 1) +
             compression_ratio) / self._stats['total_compressions']
        )
        self._stats['avg_token_savings'] = (
            (self._stats['avg_token_savings'] * (self._stats['total_compressions'] - 1) +
             token_savings) / self._stats['total_compressions']
        )

        logger.debug(
            f"Compressed: {original_count} -> {compressed_count} nodes "
            f"({compression_ratio:.1%} kept, {token_savings} tokens saved)"
        )

        return CompressionResult(
            compressed_nodes=compressed_nodes,
            original_count=original_count,
            compressed_count=compressed_count,
            compression_ratio=compression_ratio,
            token_savings=token_savings,
            importance_threshold=importance_threshold
        )

    def matryoshka_compress(
        self,
        nodes: list[str],
        importance_scores: ImportanceMap,
        embeddings: NodeEmbeddings | None = None,
        scales: list[int] | None = None,
        mi_scores: dict[str, float] | None = None
    ) -> tuple[list[str], ScaleAssignments]:
        """
        Multi-scale compression using Matryoshka embeddings.

        Assigns different embedding scales based on importance or MI:
        - High importance/MI: 384D (full detail)
        - Medium importance/MI: 256D (moderate detail)
        - Low importance/MI: 128D (minimal detail)
        - Very low: Dropped

        Phase 5 Enhancement: When mi_scores provided, uses MI-aware thresholds:
        - MI > 0.7: 384D (high information shared with query)
        - MI 0.4-0.7: 256D (moderate)
        - MI 0.2-0.4: 128D (low)
        - MI < 0.2: dropped

        Args:
            nodes: List of node IDs
            importance_scores: Dict mapping node_id -> importance
            embeddings: Optional full embeddings (unused, for future use)
            scales: Optional override of Matryoshka scales
            mi_scores: Optional MI scores for information-theoretic assignment

        Returns:
            Tuple of (kept_nodes, node_id -> assigned_scale)
        """
        scales = scales or self.config.matryoshka_scales

        kept_nodes = []
        scale_assignments = {}

        # Determine which scores to use for scale assignment
        use_mi = mi_scores is not None and len(mi_scores) > 0

        # Sort nodes by primary score (MI if available, otherwise importance)
        if use_mi:
            sorted_nodes = sorted(
                nodes,
                key=lambda n: mi_scores.get(n, 0.0),
                reverse=True
            )
        else:
            sorted_nodes = sorted(
                nodes,
                key=lambda n: importance_scores.get(n, 0.0),
                reverse=True
            )

        # Assign scales based on thresholds
        for node in sorted_nodes:
            if use_mi:
                # MI-aware assignment (Phase 5)
                score = mi_scores.get(node, 0.0)
                high_thresh = self.config.mi_high_threshold
                mid_thresh = self.config.mi_mid_threshold
                low_thresh = self.config.mi_low_threshold
            else:
                # Traditional importance-based assignment
                score = importance_scores.get(node, 0.0)
                high_thresh = self.config.high_importance_threshold
                mid_thresh = self.config.mid_importance_threshold
                low_thresh = self.config.low_importance_threshold

            if score >= high_thresh:
                # High score: full scale
                scale = scales[0] if scales else 384
                kept_nodes.append(node)
                scale_assignments[node] = scale

            elif score >= mid_thresh:
                # Medium score: middle scale
                scale = scales[1] if len(scales) > 1 else 256
                kept_nodes.append(node)
                scale_assignments[node] = scale

            elif score >= low_thresh:
                # Low score: smallest scale
                scale = scales[-1] if scales else 128
                kept_nodes.append(node)
                scale_assignments[node] = scale

            else:
                # Very low score: drop
                pass

        # Ensure we keep at least preserve_top_k nodes
        if len(kept_nodes) < self.config.preserve_top_k:
            missing = self.config.preserve_top_k - len(kept_nodes)
            for node in sorted_nodes:
                if node not in kept_nodes and missing > 0:
                    kept_nodes.append(node)
                    scale_assignments[node] = scales[-1] if scales else 128
                    missing -= 1

        mode_str = "MI-aware" if use_mi else "importance-based"
        logger.debug(
            f"Matryoshka compression ({mode_str}): {len(nodes)} -> {len(kept_nodes)} nodes "
            f"(scales: {self._count_by_scale(scale_assignments)})"
        )

        return kept_nodes, scale_assignments

    def adaptive_compress(
        self,
        nodes: list[str],
        importance_scores: ImportanceMap,
        min_tokens: int = 500,
        max_tokens: int = 4000
    ) -> CompressionResult:
        """
        Adaptively compress context within token budget range.

        Finds optimal compression that maximizes information density
        while staying within budget.

        Args:
            nodes: List of node IDs
            importance_scores: Dict mapping node_id -> importance
            min_tokens: Minimum acceptable token count
            max_tokens: Maximum acceptable token count

        Returns:
            CompressionResult with adaptively compressed context
        """
        original_count = len(nodes)

        if original_count == 0:
            return self.compress(nodes, importance_scores)

        # Binary search for optimal ratio
        low_ratio = 0.1
        high_ratio = 1.0
        best_result = None

        while high_ratio - low_ratio > 0.05:  # 5% precision
            mid_ratio = (low_ratio + high_ratio) / 2.0

            # Try compression at this ratio
            result = self.compress(nodes, importance_scores, target_ratio=mid_ratio)

            # Estimate tokens
            estimated_tokens = result.compressed_count * self.config.avg_tokens_per_node

            if estimated_tokens < min_tokens:
                # Need more nodes
                low_ratio = mid_ratio
            elif estimated_tokens > max_tokens:
                # Need fewer nodes
                high_ratio = mid_ratio
            else:
                # Within budget - save this result
                best_result = result
                # Try to compress more (greedy for token savings)
                high_ratio = mid_ratio

        # Use best found result, or compress to max if nothing found
        if best_result is None:
            target_count = max_tokens // self.config.avg_tokens_per_node
            target_ratio = target_count / original_count if original_count > 0 else 1.0
            best_result = self.compress(nodes, importance_scores, target_ratio=target_ratio)

        logger.debug(
            f"Adaptive compression: {original_count} -> {best_result.compressed_count} nodes "
            f"(~{best_result.compressed_count * self.config.avg_tokens_per_node} tokens)"
        )

        return best_result

    def information_budget_compress(
        self,
        nodes: list[str],
        importance_scores: ImportanceMap,
        mi_scores: dict[str, float],
        information_budget: float | None = None,
        token_budget: int | None = None
    ) -> CompressionResult:
        """
        Information Bottleneck-based compression using Tishby's principle.

        Greedily selects nodes by mutual information until:
        1. Information budget is exhausted (cumulative MI >= budget)
        2. Token budget is exhausted
        3. Marginal MI gain falls below threshold (diminishing returns)

        This implements the Information Bottleneck objective:
        Maximize I(Context; Query) subject to constraints.

        Args:
            nodes: List of node IDs to compress
            importance_scores: Dict mapping node_id -> importance (0-1)
            mi_scores: Dict mapping node_id -> mutual information score
            information_budget: Maximum cumulative MI (default: from config)
            token_budget: Maximum token count (optional additional constraint)

        Returns:
            CompressionResult with information-optimal compressed nodes
        """
        # Use config defaults if not specified
        if information_budget is None:
            information_budget = self.config.information_budget

        if not self.config.enable_information_budget:
            # Fall back to standard compression
            return self.compress(nodes, importance_scores)

        original_count = len(nodes)

        if original_count == 0:
            return CompressionResult(
                compressed_nodes=[],
                original_count=0,
                compressed_count=0,
                compression_ratio=0.0,
                token_savings=0,
                importance_threshold=0.0
            )

        # Sort nodes by MI (descending) - greedy selection
        sorted_nodes = sorted(
            nodes,
            key=lambda n: mi_scores.get(n, 0.0),
            reverse=True
        )

        # Greedy selection until budget exhausted
        selected_nodes = []
        cumulative_mi = 0.0
        cumulative_tokens = 0
        last_mi = float('inf')  # For diminishing returns detection

        for node in sorted_nodes:
            node_mi = mi_scores.get(node, 0.0)
            node_tokens = self.config.avg_tokens_per_node

            # Check stopping conditions

            # 1. Information budget exhausted
            if cumulative_mi + node_mi > information_budget:
                # Check if we should include this node (might still have value)
                remaining_budget = information_budget - cumulative_mi
                if remaining_budget > 0 and node_mi > 0:
                    # Proportional inclusion (node partially contributes)
                    # Include if at least 50% of node's MI fits in budget
                    if remaining_budget >= node_mi * 0.5:
                        selected_nodes.append(node)
                        cumulative_mi += node_mi
                        cumulative_tokens += node_tokens
                break

            # 2. Token budget exhausted
            if token_budget is not None:
                if cumulative_tokens + node_tokens > token_budget:
                    break

            # 3. Diminishing returns - marginal MI gain too small
            marginal_gain = node_mi
            if marginal_gain < self.config.mi_diminishing_returns_threshold:
                logger.debug(
                    f"Stopping: marginal MI gain {marginal_gain:.4f} < "
                    f"threshold {self.config.mi_diminishing_returns_threshold}"
                )
                break

            # Include this node
            selected_nodes.append(node)
            cumulative_mi += node_mi
            cumulative_tokens += node_tokens
            last_mi = node_mi

        # Ensure we keep at least preserve_top_k nodes
        if len(selected_nodes) < self.config.preserve_top_k:
            # Add more nodes from sorted list
            for node in sorted_nodes:
                if node not in selected_nodes:
                    selected_nodes.append(node)
                    if len(selected_nodes) >= self.config.preserve_top_k:
                        break

        # Calculate metrics
        compressed_count = len(selected_nodes)
        compression_ratio = compressed_count / original_count if original_count > 0 else 0.0
        dropped_count = original_count - compressed_count
        token_savings = dropped_count * self.config.avg_tokens_per_node

        # Determine threshold (MI of last kept node)
        if selected_nodes:
            importance_threshold = importance_scores.get(selected_nodes[-1], 0.0)
        else:
            importance_threshold = 0.0

        # Update stats
        self._stats['total_compressions'] += 1
        self._update_running_average('avg_compression_ratio', compression_ratio)
        self._update_running_average('avg_token_savings', token_savings)

        # Track MI-specific stats
        if 'avg_cumulative_mi' not in self._stats:
            self._stats['avg_cumulative_mi'] = 0.0
            self._stats['mi_budget_compressions'] = 0

        self._stats['mi_budget_compressions'] += 1
        self._stats['avg_cumulative_mi'] = (
            (self._stats['avg_cumulative_mi'] * (self._stats['mi_budget_compressions'] - 1) +
             cumulative_mi) / self._stats['mi_budget_compressions']
        )

        logger.debug(
            f"Information budget compression: {original_count} -> {compressed_count} nodes "
            f"(cumulative MI: {cumulative_mi:.3f} / {information_budget}, "
            f"{token_savings} tokens saved)"
        )

        return CompressionResult(
            compressed_nodes=selected_nodes,
            original_count=original_count,
            compressed_count=compressed_count,
            compression_ratio=compression_ratio,
            token_savings=token_savings,
            importance_threshold=importance_threshold
        )

    def _update_running_average(self, stat_name: str, value: float):
        """Helper to update running average statistics."""
        count = self._stats['total_compressions']
        if count > 0:
            self._stats[stat_name] = (
                (self._stats[stat_name] * (count - 1) + value) / count
            )

    def estimate_tokens(
        self,
        nodes: list[str],
        scale_assignments: ScaleAssignments | None = None
    ) -> int:
        """
        Estimate token count for node list.

        If scale_assignments provided, accounts for different scales:
        - 384D: avg_tokens_per_node
        - 256D: avg_tokens_per_node * 0.66
        - 128D: avg_tokens_per_node * 0.33

        Args:
            nodes: List of nodes
            scale_assignments: Optional dict of node_id -> scale

        Returns:
            Estimated token count
        """
        if scale_assignments is None:
            # Simple estimate
            return len(nodes) * self.config.avg_tokens_per_node

        # Account for different scales
        total_tokens = 0
        base_tokens = self.config.avg_tokens_per_node

        for node in nodes:
            scale = scale_assignments.get(node, self.config.matryoshka_scales[0])

            # Token multiplier based on scale
            if scale >= 384:
                multiplier = 1.0
            elif scale >= 256:
                multiplier = 0.66
            elif scale >= 128:
                multiplier = 0.33
            else:
                multiplier = 0.1  # Very small

            total_tokens += int(base_tokens * multiplier)

        return total_tokens

    def get_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        return self._stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            'total_compressions': 0,
            'avg_compression_ratio': 0.0,
            'avg_token_savings': 0.0
        }

    # Helper methods

    def _count_by_scale(self, scale_assignments: ScaleAssignments) -> dict[int, int]:
        """Count nodes by assigned scale."""
        counts = defaultdict(int)
        for scale in scale_assignments.values():
            counts[scale] += 1
        return dict(counts)


# Convenience functions

def quick_compress(
    nodes: list[str],
    importance_scores: ImportanceMap,
    target_ratio: float = 0.5
) -> list[str]:
    """
    Quick compression - just returns compressed node list.

    Args:
        nodes: Nodes to compress
        importance_scores: Importance scores
        target_ratio: Target compression ratio

    Returns:
        List of compressed nodes
    """
    compressor = ContextCompressor()
    result = compressor.compress(nodes, importance_scores, target_ratio=target_ratio)
    return result.compressed_nodes


def compress_to_budget(
    nodes: list[str],
    importance_scores: ImportanceMap,
    max_tokens: int = 2000,
    avg_tokens_per_node: int = 50
) -> list[str]:
    """
    Compress to fit within token budget.

    Args:
        nodes: Nodes to compress
        importance_scores: Importance scores
        max_tokens: Maximum token budget
        avg_tokens_per_node: Estimated tokens per node

    Returns:
        List of compressed nodes
    """
    config = CompressionConfig(avg_tokens_per_node=avg_tokens_per_node)
    compressor = ContextCompressor(config)

    result = compressor.adaptive_compress(
        nodes, importance_scores,
        min_tokens=max_tokens // 2,
        max_tokens=max_tokens
    )

    return result.compressed_nodes

"""
Matryoshka Gating - Multi-Scale Importance Filtering
====================================================
Progressive filtering using coarse-to-fine embedding scales.

Philosophy:
Like Russian nesting dolls, we filter candidates at each scale:
- 96d (coarse): Fast filtering, keep top 20-30%
- 192d (medium): Balanced filtering, keep top 10-15%
- 384d (fine): Precise ranking, keep top K

This is the SAME concept as recursive crawling importance gates,
but applied to retrieval instead of web scraping!

Benefits:
- Computational efficiency (avoid computing fine embeddings for all)
- Progressive refinement (coarse → fine)
- Adaptive quality (stop early for easy queries)
- Importance-based gating (threshold increases with depth)
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Matryoshka Gate Configuration
# ============================================================================

class GateStrategy(Enum):
    """Gating strategies for multi-scale filtering."""
    FIXED_THRESHOLD = "fixed_threshold"  # Fixed threshold at each scale
    FIXED_TOPK = "fixed_topk"  # Fixed top-K at each scale
    ADAPTIVE = "adaptive"  # Adaptive based on score distribution
    PROGRESSIVE = "progressive"  # Increasing threshold with depth


@dataclass
class GateConfig:
    """Configuration for matryoshka gating."""
    scales: List[int] = None  # Embedding dimensions [96, 192, 384]
    thresholds: List[float] = None  # Importance thresholds per scale
    topk_ratios: List[float] = None  # Keep ratio per scale [0.3, 0.5, 1.0]
    strategy: GateStrategy = GateStrategy.PROGRESSIVE
    min_candidates: int = 5  # Minimum candidates to keep at each stage

    def __post_init__(self):
        if self.scales is None:
            self.scales = [96, 192, 384]

        if self.thresholds is None:
            # Progressive thresholds (like recursive crawler)
            self.thresholds = [0.6, 0.75, 0.85]

        if self.topk_ratios is None:
            # Keep 30% → 50% → 100%
            self.topk_ratios = [0.3, 0.5, 1.0]


# ============================================================================
# Matryoshka Gate
# ============================================================================

@dataclass
class GateResult:
    """Result from a single gating stage."""
    scale: int
    candidates_in: int
    candidates_out: int
    threshold_used: float
    scores: np.ndarray
    kept_indices: List[int]


class MatryoshkaGate:
    """
    Multi-scale importance gating for efficient retrieval.

    Filters candidates progressively through embedding scales:
    1. Compute coarse embeddings (96d) for ALL candidates
    2. Filter to top-K based on threshold
    3. Compute medium embeddings (192d) for survivors
    4. Filter again
    5. Compute fine embeddings (384d) for final ranking

    This is MUCH more efficient than computing 384d for everything!
    """

    def __init__(
        self,
        embedder,
        config: Optional[GateConfig] = None
    ):
        """
        Initialize matryoshka gate.

        Args:
            embedder: MatryoshkaEmbeddings instance
            config: Gate configuration
        """
        self.embedder = embedder
        self.config = config or GateConfig()

        # Statistics
        self.total_gates = 0
        self.total_filtered = 0
        self.scale_stats = {scale: {"passed": 0, "filtered": 0} for scale in self.config.scales}

        logger.info(f"MatryoshkaGate initialized: {self.config.scales}, strategy={self.config.strategy.value}")

    def gate(
        self,
        query: str,
        candidates: List[str],
        final_k: int = 10
    ) -> Tuple[List[int], List[GateResult]]:
        """
        Gate candidates through multi-scale filtering.

        Args:
            query: Query text
            candidates: Candidate texts to filter
            final_k: Final number of results to return

        Returns:
            Tuple of (final_indices, gate_results)
        """
        self.total_gates += 1

        if len(candidates) == 0:
            return [], []

        # Track which candidates survive at each stage
        active_indices = list(range(len(candidates)))
        gate_results = []

        # Encode query at all scales
        query_embeds = self._encode_at_scales(query)

        # Progressive filtering through scales
        for scale_idx, scale in enumerate(self.config.scales):
            if len(active_indices) == 0:
                break

            logger.debug(f"Gate stage {scale_idx+1}/{len(self.config.scales)}: "
                        f"scale={scale}d, candidates={len(active_indices)}")

            # Encode active candidates at this scale
            active_candidates = [candidates[i] for i in active_indices]
            candidate_embeds = self._encode_at_scale(active_candidates, scale)

            # Compute similarity scores
            scores = self._compute_similarity(query_embeds[scale], candidate_embeds)

            # Apply gating threshold
            threshold = self._get_threshold(scale_idx, scores)

            # Determine which candidates pass
            if scale_idx < len(self.config.scales) - 1:
                # Intermediate stage: filter
                kept_mask = self._apply_gate(scores, threshold, scale_idx)
            else:
                # Final stage: take top-K
                kept_mask = self._apply_final_gate(scores, final_k)

            # Update active indices
            kept_local = np.where(kept_mask)[0]
            kept_global = [active_indices[i] for i in kept_local]

            # Record gate result
            gate_result = GateResult(
                scale=scale,
                candidates_in=len(active_indices),
                candidates_out=len(kept_global),
                threshold_used=threshold,
                scores=scores,
                kept_indices=kept_global
            )
            gate_results.append(gate_result)

            # Update statistics
            filtered_count = len(active_indices) - len(kept_global)
            self.scale_stats[scale]["passed"] += len(kept_global)
            self.scale_stats[scale]["filtered"] += filtered_count
            self.total_filtered += filtered_count

            # Update active set
            active_indices = kept_global

            logger.debug(f"  → Kept {len(kept_global)}/{gate_result.candidates_in} "
                        f"(threshold={threshold:.3f})")

        return active_indices, gate_results

    def _encode_at_scales(self, text: str) -> Dict[int, np.ndarray]:
        """Encode text at all scales."""
        embeds = {}
        for scale in self.config.scales:
            embeds[scale] = self._encode_at_scale([text], scale)[0]
        return embeds

    def _encode_at_scale(self, texts: List[str], scale: int) -> np.ndarray:
        """
        Encode texts at specific scale.

        Args:
            texts: List of texts
            scale: Embedding dimension

        Returns:
            Embeddings array (n_texts, scale)
        """
        # Get full embeddings
        full_embeds = self.embedder.encode(texts)

        # Project to target scale
        if scale == self.embedder.base_dim:
            return full_embeds
        elif scale in self.embedder.proj:
            # Use projection matrix
            return full_embeds @ self.embedder.proj[scale]
        else:
            # Truncate
            return full_embeds[:, :scale]

    def _compute_similarity(
        self,
        query_embed: np.ndarray,
        candidate_embeds: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity scores.

        Args:
            query_embed: Query embedding (scale,)
            candidate_embeds: Candidate embeddings (n_candidates, scale)

        Returns:
            Similarity scores (n_candidates,)
        """
        # Normalize
        query_norm = query_embed / (np.linalg.norm(query_embed) + 1e-8)
        cand_norm = candidate_embeds / (np.linalg.norm(candidate_embeds, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        scores = cand_norm @ query_norm

        return scores

    def _get_threshold(self, scale_idx: int, scores: np.ndarray) -> float:
        """
        Get gating threshold for current scale.

        Args:
            scale_idx: Index of current scale
            scores: Similarity scores

        Returns:
            Threshold value
        """
        if self.config.strategy == GateStrategy.FIXED_THRESHOLD:
            return self.config.thresholds[scale_idx]

        elif self.config.strategy == GateStrategy.PROGRESSIVE:
            # Increasing threshold with depth (like crawler)
            return self.config.thresholds[scale_idx]

        elif self.config.strategy == GateStrategy.ADAPTIVE:
            # Adaptive based on score distribution
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            return mean_score + 0.5 * std_score  # Keep above-average

        else:
            return self.config.thresholds[scale_idx]

    def _apply_gate(
        self,
        scores: np.ndarray,
        threshold: float,
        scale_idx: int
    ) -> np.ndarray:
        """
        Apply gating logic to filter candidates.

        Args:
            scores: Similarity scores
            threshold: Threshold value
            scale_idx: Current scale index

        Returns:
            Boolean mask of kept candidates
        """
        # Threshold-based filtering
        threshold_mask = scores >= threshold

        # Also keep top-K ratio
        topk_ratio = self.config.topk_ratios[scale_idx]
        n_keep = max(int(len(scores) * topk_ratio), self.config.min_candidates)
        n_keep = min(n_keep, len(scores))  # Don't exceed available

        # Get top-K indices
        topk_indices = np.argsort(scores)[-n_keep:]
        topk_mask = np.zeros(len(scores), dtype=bool)
        topk_mask[topk_indices] = True

        # Combine: pass if EITHER threshold OR top-K
        combined_mask = threshold_mask | topk_mask

        return combined_mask

    def _apply_final_gate(self, scores: np.ndarray, final_k: int) -> np.ndarray:
        """Apply final gate (just top-K)."""
        n_keep = min(final_k, len(scores))
        topk_indices = np.argsort(scores)[-n_keep:]
        mask = np.zeros(len(scores), dtype=bool)
        mask[topk_indices] = True
        return mask

    def get_statistics(self) -> Dict:
        """Get gating statistics."""
        return {
            "total_gates": self.total_gates,
            "total_filtered": self.total_filtered,
            "scale_stats": self.scale_stats,
            "avg_filtered_per_gate": self.total_filtered / max(self.total_gates, 1)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    # Add repository root to path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo_root)

    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

    print("="*80)
    print("MATRYOSHKA GATING DEMO")
    print("="*80)

    # Create embedder
    print("\nInitializing embedder...")
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    # Create gate
    config = GateConfig(
        scales=[96, 192, 384],
        thresholds=[0.6, 0.75, 0.85],  # Progressive (like crawler)
        topk_ratios=[0.3, 0.5, 1.0],    # Keep 30% → 50% → all
        strategy=GateStrategy.PROGRESSIVE
    )
    gate = MatryoshkaGate(embedder, config)

    # Test data
    query = "What is machine learning?"

    candidates = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Python is a popular programming language",
        "Natural language processing analyzes human language",
        "The weather is nice today",
        "Reinforcement learning trains agents through rewards",
        "Cooking pasta requires boiling water",
        "Neural networks mimic biological neurons",
        "Basketball is played with a ball and hoop",
        "Supervised learning uses labeled training data",
        "The ocean contains many fish species",
        "Computer vision processes images and video",
        "Yesterday I went to the store",
        "Transfer learning reuses pre-trained models",
        "Music has different genres and styles",
    ]

    print(f"\nQuery: '{query}'")
    print(f"Candidates: {len(candidates)}")
    print()

    # Run gating
    final_indices, gate_results = gate.gate(query, candidates, final_k=5)

    # Show results
    print("\nGating Results:")
    print("-" * 80)

    for i, result in enumerate(gate_results):
        print(f"\nStage {i+1}: {result.scale}d")
        print(f"  Input: {result.candidates_in} candidates")
        print(f"  Threshold: {result.threshold_used:.3f}")
        print(f"  Output: {result.candidates_out} candidates")
        print(f"  Filtered: {result.candidates_in - result.candidates_out}")

        if result.candidates_out <= 10:
            print(f"  Survivors:")
            for idx in result.kept_indices:
                score = result.scores[result.kept_indices.index(idx)] if idx in result.kept_indices else 0
                print(f"    [{idx}] {score:.3f}: {candidates[idx][:60]}...")

    print("\n" + "="*80)
    print("FINAL RESULTS (Top 5)")
    print("="*80)
    for rank, idx in enumerate(final_indices, 1):
        print(f"{rank}. [{idx}] {candidates[idx]}")

    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    stats = gate.get_statistics()
    print(f"Total gates run: {stats['total_gates']}")
    print(f"Total filtered: {stats['total_filtered']}")
    print(f"Avg filtered per gate: {stats['avg_filtered_per_gate']:.1f}")
    print("\nPer-scale stats:")
    for scale, s in stats['scale_stats'].items():
        print(f"  {scale}d: passed={s['passed']}, filtered={s['filtered']}")

    print("\nMatryoshka gating operational!")

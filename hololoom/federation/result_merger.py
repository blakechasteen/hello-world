#!/usr/bin/env python3
"""
RAG Result Merger with MI-Based Deduplication
==============================================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 18: Agentic RAG

Merges RAG results from multiple federation nodes using mutual information
scoring for intelligent deduplication and ranking.

Author: Claude Code (Phase 4 - December 2025)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hololoom.federation.types import GuildTrustLevel

logger = logging.getLogger(__name__)


# ============================================================================
#  Constants
# ============================================================================

# Trust level weights for confidence aggregation
TRUST_WEIGHTS: dict[GuildTrustLevel, float] = {
    GuildTrustLevel.STARTER: 0.3,
    GuildTrustLevel.ESTABLISHED: 0.6,
    GuildTrustLevel.VETERAN: 1.0,
}

# Default similarity threshold for deduplication
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# Minimum MI score to keep a source
MIN_MI_THRESHOLD = 0.1


# ============================================================================
#  Data Classes
# ============================================================================

@dataclass(frozen=True)
class SourceWithProvenance:
    """A source text with provenance tracking."""

    content: str
    node_id: str
    confidence: float
    trust_level: GuildTrustLevel = GuildTrustLevel.STARTER
    mi_score: float = 0.5

    def trust_weight(self) -> float:
        """Get trust weight for this source."""
        return TRUST_WEIGHTS.get(self.trust_level, 0.3)

    def effective_score(self) -> float:
        """Calculate effective score: MI × trust × confidence."""
        return self.mi_score * self.trust_weight() * self.confidence


@dataclass
class NodeRAGResult:
    """RAG result from a single federation node."""

    node_id: str
    response: str
    sources: list[str]
    confidence: float
    trust_level: GuildTrustLevel = GuildTrustLevel.STARTER
    metadata: dict[str, Any] = field(default_factory=dict)

    def source_count(self) -> int:
        """Number of sources."""
        return len(self.sources)


@dataclass
class MergedRAGResult:
    """Merged RAG result from multiple federation nodes."""

    sources: list[SourceWithProvenance]
    confidence: float
    original_count: int
    merged_count: int
    contributing_nodes: list[str]
    mi_scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def unique_sources(self) -> list[str]:
        """Get unique source contents."""
        return [s.content for s in self.sources]

    @property
    def top_sources(self) -> list[str]:
        """Get top sources by effective score (for RAGResult compatibility)."""
        sorted_sources = sorted(
            self.sources,
            key=lambda s: s.effective_score(),
            reverse=True
        )
        return [s.content for s in sorted_sources]

    @property
    def avg_mi(self) -> float:
        """Average MI score across all sources."""
        if not self.sources:
            return 0.0
        return sum(s.mi_score for s in self.sources) / len(self.sources)

    @property
    def redundancy_removed(self) -> float:
        """Fraction of sources removed due to redundancy."""
        if self.original_count == 0:
            return 0.0
        return 1.0 - (self.merged_count / self.original_count)


# ============================================================================
#  RAG Result Merger
# ============================================================================

class RAGResultMerger:
    """
    Merges RAG results from multiple federation nodes.

    Uses mutual information scoring to:
    1. Deduplicate semantically similar sources
    2. Rank sources by information value
    3. Weight by node trust level

    Example:
        >>> from hololoom.context_packing import ImportanceScorer
        >>> scorer = ImportanceScorer()
        >>> merger = RAGResultMerger(importance_scorer=scorer)
        >>>
        >>> results = [node_a_result, node_b_result, node_c_result]
        >>> merged = merger.merge(results, query="What is Thompson Sampling?")
        >>> print(f"Merged {merged.original_count} → {merged.merged_count} sources")
    """

    def __init__(
        self,
        importance_scorer: Any | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_mi_threshold: float = MIN_MI_THRESHOLD,
    ):
        """
        Initialize result merger.

        Args:
            importance_scorer: ImportanceScorer instance for MI computation
            similarity_threshold: Threshold for deduplication (0.0-1.0)
            min_mi_threshold: Minimum MI score to keep a source
        """
        self._scorer = importance_scorer
        self._similarity_threshold = similarity_threshold
        self._min_mi_threshold = min_mi_threshold

        # Lazy-load scorer if not provided
        self._scorer_loaded = importance_scorer is not None

    def _ensure_scorer(self) -> None:
        """Ensure importance scorer is available."""
        if self._scorer is None and not self._scorer_loaded:
            try:
                from hololoom.context_packing import ImportanceScorer
                self._scorer = ImportanceScorer()
                self._scorer_loaded = True
            except ImportError:
                logger.warning("ImportanceScorer not available, using fallback MI scoring")
                self._scorer_loaded = True

    def merge(
        self,
        results: list[NodeRAGResult],
        query: str,
        max_sources: int = 20,
    ) -> MergedRAGResult:
        """
        Merge RAG results from multiple nodes.

        Steps:
        1. Extract all sources from all nodes
        2. Compute MI scores via batch_compute_mi_scores()
        3. Deduplicate by similarity threshold (0.85)
        4. Rank by MI × trust weight
        5. Return top-k with combined confidence

        Args:
            results: List of NodeRAGResult from federation nodes
            query: The query text (used for MI scoring)
            max_sources: Maximum sources to return

        Returns:
            MergedRAGResult with deduplicated, ranked sources
        """
        if not results:
            return MergedRAGResult(
                sources=[],
                confidence=0.0,
                original_count=0,
                merged_count=0,
                contributing_nodes=[],
                mi_scores={},
            )

        # Step 1: Extract all sources with provenance
        all_sources = self._combine_sources(results)
        original_count = len(all_sources)

        if not all_sources:
            return MergedRAGResult(
                sources=[],
                confidence=0.0,
                original_count=0,
                merged_count=0,
                contributing_nodes=[r.node_id for r in results],
                mi_scores={},
            )

        # Step 2: Compute MI scores
        self._ensure_scorer()
        mi_scores = self._compute_mi_scores(all_sources, query)

        # Update sources with MI scores
        sources_with_mi = []
        for source in all_sources:
            source_hash = self._content_hash(source.content)
            mi_score = mi_scores.get(source_hash, 0.5)

            # Skip low-MI sources
            if mi_score < self._min_mi_threshold:
                continue

            sources_with_mi.append(SourceWithProvenance(
                content=source.content,
                node_id=source.node_id,
                confidence=source.confidence,
                trust_level=source.trust_level,
                mi_score=mi_score,
            ))

        # Step 3: Deduplicate
        deduplicated = self._deduplicate(sources_with_mi)

        # Step 4: Rank by effective score
        ranked = sorted(
            deduplicated,
            key=lambda s: s.effective_score(),
            reverse=True
        )

        # Step 5: Take top-k
        top_sources = ranked[:max_sources]

        # Calculate aggregated confidence
        aggregated_confidence = self._aggregate_confidence(results, top_sources)

        return MergedRAGResult(
            sources=top_sources,
            confidence=aggregated_confidence,
            original_count=original_count,
            merged_count=len(top_sources),
            contributing_nodes=list(set(r.node_id for r in results)),
            mi_scores=mi_scores,
            metadata={
                "query": query,
                "similarity_threshold": self._similarity_threshold,
                "sources_filtered_by_mi": original_count - len(sources_with_mi),
                "sources_deduplicated": len(sources_with_mi) - len(deduplicated),
            }
        )

    def _combine_sources(
        self,
        results: list[NodeRAGResult],
    ) -> list[SourceWithProvenance]:
        """
        Combine sources from all nodes with provenance tracking.

        Args:
            results: List of node results

        Returns:
            List of sources with provenance
        """
        all_sources = []

        for result in results:
            for source in result.sources:
                all_sources.append(SourceWithProvenance(
                    content=source,
                    node_id=result.node_id,
                    confidence=result.confidence,
                    trust_level=result.trust_level,
                    mi_score=0.5,  # Will be updated
                ))

        return all_sources

    def _compute_mi_scores(
        self,
        sources: list[SourceWithProvenance],
        query: str,
    ) -> dict[str, float]:
        """
        Compute MI scores for all sources.

        Args:
            sources: List of sources
            query: Query text

        Returns:
            Dict mapping content_hash → MI score
        """
        mi_scores = {}

        if self._scorer is None:
            # Fallback: word overlap-based MI
            for source in sources:
                content_hash = self._content_hash(source.content)
                mi_scores[content_hash] = self._compute_word_overlap_mi(
                    source.content, query
                )
            return mi_scores

        # Use ImportanceScorer.batch_compute_mi_scores
        node_ids = []
        node_contents = {}

        for source in sources:
            content_hash = self._content_hash(source.content)
            node_ids.append(content_hash)
            node_contents[content_hash] = source.content

        try:
            mi_scores = self._scorer.batch_compute_mi_scores(
                node_ids=node_ids,
                query=query,
                node_contents=node_contents,
            )
        except Exception as e:
            logger.warning(f"batch_compute_mi_scores failed: {e}, using fallback")
            for source in sources:
                content_hash = self._content_hash(source.content)
                mi_scores[content_hash] = self._compute_word_overlap_mi(
                    source.content, query
                )

        return mi_scores

    def _compute_word_overlap_mi(self, content: str, query: str) -> float:
        """
        Fallback MI computation using word overlap.

        Approximates I(Content; Query) using Jaccard similarity.
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words or not content_words:
            return 0.0

        overlap = len(query_words & content_words)
        union = len(query_words | content_words)

        if union == 0:
            return 0.0

        return overlap / union

    def _deduplicate(
        self,
        sources: list[SourceWithProvenance],
    ) -> list[SourceWithProvenance]:
        """
        Deduplicate sources by semantic similarity.

        Uses word overlap as similarity measure.
        Keeps the source with higher effective score when duplicates found.

        Args:
            sources: List of sources with MI scores

        Returns:
            Deduplicated list
        """
        if not sources:
            return []

        # Sort by effective score (best first)
        sorted_sources = sorted(
            sources,
            key=lambda s: s.effective_score(),
            reverse=True
        )

        kept: list[SourceWithProvenance] = []

        for source in sorted_sources:
            is_duplicate = False

            for kept_source in kept:
                similarity = self._compute_similarity(
                    source.content,
                    kept_source.content
                )

                if similarity >= self._similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(source)

        return kept

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.

        Uses Jaccard similarity on word sets.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0

        return intersection / union

    def _aggregate_confidence(
        self,
        results: list[NodeRAGResult],
        selected_sources: list[SourceWithProvenance],
    ) -> float:
        """
        Aggregate confidence across nodes.

        Formula: Σ(conf × trust × sources) / Σ(trust × sources)

        Args:
            results: All node results
            selected_sources: Sources selected for final result

        Returns:
            Aggregated confidence score (0.0-1.0)
        """
        if not results:
            return 0.0

        # Weight by trust level and source count
        numerator = 0.0
        denominator = 0.0

        for result in results:
            trust_weight = TRUST_WEIGHTS.get(result.trust_level, 0.3)
            source_count = result.source_count()

            numerator += result.confidence * trust_weight * source_count
            denominator += trust_weight * source_count

        if denominator == 0:
            return 0.0

        return min(1.0, max(0.0, numerator / denominator))

    def _content_hash(self, content: str) -> str:
        """
        Create a hash for content lookup.

        Uses first 100 chars + length for fast hashing.
        """
        return f"{content[:100]}:{len(content)}"


# ============================================================================
#  Factory Function
# ============================================================================

def create_result_merger(
    importance_scorer: Any | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_mi_threshold: float = MIN_MI_THRESHOLD,
) -> RAGResultMerger:
    """
    Create a RAGResultMerger instance.

    Args:
        importance_scorer: Optional ImportanceScorer (auto-created if None)
        similarity_threshold: Deduplication threshold
        min_mi_threshold: Minimum MI score to keep

    Returns:
        Configured RAGResultMerger
    """
    return RAGResultMerger(
        importance_scorer=importance_scorer,
        similarity_threshold=similarity_threshold,
        min_mi_threshold=min_mi_threshold,
    )


# ============================================================================
#  Exports
# ============================================================================

__all__ = [
    # Data classes
    "SourceWithProvenance",
    "NodeRAGResult",
    "MergedRAGResult",
    # Main class
    "RAGResultMerger",
    # Factory
    "create_result_merger",
    # Constants
    "TRUST_WEIGHTS",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "MIN_MI_THRESHOLD",
]

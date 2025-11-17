"""
DreamEngine Pattern Synthesis
==============================

Discovers recurring patterns in high-confidence interactions and creates
summary memories.

**Philosophy:** Similar to how humans consolidate episodic memories into
semantic knowledge during sleep.

Author: HoloLoom Team
Date: 2025-11-17
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from HoloLoom.synthesis.types import (
    SyntheticMemory,
    SynthesisType,
    PatternCluster
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PatternSynthesisConfig:
    """Configuration for pattern synthesis."""
    enabled: bool = True
    confidence_threshold: float = 0.85  # Minimum confidence
    min_cluster_size: int = 3  # Min queries to form pattern
    similarity_threshold: float = 0.1  # Word overlap similarity (Jaccard, very permissive for MVP)
    window_size: int = 100  # Look at last N queries
    trigger_interval: int = 100  # Run every N queries


# ============================================================================
# Pattern Synthesis Engine
# ============================================================================

class PatternSynthesizer:
    """
    Discovers patterns in high-confidence queries and creates summaries.

    Example:
        User asks 3+ related questions about Thompson Sampling with high
        confidence → System creates a summary memory combining the insights.
    """

    def __init__(self, config: Optional[PatternSynthesisConfig] = None):
        """
        Initialize pattern synthesizer.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or PatternSynthesisConfig()
        self.query_history: List[Dict[str, Any]] = []

    def add_query(self, query: Dict[str, Any]):
        """
        Add query to history for pattern detection.

        Args:
            query: Query dict with keys:
                - text: Query text
                - confidence: Query confidence (0.0-1.0)
                - timestamp: When query was made
                - entities: Extracted entities (optional)
                - motifs: Extracted motifs (optional)
                - embedding: Query embedding (optional)
        """
        self.query_history.append(query)

        # Keep only recent queries
        if len(self.query_history) > self.config.window_size:
            self.query_history = self.query_history[-self.config.window_size:]

    def synthesize_patterns(self) -> List[SyntheticMemory]:
        """
        Synthesize patterns from recent high-confidence queries.

        Returns:
            List of synthetic summary memories
        """
        if not self.config.enabled:
            return []

        # 1. Filter high-confidence queries
        high_conf_queries = [
            q for q in self.query_history
            if q.get('confidence', 0.0) >= self.config.confidence_threshold
        ]

        if len(high_conf_queries) < self.config.min_cluster_size:
            return []  # Not enough data yet

        # 2. Cluster by semantic similarity
        clusters = self._cluster_queries(high_conf_queries)

        # 3. Generate summary for each cluster
        synthetic_memories = []
        for cluster in clusters:
            if len(cluster) >= self.config.min_cluster_size:
                summary = self._generate_pattern_summary(cluster)
                if summary:
                    synthetic_memories.append(summary)

        return synthetic_memories

    def _cluster_queries(
        self,
        queries: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster queries by semantic similarity.

        Uses simple word overlap for MVP (can upgrade to embeddings later).

        Args:
            queries: List of query dicts

        Returns:
            List of query clusters
        """
        clusters = []

        for query in queries:
            # Try to add to existing cluster
            added = False
            for cluster in clusters:
                if self._is_similar_to_cluster(query, cluster):
                    cluster.append(query)
                    added = True
                    break

            # Create new cluster if no match
            if not added:
                clusters.append([query])

        return clusters

    def _is_similar_to_cluster(
        self,
        query: Dict[str, Any],
        cluster: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if query is similar enough to join cluster.

        Uses word overlap similarity for MVP.

        Args:
            query: Query to check
            cluster: Existing cluster

        Returns:
            True if similar enough, False otherwise
        """
        query_words = set(query.get('text', '').lower().split())

        # Calculate average overlap with cluster members
        overlaps = []
        for member in cluster:
            member_words = set(member.get('text', '').lower().split())
            if len(query_words | member_words) > 0:
                overlap = len(query_words & member_words) / len(query_words | member_words)
                overlaps.append(overlap)

        if not overlaps:
            return False

        avg_overlap = sum(overlaps) / len(overlaps)
        return avg_overlap >= self.config.similarity_threshold

    def _generate_pattern_summary(
        self,
        cluster: List[Dict[str, Any]]
    ) -> Optional[SyntheticMemory]:
        """
        Generate summary memory from query cluster.

        Args:
            cluster: List of related queries

        Returns:
            Synthetic memory or None if generation fails
        """
        # Extract common entities and motifs
        all_entities = []
        all_motifs = []

        for query in cluster:
            all_entities.extend(query.get('entities', []))
            all_motifs.extend(query.get('motifs', []))

        # Find most common
        common_entities = self._find_common_terms(all_entities, min_count=2)
        common_motifs = self._find_common_terms(all_motifs, min_count=2)

        if not common_entities and not common_motifs:
            return None  # No clear pattern

        # Generate summary text
        summary_text = self._create_summary_text(
            cluster=cluster,
            common_entities=common_entities,
            common_motifs=common_motifs
        )

        # Calculate average confidence
        confidences = [q.get('confidence', 0.0) for q in cluster]
        avg_confidence = sum(confidences) / len(confidences)

        # Create synthetic memory
        return SyntheticMemory(
            content=summary_text,
            provenance="pattern_synthesis",
            source_memories=[q.get('id', f"query_{i}") for i, q in enumerate(cluster)],
            confidence=avg_confidence,
            synthesis_type=SynthesisType.PATTERN,
            metadata={
                'cluster_size': len(cluster),
                'common_entities': common_entities,
                'common_motifs': common_motifs,
                'time_span_hours': self._get_time_span(cluster)
            }
        )

    def _find_common_terms(
        self,
        terms: List[str],
        min_count: int = 2
    ) -> List[str]:
        """
        Find terms that appear multiple times.

        Args:
            terms: List of terms
            min_count: Minimum occurrences

        Returns:
            List of common terms
        """
        from collections import Counter
        counts = Counter(terms)
        return [term for term, count in counts.items() if count >= min_count]

    def _create_summary_text(
        self,
        cluster: List[Dict[str, Any]],
        common_entities: List[str],
        common_motifs: List[str]
    ) -> str:
        """
        Create human-readable summary from cluster.

        Args:
            cluster: Query cluster
            common_entities: Common entities across queries
            common_motifs: Common motifs across queries

        Returns:
            Summary text
        """
        # Extract key themes
        themes = common_entities + common_motifs

        # Create summary
        if len(themes) == 1:
            summary = f"Pattern detected: Recurring interest in {themes[0]}. "
        else:
            summary = f"Pattern detected: Recurring interest in {', '.join(themes[:3])}. "

        # Add query count
        summary += f"Explored through {len(cluster)} related questions. "

        # Add time context
        time_span = self._get_time_span(cluster)
        if time_span:
            if time_span < 1:
                summary += f"Concentrated within {int(time_span * 60)} minutes."
            elif time_span < 24:
                summary += f"Explored over {int(time_span)} hours."
            else:
                summary += f"Explored over {int(time_span / 24)} days."

        return summary

    def _get_time_span(self, cluster: List[Dict[str, Any]]) -> Optional[float]:
        """
        Get time span of cluster in hours.

        Args:
            cluster: Query cluster

        Returns:
            Time span in hours or None if timestamps unavailable
        """
        timestamps = [q.get('timestamp') for q in cluster if 'timestamp' in q]

        if len(timestamps) < 2:
            return None

        # Convert to datetime if needed
        timestamps = [
            ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
            for ts in timestamps
        ]

        earliest = min(timestamps)
        latest = max(timestamps)
        span = (latest - earliest).total_seconds() / 3600  # hours

        return span

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pattern synthesis statistics.

        Returns:
            Dict with statistics
        """
        high_conf_count = sum(
            1 for q in self.query_history
            if q.get('confidence', 0.0) >= self.config.confidence_threshold
        )

        return {
            'total_queries': len(self.query_history),
            'high_confidence_queries': high_conf_count,
            'high_conf_rate': high_conf_count / len(self.query_history) if self.query_history else 0,
            'window_size': self.config.window_size,
            'confidence_threshold': self.config.confidence_threshold
        }


# ============================================================================
# Standalone Functions
# ============================================================================

def synthesize_patterns_from_queries(
    queries: List[Dict[str, Any]],
    config: Optional[PatternSynthesisConfig] = None
) -> List[SyntheticMemory]:
    """
    Synthesize patterns from a list of queries (standalone function).

    Args:
        queries: List of query dicts
        config: Optional configuration

    Returns:
        List of synthetic memories

    Example:
        >>> queries = [
        ...     {'text': 'What is Thompson Sampling?', 'confidence': 0.92},
        ...     {'text': 'How does Thompson Sampling work?', 'confidence': 0.89},
        ...     {'text': 'Thompson vs epsilon-greedy', 'confidence': 0.91}
        ... ]
        >>> patterns = synthesize_patterns_from_queries(queries)
        >>> print(len(patterns))
        1
        >>> print(patterns[0].content)
        'Pattern detected: Recurring interest in Thompson Sampling...'
    """
    synthesizer = PatternSynthesizer(config)

    for query in queries:
        synthesizer.add_query(query)

    return synthesizer.synthesize_patterns()

"""
HoloLoom Curiosity Engine
==========================

Proactive exploration suggestion system that analyzes knowledge gaps, contradictions,
and access patterns to suggest learning opportunities.

**Philosophy:** Great learning is proactive, not reactive. The Curiosity Engine acts as
a curious companion, always looking for interesting paths to explore, contradictions to
resolve, and connections to make.

Author: HoloLoom Team
Date: 2025-11-17
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import random

from HoloLoom.synthesis.gap_identification import GapIdentifier, GapIdentificationConfig
from HoloLoom.synthesis.contradiction_detection import ContradictionDetector, ContradictionConfig
from HoloLoom.synthesis.types import KnowledgeGap, Contradiction, GapType, ContradictionType
from HoloLoom.memory.graph import KG


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ExplorationSuggestion:
    """
    A proactive suggestion to explore a concept or resolve an issue.

    Designed to be user-friendly with natural language explanations.
    """
    type: str  # 'gap', 'contradiction', 'related_concept', 'trending', 'deep_dive'
    concept: str  # The main concept to explore
    reason: str  # Natural language explanation of why this is interesting
    importance: float  # 0-1 score (higher = more important)
    suggested_query: str  # Ready-to-use query string
    expected_benefit: str  # What the user will learn/gain
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context
    created_at: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"""
💡 {self.concept} ({self.importance:.0%} important)

{self.reason}

Suggested query: "{self.suggested_query}"

Expected benefit: {self.expected_benefit}
        """.strip()


@dataclass
class CuriosityConfig:
    """Configuration for the Curiosity Engine."""
    enabled: bool = True
    max_suggestions: int = 5  # Max suggestions per request
    gap_importance_threshold: float = 0.5  # Only suggest high-importance gaps
    contradiction_score_threshold: float = 0.6  # Only suggest clear contradictions
    trending_window_hours: int = 24  # Look back 24 hours for trending topics
    related_concepts_depth: int = 2  # How far to traverse graph for related concepts
    access_pattern_min_frequency: int = 3  # Min accesses to consider trending
    enable_serendipity: bool = True  # Enable random exploration suggestions
    serendipity_probability: float = 0.2  # 20% chance of serendipitous suggestion


# ============================================================================
# Curiosity Engine
# ============================================================================

class CuriosityEngine:
    """
    Proactive exploration suggestion engine for HoloLoom.

    Analyzes knowledge graph structure, contradiction patterns, and user behavior
    to suggest interesting learning opportunities.

    Key Features:
    - Gap-based suggestions (mentioned but unexplored concepts)
    - Contradiction resolution suggestions (conflicting information)
    - Access pattern analysis (trending topics, related concepts)
    - Serendipitous exploration (random interesting paths)
    - Deep dive suggestions (comprehensive topic exploration)

    Usage:
        engine = CuriosityEngine(kg=knowledge_graph)

        # Get proactive suggestions
        suggestions = await engine.suggest_exploration(limit=5)

        for suggestion in suggestions:
            print(suggestion)
            if user_accepts:
                result = await orchestrator.weave(Query(text=suggestion.suggested_query))
    """

    def __init__(
        self,
        kg: Optional[KG] = None,
        config: Optional[CuriosityConfig] = None,
        gap_config: Optional[GapIdentificationConfig] = None,
        contradiction_config: Optional[ContradictionConfig] = None
    ):
        """
        Initialize Curiosity Engine.

        Args:
            kg: Knowledge graph (optional, can be set later)
            config: Curiosity engine configuration
            gap_config: Gap identification configuration
            contradiction_config: Contradiction detection configuration
        """
        self.kg = kg
        self.config = config or CuriosityConfig()

        # Initialize sub-engines
        self.gap_identifier = GapIdentifier(gap_config)
        self.contradiction_detector = ContradictionDetector(contradiction_config)

        # Track query history for pattern analysis
        self.query_history: List[Dict[str, Any]] = []
        self.entity_access_counts: Dict[str, int] = Counter()
        self.entity_last_access: Dict[str, datetime] = {}

        # Cache for expensive computations
        self._suggestion_cache: List[ExplorationSuggestion] = []
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds: int = 300  # 5 minutes

    def set_knowledge_graph(self, kg: KG):
        """Set or update the knowledge graph."""
        self.kg = kg
        self._invalidate_cache()

    def track_query(
        self,
        query_text: str,
        entities: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Track a user query for pattern analysis.

        Args:
            query_text: The query string
            entities: Extracted entities from query
            timestamp: When the query occurred (defaults to now)
        """
        timestamp = timestamp or datetime.now()

        # Record query
        self.query_history.append({
            'text': query_text,
            'entities': entities or [],
            'timestamp': timestamp
        })

        # Update entity access tracking
        if entities:
            for entity in entities:
                self.entity_access_counts[entity] += 1
                self.entity_last_access[entity] = timestamp

        # Prune old queries (keep last 1000)
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

        # Invalidate cache on new query
        self._invalidate_cache()

    async def suggest_exploration(self, limit: int = 5) -> List[ExplorationSuggestion]:
        """
        Get comprehensive exploration suggestions.

        Combines gap-based, contradiction-based, access pattern, and serendipitous
        suggestions into a prioritized list.

        Args:
            limit: Maximum number of suggestions to return

        Returns:
            List of exploration suggestions, sorted by importance
        """
        if not self.config.enabled:
            return []

        # Check cache first
        if self._is_cache_valid():
            return self._suggestion_cache[:limit]

        suggestions: List[ExplorationSuggestion] = []

        # 1. Gap-based suggestions (highest priority)
        if self.kg:
            gap_suggestions = await self.suggest_gap_filling(
                self._kg_to_dict(self.kg)
            )
            suggestions.extend(gap_suggestions)

        # 2. Contradiction resolution suggestions
        contradiction_suggestions = await self.suggest_contradiction_resolution()
        suggestions.extend(contradiction_suggestions)

        # 3. Access pattern suggestions (trending, related)
        if self.query_history:
            pattern_suggestions = await self._suggest_from_access_patterns()
            suggestions.extend(pattern_suggestions)

        # 4. Deep dive suggestions (comprehensive exploration)
        deep_dive_suggestions = await self._suggest_deep_dives()
        suggestions.extend(deep_dive_suggestions)

        # 5. Serendipitous suggestions (random exploration)
        if self.config.enable_serendipity and random.random() < self.config.serendipity_probability:
            serendipity_suggestions = await self._suggest_serendipitous()
            suggestions.extend(serendipity_suggestions)

        # Sort by importance and limit
        suggestions.sort(key=lambda s: s.importance, reverse=True)
        suggestions = suggestions[:self.config.max_suggestions]

        # Update cache
        self._suggestion_cache = suggestions
        self._cache_timestamp = datetime.now()

        return suggestions[:limit]

    async def suggest_gap_filling(
        self,
        knowledge_graph: Dict[str, Any]
    ) -> List[ExplorationSuggestion]:
        """
        Suggest explorations to fill knowledge gaps.

        Args:
            knowledge_graph: Dict representation of KG with:
                - nodes: List of concept nodes
                - edges: List of (source, target, type) edges
                - node_counts: Dict of node -> access count

        Returns:
            List of gap-filling suggestions
        """
        suggestions = []

        # Identify gaps
        gaps = self.gap_identifier.identify_gaps(knowledge_graph)

        for gap in gaps:
            # Filter by importance threshold
            if gap.importance < self.config.gap_importance_threshold:
                continue

            # Generate suggestion
            suggestion = self._gap_to_suggestion(gap)
            suggestions.append(suggestion)

        return suggestions

    async def suggest_contradiction_resolution(self) -> List[ExplorationSuggestion]:
        """
        Suggest explorations to resolve contradictions.

        Returns:
            List of contradiction resolution suggestions
        """
        suggestions = []

        # Detect contradictions
        contradictions = self.contradiction_detector.detect_contradictions()

        for contradiction in contradictions:
            # Filter by score threshold
            if contradiction.score < self.config.contradiction_score_threshold:
                continue

            # Generate suggestion
            suggestion = self._contradiction_to_suggestion(contradiction)
            suggestions.append(suggestion)

        return suggestions

    async def suggest_related_concepts(
        self,
        current_concept: str,
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Suggest related concepts to explore based on current concept.

        Args:
            current_concept: The concept user is currently exploring
            max_suggestions: Maximum number of related concepts to return

        Returns:
            List of related concept names
        """
        if not self.kg:
            return []

        related = []

        # Get direct neighbors in KG
        neighbors = self.kg.get_neighbors(current_concept)

        # Filter out already heavily-accessed concepts
        unexplored = [
            n for n in neighbors
            if self.entity_access_counts.get(n, 0) < 5  # Less than 5 accesses
        ]

        # Sort by edge weight (stronger connections first)
        # For now, just use neighbors as-is
        related.extend(unexplored[:max_suggestions])

        # If we need more suggestions, do 2-hop traversal
        if len(related) < max_suggestions and self.config.related_concepts_depth >= 2:
            for neighbor in neighbors[:3]:  # Check top 3 neighbors
                second_hop = self.kg.get_neighbors(neighbor)
                for concept in second_hop:
                    if concept != current_concept and concept not in related:
                        if self.entity_access_counts.get(concept, 0) < 5:
                            related.append(concept)
                            if len(related) >= max_suggestions:
                                break
                if len(related) >= max_suggestions:
                    break

        return related[:max_suggestions]

    async def _suggest_from_access_patterns(self) -> List[ExplorationSuggestion]:
        """
        Generate suggestions based on user access patterns.

        Identifies:
        - Trending topics (frequently accessed recently)
        - Related concepts (unexplored neighbors of trending topics)

        Returns:
            List of access pattern-based suggestions
        """
        suggestions = []

        # Find trending topics (frequently accessed in recent window)
        cutoff = datetime.now() - timedelta(hours=self.config.trending_window_hours)
        recent_entities = [
            entity for entity, last_access in self.entity_last_access.items()
            if last_access >= cutoff
        ]

        # Count recent accesses
        recent_counts = Counter(recent_entities)
        trending = [
            entity for entity, count in recent_counts.most_common(5)
            if count >= self.config.access_pattern_min_frequency
        ]

        # Generate related concept suggestions for trending topics
        for trend in trending:
            related = await self.suggest_related_concepts(trend, max_suggestions=3)

            for concept in related:
                suggestion = ExplorationSuggestion(
                    type='related_concept',
                    concept=concept,
                    reason=f"You've been exploring {trend} recently. {concept} is closely related and might interest you.",
                    importance=0.6,
                    suggested_query=f"Tell me about {concept} and how it relates to {trend}",
                    expected_benefit=f"Understanding {concept} will deepen your knowledge of {trend}",
                    metadata={
                        'parent_concept': trend,
                        'access_count': recent_counts[trend]
                    }
                )
                suggestions.append(suggestion)

        return suggestions

    async def _suggest_deep_dives(self) -> List[ExplorationSuggestion]:
        """
        Suggest comprehensive exploration of partially-known topics.

        Identifies concepts that have been accessed a few times but not deeply explored.

        Returns:
            List of deep dive suggestions
        """
        suggestions = []

        # Find concepts with moderate access (2-4 times) - partial knowledge
        moderate_access = [
            (entity, count) for entity, count in self.entity_access_counts.items()
            if 2 <= count <= 4
        ]

        # Sort by recency
        moderate_access.sort(
            key=lambda x: self.entity_last_access.get(x[0], datetime.min),
            reverse=True
        )

        for entity, count in moderate_access[:3]:
            suggestion = ExplorationSuggestion(
                type='deep_dive',
                concept=entity,
                reason=f"You've touched on {entity} {count} times but haven't explored it deeply. Time for a comprehensive dive?",
                importance=0.5,
                suggested_query=f"Give me a comprehensive overview of {entity} with examples and use cases",
                expected_benefit=f"Transform partial knowledge into deep understanding of {entity}",
                metadata={
                    'access_count': count,
                    'last_access': self.entity_last_access.get(entity)
                }
            )
            suggestions.append(suggestion)

        return suggestions

    async def _suggest_serendipitous(self) -> List[ExplorationSuggestion]:
        """
        Suggest random but potentially interesting explorations.

        Enables serendipitous discovery by suggesting concepts the user hasn't
        encountered yet but are in the knowledge graph.

        Returns:
            List of serendipitous suggestions
        """
        suggestions = []

        if not self.kg:
            return suggestions

        # Get all nodes in KG
        all_nodes = set(self.kg.get_all_nodes())

        # Filter to unexplored nodes (never or rarely accessed)
        unexplored = [
            node for node in all_nodes
            if self.entity_access_counts.get(node, 0) <= 1
        ]

        if not unexplored:
            return suggestions

        # Pick a random unexplored concept
        random_concept = random.choice(unexplored)

        suggestion = ExplorationSuggestion(
            type='trending',  # Using 'trending' as a catch-all for serendipity
            concept=random_concept,
            reason=f"How about exploring something completely new? {random_concept} is in your knowledge graph but you haven't discovered it yet.",
            importance=0.4,
            suggested_query=f"What is {random_concept}?",
            expected_benefit="Serendipitous discovery - you never know what fascinating connections you'll find!",
            metadata={'serendipity': True}
        )
        suggestions.append(suggestion)

        return suggestions

    def _gap_to_suggestion(self, gap: KnowledgeGap) -> ExplorationSuggestion:
        """
        Convert a KnowledgeGap to an ExplorationSuggestion.

        Args:
            gap: Knowledge gap identified by GapIdentifier

        Returns:
            User-friendly exploration suggestion
        """
        # Customize reason based on gap type
        if gap.gap_type == GapType.BRIDGE_GAP:
            reason = f"You've mentioned {gap.bridge_concept} but haven't explored it deeply. It connects to {', '.join(gap.missing_concepts[:2])} which you also don't know yet."
            benefit = f"Understanding {gap.bridge_concept} will fill a key gap in your knowledge network"

        elif gap.gap_type == GapType.MISSING_PREREQUISITE:
            reason = f"You know about {gap.bridge_concept}, but you're missing the foundational concepts: {', '.join(gap.missing_concepts[:2])}."
            benefit = f"Learning these prerequisites will deepen your understanding of {gap.bridge_concept}"

        elif gap.gap_type == GapType.INCOMPLETE_CATEGORY:
            known = [c for c in self.gap_identifier.related_concepts.get(gap.bridge_concept.lower(), [])
                     if c not in gap.missing_concepts]
            reason = f"You know some aspects of {gap.bridge_concept} ({', '.join(known[:2])}), but you're missing {', '.join(gap.missing_concepts[:2])}."
            benefit = f"Complete your knowledge of {gap.bridge_concept} for a comprehensive understanding"

        else:  # STALE_KNOWLEDGE or unknown
            reason = f"There's a gap in your knowledge around {gap.bridge_concept}."
            benefit = "Fill this gap to strengthen your understanding"

        return ExplorationSuggestion(
            type='gap',
            concept=gap.bridge_concept,
            reason=reason,
            importance=gap.importance,
            suggested_query=gap.suggested_queries[0] if gap.suggested_queries else f"What is {gap.bridge_concept}?",
            expected_benefit=benefit,
            metadata={
                'gap_type': gap.gap_type.value,
                'missing_concepts': gap.missing_concepts
            }
        )

    def _contradiction_to_suggestion(self, contradiction: Contradiction) -> ExplorationSuggestion:
        """
        Convert a Contradiction to an ExplorationSuggestion.

        Args:
            contradiction: Detected contradiction

        Returns:
            User-friendly resolution suggestion
        """
        # Extract key concept from contradiction
        concept = self._extract_key_concept(contradiction)

        # Customize reason based on contradiction type
        if contradiction.type == ContradictionType.PREFERENCE_REVERSAL:
            reason = f"You seem to have changed your mind about {concept}. Earlier you said '{contradiction.memory1_content[:40]}...' but later '{contradiction.memory2_content[:40]}...'"
            benefit = "Clarify your current perspective and understand what changed"

        elif contradiction.type == ContradictionType.FACT_UPDATE:
            reason = f"Your knowledge about {concept} appears to have been updated. The facts seem to have changed over {contradiction.temporal_gap_days} days."
            benefit = "Verify the current facts and understand what changed"

        elif contradiction.type == ContradictionType.BELIEF_CHANGE:
            reason = f"Your beliefs about {concept} have evolved. {contradiction.temporal_gap_days} days ago you thought differently."
            benefit = "Reflect on this evolution and understand why your thinking changed"

        elif contradiction.type == ContradictionType.LOGICAL_CONFLICT:
            reason = f"There's a logical inconsistency about {concept} in your knowledge. These statements conflict: '{contradiction.memory1_content[:40]}...' vs '{contradiction.memory2_content[:40]}...'"
            benefit = "Resolve this contradiction to maintain consistent knowledge"

        else:  # CONTEXT_DEPENDENT or unknown
            reason = f"There's conflicting information about {concept}. Context might matter."
            benefit = "Clarify when each perspective applies"

        return ExplorationSuggestion(
            type='contradiction',
            concept=concept,
            reason=reason,
            importance=contradiction.score,
            suggested_query=f"Clarify: {contradiction.memory1_content[:50]}... vs {contradiction.memory2_content[:50]}...",
            expected_benefit=benefit,
            metadata={
                'contradiction_type': contradiction.type.value,
                'memory1_id': contradiction.memory1_id,
                'memory2_id': contradiction.memory2_id,
                'temporal_gap_days': contradiction.temporal_gap_days
            }
        )

    def _extract_key_concept(self, contradiction: Contradiction) -> str:
        """
        Extract the main concept from a contradiction.

        Simple heuristic: find the most common non-stopword across both memories.

        Args:
            contradiction: The contradiction object

        Returns:
            Key concept string
        """
        # Combine both memories
        text = f"{contradiction.memory1_content} {contradiction.memory2_content}".lower()

        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of',
                    'and', 'in', 'on', 'at', 'for', 'with', 'but', 'or', 'as', 'by',
                    'i', 'you', 'it', 'that', 'this', 'they', 'them', 'their'}

        words = text.split()
        words = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in stopwords]

        # Find most common word
        if words:
            counts = Counter(words)
            return counts.most_common(1)[0][0]

        return "this topic"

    def _kg_to_dict(self, kg: KG) -> Dict[str, Any]:
        """
        Convert KG to dict format expected by GapIdentifier.

        Args:
            kg: Knowledge graph

        Returns:
            Dict with nodes, edges, node_counts
        """
        # Get all nodes
        nodes = kg.get_all_nodes()

        # Get all edges (simplified - just get edge triples)
        edges = []
        for node in nodes:
            neighbors = kg.get_neighbors(node)
            for neighbor in neighbors:
                # Get edge data
                edge_data = kg.G.get_edge_data(node, neighbor)
                if edge_data:
                    # NetworkX MultiDiGraph can have multiple edges
                    for key, data in edge_data.items():
                        edge_type = data.get('type', 'RELATED')
                        edges.append((node, neighbor, edge_type))

        # Node counts from access tracking
        node_counts = dict(self.entity_access_counts)

        return {
            'nodes': list(nodes),
            'edges': edges,
            'node_counts': node_counts
        }

    def _is_cache_valid(self) -> bool:
        """Check if suggestion cache is still valid."""
        if not self._cache_timestamp:
            return False

        age_seconds = (datetime.now() - self._cache_timestamp).total_seconds()
        return age_seconds < self._cache_ttl_seconds

    def _invalidate_cache(self):
        """Invalidate the suggestion cache."""
        self._cache_timestamp = None
        self._suggestion_cache = []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get curiosity engine statistics.

        Returns:
            Dict with statistics
        """
        return {
            'total_queries_tracked': len(self.query_history),
            'unique_entities_accessed': len(self.entity_access_counts),
            'top_trending_entities': [
                (entity, count) for entity, count in self.entity_access_counts.most_common(5)
            ],
            'cache_valid': self._is_cache_valid(),
            'suggestions_cached': len(self._suggestion_cache),
            'config': {
                'max_suggestions': self.config.max_suggestions,
                'gap_threshold': self.config.gap_importance_threshold,
                'contradiction_threshold': self.config.contradiction_score_threshold,
                'serendipity_enabled': self.config.enable_serendipity
            }
        }


# ============================================================================
# Standalone Functions
# ============================================================================

async def suggest_explorations(
    kg: Optional[KG] = None,
    query_history: Optional[List[Dict[str, Any]]] = None,
    config: Optional[CuriosityConfig] = None,
    limit: int = 5
) -> List[ExplorationSuggestion]:
    """
    Standalone function to get exploration suggestions.

    Args:
        kg: Knowledge graph (optional)
        query_history: List of past queries with entities (optional)
        config: Curiosity configuration (optional)
        limit: Max suggestions to return

    Returns:
        List of exploration suggestions

    Example:
        >>> kg = KG()
        >>> # ... populate KG ...
        >>> suggestions = await suggest_explorations(kg=kg, limit=3)
        >>> for s in suggestions:
        ...     print(s.suggested_query)
    """
    engine = CuriosityEngine(kg=kg, config=config)

    # Track query history if provided
    if query_history:
        for query in query_history:
            engine.track_query(
                query_text=query.get('text', ''),
                entities=query.get('entities', []),
                timestamp=query.get('timestamp')
            )

    return await engine.suggest_exploration(limit=limit)

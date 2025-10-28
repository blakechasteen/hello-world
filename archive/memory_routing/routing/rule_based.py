"""
Rule-Based Router - Deterministic Backend Selection
==================================================

Simple, fast, interpretable routing based on query patterns.
Good baseline for comparison with learned strategies.
"""

import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

from .protocol import (
    RoutingStrategy,
    RoutingDecision,
    RoutingOutcome,
    BackendType,
    QueryType
)


class RuleBasedRouter:
    """
    Rule-based routing strategy using query pattern matching.

    Rules:
    - Relationship queries (who/when/where) → Neo4j
    - Similarity queries (find/similar/like) → Qdrant
    - Personal queries (my/I/preference) → Mem0
    - Temporal queries (recent/today/latest) → InMemory
    - Default: Qdrant (semantic similarity)
    """

    def __init__(self):
        self.outcomes: List[RoutingOutcome] = []
        self.stats = {
            'total_queries': 0,
            'backend_counts': defaultdict(int),
            'query_type_counts': defaultdict(int),
            'avg_latency': defaultdict(list),
            'avg_relevance': defaultdict(list),
        }

        # Rule patterns
        self.patterns = {
            QueryType.RELATIONSHIP: [
                r'\b(who|whom|whose)\b',
                r'\b(when|where)\b',
                r'\b(related|connected|between)\b',
                r'\b(relationship|connection)\b',
            ],
            QueryType.SIMILARITY: [
                r'\b(similar|like|resembles)\b',
                r'\b(find|search|look for)\b',
                r'\b(compare|comparison)\b',
            ],
            QueryType.PERSONAL: [
                r'\b(my|mine|i|me)\b',
                r'\b(personal|preference|favorite)\b',
                r'\b(user|profile)\b',
            ],
            QueryType.TEMPORAL: [
                r'\b(recent|latest|today|yesterday)\b',
                r'\b(this week|last week|past)\b',
                r'\b(now|current|currently)\b',
            ],
        }

    def select_backend(
        self,
        query: str,
        available_backends: List[BackendType],
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """Select backend using rule-based pattern matching."""

        # Classify query type
        query_type = self._classify_query(query)

        # Map query type to backend preference
        backend_preference = {
            QueryType.RELATIONSHIP: BackendType.NEO4J,
            QueryType.SIMILARITY: BackendType.QDRANT,
            QueryType.PERSONAL: BackendType.MEM0,
            QueryType.TEMPORAL: BackendType.INMEMORY,
            QueryType.UNKNOWN: BackendType.QDRANT,  # Default to semantic
        }

        preferred_backend = backend_preference[query_type]

        # Fallback if preferred not available
        if preferred_backend not in available_backends:
            # Try alternatives in order
            alternatives_order = [
                BackendType.QDRANT,  # Best general-purpose
                BackendType.NEO4J,
                BackendType.MEM0,
                BackendType.INMEMORY,
            ]

            for alt in alternatives_order:
                if alt in available_backends:
                    preferred_backend = alt
                    break

        # Build alternatives list
        alternatives = [b for b in available_backends if b != preferred_backend]

        # Record stats
        self.stats['total_queries'] += 1
        self.stats['backend_counts'][preferred_backend.value] += 1
        self.stats['query_type_counts'][query_type.value] += 1

        return RoutingDecision(
            backend_type=preferred_backend,
            confidence=0.8 if query_type != QueryType.UNKNOWN else 0.5,
            query_type=query_type,
            reasoning=self._explain_choice(query_type, preferred_backend),
            alternatives=alternatives,
            metadata={
                'strategy': 'rule_based',
                'patterns_matched': self._get_matched_patterns(query, query_type)
            }
        )

    def _classify_query(self, query: str) -> QueryType:
        """Classify query using pattern matching."""
        query_lower = query.lower()

        # Count pattern matches for each type
        scores = defaultdict(int)

        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[query_type] += 1

        # Return type with most matches
        if scores:
            return max(scores, key=scores.get)

        return QueryType.UNKNOWN

    def _get_matched_patterns(self, query: str, query_type: QueryType) -> List[str]:
        """Get which patterns matched for this query type."""
        query_lower = query.lower()
        matched = []

        if query_type in self.patterns:
            for pattern in self.patterns[query_type]:
                if re.search(pattern, query_lower):
                    matched.append(pattern)

        return matched

    def _explain_choice(self, query_type: QueryType, backend: BackendType) -> str:
        """Generate human-readable explanation."""
        explanations = {
            (QueryType.RELATIONSHIP, BackendType.NEO4J):
                "Relationship query detected → Neo4j graph traversal optimal",
            (QueryType.SIMILARITY, BackendType.QDRANT):
                "Similarity query detected → Qdrant vector search optimal",
            (QueryType.PERSONAL, BackendType.MEM0):
                "Personal query detected → Mem0 user context optimal",
            (QueryType.TEMPORAL, BackendType.INMEMORY):
                "Temporal query detected → InMemory cache optimal for recent data",
            (QueryType.UNKNOWN, BackendType.QDRANT):
                "General query → Qdrant semantic search as default",
        }

        return explanations.get(
            (query_type, backend),
            f"Query type {query_type.value} → {backend.value} backend"
        )

    def record_outcome(self, outcome: RoutingOutcome):
        """Record outcome for statistics."""
        self.outcomes.append(outcome)

        # Update running stats
        backend = outcome.decision.backend_type.value
        self.stats['avg_latency'][backend].append(outcome.latency_ms)
        self.stats['avg_relevance'][backend].append(outcome.avg_relevance)

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = dict(self.stats)

        # Compute averages
        stats['avg_latency_by_backend'] = {
            backend: sum(latencies) / len(latencies) if latencies else 0
            for backend, latencies in self.stats['avg_latency'].items()
        }

        stats['avg_relevance_by_backend'] = {
            backend: sum(relevances) / len(relevances) if relevances else 0
            for backend, relevances in self.stats['avg_relevance'].items()
        }

        # Overall accuracy (queries with good relevance)
        all_relevances = [o.avg_relevance for o in self.outcomes]
        stats['accuracy'] = (
            sum(1 for r in all_relevances if r > 0.7) / len(all_relevances)
            if all_relevances else 0
        )

        return stats
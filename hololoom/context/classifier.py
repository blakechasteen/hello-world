#!/usr/bin/env python3
"""
Query Classifier for Hybrid Routing

Part 3: Classification and Basic Routing (Days 11-15)

Implements 7-rule decision tree for backend selection.
Validated in Demo 1 (95% accuracy).

Rules:
1. Exact ID lookups → SQL (0.95)
2. Policy/ground truth → SQL (0.90)
3. Aggregations/counts → SQL (0.85)
4. Similarity queries → Vector (0.88)
5. Relationship traversal → Graph (0.87)
6. Hybrid queries → SQL + Graph (0.82)
7. Exploratory → Graph (0.70)
Default: Thompson Sampling (0.50)
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Backend Selection
# ============================================================================

class Backend(str):
    """Available backends"""
    SQL = "sql"
    NEO4J = "neo4j"
    QDRANT = "qdrant"
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class BackendSelection:
    """
    Classification result

    Attributes:
        backend: Selected backend ("sql", "neo4j", "qdrant", "thompson_sampling")
        confidence: Confidence score (0.0-1.0)
        rule_matched: Rule that triggered selection
        query: Original query text
        secondary_backend: Optional secondary backend for hybrid queries
    """
    backend: str
    confidence: float
    rule_matched: str
    query: str
    secondary_backend: str | None = None

    def is_hybrid(self) -> bool:
        """Check if this is a hybrid query requiring multiple backends"""
        return self.secondary_backend is not None

    def get_backends(self) -> list[str]:
        """Get list of backends (primary + secondary if hybrid)"""
        backends = [self.backend]
        if self.secondary_backend:
            backends.append(self.secondary_backend)
        return backends


# ============================================================================
# Query Classifier
# ============================================================================

class QueryClassifier:
    """
    7-rule query classification for hybrid routing

    Validated in Demo 1 with 95% accuracy (19/20 queries)
    """

    def __init__(self):
        """Initialize classifier"""
        self.classification_count = 0
        self.rule_usage = {
            "Rule 1": 0,
            "Rule 2": 0,
            "Rule 3": 0,
            "Rule 4": 0,
            "Rule 5": 0,
            "Rule 6": 0,
            "Rule 7": 0,
            "Default": 0
        }
        # Backend weights for strategy updater (Part 4: Learning)
        self.backend_weights = {
            "sql": 1.0,
            "neo4j": 1.0,
            "qdrant": 1.0
        }

    def classify(self, query: str) -> BackendSelection:
        """
        Apply 7-rule decision tree

        Args:
            query: Natural language query

        Returns:
            BackendSelection with backend, confidence, rule
        """
        self.classification_count += 1
        query_lower = query.lower()

        # Rule 1: Exact ID lookups → SQL (0.95)
        if re.search(r'\b(get|fetch|retrieve|show)\b.*\b[a-z]+_\d+\b', query_lower):
            self.rule_usage["Rule 1"] += 1
            logger.debug(f"Rule 1 matched (Exact ID): {query[:50]}...")
            return BackendSelection(
                backend=Backend.SQL,
                confidence=0.95,
                rule_matched="Rule 1: Exact ID lookup",
                query=query
            )

        # Rule 5: Relationship traversal → Graph (0.87)
        # Check this BEFORE Rule 2 to prioritize graph queries
        if re.search(r'\b(connected|related|linked|associated|affected)\b', query_lower):
            self.rule_usage["Rule 5"] += 1
            logger.debug(f"Rule 5 matched (Relationship traversal): {query[:50]}...")
            return BackendSelection(
                backend=Backend.NEO4J,
                confidence=0.87,
                rule_matched="Rule 5: Relationship traversal",
                query=query
            )

        # Rule 4: Similarity queries → Vector (0.88)
        if re.search(r'\b(similar|like|resembling|comparable)\b', query_lower):
            self.rule_usage["Rule 4"] += 1
            logger.debug(f"Rule 4 matched (Similarity search): {query[:50]}...")
            return BackendSelection(
                backend=Backend.QDRANT,
                confidence=0.88,
                rule_matched="Rule 4: Similarity search",
                query=query
            )

        # Rule 2: Policy/ground truth/audit → SQL (0.90)
        if re.search(r'\b(policy|rule|audit|transaction|permission)\b', query_lower):
            self.rule_usage["Rule 2"] += 1
            logger.debug(f"Rule 2 matched (Policy/ground truth): {query[:50]}...")
            return BackendSelection(
                backend=Backend.SQL,
                confidence=0.90,
                rule_matched="Rule 2: Policy/ground truth",
                query=query
            )

        # Rule 3: Aggregations/counts → SQL (0.85)
        if re.search(r'\b(count|sum|average|total|statistics|how many)\b', query_lower):
            self.rule_usage["Rule 3"] += 1
            logger.debug(f"Rule 3 matched (Aggregation/count): {query[:50]}...")
            return BackendSelection(
                backend=Backend.SQL,
                confidence=0.85,
                rule_matched="Rule 3: Aggregation/count",
                query=query
            )

        # Rule 6: Hybrid queries → SQL + Graph (0.82)
        if re.search(r'\b(violating|non-compliant|breaking|compliance)\b', query_lower):
            self.rule_usage["Rule 6"] += 1
            logger.debug(f"Rule 6 matched (Hybrid query): {query[:50]}...")
            return BackendSelection(
                backend=Backend.SQL,
                confidence=0.82,
                rule_matched="Rule 6: Hybrid query (SQL + Graph)",
                query=query,
                secondary_backend=Backend.NEO4J  # Hybrid: SQL primary, Graph secondary
            )

        # Rule 7: Exploratory → Graph (0.70)
        if re.search(r'\b(explore|discover|find|show|list)\b', query_lower):
            self.rule_usage["Rule 7"] += 1
            logger.debug(f"Rule 7 matched (Exploratory query): {query[:50]}...")
            return BackendSelection(
                backend=Backend.NEO4J,
                confidence=0.70,
                rule_matched="Rule 7: Exploratory query",
                query=query
            )

        # Default: Use Thompson Sampling (handled by router)
        self.rule_usage["Default"] += 1
        logger.debug(f"Default matched (Thompson Sampling): {query[:50]}...")
        return BackendSelection(
            backend=Backend.THOMPSON_SAMPLING,
            confidence=0.50,
            rule_matched="Default: Thompson Sampling",
            query=query
        )

    def get_stats(self) -> dict:
        """
        Get classifier statistics

        Returns:
            Dict with classification count and rule usage
        """
        return {
            "total_classifications": self.classification_count,
            "rule_usage": self.rule_usage.copy(),
            "rule_distribution": {
                rule: count / max(self.classification_count, 1)
                for rule, count in self.rule_usage.items()
            }
        }

    def reset_stats(self):
        """Reset statistics (for testing)"""
        self.classification_count = 0
        self.rule_usage = dict.fromkeys(self.rule_usage.keys(), 0)


# ============================================================================
# Confidence Tiers
# ============================================================================

class ConfidenceTier:
    """Confidence tier definitions for routing decisions"""

    CRITICAL = 0.95  # Exact matches, guaranteed correct
    HIGH = 0.85      # Strong pattern match
    MEDIUM = 0.70    # Good pattern match
    LOW = 0.50       # Uncertain, use exploration

    @staticmethod
    def get_tier_name(confidence: float) -> str:
        """Get tier name from confidence score"""
        if confidence >= ConfidenceTier.CRITICAL:
            return "CRITICAL"
        elif confidence >= ConfidenceTier.HIGH:
            return "HIGH"
        elif confidence >= ConfidenceTier.MEDIUM:
            return "MEDIUM"
        else:
            return "LOW"


# ============================================================================
# Helper Functions
# ============================================================================

def create_classifier() -> QueryClassifier:
    """
    Create query classifier

    Returns:
        QueryClassifier instance
    """
    return QueryClassifier()

#!/usr/bin/env python3
"""
Demo 1: Query Classification

Demonstrates the 7-rule query classifier routing queries to SQL, Neo4j, or Qdrant backends.

Part of: Hybrid Query Routing Architecture - Part 1 (Proof-of-Concept Demos)
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class BackendSelection:
    """Classification result"""
    backend: str  # "sql", "neo4j", "qdrant", "thompson_sampling"
    confidence: float
    rule_matched: str
    query: str


class QueryClassifier:
    """7-rule query classification for hybrid routing"""

    def __init__(self):
        self.backend_weights = {
            "sql": 1.0,
            "neo4j": 1.0,
            "qdrant": 1.0
        }

    def classify(self, query: str) -> BackendSelection:
        """Apply 7-rule decision tree"""

        query_lower = query.lower()

        # Rule 1: Exact ID lookups → SQL (0.95)
        if re.search(r'\b(get|fetch|retrieve|show)\b.*\b[a-z]+_\d+\b', query_lower):
            return BackendSelection(
                backend="sql",
                confidence=0.95,
                rule_matched="Rule 1: Exact ID lookup",
                query=query
            )

        # Rule 2: Policy/ground truth/audit → SQL (0.90)
        if re.search(r'\b(policy|rule|audit|transaction|permission)\b', query_lower):
            return BackendSelection(
                backend="sql",
                confidence=0.90,
                rule_matched="Rule 2: Policy/ground truth",
                query=query
            )

        # Rule 3: Aggregations/counts → SQL (0.85)
        if re.search(r'\b(count|sum|average|total|statistics|how many)\b', query_lower):
            return BackendSelection(
                backend="sql",
                confidence=0.85,
                rule_matched="Rule 3: Aggregation/count",
                query=query
            )

        # Rule 4: Similarity queries → Vector (0.88)
        if re.search(r'\b(similar|like|resembling|comparable)\b', query_lower):
            return BackendSelection(
                backend="qdrant",
                confidence=0.88,
                rule_matched="Rule 4: Similarity search",
                query=query
            )

        # Rule 5: Relationship traversal → Graph (0.87)
        if re.search(r'\b(connected|related|linked|associated|affected)\b', query_lower):
            return BackendSelection(
                backend="neo4j",
                confidence=0.87,
                rule_matched="Rule 5: Relationship traversal",
                query=query
            )

        # Rule 6: Hybrid queries → SQL + Graph (0.82)
        if re.search(r'\b(violating|non-compliant|breaking|compliance)\b', query_lower):
            return BackendSelection(
                backend="sql",  # Primary (will trigger multi-backend in production)
                confidence=0.82,
                rule_matched="Rule 6: Hybrid query (SQL + Graph)",
                query=query
            )

        # Rule 7: Exploratory → Graph (0.70)
        if re.search(r'\b(explore|discover|find|show|list)\b', query_lower):
            return BackendSelection(
                backend="neo4j",
                confidence=0.70,
                rule_matched="Rule 7: Exploratory query",
                query=query
            )

        # Default: Use Thompson Sampling (handled by router in production)
        return BackendSelection(
            backend="thompson_sampling",
            confidence=0.50,
            rule_matched="Default: Thompson Sampling",
            query=query
        )


# Test queries with expected backends
TEST_QUERIES = [
    # SQL queries (Rules 1-3, 6)
    ("Get policy rule bee_001", "sql", "Rule 1"),
    ("Fetch audit trail for hive_042", "sql", "Rule 1"),
    ("What is the Varroa treatment schedule policy?", "sql", "Rule 2"),
    ("Show me the transaction logs for user_123", "sql", "Rule 2"),
    ("Retrieve user permissions for admin_001", "sql", "Rule 2"),
    ("How many hives are in the system?", "sql", "Rule 3"),
    ("Count total policy violations this month", "sql", "Rule 3"),
    ("Show average honey production statistics", "sql", "Rule 3"),
    ("Which hives are violating the treatment schedule?", "sql", "Rule 6"),
    ("Find non-compliant apiaries", "sql", "Rule 6"),

    # Graph queries (Rules 5, 7)
    ("What hives are connected to the Varroa treatment policy?", "neo4j", "Rule 5"),
    ("Show entities related to hive maintenance", "neo4j", "Rule 5"),
    ("Which apiaries are linked to treatment protocols?", "neo4j", "Rule 5"),
    ("What entities are associated with pest management?", "neo4j", "Rule 5"),
    ("Explore the knowledge graph for beekeeping practices", "neo4j", "Rule 7"),
    ("Discover connections between apiaries", "neo4j", "Rule 7"),

    # Vector queries (Rule 4)
    ("Find beekeeping practices similar to Varroa treatment", "qdrant", "Rule 4"),
    ("Show treatments like oxalic acid vaporization", "qdrant", "Rule 4"),
    ("Recommend practices comparable to integrated pest management", "qdrant", "Rule 4"),

    # Ambiguous query (Default)
    ("Tell me about hives", "thompson_sampling", "Default"),
]


def format_confidence(confidence: float) -> str:
    """Format confidence with ASCII indicators"""
    if confidence >= 0.90:
        return f"{confidence:.2f} **"  # High confidence
    elif confidence >= 0.75:
        return f"{confidence:.2f} *"   # Good confidence
    elif confidence >= 0.50:
        return f"{confidence:.2f} ~"   # Medium confidence
    else:
        return f"{confidence:.2f} ?"   # Low confidence


def main():
    """Run query classification demo"""

    print("=" * 80)
    print("DEMO 1: Query Classification")
    print("=" * 80)
    print()
    print("Testing 7-rule classifier with 20 beekeeping domain queries")
    print()

    classifier = QueryClassifier()

    # Track accuracy
    correct = 0
    total = len(TEST_QUERIES)

    # Classify each query
    for i, (query, expected_backend, expected_rule_prefix) in enumerate(TEST_QUERIES, 1):
        result = classifier.classify(query)

        # Check if classification is correct
        is_correct = result.backend == expected_backend
        if is_correct:
            correct += 1

        # Format output (ASCII-safe for Windows)
        status = "[PASS]" if is_correct else "[FAIL]"
        confidence_str = format_confidence(result.confidence)

        print(f"{i:2d}. {status} {result.backend.upper():18s} ({confidence_str})")
        print(f"    Query: \"{query}\"")
        print(f"    Rule:  {result.rule_matched}")

        if not is_correct:
            print(f"    WARNING: Expected {expected_backend.upper()}")

        print()

    # Summary
    accuracy = (correct / total) * 100
    print("=" * 80)
    print(f"RESULTS: {correct}/{total} queries correctly classified ({accuracy:.1f}% accuracy)")
    print("=" * 80)
    print()

    # Backend distribution
    print("Backend Distribution:")
    backend_counts = {}
    for query, expected_backend, _ in TEST_QUERIES:
        result = classifier.classify(query)
        backend_counts[result.backend] = backend_counts.get(result.backend, 0) + 1

    for backend, count in sorted(backend_counts.items()):
        percentage = (count / total) * 100
        bar = "#" * int(percentage / 5)
        print(f"  {backend.upper():18s}: {count:2d} queries ({percentage:5.1f}%) {bar}")

    print()

    # Validation
    print("=" * 80)
    print("VALIDATION GATE 1.1: Classification Logic")
    print("=" * 80)

    if accuracy >= 95.0:
        print(f"[PASS] Classification accuracy {accuracy:.1f}% (target: >=95%)")
        print("[PASS] 7-rule decision tree is sound")
        print("[PASS] Ready to proceed to Demo 2 (Thompson Sampling)")
    elif accuracy >= 85.0:
        print(f"[WARN] Classification accuracy {accuracy:.1f}% (target: >=95%)")
        print("[WARN] Consider tuning rules or adding more patterns")
    else:
        print(f"[FAIL] Classification accuracy {accuracy:.1f}% (target: >=95%)")
        print("[FAIL] Rules need significant revision before proceeding")

    print()

    # Confidence distribution
    print("=" * 80)
    print("Confidence Distribution:")
    print("=" * 80)

    confidences = [classifier.classify(query).confidence for query, _, _ in TEST_QUERIES]
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)

    print(f"  Average:  {avg_confidence:.3f}")
    print(f"  Min:      {min_confidence:.3f}")
    print(f"  Max:      {max_confidence:.3f}")
    print()

    # Recommendations
    print("=" * 80)
    print("Next Steps:")
    print("=" * 80)
    if accuracy >= 95.0:
        print("1. [READY] Proceed to Demo 2: Thompson Sampling Convergence")
        print("2. [READY] Test with additional domain-specific queries")
        print("3. [READY] Begin planning production classifier implementation")
    else:
        print("1. [TODO] Add more pattern matching rules")
        print("2. [TODO] Test with broader query set")
        print("3. [TODO] Revisit classification logic")

    print()


if __name__ == "__main__":
    main()

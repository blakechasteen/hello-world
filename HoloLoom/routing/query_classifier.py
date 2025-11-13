"""
Smart Query Routing - Complexity Classification
================================================
Classifies queries by complexity to route them to appropriate fast paths.

Query Types:
- TRIVIAL: Greetings, acknowledgments (<10ms target)
- SIMPLE: Factual lookups, definitions (<50ms target)
- COMPLEX: Multi-step reasoning, analysis (<150ms target)
- RESEARCH: Deep exploration, no time limit

Author: Claude Code
Date: 2025-11-12
"""

from enum import Enum
from typing import Set, Pattern
import re
from dataclasses import dataclass


class QueryComplexity(Enum):
    """Query complexity levels for routing."""
    TRIVIAL = "trivial"      # <10ms: "hi", "thanks", "ok"
    SIMPLE = "simple"        # <50ms: "what is X?", "define Y"
    COMPLEX = "complex"      # <150ms: "explain X in detail"
    RESEARCH = "research"    # No limit: "analyze all tradeoffs of X vs Y"


@dataclass
class ClassificationResult:
    """Result of query classification."""
    complexity: QueryComplexity
    confidence: float  # 0-1
    reasoning: str
    detected_patterns: Set[str] = None

    def __post_init__(self):
        if self.detected_patterns is None:
            self.detected_patterns = set()


class QueryClassifier:
    """
    Classifies queries by complexity for smart routing.

    Uses pattern matching and heuristics to determine query complexity.
    Fast (<1ms overhead) to avoid slowing down the pipeline.
    """

    # Trivial patterns (greetings, acknowledgments)
    TRIVIAL_PATTERNS = {
        r'^(hi|hello|hey|yo)[\s!?]*$',
        r'^(thanks|thank you|thx)[\s!?.]*$',
        r'^(bye|goodbye|see you|cya)[\s!?.]*$',
        r'^(ok|okay|sure|yes|no|yep|nope)[\s!?.]*$',
        r'^(help|info)[\s!?]*$',
    }

    # Simple patterns (factual lookups)
    SIMPLE_PATTERNS = {
        r'^what is \w+\??$',  # "what is X?"
        r'^define \w+',  # "define X"
        r'^who is \w+',  # "who is X"
        r'^when (is|was|will)',  # "when is X"
        r'^where (is|was)',  # "where is X"
    }

    # Research indicators (deep exploration)
    RESEARCH_KEYWORDS = {
        'analyze', 'compare', 'evaluate', 'assess', 'examine',
        'tradeoffs', 'trade-offs', 'comprehensive', 'detailed',
        'thorough', 'in-depth', 'extensive', 'deep dive',
        'versus', 'vs', 'pros and cons', 'advantages and disadvantages'
    }

    # Complexity indicators
    COMPLEXITY_KEYWORDS = {
        'explain': 0.3,  # Medium complexity
        'describe': 0.2,
        'how': 0.4,
        'why': 0.5,
        'elaborate': 0.6,
        'clarify': 0.3,
    }

    def __init__(self):
        """Initialize classifier."""
        # Compile patterns for performance
        self._trivial_regex = [re.compile(p, re.IGNORECASE) for p in self.TRIVIAL_PATTERNS]
        self._simple_regex = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]

    def classify(self, query_text: str) -> ClassificationResult:
        """
        Classify query complexity.

        Args:
            query_text: Query string

        Returns:
            ClassificationResult with complexity level and reasoning
        """
        text = query_text.strip().lower()
        detected_patterns = set()

        # Check trivial patterns first (fastest path)
        for pattern in self._trivial_regex:
            if pattern.match(text):
                detected_patterns.add("trivial_match")
                return ClassificationResult(
                    complexity=QueryComplexity.TRIVIAL,
                    confidence=0.95,
                    reasoning="Matches trivial pattern (greeting/acknowledgment)",
                    detected_patterns=detected_patterns
                )

        # Check simple patterns
        for pattern in self._simple_regex:
            if pattern.match(text):
                detected_patterns.add("simple_match")
                # Additional check: very short queries are likely simple
                if len(text.split()) <= 5:
                    return ClassificationResult(
                        complexity=QueryComplexity.SIMPLE,
                        confidence=0.85,
                        reasoning="Matches simple pattern (factual lookup)",
                        detected_patterns=detected_patterns
                    )

        # Check for research indicators
        research_score = sum(
            1 for keyword in self.RESEARCH_KEYWORDS
            if keyword in text
        )

        if research_score >= 2:
            detected_patterns.add("research_keywords")
            return ClassificationResult(
                complexity=QueryComplexity.RESEARCH,
                confidence=0.90,
                reasoning=f"Contains {research_score} research keywords",
                detected_patterns=detected_patterns
            )

        # Heuristic: Length-based complexity estimation
        word_count = len(text.split())
        detected_patterns.add(f"word_count_{word_count}")

        if word_count <= 3:
            # Very short queries are usually simple
            return ClassificationResult(
                complexity=QueryComplexity.SIMPLE,
                confidence=0.70,
                reasoning=f"Short query ({word_count} words)",
                detected_patterns=detected_patterns
            )

        if word_count >= 20:
            # Long queries are usually complex/research
            detected_patterns.add("long_query")
            return ClassificationResult(
                complexity=QueryComplexity.RESEARCH if word_count >= 30 else QueryComplexity.COMPLEX,
                confidence=0.75,
                reasoning=f"Long query ({word_count} words)",
                detected_patterns=detected_patterns
            )

        # Check for complexity keywords
        complexity_score = sum(
            weight for keyword, weight in self.COMPLEXITY_KEYWORDS.items()
            if keyword in text
        )

        if complexity_score >= 0.5:
            detected_patterns.add("complexity_keywords")
            return ClassificationResult(
                complexity=QueryComplexity.COMPLEX,
                confidence=0.80,
                reasoning=f"Contains complexity keywords (score: {complexity_score:.1f})",
                detected_patterns=detected_patterns
            )

        # Default: SIMPLE (conservative routing)
        return ClassificationResult(
            complexity=QueryComplexity.SIMPLE,
            confidence=0.60,
            reasoning="Default classification (no strong indicators)",
            detected_patterns=detected_patterns
        )

    def estimate_latency_budget(self, complexity: QueryComplexity) -> float:
        """
        Estimate latency budget for a given complexity.

        Args:
            complexity: Query complexity level

        Returns:
            Latency budget in milliseconds
        """
        budgets = {
            QueryComplexity.TRIVIAL: 10,
            QueryComplexity.SIMPLE: 50,
            QueryComplexity.COMPLEX: 150,
            QueryComplexity.RESEARCH: float('inf')
        }
        return budgets.get(complexity, 150)

    def should_use_fast_path(self, complexity: QueryComplexity) -> bool:
        """
        Determine if query should use fast path.

        Args:
            complexity: Query complexity level

        Returns:
            True if should use fast path
        """
        return complexity in {QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE}


# Convenience function
def classify_query(query_text: str) -> ClassificationResult:
    """
    Classify a query (convenience function).

    Args:
        query_text: Query string

    Returns:
        ClassificationResult
    """
    classifier = QueryClassifier()
    return classifier.classify(query_text)

"""
Query Decomposition - Multi-Level Query Breaking
=================================================

Decomposes complex queries into sub-queries for parallel solving and synthesis.

Philosophy:
Complex queries often require breaking down into simpler sub-problems that can
be solved independently and then synthesized into a complete answer. This module
implements recursive query decomposition with configurable depth and synthesis
strategies.

Created: 2025-11-20
Author: HoloLoom Team
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from hololoom.protocols.recursive_reasoning import DecompositionTree

logger = logging.getLogger(__name__)


# ============================================================================
# Complexity Detection Heuristics
# ============================================================================

class ComplexityDetector:
    """
    Detect query complexity using multiple heuristics.

    Factors considered:
    - Query length
    - Number of conjunctions (and, or, but)
    - Number of questions (?,  how, what, why)
    - Technical keywords
    - Nested structure (parentheses, clauses)
    """

    def __init__(self):
        self.conjunction_words = ["and", "or", "but", "also", "additionally", "furthermore"]
        self.question_words = ["what", "why", "how", "when", "where", "who", "which"]
        self.complexity_keywords = [
            "compare", "contrast", "analyze", "evaluate", "synthesize",
            "tradeoffs", "implications", "relationship", "impact"
        ]

    def detect(self, query: str) -> float:
        """
        Detect query complexity (0.0-1.0).

        Args:
            query: Query text

        Returns:
            Complexity score (0.0 = simple, 1.0 = very complex)
        """
        query_lower = query.lower()
        words = query.split()
        score = 0.0

        # Factor 1: Length (longer = more complex)
        length_score = min(len(words) / 50.0, 0.3)  # Max 0.3
        score += length_score

        # Factor 2: Conjunctions (multiple parts)
        conjunction_count = sum(1 for word in self.conjunction_words if word in query_lower)
        conjunction_score = min(conjunction_count / 5.0, 0.2)  # Max 0.2
        score += conjunction_score

        # Factor 3: Multiple questions
        question_count = query.count("?") + sum(1 for word in self.question_words if word in query_lower)
        question_score = min(question_count / 3.0, 0.2)  # Max 0.2
        score += question_score

        # Factor 4: Complexity keywords
        keyword_count = sum(1 for keyword in self.complexity_keywords if keyword in query_lower)
        keyword_score = min(keyword_count / 3.0, 0.2)  # Max 0.2
        score += keyword_score

        # Factor 5: Nested structure (parentheses, semicolons)
        nesting_score = min((query.count("(") + query.count(";")) / 3.0, 0.1)  # Max 0.1
        score += nesting_score

        return min(score, 1.0)  # Cap at 1.0


# ============================================================================
# Query Decomposer
# ============================================================================

class QueryDecomposer:
    """
    Decompose complex queries into sub-queries.

    Strategies:
    1. **Conjunction splitting**: Split on "and", "or" conjunctions
    2. **Question extraction**: Extract multiple questions
    3. **Aspect-based**: Break into aspects (causes, effects, examples)
    4. **Hierarchical**: Break into high-level → detailed sub-queries
    """

    def __init__(
        self,
        max_depth: int = 3,
        max_sub_queries: int = 5,
        weaving_fn: Callable | None = None
    ):
        """
        Initialize decomposer.

        Args:
            max_depth: Maximum decomposition depth
            max_sub_queries: Maximum sub-queries per level
            weaving_fn: Optional function to execute queries (for synthesis)
        """
        self.max_depth = max_depth
        self.max_sub_queries = max_sub_queries
        self.weaving_fn = weaving_fn
        self.complexity_detector = ComplexityDetector()

        logger.info(f"QueryDecomposer initialized (max_depth={max_depth}, max_sub_queries={max_sub_queries})")

    def detect_complexity(self, query: str) -> float:
        """
        Estimate query complexity.

        Args:
            query: Query to analyze

        Returns:
            Complexity score (0.0-1.0)
        """
        return self.complexity_detector.detect(query)

    async def decompose(
        self,
        query: str,
        complexity_threshold: float = 0.7
    ) -> DecompositionTree:
        """
        Decompose query into sub-queries if complex enough.

        Args:
            query: Query to decompose
            complexity_threshold: Minimum complexity to trigger decomposition

        Returns:
            DecompositionTree with sub-queries
        """
        complexity = self.detect_complexity(query)
        logger.info(f"Query complexity: {complexity:.2f} (threshold={complexity_threshold})")

        if complexity < complexity_threshold:
            # Not complex enough, return single query
            return DecompositionTree(
                root_query=query,
                sub_queries=[query],
                depth=1,
                complexity_score=complexity,
                synthesis_strategy="direct"
            )

        # Decompose using appropriate strategy
        sub_queries = self._decompose_strategy(query, complexity)

        # Limit to max_sub_queries
        sub_queries = sub_queries[:self.max_sub_queries]

        # Determine synthesis strategy
        synthesis_strategy = self._select_synthesis_strategy(query, sub_queries)

        logger.info(f"Decomposed into {len(sub_queries)} sub-queries (strategy={synthesis_strategy})")

        return DecompositionTree(
            root_query=query,
            sub_queries=sub_queries,
            depth=1,  # Currently only 1-level deep (could extend to tree)
            complexity_score=complexity,
            synthesis_strategy=synthesis_strategy
        )

    def _decompose_strategy(self, query: str, complexity: float) -> list[str]:
        """
        Select and apply decomposition strategy.

        Args:
            query: Query to decompose
            complexity: Complexity score

        Returns:
            List of sub-queries
        """
        query_lower = query.lower()

        # Strategy 1: Conjunction splitting (for "and" queries)
        if " and " in query_lower or " or " in query_lower:
            return self._split_conjunctions(query)

        # Strategy 2: Multiple questions (for multi-question queries)
        if query.count("?") > 1:
            return self._extract_questions(query)

        # Strategy 3: Comparison splitting (for "compare X and Y")
        if "compare" in query_lower or "vs" in query_lower or "versus" in query_lower:
            return self._split_comparison(query)

        # Strategy 4: Aspect-based (for analytical queries)
        if any(word in query_lower for word in ["analyze", "evaluate", "assess"]):
            return self._aspect_based_decomposition(query)

        # Strategy 5: Fallback - hierarchical breakdown
        return self._hierarchical_decomposition(query)

    def _split_conjunctions(self, query: str) -> list[str]:
        """Split query on conjunctions (and, or)."""
        # Split on " and " and " or "
        parts = re.split(r'\s+(?:and|or)\s+', query, flags=re.IGNORECASE)

        # Clean up parts
        sub_queries = []
        for part in parts:
            part = part.strip()
            if part and len(part.split()) > 2:  # At least 3 words
                # Add question mark if missing
                if not part.endswith("?"):
                    part += "?"
                sub_queries.append(part)

        return sub_queries if sub_queries else [query]

    def _extract_questions(self, query: str) -> list[str]:
        """Extract multiple questions from query."""
        # Split on sentence boundaries with question marks
        questions = [q.strip() + "?" for q in query.split("?") if q.strip()]
        return questions if len(questions) > 1 else [query]

    def _split_comparison(self, query: str) -> list[str]:
        """Split comparison queries into individual analyses."""
        query_lower = query.lower()

        # Extract entities being compared
        if "compare" in query_lower:
            # "Compare X and Y" → "What is X?", "What is Y?", "How do X and Y differ?"
            match = re.search(r'compare\s+(.+?)\s+and\s+(.+?)(?:\?|$)', query, re.IGNORECASE)
            if match:
                entity1, entity2 = match.groups()
                return [
                    f"What is {entity1.strip()}?",
                    f"What is {entity2.strip()}?",
                    f"How do {entity1.strip()} and {entity2.strip()} differ?"
                ]

        # "X vs Y" → Similar breakdown
        if " vs " in query_lower or " versus " in query_lower:
            match = re.search(r'(.+?)\s+(?:vs|versus)\s+(.+?)(?:\?|$)', query, re.IGNORECASE)
            if match:
                entity1, entity2 = match.groups()
                return [
                    f"What is {entity1.strip()}?",
                    f"What is {entity2.strip()}?",
                    "What are the key differences?"
                ]

        return [query]

    def _aspect_based_decomposition(self, query: str) -> list[str]:
        """Decompose analytical queries into aspects."""
        # Common aspects for analysis
        aspects = [
            "definition",
            "advantages",
            "disadvantages",
            "use cases",
            "implementation"
        ]

        # Extract main topic
        topic = self._extract_topic(query)

        sub_queries = [
            f"What is {topic}?",
            f"What are the advantages of {topic}?",
            f"What are the disadvantages or limitations of {topic}?",
            f"What are common use cases for {topic}?",
        ]

        return sub_queries

    def _hierarchical_decomposition(self, query: str) -> list[str]:
        """Hierarchical breakdown: overview → details → examples."""
        topic = self._extract_topic(query)

        sub_queries = [
            f"What is a brief overview of {topic}?",
            f"What are the key details and components of {topic}?",
            f"Can you provide examples of {topic}?",
        ]

        return sub_queries

    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query (simple heuristic)."""
        # Remove question words
        query_clean = re.sub(r'\b(what|why|how|when|where|who|which|is|are|the)\b', '', query, flags=re.IGNORECASE)

        # Remove question mark
        query_clean = query_clean.replace("?", "").strip()

        # Take first few meaningful words (max 5)
        words = [w for w in query_clean.split() if len(w) > 2]
        topic = " ".join(words[:5])

        return topic if topic else query

    def _select_synthesis_strategy(self, root_query: str, sub_queries: list[str]) -> str:
        """
        Select synthesis strategy based on query type.

        Args:
            root_query: Original query
            sub_queries: Sub-queries

        Returns:
            Synthesis strategy name
        """
        query_lower = root_query.lower()

        if "compare" in query_lower or "vs" in query_lower:
            return "comparison_synthesis"
        elif any(word in query_lower for word in ["analyze", "evaluate", "assess"]):
            return "analytical_synthesis"
        elif len(sub_queries) > 3:
            return "hierarchical_synthesis"
        else:
            return "sequential_synthesis"

    async def synthesize_sub_answers(
        self,
        sub_results: list[Any],
        synthesis_strategy: str = "sequential_synthesis"
    ) -> str:
        """
        Combine sub-query answers into final answer.

        Args:
            sub_results: List of sub-query results (Spacetime objects or strings)
            synthesis_strategy: How to combine answers

        Returns:
            Synthesized final answer
        """
        # Extract response texts from results
        answers = []
        for result in sub_results:
            if hasattr(result, 'response'):
                answers.append(result.response)
            elif isinstance(result, str):
                answers.append(result)
            else:
                answers.append(str(result))

        # Apply synthesis strategy
        if synthesis_strategy == "comparison_synthesis":
            return self._synthesize_comparison(answers)
        elif synthesis_strategy == "analytical_synthesis":
            return self._synthesize_analytical(answers)
        elif synthesis_strategy == "hierarchical_synthesis":
            return self._synthesize_hierarchical(answers)
        else:
            return self._synthesize_sequential(answers)

    def _synthesize_sequential(self, answers: list[str]) -> str:
        """Concatenate answers in sequence."""
        synthesized = "\n\n".join(f"**Part {i+1}:**\n{answer}" for i, answer in enumerate(answers))
        return synthesized

    def _synthesize_comparison(self, answers: list[str]) -> str:
        """Synthesize comparison answers."""
        if len(answers) < 3:
            return self._synthesize_sequential(answers)

        synthesized = f"""**Entity 1:**
{answers[0]}

**Entity 2:**
{answers[1]}

**Key Differences:**
{answers[2]}
"""
        return synthesized

    def _synthesize_analytical(self, answers: list[str]) -> str:
        """Synthesize analytical answers with structure."""
        sections = ["Overview", "Advantages", "Disadvantages", "Use Cases"]
        synthesized_parts = []

        for i, (section, answer) in enumerate(zip(sections, answers)):
            synthesized_parts.append(f"**{section}:**\n{answer}")

        return "\n\n".join(synthesized_parts)

    def _synthesize_hierarchical(self, answers: list[str]) -> str:
        """Synthesize hierarchical answers."""
        if len(answers) >= 3:
            return f"""**Overview:**
{answers[0]}

**Details:**
{answers[1]}

**Examples:**
{answers[2]}
"""
        else:
            return self._synthesize_sequential(answers)


# ============================================================================
# Factory
# ============================================================================

def create_query_decomposer(
    max_depth: int = 3,
    max_sub_queries: int = 5,
    weaving_fn: Callable | None = None
) -> QueryDecomposer:
    """
    Create query decomposer.

    Args:
        max_depth: Maximum decomposition depth
        max_sub_queries: Maximum sub-queries per level
        weaving_fn: Optional function to execute queries

    Returns:
        Configured QueryDecomposer
    """
    return QueryDecomposer(
        max_depth=max_depth,
        max_sub_queries=max_sub_queries,
        weaving_fn=weaving_fn
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("Query Decomposition Demo")
        print("="*80 + "\n")

        decomposer = QueryDecomposer(max_depth=2, max_sub_queries=5)

        # Test queries
        test_queries = [
            "What is Thompson Sampling?",  # Simple
            "Compare Thompson Sampling and Upper Confidence Bound algorithms",  # Comparison
            "Analyze the tradeoffs of reinforcement learning in production systems",  # Analytical
            "What is recursion and how does it work in Python and when should I use it?",  # Multiple questions
        ]

        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")

            # Detect complexity
            complexity = decomposer.detect_complexity(query)
            print(f"\nComplexity: {complexity:.2f}")

            # Decompose
            tree = await decomposer.decompose(query, complexity_threshold=0.5)

            print("\nDecomposition:")
            print(f"  Root: {tree.root_query}")
            print(f"  Depth: {tree.depth}")
            print(f"  Synthesis Strategy: {tree.synthesis_strategy}")
            print("\n  Sub-queries:")
            for i, sub in enumerate(tree.sub_queries, 1):
                print(f"    {i}. {sub}")

        print("\n" + "="*80)
        print("✓ Demo complete!")

    asyncio.run(demo())

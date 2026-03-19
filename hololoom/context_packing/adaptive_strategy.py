"""
Adaptive Compression Strategy - Query Complexity to MI Budget Mapping
=====================================================================

Phase 6.1: Automatically selects compression strategy based on query
complexity using the existing query classifier.

Complexity Levels -> MI Budget Mapping:
- TRIVIAL: 2.0 bits (aggressive compression, greetings/acks)
- SIMPLE: 3.0 bits (aggressive, factual lookups)
- MODERATE: 5.0 bits (balanced, explanations)
- COMPLEX: 8.0 bits (conservative, comparisons)
- RESEARCH: 15.0 bits (minimal compression, full exploration)

Author: Claude Code
Date: 2025-12-09
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from .config import ContextPackerConfig


class AdaptiveComplexity(Enum):
    """
    Extended query complexity levels with MI budget mapping.

    Extends routing.QueryComplexity with MODERATE level for
    finer-grained compression strategy selection.
    """
    TRIVIAL = "trivial"      # Greetings, acknowledgments
    SIMPLE = "simple"        # Factual lookups
    MODERATE = "moderate"    # Explanations, descriptions (NEW)
    COMPLEX = "complex"      # Comparisons, analysis
    RESEARCH = "research"    # Deep exploration


@dataclass
class StrategySelection:
    """Result of adaptive strategy selection."""
    complexity: AdaptiveComplexity
    config: ContextPackerConfig
    mi_budget: float
    compression_ratio: float
    confidence: float
    reasoning: str
    detected_indicators: set = field(default_factory=set)


# MI Budget constants per complexity level
MI_BUDGETS = {
    AdaptiveComplexity.TRIVIAL: 2.0,    # Minimal context needed
    AdaptiveComplexity.SIMPLE: 3.0,     # Basic factual lookup
    AdaptiveComplexity.MODERATE: 5.0,   # Standard explanation
    AdaptiveComplexity.COMPLEX: 8.0,    # Multi-factor analysis
    AdaptiveComplexity.RESEARCH: 15.0,  # Full exploration
}

# Compression ratios (how much to keep)
COMPRESSION_RATIOS = {
    AdaptiveComplexity.TRIVIAL: 0.30,   # Keep 30%
    AdaptiveComplexity.SIMPLE: 0.30,    # Keep 30%
    AdaptiveComplexity.MODERATE: 0.50,  # Keep 50%
    AdaptiveComplexity.COMPLEX: 0.70,   # Keep 70%
    AdaptiveComplexity.RESEARCH: 0.90,  # Keep 90%
}


class AdaptiveCompressionStrategy:
    """
    Selects optimal compression strategy based on query characteristics.

    Uses lightweight heuristics to classify query complexity and map
    to appropriate MI budget and ContextPackerConfig preset.

    Design Principle: Fast classification (<1ms) with reasonable accuracy.
    Learning can improve accuracy over time (Phase 6.4).
    """

    # Trivial patterns (greetings, acknowledgments)
    TRIVIAL_PATTERNS = [
        r'^(hi|hello|hey|yo)[\s!?]*$',
        r'^(thanks|thank you|thx)[\s!?.]*$',
        r'^(bye|goodbye|see you|cya)[\s!?.]*$',
        r'^(ok|okay|sure|yes|no|yep|nope)[\s!?.]*$',
        r'^(help|info|\?)[\s!?]*$',
    ]

    # Simple patterns (factual lookups, definitions)
    SIMPLE_PATTERNS = [
        r'^what is \w+\??$',       # "what is X?"
        r'^define \w+',            # "define X"
        r'^who is \w+',            # "who is X"
        r'^when (is|was|will)',    # "when is X"
        r'^where (is|was)',        # "where is X"
        r'^list \w+',              # "list X"
    ]

    # Moderate indicators (explanations, descriptions)
    MODERATE_KEYWORDS = {
        'explain', 'describe', 'what does', 'how does',
        'tell me about', 'summarize', 'overview', 'brief',
    }

    # Complex indicators (analysis, comparison)
    COMPLEX_KEYWORDS = {
        'compare', 'contrast', 'versus', 'vs', 'difference',
        'pros and cons', 'advantages', 'disadvantages',
        'why', 'how', 'relationship', 'impact',
    }

    # Research indicators (deep exploration)
    RESEARCH_KEYWORDS = {
        'analyze', 'evaluate', 'assess', 'examine', 'investigate',
        'comprehensive', 'detailed', 'thorough', 'in-depth', 'extensive',
        'tradeoffs', 'trade-offs', 'deep dive', 'all aspects',
        'research', 'study', 'explore thoroughly',
    }

    def __init__(self, enable_learning: bool = False):
        """
        Initialize adaptive strategy selector.

        Args:
            enable_learning: Whether to enable learning from outcomes (Phase 6.4)
        """
        self._trivial_regex = [re.compile(p, re.IGNORECASE) for p in self.TRIVIAL_PATTERNS]
        self._simple_regex = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
        self._enable_learning = enable_learning

        # Statistics for learning (Phase 6.4)
        self._selection_counts: dict[AdaptiveComplexity, int] = dict.fromkeys(AdaptiveComplexity, 0)
        self._outcome_scores: dict[AdaptiveComplexity, list] = {c: [] for c in AdaptiveComplexity}

    def select_strategy(self, query: str) -> StrategySelection:
        """
        Select optimal compression strategy for query.

        Args:
            query: Query text

        Returns:
            StrategySelection with complexity, config, and MI budget
        """
        text = query.strip().lower()
        indicators = set()

        # Step 1: Check trivial patterns (fastest path)
        for pattern in self._trivial_regex:
            if pattern.match(text):
                indicators.add("trivial_pattern_match")
                return self._build_selection(
                    AdaptiveComplexity.TRIVIAL,
                    confidence=0.95,
                    reasoning="Matches trivial pattern (greeting/acknowledgment)",
                    indicators=indicators
                )

        # Step 2: Check simple patterns
        for pattern in self._simple_regex:
            if pattern.match(text):
                indicators.add("simple_pattern_match")
                return self._build_selection(
                    AdaptiveComplexity.SIMPLE,
                    confidence=0.90,
                    reasoning="Matches simple factual pattern",
                    indicators=indicators
                )

        # Step 3: Count keyword matches at each level
        research_score = sum(1 for kw in self.RESEARCH_KEYWORDS if kw in text)
        complex_score = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in text)
        moderate_score = sum(1 for kw in self.MODERATE_KEYWORDS if kw in text)

        # Step 4: Check for research indicators (highest priority)
        if research_score >= 2 or (research_score >= 1 and len(text.split()) >= 25):
            indicators.add(f"research_keywords_{research_score}")
            indicators.add("long_query" if len(text.split()) >= 25 else "keyword_match")
            return self._build_selection(
                AdaptiveComplexity.RESEARCH,
                confidence=0.90,
                reasoning=f"Research-level query ({research_score} research keywords)",
                indicators=indicators
            )

        # Step 5: Check for complex indicators
        if complex_score >= 2 or (complex_score >= 1 and len(text.split()) >= 15):
            indicators.add(f"complex_keywords_{complex_score}")
            return self._build_selection(
                AdaptiveComplexity.COMPLEX,
                confidence=0.85,
                reasoning=f"Complex query ({complex_score} analysis keywords)",
                indicators=indicators
            )

        # Step 6: Check for moderate indicators
        if moderate_score >= 1 or len(text.split()) >= 8:
            indicators.add(f"moderate_keywords_{moderate_score}")
            indicators.add(f"word_count_{len(text.split())}")
            return self._build_selection(
                AdaptiveComplexity.MODERATE,
                confidence=0.80,
                reasoning="Moderate query (explanation/description)",
                indicators=indicators
            )

        # Step 7: Length-based fallback
        word_count = len(text.split())
        indicators.add(f"word_count_{word_count}")

        if word_count <= 3:
            return self._build_selection(
                AdaptiveComplexity.SIMPLE,
                confidence=0.70,
                reasoning=f"Short query ({word_count} words)",
                indicators=indicators
            )

        # Default: MODERATE (balanced)
        return self._build_selection(
            AdaptiveComplexity.MODERATE,
            confidence=0.60,
            reasoning="Default classification (moderate complexity)",
            indicators=indicators
        )

    def _build_selection(
        self,
        complexity: AdaptiveComplexity,
        confidence: float,
        reasoning: str,
        indicators: set
    ) -> StrategySelection:
        """Build StrategySelection with appropriate config."""
        config = self._get_config_for_complexity(complexity)
        mi_budget = MI_BUDGETS[complexity]
        compression_ratio = COMPRESSION_RATIOS[complexity]

        # Update config with MI budget
        config.compression.information_budget = mi_budget
        config.compression.target_ratio = compression_ratio

        # Track for learning
        self._selection_counts[complexity] += 1

        return StrategySelection(
            complexity=complexity,
            config=config,
            mi_budget=mi_budget,
            compression_ratio=compression_ratio,
            confidence=confidence,
            reasoning=reasoning,
            detected_indicators=indicators
        )

    def _get_config_for_complexity(self, complexity: AdaptiveComplexity) -> ContextPackerConfig:
        """Get ContextPackerConfig for complexity level."""
        if complexity == AdaptiveComplexity.TRIVIAL:
            return ContextPackerConfig.aggressive()
        elif complexity == AdaptiveComplexity.SIMPLE:
            return ContextPackerConfig.aggressive()
        elif complexity == AdaptiveComplexity.MODERATE:
            return ContextPackerConfig.balanced()
        elif complexity == AdaptiveComplexity.COMPLEX:
            return ContextPackerConfig.conservative()
        else:  # RESEARCH
            return ContextPackerConfig.research()

    def get_mi_budget(self, query: str) -> float:
        """
        Get MI budget for query (convenience method).

        Args:
            query: Query text

        Returns:
            MI budget in bits
        """
        selection = self.select_strategy(query)
        return selection.mi_budget

    def get_statistics(self) -> dict[str, any]:
        """Get selection statistics for monitoring."""
        total = sum(self._selection_counts.values())
        return {
            "total_selections": total,
            "by_complexity": {
                c.value: {
                    "count": self._selection_counts[c],
                    "percentage": (self._selection_counts[c] / total * 100) if total > 0 else 0
                }
                for c in AdaptiveComplexity
            }
        }

    def record_outcome(self, complexity: AdaptiveComplexity, quality_score: float):
        """
        Record outcome for learning (Phase 6.4).

        Args:
            complexity: Complexity level used
            quality_score: Quality of result (0-1)
        """
        if self._enable_learning:
            self._outcome_scores[complexity].append(quality_score)
            # Keep last 100 outcomes per complexity
            if len(self._outcome_scores[complexity]) > 100:
                self._outcome_scores[complexity] = self._outcome_scores[complexity][-100:]


# Convenience functions

def select_adaptive_strategy(query: str) -> StrategySelection:
    """
    Select adaptive compression strategy for query.

    Args:
        query: Query text

    Returns:
        StrategySelection with complexity, config, and MI budget
    """
    strategy = AdaptiveCompressionStrategy()
    return strategy.select_strategy(query)


def get_adaptive_mi_budget(query: str) -> float:
    """
    Get adaptive MI budget for query.

    Args:
        query: Query text

    Returns:
        MI budget in bits
    """
    strategy = AdaptiveCompressionStrategy()
    return strategy.get_mi_budget(query)


def get_adaptive_config(query: str) -> tuple[ContextPackerConfig, float]:
    """
    Get adaptive ContextPackerConfig and MI budget for query.

    Args:
        query: Query text

    Returns:
        Tuple of (ContextPackerConfig, MI budget)
    """
    selection = select_adaptive_strategy(query)
    return selection.config, selection.mi_budget

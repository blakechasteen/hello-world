#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Complexity Analyzer

Analyzes query complexity to determine optimal token budget:
- Simple queries: "What is X?" → Small budget (2k tokens)
- Moderate queries: "Explain X and Y" → Medium budget (4k-8k tokens)
- Complex queries: "Compare X, Y, Z with examples" → Large budget (16k tokens)
- Research queries: "Comprehensive analysis of..." → Very large budget (32k+ tokens)

Complexity Factors:
1. Query length (longer = more complex)
2. Question depth (what < how < why < compare)
3. Number of entities/concepts (multiple entities = more complex)
4. Technical terminology (technical = more complex)
5. Requested detail level (simple vs. detailed vs. comprehensive)
6. Uncertainty level (high uncertainty = more context needed)

Usage:
    analyzer = QueryComplexityAnalyzer()
    complexity = analyzer.analyze("What is quantum tunneling?")

    print(complexity.level)  # "simple"
    print(complexity.score)  # 0.3 (0.0-1.0)
    print(complexity.recommended_budget)  # 2000 tokens
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import re


class ComplexityLevel(str, Enum):
    """Query complexity levels"""
    SIMPLE = "simple"          # "What is X?" - Basic factual queries
    MODERATE = "moderate"      # "Explain X and Y" - Multi-concept queries
    COMPLEX = "complex"        # "Compare X, Y, Z" - Deep analysis queries
    RESEARCH = "research"      # "Comprehensive analysis" - Research-level queries


@dataclass
class ComplexityAnalysis:
    """
    Query complexity analysis result.

    Contains:
    - Complexity level (SIMPLE/MODERATE/COMPLEX/RESEARCH)
    - Complexity score (0.0-1.0, continuous measure)
    - Recommended token budget (2k-32k+ tokens)
    - Contributing factors (what made it complex?)
    """

    # Core results
    level: ComplexityLevel
    score: float  # 0.0-1.0 (continuous complexity measure)

    # Budget recommendation
    recommended_budget: int  # Total tokens (2k-32k+)

    # Breakdown of factors
    factors: Dict[str, float]  # factor_name → contribution (0.0-1.0)

    # Metadata
    query: str
    query_length: int
    entity_count: int

    def __str__(self) -> str:
        return (
            f"ComplexityAnalysis(\n"
            f"  Level: {self.level.value}\n"
            f"  Score: {self.score:.2f}\n"
            f"  Budget: {self.recommended_budget} tokens\n"
            f"  Query: {self.query[:60]}...\n"
            f")"
        )


class QueryComplexityAnalyzer:
    """
    Analyze query complexity to determine optimal token budget.

    Uses heuristics to classify queries into complexity levels:
    - SIMPLE (0.0-0.3): Basic factual queries
    - MODERATE (0.3-0.6): Multi-concept explanations
    - COMPLEX (0.6-0.8): Deep analysis, comparisons
    - RESEARCH (0.8-1.0): Comprehensive research

    Factors considered:
    1. Query length (longer = more complex)
    2. Question type (what < how < why < compare)
    3. Entity count (multiple concepts)
    4. Technical terminology
    5. Detail level requests (simple/detailed/comprehensive)
    6. Uncertainty signals (if available from awareness)
    """

    def __init__(self):
        """Initialize complexity analyzer"""

        # Question type complexity weights
        self.question_weights = {
            'what': 0.2,      # Simple factual
            'who': 0.2,       # Simple factual
            'when': 0.2,      # Simple factual
            'where': 0.2,     # Simple factual
            'which': 0.3,     # Selection/comparison
            'how': 0.5,       # Process/mechanism
            'why': 0.6,       # Reasoning/causation
            'compare': 0.8,   # Deep analysis
            'analyze': 0.8,   # Deep analysis
            'evaluate': 0.8,  # Critical thinking
            'explain': 0.4,   # Explanation (moderate)
            'describe': 0.3,  # Description (moderate)
        }

        # Detail level indicators
        self.detail_indicators = {
            'simple': -0.1,       # Requesting simplicity
            'briefly': -0.1,      # Requesting brevity
            'detailed': 0.2,      # Requesting detail
            'comprehensive': 0.4, # Requesting thoroughness
            'in depth': 0.3,      # Requesting depth
            'thoroughly': 0.3,    # Requesting thoroughness
            'complete': 0.3,      # Requesting completeness
        }

        # Technical terminology indicators (common technical words)
        self.technical_terms = {
            'algorithm', 'architecture', 'framework', 'methodology',
            'implementation', 'optimization', 'integration', 'protocol',
            'quantum', 'neural', 'distributed', 'asynchronous',
            'paradigm', 'heuristic', 'stochastic', 'deterministic'
        }

    def analyze(
        self,
        query: str,
        awareness_context=None
    ) -> ComplexityAnalysis:
        """
        Analyze query complexity.

        Args:
            query: User query
            awareness_context: Optional UnifiedAwarenessContext (for uncertainty)

        Returns:
            ComplexityAnalysis with level, score, budget
        """
        factors = {}

        # Factor 1: Query length
        query_length = len(query)
        length_score = min(query_length / 200.0, 1.0)  # Normalize to 0-1
        factors['length'] = length_score * 0.2  # 20% weight

        # Factor 2: Question type
        question_score = self._analyze_question_type(query)
        factors['question_type'] = question_score * 0.25  # 25% weight

        # Factor 3: Entity count (rough estimate)
        entity_count = self._estimate_entity_count(query)
        entity_score = min(entity_count / 5.0, 1.0)  # 5+ entities = max complexity
        factors['entity_count'] = entity_score * 0.15  # 15% weight

        # Factor 4: Technical terminology
        technical_score = self._analyze_technical_terminology(query)
        factors['technical'] = technical_score * 0.15  # 15% weight

        # Factor 5: Detail level requests
        detail_score = self._analyze_detail_level(query)
        factors['detail_level'] = detail_score * 0.15  # 15% weight

        # Factor 6: Uncertainty (if awareness context available)
        uncertainty_score = 0.0
        if awareness_context:
            uncertainty_score = self._analyze_uncertainty(awareness_context)
            factors['uncertainty'] = uncertainty_score * 0.10  # 10% weight

        # Calculate total complexity score (0.0-1.0)
        total_score = sum(factors.values())
        total_score = max(0.0, min(1.0, total_score))  # Clamp to [0, 1]

        # Determine complexity level
        if total_score < 0.3:
            level = ComplexityLevel.SIMPLE
        elif total_score < 0.6:
            level = ComplexityLevel.MODERATE
        elif total_score < 0.8:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.RESEARCH

        # Calculate recommended budget
        recommended_budget = self._calculate_budget(total_score, level)

        return ComplexityAnalysis(
            level=level,
            score=total_score,
            recommended_budget=recommended_budget,
            factors=factors,
            query=query,
            query_length=query_length,
            entity_count=entity_count
        )

    def _analyze_question_type(self, query: str) -> float:
        """Analyze question type complexity (0.0-1.0)"""
        query_lower = query.lower()

        # Check for each question type
        max_weight = 0.0
        for question_word, weight in self.question_weights.items():
            if question_word in query_lower:
                max_weight = max(max_weight, weight)

        return max_weight

    def _estimate_entity_count(self, query: str) -> int:
        """Estimate number of entities/concepts in query"""
        # Simple heuristic: count capitalized words and technical terms
        words = query.split()

        # Count capitalized words (likely entities)
        capitalized = sum(1 for word in words if word and word[0].isupper())

        # Count conjunctions (and, or) as indicators of multiple concepts
        conjunctions = query.lower().count(' and ') + query.lower().count(' or ')

        # Estimate entity count
        entity_count = max(capitalized, conjunctions + 1)

        return entity_count

    def _analyze_technical_terminology(self, query: str) -> float:
        """Analyze technical terminology density (0.0-1.0)"""
        query_lower = query.lower()

        # Count technical terms
        technical_count = sum(
            1 for term in self.technical_terms
            if term in query_lower
        )

        # Normalize by query length (words)
        words = query_lower.split()
        if len(words) == 0:
            return 0.0

        # Technical density: technical terms / total words
        density = technical_count / len(words)

        # Scale to 0-1 (0.2 density = high complexity)
        score = min(density / 0.2, 1.0)

        return score

    def _analyze_detail_level(self, query: str) -> float:
        """Analyze requested detail level (0.0-1.0)"""
        query_lower = query.lower()

        # Check for detail indicators
        detail_adjustment = 0.0
        for indicator, adjustment in self.detail_indicators.items():
            if indicator in query_lower:
                detail_adjustment += adjustment

        # Base level: 0.5 (moderate)
        # Adjustments: -0.2 to +0.6
        score = 0.5 + detail_adjustment

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        return score

    def _analyze_uncertainty(self, awareness_context) -> float:
        """Analyze uncertainty from awareness context (0.0-1.0)"""
        # Extract uncertainty level
        if hasattr(awareness_context, 'confidence'):
            uncertainty = awareness_context.confidence.uncertainty_level
            return uncertainty  # Already 0-1

        return 0.0  # Default: no uncertainty boost

    def _calculate_budget(self, score: float, level: ComplexityLevel) -> int:
        """
        Calculate recommended token budget based on complexity.

        Budget ranges:
        - SIMPLE: 2k-4k tokens
        - MODERATE: 4k-8k tokens
        - COMPLEX: 8k-16k tokens
        - RESEARCH: 16k-32k tokens

        Uses linear interpolation within ranges.
        """
        if level == ComplexityLevel.SIMPLE:
            # 2k-4k tokens (score 0.0-0.3)
            min_budget, max_budget = 2000, 4000
            normalized_score = score / 0.3  # 0.0-1.0 within SIMPLE range

        elif level == ComplexityLevel.MODERATE:
            # 4k-8k tokens (score 0.3-0.6)
            min_budget, max_budget = 4000, 8000
            normalized_score = (score - 0.3) / 0.3  # 0.0-1.0 within MODERATE range

        elif level == ComplexityLevel.COMPLEX:
            # 8k-16k tokens (score 0.6-0.8)
            min_budget, max_budget = 8000, 16000
            normalized_score = (score - 0.6) / 0.2  # 0.0-1.0 within COMPLEX range

        else:  # RESEARCH
            # 16k-32k tokens (score 0.8-1.0)
            min_budget, max_budget = 16000, 32000
            normalized_score = (score - 0.8) / 0.2  # 0.0-1.0 within RESEARCH range

        # Linear interpolation
        budget = int(min_budget + (max_budget - min_budget) * normalized_score)

        # Round to nearest 500
        budget = round(budget / 500) * 500

        return budget

    def analyze_batch(
        self,
        queries: List[str],
        awareness_contexts: Optional[List[Any]] = None
    ) -> List[ComplexityAnalysis]:
        """
        Analyze multiple queries in batch.

        Args:
            queries: List of queries to analyze
            awareness_contexts: Optional list of awareness contexts (1:1 with queries)

        Returns:
            List of ComplexityAnalysis results
        """
        results = []

        for i, query in enumerate(queries):
            awareness_ctx = None
            if awareness_contexts and i < len(awareness_contexts):
                awareness_ctx = awareness_contexts[i]

            result = self.analyze(query, awareness_ctx)
            results.append(result)

        return results

    def get_statistics(self, analyses: List[ComplexityAnalysis]) -> Dict[str, Any]:
        """
        Get statistics from multiple analyses.

        Args:
            analyses: List of ComplexityAnalysis results

        Returns:
            Statistics dict with averages, distributions
        """
        if not analyses:
            return {}

        # Level distribution
        level_counts = {
            ComplexityLevel.SIMPLE: 0,
            ComplexityLevel.MODERATE: 0,
            ComplexityLevel.COMPLEX: 0,
            ComplexityLevel.RESEARCH: 0
        }

        for analysis in analyses:
            level_counts[analysis.level] += 1

        # Average scores
        avg_score = sum(a.score for a in analyses) / len(analyses)
        avg_budget = sum(a.recommended_budget for a in analyses) / len(analyses)

        # Budget range
        min_budget = min(a.recommended_budget for a in analyses)
        max_budget = max(a.recommended_budget for a in analyses)

        return {
            'total_queries': len(analyses),
            'level_distribution': {
                level.value: count for level, count in level_counts.items()
            },
            'avg_complexity_score': avg_score,
            'avg_recommended_budget': int(avg_budget),
            'budget_range': {
                'min': min_budget,
                'max': max_budget
            },
            'factor_averages': self._calculate_factor_averages(analyses)
        }

    def _calculate_factor_averages(
        self,
        analyses: List[ComplexityAnalysis]
    ) -> Dict[str, float]:
        """Calculate average contribution of each factor"""
        factor_sums = {}

        for analysis in analyses:
            for factor, contribution in analysis.factors.items():
                if factor not in factor_sums:
                    factor_sums[factor] = 0.0
                factor_sums[factor] += contribution

        # Calculate averages
        factor_averages = {
            factor: total / len(analyses)
            for factor, total in factor_sums.items()
        }

        return factor_averages

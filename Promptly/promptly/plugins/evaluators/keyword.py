#!/usr/bin/env python3
"""
Keyword-based Evaluator Plugin

Evaluates output based on presence and frequency of expected keywords.
"""

from typing import Dict, Any, Optional, List
import re
from ..base import BaseEvaluator


class KeywordEvaluator(BaseEvaluator):
    """
    Evaluates text based on keyword matching.

    Supports:
    - Required keywords (must be present)
    - Optional keywords (bonus points)
    - Forbidden keywords (penalty)
    - Case-insensitive matching
    - Frequency-based scoring
    """

    def __init__(self, case_sensitive: bool = False, normalize_whitespace: bool = True):
        super().__init__(
            name="keyword",
            description="Keyword-based evaluator for text matching"
        )
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate based on keyword matching

        Expected format (if context is provided):
        context = {
            'required_keywords': ['word1', 'word2'],  # Must be present
            'optional_keywords': ['word3', 'word4'],  # Bonus points
            'forbidden_keywords': ['bad1', 'bad2'],   # Penalties
            'min_frequency': 2  # Minimum keyword occurrences
        }

        If no context, uses simple word overlap between actual and expected.

        Args:
            actual: The actual output text
            expected: The expected text or reference
            context: Optional context with keyword configuration

        Returns:
            float: Score between 0.0 and 1.0
        """
        if not actual:
            return 0.0

        # Normalize text
        actual_text = self._normalize(actual)
        expected_text = self._normalize(expected)

        # If context provided, use structured keyword evaluation
        if context and ('required_keywords' in context or 'optional_keywords' in context):
            return self._evaluate_with_keywords(actual_text, context)

        # Otherwise, use simple word overlap
        return self._simple_word_overlap(actual_text, expected_text)

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""

        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        if not self.case_sensitive:
            text = text.lower()

        return text

    def _simple_word_overlap(self, actual: str, expected: str) -> float:
        """Calculate simple word overlap score"""
        if not expected:
            return 0.0

        actual_words = set(actual.split())
        expected_words = set(expected.split())

        if not expected_words:
            return 0.0

        overlap = actual_words & expected_words
        return len(overlap) / len(expected_words)

    def _evaluate_with_keywords(self, actual_text: str, context: Dict[str, Any]) -> float:
        """Evaluate using structured keyword configuration"""
        score = 0.0
        max_score = 0.0

        # Required keywords (70% of score)
        required = context.get('required_keywords', [])
        if required:
            required_score = self._check_keywords(actual_text, required, context.get('min_frequency', 1))
            score += required_score * 0.7
            max_score += 0.7

        # Optional keywords (20% of score)
        optional = context.get('optional_keywords', [])
        if optional:
            optional_score = self._check_keywords(actual_text, optional, 1)
            score += optional_score * 0.2
            max_score += 0.2

        # Length check (10% of score)
        min_length = context.get('min_length', 0)
        max_length = context.get('max_length', float('inf'))
        if min_length <= len(actual_text) <= max_length:
            score += 0.1
        max_score += 0.1

        # Forbidden keywords (penalty)
        forbidden = context.get('forbidden_keywords', [])
        if forbidden:
            forbidden_count = sum(1 for kw in forbidden if kw.lower() in actual_text)
            if forbidden_count > 0:
                penalty = min(0.3, forbidden_count * 0.1)  # Max 30% penalty
                score = max(0.0, score - penalty)

        return score / max_score if max_score > 0 else 0.0

    def _check_keywords(self, text: str, keywords: List[str], min_frequency: int = 1) -> float:
        """Check if keywords are present with minimum frequency"""
        if not keywords:
            return 1.0

        matches = 0
        for keyword in keywords:
            kw = keyword.lower() if not self.case_sensitive else keyword
            count = text.count(kw)
            if count >= min_frequency:
                matches += 1

        return matches / len(keywords)

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed metrics including keyword matches"""
        score = self.evaluate(actual, expected, context)
        actual_text = self._normalize(actual)

        metrics = {
            "score": score,
            "actual_length": len(actual),
            "actual_word_count": len(actual.split())
        }

        if context:
            # Add keyword-specific metrics
            required = context.get('required_keywords', [])
            optional = context.get('optional_keywords', [])
            forbidden = context.get('forbidden_keywords', [])

            if required:
                metrics['required_keywords_found'] = [
                    kw for kw in required
                    if (kw.lower() if not self.case_sensitive else kw) in actual_text
                ]
                metrics['required_keywords_missing'] = [
                    kw for kw in required
                    if (kw.lower() if not self.case_sensitive else kw) not in actual_text
                ]

            if optional:
                metrics['optional_keywords_found'] = [
                    kw for kw in optional
                    if (kw.lower() if not self.case_sensitive else kw) in actual_text
                ]

            if forbidden:
                metrics['forbidden_keywords_found'] = [
                    kw for kw in forbidden
                    if (kw.lower() if not self.case_sensitive else kw) in actual_text
                ]
        else:
            # Simple word overlap metrics
            expected_text = self._normalize(expected)
            actual_words = set(actual_text.split())
            expected_words = set(expected_text.split())
            metrics['overlap_words'] = list(actual_words & expected_words)
            metrics['missing_words'] = list(expected_words - actual_words)

        return metrics

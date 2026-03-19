#!/usr/bin/env python3
"""
DSPy Metrics System for HoloLoom

Implements quantifiable evaluation metrics for DSPy optimization.
Based on principles from systematic prompt engineering.

Key Metrics:
- Functionality: Does it accomplish the task?
- Format: Does it match expected format?
- Completeness: Does it include all necessary information?
- Accuracy: Is it factually correct?
- Clarity: Is it clear and well-structured?
- Efficiency: Is it concise without sacrificing quality?
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricType(Enum):
    """Types of metrics for evaluation"""
    FUNCTIONALITY = "functionality"
    FORMAT = "format"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    EFFICIENCY = "efficiency"
    RELEVANCE = "relevance"
    SAFETY = "safety"


@dataclass
class MetricResult:
    """Result of a single metric evaluation"""
    metric_type: MetricType
    score: float  # 0.0 to 1.0
    max_score: float = 1.0
    explanation: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def percentage(self) -> float:
        """Get score as percentage"""
        return (self.score / self.max_score) * 100


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    overall_score: float
    metric_results: list[MetricResult]
    passed: bool
    feedback: str = ""

    def get_metric(self, metric_type: MetricType) -> MetricResult | None:
        """Get result for specific metric"""
        for result in self.metric_results:
            if result.metric_type == metric_type:
                return result
        return None

    def lowest_scoring_metric(self) -> MetricResult | None:
        """Get the lowest scoring metric"""
        if not self.metric_results:
            return None
        return min(self.metric_results, key=lambda r: r.score / r.max_score)


class MetricsEvaluator:
    """Evaluates outputs using defined metrics"""

    def __init__(
        self,
        metrics: list[MetricType],
        threshold: float = 0.7
    ):
        """
        Initialize evaluator.

        Args:
            metrics: List of metrics to evaluate
            threshold: Minimum score to pass (0.0 to 1.0)
        """
        self.metrics = metrics
        self.threshold = threshold

        # Map metrics to evaluation functions
        self.metric_functions: dict[MetricType, Callable] = {
            MetricType.FUNCTIONALITY: self.evaluate_functionality,
            MetricType.FORMAT: self.evaluate_format,
            MetricType.COMPLETENESS: self.evaluate_completeness,
            MetricType.ACCURACY: self.evaluate_accuracy,
            MetricType.CLARITY: self.evaluate_clarity,
            MetricType.EFFICIENCY: self.evaluate_efficiency,
            MetricType.RELEVANCE: self.evaluate_relevance,
            MetricType.SAFETY: self.evaluate_safety
        }

    def evaluate(
        self,
        example: Any,
        prediction: Any,
        context: dict[str, Any] | None = None
    ) -> EvaluationResult:
        """
        Evaluate a prediction against expected output.

        Args:
            example: Expected output (dspy.Example)
            prediction: Actual output from program
            context: Optional additional context

        Returns:
            EvaluationResult with scores
        """
        metric_results = []

        for metric_type in self.metrics:
            if metric_type in self.metric_functions:
                result = self.metric_functions[metric_type](example, prediction, context)
                metric_results.append(result)

        # Calculate overall score (weighted average)
        if metric_results:
            overall = sum(r.score / r.max_score for r in metric_results) / len(metric_results)
        else:
            overall = 0.0

        passed = overall >= self.threshold

        # Generate feedback
        feedback = self._generate_feedback(metric_results, overall, passed)

        return EvaluationResult(
            overall_score=overall,
            metric_results=metric_results,
            passed=passed,
            feedback=feedback
        )

    def evaluate_functionality(
        self,
        example: Any,
        prediction: Any,
        context: dict | None = None
    ) -> MetricResult:
        """Evaluate if output accomplishes the task"""
        # Check if prediction has expected fields
        expected_fields = []
        if hasattr(example, '__dict__'):
            expected_fields = list(vars(example).keys())

        score = 0.0
        details = {}

        if expected_fields:
            # Check field presence
            present = sum(1 for field in expected_fields if hasattr(prediction, field))
            score = present / len(expected_fields)
            details["expected_fields"] = expected_fields
            details["present_fields"] = present

            explanation = f"Found {present}/{len(expected_fields)} expected fields"
        else:
            # Fallback: just check if prediction exists and is non-empty
            if prediction and (
                (isinstance(prediction, str) and prediction.strip()) or
                (hasattr(prediction, '__dict__') and vars(prediction))
            ):
                score = 1.0
                explanation = "Output generated successfully"
            else:
                score = 0.0
                explanation = "No valid output generated"

        return MetricResult(
            metric_type=MetricType.FUNCTIONALITY,
            score=score,
            explanation=explanation,
            details=details
        )

    def evaluate_format(
        self,
        example: Any,
        prediction: Any,
        context: dict | None = None
    ) -> MetricResult:
        """Evaluate if output matches expected format"""
        score = 0.0
        details = {}

        # Check type match
        if type(example) == type(prediction):
            score += 0.5
            details["type_match"] = True
        else:
            details["type_match"] = False

        # Check structure match (for objects with attributes)
        if hasattr(example, '__dict__') and hasattr(prediction, '__dict__'):
            example_attrs = set(vars(example).keys())
            prediction_attrs = set(vars(prediction).keys())

            matching = len(example_attrs & prediction_attrs)
            total = len(example_attrs)

            if total > 0:
                structure_score = matching / total
                score += 0.5 * structure_score
                details["attribute_match"] = f"{matching}/{total}"
            else:
                score += 0.5
        else:
            # For strings, check basic formatting
            if isinstance(prediction, str) and isinstance(example, str):
                # Check if similar structure (paragraphs, lists, etc.)
                example_lines = example.strip().split('\n')
                prediction_lines = prediction.strip().split('\n')

                if abs(len(example_lines) - len(prediction_lines)) <= 2:
                    score += 0.5
                    details["line_count_similar"] = True
                else:
                    details["line_count_similar"] = False

        explanation = f"Format match: {score:.1%}"
        if not details.get("type_match"):
            explanation += f" (type mismatch: expected {type(example).__name__}, got {type(prediction).__name__})"

        return MetricResult(
            metric_type=MetricType.FORMAT,
            score=score,
            explanation=explanation,
            details=details
        )

    def evaluate_completeness(
        self,
        example: Any,
        prediction: Any,
        context: dict | None = None
    ) -> MetricResult:
        """Evaluate if output includes all necessary information"""
        score = 0.0
        details = {}

        # For text outputs, check key terms coverage
        if isinstance(example, str) and isinstance(prediction, str):
            # Extract key terms (words >4 chars, excluding common words)
            common_words = {'this', 'that', 'with', 'from', 'have', 'what', 'when', 'where', 'which'}

            example_terms = set(
                word.lower() for word in re.findall(r'\b\w{5,}\b', example)
                if word.lower() not in common_words
            )

            prediction_terms = set(
                word.lower() for word in re.findall(r'\b\w{5,}\b', prediction)
                if word.lower() not in common_words
            )

            if example_terms:
                covered = len(example_terms & prediction_terms)
                score = covered / len(example_terms)
                details["key_terms_covered"] = f"{covered}/{len(example_terms)}"
                details["missing_terms"] = list(example_terms - prediction_terms)[:5]
                explanation = f"Covered {covered}/{len(example_terms)} key terms"
            else:
                score = 1.0
                explanation = "No key terms to check"

        # For objects, check attribute completeness
        elif hasattr(example, '__dict__') and hasattr(prediction, '__dict__'):
            example_attrs = vars(example)
            prediction_attrs = vars(prediction)

            complete = sum(
                1 for attr in example_attrs
                if attr in prediction_attrs and prediction_attrs[attr]
            )

            score = complete / len(example_attrs) if example_attrs else 1.0
            details["complete_attributes"] = f"{complete}/{len(example_attrs)}"
            explanation = f"{complete}/{len(example_attrs)} attributes complete"

        else:
            score = 1.0
            explanation = "Unable to evaluate completeness for this type"

        return MetricResult(
            metric_type=MetricType.COMPLETENESS,
            score=score,
            explanation=explanation,
            details=details
        )

    def evaluate_accuracy(
        self,
        example: Any,
        prediction: Any,
        context: dict | None = None
    ) -> MetricResult:
        """Evaluate factual accuracy"""
        # This is a placeholder - real accuracy checking would need
        # fact-checking against a knowledge base or ground truth
        score = 0.8  # Default optimistic score

        # Basic checks we can do
        details = {}

        if isinstance(example, str) and isinstance(prediction, str):
            # Check if prediction contradicts example
            example_lower = example.lower()
            prediction_lower = prediction.lower()

            # Look for negation patterns
            negation_patterns = [
                (r'\bnot\b', r'\bis\b'),
                (r'\bno\b', r'\byes\b'),
                (r'\bfalse\b', r'\btrue\b')
            ]

            contradictions = 0
            for neg_pattern, pos_pattern in negation_patterns:
                if re.search(neg_pattern, example_lower) and re.search(pos_pattern, prediction_lower):
                    contradictions += 1
                if re.search(pos_pattern, example_lower) and re.search(neg_pattern, prediction_lower):
                    contradictions += 1

            if contradictions > 0:
                score = max(0.0, 1.0 - (contradictions * 0.2))
                details["contradictions"] = contradictions
                explanation = f"Detected {contradictions} potential contradictions"
            else:
                explanation = "No obvious contradictions detected"

        else:
            explanation = "Accuracy check not applicable for this type"

        return MetricResult(
            metric_type=MetricType.ACCURACY,
            score=score,
            explanation=explanation,
            details=details
        )

    def evaluate_clarity(
        self,
        example: Any,
        prediction: Any,
        context: dict | None = None
    ) -> MetricResult:
        """Evaluate clarity and readability"""
        score = 0.0
        details = {}

        if isinstance(prediction, str):
            text = prediction.strip()

            # Check for clear structure
            has_paragraphs = '\n\n' in text
            has_lists = bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE))
            has_numbers = bool(re.search(r'^\s*\d+\.', text, re.MULTILINE))

            structure_score = sum([has_paragraphs, has_lists, has_numbers]) / 3
            details["structure_elements"] = {
                "paragraphs": has_paragraphs,
                "lists": has_lists,
                "numbered_lists": has_numbers
            }

            # Check sentence length (readable if average <25 words)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if sentences:
                words_per_sentence = [len(s.split()) for s in sentences]
                avg_words = sum(words_per_sentence) / len(words_per_sentence)
                length_score = 1.0 if avg_words <= 25 else max(0.0, 1.0 - (avg_words - 25) / 25)
                details["avg_sentence_length"] = f"{avg_words:.1f} words"
            else:
                length_score = 0.5

            score = (structure_score * 0.5) + (length_score * 0.5)
            explanation = f"Clarity score: {score:.1%} (structure + readability)"

        else:
            score = 0.5
            explanation = "Clarity check not applicable for non-text"

        return MetricResult(
            metric_type=MetricType.CLARITY,
            score=score,
            explanation=explanation,
            details=details
        )

    def evaluate_efficiency(
        self,
        example: Any,
        prediction: Any,
        context: dict | None = None
    ) -> MetricResult:
        """Evaluate if output is concise without sacrificing quality"""
        score = 0.0
        details = {}

        if isinstance(example, str) and isinstance(prediction, str):
            example_len = len(example.split())
            prediction_len = len(prediction.split())

            # Ideal: within 80-120% of example length
            ratio = prediction_len / example_len if example_len > 0 else 1.0

            if 0.8 <= ratio <= 1.2:
                score = 1.0
                explanation = "Optimal length (±20% of example)"
            elif ratio < 0.8:
                score = max(0.0, 1.0 - (0.8 - ratio))
                explanation = f"Too concise ({ratio:.0%} of example length)"
            else:
                score = max(0.0, 1.0 - (ratio - 1.2) / 2)
                explanation = f"Too verbose ({ratio:.0%} of example length)"

            details["length_ratio"] = f"{ratio:.1%}"
            details["example_words"] = example_len
            details["prediction_words"] = prediction_len

        else:
            score = 0.5
            explanation = "Efficiency check not applicable"

        return MetricResult(
            metric_type=MetricType.EFFICIENCY,
            score=score,
            explanation=explanation,
            details=details
        )

    def evaluate_relevance(
        self,
        example: Any,
        prediction: Any,
        context: dict | None = None
    ) -> MetricResult:
        """Evaluate if output is relevant to the task"""
        # Similar to completeness but focused on staying on-topic
        score = 0.8  # Default
        explanation = "Relevance assumed (no context to check)"

        return MetricResult(
            metric_type=MetricType.RELEVANCE,
            score=score,
            explanation=explanation
        )

    def evaluate_safety(
        self,
        example: Any,
        prediction: Any,
        context: dict | None = None
    ) -> MetricResult:
        """Evaluate if output is safe (no harmful content)"""
        score = 1.0  # Default to safe
        details = {}

        if isinstance(prediction, str):
            # Check for potentially unsafe patterns
            unsafe_patterns = [
                r'\bpassword\b.*=',
                r'\bapi[_-]?key\b.*=',
                r'\bsecret\b.*=',
                r'<script',
                r'eval\(',
                r'exec\('
            ]

            issues = []
            for pattern in unsafe_patterns:
                if re.search(pattern, prediction, re.IGNORECASE):
                    issues.append(pattern)

            if issues:
                score = max(0.0, 1.0 - len(issues) * 0.2)
                details["safety_issues"] = issues
                explanation = f"Detected {len(issues)} potential safety issues"
            else:
                explanation = "No safety issues detected"

        else:
            explanation = "Safety check not applicable"

        return MetricResult(
            metric_type=MetricType.SAFETY,
            score=score,
            explanation=explanation,
            details=details
        )

    def _generate_feedback(
        self,
        metric_results: list[MetricResult],
        overall_score: float,
        passed: bool
    ) -> str:
        """Generate human-readable feedback"""
        feedback_parts = []

        if passed:
            feedback_parts.append(f"✅ PASSED (Overall: {overall_score:.1%})")
        else:
            feedback_parts.append(f"❌ FAILED (Overall: {overall_score:.1%}, Threshold: {self.threshold:.1%})")

        feedback_parts.append("\nMetric Breakdown:")

        for result in metric_results:
            percentage = result.percentage()
            status = "✓" if percentage >= 70 else "✗"
            feedback_parts.append(f"  {status} {result.metric_type.value.title()}: {percentage:.0f}%")
            if result.explanation:
                feedback_parts.append(f"     └─ {result.explanation}")

        # Suggest improvement
        lowest = min(metric_results, key=lambda r: r.score / r.max_score) if metric_results else None
        if lowest and (lowest.score / lowest.max_score) < 0.7:
            feedback_parts.append(f"\n💡 Focus on improving: {lowest.metric_type.value.title()}")

        return "\n".join(feedback_parts)


# Convenience function for DSPy integration
def create_hololoom_metric(
    metric_types: list[MetricType] = None,
    threshold: float = 0.7
) -> Callable:
    """
    Create a metric function for DSPy optimization.

    Args:
        metric_types: List of metrics to evaluate (default: all)
        threshold: Minimum score to pass

    Returns:
        Metric function compatible with DSPy optimizers
    """
    if metric_types is None:
        metric_types = [
            MetricType.FUNCTIONALITY,
            MetricType.FORMAT,
            MetricType.COMPLETENESS,
            MetricType.ACCURACY
        ]

    evaluator = MetricsEvaluator(metric_types, threshold)

    def metric_function(example, prediction, trace=None):
        """DSPy-compatible metric function"""
        result = evaluator.evaluate(example, prediction)
        return result.overall_score

    return metric_function


if __name__ == "__main__":
    # Demo
    print("🎯 DSPy Metrics System Demo\n")

    # Create evaluator
    evaluator = MetricsEvaluator(
        metrics=[
            MetricType.FUNCTIONALITY,
            MetricType.FORMAT,
            MetricType.COMPLETENESS,
            MetricType.CLARITY
        ],
        threshold=0.7
    )

    # Mock example and prediction
    class MockOutput:
        def __init__(self, answer, confidence):
            self.answer = answer
            self.confidence = confidence

    example = MockOutput(
        answer="Thompson Sampling is a Bayesian approach to multi-armed bandits.",
        confidence="0.95"
    )

    prediction = MockOutput(
        answer="Thompson Sampling uses Bayesian inference for bandit problems.",
        confidence="0.92"
    )

    # Evaluate
    result = evaluator.evaluate(example, prediction)

    # Show results
    print(result.feedback)

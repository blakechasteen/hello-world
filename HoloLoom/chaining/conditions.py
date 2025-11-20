"""
Common Condition Helpers for Chain Orchestration

Provides pre-built condition functions for:
- Confidence thresholds
- Source validation
- Verification checks
- Loop conditions
- Custom matchers

All conditions take a context dict and return a bool.

Author: HoloLoom Architecture Team
Date: November 2025
"""

from typing import Callable, Dict, Any, List
import re


class Conditions:
    """Common condition functions for chains."""

    @staticmethod
    def confidence_above(threshold: float) -> Callable:
        """Confidence score >= threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            confidence = ctx.get("confidence", 0.0)
            return confidence >= threshold
        return condition

    @staticmethod
    def confidence_below(threshold: float) -> Callable:
        """Confidence score < threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            confidence = ctx.get("confidence", 1.0)
            return confidence < threshold
        return condition

    @staticmethod
    def confidence_between(min_threshold: float, max_threshold: float) -> Callable:
        """Confidence between min and max (inclusive)."""
        def condition(ctx: Dict[str, Any]) -> bool:
            confidence = ctx.get("confidence", 0.0)
            return min_threshold <= confidence <= max_threshold
        return condition

    @staticmethod
    def has_sources(min_count: int = 1) -> Callable:
        """Has at least N sources."""
        def condition(ctx: Dict[str, Any]) -> bool:
            sources = ctx.get("sources", [])
            return len(sources) >= min_count
        return condition

    @staticmethod
    def sources_above(count: int) -> Callable:
        """Number of sources > count."""
        def condition(ctx: Dict[str, Any]) -> bool:
            sources = ctx.get("sources", [])
            return len(sources) > count
        return condition

    @staticmethod
    def all_checks_passed() -> Callable:
        """All verification checks passed."""
        def condition(ctx: Dict[str, Any]) -> bool:
            checks = ctx.get("verification_checks", [])
            if not checks:
                return False
            return all(check.get("passed", False) for check in checks)
        return condition

    @staticmethod
    def specific_check_passed(dimension: str) -> Callable:
        """Specific verification check passed (e.g., 'Domain', 'Sensibility')."""
        def condition(ctx: Dict[str, Any]) -> bool:
            checks = ctx.get("verification_checks", [])
            for check in checks:
                if check.get("dimension") == dimension:
                    return check.get("passed", False)
            return False
        return condition

    @staticmethod
    def verification_score_above(threshold: float) -> Callable:
        """Overall verification score >= threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            score = ctx.get("verification_score", 0.0)
            return score >= threshold
        return condition

    @staticmethod
    def response_exists() -> Callable:
        """Response exists and is not empty."""
        def condition(ctx: Dict[str, Any]) -> bool:
            response = ctx.get("response", {})
            answer = response.get("answer", "")
            return bool(answer and len(answer.strip()) > 0)
        return condition

    @staticmethod
    def response_has_content(min_length: int = 10) -> Callable:
        """Response has at least N characters."""
        def condition(ctx: Dict[str, Any]) -> bool:
            response = ctx.get("response", {})
            answer = response.get("answer", "")
            return len(answer) >= min_length
        return condition

    @staticmethod
    def error_occurred() -> Callable:
        """An error occurred during execution."""
        def condition(ctx: Dict[str, Any]) -> bool:
            error = ctx.get("error")
            return error is not None
        return condition

    @staticmethod
    def max_iterations_reached(max_iter: int) -> Callable:
        """Reached maximum iterations."""
        def condition(ctx: Dict[str, Any]) -> bool:
            iteration = ctx.get("iteration_count", 0)
            return iteration >= max_iter
        return condition

    @staticmethod
    def response_contains(text: str, case_sensitive: bool = False) -> Callable:
        """Response contains specific text."""
        def condition(ctx: Dict[str, Any]) -> bool:
            response = ctx.get("response", {})
            answer = response.get("answer", "")
            if case_sensitive:
                return text in answer
            else:
                return text.lower() in answer.lower()
        return condition

    @staticmethod
    def response_matches_pattern(pattern: str) -> Callable:
        """Response matches regex pattern."""
        def condition(ctx: Dict[str, Any]) -> bool:
            response = ctx.get("response", {})
            answer = response.get("answer", "")
            try:
                return bool(re.search(pattern, answer))
            except re.error:
                return False
        return condition

    @staticmethod
    def reasoning_mode_is(mode: str) -> Callable:
        """Reasoning mode matches specified value."""
        def condition(ctx: Dict[str, Any]) -> bool:
            current_mode = ctx.get("reasoning_mode", "")
            return current_mode == mode
        return condition

    @staticmethod
    def field_exists(field_name: str) -> Callable:
        """Field exists in context."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return field_name in ctx
        return condition

    @staticmethod
    def field_equals(field_name: str, value: Any) -> Callable:
        """Field value equals specified value."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return ctx.get(field_name) == value
        return condition

    @staticmethod
    def combine_and(*conditions: Callable) -> Callable:
        """Combine conditions with AND (all must be true)."""
        def combined(ctx: Dict[str, Any]) -> bool:
            return all(cond(ctx) for cond in conditions)
        return combined

    @staticmethod
    def combine_or(*conditions: Callable) -> Callable:
        """Combine conditions with OR (any can be true)."""
        def combined(ctx: Dict[str, Any]) -> bool:
            return any(cond(ctx) for cond in conditions)
        return combined

    @staticmethod
    def combine_not(condition: Callable) -> Callable:
        """Negate a condition."""
        def negated(ctx: Dict[str, Any]) -> bool:
            return not condition(ctx)
        return negated

    @staticmethod
    def always_true() -> Callable:
        """Always returns true."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return True
        return condition

    @staticmethod
    def always_false() -> Callable:
        """Always returns false."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return False
        return condition


# Common condition combinations
class CommonConditions:
    """Pre-built condition combinations."""

    @staticmethod
    def high_confidence() -> Callable:
        """High confidence (>= 0.75)."""
        return Conditions.confidence_above(0.75)

    @staticmethod
    def low_confidence() -> Callable:
        """Low confidence (< 0.75)."""
        return Conditions.confidence_below(0.75)

    @staticmethod
    def very_low_confidence() -> Callable:
        """Very low confidence (< 0.5)."""
        return Conditions.confidence_below(0.5)

    @staticmethod
    def verified_response() -> Callable:
        """Response is verified (all checks passed)."""
        return Conditions.combine_and(
            Conditions.response_exists(),
            Conditions.all_checks_passed(),
        )

    @staticmethod
    def needs_refinement() -> Callable:
        """Response needs refinement (low confidence + has content)."""
        return Conditions.combine_and(
            Conditions.confidence_below(0.75),
            Conditions.response_has_content(),
        )

    @staticmethod
    def ready_to_output() -> Callable:
        """Response is ready to output (not low confidence)."""
        return Conditions.confidence_above(0.5)

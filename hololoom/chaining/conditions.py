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


# =============================================================================
# Conditions for New Chain Templates (December 2025)
# =============================================================================

class FactCheckConditions:
    """Conditions specific to fact-checking chains."""

    @staticmethod
    def all_claims_verified() -> Callable:
        """All extracted claims have been verified."""
        def condition(ctx: Dict[str, Any]) -> bool:
            claims = ctx.get("claims", [])
            if not claims:
                return False
            return all(
                claim.get("verified", False) or claim.get("status") == "verified"
                for claim in claims
            )
        return condition

    @staticmethod
    def claim_count_above(count: int) -> Callable:
        """Number of claims > count."""
        def condition(ctx: Dict[str, Any]) -> bool:
            claims = ctx.get("claims", [])
            return len(claims) > count
        return condition

    @staticmethod
    def verified_claim_ratio_above(threshold: float) -> Callable:
        """Ratio of verified claims >= threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            claims = ctx.get("claims", [])
            if not claims:
                return False
            verified = sum(
                1 for c in claims
                if c.get("verified", False) or c.get("status") == "verified"
            )
            return (verified / len(claims)) >= threshold
        return condition

    @staticmethod
    def citation_coverage_above(threshold: float) -> Callable:
        """Citation coverage >= threshold (ratio of claims with citations)."""
        def condition(ctx: Dict[str, Any]) -> bool:
            claims = ctx.get("claims", [])
            if not claims:
                return False
            cited = sum(
                1 for c in claims
                if c.get("citations") or c.get("sources")
            )
            return (cited / len(claims)) >= threshold
        return condition

    @staticmethod
    def has_unverified_claims() -> Callable:
        """At least one claim is unverified."""
        def condition(ctx: Dict[str, Any]) -> bool:
            claims = ctx.get("claims", [])
            return any(
                not (c.get("verified", False) or c.get("status") == "verified")
                for c in claims
            )
        return condition


class CodeReviewConditions:
    """Conditions specific to code review chains."""

    @staticmethod
    def no_security_issues() -> Callable:
        """No security issues detected."""
        def condition(ctx: Dict[str, Any]) -> bool:
            security_issues = ctx.get("security_issues", [])
            security_score = ctx.get("security_score", 1.0)
            return len(security_issues) == 0 and security_score >= 0.9
        return condition

    @staticmethod
    def security_score_above(threshold: float) -> Callable:
        """Security score >= threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            score = ctx.get("security_score", 0.0)
            return score >= threshold
        return condition

    @staticmethod
    def style_score_above(threshold: float) -> Callable:
        """Style/lint score >= threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            score = ctx.get("style_score", 0.0)
            return score >= threshold
        return condition

    @staticmethod
    def has_critical_issues() -> Callable:
        """Has critical security or code issues."""
        def condition(ctx: Dict[str, Any]) -> bool:
            issues = ctx.get("issues", [])
            return any(
                issue.get("severity") in ["critical", "high"]
                for issue in issues
            )
        return condition

    @staticmethod
    def all_tests_passing() -> Callable:
        """All code tests are passing."""
        def condition(ctx: Dict[str, Any]) -> bool:
            tests = ctx.get("test_results", {})
            passed = tests.get("passed", 0)
            total = tests.get("total", 0)
            return total > 0 and passed == total
        return condition

    @staticmethod
    def code_complexity_below(threshold: float) -> Callable:
        """Cyclomatic complexity below threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            complexity = ctx.get("complexity", float("inf"))
            return complexity < threshold
        return condition


class SafetyConditions:
    """Conditions specific to safety-gated chains."""

    @staticmethod
    def safety_score_above(threshold: float) -> Callable:
        """Safety score >= threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            score = ctx.get("safety_score", 0.0)
            return score >= threshold
        return condition

    @staticmethod
    def safety_check_passed() -> Callable:
        """Primary safety check passed."""
        def condition(ctx: Dict[str, Any]) -> bool:
            passed = ctx.get("safety_check_passed", False)
            risk_level = ctx.get("risk_level", "unknown")
            return passed and risk_level in ["low", "medium"]
        return condition

    @staticmethod
    def audit_trail_complete() -> Callable:
        """Audit trail is complete with all required entries."""
        def condition(ctx: Dict[str, Any]) -> bool:
            audit = ctx.get("audit_trail", [])
            if not audit:
                return False
            required = {"input", "decision", "output"}
            entry_types = {entry.get("type") for entry in audit}
            return required.issubset(entry_types)
        return condition

    @staticmethod
    def no_blocked_content() -> Callable:
        """No content was blocked by safety filters."""
        def condition(ctx: Dict[str, Any]) -> bool:
            blocked = ctx.get("blocked_content", [])
            return len(blocked) == 0
        return condition

    @staticmethod
    def risk_level_acceptable() -> Callable:
        """Risk level is acceptable (low or medium)."""
        def condition(ctx: Dict[str, Any]) -> bool:
            level = ctx.get("risk_level", "unknown")
            return level in ["low", "medium"]
        return condition


class HallucinationConditions:
    """Conditions specific to hallucination guard chains."""

    @staticmethod
    def all_claims_sourced() -> Callable:
        """All claims in response have sources."""
        def condition(ctx: Dict[str, Any]) -> bool:
            claims = ctx.get("claims", [])
            if not claims:
                # No claims means nothing to verify
                return True
            return all(
                claim.get("sourced", False) or
                len(claim.get("sources", [])) > 0
                for claim in claims
            )
        return condition

    @staticmethod
    def hallucination_score_below(threshold: float) -> Callable:
        """Hallucination score < threshold (lower is better)."""
        def condition(ctx: Dict[str, Any]) -> bool:
            score = ctx.get("hallucination_score", 1.0)
            return score < threshold
        return condition

    @staticmethod
    def grounding_score_above(threshold: float) -> Callable:
        """Grounding/factuality score >= threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            score = ctx.get("grounding_score", 0.0)
            return score >= threshold
        return condition

    @staticmethod
    def has_unsourced_claims() -> Callable:
        """At least one claim lacks sources."""
        def condition(ctx: Dict[str, Any]) -> bool:
            claims = ctx.get("claims", [])
            return any(
                not claim.get("sourced", False) and
                len(claim.get("sources", [])) == 0
                for claim in claims
            )
        return condition


class RAGConditions:
    """Conditions specific to RAG chains."""

    @staticmethod
    def retrieval_quality_above(threshold: float) -> Callable:
        """Retrieval quality score >= threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            score = ctx.get("retrieval_quality", 0.0)
            return score >= threshold
        return condition

    @staticmethod
    def has_relevant_documents(min_count: int = 1) -> Callable:
        """Has at least N relevant documents retrieved."""
        def condition(ctx: Dict[str, Any]) -> bool:
            docs = ctx.get("retrieved_documents", [])
            relevant = [d for d in docs if d.get("relevance", 0) >= 0.5]
            return len(relevant) >= min_count
        return condition

    @staticmethod
    def rerank_improved() -> Callable:
        """Reranking improved document order."""
        def condition(ctx: Dict[str, Any]) -> bool:
            original = ctx.get("original_ranking_score", 0.0)
            reranked = ctx.get("reranked_score", 0.0)
            return reranked > original
        return condition

    @staticmethod
    def all_sources_cited() -> Callable:
        """All retrieved sources are cited in response."""
        def condition(ctx: Dict[str, Any]) -> bool:
            retrieved = ctx.get("retrieved_documents", [])
            citations = ctx.get("citations", [])
            if not retrieved:
                return True
            retrieved_ids = {d.get("id") for d in retrieved}
            cited_ids = {c.get("source_id") for c in citations}
            return retrieved_ids.issubset(cited_ids)
        return condition


class MemoryConditions:
    """Conditions specific to memory-augmented chains."""

    @staticmethod
    def memory_recall_successful() -> Callable:
        """Successfully recalled relevant memories."""
        def condition(ctx: Dict[str, Any]) -> bool:
            memories = ctx.get("recalled_memories", [])
            return len(memories) > 0
        return condition

    @staticmethod
    def memory_confidence_above(threshold: float) -> Callable:
        """Average memory confidence >= threshold."""
        def condition(ctx: Dict[str, Any]) -> bool:
            memories = ctx.get("recalled_memories", [])
            if not memories:
                return False
            avg_conf = sum(m.get("confidence", 0) for m in memories) / len(memories)
            return avg_conf >= threshold
        return condition

    @staticmethod
    def new_knowledge_detected() -> Callable:
        """New knowledge was detected to store."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return ctx.get("new_knowledge_detected", False)
        return condition

    @staticmethod
    def memory_stored_successfully() -> Callable:
        """Memory was stored successfully."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return ctx.get("memory_stored", False)
        return condition


class AgentConditions:
    """Conditions specific to agent planning chains."""

    @staticmethod
    def plan_valid() -> Callable:
        """Generated plan is valid and executable."""
        def condition(ctx: Dict[str, Any]) -> bool:
            plan = ctx.get("plan", {})
            steps = plan.get("steps", [])
            return (
                len(steps) > 0 and
                plan.get("valid", False)
            )
        return condition

    @staticmethod
    def all_steps_complete() -> Callable:
        """All plan steps have been executed."""
        def condition(ctx: Dict[str, Any]) -> bool:
            plan = ctx.get("plan", {})
            steps = plan.get("steps", [])
            if not steps:
                return False
            return all(
                step.get("status") == "complete"
                for step in steps
            )
        return condition

    @staticmethod
    def step_count_below(max_steps: int) -> Callable:
        """Number of plan steps < max_steps."""
        def condition(ctx: Dict[str, Any]) -> bool:
            plan = ctx.get("plan", {})
            steps = plan.get("steps", [])
            return len(steps) < max_steps
        return condition

    @staticmethod
    def reflection_suggests_retry() -> Callable:
        """Reflection step suggests retrying."""
        def condition(ctx: Dict[str, Any]) -> bool:
            reflection = ctx.get("reflection", {})
            return reflection.get("suggest_retry", False)
        return condition

    @staticmethod
    def goal_achieved() -> Callable:
        """Primary goal has been achieved."""
        def condition(ctx: Dict[str, Any]) -> bool:
            return ctx.get("goal_achieved", False)
        return condition

    @staticmethod
    def subtasks_remaining() -> Callable:
        """There are remaining subtasks to complete."""
        def condition(ctx: Dict[str, Any]) -> bool:
            subtasks = ctx.get("subtasks", [])
            return any(
                task.get("status") != "complete"
                for task in subtasks
            )
        return condition

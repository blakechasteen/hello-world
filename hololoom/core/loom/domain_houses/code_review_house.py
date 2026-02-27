"""
Code Review House - Multi-Perspective Code Analysis
====================================================

A pre-configured WeaveHouse optimized for code review tasks.

Combines 4 core looms with code-specific configurations:
- RECALL: Find similar code patterns, past reviews
- REASON: Trace logic flow, identify potential bugs
- REFLECT: Verify correctness, check consistency
- REFUSE: Find edge cases, security vulnerabilities

Date: December 2025
Author: HoloLoom Architecture Team
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hololoom.core.loom.weave_house import WeaveHouse, WeaveResult
from hololoom.core.loom.consensus import LoomConsensus, create_loom_consensus
from hololoom.core.loom.core_looms.recall_loom import RecallLoom
from hololoom.core.loom.core_looms.reason_loom import ReasonLoom
from hololoom.core.loom.core_looms.reflect_loom import ReflectLoom
from hololoom.core.loom.core_looms.refuse_loom import RefuseLoom


@dataclass
class CodeReviewConfig:
    """Configuration for code review house."""

    # Recall settings (finding similar code)
    recall_k: int = 15
    recall_temperature: float = 0.3

    # Reason settings (logic analysis)
    max_inference_depth: int = 7  # Deeper for complex code
    require_explicit_links: bool = True

    # Reflect settings (verification)
    verification_passes: int = 2
    consistency_threshold: float = 0.85

    # Refuse settings (adversarial probing)
    attack_intensity: float = 0.7  # Higher for security review
    coverage_target: float = 0.8

    # Consensus settings
    exploration_depth: int = 2
    tension_threshold: float = 0.25  # Lower = more exploration

    # Code-specific
    check_security: bool = True
    check_performance: bool = True
    check_maintainability: bool = True


class CodeReviewHouse(WeaveHouse):
    """
    A WeaveHouse specialized for code review.

    Provides multi-perspective code analysis:
    - Pattern matching (similar code, past reviews)
    - Logic verification (bug detection, flow analysis)
    - Consistency checking (style, conventions)
    - Security probing (vulnerabilities, edge cases)

    Example:
        ```python
        async with create_code_review_house() as house:
            result = await house.review(code_snippet)
            print(result.issues)
            print(result.suggestions)
        ```
    """

    def __init__(
        self,
        config: Optional[CodeReviewConfig] = None,
        **kwargs
    ):
        """
        Initialize the Code Review House.

        Args:
            config: Optional configuration for the house
            **kwargs: Arguments passed to WeaveHouse
        """
        self.code_config = config or CodeReviewConfig()

        # Create specialized looms for code review
        looms = self._create_code_looms()

        # Create consensus with code-specific settings
        consensus = create_loom_consensus(
            exploration_enabled=True,
            agreement_threshold=0.7
        )

        super().__init__(
            looms=looms,
            consensus=consensus,
            exploration_depth=self.code_config.exploration_depth,
            tension_threshold=self.code_config.tension_threshold,
            **kwargs
        )

    def _create_code_looms(self) -> List:
        """Create looms configured for code review."""
        return [
            # RECALL: Find similar code patterns
            RecallLoom(
                retrieval_k=self.code_config.recall_k,
                temperature=self.code_config.recall_temperature,
                trust_memory_weight=0.8,
                trust_inference_weight=0.2,
            ),

            # REASON: Analyze code logic
            ReasonLoom(
                max_inference_depth=self.code_config.max_inference_depth,
                require_explicit_links=self.code_config.require_explicit_links,
                min_step_confidence=0.65,
            ),

            # REFLECT: Verify correctness
            ReflectLoom(
                verification_passes=self.code_config.verification_passes,
                consistency_threshold=self.code_config.consistency_threshold,
                require_evidence=True,
            ),

            # REFUSE: Find vulnerabilities
            RefuseLoom(
                attack_intensity=self.code_config.attack_intensity,
                coverage_target=self.code_config.coverage_target,
                max_attack_depth=5,
            ),
        ]

    async def review(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        focus: Optional[List[str]] = None
    ) -> "CodeReviewResult":
        """
        Review code from multiple perspectives.

        Args:
            code: The code to review
            context: Optional context (language, file path, etc.)
            focus: Optional focus areas ["security", "performance", "style"]

        Returns:
            CodeReviewResult with issues, suggestions, and perspectives
        """
        # Build review query
        query = self._build_review_query(code, context, focus)

        # Weave through all perspectives
        result = await self.weave(query)

        # Parse into structured review
        return self._parse_review_result(result, code)

    def _build_review_query(
        self,
        code: str,
        context: Optional[Dict[str, Any]],
        focus: Optional[List[str]]
    ) -> str:
        """Build a query for code review."""
        parts = ["Review the following code:\n\n```\n", code, "\n```\n"]

        if context:
            parts.append(f"\nContext: {context}\n")

        if focus:
            parts.append(f"\nFocus on: {', '.join(focus)}\n")

        parts.append("\nProvide analysis from multiple perspectives.")

        return "".join(parts)

    def _parse_review_result(
        self,
        result: WeaveResult,
        code: str
    ) -> "CodeReviewResult":
        """Parse weave result into structured review."""
        return CodeReviewResult(
            code=code,
            summary=result.response,
            confidence=result.confidence,
            epistemic_confidence=result.epistemic_confidence,
            issues=self._extract_issues(result),
            suggestions=self._extract_suggestions(result),
            perspectives={
                fab.perspective: fab.response
                for fab in result.fabrics
            },
            tensions=result.tensions,
            metadata=result.metadata,
        )

    def _extract_issues(self, result: WeaveResult) -> List[Dict[str, Any]]:
        """Extract issues from review result."""
        issues = []

        # Extract from REFUSE loom (vulnerabilities)
        for fab in result.fabrics:
            if fab.perspective == "refuse":
                if "vulnerabilities" in fab.metadata:
                    for vuln in fab.metadata["vulnerabilities"]:
                        issues.append({
                            "type": "security",
                            "severity": vuln.get("severity", "medium"),
                            "description": vuln.get("description", ""),
                            "source": "refuse_loom",
                        })

            # Extract from REFLECT loom (contradictions)
            elif fab.perspective == "reflect":
                if "contradictions" in fab.metadata:
                    for contra in fab.metadata["contradictions"]:
                        issues.append({
                            "type": "consistency",
                            "severity": "low",
                            "description": str(contra),
                            "source": "reflect_loom",
                        })

        return issues

    def _extract_suggestions(self, result: WeaveResult) -> List[str]:
        """Extract suggestions from review result."""
        suggestions = []

        for fab in result.fabrics:
            # REASON loom often has improvement suggestions
            if fab.perspective == "reason":
                if "improvements" in fab.metadata:
                    suggestions.extend(fab.metadata["improvements"])

            # RECALL loom might have patterns to follow
            elif fab.perspective == "recall":
                if "similar_patterns" in fab.metadata:
                    for pattern in fab.metadata["similar_patterns"]:
                        suggestions.append(f"Consider pattern: {pattern}")

        return suggestions


@dataclass
class CodeReviewResult:
    """Result of a code review."""

    code: str
    summary: str
    confidence: float
    epistemic_confidence: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    perspectives: Dict[str, str] = field(default_factory=dict)
    tensions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_security_issues(self) -> bool:
        """Check if there are security issues."""
        return any(i["type"] == "security" for i in self.issues)

    @property
    def issue_count(self) -> int:
        """Total number of issues found."""
        return len(self.issues)

    def issues_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get issues by severity level."""
        return [i for i in self.issues if i.get("severity") == severity]


def create_code_review_house(
    config: Optional[CodeReviewConfig] = None,
    **kwargs
) -> CodeReviewHouse:
    """
    Factory function to create a CodeReviewHouse.

    Args:
        config: Optional configuration
        **kwargs: Additional arguments for WeaveHouse

    Returns:
        Configured CodeReviewHouse instance
    """
    return CodeReviewHouse(config=config, **kwargs)


__all__ = ['CodeReviewHouse', 'CodeReviewConfig', 'CodeReviewResult', 'create_code_review_house']

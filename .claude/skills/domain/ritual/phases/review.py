"""
Ritual REVIEW Phase Skill
=========================

Created: December 30, 2025
Purpose: Quality check after implementation, capture learnings, store decisions in memory

The REVIEW phase:
1. Reviews implementation for quality issues
2. Identifies key decisions made during implementation
3. Extracts learnings worth remembering
4. Stores decisions and learnings in HoloLoom memory

Decision Point: "Approve", "Address issues first", "Skip review"

Follows DockableSkill protocol for consistent lifecycle management.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re

from .base import (
    AbstractRitualPhase,
    AgentCapability,
    PhaseResult,
    PhaseExecutionContext,
    PhasePreflightResult,
)

# Import ritual-specific types
from ..ritual_events import RitualPhase

logger = logging.getLogger(__name__)


# ============================================================================
# Review-Specific Data Structures
# ============================================================================

@dataclass
class QualityIssue:
    """A quality issue found during review."""
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "security", "performance", "maintainability", "correctness", "style"
    description: str
    location: str = ""  # File/function where issue found
    suggestion: str = ""  # Suggested fix
    auto_fixable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion,
            "auto_fixable": self.auto_fixable,
        }


@dataclass
class ImplementationDecision:
    """A key decision made during implementation."""
    category: str  # "architecture", "algorithm", "dependency", "tradeoff", "convention"
    decision: str
    rationale: str
    alternatives_considered: List[str] = field(default_factory=list)
    impact: str = ""  # What this decision affects
    reversible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "decision": self.decision,
            "rationale": self.rationale,
            "alternatives_considered": self.alternatives_considered,
            "impact": self.impact,
            "reversible": self.reversible,
        }


@dataclass
class Learning:
    """A learning extracted from the implementation."""
    topic: str
    insight: str
    applicable_to: List[str] = field(default_factory=list)  # Future use cases
    source: str = ""  # Where this learning came from
    confidence: float = 0.5  # How confident in this insight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "insight": self.insight,
            "applicable_to": self.applicable_to,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class ReviewResult:
    """Complete result of the review phase."""
    quality_assessment: str
    issues: List[QualityIssue]
    decisions: List[ImplementationDecision]
    learnings: List[Learning]
    overall_quality_score: float  # 0.0-1.0
    ready_for_approval: bool
    blocking_issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == "critical" for issue in self.issues)

    @property
    def has_high_issues(self) -> bool:
        return any(issue.severity == "high" for issue in self.issues)

    @property
    def issue_count_by_severity(self) -> Dict[str, int]:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in self.issues:
            if issue.severity in counts:
                counts[issue.severity] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_assessment": self.quality_assessment,
            "issues": [i.to_dict() for i in self.issues],
            "decisions": [d.to_dict() for d in self.decisions],
            "learnings": [l.to_dict() for l in self.learnings],
            "overall_quality_score": self.overall_quality_score,
            "ready_for_approval": self.ready_for_approval,
            "blocking_issues": [i.to_dict() for i in self.blocking_issues],
            "recommendations": self.recommendations,
            "issue_counts": self.issue_count_by_severity,
        }


# ============================================================================
# Review Phase Implementation
# ============================================================================

class ReviewPhase(AbstractRitualPhase):
    """
    REVIEW phase skill implementation.

    Purpose: Quality check after implementation. Captures learnings,
    stores key decisions in memory for future recall.

    Capabilities:
    - QUALITY_ASSURANCE: Review implementation quality
    - MEMORY_STORAGE: Store decisions and learnings

    Decision Point:
    After review, user decides:
    - "Approve": Proceed to REFLECT phase
    - "Address issues first": Return to fix issues
    - "Skip review": Skip and proceed (not recommended)
    """

    def __init__(self):
        super().__init__(
            phase=RitualPhase.REVIEW,
            name="ritual-review",
            version="1.0.0",
            description="Quality check after implementation, capture learnings and decisions",
            capabilities=[
                AgentCapability.QUALITY_ASSURANCE,
                AgentCapability.MEMORY_STORAGE,
            ],
        )

        # Quality check categories to evaluate
        self._quality_categories = [
            "correctness",
            "security",
            "performance",
            "maintainability",
            "style",
            "documentation",
            "testing",
        ]

        # Severity thresholds for approval
        self._auto_approve_threshold = 0.8  # Quality score above this = auto-approve suggestion
        self._blocking_severities = {"critical", "high"}

    # ========================================================================
    # Phase-Specific Preflight
    # ========================================================================

    async def _phase_specific_preflight(
        self,
        context: PhaseExecutionContext
    ) -> Dict[str, Any]:
        """
        REVIEW-specific preflight checks.

        Validates:
        1. Implementation summary exists (from IMPLEMENT phase handoff)
        2. Feature name is specified
        3. Memory system ready for storing learnings

        Returns preflight result dict with checks, failures, warnings, memory_context.
        """
        checks = []
        failures = []
        warnings = []
        memory_context = {}

        # Check 1: Feature name available
        feature_name = (
            context.feature_name or
            context.ritual_context.feature_name
        )
        if feature_name:
            checks.append("feature_name_available")
            memory_context["feature_name"] = feature_name
        else:
            warnings.append("No feature name specified - review will be generic")

        # Check 2: Handoff context from previous phases
        if context.handoff_context:
            checks.append("handoff_context_present")
            memory_context["handoff_tokens"] = context.handoff_tokens
            memory_context["handoff_preview"] = context.handoff_context[:200] + "..."
        else:
            warnings.append("No handoff context from previous phases")

        # Check 3: Memory system available for storing learnings
        if context.warp_space is not None:
            checks.append("memory_storage_available")
        else:
            warnings.append("Memory storage unavailable - learnings won't persist")

        # Pre-fetch any existing decisions/learnings for this feature
        if feature_name and context.warp_space:
            try:
                existing = await self._fetch_existing_knowledge(
                    feature_name, context
                )
                memory_context["existing_decisions"] = existing.get("decisions", [])
                memory_context["existing_learnings"] = existing.get("learnings", [])
                checks.append("existing_knowledge_retrieved")
            except Exception as e:
                warnings.append(f"Could not retrieve existing knowledge: {e}")

        return {
            "checks": checks,
            "failures": failures,
            "warnings": warnings,
            "memory_context": memory_context,
        }

    async def _fetch_existing_knowledge(
        self,
        feature_name: str,
        context: PhaseExecutionContext
    ) -> Dict[str, List]:
        """Fetch existing decisions and learnings for this feature."""
        existing = {"decisions": [], "learnings": []}

        # Query for past decisions
        decision_memories = await self.recall_memories(
            query=f"decisions made for {feature_name}",
            context=context,
            strategy="semantic",
            limit=5
        )
        existing["decisions"] = decision_memories

        # Query for past learnings
        learning_memories = await self.recall_memories(
            query=f"learnings from {feature_name}",
            context=context,
            strategy="semantic",
            limit=5
        )
        existing["learnings"] = learning_memories

        return existing

    def _estimate_duration(self, context: PhaseExecutionContext) -> float:
        """Estimate execution duration based on complexity."""
        base_duration = 8000.0  # 8 seconds base

        # Adjust based on handoff context size
        if context.handoff_tokens:
            # More context = more review time
            token_factor = min(context.handoff_tokens / 1000, 3.0)
            base_duration += token_factor * 2000

        return base_duration

    # ========================================================================
    # Phase-Specific Execution
    # ========================================================================

    async def _phase_specific_execute(
        self,
        parameters: Dict[str, Any],
        context: PhaseExecutionContext
    ) -> PhaseResult:
        """
        Execute the REVIEW phase.

        Steps:
        1. Analyze implementation for quality issues
        2. Extract key decisions made
        3. Identify learnings worth remembering
        4. Store in memory if quality threshold met
        5. Present decision point to user
        """
        feature_name = parameters.get("_feature_name", "unknown feature")
        handoff_context = parameters.get("_handoff_context", "")
        memory_context = parameters.get("_memory_context", {})

        # Implementation summary (from user or handoff)
        implementation_summary = parameters.get(
            "implementation_summary",
            handoff_context or "No implementation summary provided"
        )

        # Code changes if provided
        code_changes = parameters.get("code_changes", "")

        memories_stored = []
        memories_retrieved = []

        try:
            # Step 1: Perform quality assessment
            issues = await self._analyze_quality(
                implementation_summary,
                code_changes,
                feature_name,
                context
            )

            # Step 2: Extract key decisions
            decisions = await self._extract_decisions(
                implementation_summary,
                handoff_context,
                feature_name,
                context
            )

            # Step 3: Identify learnings
            learnings = await self._extract_learnings(
                implementation_summary,
                issues,
                decisions,
                feature_name,
                context
            )

            # Step 4: Calculate overall quality score
            quality_score = self._calculate_quality_score(issues)

            # Step 5: Determine blocking issues
            blocking_issues = [
                issue for issue in issues
                if issue.severity in self._blocking_severities
            ]

            # Step 6: Build review result
            review_result = ReviewResult(
                quality_assessment=self._generate_quality_summary(issues, quality_score),
                issues=issues,
                decisions=decisions,
                learnings=learnings,
                overall_quality_score=quality_score,
                ready_for_approval=len(blocking_issues) == 0,
                blocking_issues=blocking_issues,
                recommendations=self._generate_recommendations(issues, decisions),
            )

            # Step 7: Store decisions and learnings in memory
            if quality_score >= 0.5:  # Only store if reasonable quality
                stored_ids = await self._store_knowledge(
                    review_result, feature_name, context
                )
                memories_stored.extend(stored_ids)

            # Step 8: Track retrieved memories
            existing_decisions = memory_context.get("existing_decisions", [])
            existing_learnings = memory_context.get("existing_learnings", [])
            memories_retrieved.extend([
                m.get("id", str(i)) for i, m in enumerate(existing_decisions)
            ])
            memories_retrieved.extend([
                m.get("id", str(i)) for i, m in enumerate(existing_learnings)
            ])

            # Step 9: Determine decision suggestion
            decision_needed, suggested_decision, decision_reasoning = (
                self._determine_decision(review_result)
            )

            return PhaseResult(
                success=True,
                output=review_result.to_dict(),
                confidence=quality_score,
                memories_stored=memories_stored,
                memories_retrieved=memories_retrieved,
                decision_needed=decision_needed,
                decision_options=self.get_decision_options(),
                suggested_decision=suggested_decision,
                decision_reasoning=decision_reasoning,
                metadata={
                    "feature_name": feature_name,
                    "issue_count": len(issues),
                    "decision_count": len(decisions),
                    "learning_count": len(learnings),
                    "blocking_issue_count": len(blocking_issues),
                    "quality_score": quality_score,
                },
            )

        except Exception as e:
            logger.error(f"REVIEW phase execution failed: {e}")
            return PhaseResult(
                success=False,
                output=None,
                confidence=0.0,
                errors=[str(e)],
                decision_needed=True,
                decision_options=self.get_decision_options(),
                suggested_decision="Address issues first",
                decision_reasoning=f"Review failed with error: {e}",
            )

    # ========================================================================
    # Quality Analysis
    # ========================================================================

    async def _analyze_quality(
        self,
        implementation_summary: str,
        code_changes: str,
        feature_name: str,
        context: PhaseExecutionContext
    ) -> List[QualityIssue]:
        """Analyze implementation for quality issues."""
        issues = []

        combined_content = f"{implementation_summary}\n{code_changes}"

        # Security checks
        security_issues = self._check_security(combined_content)
        issues.extend(security_issues)

        # Performance checks
        performance_issues = self._check_performance(combined_content)
        issues.extend(performance_issues)

        # Maintainability checks
        maintainability_issues = self._check_maintainability(combined_content)
        issues.extend(maintainability_issues)

        # Documentation checks
        doc_issues = self._check_documentation(combined_content, feature_name)
        issues.extend(doc_issues)

        # Testing checks
        testing_issues = self._check_testing(combined_content)
        issues.extend(testing_issues)

        return issues

    def _check_security(self, content: str) -> List[QualityIssue]:
        """Check for common security issues."""
        issues = []
        content_lower = content.lower()

        # Hardcoded secrets
        secret_patterns = [
            (r'api[_-]?key\s*[=:]\s*["\'][^"\']+["\']', "Potential hardcoded API key"),
            (r'password\s*[=:]\s*["\'][^"\']+["\']', "Potential hardcoded password"),
            (r'secret\s*[=:]\s*["\'][^"\']+["\']', "Potential hardcoded secret"),
        ]
        for pattern, desc in secret_patterns:
            if re.search(pattern, content_lower):
                issues.append(QualityIssue(
                    severity="critical",
                    category="security",
                    description=desc,
                    suggestion="Use environment variables or secure credential management",
                ))

        # SQL injection risk
        if "execute(" in content_lower and ("format" in content_lower or "%" in content):
            issues.append(QualityIssue(
                severity="high",
                category="security",
                description="Potential SQL injection risk - string formatting in SQL",
                suggestion="Use parameterized queries instead",
            ))

        # Command injection risk
        if "os.system" in content_lower or "subprocess.call" in content_lower:
            issues.append(QualityIssue(
                severity="medium",
                category="security",
                description="Shell command execution detected",
                suggestion="Use subprocess with shell=False and explicit arguments",
            ))

        return issues

    def _check_performance(self, content: str) -> List[QualityIssue]:
        """Check for common performance issues."""
        issues = []
        content_lower = content.lower()

        # N+1 query patterns
        if "for" in content_lower and ("query" in content_lower or ".get(" in content_lower):
            issues.append(QualityIssue(
                severity="medium",
                category="performance",
                description="Potential N+1 query pattern detected",
                suggestion="Consider batch fetching or eager loading",
            ))

        # Large list comprehensions with nested loops
        if content.count("for") > 2 and "[" in content and "]" in content:
            issues.append(QualityIssue(
                severity="low",
                category="performance",
                description="Complex nested comprehension detected",
                suggestion="Consider breaking into separate steps for clarity",
            ))

        return issues

    def _check_maintainability(self, content: str) -> List[QualityIssue]:
        """Check for maintainability issues."""
        issues = []

        # Long functions (heuristic: many lines mentioned)
        if content.count("\n") > 100:
            issues.append(QualityIssue(
                severity="low",
                category="maintainability",
                description="Implementation appears to be lengthy",
                suggestion="Consider breaking into smaller, focused functions",
            ))

        # Magic numbers
        magic_number_pattern = r'\b(?<!["\'])\d{3,}(?!["\'])\b'
        if re.search(magic_number_pattern, content):
            issues.append(QualityIssue(
                severity="low",
                category="maintainability",
                description="Magic numbers detected",
                suggestion="Extract to named constants for clarity",
                auto_fixable=True,
            ))

        # TODO/FIXME comments
        if "todo" in content.lower() or "fixme" in content.lower():
            issues.append(QualityIssue(
                severity="low",
                category="maintainability",
                description="TODO/FIXME comments found - incomplete work",
                suggestion="Address or create tracked issues for outstanding items",
            ))

        return issues

    def _check_documentation(self, content: str, feature_name: str) -> List[QualityIssue]:
        """Check documentation quality."""
        issues = []
        content_lower = content.lower()

        # Missing docstrings
        if "def " in content_lower and '"""' not in content and "'''" not in content:
            issues.append(QualityIssue(
                severity="low",
                category="documentation",
                description="Functions may be missing docstrings",
                suggestion="Add docstrings explaining purpose and parameters",
            ))

        # No mention of feature name in docs
        if feature_name.lower() not in content_lower:
            issues.append(QualityIssue(
                severity="low",
                category="documentation",
                description=f"Feature name '{feature_name}' not documented in implementation",
                suggestion="Add clear references to feature in comments/docstrings",
            ))

        return issues

    def _check_testing(self, content: str) -> List[QualityIssue]:
        """Check testing considerations."""
        issues = []
        content_lower = content.lower()

        # No test mentions
        if "test" not in content_lower and "assert" not in content_lower:
            issues.append(QualityIssue(
                severity="medium",
                category="testing",
                description="No testing considerations mentioned",
                suggestion="Add unit tests or document test strategy",
            ))

        return issues

    def _calculate_quality_score(self, issues: List[QualityIssue]) -> float:
        """Calculate overall quality score from issues."""
        if not issues:
            return 1.0

        # Penalty per severity
        penalties = {
            "critical": 0.4,
            "high": 0.2,
            "medium": 0.1,
            "low": 0.05,
        }

        total_penalty = sum(
            penalties.get(issue.severity, 0.05)
            for issue in issues
        )

        # Cap penalty at 0.9 (minimum score is 0.1)
        total_penalty = min(total_penalty, 0.9)

        return round(1.0 - total_penalty, 2)

    def _generate_quality_summary(
        self,
        issues: List[QualityIssue],
        score: float
    ) -> str:
        """Generate human-readable quality summary."""
        if score >= 0.9:
            assessment = "Excellent quality - ready for approval"
        elif score >= 0.7:
            assessment = "Good quality - minor issues to address"
        elif score >= 0.5:
            assessment = "Acceptable quality - some issues need attention"
        else:
            assessment = "Quality concerns - significant issues found"

        counts = {}
        for issue in issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1

        if counts:
            issue_summary = ", ".join(
                f"{count} {severity}" for severity, count in counts.items()
            )
            return f"{assessment}. Issues: {issue_summary}"

        return f"{assessment}. No issues found."

    # ========================================================================
    # Decision Extraction
    # ========================================================================

    async def _extract_decisions(
        self,
        implementation_summary: str,
        handoff_context: str,
        feature_name: str,
        context: PhaseExecutionContext
    ) -> List[ImplementationDecision]:
        """Extract key decisions from implementation."""
        decisions = []

        combined = f"{implementation_summary}\n{handoff_context}"
        combined_lower = combined.lower()

        # Look for decision indicators
        decision_indicators = [
            ("chose", "decision"),
            ("decided", "decision"),
            ("selected", "decision"),
            ("using", "dependency"),
            ("approach", "architecture"),
            ("pattern", "architecture"),
            ("tradeoff", "tradeoff"),
            ("trade-off", "tradeoff"),
            ("instead of", "tradeoff"),
            ("rather than", "tradeoff"),
        ]

        for indicator, category in decision_indicators:
            if indicator in combined_lower:
                # Extract sentence containing the indicator
                sentences = combined.split(".")
                for sentence in sentences:
                    if indicator in sentence.lower():
                        decisions.append(ImplementationDecision(
                            category=category,
                            decision=sentence.strip()[:200],
                            rationale="Extracted from implementation context",
                        ))
                        break

        # Architecture decisions from patterns
        arch_patterns = [
            ("factory", "Using factory pattern"),
            ("singleton", "Using singleton pattern"),
            ("observer", "Using observer pattern"),
            ("decorator", "Using decorator pattern"),
            ("protocol", "Using protocol/interface pattern"),
        ]
        for pattern, description in arch_patterns:
            if pattern in combined_lower:
                decisions.append(ImplementationDecision(
                    category="architecture",
                    decision=description,
                    rationale=f"Pattern '{pattern}' identified in implementation",
                ))

        return decisions[:10]  # Limit to top 10 decisions

    # ========================================================================
    # Learning Extraction
    # ========================================================================

    async def _extract_learnings(
        self,
        implementation_summary: str,
        issues: List[QualityIssue],
        decisions: List[ImplementationDecision],
        feature_name: str,
        context: PhaseExecutionContext
    ) -> List[Learning]:
        """Extract learnings worth remembering."""
        learnings = []

        # Learning from issues
        for issue in issues:
            if issue.severity in ("critical", "high"):
                learnings.append(Learning(
                    topic=f"{issue.category} issue",
                    insight=f"Avoid: {issue.description}. {issue.suggestion}",
                    applicable_to=[feature_name, issue.category],
                    source="quality_review",
                    confidence=0.8,
                ))

        # Learning from decisions
        for decision in decisions:
            if decision.category in ("architecture", "tradeoff"):
                learnings.append(Learning(
                    topic=decision.category,
                    insight=decision.decision,
                    applicable_to=[feature_name, decision.category],
                    source="implementation_decision",
                    confidence=0.7,
                ))

        # Look for explicit learnings in content
        learning_indicators = ["learned", "discovered", "realized", "found that"]
        for indicator in learning_indicators:
            if indicator in implementation_summary.lower():
                sentences = implementation_summary.split(".")
                for sentence in sentences:
                    if indicator in sentence.lower():
                        learnings.append(Learning(
                            topic="implementation insight",
                            insight=sentence.strip()[:200],
                            applicable_to=[feature_name],
                            source="explicit_mention",
                            confidence=0.9,
                        ))
                        break

        return learnings[:10]  # Limit to top 10 learnings

    # ========================================================================
    # Knowledge Storage
    # ========================================================================

    async def _store_knowledge(
        self,
        review_result: ReviewResult,
        feature_name: str,
        context: PhaseExecutionContext
    ) -> List[str]:
        """Store decisions and learnings in HoloLoom memory."""
        stored_ids = []

        # Store decisions
        for decision in review_result.decisions:
            content = (
                f"[Decision - {decision.category}] {feature_name}: "
                f"{decision.decision}. Rationale: {decision.rationale}"
            )
            memory_id = await self.store_memory(
                content=content,
                context=context,
                tags=["decision", decision.category, feature_name],
                importance=0.7,
            )
            if memory_id:
                stored_ids.append(memory_id)

        # Store learnings
        for learning in review_result.learnings:
            content = (
                f"[Learning - {learning.topic}] {feature_name}: "
                f"{learning.insight}"
            )
            memory_id = await self.store_memory(
                content=content,
                context=context,
                tags=["learning", learning.topic, feature_name] + learning.applicable_to,
                importance=learning.confidence,
            )
            if memory_id:
                stored_ids.append(memory_id)

        # Store overall review summary if quality is good
        if review_result.overall_quality_score >= 0.7:
            summary_content = (
                f"[Review Summary] {feature_name}: "
                f"Quality score {review_result.overall_quality_score:.0%}. "
                f"{review_result.quality_assessment}"
            )
            memory_id = await self.store_memory(
                content=summary_content,
                context=context,
                tags=["review_summary", feature_name],
                importance=0.6,
            )
            if memory_id:
                stored_ids.append(memory_id)

        return stored_ids

    # ========================================================================
    # Recommendations
    # ========================================================================

    def _generate_recommendations(
        self,
        issues: List[QualityIssue],
        decisions: List[ImplementationDecision]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Priority fixes
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            recommendations.append(
                f"🚨 Address {len(critical_issues)} critical issue(s) before proceeding"
            )

        high_issues = [i for i in issues if i.severity == "high"]
        if high_issues:
            recommendations.append(
                f"⚠️ Review {len(high_issues)} high-severity issue(s)"
            )

        # Auto-fixable suggestions
        auto_fixable = [i for i in issues if i.auto_fixable]
        if auto_fixable:
            recommendations.append(
                f"🔧 {len(auto_fixable)} issue(s) can be auto-fixed"
            )

        # Testing recommendation
        testing_issues = [i for i in issues if i.category == "testing"]
        if testing_issues:
            recommendations.append(
                "📝 Consider adding tests for the new implementation"
            )

        # Documentation recommendation
        doc_issues = [i for i in issues if i.category == "documentation"]
        if doc_issues:
            recommendations.append(
                "📚 Improve documentation for maintainability"
            )

        return recommendations

    # ========================================================================
    # Decision Determination
    # ========================================================================

    def _determine_decision(
        self,
        review_result: ReviewResult
    ) -> tuple[bool, str, str]:
        """
        Determine suggested decision based on review results.

        Returns: (decision_needed, suggested_decision, reasoning)
        """
        if review_result.has_critical_issues:
            return (
                True,
                "Address issues first",
                f"Critical issues found: {len(review_result.blocking_issues)} blocking issue(s) must be resolved"
            )

        if review_result.has_high_issues:
            return (
                True,
                "Address issues first",
                f"High-severity issues found: recommend addressing before approval"
            )

        if review_result.overall_quality_score >= self._auto_approve_threshold:
            return (
                True,
                "Approve",
                f"Quality score {review_result.overall_quality_score:.0%} meets approval threshold"
            )

        return (
            True,
            "Approve",
            f"Quality score {review_result.overall_quality_score:.0%} - minor issues can be addressed later"
        )

    # ========================================================================
    # Decision Options Override
    # ========================================================================

    def get_decision_options(self) -> List[str]:
        """Get available decision options for REVIEW phase."""
        return ["Approve", "Address issues first", "Skip review"]


# ============================================================================
# Factory Function
# ============================================================================

def create_review_phase() -> ReviewPhase:
    """Factory function to create REVIEW phase skill instance."""
    return ReviewPhase()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Data structures
    'QualityIssue',
    'ImplementationDecision',
    'Learning',
    'ReviewResult',

    # Phase implementation
    'ReviewPhase',

    # Factory
    'create_review_phase',
]
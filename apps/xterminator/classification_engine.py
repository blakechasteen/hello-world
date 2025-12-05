"""
Classification Engine
=====================

Orchestrates the complete classification pipeline:
1. Context Detection - Where is the issue?
2. Risk Assessment - How risky is fixing it?
3. Strategy Selection - What fix approach should we use?
4. Confidence Scoring - How confident are we?

Produces a ClassificationResult and FixProposal.
"""

import asyncio
from typing import Optional
from pathlib import Path

from .context_detector import ContextDetector
from .risk_assessor import RiskAssessor
from .strategy_selector import StrategySelector
from .confidence_scorer import ConfidenceScorer
from .xterminator_types import (
    ClassificationResult,
    FixProposal,
    CodeContext,
    RiskLevel,
    FixStrategy
)


class ClassificationEngine:
    """Complete classification pipeline for fix proposals"""

    def __init__(self):
        self.context_detector = ContextDetector()
        self.risk_assessor = RiskAssessor()
        self.strategy_selector = StrategySelector()
        self.confidence_scorer = ConfidenceScorer()

    async def classify(
        self,
        issue,  # SlopIssue from Trough
        full_code: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify an issue and determine fix approach.

        Args:
            issue: SlopIssue from Trough detection
            full_code: Full file content (for AST analysis)
            file_path: Path to file (overrides issue.file_path)

        Returns:
            ClassificationResult with complete analysis
        """
        # Extract issue details
        issue_category = issue.category.value if hasattr(issue.category, 'value') else str(issue.category)
        issue_severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
        detection_confidence = getattr(issue, 'confidence', 0.8)
        file_path = file_path or getattr(issue, 'file_path', 'unknown')
        line_number = getattr(issue, 'line_number', 1)

        # Read file if not provided
        if full_code is None and file_path != 'unknown':
            try:
                full_code = Path(file_path).read_text()
            except Exception:
                full_code = None

        # Parse into lines
        code_lines = full_code.split('\n') if full_code else [getattr(issue, 'code_snippet', '')]

        # Step 1: Context Detection
        context = self.context_detector.analyze_context(
            file_path=file_path,
            line_number=line_number,
            code_lines=code_lines,
            full_code=full_code
        )

        # Step 2: Risk Assessment
        risk_level, risk_factors = self.risk_assessor.assess_risk(
            issue_category=issue_category,
            issue_severity=issue_severity,
            context=context,
            code_lines=code_lines
        )

        # Step 3: Strategy Selection
        selected_strategy, strategy_reason, alternatives = self.strategy_selector.select_strategy(
            issue_category=issue_category,
            risk_level=risk_level,
            context=context,
            confidence=detection_confidence
        )

        # Step 4: Confidence Scoring
        complexity = self.risk_assessor._estimate_complexity(context, code_lines)
        confidence, confidence_breakdown = self.confidence_scorer.calculate_confidence(
            detection_confidence=detection_confidence,
            issue_category=issue_category,
            context=context,
            risk_level=risk_level,
            selected_strategy=selected_strategy,
            code_complexity=complexity
        )

        # Check for false positives
        is_false_positive = (
            confidence < 0.50 or
            selected_strategy == FixStrategy.SKIP or
            not context.is_safe_location()
        )

        false_positive_reason = ""
        if is_false_positive:
            if not context.is_safe_location():
                false_positive_reason = f"Issue in {context.context_type.value}, not executable code"
            elif confidence < 0.50:
                false_positive_reason = f"Low confidence ({confidence:.2f})"
            elif selected_strategy == FixStrategy.SKIP:
                false_positive_reason = strategy_reason

        # Generate recommendation
        recommendation = self._generate_recommendation(
            issue_category=issue_category,
            risk_level=risk_level,
            selected_strategy=selected_strategy,
            confidence=confidence,
            context=context,
            is_false_positive=is_false_positive
        )

        # Build classification result
        result = ClassificationResult(
            issue_id=f"cls_{line_number}_{issue_category}",
            context=context,
            is_false_positive=is_false_positive,
            false_positive_reason=false_positive_reason,
            risk_level=risk_level,
            risk_factors=risk_factors,
            selected_strategy=selected_strategy,
            strategy_reason=strategy_reason,
            alternative_strategies=alternatives,
            confidence=confidence,
            confidence_factors=confidence_breakdown,
            recommendation=recommendation
        )

        return result

    async def classify_and_propose(
        self,
        issue,
        full_code: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> FixProposal:
        """
        Classify issue and create fix proposal.

        Args:
            issue: SlopIssue from Trough
            full_code: Full file content
            file_path: Path to file

        Returns:
            FixProposal ready for fixing engine
        """
        # Classify
        classification = await self.classify(issue, full_code, file_path)

        # Get original code
        original_code = getattr(issue, 'code_snippet', '')

        # Convert to fix proposal
        proposal = classification.to_fix_proposal(issue, original_code)

        # Add implementation hints
        hints = self.strategy_selector.get_implementation_hints(
            strategy=proposal.fix_strategy,
            issue_category=proposal.issue_category,
            context=proposal.context
        )

        proposal.metadata['implementation_hints'] = hints
        proposal.metadata['classification'] = classification

        return proposal

    async def classify_batch(
        self,
        issues: list,
        full_code: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> list[ClassificationResult]:
        """
        Classify multiple issues in parallel.

        Args:
            issues: List of SlopIssue objects
            full_code: Full file content (shared)
            file_path: Path to file (shared)

        Returns:
            List of ClassificationResult objects
        """
        tasks = [
            self.classify(issue, full_code, file_path)
            for issue in issues
        ]
        return await asyncio.gather(*tasks)

    def _generate_recommendation(
        self,
        issue_category: str,
        risk_level: RiskLevel,
        selected_strategy: FixStrategy,
        confidence: float,
        context: CodeContext,
        is_false_positive: bool
    ) -> str:
        """Generate human-readable recommendation"""

        if is_false_positive:
            return f"[WARN] Likely false positive - skip fixing"

        confidence_label = self.confidence_scorer.confidence_label(confidence)

        if selected_strategy == FixStrategy.SKIP:
            return f"Skip: Issue in {context.context_type.value}"

        elif selected_strategy == FixStrategy.MANUAL:
            return (
                f"Manual fix required ({risk_level.value} risk, {confidence_label} confidence). "
                f"Review carefully and add tests before fixing."
            )

        elif selected_strategy == FixStrategy.AST:
            if self.confidence_scorer.should_autofix(confidence, risk_level, context.has_tests):
                return (
                    f"[AUTO] Safe to autofix using AST transformation "
                    f"({risk_level.value} risk, {confidence_label} confidence, tests present)"
                )
            else:
                return (
                    f"AST transformation available but needs review "
                    f"({risk_level.value} risk, {confidence_label} confidence, "
                    f"{'no tests' if not context.has_tests else 'manual approval required'})"
                )

        elif selected_strategy == FixStrategy.TEMPLATE:
            if self.confidence_scorer.should_autofix(confidence, risk_level, context.has_tests):
                return (
                    f"[AUTO] Safe to autofix using template "
                    f"({risk_level.value} risk, {confidence_label} confidence, tests present)"
                )
            else:
                return (
                    f"Template fix available but needs review "
                    f"({risk_level.value} risk, {confidence_label} confidence, "
                    f"{'no tests' if not context.has_tests else 'manual approval required'})"
                )

        return f"Strategy: {selected_strategy.value}, Risk: {risk_level.value}, Confidence: {confidence_label}"

    def get_statistics(self, results: list[ClassificationResult]) -> dict:
        """Get statistics from classification results"""
        if not results:
            return {}

        total = len(results)

        # Count by strategy
        strategy_counts = {}
        for result in results:
            strategy = result.selected_strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Count by risk
        risk_counts = {}
        for result in results:
            risk = result.risk_level.value
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        # Count false positives
        false_positives = sum(1 for r in results if r.is_false_positive)

        # Count auto-fixable
        auto_fixable = sum(
            1 for r in results
            if (r.selected_strategy in {FixStrategy.AST, FixStrategy.TEMPLATE} and
                r.risk_level == RiskLevel.LOW and
                r.confidence >= 0.85 and
                r.context.has_tests)
        )

        # Average confidence
        avg_confidence = sum(r.confidence for r in results) / total

        return {
            'total_issues': total,
            'false_positives': false_positives,
            'false_positive_rate': false_positives / total,
            'auto_fixable': auto_fixable,
            'auto_fixable_rate': auto_fixable / total,
            'needs_review': total - false_positives - auto_fixable,
            'avg_confidence': avg_confidence,
            'strategy_distribution': strategy_counts,
            'risk_distribution': risk_counts,
        }

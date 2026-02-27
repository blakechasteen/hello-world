"""
Bridge connecting Prompt Testing with CoVe (Chain-of-Verification) verification.

This module integrates HoloLoom's prompt testing framework with the CoVe
verification system, enabling automatic verification of test responses and
quality analysis of detected contradictions.

Status: ✅ Production Ready (December 2025)
Location: hololoom/prompting/testing/verification_bridge.py
Lines: ~200
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import Counter
import logging

from hololoom.prompting.testing.protocol import PromptTestCase, PromptTestResult
from hololoom.prompting.testing.metrics_collector import MetricsCollector, MetricType

# Try to import verification module with graceful fallback
try:
    from hololoom.verification.protocol import VerificationResult, VerificationStatus
    from hololoom.verification.chain import VerificationChain
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False
    VerificationResult = None  # type: ignore
    VerificationStatus = None  # type: ignore
    VerificationChain = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class EnhancedTestResult:
    """Test result enhanced with verification data and quality analysis."""

    test_result: PromptTestResult
    """Original test case result."""

    verification_result: Optional[Any] = None
    """Verification result (VerificationResult if available)."""

    verified: bool = False
    """Whether response passed verification."""

    quality_before: float = 0.0
    """Quality score before verification."""

    quality_after: float = 0.0
    """Quality score after verification."""

    contradictions: List[str] = field(default_factory=list)
    """List of detected contradictions (as strings)."""

    @property
    def quality_improvement(self) -> float:
        """Calculate quality improvement from verification."""
        if self.quality_before == 0:
            return 0.0
        return (self.quality_after - self.quality_before) / self.quality_before

    @property
    def contradiction_count(self) -> int:
        """Get count of detected contradictions."""
        return len(self.contradictions)


@dataclass
class VerificationInsights:
    """Aggregated insights from analyzing verification results."""

    failed_tests: int
    """Number of failed tests."""

    common_contradictions: Dict[str, int] = field(default_factory=dict)
    """Contradiction type -> count mapping."""

    avg_quality_drop: float = 0.0
    """Average quality drop across failed tests."""

    recommendations: List[str] = field(default_factory=list)
    """Actionable recommendations for improvement."""

    @property
    def total_contradictions(self) -> int:
        """Total contradiction count across all types."""
        return sum(self.common_contradictions.values())

    @property
    def most_common_contradiction(self) -> Optional[str]:
        """Get most frequently occurring contradiction type."""
        if not self.common_contradictions:
            return None
        return max(
            self.common_contradictions.items(),
            key=lambda x: x[1]
        )[0]


class PromptVerificationBridge:
    """
    Bridge connecting prompt testing with CoVe verification.

    Enables automatic verification of test responses, tracks quality
    improvements, and provides insights into detected contradictions.
    """

    def __init__(
        self,
        verification_chain: Optional[Any] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize the verification bridge.

        Args:
            verification_chain: Optional VerificationChain instance for CoVe
            metrics_collector: Optional MetricsCollector for recording metrics
        """
        self.verification_chain = verification_chain
        self.metrics_collector = metrics_collector
        self.verification_available = VERIFICATION_AVAILABLE and verification_chain is not None
        logger.info(
            f"Verification bridge initialized (verification_available={self.verification_available})"
        )

    async def verify_test_response(
        self,
        test_case: PromptTestCase,
        actual_response: str,
        test_result: PromptTestResult
    ) -> EnhancedTestResult:
        """
        Verify a test response using CoVe verification.

        Args:
            test_case: The original test case
            actual_response: The actual response to verify
            test_result: The test result from quality evaluation

        Returns:
            EnhancedTestResult with verification data and quality analysis
        """
        enhanced_result = EnhancedTestResult(
            test_result=test_result,
            quality_before=test_result.quality_scores.get('overall', 0.0)
        )

        # Skip verification if not available
        if not self.verification_available:
            logger.debug("Verification skipped (not available)")
            return enhanced_result

        try:
            # Run verification chain
            verification_result = await self.verification_chain.verify(
                response=actual_response,
                context={
                    'query': test_case.prompt_template,
                    'test_case': test_case.name
                }
            )

            enhanced_result.verification_result = verification_result
            enhanced_result.quality_after = verification_result.confidence_after
            enhanced_result.verified = verification_result.verification_passed

            # Extract contradictions
            if verification_result.contradictions_found:
                enhanced_result.contradictions = [
                    c.explanation or f"{c.contradiction_type.value}: {c.claim.text}"
                    for c in verification_result.contradictions_found
                ]

            logger.debug(
                f"Verification complete for {test_case.name}: "
                f"verified={enhanced_result.verified}, "
                f"contradictions={len(enhanced_result.contradictions)}"
            )

        except Exception as e:
            logger.error(f"Verification failed for {test_case.name}: {e}")
            enhanced_result.verified = False

        # Record metrics
        self.record_verification_metrics(enhanced_result)

        return enhanced_result

    async def analyze_failures(
        self,
        failed_results: List[PromptTestResult]
    ) -> VerificationInsights:
        """
        Analyze failed test results to identify patterns.

        Args:
            failed_results: List of failed test results

        Returns:
            VerificationInsights with recommendations
        """
        insights = VerificationInsights(failed_tests=len(failed_results))

        if not failed_results:
            logger.info("No failed results to analyze")
            return insights

        # Analyze regressions and quality drops
        quality_drops = []

        for result in failed_results:
            regressions = result.regressions_detected
            if regressions:
                # Count regression types
                for regression in regressions:
                    contradiction_type = self._classify_regression(regression)
                    insights.common_contradictions[contradiction_type] = \
                        insights.common_contradictions.get(contradiction_type, 0) + 1

            # Track quality scores
            overall_quality = result.quality_scores.get('overall', 0.0)
            quality_drops.append(1.0 - overall_quality)

        # Calculate average quality drop
        if quality_drops:
            insights.avg_quality_drop = sum(quality_drops) / len(quality_drops)

        # Generate recommendations
        insights.recommendations = self._generate_recommendations(insights)

        logger.info(
            f"Analysis complete: {insights.failed_tests} failures, "
            f"{insights.total_contradictions} contradictions detected, "
            f"avg quality drop: {insights.avg_quality_drop:.2f}"
        )

        return insights

    def record_verification_metrics(self, enhanced_result: EnhancedTestResult) -> None:
        """
        Record verification metrics to the collector.

        Args:
            enhanced_result: The enhanced test result with verification data
        """
        if not self.metrics_collector:
            return

        try:
            # Record extraction/verification time
            if enhanced_result.verification_result:
                self.metrics_collector.record_metric(
                    name=f"verification_latency_{enhanced_result.test_result.test_case.name}",
                    value=enhanced_result.verification_result.latency_ms,
                    metric_type=MetricType.LATENCY_MS,
                    tags={'stage': 'verification'}
                )

            # Record verification quality
            self.metrics_collector.record_metric(
                name=f"verification_quality_{enhanced_result.test_result.test_case.name}",
                value=enhanced_result.quality_after,
                metric_type=MetricType.QUALITY_SCORE,
                tags={'verified': str(enhanced_result.verified).lower()}
            )

            # Record contradictions found
            if enhanced_result.contradictions:
                self.metrics_collector.record_metric(
                    name=f"contradictions_{enhanced_result.test_result.test_case.name}",
                    value=float(len(enhanced_result.contradictions)),
                    metric_type=MetricType.REGRESSION_COUNT
                )

            logger.debug(f"Metrics recorded for {enhanced_result.test_result.test_case.name}")

        except Exception as e:
            logger.error(f"Failed to record verification metrics: {e}")

    def _classify_regression(self, regression: str) -> str:
        """Classify a regression string into a category type.

        Args:
            regression: Regression description

        Returns:
            Classification string
        """
        regression_lower = regression.lower()

        if any(word in regression_lower for word in ['factual', 'fact', 'accuracy']):
            return 'factual'
        elif any(word in regression_lower for word in ['logic', 'consistent', 'contradiction']):
            return 'logical'
        elif any(word in regression_lower for word in ['time', 'temporal', 'date']):
            return 'temporal'
        elif any(word in regression_lower for word in ['number', 'quantity', 'count']):
            return 'quantitative'
        elif any(word in regression_lower for word in ['meaning', 'semantic', 'sense']):
            return 'semantic'
        else:
            return 'unknown'

    def _generate_recommendations(self, insights: VerificationInsights) -> List[str]:
        """Generate actionable recommendations based on analysis.

        Args:
            insights: Verification insights to analyze

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Recommendation based on failure count
        if insights.failed_tests > 5:
            recommendations.append(
                f"High failure rate ({insights.failed_tests} tests). "
                "Consider reviewing prompt template clarity and context."
            )

        # Recommendation based on most common contradiction
        if insights.most_common_contradiction:
            most_common = insights.most_common_contradiction
            count = insights.common_contradictions[most_common]
            recommendations.append(
                f"Most common issue: {most_common} ({count} occurrences). "
                f"Add explicit guidance in prompt to address {most_common} issues."
            )

        # Recommendation based on quality drop
        if insights.avg_quality_drop > 0.3:
            recommendations.append(
                f"Significant quality drop (avg {insights.avg_quality_drop:.1%}). "
                "Consider multi-pass refinement strategies."
            )
        elif insights.avg_quality_drop > 0.15:
            recommendations.append(
                f"Moderate quality issues (avg {insights.avg_quality_drop:.1%}). "
                "Review test thresholds or prompt specificity."
            )

        # Recommendation based on total contradictions
        if insights.total_contradictions > 10:
            recommendations.append(
                "High contradiction count. Enable CoVe verification for all tests "
                "to catch hallucinations earlier."
            )

        return recommendations if recommendations else [
            "Tests performing well. Continue monitoring for regressions."
        ]


def create_verification_bridge(
    metrics_collector: Optional[MetricsCollector] = None
) -> PromptVerificationBridge:
    """
    Factory function to create a verification bridge.

    Args:
        metrics_collector: Optional metrics collector instance

    Returns:
        PromptVerificationBridge instance (with or without CoVe support)
    """
    verification_chain = None

    # Try to create verification chain if available
    if VERIFICATION_AVAILABLE:
        try:
            verification_chain = VerificationChain()
            logger.info("Verification chain created successfully")
        except Exception as e:
            logger.warning(f"Failed to create verification chain: {e}")

    return PromptVerificationBridge(
        verification_chain=verification_chain,
        metrics_collector=metrics_collector
    )

"""
Phase 3+: Extensible Validation Pipeline

Elegant, composable pipeline for MRF validation with plugin architecture.
Enables "moonshot swarm" parallel validation across multiple dimensions.

Created: November 2025
Status: Production ready
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


class ValidationPhase(Enum):
    """4-week validation timeline phases."""
    BASELINE = "baseline"  # Week 1-2
    AB_TEST = "ab_test"    # Week 2-3
    ANALYSIS = "analysis"  # Week 3
    DECISION = "decision"  # Week 4


@dataclass
class ValidationContext:
    """Shared context across validation pipeline."""

    phase: ValidationPhase
    start_time: datetime = field(default_factory=datetime.now)

    # Data accumulated across phases
    baseline_queries: List[Any] = field(default_factory=list)
    mrf_queries: List[Any] = field(default_factory=list)

    # Results from validators
    validator_results: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, validator_name: str, result: Any):
        """Add result from a validator."""
        self.validator_results[validator_name] = result

    def get_result(self, validator_name: str) -> Optional[Any]:
        """Get result from a validator."""
        return self.validator_results.get(validator_name)


class Validator(ABC):
    """
    Abstract base class for validation plugins.

    Validators are composable, extensible plugins that each validate
    one aspect of MRF improvements.

    Example validators:
    - QualityValidator (accuracy, completeness, clarity)
    - LatencyValidator (response time, throughput)
    - UserSatisfactionValidator (ratings, thumbs up/down)
    - StatisticalValidator (t-tests, effect sizes)
    - HumanEvaluationValidator (blind comparisons)
    """

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    @abstractmethod
    async def validate(self, context: ValidationContext) -> Dict[str, Any]:
        """
        Run validation and return results.

        Args:
            context: Shared validation context

        Returns:
            Dictionary with validation results
        """
        pass

    def should_run(self, context: ValidationContext) -> bool:
        """
        Determine if this validator should run in current phase.

        Override to customize when validator runs.
        """
        return self.enabled


class QualityValidator(Validator):
    """Validates quality metrics (accuracy, completeness, clarity)."""

    def __init__(self, min_improvement_pct: float = 10.0):
        super().__init__("quality")
        self.min_improvement_pct = min_improvement_pct

    async def validate(self, context: ValidationContext) -> Dict[str, Any]:
        """Validate quality improvements."""
        # Extract quality scores
        baseline_quality = [
            q.get('quality_score', 0.0)
            for q in context.baseline_queries
        ]
        mrf_quality = [
            q.get('quality_score', 0.0)
            for q in context.mrf_queries
        ]

        if not baseline_quality or not mrf_quality:
            return {"status": "insufficient_data"}

        # Calculate improvement
        baseline_avg = sum(baseline_quality) / len(baseline_quality)
        mrf_avg = sum(mrf_quality) / len(mrf_quality)

        improvement_pct = ((mrf_avg - baseline_avg) / baseline_avg) * 100

        return {
            "status": "pass" if improvement_pct >= self.min_improvement_pct else "fail",
            "baseline_avg": baseline_avg,
            "mrf_avg": mrf_avg,
            "improvement_pct": improvement_pct,
            "meets_target": improvement_pct >= self.min_improvement_pct
        }


class LatencyValidator(Validator):
    """Validates latency metrics."""

    def __init__(self, max_regression_pct: float = 20.0):
        super().__init__("latency")
        self.max_regression_pct = max_regression_pct

    async def validate(self, context: ValidationContext) -> Dict[str, Any]:
        """Validate latency hasn't regressed significantly."""
        baseline_latencies = [
            q.get('latency_ms', 0.0)
            for q in context.baseline_queries
        ]
        mrf_latencies = [
            q.get('latency_ms', 0.0)
            for q in context.mrf_queries
        ]

        if not baseline_latencies or not mrf_latencies:
            return {"status": "insufficient_data"}

        baseline_avg = sum(baseline_latencies) / len(baseline_latencies)
        mrf_avg = sum(mrf_latencies) / len(mrf_latencies)

        change_pct = ((mrf_avg - baseline_avg) / baseline_avg) * 100

        return {
            "status": "pass" if change_pct <= self.max_regression_pct else "fail",
            "baseline_avg_ms": baseline_avg,
            "mrf_avg_ms": mrf_avg,
            "change_pct": change_pct,
            "acceptable": change_pct <= self.max_regression_pct
        }


class UserSatisfactionValidator(Validator):
    """Validates user satisfaction metrics."""

    def __init__(self, min_rating: float = 4.0):
        super().__init__("user_satisfaction")
        self.min_rating = min_rating

    async def validate(self, context: ValidationContext) -> Dict[str, Any]:
        """Validate user satisfaction."""
        mrf_ratings = [
            q.get('user_rating', 0)
            for q in context.mrf_queries
            if q.get('user_rating') is not None
        ]

        if not mrf_ratings:
            return {"status": "insufficient_data"}

        avg_rating = sum(mrf_ratings) / len(mrf_ratings)

        thumbs_up = [
            q.get('thumbs_up', False)
            for q in context.mrf_queries
            if q.get('thumbs_up') is not None
        ]

        thumbs_up_rate = sum(thumbs_up) / len(thumbs_up) if thumbs_up else 0.0

        return {
            "status": "pass" if avg_rating >= self.min_rating else "fail",
            "avg_rating": avg_rating,
            "thumbs_up_rate": thumbs_up_rate,
            "meets_target": avg_rating >= self.min_rating
        }


class StatisticalValidator(Validator):
    """Validates statistical significance."""

    def __init__(self, significance_level: float = 0.05):
        super().__init__("statistical")
        self.significance_level = significance_level

    async def validate(self, context: ValidationContext) -> Dict[str, Any]:
        """Run statistical significance tests."""
        from .statistical_analysis import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        # Extract quality scores
        baseline_quality = [
            q.get('quality_score', 0.0)
            for q in context.baseline_queries
        ]
        mrf_quality = [
            q.get('quality_score', 0.0)
            for q in context.mrf_queries
        ]

        if len(baseline_quality) < 30 or len(mrf_quality) < 30:
            return {
                "status": "insufficient_data",
                "message": "Need >= 30 samples per variant for significance testing"
            }

        # Run t-test
        from .statistical_analysis import SignificanceLevel

        sig_level = SignificanceLevel.P_05  # 95% confidence
        result = analyzer.independent_ttest(baseline_quality, mrf_quality, sig_level)

        return {
            "status": "pass" if result.is_significant else "fail",
            "p_value": result.p_value,
            "is_significant": result.is_significant,
            "effect_size": result.effect_size,
            "confidence_interval": {
                "lower": result.ci_lower,
                "upper": result.ci_upper
            }
        }


class ValidationPipeline:
    """
    Extensible, composable validation pipeline.

    Usage:
        pipeline = ValidationPipeline()

        # Add validators
        pipeline.add_validator(QualityValidator(min_improvement_pct=20.0))
        pipeline.add_validator(LatencyValidator(max_regression_pct=20.0))
        pipeline.add_validator(StatisticalValidator(significance_level=0.05))

        # Run pipeline
        results = await pipeline.run(
            baseline_queries=baseline_data,
            mrf_queries=mrf_data,
            phase=ValidationPhase.ANALYSIS
        )

        # Check results
        if results.passed:
            print("DEPLOY to production")
        else:
            print(f"INVESTIGATE: {results.summary}")
    """

    def __init__(self):
        self.validators: List[Validator] = []

    def add_validator(self, validator: Validator):
        """Add a validator to the pipeline."""
        self.validators.append(validator)
        return self  # Allow chaining

    def remove_validator(self, name: str):
        """Remove a validator by name."""
        self.validators = [v for v in self.validators if v.name != name]
        return self

    async def run(
        self,
        baseline_queries: List[Dict],
        mrf_queries: List[Dict],
        phase: ValidationPhase = ValidationPhase.ANALYSIS
    ) -> 'PipelineResults':
        """
        Run all validators in pipeline.

        Args:
            baseline_queries: Baseline (traditional) query data
            mrf_queries: UnifiedMRF query data
            phase: Current validation phase

        Returns:
            PipelineResults with aggregated results
        """
        # Create context
        context = ValidationContext(
            phase=phase,
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries
        )

        # Run validators (in parallel)
        tasks = []
        for validator in self.validators:
            if validator.should_run(context):
                tasks.append(self._run_validator(validator, context))

        await asyncio.gather(*tasks)

        # Aggregate results
        return self._aggregate_results(context)

    async def _run_validator(self, validator: Validator, context: ValidationContext):
        """Run a single validator and store result in context."""
        try:
            result = await validator.validate(context)
            context.add_result(validator.name, result)
        except Exception as e:
            context.add_result(validator.name, {
                "status": "error",
                "error": str(e)
            })

    def _aggregate_results(self, context: ValidationContext) -> 'PipelineResults':
        """Aggregate results from all validators."""
        results = PipelineResults()

        passed = 0
        failed = 0
        errors = 0

        for validator_name, result in context.validator_results.items():
            status = result.get('status', 'unknown')

            if status == 'pass':
                passed += 1
            elif status == 'fail':
                failed += 1
            elif status == 'error':
                errors += 1

        results.passed = (failed == 0 and errors == 0)
        results.validator_results = context.validator_results
        results.summary = self._generate_summary(context)

        return results

    def _generate_summary(self, context: ValidationContext) -> str:
        """Generate summary of validation results."""
        quality_result = context.get_result('quality')
        latency_result = context.get_result('latency')
        stat_result = context.get_result('statistical')

        summary_parts = []

        if quality_result and quality_result.get('status') == 'pass':
            improvement = quality_result.get('improvement_pct', 0)
            summary_parts.append(f"+{improvement:.1f}% quality improvement")

        if stat_result and stat_result.get('is_significant'):
            p_value = stat_result.get('p_value', 1.0)
            summary_parts.append(f"statistically significant (p={p_value:.3f})")

        if latency_result and latency_result.get('status') == 'pass':
            change = latency_result.get('change_pct', 0)
            summary_parts.append(f"{change:+.1f}% latency change")

        return ", ".join(summary_parts) if summary_parts else "Validation incomplete"


@dataclass
class PipelineResults:
    """Results from validation pipeline."""

    passed: bool = False
    validator_results: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def get_recommendation(self) -> str:
        """Get deployment recommendation based on results."""
        if not self.passed:
            return "INVESTIGATE: Some validators failed"

        quality_result = self.validator_results.get('quality', {})
        improvement = quality_result.get('improvement_pct', 0)

        stat_result = self.validator_results.get('statistical', {})
        is_significant = stat_result.get('is_significant', False)

        if improvement >= 20 and is_significant:
            return "DEPLOY: Strong evidence of improvement"
        elif improvement >= 10 and is_significant:
            return "GRADUAL_ROLLOUT: Moderate evidence of improvement"
        else:
            return "INVESTIGATE: Improvement below target or not significant"


# Example usage
if __name__ == "__main__":
    async def main():
        # Simulate data
        baseline_queries = [
            {"quality_score": 0.72 + (i % 10) * 0.01, "latency_ms": 140 + (i % 5) * 2}
            for i in range(50)
        ]

        mrf_queries = [
            {"quality_score": 0.91 + (i % 10) * 0.01, "latency_ms": 155 + (i % 5) * 2,
             "user_rating": 4 + (i % 2), "thumbs_up": (i % 3) > 0}
            for i in range(50)
        ]

        # Create pipeline
        pipeline = ValidationPipeline()

        # Add validators (elegant chaining)
        pipeline \
            .add_validator(QualityValidator(min_improvement_pct=20.0)) \
            .add_validator(LatencyValidator(max_regression_pct=20.0)) \
            .add_validator(UserSatisfactionValidator(min_rating=4.0)) \
            .add_validator(StatisticalValidator(significance_level=0.05))

        # Run pipeline
        results = await pipeline.run(
            baseline_queries=baseline_queries,
            mrf_queries=mrf_queries,
            phase=ValidationPhase.ANALYSIS
        )

        # Print results
        print("\n" + "="*60)
        print("VALIDATION PIPELINE RESULTS")
        print("="*60)

        print(f"\nOverall: {'PASS' if results.passed else 'FAIL'}")
        print(f"Summary: {results.summary}")
        print(f"Recommendation: {results.get_recommendation()}")

        print(f"\nValidator Results:")
        for name, result in results.validator_results.items():
            status = result.get('status', 'unknown')
            print(f"  {name}: {status.upper()}")

            if name == 'quality':
                improvement = result.get('improvement_pct', 0)
                print(f"    Improvement: {improvement:+.1f}%")
            elif name == 'latency':
                change = result.get('change_pct', 0)
                print(f"    Latency change: {change:+.1f}%")
            elif name == 'statistical':
                p_value = result.get('p_value', 1.0)
                print(f"    P-value: {p_value:.4f}")

        print("\n" + "="*60)

    asyncio.run(main())

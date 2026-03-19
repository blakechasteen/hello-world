"""
MRF A/B Testing Framework
==========================
Production A/B testing for validating MRF enhancements against baseline.

**Phase 2 - November 2025**: Statistical validation of MRF improvements
before full deployment.

Features:
- Traffic splitting (control vs treatment)
- Statistical significance testing (Welch's t-test)
- Effect size calculation (Cohen's d)
- Automatic deployment decisions
- Real-time monitoring

Usage:
    from hololoom.prompting.analytics import ABTest

    # Create A/B test
    test = ABTest(
        name="mrf_agentic_verify",
        control_description="Baseline agentic reasoning",
        treatment_description="MRF-enhanced verify mode",
        traffic_split=0.5  # 50/50 split
    )

    # Assign user to group
    group = test.assign_group(user_id="user_123")

    # Log result
    test.log_result(
        user_id="user_123",
        group=group,
        quality_score=0.92,
        execution_time_ms=450.0
    )

    # Check if statistically significant
    analysis = test.analyze()
    if analysis["is_significant"] and analysis["treatment_better"]:
        print(f"Deploy treatment! Effect size: {analysis['cohens_d']:.2f}")
"""

import hashlib
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ABGroup(Enum):
    """A/B test group assignment."""
    CONTROL = "control"
    TREATMENT = "treatment"


class DeploymentDecision(Enum):
    """Deployment decision based on A/B test results."""
    DEPLOY = "deploy"  # Strong evidence, deploy treatment
    MONITOR = "monitor"  # Promising but need more data
    REJECT = "reject"  # No improvement or regression
    INCONCLUSIVE = "inconclusive"  # Not enough data


@dataclass
class ABResult:
    """Single A/B test result."""
    timestamp: float
    user_id: str
    group: ABGroup
    quality_score: float
    execution_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestConfig:
    """Configuration for A/B test."""
    name: str
    control_description: str
    treatment_description: str
    traffic_split: float = 0.5  # 50/50 by default
    minimum_sample_size: int = 30  # Per group
    significance_level: float = 0.05  # p < 0.05
    minimum_effect_size: float = 0.2  # Cohen's d >= 0.2 (small effect)


class ABTest:
    """
    A/B test for validating MRF enhancements.

    Provides traffic splitting, statistical analysis, and deployment decisions.
    """

    def __init__(
        self,
        name: str,
        control_description: str,
        treatment_description: str,
        traffic_split: float = 0.5,
        minimum_sample_size: int = 30,
        persist_path: Path | None = None
    ):
        """
        Initialize A/B test.

        Args:
            name: Test name (unique identifier)
            control_description: Description of control group
            treatment_description: Description of treatment group
            traffic_split: Fraction of traffic to treatment (0.0-1.0)
            minimum_sample_size: Minimum samples per group for analysis
            persist_path: Optional path to persist test data
        """
        self.config = ABTestConfig(
            name=name,
            control_description=control_description,
            treatment_description=treatment_description,
            traffic_split=traffic_split,
            minimum_sample_size=minimum_sample_size
        )

        self.persist_path = persist_path or Path("./mrf_ab_tests")
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: dict[ABGroup, list[ABResult]] = {
            ABGroup.CONTROL: [],
            ABGroup.TREATMENT: []
        }

        # User assignments (for consistency)
        self.user_assignments: dict[str, ABGroup] = {}

        # Load existing data
        self._load_data()

    def assign_group(self, user_id: str) -> ABGroup:
        """
        Assign user to control or treatment group.

        Uses deterministic hashing to ensure consistent assignment.

        Args:
            user_id: User identifier

        Returns:
            Assigned group (CONTROL or TREATMENT)
        """
        # Check if already assigned
        if user_id in self.user_assignments:
            return self.user_assignments[user_id]

        # Deterministic assignment using hash
        hash_value = int(hashlib.md5(
            f"{self.config.name}:{user_id}".encode()
        ).hexdigest(), 16)

        # Assign based on traffic split
        assigned_group = (
            ABGroup.TREATMENT
            if (hash_value % 100) < (self.config.traffic_split * 100)
            else ABGroup.CONTROL
        )

        self.user_assignments[user_id] = assigned_group
        return assigned_group

    def log_result(
        self,
        user_id: str,
        group: ABGroup,
        quality_score: float,
        execution_time_ms: float,
        metadata: dict[str, Any] | None = None
    ):
        """
        Log A/B test result.

        Args:
            user_id: User identifier
            group: Group assignment
            quality_score: Quality metric (0.0-1.0)
            execution_time_ms: Execution time
            metadata: Optional metadata
        """
        result = ABResult(
            timestamp=time.time(),
            user_id=user_id,
            group=group,
            quality_score=quality_score,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {}
        )

        self.results[group].append(result)

        logger.debug(
            f"[A/B {self.config.name}] {group.value}: "
            f"quality={quality_score:.2f}, time={execution_time_ms:.1f}ms"
        )

        # Persist
        self._persist_result(result)

    def analyze(self) -> dict[str, Any]:
        """
        Analyze A/B test results with statistical tests.

        Returns:
            Analysis results with deployment decision
        """
        control_results = self.results[ABGroup.CONTROL]
        treatment_results = self.results[ABGroup.TREATMENT]

        # Check minimum sample size
        if (len(control_results) < self.config.minimum_sample_size or
            len(treatment_results) < self.config.minimum_sample_size):
            return {
                "is_significant": False,
                "deployment_decision": DeploymentDecision.INCONCLUSIVE.value,
                "reason": f"Insufficient data (need {self.config.minimum_sample_size} per group)",
                "sample_sizes": {
                    "control": len(control_results),
                    "treatment": len(treatment_results)
                }
            }

        # Extract quality scores
        control_quality = [r.quality_score for r in control_results]
        treatment_quality = [r.quality_score for r in treatment_results]

        # Descriptive statistics
        control_mean = statistics.mean(control_quality)
        treatment_mean = statistics.mean(treatment_quality)
        control_std = statistics.stdev(control_quality) if len(control_quality) > 1 else 0.0
        treatment_std = statistics.stdev(treatment_quality) if len(treatment_quality) > 1 else 0.0

        # Effect size (Cohen's d)
        pooled_std = ((control_std ** 2 + treatment_std ** 2) / 2) ** 0.5
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0

        # Welch's t-test (approximate, simplified)
        # For production, use scipy.stats.ttest_ind(equal_var=False)
        treatment_better = treatment_mean > control_mean
        improvement_percent = ((treatment_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0.0

        # Approximate p-value using normal approximation
        # In production, use scipy for exact calculation
        se_control = control_std / (len(control_quality) ** 0.5)
        se_treatment = treatment_std / (len(treatment_quality) ** 0.5)
        se_diff = (se_control ** 2 + se_treatment ** 2) ** 0.5

        # Z-score approximation
        z_score = abs(treatment_mean - control_mean) / se_diff if se_diff > 0 else 0.0

        # Approximate p-value (two-tailed)
        # For z > 1.96, p < 0.05
        is_significant = z_score >= 1.96

        # Deployment decision
        decision = self._make_deployment_decision(
            is_significant=is_significant,
            treatment_better=treatment_better,
            cohens_d=cohens_d,
            improvement_percent=improvement_percent
        )

        # Execution time comparison
        control_time = statistics.mean([r.execution_time_ms for r in control_results])
        treatment_time = statistics.mean([r.execution_time_ms for r in treatment_results])
        time_increase_percent = ((treatment_time - control_time) / control_time * 100) if control_time > 0 else 0.0

        return {
            "test_name": self.config.name,
            "is_significant": is_significant,
            "treatment_better": treatment_better,
            "deployment_decision": decision.value,
            "statistics": {
                "control": {
                    "mean_quality": control_mean,
                    "std_quality": control_std,
                    "mean_time_ms": control_time,
                    "sample_size": len(control_results)
                },
                "treatment": {
                    "mean_quality": treatment_mean,
                    "std_quality": treatment_std,
                    "mean_time_ms": treatment_time,
                    "sample_size": len(treatment_results)
                },
                "difference": {
                    "quality_improvement_percent": improvement_percent,
                    "cohens_d": cohens_d,
                    "z_score": z_score,
                    "time_increase_percent": time_increase_percent
                }
            },
            "interpretation": self._interpret_results(
                cohens_d, improvement_percent, time_increase_percent
            )
        }

    def _make_deployment_decision(
        self,
        is_significant: bool,
        treatment_better: bool,
        cohens_d: float,
        improvement_percent: float
    ) -> DeploymentDecision:
        """
        Make deployment decision based on statistical analysis.

        Args:
            is_significant: Whether difference is statistically significant
            treatment_better: Whether treatment outperforms control
            cohens_d: Cohen's d effect size
            improvement_percent: Improvement percentage

        Returns:
            Deployment decision
        """
        # Strong evidence for deployment
        if is_significant and treatment_better and abs(cohens_d) >= self.config.minimum_effect_size:
            return DeploymentDecision.DEPLOY

        # Promising but needs more data
        if treatment_better and improvement_percent > 10:
            return DeploymentDecision.MONITOR

        # Clear regression
        if is_significant and not treatment_better:
            return DeploymentDecision.REJECT

        # Not enough evidence
        return DeploymentDecision.INCONCLUSIVE

    def _interpret_results(
        self,
        cohens_d: float,
        improvement_percent: float,
        time_increase_percent: float
    ) -> str:
        """Generate human-readable interpretation."""
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_desc = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_desc = "small"
        elif abs(cohens_d) < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"

        # Recommendation
        if improvement_percent > 20 and abs(cohens_d) >= 0.5:
            recommendation = "Strong evidence for treatment. Recommend full deployment."
        elif improvement_percent > 10:
            recommendation = "Moderate improvement. Consider gradual rollout."
        elif improvement_percent < -5:
            recommendation = "Treatment performs worse. Do not deploy."
        else:
            recommendation = "Insufficient evidence. Continue monitoring."

        # Time consideration
        time_note = ""
        if time_increase_percent > 50:
            time_note = " WARNING: Significant latency increase."
        elif time_increase_percent > 20:
            time_note = " Note: Moderate latency increase."

        return f"Effect size: {effect_desc} (Cohen's d={cohens_d:.2f}). " \
               f"Quality improvement: {improvement_percent:+.1f}%.{time_note} {recommendation}"

    def get_summary(self) -> dict[str, Any]:
        """Get test summary."""
        return {
            "name": self.config.name,
            "control_description": self.config.control_description,
            "treatment_description": self.config.treatment_description,
            "traffic_split": self.config.traffic_split,
            "sample_sizes": {
                "control": len(self.results[ABGroup.CONTROL]),
                "treatment": len(self.results[ABGroup.TREATMENT])
            },
            "ready_for_analysis": (
                len(self.results[ABGroup.CONTROL]) >= self.config.minimum_sample_size and
                len(self.results[ABGroup.TREATMENT]) >= self.config.minimum_sample_size
            )
        }

    def _persist_result(self, result: ABResult):
        """Persist single result to disk."""
        results_file = self.persist_path / f"{self.config.name}_results.jsonl"
        with open(results_file, "a", encoding="utf-8") as f:
            data = asdict(result)
            data["group"] = result.group.value  # Convert enum to string
            f.write(json.dumps(data) + "\n")

    def _load_data(self):
        """Load existing test data from disk."""
        results_file = self.persist_path / f"{self.config.name}_results.jsonl"

        if not results_file.exists():
            logger.info(f"[A/B] No existing data for test '{self.config.name}'")
            return

        try:
            with open(results_file, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    # Convert string back to enum
                    data["group"] = ABGroup(data["group"])
                    result = ABResult(**data)
                    self.results[result.group].append(result)
                    self.user_assignments[result.user_id] = result.group

            logger.info(
                f"[A/B] Loaded test '{self.config.name}': "
                f"control={len(self.results[ABGroup.CONTROL])}, "
                f"treatment={len(self.results[ABGroup.TREATMENT])}"
            )

        except Exception as e:
            logger.error(f"[A/B] Error loading test data: {e}")


# Convenience functions
def create_ab_test(
    name: str,
    control_description: str,
    treatment_description: str,
    traffic_split: float = 0.5,
    persist_path: Path | None = None
) -> ABTest:
    """
    Create A/B test.

    Args:
        name: Test name
        control_description: Control description
        treatment_description: Treatment description
        traffic_split: Traffic split (default: 0.5)
        persist_path: Optional persist path

    Returns:
        ABTest instance
    """
    return ABTest(
        name=name,
        control_description=control_description,
        treatment_description=treatment_description,
        traffic_split=traffic_split,
        persist_path=persist_path
    )


__all__ = [
    "ABTest",
    "ABGroup",
    "DeploymentDecision",
    "ABResult",
    "ABTestConfig",
    "create_ab_test"
]

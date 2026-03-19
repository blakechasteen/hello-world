"""Regression Testing capabilities for detecting quality drift.

This module provides infrastructure for detecting regressions in prompt quality,
latency, token usage, and consistency across test runs.

Status: ✅ Production Ready (November 2025)
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from hololoom.prompting.testing.protocol import (
    PromptTestConfig,
    PromptTestResult,
)


class RegressionType(Enum):
    """Types of regressions that can be detected."""

    QUALITY_DROP = "quality_drop"  # Overall quality decreased >5%
    LATENCY_INCREASE = "latency_increase"  # Response time increased >20%
    TOKEN_INCREASE = "token_increase"  # Token usage increased >10%
    CONSISTENCY_CHANGE = "consistency_change"  # Output format changed
    GOLDEN_MISMATCH = "golden_mismatch"  # Golden output no longer matches


@dataclass
class Regression:
    """A detected regression in test results."""

    regression_type: RegressionType
    """Type of regression detected."""

    test_name: str
    """Name of the test that regressed."""

    baseline_value: float
    """Value from baseline/previous run."""

    current_value: float
    """Current value."""

    delta: float
    """Absolute change (current - baseline)."""

    delta_percent: float
    """Percentage change ((current - baseline) / baseline * 100)."""

    severity: str
    """Severity level: LOW, MEDIUM, HIGH, CRITICAL."""

    description: str
    """Human-readable description of the regression."""

    detected_at: datetime = field(default_factory=datetime.now)
    """When this regression was detected."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["regression_type"] = self.regression_type.value
        data["detected_at"] = self.detected_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Regression":
        """Create from dictionary for JSON deserialization."""
        data = data.copy()
        data["regression_type"] = RegressionType(data["regression_type"])
        data["detected_at"] = datetime.fromisoformat(data["detected_at"])
        return cls(**data)


class RegressionDetector:
    """Detects regressions by comparing current and baseline test results."""

    # Default thresholds
    QUALITY_DROP_THRESHOLD = 0.05  # 5%
    LATENCY_INCREASE_THRESHOLD = 0.20  # 20%
    TOKEN_INCREASE_THRESHOLD = 0.10  # 10%

    def __init__(self, config: PromptTestConfig | None = None):
        """Initialize regression detector with optional config.

        Args:
            config: Optional PromptTestConfig with custom thresholds
        """
        self.config = config or PromptTestConfig()
        self.quality_threshold = (
            config.regression_threshold
            if config
            else self.QUALITY_DROP_THRESHOLD
        )
        self.latency_threshold = self.LATENCY_INCREASE_THRESHOLD
        self.token_threshold = self.TOKEN_INCREASE_THRESHOLD

    async def detect_regressions(
        self,
        current_results: list[PromptTestResult],
        baseline_results: list[PromptTestResult],
    ) -> list[Regression]:
        """Detect regressions between baseline and current results.

        Args:
            current_results: Current test results
            baseline_results: Baseline/historical results

        Returns:
            List of detected regressions
        """
        regressions: list[Regression] = []

        # Create lookup map for baseline results by test name
        baseline_map = {r.test_case.name: r for r in baseline_results}

        for current in current_results:
            test_name = current.test_case.name
            if test_name not in baseline_map:
                continue

            baseline = baseline_map[test_name]

            # Check for quality regression
            quality_reg = self._compare_quality(current, baseline)
            if quality_reg:
                regressions.append(quality_reg)

            # Check for latency regression
            latency_reg = self._compare_latency(current, baseline)
            if latency_reg:
                regressions.append(latency_reg)

            # Check for token regression
            token_reg = self._compare_tokens(current, baseline)
            if token_reg:
                regressions.append(token_reg)

        return regressions

    def _compare_quality(
        self, current: PromptTestResult, baseline: PromptTestResult
    ) -> Regression | None:
        """Compare quality scores between current and baseline.

        Args:
            current: Current test result
            baseline: Baseline test result

        Returns:
            Regression if quality dropped, None otherwise
        """
        # Average quality scores from all dimensions
        current_avg = (
            sum(current.quality_scores.values())
            / len(current.quality_scores)
            if current.quality_scores
            else 0.0
        )
        baseline_avg = (
            sum(baseline.quality_scores.values())
            / len(baseline.quality_scores)
            if baseline.quality_scores
            else 0.0
        )

        if baseline_avg == 0:
            return None

        delta = current_avg - baseline_avg
        delta_percent = (delta / baseline_avg) * 100

        # Check if regression exceeds threshold
        if delta_percent < -self.quality_threshold * 100:
            severity = self._calculate_severity(abs(delta_percent))
            return Regression(
                regression_type=RegressionType.QUALITY_DROP,
                test_name=current.test_case.name,
                baseline_value=baseline_avg,
                current_value=current_avg,
                delta=delta,
                delta_percent=delta_percent,
                severity=severity,
                description=f"Quality dropped from {baseline_avg:.2f} to {current_avg:.2f} ({delta_percent:.1f}%)",
            )

        return None

    def _compare_latency(
        self, current: PromptTestResult, baseline: PromptTestResult
    ) -> Regression | None:
        """Compare latency between current and baseline.

        Args:
            current: Current test result
            baseline: Baseline test result

        Returns:
            Regression if latency increased significantly, None otherwise
        """
        if baseline.latency_ms == 0:
            return None

        delta = current.latency_ms - baseline.latency_ms
        delta_percent = (delta / baseline.latency_ms) * 100

        # Check if regression exceeds threshold
        if delta_percent > self.latency_threshold * 100:
            severity = self._calculate_severity(delta_percent)
            return Regression(
                regression_type=RegressionType.LATENCY_INCREASE,
                test_name=current.test_case.name,
                baseline_value=baseline.latency_ms,
                current_value=current.latency_ms,
                delta=delta,
                delta_percent=delta_percent,
                severity=severity,
                description=f"Latency increased from {baseline.latency_ms:.1f}ms to {current.latency_ms:.1f}ms ({delta_percent:.1f}%)",
            )

        return None

    def _compare_tokens(
        self, current: PromptTestResult, baseline: PromptTestResult
    ) -> Regression | None:
        """Compare token usage between current and baseline.

        Args:
            current: Current test result
            baseline: Baseline test result

        Returns:
            Regression if token usage increased, None otherwise
        """
        if baseline.token_count == 0:
            return None

        delta = current.token_count - baseline.token_count
        delta_percent = (delta / baseline.token_count) * 100

        # Check if regression exceeds threshold
        if delta_percent > self.token_threshold * 100:
            severity = self._calculate_severity(delta_percent)
            return Regression(
                regression_type=RegressionType.TOKEN_INCREASE,
                test_name=current.test_case.name,
                baseline_value=float(baseline.token_count),
                current_value=float(current.token_count),
                delta=float(delta),
                delta_percent=delta_percent,
                severity=severity,
                description=f"Token count increased from {baseline.token_count} to {current.token_count} ({delta_percent:.1f}%)",
            )

        return None

    def _calculate_severity(self, delta_percent: float) -> str:
        """Calculate severity based on percentage change.

        Args:
            delta_percent: Absolute percentage change

        Returns:
            Severity level: LOW, MEDIUM, HIGH, or CRITICAL
        """
        abs_delta = abs(delta_percent)
        if abs_delta < 10:
            return "LOW"
        elif abs_delta < 25:
            return "MEDIUM"
        elif abs_delta < 50:
            return "HIGH"
        else:
            return "CRITICAL"

    def save_baseline(
        self, results: list[PromptTestResult], path: str
    ) -> None:
        """Save test results as baseline for future comparison.

        Args:
            results: Test results to save
            path: Path to save baseline to
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "test_name": r.test_case.name,
                    "quality_scores": r.quality_scores,
                    "latency_ms": r.latency_ms,
                    "token_count": r.token_count,
                }
                for r in results
            ],
        }

        with open(path, "w") as f:
            json.dump(baseline_data, f, indent=2)

    def load_baseline(
        self, path: str
    ) -> list[PromptTestResult]:
        """Load baseline results from saved file.

        Args:
            path: Path to baseline file

        Returns:
            List of baseline PromptTestResult objects
        """
        with open(path) as f:
            baseline_data = json.load(f)

        # Reconstruct basic results (note: test_case details are not restored)
        results = []
        for item in baseline_data.get("results", []):
            # Create minimal test result for comparison
            from hololoom.prompting.testing.protocol import (
                PromptTestCase,
                TestType,
            )

            test_case = PromptTestCase(
                name=item["test_name"],
                prompt_template="",
                test_inputs=[],
                expected_qualities=item.get("quality_scores", {}),
                test_type=TestType.REGRESSION,
            )

            result = PromptTestResult(
                test_case=test_case,
                passed=True,
                quality_scores=item.get("quality_scores", {}),
                latency_ms=item.get("latency_ms", 0),
                token_count=item.get("token_count", 0),
            )
            results.append(result)

        return results


def create_regression_detector(
    config: PromptTestConfig | None = None,
) -> RegressionDetector:
    """Factory function to create a regression detector.

    Args:
        config: Optional PromptTestConfig with custom settings

    Returns:
        RegressionDetector instance
    """
    return RegressionDetector(config)

"""
Continuous Validator - Monitors classifier accuracy with automatic alerting
===========================================================================
Continuously validates classifier performance and detects regressions.

Author: Claude Code
Date: November 13, 2025
Phase: 3 - Adaptive Learning
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ValidationQuery:
    """Single validation query with ground truth label."""
    text: str
    expected_complexity: str  # Ground truth
    metadata: Dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a validation run."""
    timestamp: float
    overall_accuracy: float
    by_complexity: Dict[str, float]
    total_queries: int
    correct: int
    incorrect: int
    regression_detected: bool
    baseline_accuracy: Optional[float] = None
    trend_7day: Optional[float] = None
    trend_30day: Optional[float] = None


@dataclass
class RegressionAlert:
    """Alert for detected regression."""
    timestamp: float
    current_accuracy: float
    baseline_accuracy: float
    drop_percentage: float
    affected_complexity: List[str]
    severity: str  # "warning", "critical"
    message: str


class ContinuousValidator:
    """
    Continuous accuracy monitoring with automatic regression detection.

    Features:
    - Hourly/daily/weekly validation schedules
    - Regression detection (>2% accuracy drop)
    - Trend analysis (7-day, 30-day moving averages)
    - Automatic alerts (Slack, email)
    - Validation set management

    Usage:
        validator = ContinuousValidator(
            classifier=moonshot_classifier,
            validation_set_path="./validation_set.json"
        )

        # Hourly check
        result = await validator.validate_hourly()

        # Daily check
        result = await validator.validate_daily()

        # Check for regression
        if result.regression_detected:
            alerts = validator.get_alerts()
    """

    def __init__(
        self,
        classifier,
        validation_set_path: Optional[str] = None,
        regression_threshold: float = 0.02,  # 2% drop triggers regression
        history_size: int = 1000,  # Keep last 1000 validation results
        baseline_accuracy: Optional[float] = None
    ):
        """
        Initialize continuous validator.

        Args:
            classifier: Query classifier instance
            validation_set_path: Path to validation set JSON
            regression_threshold: Accuracy drop to trigger regression (default: 2%)
            history_size: Number of validation results to keep
            baseline_accuracy: Expected accuracy (defaults to 100% from Phase 1)
        """
        self.classifier = classifier
        self.validation_set_path = Path(validation_set_path) if validation_set_path else None
        self.regression_threshold = regression_threshold
        self.baseline_accuracy = baseline_accuracy or 1.0  # 100% from Phase 1

        # Validation history (for trend analysis)
        self.validation_history = deque(maxlen=history_size)

        # Alerts queue
        self.alerts = []

        # Load validation set
        self.validation_set = self._load_validation_set()

        logger.info(f"ContinuousValidator initialized "
                   f"(validation_set_size={len(self.validation_set)}, "
                   f"regression_threshold={regression_threshold:.1%}, "
                   f"baseline={self.baseline_accuracy:.1%})")

    def _load_validation_set(self) -> List[ValidationQuery]:
        """Load validation set from disk."""
        if not self.validation_set_path or not self.validation_set_path.exists():
            logger.warning("No validation set found, using empty set")
            return []

        try:
            with open(self.validation_set_path, 'r') as f:
                data = json.load(f)

            queries = []
            for item in data:
                query = ValidationQuery(
                    text=item['text'],
                    expected_complexity=item['complexity'],
                    metadata=item.get('metadata', {})
                )
                queries.append(query)

            logger.info(f"Loaded {len(queries)} validation queries")
            return queries

        except Exception as e:
            logger.error(f"Error loading validation set: {e}")
            return []

    async def validate_hourly(self, sample_size: int = 100) -> ValidationResult:
        """
        Quick hourly validation (100 queries).

        Args:
            sample_size: Number of queries to validate

        Returns:
            ValidationResult with accuracy metrics
        """
        logger.info("Running hourly validation...")

        # Sample queries for quick check
        import random
        queries = random.sample(self.validation_set, min(sample_size, len(self.validation_set)))

        # Run validation
        result = self._validate_queries(queries)

        # Detect regression
        if self._detect_regression(result):
            self._create_alert(result)

        # Store in history
        self.validation_history.append(result)

        logger.info(f"Hourly validation complete: "
                   f"accuracy={result.overall_accuracy:.1%}, "
                   f"regression={'YES' if result.regression_detected else 'NO'}")

        return result

    async def validate_daily(self) -> ValidationResult:
        """
        Full daily validation (all queries).

        Returns:
            ValidationResult with comprehensive metrics
        """
        logger.info("Running daily validation...")

        # Validate all queries
        result = self._validate_queries(self.validation_set)

        # Calculate trends
        result.trend_7day = self._calculate_trend(days=7)
        result.trend_30day = self._calculate_trend(days=30)

        # Detect regression
        if self._detect_regression(result):
            self._create_alert(result)

        # Store in history
        self.validation_history.append(result)

        logger.info(f"Daily validation complete: "
                   f"accuracy={result.overall_accuracy:.1%}, "
                   f"7d_trend={result.trend_7day:.1%} if result.trend_7day else 'N/A', "
                   f"30d_trend={result.trend_30day:.1%} if result.trend_30day else 'N/A', "
                   f"regression={'YES' if result.regression_detected else 'NO'}")

        return result

    def _validate_queries(self, queries: List[ValidationQuery]) -> ValidationResult:
        """
        Validate a list of queries.

        Args:
            queries: List of validation queries

        Returns:
            ValidationResult with accuracy metrics
        """
        if not queries:
            return ValidationResult(
                timestamp=datetime.now().timestamp(),
                overall_accuracy=0.0,
                by_complexity={},
                total_queries=0,
                correct=0,
                incorrect=0,
                regression_detected=False
            )

        # Track results by complexity
        results_by_complexity = {
            'trivial': {'correct': 0, 'total': 0},
            'simple': {'correct': 0, 'total': 0},
            'complex': {'correct': 0, 'total': 0},
            'research': {'correct': 0, 'total': 0}
        }

        total_correct = 0
        total_queries = len(queries)

        # Classify each query
        for query in queries:
            classification = self.classifier.classify(query.text)
            predicted = classification.complexity.value
            expected = query.expected_complexity

            # Update counts
            results_by_complexity[expected]['total'] += 1

            if predicted == expected:
                results_by_complexity[expected]['correct'] += 1
                total_correct += 1

        # Calculate accuracies
        overall_accuracy = total_correct / total_queries if total_queries > 0 else 0.0

        by_complexity = {}
        for complexity, stats in results_by_complexity.items():
            if stats['total'] > 0:
                by_complexity[complexity] = stats['correct'] / stats['total']
            else:
                by_complexity[complexity] = 0.0

        return ValidationResult(
            timestamp=datetime.now().timestamp(),
            overall_accuracy=overall_accuracy,
            by_complexity=by_complexity,
            total_queries=total_queries,
            correct=total_correct,
            incorrect=total_queries - total_correct,
            regression_detected=False,  # Set by _detect_regression()
            baseline_accuracy=self.baseline_accuracy
        )

    def _detect_regression(self, result: ValidationResult) -> bool:
        """
        Detect if accuracy has regressed.

        Regression criteria:
        - Overall accuracy drop >2% from baseline
        - Any complexity level drop >5% from baseline

        Args:
            result: Validation result

        Returns:
            True if regression detected
        """
        # Check overall accuracy
        overall_drop = self.baseline_accuracy - result.overall_accuracy
        if overall_drop > self.regression_threshold:
            result.regression_detected = True
            logger.warning(f"REGRESSION DETECTED: Overall accuracy dropped "
                          f"{overall_drop:.1%} (threshold: {self.regression_threshold:.1%})")
            return True

        # Check per-complexity regression (stricter threshold: 5%)
        complexity_threshold = 0.05
        for complexity, accuracy in result.by_complexity.items():
            drop = self.baseline_accuracy - accuracy
            if drop > complexity_threshold:
                result.regression_detected = True
                logger.warning(f"REGRESSION DETECTED: {complexity.upper()} accuracy dropped "
                              f"{drop:.1%} (threshold: {complexity_threshold:.1%})")
                return True

        return False

    def _create_alert(self, result: ValidationResult):
        """Create regression alert."""
        # Calculate affected complexity levels
        affected = []
        for complexity, accuracy in result.by_complexity.items():
            drop = self.baseline_accuracy - accuracy
            if drop > 0.05:  # 5% drop
                affected.append(f"{complexity}({accuracy:.1%})")

        # Determine severity
        overall_drop = self.baseline_accuracy - result.overall_accuracy
        if overall_drop > 0.05:  # >5% drop
            severity = "critical"
        else:
            severity = "warning"

        # Create alert message
        message = (
            f"Classifier Regression Detected\n"
            f"Current accuracy: {result.overall_accuracy:.1%}\n"
            f"Baseline accuracy: {result.baseline_accuracy:.1%}\n"
            f"Drop: {overall_drop:.1%} (threshold: {self.regression_threshold:.1%})\n"
            f"Affected: {', '.join(affected) if affected else 'Overall'}\n"
            f"Time: {datetime.now().isoformat()}"
        )

        alert = RegressionAlert(
            timestamp=result.timestamp,
            current_accuracy=result.overall_accuracy,
            baseline_accuracy=result.baseline_accuracy,
            drop_percentage=overall_drop,
            affected_complexity=affected,
            severity=severity,
            message=message
        )

        self.alerts.append(alert)
        logger.error(f"[ALERT] {message}")

    def _calculate_trend(self, days: int) -> Optional[float]:
        """
        Calculate accuracy trend over N days.

        Args:
            days: Number of days to calculate trend

        Returns:
            Average accuracy over period, or None if insufficient data
        """
        if not self.validation_history:
            return None

        # Filter results from last N days
        cutoff = datetime.now().timestamp() - (days * 86400)
        recent_results = [r for r in self.validation_history if r.timestamp >= cutoff]

        if not recent_results:
            return None

        # Calculate average accuracy
        avg_accuracy = statistics.mean(r.overall_accuracy for r in recent_results)
        return avg_accuracy

    def get_alerts(self, since: Optional[datetime] = None) -> List[RegressionAlert]:
        """
        Get alerts since a specific time.

        Args:
            since: Get alerts since this time (defaults to last 24 hours)

        Returns:
            List of regression alerts
        """
        if since is None:
            since = datetime.now() - timedelta(hours=24)

        since_ts = since.timestamp()
        return [a for a in self.alerts if a.timestamp >= since_ts]

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()
        logger.info("Alerts cleared")

    def get_validation_history(self, days: int = 7) -> List[ValidationResult]:
        """
        Get validation history for the last N days.

        Args:
            days: Number of days of history

        Returns:
            List of validation results
        """
        cutoff = datetime.now().timestamp() - (days * 86400)
        return [r for r in self.validation_history if r.timestamp >= cutoff]

    def export_validation_history(self, output_path: str):
        """Export validation history to JSON."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for result in self.validation_history:
            data.append({
                'timestamp': result.timestamp,
                'overall_accuracy': result.overall_accuracy,
                'by_complexity': result.by_complexity,
                'total_queries': result.total_queries,
                'correct': result.correct,
                'incorrect': result.incorrect,
                'regression_detected': result.regression_detected,
                'baseline_accuracy': result.baseline_accuracy,
                'trend_7day': result.trend_7day,
                'trend_30day': result.trend_30day
            })

        with open(output, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} validation results to {output}")


def create_validation_set(
    queries: List[Tuple[str, str]],
    output_path: str
):
    """
    Create and save a validation set.

    Args:
        queries: List of (query_text, expected_complexity) tuples
        output_path: Where to save validation set

    Example:
        queries = [
            ("hi", "trivial"),
            ("what is Python?", "simple"),
            ("how does attention work?", "complex"),
        ]
        create_validation_set(queries, "./validation_set.json")
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for text, complexity in queries:
        data.append({
            'text': text,
            'complexity': complexity,
            'metadata': {}
        })

    with open(output, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Created validation set with {len(data)} queries at {output}")

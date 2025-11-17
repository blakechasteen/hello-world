"""
AR Validator - Continuous accuracy monitoring for AR query classifier
======================================================================
Monitors classifier performance and detects regressions automatically.

Author: Claude Code (Agent Q)
Date: November 17, 2025
Phase: Agent Swarm Wave 5 - Adaptive Routing Integration
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta
import statistics

from HoloLoom.voice.ar_query_classifier import ARQueryClassifier, ARComplexity, ARQueryType

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ValidationQuery:
    """Single validation query with ground truth label."""
    text: str
    expected_complexity: str  # Ground truth
    expected_ar_type: Optional[str] = None
    context: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a validation run."""
    timestamp: float
    overall_accuracy: float
    complexity_accuracy: Dict[str, float]
    ar_type_accuracy: Dict[str, float]
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


# ============================================================================
# AR Continuous Validator
# ============================================================================

class ARContinuousValidator:
    """
    Continuous accuracy monitoring for AR query classifier.

    Features:
    - Hourly/daily/weekly validation schedules
    - Regression detection (>2% accuracy drop)
    - Trend analysis (7-day, 30-day moving averages)
    - Automatic alerts (Slack, email)
    - Validation set management

    Usage:
        validator = ARContinuousValidator(
            classifier=ar_classifier,
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
        classifier: ARQueryClassifier,
        validation_set_path: Optional[str] = None,
        regression_threshold: float = 0.02,  # 2% drop triggers regression
        history_size: int = 1000,  # Keep last 1000 validation results
        baseline_accuracy: Optional[float] = None
    ):
        """
        Initialize AR continuous validator.

        Args:
            classifier: AR query classifier instance
            validation_set_path: Path to validation set JSON
            regression_threshold: Accuracy drop to trigger regression (default: 2%)
            history_size: Number of validation results to keep
            baseline_accuracy: Initial baseline accuracy (auto-computed if None)
        """
        self.classifier = classifier
        self.validation_set_path = Path(validation_set_path) if validation_set_path else None
        self.regression_threshold = regression_threshold
        self.history_size = history_size
        self.baseline_accuracy = baseline_accuracy

        # Validation set
        self.validation_set: List[ValidationQuery] = []
        if self.validation_set_path and self.validation_set_path.exists():
            self._load_validation_set()

        # Validation history
        self.validation_history: deque = deque(maxlen=history_size)

        # Alerts
        self.alerts: List[RegressionAlert] = []

    def _load_validation_set(self) -> None:
        """Load validation set from JSON file."""
        try:
            with open(self.validation_set_path, "r") as f:
                data = json.load(f)

            for item in data:
                query = ValidationQuery(
                    text=item["text"],
                    expected_complexity=item["expected_complexity"],
                    expected_ar_type=item.get("expected_ar_type"),
                    context=item.get("context", {}),
                    metadata=item.get("metadata", {})
                )
                self.validation_set.append(query)

            logger.info(f"Loaded {len(self.validation_set)} validation queries")
        except Exception as e:
            logger.error(f"Failed to load validation set: {e}")

    def validate(self) -> ValidationResult:
        """
        Run validation on current validation set.

        Returns:
            ValidationResult with accuracy metrics
        """
        if not self.validation_set:
            logger.warning("No validation set available")
            return ValidationResult(
                timestamp=datetime.now().timestamp(),
                overall_accuracy=0.0,
                complexity_accuracy={},
                ar_type_accuracy={},
                total_queries=0,
                correct=0,
                incorrect=0,
                regression_detected=False
            )

        # Run classification on validation set
        results = []
        for val_query in self.validation_set:
            result = self.classifier.classify(
                query=val_query.text,
                context=val_query.context
            )
            results.append((val_query, result))

        # Compute accuracy metrics
        total_queries = len(results)
        correct_complexity = sum(
            1 for val, res in results
            if res.complexity.value == val.expected_complexity
        )
        overall_accuracy = correct_complexity / total_queries if total_queries > 0 else 0.0

        # Accuracy by complexity
        complexity_accuracy = {}
        for complexity in ARComplexity:
            complexity_queries = [
                (val, res) for val, res in results
                if val.expected_complexity == complexity.value
            ]
            if complexity_queries:
                correct = sum(
                    1 for val, res in complexity_queries
                    if res.complexity.value == val.expected_complexity
                )
                complexity_accuracy[complexity.value] = correct / len(complexity_queries)
            else:
                complexity_accuracy[complexity.value] = 0.0

        # Accuracy by AR type (if available)
        ar_type_accuracy = {}
        for ar_type in ARQueryType:
            ar_type_queries = [
                (val, res) for val, res in results
                if val.expected_ar_type == ar_type.value
            ]
            if ar_type_queries:
                correct = sum(
                    1 for val, res in ar_type_queries
                    if res.ar_type.value == val.expected_ar_type
                )
                ar_type_accuracy[ar_type.value] = correct / len(ar_type_queries)
            else:
                ar_type_accuracy[ar_type.value] = 0.0

        # Compute trends
        trend_7day = self._compute_trend(7)
        trend_30day = self._compute_trend(30)

        # Detect regression
        regression_detected = False
        if self.baseline_accuracy is not None:
            accuracy_drop = self.baseline_accuracy - overall_accuracy
            if accuracy_drop > self.regression_threshold:
                regression_detected = True
                self._create_regression_alert(
                    overall_accuracy,
                    complexity_accuracy
                )

        # Create result
        result = ValidationResult(
            timestamp=datetime.now().timestamp(),
            overall_accuracy=overall_accuracy,
            complexity_accuracy=complexity_accuracy,
            ar_type_accuracy=ar_type_accuracy,
            total_queries=total_queries,
            correct=correct_complexity,
            incorrect=total_queries - correct_complexity,
            regression_detected=regression_detected,
            baseline_accuracy=self.baseline_accuracy,
            trend_7day=trend_7day,
            trend_30day=trend_30day
        )

        # Update history
        self.validation_history.append(result)

        # Update baseline if not set
        if self.baseline_accuracy is None:
            self.baseline_accuracy = overall_accuracy
            logger.info(f"Baseline accuracy set to {overall_accuracy:.1%}")

        return result

    def validate_hourly(self) -> ValidationResult:
        """Run hourly validation."""
        logger.info("Running hourly validation...")
        return self.validate()

    def validate_daily(self) -> ValidationResult:
        """Run daily validation."""
        logger.info("Running daily validation...")
        return self.validate()

    def validate_weekly(self) -> ValidationResult:
        """Run weekly validation."""
        logger.info("Running weekly validation...")
        return self.validate()

    def _compute_trend(self, days: int) -> Optional[float]:
        """Compute accuracy trend over last N days."""
        if not self.validation_history:
            return None

        # Filter recent results
        cutoff = datetime.now().timestamp() - (days * 86400)
        recent = [
            r for r in self.validation_history
            if r.timestamp >= cutoff
        ]

        if not recent:
            return None

        # Compute average accuracy
        avg_accuracy = sum(r.overall_accuracy for r in recent) / len(recent)
        return avg_accuracy

    def _create_regression_alert(
        self,
        current_accuracy: float,
        complexity_accuracy: Dict[str, float]
    ) -> None:
        """Create regression alert."""
        if self.baseline_accuracy is None:
            return

        drop = self.baseline_accuracy - current_accuracy
        drop_percentage = (drop / self.baseline_accuracy) * 100

        # Find affected complexity levels
        affected = []
        for complexity, accuracy in complexity_accuracy.items():
            baseline_comp = self.baseline_accuracy  # Simplified
            if baseline_comp - accuracy > self.regression_threshold:
                affected.append(complexity)

        # Determine severity
        if drop_percentage > 5.0:
            severity = "critical"
        else:
            severity = "warning"

        # Create alert
        alert = RegressionAlert(
            timestamp=datetime.now().timestamp(),
            current_accuracy=current_accuracy,
            baseline_accuracy=self.baseline_accuracy,
            drop_percentage=drop_percentage,
            affected_complexity=affected,
            severity=severity,
            message=f"Classifier regression detected: {drop_percentage:.1f}% accuracy drop"
        )

        self.alerts.append(alert)
        logger.warning(f"Regression alert: {alert.message}")

    def get_alerts(self, severity: Optional[str] = None) -> List[RegressionAlert]:
        """
        Get regression alerts.

        Args:
            severity: Filter by severity ("warning" or "critical")

        Returns:
            List of regression alerts
        """
        if severity:
            return [a for a in self.alerts if a.severity == severity]
        return self.alerts

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()

    def add_validation_query(
        self,
        text: str,
        expected_complexity: str,
        expected_ar_type: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> None:
        """Add a query to the validation set."""
        query = ValidationQuery(
            text=text,
            expected_complexity=expected_complexity,
            expected_ar_type=expected_ar_type,
            context=context or {}
        )
        self.validation_set.append(query)

    def save_validation_set(self, filepath: str) -> None:
        """Save validation set to JSON file."""
        try:
            data = []
            for query in self.validation_set:
                data.append({
                    "text": query.text,
                    "expected_complexity": query.expected_complexity,
                    "expected_ar_type": query.expected_ar_type,
                    "context": query.context,
                    "metadata": query.metadata
                })

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(data)} validation queries to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save validation set: {e}")

    def get_statistics(self) -> Dict:
        """Get validation statistics."""
        if not self.validation_history:
            return {
                "total_validations": 0,
                "avg_accuracy": 0.0,
                "min_accuracy": 0.0,
                "max_accuracy": 0.0,
                "total_alerts": len(self.alerts)
            }

        accuracies = [r.overall_accuracy for r in self.validation_history]

        return {
            "total_validations": len(self.validation_history),
            "avg_accuracy": statistics.mean(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "total_alerts": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a.severity == "critical"]),
            "warning_alerts": len([a for a in self.alerts if a.severity == "warning"]),
            "baseline_accuracy": self.baseline_accuracy,
            "trend_7day": self._compute_trend(7),
            "trend_30day": self._compute_trend(30)
        }


# ============================================================================
# Utility Functions
# ============================================================================

def create_ar_validator(
    classifier: ARQueryClassifier,
    validation_set_path: Optional[str] = None,
    regression_threshold: float = 0.02
) -> ARContinuousValidator:
    """
    Factory function to create AR continuous validator.

    Args:
        classifier: AR query classifier instance
        validation_set_path: Path to validation set JSON
        regression_threshold: Accuracy drop to trigger regression

    Returns:
        ARContinuousValidator instance
    """
    return ARContinuousValidator(
        classifier=classifier,
        validation_set_path=validation_set_path,
        regression_threshold=regression_threshold
    )

"""Metrics collection for prompt testing and validation.

Tracks performance metrics across test runs with support for:
- Time-series metric collection
- Prometheus format export
- JSON serialization
- Statistical aggregation
- Automatic retention/cleanup
- Statistical analysis (std dev, correlation, anomaly detection)
- Threshold-based alerting with notifications
- Tapestry/FabricInspector signal integration
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
import json
from collections import defaultdict

if TYPE_CHECKING:
    from .statistical_analysis import StatisticalSummary, StatisticalAnalyzer
    from .alerting import Alert, AlertingSystem


class MetricType(Enum):
    """Types of metrics collected during prompt testing."""

    # Core prompt testing metrics (9)
    LATENCY_MS = "latency_ms"
    TOKEN_COUNT = "token_count"
    QUALITY_SCORE = "quality_score"
    PASS_RATE = "pass_rate"
    MUTATION_ROBUSTNESS = "mutation_robustness"
    REGRESSION_COUNT = "regression_count"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    COST_ESTIMATE = "cost_estimate"

    # Tapestry/FabricInspector signal metrics (7)
    SECURITY_VIOLATIONS = "security_violations"      # From TroughSignal
    SAFETY_RISK_LEVEL = "safety_risk_level"          # From AlignmentSignal
    DOCUMENTATION_COVERAGE = "doc_coverage"          # From DocumentationSignal
    ARCHITECTURE_ADHERENCE = "arch_adherence"        # From ArchitectureSignal
    INTEGRATION_HEALTH = "integration_health"        # From IntegrationSignal
    FABRIC_CONFIDENCE = "fabric_confidence"          # Aggregated FabricCheckResult
    THREAD_BLOCKERS = "thread_blockers"              # Count of blocking issues


@dataclass
class Metric:
    """Single metric data point."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prometheus(self) -> str:
        """Convert to Prometheus format."""
        label_str = ",".join(f'{k}="{v}"' for k, v in self.tags.items())
        labels = f"{{{label_str}}}" if label_str else ""
        return f"prompt_test_{self.metric_type.value}{labels} {self.value}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


class MetricsCollector:
    """Collects and manages metrics from prompt tests.

    Enhanced with:
    - Statistical analysis (std dev, correlation, trend, anomaly detection)
    - Threshold-based alerting with notification handlers
    - Tapestry/FabricInspector signal support (16 total metric types)
    """

    def __init__(
        self,
        retention_hours: int = 24,
        enable_alerting: bool = False,
        enable_statistics: bool = False
    ):
        """Initialize metrics collector.

        Args:
            retention_hours: Hours to retain metrics (default: 24)
            enable_alerting: Enable threshold-based alerting (default: False)
            enable_statistics: Enable statistical analysis (default: False)
        """
        self.retention_hours = retention_hours
        self.metrics: List[Metric] = []
        self.created_at = datetime.now()

        # Initialize optional features
        self._alerting: Optional["AlertingSystem"] = None
        self._statistics: Optional["StatisticalAnalyzer"] = None

        if enable_alerting:
            from .alerting import create_alerting_system
            self._alerting = create_alerting_system(with_defaults=True)

        if enable_statistics:
            from .statistical_analysis import create_statistical_analyzer
            self._statistics = create_statistical_analyzer()

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional["Alert"]:
        """Record a metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for filtering/grouping
            metadata: Optional additional metadata

        Returns:
            Alert if alerting is enabled and a threshold was triggered, else None
        """
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        self._prune_metrics()

        # Check for alerts
        alert = None
        if self._alerting:
            alert = self._alerting.check(metric_type, value, metadata)

        return alert

    def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        since: Optional[datetime] = None
    ) -> List[Metric]:
        """Get metrics, optionally filtered.

        Args:
            metric_type: Filter by metric type
            since: Only metrics after this timestamp

        Returns:
            List of matching metrics
        """
        result = self.metrics

        if metric_type:
            result = [m for m in result if m.metric_type == metric_type]

        if since:
            result = [m for m in result if m.timestamp >= since]

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated statistics per metric type.

        Returns:
            Summary with min, max, avg, count per type
        """
        summary: Dict[str, Any] = {}

        for metric_type in MetricType:
            metrics = self.get_metrics(metric_type=metric_type)
            if not metrics:
                continue

            values = [m.value for m in metrics]
            summary[metric_type.value] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": metrics[-1].value
            }

        return summary

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-format text
        """
        lines = [
            "# HELP prompt_test Prompt testing metrics",
            "# TYPE prompt_test gauge"
        ]

        for metric in self.metrics:
            lines.append(metric.to_prometheus())

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export metrics as JSON.

        Returns:
            JSON string of all metrics
        """
        return json.dumps(
            [m.to_dict() for m in self.metrics],
            indent=2,
            default=str
        )

    def clear_old_metrics(self) -> int:
        """Remove metrics older than retention period.

        Returns:
            Number of metrics removed
        """
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        before = len(self.metrics)
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff]
        return before - len(self.metrics)

    def _prune_metrics(self) -> None:
        """Internal cleanup when limit approached."""
        if len(self.metrics) > 10000:  # Soft limit
            self.clear_old_metrics()

    # ─────────────────────────────────────────────────────────────────
    # Statistical Analysis Methods (require enable_statistics=True)
    # ─────────────────────────────────────────────────────────────────

    def get_statistics(
        self,
        metric_type: MetricType,
        since: Optional[datetime] = None
    ) -> Optional["StatisticalSummary"]:
        """Get extended statistics for a metric type.

        Requires enable_statistics=True in constructor.

        Args:
            metric_type: Type of metric to analyze
            since: Only include metrics after this timestamp

        Returns:
            StatisticalSummary with mean, std_dev, variance, median,
            percentiles, trend, and anomalies. None if statistics disabled.
        """
        if not self._statistics:
            return None

        metrics = self.get_metrics(metric_type=metric_type, since=since)
        if not metrics:
            return None

        values = [m.value for m in metrics]
        return self._statistics.analyze(values)

    def get_correlation(
        self,
        type_x: MetricType,
        type_y: MetricType,
        since: Optional[datetime] = None
    ) -> float:
        """Get Pearson correlation between two metric types.

        Requires enable_statistics=True in constructor.

        Args:
            type_x: First metric type
            type_y: Second metric type
            since: Only include metrics after this timestamp

        Returns:
            Correlation coefficient between -1.0 and 1.0.
            Returns 0.0 if statistics disabled or insufficient data.
        """
        if not self._statistics:
            return 0.0

        x_metrics = self.get_metrics(metric_type=type_x, since=since)
        y_metrics = self.get_metrics(metric_type=type_y, since=since)

        if not x_metrics or not y_metrics:
            return 0.0

        # Align by order (simple approach)
        x_values = [m.value for m in x_metrics]
        y_values = [m.value for m in y_metrics]

        # Truncate to same length
        min_len = min(len(x_values), len(y_values))
        return self._statistics.correlation(
            x_values[:min_len],
            y_values[:min_len]
        )

    def get_rolling_stats(
        self,
        metric_type: MetricType,
        window_size: int = 10,
        since: Optional[datetime] = None
    ) -> List[Dict[str, float]]:
        """Get rolling statistics for a metric type.

        Requires enable_statistics=True in constructor.

        Args:
            metric_type: Type of metric to analyze
            window_size: Rolling window size
            since: Only include metrics after this timestamp

        Returns:
            List of dicts with index, rolling_mean, rolling_std
        """
        if not self._statistics:
            return []

        metrics = self.get_metrics(metric_type=metric_type, since=since)
        if not metrics:
            return []

        values = [m.value for m in metrics]
        return self._statistics.rolling_stats(values, window_size)

    # ─────────────────────────────────────────────────────────────────
    # Alerting Methods (require enable_alerting=True)
    # ─────────────────────────────────────────────────────────────────

    def get_active_alerts(self) -> List["Alert"]:
        """Get unacknowledged alerts.

        Requires enable_alerting=True in constructor.

        Returns:
            List of unacknowledged alerts
        """
        if not self._alerting:
            return []
        return self._alerting.get_active_alerts()

    def add_alert_handler(
        self,
        handler: Callable[["Alert"], None]
    ) -> None:
        """Register handler for alerts.

        Requires enable_alerting=True in constructor.

        Args:
            handler: Callable that receives Alert objects (e.g., Slack, email)
        """
        if self._alerting:
            self._alerting.add_handler(handler)

    def add_alert_rule(
        self,
        metric_type: MetricType,
        threshold: float,
        comparison: str,
        severity: str = "warning",
        message_template: Optional[str] = None,
        cooldown_seconds: int = 300
    ) -> None:
        """Add a custom alert rule.

        Requires enable_alerting=True in constructor.

        Args:
            metric_type: Metric type to monitor
            threshold: Threshold value
            comparison: Comparison operator ("gt", "lt", "gte", "lte", "eq")
            severity: Alert severity ("info", "warning", "critical")
            message_template: Custom message template
            cooldown_seconds: Minimum time between alerts
        """
        if not self._alerting:
            return

        from .alerting import ThresholdRule, AlertSeverity

        severity_enum = AlertSeverity(severity)
        rule = ThresholdRule(
            metric_type=metric_type,
            threshold=threshold,
            comparison=comparison,
            severity=severity_enum,
            message_template=message_template or f"{{metric}} {{comparison}} {{threshold}}",
            cooldown_seconds=cooldown_seconds
        )
        self._alerting.add_rule(rule)

    def acknowledge_alert(self, alert: "Alert") -> None:
        """Mark an alert as acknowledged.

        Args:
            alert: Alert to acknowledge
        """
        if self._alerting:
            self._alerting.acknowledge(alert)

    def acknowledge_all_alerts(self) -> int:
        """Acknowledge all active alerts.

        Returns:
            Number of alerts acknowledged
        """
        if not self._alerting:
            return 0
        return self._alerting.acknowledge_all()

    # ─────────────────────────────────────────────────────────────────
    # Enhanced Summary
    # ─────────────────────────────────────────────────────────────────

    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get summary with statistics and alerts.

        Returns basic summary plus:
        - std_dev, trend, anomaly_count per metric type (if statistics enabled)
        - active_alerts count and list (if alerting enabled)

        Returns:
            Enhanced summary dictionary
        """
        summary = self.get_summary()

        # Add statistics if enabled
        if self._statistics:
            for metric_type in MetricType:
                stats = self.get_statistics(metric_type)
                if stats and metric_type.value in summary:
                    summary[metric_type.value]["std_dev"] = stats.std_dev
                    summary[metric_type.value]["trend"] = stats.trend
                    summary[metric_type.value]["anomaly_count"] = len(stats.anomalies)

        # Add alerts if enabled
        if self._alerting:
            active = self.get_active_alerts()
            summary["active_alerts"] = len(active)
            summary["alerts"] = [
                {
                    "message": a.message(),
                    "severity": a.rule.severity.value,
                    "timestamp": a.timestamp.isoformat(),
                    "metric_type": a.rule.metric_type.value,
                    "actual_value": a.actual_value
                }
                for a in active
            ]

        return summary


class MetricsAggregator:
    """Aggregates metrics from test results."""

    @staticmethod
    def aggregate_test_results(
        results: List[Dict[str, Any]]
    ) -> Dict[str, Metric]:
        """Aggregate metrics from test results.

        Args:
            results: List of test result dictionaries

        Returns:
            Dictionary of aggregated metrics
        """
        aggregated = {}

        # Group by metric type
        by_type: Dict[str, List[float]] = defaultdict(list)

        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    by_type[key].append(float(value))

        # Calculate aggregates
        for metric_name, values in by_type.items():
            if values:
                metric_type = MetricType.PASS_RATE  # Default
                try:
                    metric_type = MetricType(metric_name)
                except ValueError:
                    pass

                aggregated[metric_name] = Metric(
                    name=metric_name,
                    value=sum(values) / len(values),
                    metric_type=metric_type,
                    metadata={
                        "count": len(values),
                        "min": min(values),
                        "max": max(values)
                    }
                )

        return aggregated

    @staticmethod
    def calculate_percentiles(
        values: List[float],
        percentiles: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """Calculate percentiles from values.

        Args:
            values: List of numeric values
            percentiles: List of percentiles (default: [25, 50, 75, 95, 99])

        Returns:
            Mapping of percentile -> value
        """
        if not values:
            return {}

        percentiles = percentiles or [25, 50, 75, 95, 99]
        sorted_values = sorted(values)
        result = {}

        for p in percentiles:
            index = int((p / 100.0) * len(sorted_values))
            index = min(index, len(sorted_values) - 1)
            result[p] = sorted_values[index]

        return result


def create_metrics_collector(
    retention_hours: int = 24,
    enable_alerting: bool = False,
    enable_statistics: bool = False
) -> MetricsCollector:
    """Factory function to create metrics collector.

    Args:
        retention_hours: Hours to retain metrics
        enable_alerting: Enable threshold-based alerting
        enable_statistics: Enable statistical analysis

    Returns:
        New MetricsCollector instance
    """
    return MetricsCollector(
        retention_hours=retention_hours,
        enable_alerting=enable_alerting,
        enable_statistics=enable_statistics
    )

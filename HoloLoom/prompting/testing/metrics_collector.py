"""Metrics collection for prompt testing and validation.

Tracks performance metrics across test runs with support for:
- Time-series metric collection
- Prometheus format export
- JSON serialization
- Statistical aggregation
- Automatic retention/cleanup
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import json
from collections import defaultdict


class MetricType(Enum):
    """Types of metrics collected during prompt testing."""

    LATENCY_MS = "latency_ms"
    TOKEN_COUNT = "token_count"
    QUALITY_SCORE = "quality_score"
    PASS_RATE = "pass_rate"
    MUTATION_ROBUSTNESS = "mutation_robustness"
    REGRESSION_COUNT = "regression_count"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    COST_ESTIMATE = "cost_estimate"


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
    """Collects and manages metrics from prompt tests."""

    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector.

        Args:
            retention_hours: Hours to retain metrics (default: 24)
        """
        self.retention_hours = retention_hours
        self.metrics: List[Metric] = []
        self.created_at = datetime.now()

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for filtering/grouping
        """
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            timestamp=datetime.now()
        )
        self.metrics.append(metric)
        self._prune_metrics()

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


def create_metrics_collector(retention_hours: int = 24) -> MetricsCollector:
    """Factory function to create metrics collector.

    Args:
        retention_hours: Hours to retain metrics

    Returns:
        New MetricsCollector instance
    """
    return MetricsCollector(retention_hours=retention_hours)

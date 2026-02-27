"""
Prometheus Metrics - Registry and metric types.

Provides Prometheus-compatible metrics collection and export.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from HoloLoom.telemetry.protocol import MetricsProtocol, MetricType, HistogramBuckets
from HoloLoom.telemetry.config import MetricsConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  BASE METRIC
# ═══════════════════════════════════════════════════════════════════════════════


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._lock = Lock()
        self._values: Dict[Tuple[str, ...], Any] = {}

    def _validate_labels(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        """Validate and convert labels to tuple."""
        if set(labels.keys()) != set(self.label_names):
            expected = set(self.label_names)
            got = set(labels.keys())
            raise ValueError(
                f"Label mismatch for {self.name}: expected {expected}, got {got}"
            )
        return tuple(labels.get(k, "") for k in self.label_names)

    def _get_label_str(self, label_values: Tuple[str, ...]) -> str:
        """Convert label values to Prometheus format."""
        if not label_values:
            return ""
        pairs = [f'{k}="{v}"' for k, v in zip(self.label_names, label_values)]
        return "{" + ",".join(pairs) + "}"

    @abstractmethod
    def collect(self) -> List[str]:
        """Collect metric lines in Prometheus format."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
#  COUNTER
# ═══════════════════════════════════════════════════════════════════════════════


class Counter(Metric):
    """Monotonically increasing counter."""

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        label_key = self._validate_labels(labels or {}) if labels else ()
        with self._lock:
            current = self._values.get(label_key, 0.0)
            self._values[label_key] = current + value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        label_key = self._validate_labels(labels or {}) if labels else ()
        with self._lock:
            return self._values.get(label_key, 0.0)

    def collect(self) -> List[str]:
        """Collect in Prometheus format."""
        lines = []
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} counter")
        with self._lock:
            for label_values, value in self._values.items():
                label_str = self._get_label_str(label_values)
                lines.append(f"{self.name}{label_str} {value}")
        return lines


# ═══════════════════════════════════════════════════════════════════════════════
#  GAUGE
# ═══════════════════════════════════════════════════════════════════════════════


class Gauge(Metric):
    """Value that can go up and down."""

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge value."""
        label_key = self._validate_labels(labels or {}) if labels else ()
        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge."""
        label_key = self._validate_labels(labels or {}) if labels else ()
        with self._lock:
            current = self._values.get(label_key, 0.0)
            self._values[label_key] = current + value

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge."""
        label_key = self._validate_labels(labels or {}) if labels else ()
        with self._lock:
            current = self._values.get(label_key, 0.0)
            self._values[label_key] = current - value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        label_key = self._validate_labels(labels or {}) if labels else ()
        with self._lock:
            return self._values.get(label_key, 0.0)

    def collect(self) -> List[str]:
        """Collect in Prometheus format."""
        lines = []
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} gauge")
        with self._lock:
            for label_values, value in self._values.items():
                label_str = self._get_label_str(label_values)
                lines.append(f"{self.name}{label_str} {value}")
        return lines


# ═══════════════════════════════════════════════════════════════════════════════
#  HISTOGRAM
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HistogramData:
    """Data for a single histogram."""

    buckets: List[Tuple[float, int]] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0


class Histogram(Metric):
    """Bucketed observation histogram."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        super().__init__(name, description, labels)
        self.bucket_boundaries = buckets or [
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        ]
        self._histograms: Dict[Tuple[str, ...], HistogramData] = {}

    def observe(
        self, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Observe a value."""
        label_key = self._validate_labels(labels or {}) if labels else ()
        with self._lock:
            if label_key not in self._histograms:
                self._histograms[label_key] = HistogramData(
                    buckets=[(b, 0) for b in self.bucket_boundaries]
                )
            hist = self._histograms[label_key]
            hist.sum_value += value
            hist.count += 1
            # Update buckets
            new_buckets = []
            for boundary, count in hist.buckets:
                if value <= boundary:
                    count += 1
                new_buckets.append((boundary, count))
            hist.buckets = new_buckets

    def time(self, labels: Optional[Dict[str, str]] = None):
        """Context manager to time a block of code."""
        return _HistogramTimer(self, labels)

    def collect(self) -> List[str]:
        """Collect in Prometheus format."""
        lines = []
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} histogram")
        with self._lock:
            for label_values, hist in self._histograms.items():
                label_str = self._get_label_str(label_values)
                base_labels = label_str.rstrip("}") if label_str else ""

                # Bucket lines
                cumulative = 0
                for boundary, count in hist.buckets:
                    cumulative += count
                    if base_labels:
                        lines.append(
                            f'{self.name}_bucket{base_labels},le="{boundary}"}} {cumulative}'
                        )
                    else:
                        lines.append(
                            f'{self.name}_bucket{{le="{boundary}"}} {cumulative}'
                        )

                # +Inf bucket
                if base_labels:
                    lines.append(
                        f'{self.name}_bucket{base_labels},le="+Inf"}} {hist.count}'
                    )
                else:
                    lines.append(f'{self.name}_bucket{{le="+Inf"}} {hist.count}')

                # Sum and count
                lines.append(f"{self.name}_sum{label_str} {hist.sum_value}")
                lines.append(f"{self.name}_count{label_str} {hist.count}")

        return lines


class _HistogramTimer:
    """Timer context manager for histograms."""

    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]]):
        self.histogram = histogram
        self.labels = labels
        self.start_time: float = 0

    def __enter__(self) -> _HistogramTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        duration = time.perf_counter() - self.start_time
        self.histogram.observe(duration, self.labels)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SummaryData:
    """Data for a single summary."""

    observations: List[float] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0
    max_age_seconds: float = 600.0  # 10 minutes


class Summary(Metric):
    """Streaming quantile summary."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        max_observations: int = 1000,
    ):
        super().__init__(name, description, labels)
        self.quantiles = quantiles or [0.5, 0.9, 0.99]
        self.max_observations = max_observations
        self._summaries: Dict[Tuple[str, ...], SummaryData] = {}

    def observe(
        self, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Observe a value."""
        label_key = self._validate_labels(labels or {}) if labels else ()
        with self._lock:
            if label_key not in self._summaries:
                self._summaries[label_key] = SummaryData()
            summ = self._summaries[label_key]
            summ.sum_value += value
            summ.count += 1
            summ.observations.append(value)
            # Limit observations
            if len(summ.observations) > self.max_observations:
                summ.observations = summ.observations[-self.max_observations:]

    def collect(self) -> List[str]:
        """Collect in Prometheus format."""
        lines = []
        lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} summary")
        with self._lock:
            for label_values, summ in self._summaries.items():
                label_str = self._get_label_str(label_values)
                base_labels = label_str.rstrip("}") if label_str else ""

                # Calculate quantiles
                if summ.observations:
                    sorted_obs = sorted(summ.observations)
                    for q in self.quantiles:
                        idx = int(q * len(sorted_obs))
                        idx = min(idx, len(sorted_obs) - 1)
                        value = sorted_obs[idx]
                        if base_labels:
                            lines.append(
                                f'{self.name}{base_labels},quantile="{q}"}} {value}'
                            )
                        else:
                            lines.append(f'{self.name}{{quantile="{q}"}} {value}')

                # Sum and count
                lines.append(f"{self.name}_sum{label_str} {summ.sum_value}")
                lines.append(f"{self.name}_count{label_str} {summ.count}")

        return lines


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


class PrometheusRegistry(MetricsProtocol):
    """Registry for all metrics."""

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self._metrics: Dict[str, Metric] = {}
        self._lock = Lock()
        self._prefix = self.config.prefix

    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix."""
        if self._prefix:
            return f"{self._prefix}_{name}"
        return name

    def register(self, metric: Metric) -> Metric:
        """Register a metric."""
        with self._lock:
            if metric.name in self._metrics:
                return self._metrics[metric.name]
            self._metrics[metric.name] = metric
            return metric

    def get_counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Get or create a counter."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Counter):
                    raise TypeError(f"{full_name} is not a Counter")
                return metric
            counter = Counter(full_name, description, labels)
            self._metrics[full_name] = counter
            return counter

    def get_gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Get or create a gauge."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Gauge):
                    raise TypeError(f"{full_name} is not a Gauge")
                return metric
            gauge = Gauge(full_name, description, labels)
            self._metrics[full_name] = gauge
            return gauge

    def get_histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Histogram):
                    raise TypeError(f"{full_name} is not a Histogram")
                return metric
            histogram = Histogram(full_name, description, labels, buckets)
            self._metrics[full_name] = histogram
            return histogram

    def get_summary(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
    ) -> Summary:
        """Get or create a summary."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Summary):
                    raise TypeError(f"{full_name} is not a Summary")
                return metric
            summary = Summary(full_name, description, labels, quantiles)
            self._metrics[full_name] = summary
            return summary

    # MetricsProtocol implementation

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Increment a counter."""
        label_names = list(labels.keys()) if labels else None
        counter = self.get_counter(name, description, label_names)
        counter.inc(value, labels)

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Set a gauge value."""
        label_names = list(labels.keys()) if labels else None
        gauge = self.get_gauge(name, description, label_names)
        gauge.set(value, labels)

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
        description: str = "",
    ) -> None:
        """Observe a histogram value."""
        label_names = list(labels.keys()) if labels else None
        histogram = self.get_histogram(name, description, label_names, buckets)
        histogram.observe(value, labels)

    def summary(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        quantiles: Optional[List[float]] = None,
        description: str = "",
    ) -> None:
        """Observe a summary value."""
        label_names = list(labels.keys()) if labels else None
        summary = self.get_summary(name, description, label_names, quantiles)
        summary.observe(value, labels)

    def export(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        with self._lock:
            for metric in self._metrics.values():
                lines.extend(metric.collect())
                lines.append("")  # Empty line between metrics
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


_global_registry: Optional[PrometheusRegistry] = None


def create_registry(config: Optional[MetricsConfig] = None) -> PrometheusRegistry:
    """Create and set the global registry."""
    global _global_registry
    _global_registry = PrometheusRegistry(config)
    return _global_registry


def get_registry() -> PrometheusRegistry:
    """Get the global registry, creating one if needed."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PrometheusRegistry()
    return _global_registry


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def counter(
    name: str,
    value: Optional[float] = None,
    labels: Optional[Any] = None,
    description: str = "",
) -> Optional[Counter] | None:
    """Increment a counter or get/create one if value is None."""
    registry = get_registry()
    if value is None:
        return registry.get_counter(name, description, labels)
    registry.counter(name, value, labels, description)
    return None


def gauge(
    name: str,
    value: Optional[float] = None,
    labels: Optional[Any] = None,
    description: str = "",
) -> Optional[Gauge] | None:
    """Set a gauge or get/create one if value is None."""
    registry = get_registry()
    if value is None:
        return registry.get_gauge(name, description, labels)
    registry.gauge(name, value, labels, description)
    return None


def histogram(
    name: str,
    value: Optional[float] = None,
    labels: Optional[Any] = None,
    buckets: Optional[List[float]] = None,
    description: str = "",
) -> Optional[Histogram] | None:
    """Observe a histogram value or get/create one if value is None."""
    registry = get_registry()
    if value is None:
        return registry.get_histogram(name, description, labels, buckets)
    registry.histogram(name, value, labels, buckets, description)
    return None


def summary(
    name: str,
    value: Optional[float] = None,
    labels: Optional[Any] = None,
    quantiles: Optional[List[float]] = None,
    description: str = "",
) -> Optional[Summary] | None:
    """Observe a summary value or get/create one if value is None."""
    registry = get_registry()
    if value is None:
        return registry.get_summary(name, description, labels, quantiles)
    registry.summary(name, value, labels, quantiles, description)
    return None

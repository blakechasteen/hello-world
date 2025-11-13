"""
Prometheus Metrics for Promptly Matrix Bot

Provides:
- Request/response metrics (count, latency, errors)
- Circuit breaker metrics
- Rate limiting metrics
- HoloLoom weaving performance
- Database query metrics

Usage:
    from bot.metrics import metrics, track_request

    # Track request
    with track_request("api_query"):
        # Process request
        pass

    # Export metrics for Prometheus
    metrics_output = metrics.export()
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class Counter:
    """Simple counter metric"""
    name: str
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: int = 0

    def inc(self, amount: int = 1):
        """Increment counter"""
        self.value += amount

    def reset(self):
        """Reset counter to 0"""
        self.value = 0

    def export(self) -> str:
        """Export in Prometheus format"""
        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        metric_name = f"{self.name}{{{label_str}}}" if label_str else self.name
        return f"{metric_name} {self.value}"


@dataclass
class Gauge:
    """Gauge metric (can go up or down)"""
    name: str
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0

    def set(self, value: float):
        """Set gauge value"""
        self.value = value

    def inc(self, amount: float = 1.0):
        """Increment gauge"""
        self.value += amount

    def dec(self, amount: float = 1.0):
        """Decrement gauge"""
        self.value -= amount

    def export(self) -> str:
        """Export in Prometheus format"""
        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        metric_name = f"{self.name}{{{label_str}}}" if label_str else self.name
        return f"{metric_name} {self.value}"


@dataclass
class Histogram:
    """Histogram metric for tracking distributions"""
    name: str
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    observations: deque = field(default_factory=lambda: deque(maxlen=10000))
    bucket_counts: Dict[float, int] = field(default_factory=dict)
    sum: float = 0.0
    count: int = 0

    def __post_init__(self):
        # Initialize bucket counts
        for bucket in self.buckets:
            self.bucket_counts[bucket] = 0
        self.bucket_counts[float('inf')] = 0

    def observe(self, value: float):
        """Record an observation"""
        self.observations.append(value)
        self.sum += value
        self.count += 1

        # Increment bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
        self.bucket_counts[float('inf')] += 1

    def export(self) -> str:
        """Export in Prometheus format"""
        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        base_name = f"{self.name}{{{label_str}}}" if label_str else self.name

        lines = []

        # Buckets
        for bucket in sorted(self.bucket_counts.keys()):
            bucket_label = f'{base_name}_bucket{{le="{bucket}"}}'
            lines.append(f"{bucket_label} {self.bucket_counts[bucket]}")

        # Sum and count
        lines.append(f"{base_name}_sum {self.sum}")
        lines.append(f"{base_name}_count {self.count}")

        return "\n".join(lines)

    def percentile(self, p: float) -> Optional[float]:
        """Calculate percentile from observations"""
        if not self.observations:
            return None

        sorted_obs = sorted(self.observations)
        index = int(len(sorted_obs) * p)
        return sorted_obs[min(index, len(sorted_obs) - 1)]


class MetricsRegistry:
    """
    Central registry for all metrics

    Usage:
        metrics = MetricsRegistry()

        # Create metrics
        requests = metrics.counter("requests_total", "Total requests", {"endpoint": "query"})
        latency = metrics.histogram("request_latency_seconds", "Request latency")

        # Use metrics
        requests.inc()
        latency.observe(0.123)

        # Export all metrics
        output = metrics.export()
    """

    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}

    def counter(self, name: str, help: str, labels: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create counter metric"""
        key = self._make_key(name, labels)

        if key not in self.counters:
            self.counters[key] = Counter(name=name, help=help, labels=labels or {})

        return self.counters[key]

    def gauge(self, name: str, help: str, labels: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create gauge metric"""
        key = self._make_key(name, labels)

        if key not in self.gauges:
            self.gauges[key] = Gauge(name=name, help=help, labels=labels or {})

        return self.gauges[key]

    def histogram(self, name: str, help: str, labels: Optional[Dict[str, str]] = None) -> Histogram:
        """Get or create histogram metric"""
        key = self._make_key(name, labels)

        if key not in self.histograms:
            self.histograms[key] = Histogram(name=name, help=help, labels=labels or {})

        return self.histograms[key]

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric"""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def export(self) -> str:
        """Export all metrics in Prometheus format"""
        lines = []

        # Export counters
        for counter in self.counters.values():
            lines.append(f"# HELP {counter.name} {counter.help}")
            lines.append(f"# TYPE {counter.name} counter")
            lines.append(counter.export())
            lines.append("")

        # Export gauges
        for gauge in self.gauges.values():
            lines.append(f"# HELP {gauge.name} {gauge.help}")
            lines.append(f"# TYPE {gauge.name} gauge")
            lines.append(gauge.export())
            lines.append("")

        # Export histograms
        for histogram in self.histograms.values():
            lines.append(f"# HELP {histogram.name} {histogram.help}")
            lines.append(f"# TYPE {histogram.name} histogram")
            lines.append(histogram.export())
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics"""
        return {
            "counters": {
                key: counter.value
                for key, counter in self.counters.items()
            },
            "gauges": {
                key: gauge.value
                for key, gauge in self.gauges.items()
            },
            "histograms": {
                key: {
                    "count": hist.count,
                    "sum": hist.sum,
                    "avg": hist.sum / hist.count if hist.count > 0 else 0,
                    "p50": hist.percentile(0.5),
                    "p95": hist.percentile(0.95),
                    "p99": hist.percentile(0.99)
                }
                for key, hist in self.histograms.items()
            }
        }


# Global metrics registry
metrics = MetricsRegistry()


# Pre-defined metrics

# Request metrics
http_requests_total = metrics.counter(
    "http_requests_total",
    "Total HTTP requests",
    {"method": "POST", "endpoint": "/api/query"}
)

http_request_duration_seconds = metrics.histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds"
)

http_errors_total = metrics.counter(
    "http_errors_total",
    "Total HTTP errors"
)

# HoloLoom weaving metrics
hololoom_queries_total = metrics.counter(
    "hololoom_queries_total",
    "Total HoloLoom queries"
)

hololoom_query_duration_seconds = metrics.histogram(
    "hololoom_query_duration_seconds",
    "HoloLoom query latency in seconds"
)

hololoom_confidence = metrics.histogram(
    "hololoom_confidence",
    "HoloLoom response confidence"
)

hololoom_cache_hits_total = metrics.counter(
    "hololoom_cache_hits_total",
    "Total HoloLoom cache hits"
)

hololoom_cache_misses_total = metrics.counter(
    "hololoom_cache_misses_total",
    "Total HoloLoom cache misses"
)

# Circuit breaker metrics
circuit_breaker_open = metrics.gauge(
    "circuit_breaker_open",
    "Circuit breaker is open (1) or closed (0)"
)

circuit_breaker_failures_total = metrics.counter(
    "circuit_breaker_failures_total",
    "Total circuit breaker failures"
)

# Rate limiting metrics
rate_limit_requests_total = metrics.counter(
    "rate_limit_requests_total",
    "Total rate limit requests"
)

rate_limit_denied_total = metrics.counter(
    "rate_limit_denied_total",
    "Total rate limit denials"
)

# Database metrics
db_queries_total = metrics.counter(
    "db_queries_total",
    "Total database queries"
)

db_query_duration_seconds = metrics.histogram(
    "db_query_duration_seconds",
    "Database query latency in seconds"
)

# GitHub integration metrics
github_pr_created_total = metrics.counter(
    "github_pr_created_total",
    "Total GitHub PRs created"
)

github_pr_reviewed_total = metrics.counter(
    "github_pr_reviewed_total",
    "Total GitHub PRs reviewed"
)

github_api_errors_total = metrics.counter(
    "github_api_errors_total",
    "Total GitHub API errors"
)


@contextmanager
def track_request(endpoint: str):
    """
    Context manager to track request metrics

    Usage:
        with track_request("api_query"):
            # Process request
            pass
    """
    start_time = time.time()

    try:
        yield
    except Exception as e:
        # Track error
        http_errors_total.inc()
        logger.error(f"Request error in {endpoint}: {e}")
        raise
    finally:
        # Track duration
        duration = time.time() - start_time
        http_request_duration_seconds.observe(duration)

        # Track request count
        request_counter = metrics.counter(
            "http_requests_total",
            "Total HTTP requests",
            {"endpoint": endpoint}
        )
        request_counter.inc()


@contextmanager
def track_hololoom_query():
    """
    Context manager to track HoloLoom query metrics

    Usage:
        with track_hololoom_query():
            result = await orchestrator.weave(query)
            # Result available in context
    """
    start_time = time.time()

    hololoom_queries_total.inc()

    try:
        yield
    finally:
        duration = time.time() - start_time
        hololoom_query_duration_seconds.observe(duration)


def track_confidence(confidence: float):
    """Track HoloLoom confidence score"""
    hololoom_confidence.observe(confidence)


def track_cache_hit():
    """Track HoloLoom cache hit"""
    hololoom_cache_hits_total.inc()


def track_cache_miss():
    """Track HoloLoom cache miss"""
    hololoom_cache_misses_total.inc()


def track_circuit_breaker_state(name: str, is_open: bool):
    """Track circuit breaker state"""
    breaker_gauge = metrics.gauge(
        "circuit_breaker_open",
        "Circuit breaker is open (1) or closed (0)",
        {"breaker": name}
    )
    breaker_gauge.set(1.0 if is_open else 0.0)


def track_rate_limit_denial():
    """Track rate limit denial"""
    rate_limit_denied_total.inc()

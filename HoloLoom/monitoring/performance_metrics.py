"""
HoloLoom Performance Metrics Collection
========================================

Prometheus-compatible metrics collection for HoloLoom system performance.

Tracks:
- System metrics (CPU, memory, connections)
- Request metrics (count, latency, errors)
- Application metrics (weaving, cache, database)
- Business metrics (queries, strategies, quality)

Created: 2025-11-16
Integration: Prometheus + Grafana compatible
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import threading

# Try to import psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for graceful degradation
    class Counter:
        def __init__(self, *args, **kwargs):
            self._value = 0
        def labels(self, **kwargs):
            return self
        def inc(self, amount=1):
            self._value += amount

    class Histogram:
        def __init__(self, *args, **kwargs):
            self._observations = []
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            self._observations.append(value)

    class Gauge:
        def __init__(self, *args, **kwargs):
            self._value = 0
        def labels(self, **kwargs):
            return self
        def set(self, value):
            self._value = value
        def inc(self, amount=1):
            self._value += amount
        def dec(self, amount=1):
            self._value -= amount

    Summary = Histogram


# ============================================================================
# System Metrics
# ============================================================================

# System resource usage
system_cpu_percent = Gauge(
    'hololoom_system_cpu_percent',
    'CPU usage percentage'
)

system_memory_bytes = Gauge(
    'hololoom_system_memory_bytes',
    'Memory usage in bytes'
)

system_memory_percent = Gauge(
    'hololoom_system_memory_percent',
    'Memory usage percentage'
)

active_websocket_connections = Gauge(
    'hololoom_active_websocket_connections',
    'Number of active WebSocket connections'
)


# ============================================================================
# Request Metrics
# ============================================================================

# Request count by endpoint, method, and status
request_count = Counter(
    'hololoom_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

# Request latency by endpoint
request_latency = Histogram(
    'hololoom_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Error count by endpoint and error type
error_count = Counter(
    'hololoom_errors_total',
    'Total number of errors',
    ['endpoint', 'error_type']
)


# ============================================================================
# Application Metrics
# ============================================================================

# Weaving latency by complexity level
weaving_latency = Histogram(
    'hololoom_weaving_duration_seconds',
    'Weaving latency by complexity level',
    ['complexity'],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Cache hit rate (0.0 - 1.0)
cache_hit_rate = Gauge(
    'hololoom_cache_hit_rate',
    'Cache hit rate',
    ['cache_type']
)

# Cache hits/misses
cache_operations = Counter(
    'hololoom_cache_operations_total',
    'Cache operations',
    ['cache_type', 'operation']  # operation: hit, miss
)

# Database query time
database_query_duration = Histogram(
    'hololoom_database_query_duration_seconds',
    'Database query duration',
    ['query_type'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# Background task queue size
background_task_queue_size = Gauge(
    'hololoom_background_task_queue_size',
    'Number of pending background tasks'
)


# ============================================================================
# Business Metrics
# ============================================================================

# Total queries processed
queries_processed = Counter(
    'hololoom_queries_processed_total',
    'Total queries processed',
    ['complexity']
)

# Recursive reasoning iterations
recursive_iterations = Histogram(
    'hololoom_recursive_iterations',
    'Number of recursive reasoning iterations',
    buckets=[1, 2, 3, 5, 10, 20, 50]
)

# Quality improvement (delta)
quality_improvement = Histogram(
    'hololoom_quality_improvement',
    'Quality improvement from refinement',
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
)

# Strategy usage
strategy_usage = Counter(
    'hololoom_strategy_usage_total',
    'Strategy usage count',
    ['strategy']  # refine, critique, verify, elegance, hofstadter
)

# Skill executions
skill_executions = Counter(
    'hololoom_skill_executions_total',
    'Skill execution count',
    ['skill_name', 'success']
)

# Confidence scores
confidence_scores = Histogram(
    'hololoom_confidence_scores',
    'Query confidence scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


# ============================================================================
# Metrics Collector
# ============================================================================

@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    active_connections: int = 0
    request_rate: float = 0.0  # requests per second
    error_rate: float = 0.0  # errors per request
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    cache_hit_rates: Dict[str, float] = field(default_factory=dict)
    total_queries: int = 0


class PerformanceMetricsCollector:
    """
    Collects and aggregates performance metrics.

    Thread-safe for concurrent metric recording.
    Provides snapshot API for dashboard queries.
    """

    def __init__(self):
        self.start_time = time.time()
        self.lock = threading.Lock()

        # In-memory storage for percentile calculations
        self.request_latencies: Dict[str, List[float]] = defaultdict(list)
        self.weaving_latencies: Dict[str, List[float]] = defaultdict(list)

        # Cache statistics
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'hits': 0, 'misses': 0}
        )

        # Process handle for system metrics
        if PSUTIL_AVAILABLE:
            try:
                self.process = psutil.Process()
            except Exception:
                self.process = None
        else:
            self.process = None

    def record_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        duration_seconds: float
    ):
        """Record a request."""
        with self.lock:
            # Prometheus metrics
            request_count.labels(
                endpoint=endpoint,
                method=method,
                status=str(status)
            ).inc()

            request_latency.labels(endpoint=endpoint).observe(duration_seconds)

            # Store for percentile calculations
            self.request_latencies[endpoint].append(duration_seconds)

            # Keep only recent data (last 1000 requests per endpoint)
            if len(self.request_latencies[endpoint]) > 1000:
                self.request_latencies[endpoint] = self.request_latencies[endpoint][-1000:]

    def record_error(self, endpoint: str, error_type: str):
        """Record an error."""
        error_count.labels(endpoint=endpoint, error_type=error_type).inc()

    def record_weaving(self, complexity: str, duration_seconds: float, confidence: float):
        """Record a weaving operation."""
        with self.lock:
            weaving_latency.labels(complexity=complexity).observe(duration_seconds)
            queries_processed.labels(complexity=complexity).inc()
            confidence_scores.observe(confidence)

            # Store for percentile calculations
            self.weaving_latencies[complexity].append(duration_seconds)
            if len(self.weaving_latencies[complexity]) > 1000:
                self.weaving_latencies[complexity] = self.weaving_latencies[complexity][-1000:]

    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record a cache hit or miss."""
        with self.lock:
            operation = 'hit' if hit else 'miss'
            cache_operations.labels(cache_type=cache_type, operation=operation).inc()

            # Update statistics
            if hit:
                self.cache_stats[cache_type]['hits'] += 1
            else:
                self.cache_stats[cache_type]['misses'] += 1

            # Update hit rate gauge
            stats = self.cache_stats[cache_type]
            total = stats['hits'] + stats['misses']
            hit_rate = stats['hits'] / total if total > 0 else 0.0
            cache_hit_rate.labels(cache_type=cache_type).set(hit_rate)

    def record_database_query(self, query_type: str, duration_seconds: float):
        """Record a database query."""
        database_query_duration.labels(query_type=query_type).observe(duration_seconds)

    def record_recursive_refinement(
        self,
        iterations: int,
        strategy: str,
        quality_delta: float
    ):
        """Record recursive refinement metrics."""
        recursive_iterations.observe(iterations)
        strategy_usage.labels(strategy=strategy).inc()
        quality_improvement.observe(quality_delta)

    def record_skill_execution(self, skill_name: str, success: bool):
        """Record skill execution."""
        skill_executions.labels(
            skill_name=skill_name,
            success=str(success).lower()
        ).inc()

    def update_system_metrics(self):
        """Update system resource metrics."""
        if not PSUTIL_AVAILABLE or self.process is None:
            return

        try:
            # CPU usage
            cpu = psutil.cpu_percent(interval=0.1)
            system_cpu_percent.set(cpu)

            # Memory usage
            mem_info = self.process.memory_info()
            system_memory_bytes.set(mem_info.rss)

            mem_percent = self.process.memory_percent()
            system_memory_percent.set(mem_percent)
        except Exception:
            pass

    def set_active_connections(self, count: int):
        """Update active WebSocket connections count."""
        active_websocket_connections.set(count)

    def set_background_queue_size(self, size: int):
        """Update background task queue size."""
        background_task_queue_size.set(size)

    def get_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        with self.lock:
            snapshot = PerformanceSnapshot()

            # System metrics
            if PSUTIL_AVAILABLE and self.process:
                try:
                    snapshot.cpu_percent = psutil.cpu_percent(interval=0)
                    mem_info = self.process.memory_info()
                    snapshot.memory_mb = mem_info.rss / 1024 / 1024
                    snapshot.memory_percent = self.process.memory_percent()
                except Exception:
                    pass

            # Calculate percentiles for request latency
            all_latencies = []
            for latencies in self.request_latencies.values():
                all_latencies.extend(latencies)

            if all_latencies:
                sorted_latencies = sorted(all_latencies)
                n = len(sorted_latencies)
                snapshot.p50_latency_ms = sorted_latencies[int(n * 0.50)] * 1000
                snapshot.p95_latency_ms = sorted_latencies[int(n * 0.95)] * 1000
                snapshot.p99_latency_ms = sorted_latencies[int(n * 0.99)] * 1000

            # Cache hit rates
            for cache_type, stats in self.cache_stats.items():
                total = stats['hits'] + stats['misses']
                if total > 0:
                    snapshot.cache_hit_rates[cache_type] = stats['hits'] / total

            # Active connections (read from gauge)
            try:
                snapshot.active_connections = int(active_websocket_connections._value.get())
            except Exception:
                snapshot.active_connections = 0

            # Total queries (sum across all complexity levels)
            # Note: Prometheus Counter doesn't expose value directly in all versions
            # This is a limitation - use snapshot for display only

            return snapshot

    def get_percentile(self, metric_name: str, percentile: float) -> float:
        """
        Get percentile value for a metric.

        Args:
            metric_name: "request_latency" or "weaving_latency"
            percentile: 0.0 - 1.0 (e.g., 0.95 for p95)

        Returns:
            Percentile value in seconds
        """
        with self.lock:
            if metric_name == "request_latency":
                all_values = []
                for values in self.request_latencies.values():
                    all_values.extend(values)
            elif metric_name == "weaving_latency":
                all_values = []
                for values in self.weaving_latencies.values():
                    all_values.extend(values)
            else:
                return 0.0

            if not all_values:
                return 0.0

            sorted_values = sorted(all_values)
            index = int(len(sorted_values) * percentile)
            return sorted_values[min(index, len(sorted_values) - 1)]

    def calculate_rate(self, window_seconds: int = 60) -> float:
        """
        Calculate request rate (requests per second).

        Args:
            window_seconds: Time window for rate calculation

        Returns:
            Requests per second
        """
        # This is a simplified calculation
        # In production, use a proper time-series database
        with self.lock:
            total_requests = sum(len(latencies) for latencies in self.request_latencies.values())
            uptime = time.time() - self.start_time
            window = min(window_seconds, uptime)

            if window > 0:
                return total_requests / window
            return 0.0


# ============================================================================
# Singleton Instance
# ============================================================================

_global_metrics_collector: Optional[PerformanceMetricsCollector] = None


def get_metrics_collector() -> PerformanceMetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = PerformanceMetricsCollector()
    return _global_metrics_collector


def reset_metrics_collector():
    """Reset global metrics collector."""
    global _global_metrics_collector
    _global_metrics_collector = PerformanceMetricsCollector()


# ============================================================================
# Anomaly Detection Integration (2025-11-16)
# ============================================================================

# Try to import anomaly detection
try:
    from HoloLoom.monitoring.anomaly_detection import (
        get_statistical_detector,
        get_anomaly_store,
        Anomaly,
        AnomalySeverity,
        DetectionMethod
    )
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False


def check_metric_anomalies(snapshot: PerformanceSnapshot) -> List[Any]:
    """
    Check current metrics for anomalies.

    Args:
        snapshot: Current performance snapshot

    Returns:
        List of Anomaly objects detected
    """
    if not ANOMALY_DETECTION_AVAILABLE:
        return []

    detector = get_statistical_detector()
    store = get_anomaly_store()
    anomalies = []

    # Define metric checks with thresholds
    metric_checks = [
        # System metrics
        ("cpu_percent", snapshot.cpu_percent, 80.0),  # Alert if >80%
        ("memory_percent", snapshot.memory_percent, 80.0),  # Alert if >80%
        ("p95_latency_ms", snapshot.p95_latency_ms, None),  # Use detector default
        ("p99_latency_ms", snapshot.p99_latency_ms, None),  # Use detector default
    ]

    # Add cache metrics
    for cache_type, hit_rate in snapshot.cache_hit_rates.items():
        metric_checks.append((f"cache_hit_rate_{cache_type}", hit_rate, None))

    # Check each metric
    for metric_name, value, threshold in metric_checks:
        if value is None or value == 0.0:
            continue

        # Use Z-score detection
        result = detector.detect_zscore(metric_name, value)

        # Create anomaly if detected and severity is medium or higher
        if result.is_anomaly and result.severity in [
            AnomalySeverity.MEDIUM,
            AnomalySeverity.HIGH,
            AnomalySeverity.CRITICAL
        ]:
            anomaly = Anomaly(
                timestamp=datetime.now(),
                metric_name=metric_name,
                metric_value=value,
                expected_min=result.expected_min,
                expected_max=result.expected_max,
                anomaly_score=result.score,
                severity=result.severity,
                detection_method=result.method,
                metadata=result.metadata
            )

            # Store anomaly
            store.store(anomaly)
            anomalies.append(anomaly)

    return anomalies


def get_current_metrics() -> Dict[str, float]:
    """
    Get current metric values for anomaly detection.

    Returns:
        Dictionary of metric_name -> value
    """
    collector = get_metrics_collector()
    snapshot = collector.get_snapshot()

    metrics = {
        "cpu_percent": snapshot.cpu_percent,
        "memory_mb": snapshot.memory_mb,
        "memory_percent": snapshot.memory_percent,
        "active_connections": float(snapshot.active_connections),
        "p50_latency_ms": snapshot.p50_latency_ms,
        "p95_latency_ms": snapshot.p95_latency_ms,
        "p99_latency_ms": snapshot.p99_latency_ms,
    }

    # Add cache hit rates
    for cache_type, hit_rate in snapshot.cache_hit_rates.items():
        metrics[f"cache_hit_rate_{cache_type}"] = hit_rate

    return metrics

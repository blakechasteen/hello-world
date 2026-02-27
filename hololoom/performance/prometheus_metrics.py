"""
Prometheus Metrics for HoloLoom Production Monitoring
=====================================================
Comprehensive Prometheus metrics for production dashboards.

Metrics Categories:

1. Query Counters:
   - hololoom_queries_total: Total queries processed
   - hololoom_cache_hits_total: Cache hits (query_cache, semantic_cache)
   - hololoom_cache_misses_total: Cache misses

2. Latency Histograms:
   - hololoom_weaving_duration_seconds: Total weaving cycle duration
   - hololoom_stage_duration_seconds: Individual stage durations
   - hololoom_tool_execution_duration_seconds: Tool execution latency

3. Gauges:
   - hololoom_active_threads: Active memory threads by pattern
   - hololoom_confidence_score: Last query confidence by tool
   - hololoom_memory_shards_count: Total memory shards available
   - hololoom_parallel_speedup: Parallel execution speedup factor

4. Feature Metrics:
   - hololoom_motifs_detected_total: Total motifs detected
   - hololoom_reflection_updates_total: Reflection loop updates

Usage:
    from hololoom.performance.prometheus_metrics import metrics, start_metrics_server

    # Track query metrics
    metrics.track_query(pattern='FUSED', complexity='FAST', duration=0.125)

    # Track stage timing
    metrics.track_stage('feature_extraction', 45.5)

    # Track cache hits
    metrics.track_cache_hit('query_cache')

    # Start metrics endpoint
    start_metrics_server(port=8001)

Prometheus queries:
    # 95th percentile latency by complexity
    histogram_quantile(0.95, sum(rate(hololoom_weaving_duration_seconds_bucket[5m])) by (complexity, le))

    # Cache hit rate
    rate(hololoom_cache_hits_total[5m]) / (rate(hololoom_cache_hits_total[5m]) + rate(hololoom_cache_misses_total[5m]))

    # Queries per second by tool
    sum(rate(hololoom_queries_total[1m])) by (tool_used)

    # Average confidence by tool
    avg(hololoom_confidence_score) by (tool_used)
"""

import time
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, start_http_server, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not installed - metrics disabled")
    PROMETHEUS_AVAILABLE = False


# ============================================================================
# Metric Definitions - Query Counters
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Query counters with labels for complexity, pattern card, and tool
    queries_total = Counter(
        'hololoom_queries_total',
        'Total queries processed',
        ['complexity', 'pattern_card', 'tool_used']
    )

    # Cache hit/miss tracking
    cache_hits_total = Counter(
        'hololoom_cache_hits_total',
        'Total cache hits',
        ['cache_type']  # query_cache, semantic_cache, etc.
    )

    cache_misses_total = Counter(
        'hololoom_cache_misses_total',
        'Total cache misses',
        ['cache_type']
    )

    errors_total = Counter(
        'hololoom_errors_total',
        'Total errors',
        ['error_type', 'stage']
    )

    motifs_detected_total = Counter(
        'hololoom_motifs_detected_total',
        'Total motifs detected',
        ['motif_type']
    )

    reflection_updates_total = Counter(
        'hololoom_reflection_updates_total',
        'Reflection buffer updates',
        ['signal_type']  # learning_signal types
    )

    # ========================================================================
    # Latency Histograms
    # ========================================================================

    # Total weaving cycle latency with buckets for SLO targets
    # Targets: p50=100ms, p95=150ms, p99=300ms
    weaving_duration = Histogram(
        'hololoom_weaving_duration_seconds',
        'Total weaving cycle duration',
        ['complexity', 'pattern_card'],
        buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0)
    )

    # Individual stage durations
    stage_duration = Histogram(
        'hololoom_stage_duration_seconds',
        'Individual stage execution duration',
        ['stage_name'],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
    )

    # Tool execution latency
    tool_execution_duration = Histogram(
        'hololoom_tool_execution_duration_seconds',
        'Tool execution latency',
        ['tool_name'],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
    )

    # Parallel execution performance
    parallel_execution_wall_time = Histogram(
        'hololoom_parallel_execution_duration_seconds',
        'Parallel execution wall-clock time',
        ['stage_group'],
        buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
    )

    # ========================================================================
    # Gauges
    # ========================================================================

    active_threads = Gauge(
        'hololoom_active_threads',
        'Number of active threads by pattern',
        ['pattern_card']
    )

    confidence_score = Gauge(
        'hololoom_confidence_score',
        'Last query confidence score by tool',
        ['tool_used']
    )

    memory_shards_count = Gauge(
        'hololoom_memory_shards_count',
        'Total memory shards available'
    )

    parallel_speedup = Gauge(
        'hololoom_parallel_speedup',
        'Parallel execution speedup factor',
        ['stage_group']
    )

    retrieval_context_size = Gauge(
        'hololoom_retrieval_context_size',
        'Number of context shards retrieved'
    )

    backend_status = Gauge(
        'hololoom_backend_status',
        'Backend health status',
        ['backend']
    )


class PrometheusMetrics:
    """
    Prometheus metrics registry for HoloLoom production monitoring.

    Provides methods to track:
    - Query counts and latencies
    - Cache performance
    - Stage execution timings
    - Tool selection and confidence
    - Parallel execution metrics
    - Errors and warnings
    """

    def __init__(self):
        """Initialize metrics (gracefully disabled if prometheus_client not available)."""
        self.enabled = PROMETHEUS_AVAILABLE

        if not self.enabled:
            logger.warning(
                "Prometheus metrics disabled - prometheus_client not installed. "
                "Install with: pip install prometheus_client"
            )

    # ========================================================================
    # Query and Complexity Tracking
    # ========================================================================

    def track_query(self, pattern: str, complexity: str, duration: float, tool_used: str = 'unknown'):
        """
        Track a completed query.

        Args:
            pattern: Pattern card (BARE, FAST, FUSED)
            complexity: Complexity level (LITE, FAST, FULL, RESEARCH)
            duration: Query duration in seconds
            tool_used: Tool selected for execution (answer, search, etc.)
        """
        if not self.enabled:
            return

        queries_total.labels(
            complexity=complexity,
            pattern_card=pattern,
            tool_used=tool_used
        ).inc()

        weaving_duration.labels(
            complexity=complexity,
            pattern_card=pattern
        ).observe(duration)

    # ========================================================================
    # Cache Tracking
    # ========================================================================

    def track_cache_hit(self, cache_type: str = 'query_cache'):
        """
        Record a cache hit.

        Args:
            cache_type: Type of cache (query_cache, semantic_cache, linguistic_cache)
        """
        if not self.enabled:
            return

        cache_hits_total.labels(cache_type=cache_type).inc()

    def track_cache_miss(self, cache_type: str = 'query_cache'):
        """
        Record a cache miss.

        Args:
            cache_type: Type of cache (query_cache, semantic_cache, linguistic_cache)
        """
        if not self.enabled:
            return

        cache_misses_total.labels(cache_type=cache_type).inc()

    def get_cache_hit_rate(self, cache_type: str = 'query_cache') -> Optional[float]:
        """
        Calculate cache hit rate (if prometheus available).

        Args:
            cache_type: Type of cache to calculate for

        Returns:
            Hit rate 0.0-1.0, or None if unavailable
        """
        if not self.enabled:
            return None

        # Note: In real Prometheus, this would be calculated server-side
        # This is a placeholder for local calculation
        try:
            # Access internal prometheus metrics (implementation-specific)
            hits = cache_hits_total.labels(cache_type=cache_type)._value.get()
            misses = cache_misses_total.labels(cache_type=cache_type)._value.get()
            total = hits + misses
            return hits / total if total > 0 else 0.0
        except Exception:
            return None

    # ========================================================================
    # Stage Timing
    # ========================================================================

    def track_stage(self, stage_name: str, duration_ms: float):
        """
        Track individual stage execution time.

        Args:
            stage_name: Name of the stage (pattern_selection, retrieval, etc.)
            duration_ms: Duration in milliseconds
        """
        if not self.enabled:
            return

        stage_duration.labels(stage_name=stage_name).observe(duration_ms / 1000.0)

    def track_stage_batch(self, stage_timings: dict):
        """
        Track multiple stages at once.

        Args:
            stage_timings: Dict of {stage_name: duration_ms}
        """
        if not self.enabled:
            return

        for stage, duration_ms in stage_timings.items():
            self.track_stage(stage, duration_ms)

    # ========================================================================
    # Tool Execution
    # ========================================================================

    def track_tool_execution(self, tool_name: str, duration: float):
        """
        Track tool execution time.

        Args:
            tool_name: Name of tool (answer, search, notion_write, calc)
            duration: Duration in seconds
        """
        if not self.enabled:
            return

        tool_execution_duration.labels(tool_name=tool_name).observe(duration)

    def set_confidence(self, tool_used: str, confidence: float):
        """
        Set current confidence score for a tool.

        Args:
            tool_used: Tool name
            confidence: Confidence score (0.0-1.0)
        """
        if not self.enabled:
            return

        confidence_score.labels(tool_used=tool_used).set(confidence)

    # ========================================================================
    # Parallel Execution Metrics
    # ========================================================================

    def track_parallel_execution(self, stage_group: str, wall_time: float, speedup: float):
        """
        Track parallel execution performance.

        Args:
            stage_group: Name of parallel stage group (e.g., 'steps_4_6')
            wall_time: Actual wall-clock time in seconds
            speedup: Speedup factor (sequential_time / wall_time)
        """
        if not self.enabled:
            return

        parallel_execution_wall_time.labels(stage_group=stage_group).observe(wall_time)
        parallel_speedup.labels(stage_group=stage_group).set(speedup)

    # ========================================================================
    # Memory and Context Tracking
    # ========================================================================

    def set_memory_shards_count(self, count: int):
        """Set total memory shards available."""
        if not self.enabled:
            return
        memory_shards_count.set(count)

    def set_active_threads(self, pattern: str, count: int):
        """Set active thread count by pattern."""
        if not self.enabled:
            return
        active_threads.labels(pattern_card=pattern).set(count)

    def set_retrieval_context_size(self, count: int):
        """Set size of retrieved context (shards)."""
        if not self.enabled:
            return
        retrieval_context_size.set(count)

    # ========================================================================
    # Feature Tracking
    # ========================================================================

    def track_motifs(self, motif_count: int, motif_type: str = 'detected'):
        """
        Track motif detection.

        Args:
            motif_count: Number of motifs
            motif_type: Type (detected, regex, spacy, semantic)
        """
        if not self.enabled:
            return

        for _ in range(motif_count):
            motifs_detected_total.labels(motif_type=motif_type).inc()

    def track_reflection_update(self, signal_type: str):
        """
        Track reflection buffer update.

        Args:
            signal_type: Learning signal type (confidence, accuracy, diversity, etc.)
        """
        if not self.enabled:
            return

        reflection_updates_total.labels(signal_type=signal_type).inc()

    def track_breathing(self, phase: str):
        """
        Track ChronoTrigger breathing phases.

        Args:
            phase: Breathing phase (inhale, exhale, rest)
        """
        if not self.enabled:
            return

        # Track breathing phase as a reflection update with phase-specific type
        reflection_updates_total.labels(signal_type=f'breathing_{phase}').inc()

    # ========================================================================
    # Error Tracking
    # ========================================================================

    def track_error(self, error_type: str, stage: str):
        """
        Track an error occurrence.

        Args:
            error_type: Type of error (timeout, not_found, invalid_input, etc.)
            stage: Stage where error occurred (retrieval, policy, tool, etc.)
        """
        if not self.enabled:
            return

        errors_total.labels(error_type=error_type, stage=stage).inc()

    # ========================================================================
    # Backend Status
    # ========================================================================

    def set_backend_status(self, backend: str, healthy: bool):
        """
        Set backend health status.

        Args:
            backend: Backend name (neo4j, qdrant, inmemory)
            healthy: Whether backend is healthy
        """
        if not self.enabled:
            return

        backend_status.labels(backend=backend).set(1 if healthy else 0)

    # ========================================================================
    # Context Managers for Timing
    # ========================================================================

    @contextmanager
    def query_timer(self):
        """
        Context manager for timing a complete query.

        Usage:
            with metrics.query_timer():
                result = await weave(query)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            weaving_duration.labels(complexity='UNKNOWN', pattern_card='UNKNOWN').observe(duration)

    @contextmanager
    def stage_timer(self, stage_name: str):
        """
        Context manager for timing a specific stage.

        Usage:
            with metrics.stage_timer('retrieval'):
                shards = await retrieve(query)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.track_stage(stage_name, duration_ms)

    @contextmanager
    def tool_timer(self, tool_name: str):
        """
        Context manager for timing tool execution.

        Usage:
            with metrics.tool_timer('answer'):
                result = await tool_executor.execute('answer', query, context)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.track_tool_execution(tool_name, duration)


# Global metrics instance
metrics = PrometheusMetrics()


def start_metrics_server(port: int = 8001):
    """
    Start Prometheus metrics HTTP server.

    Args:
        port: Port to serve metrics on (default 8001)

    Usage:
        start_metrics_server(8001)
        # Metrics now available at http://localhost:8001/metrics
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning(
            "Cannot start metrics server - prometheus_client not installed. "
            "Install with: pip install prometheus_client"
        )
        return

    try:
        start_http_server(port)
        logger.info(
            f"Prometheus metrics server started on http://localhost:{port}/metrics"
        )
    except Exception as e:
        logger.error(f"Failed to start metrics server on port {port}: {e}")


def get_metrics_registry():
    """
    Get the Prometheus metrics registry.

    Returns:
        REGISTRY object or None if prometheus_client not available
    """
    if not PROMETHEUS_AVAILABLE:
        return None
    return REGISTRY


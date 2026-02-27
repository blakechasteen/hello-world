"""
Cache Performance Metrics - Prometheus Integration
===================================================

Provides Prometheus metrics for compositional cache monitoring in production.

Metrics exposed:
- phase5_cache_hits_total (counter) - Total cache hits by tier
- phase5_cache_misses_total (counter) - Total cache misses by tier
- phase5_cache_hit_rate (gauge) - Current cache hit rate by tier
- phase5_cache_size (gauge) - Current cache size by tier
- phase5_query_duration_seconds (histogram) - Query latency distribution
- phase5_speedup_factor (gauge) - Real-time speedup measurement

Usage in production:
    from hololoom.performance.cache_metrics import CacheMetricsCollector

    metrics = CacheMetricsCollector(cache)

    # Record query
    with metrics.track_query() as tracker:
        embedding, trace = cache.get_compositional_embedding(query)
        tracker.record_result(trace)

    # Expose metrics endpoint
    # GET /metrics → Prometheus format
"""

import time
from contextlib import contextmanager
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    duration_ms: float
    parse_hit: bool
    merge_hit: bool
    semantic_hit: bool


class CacheMetricsCollector:
    """
    Collects and exposes compositional cache metrics for Prometheus.

    Tracks:
    - Hit/miss rates per tier
    - Query latencies
    - Cache sizes
    - Speedup measurements

    Usage:
        collector = CacheMetricsCollector(cache)

        with collector.track_query() as tracker:
            result = cache.get_compositional_embedding(query)
            tracker.record_result(trace)
    """

    def __init__(
        self,
        cache,
        registry: Optional[Any] = None,
        namespace: str = "phase5"
    ):
        """
        Initialize metrics collector.

        Args:
            cache: CompositionalCache instance
            registry: Prometheus registry (optional, uses default if None)
            namespace: Metric namespace prefix
        """
        self.cache = cache
        self.namespace = namespace

        if not PROMETHEUS_AVAILABLE:
            # Graceful degradation
            self._enabled = False
            return

        self._enabled = True
        self.registry = registry or CollectorRegistry()

        # Initialize metrics
        self._init_metrics()

        # Track baseline for speedup calculation
        self._cold_path_samples = []
        self._hot_path_samples = []

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        if not self._enabled:
            return

        # Counters
        self.cache_hits = Counter(
            f'{self.namespace}_cache_hits_total',
            'Total cache hits',
            ['tier'],  # parse, merge, semantic
            registry=self.registry
        )

        self.cache_misses = Counter(
            f'{self.namespace}_cache_misses_total',
            'Total cache misses',
            ['tier'],
            registry=self.registry
        )

        # Gauges
        self.cache_hit_rate = Gauge(
            f'{self.namespace}_cache_hit_rate',
            'Current cache hit rate',
            ['tier'],
            registry=self.registry
        )

        self.cache_size = Gauge(
            f'{self.namespace}_cache_size',
            'Current cache size',
            ['tier'],
            registry=self.registry
        )

        self.cache_capacity = Gauge(
            f'{self.namespace}_cache_capacity',
            'Cache capacity',
            ['tier'],
            registry=self.registry
        )

        self.speedup_factor = Gauge(
            f'{self.namespace}_speedup_factor',
            'Current speedup (cold→hot)',
            registry=self.registry
        )

        # Histograms
        self.query_duration = Histogram(
            f'{self.namespace}_query_duration_seconds',
            'Query latency distribution',
            ['cache_state'],  # cold, warm, hot
            buckets=[.001, .005, .01, .025, .05, .075, .1, .25, .5, 1.0],
            registry=self.registry
        )

    @contextmanager
    def track_query(self):
        """
        Context manager for tracking query metrics.

        Usage:
            with collector.track_query() as tracker:
                result = do_work()
                tracker.record_result(result_trace)
        """
        start_time = time.perf_counter()

        class QueryTracker:
            def __init__(self, collector):
                self.collector = collector
                self.start_time = start_time

            def record_result(self, trace: Optional[Dict] = None):
                """Record query result with trace info."""
                duration_s = time.perf_counter() - self.start_time

                if not self.collector._enabled:
                    return

                # Extract cache hits from trace
                if trace:
                    parse_hit = "parse" in trace.get("hits", [])
                    merge_hit = any("merge" in h for h in trace.get("hits", []))
                    semantic_hit = "semantic" in trace.get("hits", [])
                else:
                    parse_hit = merge_hit = semantic_hit = False

                # Record hit/miss counters
                self._record_hit_miss("parse", parse_hit)
                self._record_hit_miss("merge", merge_hit)
                self._record_hit_miss("semantic", semantic_hit)

                # Determine cache state
                if parse_hit and merge_hit:
                    cache_state = "hot"
                elif parse_hit or merge_hit:
                    cache_state = "warm"
                else:
                    cache_state = "cold"

                # Record latency
                self.collector.query_duration.labels(cache_state=cache_state).observe(duration_s)

                # Track samples for speedup calculation
                duration_ms = duration_s * 1000
                if cache_state == "cold":
                    self.collector._cold_path_samples.append(duration_ms)
                    if len(self.collector._cold_path_samples) > 100:
                        self.collector._cold_path_samples.pop(0)
                elif cache_state == "hot":
                    self.collector._hot_path_samples.append(duration_ms)
                    if len(self.collector._hot_path_samples) > 100:
                        self.collector._hot_path_samples.pop(0)

                # Update speedup gauge
                self._update_speedup()

                # Update cache size gauges
                self._update_cache_sizes()

            def _record_hit_miss(self, tier: str, is_hit: bool):
                """Record hit or miss for a tier."""
                if is_hit:
                    self.collector.cache_hits.labels(tier=tier).inc()
                else:
                    self.collector.cache_misses.labels(tier=tier).inc()

                # Update hit rate gauge
                self._update_hit_rate(tier)

            def _update_hit_rate(self, tier: str):
                """Update hit rate gauge for a tier."""
                stats = self.collector.cache.get_statistics()

                tier_stats = {
                    "parse": stats["parse_cache"],
                    "merge": stats["merge_cache"],
                    "semantic": stats["semantic_cache"]
                }

                if tier in tier_stats:
                    hit_rate = tier_stats[tier]["hit_rate"]
                    self.collector.cache_hit_rate.labels(tier=tier).set(hit_rate)

            def _update_cache_sizes(self):
                """Update cache size gauges."""
                stats = self.collector.cache.get_statistics()

                # Parse cache
                self.collector.cache_size.labels(tier="parse").set(
                    stats["parse_cache"]["size"]
                )
                self.collector.cache_capacity.labels(tier="parse").set(
                    stats["parse_cache"]["capacity"]
                )

                # Merge cache
                self.collector.cache_size.labels(tier="merge").set(
                    stats["merge_cache"]["size"]
                )
                self.collector.cache_capacity.labels(tier="merge").set(
                    stats["merge_cache"]["capacity"]
                )

            def _update_speedup(self):
                """Update speedup gauge."""
                if len(self.collector._cold_path_samples) < 5 or len(self.collector._hot_path_samples) < 5:
                    return  # Not enough samples

                cold_avg = sum(self.collector._cold_path_samples) / len(self.collector._cold_path_samples)
                hot_avg = sum(self.collector._hot_path_samples) / len(self.collector._hot_path_samples)

                if hot_avg > 0:
                    speedup = cold_avg / hot_avg
                    self.collector.speedup_factor.set(speedup)

        tracker = QueryTracker(self)
        yield tracker

    def get_summary(self) -> Dict[str, Any]:
        """Get current metrics summary (for debugging/logging)."""
        if not self._enabled:
            return {"enabled": False}

        stats = self.cache.get_statistics()

        cold_avg = sum(self._cold_path_samples) / len(self._cold_path_samples) if self._cold_path_samples else 0
        hot_avg = sum(self._hot_path_samples) / len(self._hot_path_samples) if self._hot_path_samples else 0
        speedup = cold_avg / hot_avg if hot_avg > 0 else 0

        return {
            "enabled": True,
            "parse_cache": {
                "hit_rate": stats["parse_cache"]["hit_rate"],
                "size": stats["parse_cache"]["size"],
                "capacity": stats["parse_cache"]["capacity"]
            },
            "merge_cache": {
                "hit_rate": stats["merge_cache"]["hit_rate"],
                "size": stats["merge_cache"]["size"],
                "capacity": stats["merge_cache"]["capacity"]
            },
            "speedup": {
                "cold_avg_ms": cold_avg,
                "hot_avg_ms": hot_avg,
                "factor": speedup
            }
        }


# ============================================================================
# Flask/FastAPI Integration Helpers
# ============================================================================

def create_metrics_endpoint(collector: CacheMetricsCollector):
    """
    Create Prometheus metrics endpoint handler.

    For Flask:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        @app.route('/metrics')
        def metrics():
            return generate_latest(collector.registry), 200, {'Content-Type': CONTENT_TYPE_LATEST}

    For FastAPI:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi import Response

        @app.get('/metrics')
        def metrics():
            return Response(
                content=generate_latest(collector.registry),
                media_type=CONTENT_TYPE_LATEST
            )
    """
    if not PROMETHEUS_AVAILABLE:
        raise RuntimeError("Prometheus client not available. Install: pip install prometheus-client")

    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    def metrics_handler():
        return generate_latest(collector.registry), 200, {'Content-Type': CONTENT_TYPE_LATEST}

    return metrics_handler


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Cache Metrics Module")
    print("=" * 80)

    if PROMETHEUS_AVAILABLE:
        print("✅ Prometheus client available")
        print("\nUsage:")
        print("  collector = CacheMetricsCollector(cache)")
        print("  with collector.track_query() as tracker:")
        print("      result = cache.get_compositional_embedding(query)")
        print("      tracker.record_result(trace)")
        print("\nMetrics endpoint:")
        print("  from prometheus_client import generate_latest")
        print("  metrics_data = generate_latest(collector.registry)")
    else:
        print("⚠️  Prometheus client not available")
        print("   Install with: pip install prometheus-client")
        print("   Metrics collection will be disabled (graceful degradation)")
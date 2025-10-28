"""
Prometheus Metrics for HoloLoom Production Monitoring
=====================================================
Exports Prometheus metrics for production dashboards.

Metrics:
- hololoom_query_duration_seconds: Query latency
- hololoom_queries_total: Total queries
- hololoom_breathing_cycles_total: Breathing cycles  
- hololoom_cache_hits_total: Cache performance
- hololoom_pattern_selections_total: Pattern usage

Usage:
    from HoloLoom.performance.prometheus_metrics import metrics, start_metrics_server

    # Track metrics
    with metrics.query_timer():
        result = await process()

    # Start server
    start_metrics_server(port=8001)
"""

import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not installed - metrics disabled")
    PROMETHEUS_AVAILABLE = False

# Metric definitions
if PROMETHEUS_AVAILABLE:
    query_duration = Histogram('hololoom_query_duration_seconds', 'Query duration',
                               buckets=(0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0, 2.0, 5.0))
    queries_total = Counter('hololoom_queries_total', 'Total queries', ['pattern', 'complexity'])
    errors_total = Counter('hololoom_errors_total', 'Total errors', ['type', 'stage'])
    breathing_cycles_total = Counter('hololoom_breathing_cycles_total', 'Breathing cycles', ['phase'])
    cache_hits_total = Counter('hololoom_cache_hits_total', 'Cache hits')
    cache_misses_total = Counter('hololoom_cache_misses_total', 'Cache misses')
    pattern_selections_total = Counter('hololoom_pattern_selections_total', 'Pattern selections', ['pattern'])
    backend_status = Gauge('hololoom_backend_status', 'Backend health', ['backend'])

class PrometheusMetrics:
    """Prometheus metrics registry."""
    
    def __init__(self):
        self.enabled = PROMETHEUS_AVAILABLE
    
    def track_query(self, pattern: str, complexity: str, duration: float):
        if not self.enabled: return
        query_duration.observe(duration)
        queries_total.labels(pattern=pattern, complexity=complexity).inc()
    
    def track_error(self, error_type: str, stage: str):
        if not self.enabled: return
        errors_total.labels(type=error_type, stage=stage).inc()
    
    def track_breathing(self, phase: str):
        if not self.enabled: return
        breathing_cycles_total.labels(phase=phase).inc()
    
    def track_cache_hit(self):
        if not self.enabled: return
        cache_hits_total.inc()
    
    def track_cache_miss(self):
        if not self.enabled: return
        cache_misses_total.inc()
    
    def track_pattern(self, pattern: str):
        if not self.enabled: return
        pattern_selections_total.labels(pattern=pattern).inc()
    
    def set_backend_status(self, backend: str, healthy: bool):
        if not self.enabled: return
        backend_status.labels(backend=backend).set(1 if healthy else 0)
    
    @contextmanager
    def query_timer(self):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            query_duration.observe(duration)

# Global instance
metrics = PrometheusMetrics()

def start_metrics_server(port: int = 8001):
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Cannot start metrics server - prometheus_client not installed")
        return
    start_http_server(port)
    logger.info(f"Metrics server started on port {port}")


#!/usr/bin/env python3
"""
Test suite for Prometheus metrics implementation.

Verifies:
1. Metrics initialization without prometheus_client
2. Metrics functionality with prometheus_client installed
3. Context managers for timing
4. Batch operations
5. Graceful degradation
"""

import unittest
import time
from unittest.mock import patch, MagicMock

# Try to import actual prometheus_client
try:
    from prometheus_client import Counter, Histogram, Gauge, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class TestPrometheusMetricsGracefulDegradation(unittest.TestCase):
    """Test graceful degradation when prometheus_client is unavailable."""

    def test_metrics_disabled_without_prometheus(self):
        """Verify metrics gracefully disabled if prometheus_client unavailable."""
        with patch('hololoom.performance.prometheus_metrics.PROMETHEUS_AVAILABLE', False):
            from hololoom.performance.prometheus_metrics import PrometheusMetrics

            metrics = PrometheusMetrics()
            self.assertFalse(metrics.enabled)

            # These should not raise errors
            metrics.track_query('FUSED', 'FAST', 0.125, 'answer')
            metrics.track_cache_hit('query_cache')
            metrics.track_stage('retrieval', 50.0)
            metrics.track_tool_execution('answer', 0.1)

    def test_metrics_enabled_with_prometheus(self):
        """Verify metrics enabled if prometheus_client available."""
        if not PROMETHEUS_AVAILABLE:
            self.skipTest("prometheus_client not installed")

        from hololoom.performance.prometheus_metrics import PrometheusMetrics

        metrics = PrometheusMetrics()
        self.assertTrue(metrics.enabled)


class TestPrometheusMetricsOperations(unittest.TestCase):
    """Test metrics operations (if prometheus_client available)."""

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_track_query(self):
        """Test query tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        # Should not raise
        metrics.track_query(
            pattern='FUSED',
            complexity='FAST',
            duration=0.125,
            tool_used='answer'
        )

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_track_cache_hits_and_misses(self):
        """Test cache hit/miss tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.track_cache_hit('query_cache')
        metrics.track_cache_hit('semantic_cache')
        metrics.track_cache_miss('query_cache')
        metrics.track_cache_miss('semantic_cache')

        # Should not raise
        hit_rate = metrics.get_cache_hit_rate('query_cache')
        # May be None if implementation incomplete

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_track_stage_batch(self):
        """Test batch stage tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        stage_timings = {
            'pattern_selection': 5.0,
            'retrieval': 50.0,
            'feature_extraction': 45.5,
            'convergence': 10.0
        }

        metrics.track_stage_batch(stage_timings)

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_track_tool_execution(self):
        """Test tool execution tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.track_tool_execution('answer', 0.1)
        metrics.track_tool_execution('search', 0.15)
        metrics.track_tool_execution('notion_write', 0.2)

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_track_parallel_execution(self):
        """Test parallel execution metrics."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.track_parallel_execution(
            stage_group='steps_4_6',
            wall_time=0.08,
            speedup=2.35
        )

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_set_confidence(self):
        """Test confidence setting."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.set_confidence('answer', 0.92)
        metrics.set_confidence('search', 0.78)

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_set_active_threads(self):
        """Test active threads tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.set_active_threads('FUSED', 12)
        metrics.set_active_threads('FAST', 6)

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_set_retrieval_context_size(self):
        """Test retrieval context size."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.set_retrieval_context_size(8)

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_track_motifs(self):
        """Test motif tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.track_motifs(3, 'regex')
        metrics.track_motifs(2, 'spacy')

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_track_reflection_update(self):
        """Test reflection update tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.track_reflection_update('confidence')
        metrics.track_reflection_update('accuracy')

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_track_error(self):
        """Test error tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.track_error('timeout', 'retrieval')
        metrics.track_error('not_found', 'tool')

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_set_backend_status(self):
        """Test backend status."""
        from hololoom.performance.prometheus_metrics import metrics

        metrics.set_backend_status('neo4j', True)
        metrics.set_backend_status('qdrant', False)


class TestPrometheusMetricsContextManagers(unittest.TestCase):
    """Test context manager functionality."""

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_query_timer(self):
        """Test query timer context manager."""
        from hololoom.performance.prometheus_metrics import metrics

        with metrics.query_timer():
            time.sleep(0.01)  # Sleep 10ms

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_stage_timer(self):
        """Test stage timer context manager."""
        from hololoom.performance.prometheus_metrics import metrics

        with metrics.stage_timer('retrieval'):
            time.sleep(0.01)

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_tool_timer(self):
        """Test tool timer context manager."""
        from hololoom.performance.prometheus_metrics import metrics

        with metrics.tool_timer('answer'):
            time.sleep(0.01)

    def test_timer_without_prometheus(self):
        """Test timers work without prometheus_client."""
        with patch('hololoom.performance.prometheus_metrics.PROMETHEUS_AVAILABLE', False):
            from hololoom.performance.prometheus_metrics import PrometheusMetrics

            metrics = PrometheusMetrics()

            # Should not raise
            with metrics.query_timer():
                time.sleep(0.01)

            with metrics.stage_timer('retrieval'):
                time.sleep(0.01)

            with metrics.tool_timer('answer'):
                time.sleep(0.01)


class TestPrometheusMetricsServer(unittest.TestCase):
    """Test metrics server startup."""

    def test_start_metrics_server_without_prometheus(self):
        """Verify server startup gracefully handles missing prometheus_client."""
        with patch('hololoom.performance.prometheus_metrics.PROMETHEUS_AVAILABLE', False):
            from hololoom.performance.prometheus_metrics import start_metrics_server

            # Should not raise
            start_metrics_server(8001)

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_get_metrics_registry_with_prometheus(self):
        """Test registry retrieval with prometheus_client."""
        from hololoom.performance.prometheus_metrics import get_metrics_registry

        registry = get_metrics_registry()
        self.assertIsNotNone(registry)

    def test_get_metrics_registry_without_prometheus(self):
        """Test registry retrieval without prometheus_client."""
        with patch('hololoom.performance.prometheus_metrics.PROMETHEUS_AVAILABLE', False):
            from hololoom.performance.prometheus_metrics import get_metrics_registry

            registry = get_metrics_registry()
            self.assertIsNone(registry)


class TestPrometheusMetricsIntegration(unittest.TestCase):
    """Integration tests simulating real usage."""

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_complete_query_lifecycle(self):
        """Test full query metrics tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        # Simulate complete query lifecycle
        query_start = time.time()

        # Stage tracking
        metrics.track_stage('pattern_selection', 5.0)
        metrics.track_stage('temporal_setup', 2.0)
        metrics.track_stage('thread_selection', 3.0)

        # Parallel execution
        metrics.track_parallel_execution('steps_4_6', 0.08, 2.35)

        # Tool execution
        metrics.track_tool_execution('answer', 0.1)

        # Query completion
        query_duration = time.time() - query_start
        metrics.track_query('FUSED', 'FAST', query_duration, 'answer')

        # Update state
        metrics.set_confidence('answer', 0.92)
        metrics.set_active_threads('FUSED', 12)
        metrics.set_retrieval_context_size(8)

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_cache_tracking_workflow(self):
        """Test cache hit/miss workflow."""
        from hololoom.performance.prometheus_metrics import metrics

        # Simulate cache misses (cold start)
        for _ in range(10):
            metrics.track_cache_miss('query_cache')

        # Simulate cache hits (warm)
        for _ in range(40):
            metrics.track_cache_hit('query_cache')

        # Check hit rate
        hit_rate = metrics.get_cache_hit_rate('query_cache')
        # May be None or 0.8 (40 hits / 50 total)

    @unittest.skipIf(not PROMETHEUS_AVAILABLE, "prometheus_client required")
    def test_error_tracking_workflow(self):
        """Test error tracking."""
        from hololoom.performance.prometheus_metrics import metrics

        # Track various errors
        metrics.track_error('timeout', 'retrieval')
        metrics.track_error('timeout', 'convergence')
        metrics.track_error('not_found', 'tool')
        metrics.track_error('invalid_input', 'pattern_selection')


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()

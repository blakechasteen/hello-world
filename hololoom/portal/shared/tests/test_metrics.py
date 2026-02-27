"""
Unit tests for Prometheus-compatible Metrics.

Tests Counter, Gauge, Histogram, LabeledCounter, LabeledGauge, and MetricsRegistry.
"""

import pytest
import threading
import time

from ..metrics import (
    Counter,
    Gauge,
    Histogram,
    LabeledCounter,
    LabeledGauge,
    MetricsRegistry,
    metrics,
)


class TestCounter:
    """Tests for Counter metric type."""

    def test_counter_creation(self):
        """Counter initializes with zero value."""
        counter = Counter("test_counter", "A test counter")

        assert counter.name == "test_counter"
        assert counter.help == "A test counter"
        assert counter.value == 0.0

    def test_counter_increment_default(self):
        """Counter increments by 1 by default."""
        counter = Counter("test_counter", "A test counter")

        counter.inc()

        assert counter.value == 1.0

    def test_counter_increment_custom_amount(self):
        """Counter increments by custom amount."""
        counter = Counter("test_counter", "A test counter")

        counter.inc(5.5)

        assert counter.value == 5.5

    def test_counter_multiple_increments(self):
        """Counter accumulates multiple increments."""
        counter = Counter("test_counter", "A test counter")

        counter.inc(1.0)
        counter.inc(2.0)
        counter.inc(3.0)

        assert counter.value == 6.0

    def test_counter_reset(self):
        """Counter resets to zero."""
        counter = Counter("test_counter", "A test counter")
        counter.inc(10.0)

        counter.reset()

        assert counter.value == 0.0

    def test_counter_with_labels(self):
        """Counter stores labels."""
        counter = Counter(
            "test_counter",
            "A test counter",
            labels={"method": "POST", "status": "200"},
        )

        assert counter.labels == {"method": "POST", "status": "200"}

    def test_counter_thread_safety(self):
        """Counter is thread-safe for concurrent increments."""
        counter = Counter("test_counter", "A test counter")
        num_threads = 10
        increments_per_thread = 1000

        def increment_many():
            for _ in range(increments_per_thread):
                counter.inc()

        threads = [threading.Thread(target=increment_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.value == num_threads * increments_per_thread


class TestGauge:
    """Tests for Gauge metric type."""

    def test_gauge_creation(self):
        """Gauge initializes with zero value."""
        gauge = Gauge("test_gauge", "A test gauge")

        assert gauge.name == "test_gauge"
        assert gauge.help == "A test gauge"
        assert gauge.value == 0.0

    def test_gauge_set(self):
        """Gauge sets to specific value."""
        gauge = Gauge("test_gauge", "A test gauge")

        gauge.set(42.5)

        assert gauge.value == 42.5

    def test_gauge_set_overwrites(self):
        """Gauge set overwrites previous value."""
        gauge = Gauge("test_gauge", "A test gauge")
        gauge.set(100.0)

        gauge.set(50.0)

        assert gauge.value == 50.0

    def test_gauge_increment(self):
        """Gauge increments by amount."""
        gauge = Gauge("test_gauge", "A test gauge")
        gauge.set(10.0)

        gauge.inc(5.0)

        assert gauge.value == 15.0

    def test_gauge_increment_default(self):
        """Gauge increments by 1 by default."""
        gauge = Gauge("test_gauge", "A test gauge")
        gauge.set(10.0)

        gauge.inc()

        assert gauge.value == 11.0

    def test_gauge_decrement(self):
        """Gauge decrements by amount."""
        gauge = Gauge("test_gauge", "A test gauge")
        gauge.set(10.0)

        gauge.dec(3.0)

        assert gauge.value == 7.0

    def test_gauge_decrement_default(self):
        """Gauge decrements by 1 by default."""
        gauge = Gauge("test_gauge", "A test gauge")
        gauge.set(10.0)

        gauge.dec()

        assert gauge.value == 9.0

    def test_gauge_can_go_negative(self):
        """Gauge can have negative values."""
        gauge = Gauge("test_gauge", "A test gauge")

        gauge.dec(5.0)

        assert gauge.value == -5.0

    def test_gauge_thread_safety(self):
        """Gauge is thread-safe for concurrent operations."""
        gauge = Gauge("test_gauge", "A test gauge")
        num_threads = 10
        operations_per_thread = 100

        def inc_dec_many():
            for _ in range(operations_per_thread):
                gauge.inc(1.0)
                gauge.dec(1.0)

        threads = [threading.Thread(target=inc_dec_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After equal increments and decrements, should be back to 0
        assert gauge.value == 0.0


class TestHistogram:
    """Tests for Histogram metric type."""

    def test_histogram_creation(self):
        """Histogram initializes with default buckets."""
        histogram = Histogram("test_histogram", "A test histogram")

        assert histogram.name == "test_histogram"
        assert histogram.help == "A test histogram"
        assert histogram.count == 0
        assert histogram.sum == 0.0

    def test_histogram_custom_buckets(self):
        """Histogram uses custom buckets."""
        buckets = [1, 5, 10, 50, 100]
        histogram = Histogram("test_histogram", "A test histogram", buckets=buckets)

        assert histogram.buckets == buckets

    def test_histogram_observe(self):
        """Histogram records observations."""
        histogram = Histogram("test_histogram", "A test histogram")

        histogram.observe(50.0)

        assert histogram.count == 1
        assert histogram.sum == 50.0

    def test_histogram_multiple_observations(self):
        """Histogram accumulates multiple observations."""
        histogram = Histogram("test_histogram", "A test histogram")

        histogram.observe(10.0)
        histogram.observe(20.0)
        histogram.observe(30.0)

        assert histogram.count == 3
        assert histogram.sum == 60.0

    def test_histogram_bucket_counts(self):
        """Histogram counts values in correct buckets."""
        histogram = Histogram(
            "test_histogram",
            "A test histogram",
            buckets=[10, 50, 100],
        )

        # Observe values that fall in different buckets
        histogram.observe(5.0)   # ≤10, ≤50, ≤100
        histogram.observe(25.0)  # ≤50, ≤100
        histogram.observe(75.0)  # ≤100
        histogram.observe(150.0) # Only +Inf

        bucket_counts = histogram.get_bucket_counts()

        assert bucket_counts[10] == 1    # Only 5
        assert bucket_counts[50] == 2    # 5 and 25
        assert bucket_counts[100] == 3   # 5, 25, and 75
        assert bucket_counts[float("inf")] == 4  # All values

    def test_histogram_inf_bucket_always_includes_all(self):
        """Histogram +Inf bucket always includes all observations."""
        histogram = Histogram(
            "test_histogram",
            "A test histogram",
            buckets=[10],
        )

        histogram.observe(5.0)
        histogram.observe(500.0)
        histogram.observe(5000.0)

        bucket_counts = histogram.get_bucket_counts()
        assert bucket_counts[float("inf")] == 3

    def test_histogram_reset(self):
        """Histogram reset clears all data."""
        histogram = Histogram("test_histogram", "A test histogram")
        histogram.observe(50.0)
        histogram.observe(100.0)

        histogram.reset()

        assert histogram.count == 0
        assert histogram.sum == 0.0
        bucket_counts = histogram.get_bucket_counts()
        for count in bucket_counts.values():
            assert count == 0

    def test_histogram_thread_safety(self):
        """Histogram is thread-safe for concurrent observations."""
        histogram = Histogram("test_histogram", "A test histogram")
        num_threads = 10
        observations_per_thread = 100

        def observe_many():
            for i in range(observations_per_thread):
                histogram.observe(float(i))

        threads = [threading.Thread(target=observe_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert histogram.count == num_threads * observations_per_thread


class TestLabeledCounter:
    """Tests for LabeledCounter (counter with dynamic labels)."""

    def test_labeled_counter_creation(self):
        """LabeledCounter initializes correctly."""
        counter = LabeledCounter(
            "test_counter",
            "A test counter",
            ["method", "status"],
        )

        assert counter.name == "test_counter"
        assert counter.help == "A test counter"
        assert counter.label_names == ["method", "status"]

    def test_labeled_counter_labels(self):
        """LabeledCounter creates instances for label combinations."""
        counter = LabeledCounter(
            "test_counter",
            "A test counter",
            ["status"],
        )

        counter.labels(status="200").inc()
        counter.labels(status="404").inc(5)

        assert counter.labels(status="200").value == 1.0
        assert counter.labels(status="404").value == 5.0

    def test_labeled_counter_same_labels_returns_same_instance(self):
        """LabeledCounter returns same instance for same labels."""
        counter = LabeledCounter(
            "test_counter",
            "A test counter",
            ["status"],
        )

        c1 = counter.labels(status="200")
        c2 = counter.labels(status="200")

        c1.inc()
        assert c2.value == 1.0  # Same instance

    def test_labeled_counter_get_all(self):
        """LabeledCounter returns all counter instances."""
        counter = LabeledCounter(
            "test_counter",
            "A test counter",
            ["status"],
        )

        counter.labels(status="200").inc()
        counter.labels(status="404").inc()
        counter.labels(status="500").inc()

        all_counters = counter.get_all()
        assert len(all_counters) == 3


class TestLabeledGauge:
    """Tests for LabeledGauge (gauge with dynamic labels)."""

    def test_labeled_gauge_creation(self):
        """LabeledGauge initializes correctly."""
        gauge = LabeledGauge(
            "test_gauge",
            "A test gauge",
            ["node_id"],
        )

        assert gauge.name == "test_gauge"
        assert gauge.help == "A test gauge"
        assert gauge.label_names == ["node_id"]

    def test_labeled_gauge_labels(self):
        """LabeledGauge creates instances for label combinations."""
        gauge = LabeledGauge(
            "test_gauge",
            "A test gauge",
            ["node_id"],
        )

        gauge.labels(node_id="node-1").set(50.0)
        gauge.labels(node_id="node-2").set(75.0)

        assert gauge.labels(node_id="node-1").value == 50.0
        assert gauge.labels(node_id="node-2").value == 75.0

    def test_labeled_gauge_get_all(self):
        """LabeledGauge returns all gauge instances."""
        gauge = LabeledGauge(
            "test_gauge",
            "A test gauge",
            ["node_id"],
        )

        gauge.labels(node_id="node-1").set(50.0)
        gauge.labels(node_id="node-2").set(75.0)

        all_gauges = gauge.get_all()
        assert len(all_gauges) == 2


class TestMetricsRegistry:
    """Tests for MetricsRegistry central registry."""

    def test_registry_has_job_counters(self):
        """Registry has job lifecycle counters."""
        registry = MetricsRegistry()

        assert hasattr(registry, "jobs_submitted")
        assert hasattr(registry, "jobs_completed")
        assert hasattr(registry, "jobs_failed")
        assert hasattr(registry, "jobs_queued")
        assert hasattr(registry, "jobs_cancelled")

    def test_registry_has_job_histogram(self):
        """Registry has job duration histogram."""
        registry = MetricsRegistry()

        assert hasattr(registry, "job_duration")
        assert isinstance(registry.job_duration, Histogram)

    def test_registry_has_node_gauges(self):
        """Registry has node gauges."""
        registry = MetricsRegistry()

        assert hasattr(registry, "nodes_online")
        assert hasattr(registry, "nodes_total")
        assert isinstance(registry.nodes_online, Gauge)

    def test_registry_has_labeled_node_metrics(self):
        """Registry has per-node labeled metrics."""
        registry = MetricsRegistry()

        assert hasattr(registry, "node_cpu")
        assert hasattr(registry, "node_memory")
        assert hasattr(registry, "node_jobs_running")
        assert isinstance(registry.node_cpu, LabeledGauge)

    def test_registry_has_queue_metrics(self):
        """Registry has queue metrics."""
        registry = MetricsRegistry()

        assert hasattr(registry, "queue_depth")
        assert hasattr(registry, "queue_wait_time")

    def test_registry_has_scheduler_metrics(self):
        """Registry has scheduler metrics."""
        registry = MetricsRegistry()

        assert hasattr(registry, "scheduler_selections")
        assert hasattr(registry, "scheduler_fallbacks")

    def test_registry_increment_counters(self):
        """Registry counters can be incremented."""
        registry = MetricsRegistry()

        registry.jobs_submitted.inc()
        registry.jobs_completed.inc()
        registry.jobs_failed.inc(2)

        assert registry.jobs_submitted.value == 1.0
        assert registry.jobs_completed.value == 1.0
        assert registry.jobs_failed.value == 2.0

    def test_registry_set_gauges(self):
        """Registry gauges can be set."""
        registry = MetricsRegistry()

        registry.nodes_online.set(5)
        registry.nodes_total.set(10)
        registry.queue_depth.set(25)

        assert registry.nodes_online.value == 5.0
        assert registry.nodes_total.value == 10.0
        assert registry.queue_depth.value == 25.0

    def test_registry_observe_histograms(self):
        """Registry histograms can record observations."""
        registry = MetricsRegistry()

        registry.job_duration.observe(100.0)
        registry.job_duration.observe(200.0)
        registry.queue_wait_time.observe(500.0)

        assert registry.job_duration.count == 2
        assert registry.job_duration.sum == 300.0
        assert registry.queue_wait_time.count == 1

    def test_registry_labeled_metrics(self):
        """Registry labeled metrics work correctly."""
        registry = MetricsRegistry()

        registry.node_cpu.labels(node_id="node-1").set(45.5)
        registry.node_memory.labels(node_id="node-1").set(60.0)
        registry.jobs_by_module.labels(module_id="analyzer").inc()

        assert registry.node_cpu.labels(node_id="node-1").value == 45.5
        assert registry.node_memory.labels(node_id="node-1").value == 60.0
        assert registry.jobs_by_module.labels(module_id="analyzer").value == 1.0

    def test_registry_to_dict(self):
        """Registry exports to dictionary."""
        registry = MetricsRegistry()
        registry.jobs_submitted.inc(10)
        registry.jobs_completed.inc(8)
        registry.jobs_failed.inc(2)
        registry.nodes_online.set(5)
        registry.queue_depth.set(3)

        data = registry.to_dict()

        assert data["jobs"]["submitted"] == 10.0
        assert data["jobs"]["completed"] == 8.0
        assert data["jobs"]["failed"] == 2.0
        assert data["nodes"]["online"] == 5.0
        assert data["queue"]["depth"] == 3.0

    def test_registry_to_dict_avg_duration(self):
        """Registry dictionary includes average job duration."""
        registry = MetricsRegistry()
        registry.job_duration.observe(100.0)
        registry.job_duration.observe(200.0)
        registry.job_duration.observe(300.0)

        data = registry.to_dict()

        assert data["jobs"]["duration"]["count"] == 3
        assert data["jobs"]["duration"]["sum"] == 600.0
        assert data["jobs"]["duration"]["avg_ms"] == 200.0

    def test_registry_to_dict_zero_division_safe(self):
        """Registry dictionary handles zero observations."""
        registry = MetricsRegistry()

        data = registry.to_dict()

        assert data["jobs"]["duration"]["avg_ms"] == 0.0
        assert data["queue"]["wait_time"]["avg_ms"] == 0.0

    def test_registry_reset_all(self):
        """Registry reset_all clears all metrics."""
        registry = MetricsRegistry()
        registry.jobs_submitted.inc(100)
        registry.nodes_online.set(10)
        registry.job_duration.observe(500.0)

        registry.reset_all()

        assert registry.jobs_submitted.value == 0.0
        assert registry.nodes_online.value == 0.0
        assert registry.job_duration.count == 0


class TestPrometheusExport:
    """Tests for Prometheus text format export."""

    def test_export_counter_format(self):
        """Prometheus export formats counters correctly."""
        registry = MetricsRegistry()
        registry.jobs_submitted.inc(42)

        text = registry.export_prometheus()

        assert "# HELP portal_jobs_submitted_total Total number of jobs submitted to the system" in text
        assert "# TYPE portal_jobs_submitted_total counter" in text
        assert "portal_jobs_submitted_total 42" in text

    def test_export_gauge_format(self):
        """Prometheus export formats gauges correctly."""
        registry = MetricsRegistry()
        registry.nodes_online.set(5)

        text = registry.export_prometheus()

        assert "# HELP portal_nodes_online Number of nodes currently online and healthy" in text
        assert "# TYPE portal_nodes_online gauge" in text
        assert "portal_nodes_online 5" in text

    def test_export_histogram_format(self):
        """Prometheus export formats histograms correctly."""
        registry = MetricsRegistry()
        registry.job_duration.observe(100.0)

        text = registry.export_prometheus()

        assert "# HELP portal_job_duration_ms Job execution duration in milliseconds" in text
        assert "# TYPE portal_job_duration_ms histogram" in text
        assert "portal_job_duration_ms_bucket" in text
        assert "portal_job_duration_ms_sum 100" in text
        assert "portal_job_duration_ms_count 1" in text

    def test_export_labeled_counter_format(self):
        """Prometheus export formats labeled counters correctly."""
        registry = MetricsRegistry()
        registry.jobs_by_module.labels(module_id="analyzer").inc(5)
        registry.jobs_by_module.labels(module_id="processor").inc(3)

        text = registry.export_prometheus()

        assert 'portal_jobs_by_module_total{module_id="analyzer"} 5' in text
        assert 'portal_jobs_by_module_total{module_id="processor"} 3' in text

    def test_export_labeled_gauge_format(self):
        """Prometheus export formats labeled gauges correctly."""
        registry = MetricsRegistry()
        registry.node_cpu.labels(node_id="node-1").set(45.5)

        text = registry.export_prometheus()

        assert 'portal_node_cpu_percent{node_id="node-1"} 45.5' in text

    def test_export_histogram_buckets(self):
        """Prometheus export includes histogram bucket labels."""
        registry = MetricsRegistry()
        registry.job_duration.observe(50.0)

        text = registry.export_prometheus()

        # Check for bucket format with le label
        assert 'le="' in text
        assert 'le="+Inf"' in text


class TestGlobalMetrics:
    """Tests for global metrics singleton."""

    def test_global_metrics_is_registry(self):
        """Global metrics is a MetricsRegistry instance."""
        assert isinstance(metrics, MetricsRegistry)

    def test_global_metrics_usable(self):
        """Global metrics singleton is usable."""
        initial = metrics.health_checks.value

        metrics.health_checks.inc()

        assert metrics.health_checks.value == initial + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

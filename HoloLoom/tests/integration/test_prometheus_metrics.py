"""
Integration tests for Prometheus metrics collector.

Tests the JobMetricsCollector for job lifecycle tracking,
histogram bucket management, and Prometheus format export.
"""

import pytest
import time
from HoloLoom.apps.chatops.handlers.prometheus_metrics import (
    JobMetricsCollector,
    LatencyHistogram,
    MetricType,
    MetricValue,
    get_metrics_collector,
    set_metrics_collector,
)


class TestLatencyHistogram:
    """Tests for the LatencyHistogram component."""

    def test_observe_updates_buckets(self):
        """Observations should update the correct buckets."""
        hist = LatencyHistogram(buckets=[10, 50, 100])
        hist.observe(25.0)

        # 25.0 <= 50, so bucket 50 should be 1
        assert hist.bucket_counts[50] == 1
        # 25.0 > 10, so bucket 10 should be 0
        assert hist.bucket_counts[10] == 0
        # 25.0 <= 100, so bucket 100 should be 1
        assert hist.bucket_counts[100] == 1
        # +Inf always gets incremented
        assert hist.bucket_counts[float('inf')] == 1

    def test_percentile_calculation(self):
        """Percentiles should be calculated correctly."""
        hist = LatencyHistogram()
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            hist.observe(v)

        # p50 should be median (around 50-55)
        assert 50 <= hist.percentile(50) <= 60
        # p90 should be around 90
        assert 90 <= hist.percentile(90) <= 100

    def test_mean_calculation(self):
        """Mean should be calculated correctly."""
        hist = LatencyHistogram()
        values = [10, 20, 30, 40, 50]
        for v in values:
            hist.observe(v)

        assert hist.mean() == 30.0  # (10+20+30+40+50) / 5

    def test_empty_histogram(self):
        """Empty histogram should return 0 for percentile and mean."""
        hist = LatencyHistogram()
        assert hist.percentile(50) == 0.0
        assert hist.mean() == 0.0

    def test_observation_trimming(self):
        """Observations should be trimmed to max_observations."""
        hist = LatencyHistogram()
        hist.max_observations = 100

        # Add more than max_observations
        for i in range(150):
            hist.observe(float(i))

        # Should only keep last 100
        assert len(hist.observations) == 100
        # Count should still be accurate
        assert hist.count == 150


class TestJobMetricsCollector:
    """Tests for the JobMetricsCollector component."""

    def test_job_lifecycle_tracking(self):
        """Full job lifecycle should be tracked correctly."""
        collector = JobMetricsCollector()

        # Submit job
        collector.job_submitted("job1", "test query")
        assert collector._jobs_pending == 1
        assert collector._jobs_total["pending"] == 1

        # Start job
        collector.job_started("job1")
        assert collector._jobs_pending == 0
        assert collector._jobs_running == 1
        assert collector._jobs_total["running"] == 1

        # Complete job
        collector.job_completed("job1", 150.0, 0.92, "answer")
        assert collector._jobs_running == 0
        assert collector._jobs_total["completed"] == 1

        # Check summary
        summary = collector.get_summary()
        assert summary["jobs_total"]["completed"] == 1
        assert summary["latency_ms"]["mean"] == 150.0

    def test_prometheus_format(self):
        """Prometheus format should be valid."""
        collector = JobMetricsCollector()
        collector.job_submitted("job1", "test query")

        output = collector.format_prometheus()

        # Should contain HELP and TYPE comments
        assert "# HELP hololoom_jobs_total" in output
        assert "# TYPE hololoom_jobs_total" in output

        # Should contain the actual metric
        assert 'hololoom_jobs_total{status="pending"}' in output

    def test_concurrent_job_tracking(self):
        """Multiple concurrent jobs should be tracked correctly."""
        collector = JobMetricsCollector()

        # Submit 10 jobs
        for i in range(10):
            collector.job_submitted(f"job{i}", f"query{i}")

        assert collector._jobs_pending == 10
        assert collector._jobs_total["pending"] == 10

        # Start 5 jobs
        for i in range(5):
            collector.job_started(f"job{i}")

        assert collector._jobs_pending == 5
        assert collector._jobs_running == 5

    def test_error_aggregation(self):
        """Errors should be aggregated by type."""
        collector = JobMetricsCollector()

        # Submit and fail job with ValueError
        collector.job_submitted("job1", "test")
        collector.job_started("job1")
        collector.job_failed("job1", "ValueError: invalid input")

        summary = collector.get_summary()
        assert "ValueError" in summary["error_counts"]
        assert summary["error_counts"]["ValueError"] == 1

        # Add another error of same type
        collector.job_submitted("job2", "test")
        collector.job_started("job2")
        collector.job_failed("job2", "ValueError: another error")

        summary = collector.get_summary()
        assert summary["error_counts"]["ValueError"] == 2

    def test_tool_usage_tracking(self):
        """Tool usage should be tracked correctly."""
        collector = JobMetricsCollector()

        # Complete jobs with different tools
        for i, tool in enumerate(["answer", "answer", "research", "verify"]):
            collector.job_submitted(f"job{i}", f"query{i}")
            collector.job_started(f"job{i}")
            collector.job_completed(f"job{i}", 100.0, 0.9, tool)

        summary = collector.get_summary()
        assert summary["tool_usage"]["answer"] == 2
        assert summary["tool_usage"]["research"] == 1
        assert summary["tool_usage"]["verify"] == 1

    def test_job_cancellation(self):
        """Job cancellation should update metrics correctly."""
        collector = JobMetricsCollector()

        collector.job_submitted("job1", "test")
        collector.job_started("job1")
        collector.job_cancelled("job1")

        assert collector._jobs_running == 0
        assert collector._jobs_total["cancelled"] == 1

    def test_jobs_per_minute_calculation(self):
        """Jobs per minute should be calculated correctly."""
        collector = JobMetricsCollector(window_size_minutes=5)

        # Submit jobs
        for i in range(10):
            collector.job_submitted(f"job{i}", f"query{i}")

        # Should be 10 jobs / 5 minutes = 2 jobs/min
        jpm = collector.get_jobs_per_minute()
        assert jpm == 2.0

    def test_get_metrics_returns_list(self):
        """get_metrics should return a list of MetricValue objects."""
        collector = JobMetricsCollector()
        collector.job_submitted("job1", "test")

        metrics = collector.get_metrics()

        assert isinstance(metrics, list)
        assert all(isinstance(m, MetricValue) for m in metrics)

        # Should have jobs_total metric
        total_metrics = [m for m in metrics if m.name == "hololoom_jobs_total"]
        assert len(total_metrics) > 0


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_metric_value_creation(self):
        """MetricValue should be created with correct fields."""
        metric = MetricValue(
            name="test_metric",
            value=42.0,
            labels={"status": "success"},
            metric_type=MetricType.GAUGE,
            help_text="A test metric"
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.labels == {"status": "success"}
        assert metric.metric_type == MetricType.GAUGE
        assert metric.help_text == "A test metric"


class TestGlobalCollector:
    """Tests for global collector singleton."""

    def test_get_metrics_collector_creates_instance(self):
        """get_metrics_collector should create a singleton instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        # Should be the same instance
        assert collector1 is collector2

    def test_set_metrics_collector_replaces_instance(self):
        """set_metrics_collector should replace the global instance."""
        original = get_metrics_collector()
        new_collector = JobMetricsCollector()

        set_metrics_collector(new_collector)
        current = get_metrics_collector()

        assert current is new_collector
        assert current is not original

        # Restore original for other tests
        set_metrics_collector(original)


class TestLatencyPercentiles:
    """Tests for latency percentile reporting."""

    def test_percentiles_in_summary(self):
        """Summary should include p50, p90, p95, p99 percentiles."""
        collector = JobMetricsCollector()

        # Add some latencies
        for latency in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            collector.job_submitted(f"job{latency}", "test")
            collector.job_started(f"job{latency}")
            collector.job_completed(f"job{latency}", float(latency), 0.9, "answer")

        summary = collector.get_summary()

        assert "p50" in summary["latency_ms"]
        assert "p90" in summary["latency_ms"]
        assert "p95" in summary["latency_ms"]
        assert "p99" in summary["latency_ms"]

        # p50 should be around 250 (median of 50-500)
        assert 200 <= summary["latency_ms"]["p50"] <= 300

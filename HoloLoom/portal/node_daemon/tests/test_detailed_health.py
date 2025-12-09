"""
Unit tests for Node Daemon Detailed Health Monitoring.

Tests HealthMonitor and DetailedHealth functionality.
"""

import pytest
import time

from ..detailed_health import (
    HealthMonitor,
    DetailedHealth,
    HealthSample,
)


class TestHealthSample:
    """Tests for HealthSample dataclass."""

    def test_sample_creation(self):
        """Sample stores all metrics correctly."""
        sample = HealthSample(
            timestamp=1234567890.0,
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_mb=4096,
            job_queue_depth=5,
        )

        assert sample.timestamp == 1234567890.0
        assert sample.cpu_percent == 50.0
        assert sample.memory_percent == 60.0
        assert sample.memory_mb == 4096
        assert sample.job_queue_depth == 5


class TestDetailedHealth:
    """Tests for DetailedHealth dataclass."""

    def test_healthy_status(self):
        """Healthy status when no degradation."""
        health = DetailedHealth(healthy=True)

        assert health.healthy
        assert not health.degraded
        assert len(health.degradation_reasons) == 0

    def test_degraded_status(self):
        """Degraded status with reasons."""
        health = DetailedHealth(
            healthy=False,
            degraded=True,
            degradation_reasons=["High memory usage", "Memory leak suspected"],
        )

        assert not health.healthy
        assert health.degraded
        assert len(health.degradation_reasons) == 2

    def test_to_dict(self):
        """to_dict provides proper JSON structure."""
        health = DetailedHealth(
            healthy=True,
            cpu_percent=45.5,
            memory_percent=60.3,
            memory_mb=4096,
            memory_available_mb=2048,
            jobs_running=3,
            jobs_completed=100,
            uptime_seconds=3600.5,
            module_count=10,
            memory_trend=[50.0, 52.0, 55.0],
            cpu_trend=[40.0, 45.0, 42.0],
            memory_leak_suspected=False,
            estimated_available_slots=7,
        )

        d = health.to_dict()

        assert d["healthy"] is True
        assert d["resources"]["cpu_percent"] == 45.5
        assert d["resources"]["memory_percent"] == 60.3
        assert d["jobs"]["running"] == 3
        assert d["jobs"]["completed"] == 100
        assert d["system"]["uptime_seconds"] == 3600  # rounded (banker's rounding)
        assert len(d["trends"]["memory"]) == 3
        assert d["insights"]["memory_leak_suspected"] is False


class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    def test_record_sample(self):
        """Samples are recorded correctly."""
        monitor = HealthMonitor()

        monitor.record_sample(
            job_queue_depth=5,
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_mb=4096,
        )

        stats = monitor.get_stats()
        assert stats["sample_count"] == 1

    def test_sample_limit(self):
        """Max samples is enforced."""
        monitor = HealthMonitor(max_samples=3)

        for i in range(5):
            monitor.record_sample(
                job_queue_depth=i,
                cpu_percent=float(i * 10),
                memory_percent=float(i * 10),
                memory_mb=i * 1000,
            )

        stats = monitor.get_stats()
        assert stats["sample_count"] == 3

    def test_job_completion_tracking(self):
        """Job completions are tracked."""
        monitor = HealthMonitor()

        monitor.record_job_completion(100.0, success=True)
        monitor.record_job_completion(150.0, success=True)
        monitor.record_job_completion(200.0, success=False)

        # Record a sample to access via get_detailed_health
        monitor.record_sample(job_queue_depth=0, cpu_percent=0, memory_percent=0, memory_mb=0)
        health = monitor.get_detailed_health()

        assert health.jobs_completed == 2
        assert health.jobs_failed == 1
        # Average of 100, 150, 200 = 150
        assert health.avg_job_duration_ms == 150.0

    def test_module_error_tracking(self):
        """Module errors are tracked."""
        monitor = HealthMonitor()

        monitor.record_module_error("Failed to load module A")
        monitor.record_module_error("Failed to load module B")

        stats = monitor.get_stats()
        assert stats["module_errors_count"] == 2

    def test_memory_leak_detection_triggered(self):
        """Memory leak is detected when memory increases monotonically."""
        monitor = HealthMonitor(memory_leak_threshold=3)

        # Simulate monotonic memory increase
        for i in range(5):
            monitor.record_sample(
                job_queue_depth=0,
                cpu_percent=50.0,
                memory_percent=50.0 + i * 5,  # 50, 55, 60, 65, 70
                memory_mb=4000 + i * 500,
            )

        health = monitor.get_detailed_health()
        assert health.memory_leak_suspected is True
        assert "memory leak" in " ".join(health.degradation_reasons).lower()

    def test_memory_leak_not_triggered_with_drops(self):
        """Memory leak not detected when memory decreases."""
        monitor = HealthMonitor(memory_leak_threshold=3)

        # Simulate non-monotonic memory usage
        memory_values = [50.0, 55.0, 52.0, 58.0, 55.0]
        for mem in memory_values:
            monitor.record_sample(
                job_queue_depth=0,
                cpu_percent=50.0,
                memory_percent=mem,
                memory_mb=int(mem * 100),
            )

        health = monitor.get_detailed_health()
        assert health.memory_leak_suspected is False

    def test_high_cpu_sustained_detection(self):
        """Sustained high CPU is detected."""
        monitor = HealthMonitor(high_cpu_threshold=80.0)

        # 3 consecutive high CPU samples
        for _ in range(3):
            monitor.record_sample(
                job_queue_depth=0,
                cpu_percent=85.0,
                memory_percent=50.0,
                memory_mb=4000,
            )

        health = monitor.get_detailed_health()
        assert health.high_cpu_sustained is True
        assert "high cpu" in " ".join(health.degradation_reasons).lower()

    def test_high_cpu_not_sustained(self):
        """High CPU not flagged if not sustained."""
        monitor = HealthMonitor(high_cpu_threshold=80.0)

        # Alternating high and low CPU
        cpu_values = [85.0, 50.0, 85.0]
        for cpu in cpu_values:
            monitor.record_sample(
                job_queue_depth=0,
                cpu_percent=cpu,
                memory_percent=50.0,
                memory_mb=4000,
            )

        health = monitor.get_detailed_health()
        assert health.high_cpu_sustained is False

    def test_available_slots_calculation(self):
        """Available slots calculated correctly."""
        monitor = HealthMonitor()
        monitor.record_sample(job_queue_depth=0, cpu_percent=0, memory_percent=0, memory_mb=0)

        health = monitor.get_detailed_health(
            jobs_running=3,
            max_concurrent=10
        )

        assert health.estimated_available_slots == 7

    def test_critical_memory_triggers_unhealthy(self):
        """Critical memory (>90%) makes node unhealthy."""
        monitor = HealthMonitor()
        monitor.record_sample(
            job_queue_depth=0,
            cpu_percent=50.0,
            memory_percent=92.0,  # Critical
            memory_mb=8000,
        )

        health = monitor.get_detailed_health()
        assert health.healthy is False
        assert "Critical memory" in " ".join(health.degradation_reasons)

    def test_high_queue_depth_triggers_degradation(self):
        """High job queue depth causes degradation."""
        monitor = HealthMonitor()
        monitor.record_sample(
            job_queue_depth=15,  # > 10
            cpu_percent=50.0,
            memory_percent=50.0,
            memory_mb=4000,
        )

        health = monitor.get_detailed_health()
        assert health.degraded is True
        assert "queue depth" in " ".join(health.degradation_reasons).lower()

    def test_trends_captured(self):
        """Memory and CPU trends are captured."""
        monitor = HealthMonitor()

        for i in range(5):
            monitor.record_sample(
                job_queue_depth=0,
                cpu_percent=40.0 + i,
                memory_percent=50.0 + i,
                memory_mb=4000,
            )

        health = monitor.get_detailed_health()

        assert len(health.memory_trend) == 5
        assert len(health.cpu_trend) == 5
        assert health.memory_trend == [50.0, 51.0, 52.0, 53.0, 54.0]
        assert health.cpu_trend == [40.0, 41.0, 42.0, 43.0, 44.0]

    def test_uptime_tracked(self):
        """Uptime is tracked from monitor creation."""
        monitor = HealthMonitor()
        time.sleep(0.1)  # Wait a bit
        monitor.record_sample(job_queue_depth=0, cpu_percent=0, memory_percent=0, memory_mb=0)

        health = monitor.get_detailed_health()

        assert health.uptime_seconds >= 0.1

    def test_reset_clears_all_data(self):
        """Reset clears all tracked data."""
        monitor = HealthMonitor()

        # Add some data
        monitor.record_sample(job_queue_depth=5, cpu_percent=50, memory_percent=60, memory_mb=4000)
        monitor.record_job_completion(100.0)
        monitor.record_module_error("Test error")

        # Reset
        monitor.reset()

        stats = monitor.get_stats()
        assert stats["sample_count"] == 0
        assert stats["jobs_completed"] == 0
        assert stats["module_errors_count"] == 0

    def test_stats_returns_diagnostic_info(self):
        """get_stats returns useful diagnostic information."""
        monitor = HealthMonitor(sample_interval=30.0, max_samples=12)

        monitor.record_sample(job_queue_depth=0, cpu_percent=0, memory_percent=0, memory_mb=0)
        monitor.record_job_completion(100.0, success=True)
        monitor.record_job_completion(50.0, success=False)

        stats = monitor.get_stats()

        assert stats["sample_count"] == 1
        assert stats["max_samples"] == 12
        assert stats["sample_interval"] == 30.0
        assert stats["uptime_seconds"] >= 0
        assert stats["jobs_completed"] == 1
        assert stats["jobs_failed"] == 1


class TestHealthyVsDegradedVsUnhealthy:
    """Tests for health status transitions."""

    def test_fully_healthy(self):
        """Node is healthy with good metrics."""
        monitor = HealthMonitor()
        monitor.record_sample(
            job_queue_depth=2,
            cpu_percent=30.0,
            memory_percent=50.0,
            memory_mb=4000,
        )

        health = monitor.get_detailed_health()

        assert health.healthy is True
        assert health.degraded is False
        assert len(health.degradation_reasons) == 0

    def test_degraded_but_not_unhealthy(self):
        """Node is degraded but not unhealthy."""
        monitor = HealthMonitor(high_memory_threshold=70.0)
        monitor.record_sample(
            job_queue_depth=2,
            cpu_percent=30.0,
            memory_percent=75.0,  # High but not critical
            memory_mb=6000,
        )

        health = monitor.get_detailed_health()

        assert health.healthy is False  # Has degradation reasons
        assert health.degraded is True
        assert len(health.degradation_reasons) > 0

    def test_unhealthy_critical(self):
        """Node is unhealthy with critical memory."""
        monitor = HealthMonitor()
        monitor.record_sample(
            job_queue_depth=2,
            cpu_percent=30.0,
            memory_percent=95.0,  # Critical
            memory_mb=8000,
        )

        health = monitor.get_detailed_health()

        assert health.healthy is False
        assert "Critical" in " ".join(health.degradation_reasons)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

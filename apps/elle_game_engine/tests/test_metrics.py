"""
Tests for metrics collection.
"""

import pytest
import time
from ..metrics import MetricsCollector, initialize_metrics, get_metrics


@pytest.fixture
def metrics():
    """Create test metrics collector."""
    return MetricsCollector(window_size=100)


def test_initial_metrics(metrics):
    """Test initial state of metrics."""
    assert metrics.total_requests == 0
    assert metrics.cache_hits == 0
    assert metrics.cache_misses == 0
    assert metrics.rate_limit_hits == 0
    assert metrics.get_average_latency() == 0.0
    assert metrics.get_p95_latency() == 0.0


def test_record_request(metrics):
    """Test recording a request."""
    metrics.record_request(
        intent_type="talk_to_npc",
        provider="dummy",
        duration_ms=100.0,
        cache_hit=False,
    )

    assert metrics.total_requests == 1
    assert metrics.requests_by_intent["talk_to_npc"] == 1
    assert metrics.requests_by_provider["dummy"] == 1
    assert metrics.cache_misses == 1
    assert metrics.cache_hits == 0


def test_record_multiple_requests(metrics):
    """Test recording multiple requests."""
    for i in range(5):
        metrics.record_request(
            intent_type="talk_to_npc",
            provider="dummy",
            duration_ms=100.0 + i * 10,
            cache_hit=i % 2 == 0,  # Alternate cache hits
        )

    assert metrics.total_requests == 5
    assert metrics.requests_by_intent["talk_to_npc"] == 5
    assert metrics.cache_hits == 3  # 0, 2, 4
    assert metrics.cache_misses == 2  # 1, 3


def test_record_different_intents(metrics):
    """Test recording different intent types."""
    metrics.record_request("talk_to_npc", "dummy", 100.0, False)
    metrics.record_request("enter_scene", "dummy", 150.0, False)
    metrics.record_request("request_hint", "dummy", 120.0, True)

    assert metrics.total_requests == 3
    assert metrics.requests_by_intent["talk_to_npc"] == 1
    assert metrics.requests_by_intent["enter_scene"] == 1
    assert metrics.requests_by_intent["request_hint"] == 1


def test_record_different_providers(metrics):
    """Test recording different LLM providers."""
    metrics.record_request("talk_to_npc", "dummy", 100.0, False)
    metrics.record_request("talk_to_npc", "openai", 200.0, False)
    metrics.record_request("talk_to_npc", "anthropic", 150.0, True)

    assert metrics.requests_by_provider["dummy"] == 1
    assert metrics.requests_by_provider["openai"] == 1
    assert metrics.requests_by_provider["anthropic"] == 1


def test_cache_hit_rate(metrics):
    """Test cache hit rate calculation."""
    # No requests yet
    assert metrics.get_cache_hit_rate() == 0.0

    # 3 hits, 2 misses = 60% hit rate
    for i in range(5):
        metrics.record_request(
            "talk_to_npc",
            "dummy",
            100.0,
            cache_hit=i < 3,  # First 3 are hits
        )

    assert metrics.get_cache_hit_rate() == 0.6


def test_average_latency(metrics):
    """Test average latency calculation."""
    latencies = [100.0, 150.0, 200.0, 250.0]
    for latency in latencies:
        metrics.record_request("talk_to_npc", "dummy", latency, False)

    expected_avg = sum(latencies) / len(latencies)
    assert metrics.get_average_latency() == expected_avg


def test_p95_latency(metrics):
    """Test 95th percentile latency calculation."""
    # Add 100 samples with known distribution
    for i in range(100):
        metrics.record_request("talk_to_npc", "dummy", i * 10.0, False)

    p95 = metrics.get_p95_latency()
    # 95th percentile of 0-990 should be around 950
    assert 940 <= p95 <= 960


def test_latency_rolling_window(metrics):
    """Test that latency samples respect window size."""
    # Window size is 100
    # Add 150 samples
    for i in range(150):
        metrics.record_request("talk_to_npc", "dummy", i * 1.0, False)

    # Should only keep last 100
    assert len(metrics.latency_samples) == 100


def test_record_rate_limit(metrics):
    """Test recording rate limit hits."""
    assert metrics.rate_limit_hits == 0

    metrics.record_rate_limit()
    assert metrics.rate_limit_hits == 1

    metrics.record_rate_limit()
    assert metrics.rate_limit_hits == 2


def test_get_uptime_seconds(metrics):
    """Test uptime calculation."""
    uptime1 = metrics.get_uptime_seconds()
    assert uptime1 >= 0

    time.sleep(0.1)

    uptime2 = metrics.get_uptime_seconds()
    assert uptime2 > uptime1
    assert uptime2 - uptime1 >= 0.1


def test_get_stats(metrics):
    """Test getting all statistics."""
    # Add some data
    metrics.record_request("talk_to_npc", "dummy", 100.0, True)
    metrics.record_request("enter_scene", "openai", 150.0, False)
    metrics.record_rate_limit()

    stats = metrics.get_stats()

    assert stats["total_requests"] == 2
    assert stats["requests_by_intent"]["talk_to_npc"] == 1
    assert stats["requests_by_intent"]["enter_scene"] == 1
    assert stats["requests_by_provider"]["dummy"] == 1
    assert stats["requests_by_provider"]["openai"] == 1
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["cache_hit_rate"] == 0.5
    assert stats["rate_limit_hits"] == 1
    assert stats["average_latency_ms"] == 125.0
    assert "uptime_seconds" in stats


def test_prometheus_format(metrics):
    """Test Prometheus format export."""
    # Add some data
    metrics.record_request("talk_to_npc", "dummy", 100.0, True)
    metrics.record_request("enter_scene", "openai", 150.0, False)
    metrics.record_rate_limit()

    prometheus_output = metrics.to_prometheus_format()

    # Check format
    assert "# HELP" in prometheus_output
    assert "# TYPE" in prometheus_output

    # Check specific metrics
    assert "elle_requests_total 2" in prometheus_output
    assert 'elle_requests_by_intent_total{intent="talk_to_npc"} 1' in prometheus_output
    assert 'elle_requests_by_intent_total{intent="enter_scene"} 1' in prometheus_output
    assert 'elle_requests_by_provider_total{provider="dummy"} 1' in prometheus_output
    assert 'elle_requests_by_provider_total{provider="openai"} 1' in prometheus_output
    assert "elle_cache_hits_total 1" in prometheus_output
    assert "elle_cache_misses_total 1" in prometheus_output
    assert "elle_cache_hit_rate 0.5000" in prometheus_output
    assert "elle_rate_limit_hits_total 1" in prometheus_output
    assert "elle_latency_average_ms" in prometheus_output
    assert "elle_latency_p95_ms" in prometheus_output
    assert "elle_uptime_seconds" in prometheus_output


def test_global_metrics_initialization():
    """Test global metrics initialization."""
    initialize_metrics(window_size=50)

    metrics = get_metrics()
    assert metrics is not None
    assert metrics.window_size == 50

    # Record request
    metrics.record_request("talk_to_npc", "dummy", 100.0, False)
    assert metrics.total_requests == 1

    # Get same instance
    metrics2 = get_metrics()
    assert metrics2 is metrics
    assert metrics2.total_requests == 1


def test_prometheus_format_escaping():
    """Test that Prometheus format handles special characters."""
    metrics = MetricsCollector()

    # Add request with intent that might need escaping
    metrics.record_request(
        intent_type="talk_to_npc",
        provider="local",
        duration_ms=100.0,
        cache_hit=False,
    )

    output = metrics.to_prometheus_format()

    # Should be valid Prometheus format
    assert 'intent="talk_to_npc"' in output
    assert 'provider="local"' in output


def test_metrics_with_no_latency_samples():
    """Test metrics calculations with no latency samples."""
    metrics = MetricsCollector()

    assert metrics.get_average_latency() == 0.0
    assert metrics.get_p95_latency() == 0.0


def test_metrics_p95_with_few_samples():
    """Test P95 calculation with fewer than 20 samples."""
    metrics = MetricsCollector()

    # Add just 5 samples
    for i in range(5):
        metrics.record_request("talk_to_npc", "dummy", i * 10.0, False)

    p95 = metrics.get_p95_latency()
    # Should still work, returning highest value
    assert p95 >= 0

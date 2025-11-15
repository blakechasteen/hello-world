"""
Prometheus-style metrics for Elle Game Engine.

Tracks operational metrics for monitoring and observability.
"""

import time
import os
from typing import Dict, List
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class LatencySample:
    """Single latency measurement."""
    timestamp: float
    duration_ms: float


class MetricsCollector:
    """
    Simple Prometheus-style metrics collector.

    Tracks:
    - Total requests
    - Requests by intent type
    - Requests by LLM provider
    - Cache hit/miss rates
    - Response time statistics
    - Rate limit hits
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector.

        Args:
            window_size: Number of recent samples to keep for statistics
        """
        self.window_size = window_size

        # Counters
        self.total_requests = 0
        self.requests_by_intent: Dict[str, int] = defaultdict(int)
        self.requests_by_provider: Dict[str, int] = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.rate_limit_hits = 0

        # Latency tracking (rolling window)
        self.latency_samples: deque[LatencySample] = deque(maxlen=window_size)

        # Start time
        self.start_time = time.time()

    def record_request(
        self,
        intent_type: str,
        provider: str,
        duration_ms: float,
        cache_hit: bool,
    ):
        """
        Record a request.

        Args:
            intent_type: Type of player intent
            provider: LLM provider used
            duration_ms: Request duration in milliseconds
            cache_hit: Whether cache was hit
        """
        self.total_requests += 1
        self.requests_by_intent[intent_type] += 1
        self.requests_by_provider[provider] += 1

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # Record latency
        self.latency_samples.append(
            LatencySample(timestamp=time.time(), duration_ms=duration_ms)
        )

    def record_rate_limit(self):
        """Record a rate limit hit."""
        self.rate_limit_hits += 1

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def get_average_latency(self) -> float:
        """Calculate average latency in milliseconds."""
        if not self.latency_samples:
            return 0.0

        total = sum(sample.duration_ms for sample in self.latency_samples)
        return total / len(self.latency_samples)

    def get_p95_latency(self) -> float:
        """Calculate 95th percentile latency in milliseconds."""
        if not self.latency_samples:
            return 0.0

        durations = sorted(sample.duration_ms for sample in self.latency_samples)
        index = int(len(durations) * 0.95)
        return durations[index] if index < len(durations) else durations[-1]

    def get_uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    def get_stats(self) -> Dict:
        """Get all metrics as dictionary."""
        return {
            "total_requests": self.total_requests,
            "requests_by_intent": dict(self.requests_by_intent),
            "requests_by_provider": dict(self.requests_by_provider),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.get_cache_hit_rate(),
            "rate_limit_hits": self.rate_limit_hits,
            "average_latency_ms": self.get_average_latency(),
            "p95_latency_ms": self.get_p95_latency(),
            "uptime_seconds": self.get_uptime_seconds(),
        }

    def to_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Total requests
        lines.append("# HELP elle_requests_total Total number of requests")
        lines.append("# TYPE elle_requests_total counter")
        lines.append(f"elle_requests_total {self.total_requests}")

        # Requests by intent
        lines.append("# HELP elle_requests_by_intent_total Requests by intent type")
        lines.append("# TYPE elle_requests_by_intent_total counter")
        for intent, count in self.requests_by_intent.items():
            lines.append(f'elle_requests_by_intent_total{{intent="{intent}"}} {count}')

        # Requests by provider
        lines.append("# HELP elle_requests_by_provider_total Requests by LLM provider")
        lines.append("# TYPE elle_requests_by_provider_total counter")
        for provider, count in self.requests_by_provider.items():
            lines.append(f'elle_requests_by_provider_total{{provider="{provider}"}} {count}')

        # Cache metrics
        lines.append("# HELP elle_cache_hits_total Cache hit count")
        lines.append("# TYPE elle_cache_hits_total counter")
        lines.append(f"elle_cache_hits_total {self.cache_hits}")

        lines.append("# HELP elle_cache_misses_total Cache miss count")
        lines.append("# TYPE elle_cache_misses_total counter")
        lines.append(f"elle_cache_misses_total {self.cache_misses}")

        lines.append("# HELP elle_cache_hit_rate Cache hit rate (0.0-1.0)")
        lines.append("# TYPE elle_cache_hit_rate gauge")
        lines.append(f"elle_cache_hit_rate {self.get_cache_hit_rate():.4f}")

        # Latency metrics
        lines.append("# HELP elle_latency_average_ms Average response time in milliseconds")
        lines.append("# TYPE elle_latency_average_ms gauge")
        lines.append(f"elle_latency_average_ms {self.get_average_latency():.2f}")

        lines.append("# HELP elle_latency_p95_ms 95th percentile response time in milliseconds")
        lines.append("# TYPE elle_latency_p95_ms gauge")
        lines.append(f"elle_latency_p95_ms {self.get_p95_latency():.2f}")

        # Rate limiting
        lines.append("# HELP elle_rate_limit_hits_total Rate limit rejections")
        lines.append("# TYPE elle_rate_limit_hits_total counter")
        lines.append(f"elle_rate_limit_hits_total {self.rate_limit_hits}")

        # Uptime
        lines.append("# HELP elle_uptime_seconds Service uptime in seconds")
        lines.append("# TYPE elle_uptime_seconds counter")
        lines.append(f"elle_uptime_seconds {self.get_uptime_seconds():.0f}")

        return "\n".join(lines) + "\n"


# Global metrics instance (initialized by service)
_metrics: MetricsCollector = None


def get_metrics() -> MetricsCollector:
    """Get global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def initialize_metrics(window_size: int = 1000):
    """
    Initialize global metrics collector.

    Args:
        window_size: Number of recent samples to keep
    """
    global _metrics
    _metrics = MetricsCollector(window_size=window_size)

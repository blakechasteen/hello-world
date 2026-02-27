"""
Rate Limiting & Statistics Middleware
=====================================

Provides rate limiting and server statistics tracking for the API.

Classes:
    RateLimiter: Sliding window rate limiter
    ServerStats: Query statistics collector

SECURITY: Uses asyncio.Lock for proper async concurrency control.
"""

import asyncio
from collections import defaultdict, deque
from time import time
from typing import Dict, Any


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.

    Tracks requests per IP and enforces configurable limits.

    SECURITY: Uses asyncio.Lock for proper async concurrency control.
    Using threading.Lock in async code can block the event loop and cause
    performance issues or deadlocks. asyncio.Lock is the correct choice
    for async functions.

    Attributes:
        max_requests: Maximum requests allowed in window
        window_seconds: Time window in seconds
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()  # SECURITY: Async-safe access

    async def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client is within rate limit.

        Args:
            client_id: Client identifier (IP address)

        Returns:
            True if within limit, False if exceeded
        """
        async with self._lock:  # SECURITY: Async-safe critical section
            now = time()
            cutoff = now - self.window_seconds

            # Remove old requests outside window
            while self.requests[client_id] and self.requests[client_id][0] < cutoff:
                self.requests[client_id].popleft()

            # Check if limit exceeded
            if len(self.requests[client_id]) >= self.max_requests:
                return False

            # Record this request
            self.requests[client_id].append(now)
            return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time()
        cutoff = now - self.window_seconds

        # Count requests in current window
        count = sum(1 for t in self.requests[client_id] if t >= cutoff)
        return max(0, self.max_requests - count)


class ServerStats:
    """
    Track server statistics.

    Monitors uptime, query counts, latencies, and error rates.

    Attributes:
        start_time: Server start timestamp
        total_queries: Total queries processed
        successful_queries: Successful query count
        failed_queries: Failed query count
    """

    def __init__(self):
        self.start_time = time()
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.latencies: deque = deque(maxlen=1000)  # Last 1000 latencies
        self.queries_by_mode: Dict[str, int] = defaultdict(int)
        self.errors_by_type: Dict[str, int] = defaultdict(int)

    def record_query(self, mode: str, latency_ms: float, success: bool):
        """Record a query completion."""
        self.total_queries += 1
        self.latencies.append(latency_ms)
        self.queries_by_mode[mode] += 1

        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1

    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.errors_by_type[error_type] += 1

    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return time() - self.start_time

    def get_avg_latency(self) -> float:
        """Get average latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def get_p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx] if idx < len(sorted_latencies) else sorted_latencies[-1]

    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_queries == 0:
            return 100.0
        return (self.successful_queries / self.total_queries) * 100

    def get_stats_dict(self) -> Dict[str, Any]:
        """Get all stats as dictionary."""
        return {
            "uptime_seconds": self.get_uptime(),
            "uptime_formatted": self._format_uptime(self.get_uptime()),
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": round(self.get_success_rate(), 2),
            "avg_latency_ms": round(self.get_avg_latency(), 2),
            "p95_latency_ms": round(self.get_p95_latency(), 2),
            "queries_by_mode": dict(self.queries_by_mode),
            "errors_by_type": dict(self.errors_by_type)
        }

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime as human-readable string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"

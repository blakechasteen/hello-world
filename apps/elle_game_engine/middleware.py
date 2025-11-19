"""
Rate limiting middleware for Elle Game Engine.

Implements sliding window rate limiting for both per-IP and per-session limits.
"""

import time
import os
from typing import Dict, Tuple
from collections import defaultdict, deque
from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.

    Tracks both per-IP and per-session limits:
    - Per-IP: Prevents single IP from overwhelming service
    - Per-session: Prevents single game session from excessive requests

    Uses in-memory storage with automatic cleanup.
    """

    def __init__(
        self,
        app,
        per_minute_limit: int = 60,
        per_hour_limit: int = 100,
    ):
        """
        Initialize rate limiter.

        Args:
            app: FastAPI application
            per_minute_limit: Max requests per minute per IP
            per_hour_limit: Max requests per hour per session
        """
        super().__init__(app)
        self.per_minute_limit = per_minute_limit
        self.per_hour_limit = per_hour_limit

        # IP tracking: {ip: deque[(timestamp, session_id)]}
        self.ip_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=per_minute_limit + 10))

        # Session tracking: {session_id: deque[timestamp]}
        self.session_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=per_hour_limit + 10))

        # Last cleanup time
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes

        # Metrics
        self.total_rate_limits = 0
        self.ip_rate_limits = 0
        self.session_rate_limits = 0

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware/endpoint

        Returns:
            Response (200 if allowed, 429 JSONResponse if rate limited)
        """
        # Skip rate limiting for health checks
        if request.url.path == "/health" or request.url.path == "/metrics":
            return await call_next(request)

        # Cleanup old entries periodically
        self._cleanup_if_needed()

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Get session ID from header (optional)
        session_id = request.headers.get("X-Session-ID", f"ip_{client_ip}")

        # Check both limits before adding to trackers
        ip_allowed = self._is_ip_allowed(client_ip)
        session_allowed = self._is_session_allowed(session_id)

        if not ip_allowed:
            self.total_rate_limits += 1
            self.ip_rate_limits += 1
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"Rate limit exceeded: {self.per_minute_limit} requests per minute per IP"},
                headers={"Retry-After": "60"},
            )

        if not session_allowed:
            self.total_rate_limits += 1
            self.session_rate_limits += 1
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"Rate limit exceeded: {self.per_hour_limit} requests per hour per session"},
                headers={"Retry-After": "3600"},
            )

        # Both limits OK - record request
        self._record_request(client_ip, session_id)

        # Process request
        response = await call_next(request)
        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check X-Forwarded-For header (from proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Fallback to direct client
        return request.client.host if request.client else "unknown"

    def _is_ip_allowed(self, client_ip: str) -> bool:
        """
        Check if IP is within rate limit (doesn't record).

        Args:
            client_ip: Client IP address

        Returns:
            True if within limit, False if exceeded
        """
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Get requests for this IP
        ip_queue = self.ip_requests[client_ip]

        # Remove expired entries
        while ip_queue and ip_queue[0][0] < window_start:
            ip_queue.popleft()

        # Check if limit exceeded
        return len(ip_queue) < self.per_minute_limit

    def _is_session_allowed(self, session_id: str) -> bool:
        """
        Check if session is within rate limit (doesn't record).

        Args:
            session_id: Session identifier

        Returns:
            True if within limit, False if exceeded
        """
        now = time.time()
        window_start = now - 3600  # 1 hour window

        # Get requests for this session
        session_queue = self.session_requests[session_id]

        # Remove expired entries
        while session_queue and session_queue[0] < window_start:
            session_queue.popleft()

        # Check if limit exceeded
        return len(session_queue) < self.per_hour_limit

    def _record_request(self, client_ip: str, session_id: str):
        """
        Record a request in both IP and session trackers.

        Args:
            client_ip: Client IP address
            session_id: Session identifier
        """
        now = time.time()

        # Record in IP tracker
        self.ip_requests[client_ip].append((now, session_id))

        # Record in session tracker
        self.session_requests[session_id].append(now)

    def _check_ip_limit(self, client_ip: str, session_id: str) -> bool:
        """
        Check if IP is within rate limit (sliding window) and record if allowed.

        Args:
            client_ip: Client IP address
            session_id: Session identifier

        Returns:
            True if within limit, False if exceeded
        """
        if not self._is_ip_allowed(client_ip):
            return False

        # Record this request
        self._record_request(client_ip, session_id)
        return True

    def _check_session_limit(self, session_id: str) -> bool:
        """
        Check if session is within rate limit (sliding window).

        Note: This method checks only - use _check_ip_limit to record requests.

        Args:
            session_id: Session identifier

        Returns:
            True if within limit, False if exceeded
        """
        return self._is_session_allowed(session_id)

    def _cleanup_if_needed(self):
        """Clean up old entries to prevent memory bloat."""
        now = time.time()

        if now - self.last_cleanup < self.cleanup_interval:
            return

        # Cleanup IP tracking (remove entries older than 1 minute)
        window_start = now - 60
        ips_to_remove = []

        for ip, queue in self.ip_requests.items():
            while queue and queue[0][0] < window_start:
                queue.popleft()

            # Remove empty queues
            if not queue:
                ips_to_remove.append(ip)

        for ip in ips_to_remove:
            del self.ip_requests[ip]

        # Cleanup session tracking (remove entries older than 1 hour)
        session_start = now - 3600
        sessions_to_remove = []

        for session_id, queue in self.session_requests.items():
            while queue and queue[0] < session_start:
                queue.popleft()

            # Remove empty queues
            if not queue:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.session_requests[session_id]

        self.last_cleanup = now

    def get_stats(self) -> Dict:
        """Get rate limiting statistics."""
        return {
            "total_rate_limits": self.total_rate_limits,
            "ip_rate_limits": self.ip_rate_limits,
            "session_rate_limits": self.session_rate_limits,
            "active_ips": len(self.ip_requests),
            "active_sessions": len(self.session_requests),
            "per_minute_limit": self.per_minute_limit,
            "per_hour_limit": self.per_hour_limit,
        }


def create_rate_limiter() -> RateLimitMiddleware:
    """
    Create rate limiter from environment variables.

    Environment Variables:
        ELLE_RATE_LIMIT_PER_MINUTE: Requests per minute per IP (default: 60)
        ELLE_RATE_LIMIT_PER_HOUR: Requests per hour per session (default: 100)

    Returns:
        Configured rate limiter middleware
    """
    per_minute = int(os.getenv("ELLE_RATE_LIMIT_PER_MINUTE", "60"))
    per_hour = int(os.getenv("ELLE_RATE_LIMIT_PER_HOUR", "100"))

    # Note: app parameter will be set by FastAPI when adding middleware
    return RateLimitMiddleware(
        app=None,  # Will be set by FastAPI
        per_minute_limit=per_minute,
        per_hour_limit=per_hour,
    )

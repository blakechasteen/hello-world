"""
Rate Limiting Middleware
"""

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict
import asyncio

from ..config import get_settings

settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm"""

    def __init__(self, app, requests_per_minute: int = 60, burst: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.buckets: Dict[str, dict] = defaultdict(lambda: {
            "tokens": burst,
            "last_update": datetime.utcnow()
        })

        # Cleanup old buckets periodically
        asyncio.create_task(self._cleanup_buckets())

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        if not settings.rate_limit_enabled:
            return await call_next(request)

        # Get client identifier (IP or API key)
        client_id = self._get_client_id(request)

        # Check rate limit
        if not self._check_rate_limit(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        bucket = self.buckets[client_id]
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket["tokens"]))

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        # Try to get API key first
        api_key = request.headers.get(settings.api_key_header)
        if api_key:
            return f"key:{api_key}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0]}"

        return f"ip:{request.client.host}"

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit"""
        bucket = self.buckets[client_id]
        now = datetime.utcnow()

        # Refill tokens based on time passed
        time_passed = (now - bucket["last_update"]).total_seconds()
        tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
        bucket["tokens"] = min(self.burst, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

        # Check if we have tokens available
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True

        return False

    async def _cleanup_buckets(self):
        """Periodically cleanup old bucket entries"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=10)

            # Remove buckets that haven't been used in 10 minutes
            to_remove = [
                client_id for client_id, bucket in self.buckets.items()
                if bucket["last_update"] < cutoff
            ]

            for client_id in to_remove:
                del self.buckets[client_id]

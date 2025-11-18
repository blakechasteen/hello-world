"""
API Middleware
===============

Authentication, rate limiting, CORS, logging, and error handling middleware.

Author: HoloLoom Team
Date: 2025-11-18
"""

import time
import logging
from typing import Callable, Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib
import os

logger = logging.getLogger(__name__)


# ============================================================================
# API Key Authentication
# ============================================================================

class APIKeyAuth:
    """
    Simple API key authentication.

    In production, use proper secret management (AWS Secrets Manager, etc).
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize API key auth.

        Args:
            api_keys: Dict[key_id -> hashed_key]. If None, loads from env.
        """
        self.api_keys = api_keys or self._load_keys_from_env()
        self.enabled = bool(self.api_keys)

    def _load_keys_from_env(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        keys = {}
        api_key = os.getenv("HOLOLOOM_API_KEY")
        if api_key:
            # Hash the key for storage
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            keys["default"] = key_hash
        return keys

    def validate_key(self, provided_key: str) -> bool:
        """Validate provided API key."""
        if not self.enabled:
            return True  # Auth disabled

        key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        return key_hash in self.api_keys.values()

    async def __call__(self, request: Request) -> Optional[str]:
        """
        Middleware callable for FastAPI dependency injection.

        Returns:
            Key ID if valid, raises HTTPException if invalid.
        """
        if not self.enabled:
            return None

        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Include X-API-Key header."
            )

        if not self.validate_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        # Find key ID (for logging/metrics)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        for key_id, stored_hash in self.api_keys.items():
            if stored_hash == key_hash:
                return key_id

        return "unknown"


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter.

    In production, use Redis-backed rate limiting for distributed systems.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Sustained request rate
            burst_size: Maximum burst allowed
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.buckets: Dict[str, Dict] = defaultdict(
            lambda: {
                "tokens": burst_size,
                "last_refill": datetime.now()
            }
        )

    def _refill_tokens(self, bucket: Dict):
        """Refill tokens based on elapsed time."""
        now = datetime.now()
        elapsed = (now - bucket["last_refill"]).total_seconds()

        # Add tokens based on rate
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        bucket["tokens"] = min(
            self.burst_size,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = now

    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if request is allowed under rate limit.

        Args:
            client_id: Client identifier (IP, API key, user ID, etc)

        Returns:
            True if allowed, False if rate limited
        """
        bucket = self.buckets[client_id]
        self._refill_tokens(bucket)

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True

        return False

    def get_retry_after(self, client_id: str) -> float:
        """Get seconds until next request allowed."""
        bucket = self.buckets[client_id]
        tokens_needed = 1.0 - bucket["tokens"]
        seconds_per_token = 60.0 / self.requests_per_minute
        return tokens_needed * seconds_per_token


# ============================================================================
# Request Logging Middleware
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing information."""

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request and log."""
        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Add timing header
            response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

            # Log response
            logger.info(
                f"Response: {response.status_code} ({duration_ms:.2f}ms)",
                extra={
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "path": request.url.path
                }
            )

            return response

        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed: {exc} ({duration_ms:.2f}ms)",
                extra={
                    "error": str(exc),
                    "duration_ms": duration_ms,
                    "path": request.url.path
                },
                exc_info=True
            )
            raise


# ============================================================================
# Rate Limiting Middleware
# ============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next: Callable):
        """Check rate limit before processing request."""
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)

        # Get client identifier (prefer API key, fallback to IP)
        client_id = request.headers.get("X-API-Key", "")
        if not client_id and request.client:
            client_id = request.client.host
        if not client_id:
            client_id = "unknown"

        # Check rate limit
        if not self.rate_limiter.check_rate_limit(client_id):
            retry_after = self.rate_limiter.get_retry_after(client_id)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Try again in {retry_after:.1f} seconds",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(int(retry_after) + 1)
                }
            )

        return await call_next(request)


# ============================================================================
# Error Handling
# ============================================================================

async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(
        f"Unhandled exception: {exc}",
        extra={
            "path": request.url.path,
            "method": request.method
        },
        exc_info=True
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# CORS Configuration
# ============================================================================

def configure_cors(app):
    """
    Configure CORS middleware.

    In production, restrict origins to your frontend domains.
    """
    origins = os.getenv("CORS_ORIGINS", "*").split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-Rate-Limit-Remaining"]
    )


# ============================================================================
# Middleware Factory
# ============================================================================

def setup_middleware(app):
    """
    Setup all middleware for the application.

    Args:
        app: FastAPI application instance
    """
    # CORS (must be first)
    configure_cors(app)

    # Rate limiting
    rate_limiter = RateLimiter(
        requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
        burst_size=int(os.getenv("RATE_LIMIT_BURST", "10"))
    )
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Exception handlers
    from fastapi.exceptions import HTTPException
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Middleware setup complete")

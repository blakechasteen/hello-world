#!/usr/bin/env python3
"""
Authentication Middleware for HoloLoom SaaS API

Provides:
- API key validation via X-API-Key header
- Request signing validation (HMAC-SHA256)
- Rate limiting per API key
- Customer context injection into requests
"""

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiter (Token Bucket)
# ============================================================================

@dataclass
class RateLimitState:
    """Per-key rate limit state using token bucket algorithm"""
    tokens: float
    last_update: float
    qps_limit: float

    def consume(self) -> bool:
        """
        Try to consume a token. Returns True if allowed, False if rate limited.
        Refills tokens based on time elapsed since last update.
        """
        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens based on time elapsed (up to qps_limit)
        self.tokens = min(self.qps_limit, self.tokens + elapsed * self.qps_limit)
        self.last_update = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def time_until_available(self) -> float:
        """Calculate seconds until next token is available"""
        if self.tokens >= 1.0:
            return 0.0
        tokens_needed = 1.0 - self.tokens
        return tokens_needed / self.qps_limit


class RateLimiter:
    """
    In-memory rate limiter using token bucket algorithm.

    For production, consider Redis-based implementation for distributed rate limiting.
    """

    def __init__(self):
        self._states: dict[str, RateLimitState] = {}

    def check(self, key_id: str, qps_limit: float) -> tuple[bool, float]:
        """
        Check if request is allowed for given key.

        Returns:
            (allowed: bool, retry_after_seconds: float)
        """
        if key_id not in self._states:
            self._states[key_id] = RateLimitState(
                tokens=qps_limit,  # Start with full bucket
                last_update=time.time(),
                qps_limit=qps_limit
            )

        state = self._states[key_id]

        # Update QPS limit if changed
        if state.qps_limit != qps_limit:
            state.qps_limit = qps_limit

        allowed = state.consume()
        retry_after = state.time_until_available() if not allowed else 0.0

        return allowed, retry_after

    def get_remaining(self, key_id: str) -> int:
        """Get remaining tokens for a key"""
        if key_id not in self._states:
            return 0
        return int(self._states[key_id].tokens)

    def cleanup_old_entries(self, max_age_seconds: float = 3600.0):
        """Remove entries not accessed in max_age_seconds"""
        now = time.time()
        to_remove = [
            key_id for key_id, state in self._states.items()
            if now - state.last_update > max_age_seconds
        ]
        for key_id in to_remove:
            del self._states[key_id]


# Global rate limiter instance
_rate_limiter = RateLimiter()


# ============================================================================
# API Key Validation
# ============================================================================

# FastAPI security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class AuthContext:
    """Authenticated request context"""
    customer_id: str
    key_id: str
    plan: str
    rate_limit_qps: float
    subscription_status: str
    queries_included: int = 0
    price_per_query_cents: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


async def validate_api_key(
    request: Request,
    api_key: str | None = Depends(api_key_header)
) -> AuthContext:
    """
    FastAPI dependency for API key validation.

    Validates API key and returns authenticated context.
    Raises HTTPException if validation fails.

    Usage:
        @router.get("/protected")
        async def protected_endpoint(auth: AuthContext = Depends(validate_api_key)):
            return {"customer_id": auth.customer_id}
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "API key required",
                "code": "missing_api_key"
            }
        )

    # Get backend from app state
    backend = getattr(request.app.state, "saas_backend", None)
    if not backend:
        logger.error("SaaS backend not initialized")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Service unavailable",
                "code": "backend_not_initialized"
            }
        )

    # Look up API key
    api_key_record = await backend.get_api_key_by_secret(api_key)

    if not api_key_record:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid API key",
                "code": "invalid_api_key"
            }
        )

    # Check key status
    if api_key_record.status != "active":
        raise HTTPException(
            status_code=401,
            detail={
                "error": f"API key {api_key_record.status}",
                "code": f"key_{api_key_record.status}"
            }
        )

    # Check expiration
    if api_key_record.expires_at:
        if api_key_record.expires_at < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "API key expired",
                    "code": "key_expired"
                }
            )

    # Rate limiting
    allowed, retry_after = _rate_limiter.check(
        api_key_record.key_id,
        api_key_record.rate_limit_qps
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "code": "rate_limit_exceeded",
                "retry_after_seconds": round(retry_after, 2),
                "limit": api_key_record.rate_limit_qps
            },
            headers={
                "Retry-After": str(int(retry_after) + 1),
                "X-RateLimit-Limit": str(api_key_record.rate_limit_qps),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time() + retry_after))
            }
        )

    # Get subscription for the customer
    subscription = await backend.get_subscription(api_key_record.customer_id)

    if not subscription:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "No active subscription",
                "code": "no_subscription"
            }
        )

    # Check subscription status
    if subscription.status not in ("active", "trialing"):
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Subscription {subscription.status}",
                "code": f"subscription_{subscription.status}"
            }
        )

    # Update last_used timestamp (fire and forget)
    await backend.update_api_key_usage(api_key_record.key_id)

    # Add rate limit headers to response
    request.state.rate_limit_headers = {
        "X-RateLimit-Limit": str(api_key_record.rate_limit_qps),
        "X-RateLimit-Remaining": str(_rate_limiter.get_remaining(api_key_record.key_id)),
        "X-RateLimit-Reset": str(int(time.time() + 1))
    }

    return AuthContext(
        customer_id=api_key_record.customer_id,
        key_id=api_key_record.key_id,
        plan=subscription.plan,
        rate_limit_qps=api_key_record.rate_limit_qps,
        subscription_status=subscription.status,
        queries_included=subscription.queries_included,
        price_per_query_cents=subscription.price_per_query_cents or 0.0
    )


# ============================================================================
# Request Signing (Optional - for sensitive operations)
# ============================================================================

def verify_request_signature(
    request: Request,
    api_secret: str,
    timestamp_tolerance_seconds: int = 300
) -> bool:
    """
    Verify HMAC-SHA256 request signature.

    Expected headers:
        X-API-Timestamp: Unix timestamp
        X-API-Signature: HMAC-SHA256 signature

    String to sign:
        {method}\n{path}\n{timestamp}\n{body_hash}

    Where body_hash = SHA256(request_body) in hex

    Returns True if signature is valid, False otherwise.
    """
    timestamp_str = request.headers.get("X-API-Timestamp")
    signature = request.headers.get("X-API-Signature")

    if not timestamp_str or not signature:
        return False

    # Verify timestamp is recent
    try:
        timestamp = int(timestamp_str)
        now = int(time.time())
        if abs(now - timestamp) > timestamp_tolerance_seconds:
            return False
    except ValueError:
        return False

    # Get request body hash
    body = getattr(request.state, "_body", b"")
    body_hash = hashlib.sha256(body).hexdigest()

    # Build string to sign
    string_to_sign = f"{request.method}\n{request.url.path}\n{timestamp}\n{body_hash}"

    # Calculate expected signature
    expected_signature = hmac.new(
        api_secret.encode(),
        string_to_sign.encode(),
        hashlib.sha256
    ).hexdigest()

    # Constant-time comparison
    return hmac.compare_digest(signature, expected_signature)


# ============================================================================
# Middleware for Rate Limit Headers
# ============================================================================

async def add_rate_limit_headers(request: Request, call_next):
    """
    Middleware to add rate limit headers to responses.

    Usage:
        app.middleware("http")(add_rate_limit_headers)
    """
    response = await call_next(request)

    # Add rate limit headers if available
    rate_limit_headers = getattr(request.state, "rate_limit_headers", None)
    if rate_limit_headers:
        for header, value in rate_limit_headers.items():
            response.headers[header] = value

    return response


# ============================================================================
# Free Tier Limit Checking
# ============================================================================

async def check_free_tier_limit(
    request: Request,
    auth: AuthContext
) -> tuple[bool, str]:
    """
    Check if free tier daily limit is exceeded.

    Returns:
        (allowed: bool, message: str)
    """
    if auth.plan != "free":
        return True, ""

    backend = request.app.state.saas_backend
    today = datetime.now(timezone.utc).date()

    today_usage = await backend.get_usage_for_date(auth.customer_id, today)

    # Free tier: 100 queries/day
    FREE_DAILY_LIMIT = 100

    if today_usage >= FREE_DAILY_LIMIT:
        return False, f"Free tier daily limit ({FREE_DAILY_LIMIT} queries) exceeded. Upgrade to continue."

    return True, ""


# ============================================================================
# Dependency Aliases
# ============================================================================

# Alias for backward compatibility
get_current_customer = validate_api_key

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated Rate Limiter
======================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 15: Safety Layer Foundation

Per-identity sliding window rate limiting for federation requests.
Rate limits are tiered by guild trust level:

    STARTER:     10 requests/minute
    ESTABLISHED: 60 requests/minute
    VETERAN:     300 requests/minute
    STEWARD:     Unlimited (for trusted network stewards)

Implementation uses a sliding window algorithm with per-identity tracking.

Author: Claude Code (Phase 4 - December 2025)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Deque, Dict, Optional, Tuple

from hololoom.federation.types import GuildTrustLevel
from hololoom.federation.safety import SafetyCheckResult


# ============================================================================
#  Rate Limit Configuration
# ============================================================================

class RateLimitTier(Enum):
    """Rate limit tiers mapping to guild trust levels."""

    OBSERVER = auto()      # Read-only, very limited
    MEMBER = auto()        # Standard member
    CONTRIBUTOR = auto()   # Trusted contributor
    STEWARD = auto()       # Network steward (unlimited)


# Default rate limits (requests per minute)
DEFAULT_RATE_LIMITS: Dict[RateLimitTier, int] = {
    RateLimitTier.OBSERVER: 10,
    RateLimitTier.MEMBER: 60,
    RateLimitTier.CONTRIBUTOR: 300,
    RateLimitTier.STEWARD: -1,  # -1 = unlimited
}

# Map guild trust level to rate limit tier
TRUST_TO_TIER: Dict[GuildTrustLevel, RateLimitTier] = {
    GuildTrustLevel.STARTER: RateLimitTier.OBSERVER,
    GuildTrustLevel.ESTABLISHED: RateLimitTier.MEMBER,
    GuildTrustLevel.VETERAN: RateLimitTier.CONTRIBUTOR,
}


# ============================================================================
#  Data Classes
# ============================================================================

@dataclass(slots=True)
class RateLimitState:
    """Tracks rate limit state for a single identity."""

    identity_id: str
    tier: RateLimitTier
    window_seconds: float
    max_requests: int
    requests: Deque[float] = field(default_factory=deque)

    @property
    def current_count(self) -> int:
        """Current number of requests in window."""
        return len(self.requests)

    @property
    def remaining(self) -> int:
        """Remaining requests allowed."""
        if self.max_requests < 0:  # Unlimited
            return -1
        return max(0, self.max_requests - self.current_count)

    @property
    def is_unlimited(self) -> bool:
        """Check if this identity has unlimited requests."""
        return self.max_requests < 0

    def reset_time(self) -> Optional[float]:
        """Time until oldest request expires (Unix timestamp)."""
        if not self.requests:
            return None
        return self.requests[0] + self.window_seconds


@dataclass(frozen=True, slots=True)
class RateLimitInfo:
    """Rate limit information for response headers."""

    limit: int              # Max requests per window
    remaining: int          # Remaining requests
    reset_time: float       # Unix timestamp when window resets
    retry_after: float      # Seconds until next request allowed (0 if not limited)

    def to_headers(self) -> Dict[str, str]:
        """Convert to standard rate limit headers."""
        return {
            "X-RateLimit-Limit": str(self.limit) if self.limit >= 0 else "unlimited",
            "X-RateLimit-Remaining": str(self.remaining) if self.remaining >= 0 else "unlimited",
            "X-RateLimit-Reset": str(int(self.reset_time)) if self.reset_time else "",
            "Retry-After": str(int(self.retry_after)) if self.retry_after > 0 else "",
        }


# ============================================================================
#  Federated Rate Limiter
# ============================================================================

class FederatedRateLimiter:
    """
    Per-identity sliding window rate limiter for federation.

    Uses a sliding window algorithm where each request timestamp is tracked.
    Old timestamps outside the window are evicted on each check.

    Example:
        >>> limiter = FederatedRateLimiter()
        >>> result = limiter.check("node-123", RateLimitTier.MEMBER)
        >>> if result.allowed:
        ...     # Process request
        ...     limiter.record("node-123")
        ... else:
        ...     print(f"Rate limited. Retry after {result.details['retry_after']}s")
    """

    def __init__(
        self,
        rate_limits: Optional[Dict[RateLimitTier, int]] = None,
        window_seconds: float = 60.0,
        cleanup_interval_seconds: float = 300.0,
    ):
        """
        Initialize rate limiter.

        Args:
            rate_limits: Requests per minute per tier (uses defaults if None)
            window_seconds: Sliding window duration
            cleanup_interval_seconds: How often to clean up expired entries
        """
        self._rate_limits = rate_limits or DEFAULT_RATE_LIMITS
        self._window_seconds = window_seconds
        self._cleanup_interval = cleanup_interval_seconds

        # Per-identity state
        self._states: Dict[str, RateLimitState] = {}
        self._last_cleanup = time.time()

    def check(
        self,
        identity_id: str,
        tier: RateLimitTier,
    ) -> SafetyCheckResult:
        """
        Check if a request is allowed under rate limits.

        This does NOT record the request - call record() after successful processing.

        Args:
            identity_id: Unique identifier for the requester
            tier: Rate limit tier to apply

        Returns:
            SafetyCheckResult with rate limit decision
        """
        self._maybe_cleanup()

        state = self._get_or_create_state(identity_id, tier)
        self._evict_expired(state)

        # Unlimited tier always allowed
        if state.is_unlimited:
            return SafetyCheckResult(
                allowed=True,
                reason="Unlimited tier",
                check_type="rate_limit",
                details=self._get_info(state).to_headers(),
            )

        # Check if under limit
        if state.current_count < state.max_requests:
            return SafetyCheckResult(
                allowed=True,
                reason=f"Under limit ({state.remaining} remaining)",
                check_type="rate_limit",
                details=self._get_info(state).to_headers(),
            )

        # Rate limited
        info = self._get_info(state)
        return SafetyCheckResult(
            allowed=False,
            reason=f"Rate limit exceeded ({state.max_requests}/min)",
            check_type="rate_limit",
            details={
                **info.to_headers(),
                "retry_after": info.retry_after,
            },
        )

    def record(self, identity_id: str) -> None:
        """
        Record a request for rate limiting.

        Call this AFTER successfully processing a request.

        Args:
            identity_id: Unique identifier for the requester
        """
        if identity_id not in self._states:
            return

        state = self._states[identity_id]
        state.requests.append(time.time())

    def get_info(
        self,
        identity_id: str,
        tier: Optional[RateLimitTier] = None,
    ) -> RateLimitInfo:
        """
        Get rate limit info for an identity.

        Args:
            identity_id: Unique identifier
            tier: Override tier (uses existing state if None)

        Returns:
            RateLimitInfo with current state
        """
        if identity_id not in self._states:
            if tier is None:
                tier = RateLimitTier.OBSERVER
            state = self._get_or_create_state(identity_id, tier)
        else:
            state = self._states[identity_id]

        self._evict_expired(state)
        return self._get_info(state)

    def reset(self, identity_id: str) -> None:
        """Reset rate limit state for an identity."""
        if identity_id in self._states:
            del self._states[identity_id]

    def reset_all(self) -> None:
        """Reset all rate limit states."""
        self._states.clear()

    def _get_or_create_state(
        self,
        identity_id: str,
        tier: RateLimitTier,
    ) -> RateLimitState:
        """Get or create state for an identity."""
        if identity_id not in self._states:
            max_requests = self._rate_limits.get(tier, DEFAULT_RATE_LIMITS[RateLimitTier.OBSERVER])
            self._states[identity_id] = RateLimitState(
                identity_id=identity_id,
                tier=tier,
                window_seconds=self._window_seconds,
                max_requests=max_requests,
            )
        return self._states[identity_id]

    def _evict_expired(self, state: RateLimitState) -> None:
        """Remove requests outside the sliding window."""
        now = time.time()
        cutoff = now - state.window_seconds

        while state.requests and state.requests[0] < cutoff:
            state.requests.popleft()

    def _get_info(self, state: RateLimitState) -> RateLimitInfo:
        """Build RateLimitInfo from state."""
        now = time.time()
        reset_time = state.reset_time() or (now + state.window_seconds)

        # Calculate retry_after
        retry_after = 0.0
        if state.remaining == 0 and state.requests:
            # Time until oldest request expires
            retry_after = max(0, state.requests[0] + state.window_seconds - now)

        return RateLimitInfo(
            limit=state.max_requests,
            remaining=state.remaining,
            reset_time=reset_time,
            retry_after=retry_after,
        )

    def _maybe_cleanup(self) -> None:
        """Periodically clean up stale entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        # Remove entries with no recent requests
        cutoff = now - self._window_seconds * 2  # Keep entries for 2x window
        to_remove = []

        for identity_id, state in self._states.items():
            if not state.requests or state.requests[-1] < cutoff:
                to_remove.append(identity_id)

        for identity_id in to_remove:
            del self._states[identity_id]

        self._last_cleanup = now


# ============================================================================
#  Convenience Functions
# ============================================================================

def get_tier_for_trust_level(trust_level: GuildTrustLevel) -> RateLimitTier:
    """Get rate limit tier for a guild trust level."""
    return TRUST_TO_TIER.get(trust_level, RateLimitTier.OBSERVER)


def create_rate_limiter(
    custom_limits: Optional[Dict[RateLimitTier, int]] = None,
    window_seconds: float = 60.0,
) -> FederatedRateLimiter:
    """
    Create a configured rate limiter.

    Args:
        custom_limits: Custom rate limits per tier
        window_seconds: Sliding window duration

    Returns:
        Configured FederatedRateLimiter
    """
    return FederatedRateLimiter(
        rate_limits=custom_limits,
        window_seconds=window_seconds,
    )


# ============================================================================
#  Exports
# ============================================================================

__all__ = [
    # Enums
    "RateLimitTier",

    # Data classes
    "RateLimitState",
    "RateLimitInfo",

    # Classes
    "FederatedRateLimiter",

    # Functions
    "get_tier_for_trust_level",
    "create_rate_limiter",

    # Constants
    "DEFAULT_RATE_LIMITS",
    "TRUST_TO_TIER",
]

"""
HoloLoom Security Module

Comprehensive security infrastructure for production deployments.

Phase 1 - Critical Security (Implemented):
- api_keys: API key generation, validation, rotation
- rate_limiting: Distributed rate limiting (Redis-backed)
- secrets: Encrypted secret management
- middleware: Unified security middleware

Usage:
    >>> from HoloLoom.security import APIKeyManager, DistributedRateLimiter, SecretManager
    >>> api_keys = APIKeyManager(secret="your-secret")
    >>> rate_limiter = DistributedRateLimiter(redis_url="redis://localhost")
    >>> secrets = SecretManager()
"""

from HoloLoom.security.api_keys import (
    APIKey,
    APIKeyManager,
    create_api_key_manager
)
from HoloLoom.security.rate_limiting import (
    DistributedRateLimiter,
    RateLimitExceeded,
    create_rate_limiter
)
from HoloLoom.security.secrets import (
    SecretManager,
    create_secret_manager
)

__all__ = [
    # API Keys
    "APIKey",
    "APIKeyManager",
    "create_api_key_manager",
    # Rate Limiting
    "DistributedRateLimiter",
    "RateLimitExceeded",
    "create_rate_limiter",
    # Secrets
    "SecretManager",
    "create_secret_manager",
]

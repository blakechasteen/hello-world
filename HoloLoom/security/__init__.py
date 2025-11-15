"""
HoloLoom Security Module

Comprehensive security infrastructure for production deployments.

Phase 1 - Critical Security (Implemented):
- api_keys: API key generation, validation, rotation
- rate_limiting: Distributed rate limiting (Redis-backed)
- secrets: Encrypted secret management
- middleware: Unified security middleware

Phase 2 - OAuth2/OpenID Connect (Implemented - 2025-11-15):
- oauth2: Multi-provider OAuth2 integration (Auth0, Okta, Google, GitHub)
- jwt_validator: JWT validation with signature verification
- middleware: FastAPI authentication middleware

Phase 2 - Role-Based Access Control (Implemented - 2025-11-15):
- rbac: Hierarchical role system (admin > write > read > guest)
- permissions: Fine-grained permissions (17 permissions across 6 resources)
- policy_engine: Complex attribute-based access control (ABAC)
- decorators: FastAPI endpoint protection
- storage: In-memory and Redis storage backends

Phase 3 - ML-Based Anomaly Detection (Implemented - 2025-11-15):
- anomaly: ML-based anomaly detection (Isolation Forest, LSTM, Autoencoder)
- baseline: Baseline behavior modeling
- explainer: Anomaly explanation generation

Usage:
    >>> from HoloLoom.security import APIKeyManager, DistributedRateLimiter, SecretManager
    >>> api_keys = APIKeyManager(secret="your-secret")
    >>> rate_limiter = DistributedRateLimiter(redis_url="redis://localhost")
    >>> secrets = SecretManager()
    >>>
    >>> # RBAC
    >>> from HoloLoom.security.rbac import RBACManager, Role, Permission
    >>> rbac = create_rbac_manager(storage_type="redis")
    >>> await rbac.assign_role("user_123", Role.WRITE)
    >>> has_perm = await rbac.check_permission("user_123", Permission.QUERY_WRITE)
    >>>
    >>> # OAuth2/JWT authentication
    >>> from HoloLoom.security import OAuth2Client, JWTValidator
    >>> oauth2_client = create_oauth2_client(provider="auth0", ...)
    >>> jwt_validator = create_jwt_validator(issuer="...", audience="...")
    >>>
    >>> # Anomaly detection
    >>> from HoloLoom.security.anomaly import AnomalyDetector, SecurityEvent
    >>> detector = AnomalyDetector()
    >>> await detector.fit_baseline(events)
    >>> result = await detector.detect(event)
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

# Optional: Secrets require cryptography which may not be installed
try:
    from HoloLoom.security.secrets import (
        SecretManager,
        create_secret_manager
    )
except Exception as e:
    import warnings
    warnings.warn(f"Secrets module unavailable: {type(e).__name__}: {e}")
    SecretManager = None
    create_secret_manager = None

# OAuth2/JWT components
from HoloLoom.security.oauth2 import (
    OAuth2Provider,
    OAuth2Config,
    OAuth2Token,
    OAuth2Client,
    PKCEChallenge,
    create_oauth2_client
)
from HoloLoom.security.jwt_validator import (
    JWTValidator,
    JWTClaims,
    create_jwt_validator
)

# Middleware (optional, requires FastAPI)
try:
    from HoloLoom.security.middleware import (
        AuthMiddleware,
        AuthContext,
        create_auth_middleware
    )
except Exception as e:
    import warnings
    warnings.warn(f"Middleware unavailable (FastAPI not installed): {type(e).__name__}: {e}")
    AuthMiddleware = None
    AuthContext = None
    create_auth_middleware = None

__all__ = [
    # API Keys
    "APIKey",
    "APIKeyManager",
    "create_api_key_manager",
    # Rate Limiting
    "DistributedRateLimiter",
    "RateLimitExceeded",
    "create_rate_limiter",
    # OAuth2
    "OAuth2Provider",
    "OAuth2Config",
    "OAuth2Token",
    "OAuth2Client",
    "PKCEChallenge",
    "create_oauth2_client",
    # JWT
    "JWTValidator",
    "JWTClaims",
    "create_jwt_validator",
]

# Add secrets only if available
if SecretManager is not None:
    __all__.extend([
        "SecretManager",
        "create_secret_manager",
    ])

# Add middleware only if available
if AuthMiddleware is not None:
    __all__.extend([
        "AuthMiddleware",
        "AuthContext",
        "create_auth_middleware",
    ])

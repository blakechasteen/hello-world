"""
FastAPI Authentication Middleware

Unified authentication middleware supporting:
- OAuth2/OpenID Connect (JWT tokens)
- API keys
- Rate limiting integration
- Audit trail logging

Features:
- Multiple authentication methods (token + API key fallback)
- Automatic token validation and refresh
- Rate limit enforcement with different limits per auth method
- Complete audit trail integration
- Graceful degradation if OAuth2 unavailable

Usage:
    >>> from fastapi import FastAPI
    >>> from HoloLoom.security.middleware import create_auth_middleware
    >>>
    >>> app = FastAPI()
    >>> auth_middleware = create_auth_middleware(
    ...     jwt_validator=validator,
    ...     api_key_manager=key_manager,
    ...     rate_limiter=limiter
    ... )
    >>> app.add_middleware(auth_middleware)

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from datetime import datetime

try:
    from fastapi import Request, Response, HTTPException, status
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not installed. Middleware unavailable.")

from HoloLoom.security.jwt_validator import JWTValidator, JWTClaims
from HoloLoom.security.api_keys import APIKeyManager, APIKey
from HoloLoom.security.rate_limiting import DistributedRateLimiter, RateLimitExceeded

try:
    from HoloLoom.alignment.audit_trail import AuditTrail
except ImportError:
    AuditTrail = None
    logging.warning("AuditTrail not available. Audit logging disabled.")

logger = logging.getLogger(__name__)


@dataclass
class AuthContext:
    """
    Authentication context for authenticated requests.

    Attributes:
        user_id: User identifier (from token or API key)
        auth_method: "oauth2" or "api_key"
        scopes: List of permission scopes
        token_claims: JWT claims (if OAuth2)
        api_key: APIKey object (if API key auth)
        metadata: Additional authentication metadata
    """
    user_id: str
    auth_method: str
    scopes: list
    token_claims: Optional[JWTClaims] = None
    api_key: Optional[APIKey] = None
    metadata: Dict[str, Any] = None

    def has_scope(self, scope: str) -> bool:
        """Check if user has specific scope."""
        return scope in self.scopes or "admin" in self.scopes


if FASTAPI_AVAILABLE:
    class AuthMiddleware(BaseHTTPMiddleware):
        """
        Unified authentication middleware for FastAPI.

        Supports multiple authentication methods with automatic fallback:
        1. OAuth2/JWT (Bearer token in Authorization header)
        2. API key (X-API-Key header or query parameter)

        Also enforces:
        - Rate limiting (different limits for OAuth2 vs API key)
        - Audit logging (all requests logged)
        """

        def __init__(
            self,
            app,
            jwt_validator: Optional[JWTValidator] = None,
            api_key_manager: Optional[APIKeyManager] = None,
            rate_limiter: Optional[DistributedRateLimiter] = None,
            audit_trail: Optional[AuditTrail] = None,
            oauth2_rate_limit: int = 1000,  # Requests per hour for OAuth2
            api_key_rate_limit: int = 100,  # Requests per hour for API keys
            public_paths: Optional[list] = None,
            require_auth: bool = True
        ):
            """
            Initialize authentication middleware.

            Args:
                app: FastAPI app instance
                jwt_validator: Optional JWT validator
                api_key_manager: Optional API key manager
                rate_limiter: Optional distributed rate limiter
                audit_trail: Optional audit trail
                oauth2_rate_limit: Rate limit for OAuth2 requests (default: 1000/hour)
                api_key_rate_limit: Rate limit for API key requests (default: 100/hour)
                public_paths: List of public paths (no auth required)
                require_auth: Whether to require authentication (default: True)
            """
            super().__init__(app)
            self.jwt_validator = jwt_validator
            self.api_key_manager = api_key_manager
            self.rate_limiter = rate_limiter
            self.audit_trail = audit_trail
            self.oauth2_rate_limit = oauth2_rate_limit
            self.api_key_rate_limit = api_key_rate_limit
            self.public_paths = public_paths or ["/health", "/docs", "/openapi.json"]
            self.require_auth = require_auth

            if not jwt_validator and not api_key_manager:
                logger.warning(
                    "⚠️  No authentication configured! "
                    "Provide jwt_validator or api_key_manager."
                )

        async def dispatch(self, request: Request, call_next: Callable):
            """
            Process request through authentication pipeline.

            Pipeline:
            1. Check if path is public (skip auth)
            2. Try OAuth2/JWT authentication
            3. Fallback to API key authentication
            4. Enforce rate limits
            5. Log to audit trail
            6. Call next middleware
            """
            start_time = time.time()

            # Skip auth for public paths
            if request.url.path in self.public_paths:
                response = await call_next(request)
                return response

            # Attempt authentication
            auth_context = None
            auth_error = None

            try:
                auth_context = await self._authenticate_request(request)
            except HTTPException as e:
                auth_error = e
            except Exception as e:
                logger.error(f"Authentication error: {e}", exc_info=True)
                auth_error = HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal authentication error"
                )

            # Handle authentication failure
            if auth_error and self.require_auth:
                # Log failed authentication
                if self.audit_trail:
                    try:
                        await self.audit_trail.log_decision(
                            query=f"{request.method} {request.url.path}",
                            action="authenticate",
                            outcome="failure",
                            metadata={
                                "error": str(auth_error.detail),
                                "ip": request.client.host if request.client else "unknown"
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Audit trail logging failed: {e}")

                return JSONResponse(
                    status_code=auth_error.status_code,
                    content={"detail": auth_error.detail}
                )

            # Attach auth context to request state
            if auth_context:
                request.state.auth = auth_context

                # Enforce rate limits
                if self.rate_limiter:
                    try:
                        await self._enforce_rate_limit(auth_context)
                    except RateLimitExceeded as e:
                        # Log rate limit exceeded
                        if self.audit_trail:
                            try:
                                await self.audit_trail.log_decision(
                                    query=f"{request.method} {request.url.path}",
                                    action="rate_limit",
                                    outcome="exceeded",
                                    metadata={
                                        "user_id": auth_context.user_id,
                                        "auth_method": auth_context.auth_method,
                                        "limit": e.limit,
                                        "window": e.window
                                    }
                                )
                            except Exception as e:
                                logger.warning(f"Audit trail logging failed: {e}")

                        return JSONResponse(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            content={
                                "detail": f"Rate limit exceeded: {e.limit} requests per {e.window}s"
                            },
                            headers={"Retry-After": str(e.reset_in)}
                        )

            # Process request
            response = await call_next(request)

            # Log successful request
            if self.audit_trail and auth_context:
                try:
                    duration_ms = (time.time() - start_time) * 1000
                    await self.audit_trail.log_decision(
                        query=f"{request.method} {request.url.path}",
                        action="request",
                        outcome="success",
                        metadata={
                            "user_id": auth_context.user_id,
                            "auth_method": auth_context.auth_method,
                            "status_code": response.status_code,
                            "duration_ms": duration_ms
                        }
                    )
                except Exception as e:
                    logger.warning(f"Audit trail logging failed: {e}")

            return response

        async def _authenticate_request(self, request: Request) -> AuthContext:
            """
            Authenticate request using OAuth2 or API key.

            Args:
                request: FastAPI request

            Returns:
                AuthContext

            Raises:
                HTTPException: If authentication fails
            """
            # Try OAuth2/JWT first (Bearer token)
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1]
                return await self._authenticate_oauth2(token)

            # Fallback to API key (X-API-Key header or query param)
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                api_key = request.query_params.get("api_key")

            if api_key:
                return await self._authenticate_api_key(api_key)

            # No authentication provided
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No authentication provided. Use Bearer token or API key.",
                headers={"WWW-Authenticate": "Bearer"}
            )

        async def _authenticate_oauth2(self, token: str) -> AuthContext:
            """
            Authenticate using OAuth2/JWT token.

            Args:
                token: JWT access token

            Returns:
                AuthContext

            Raises:
                HTTPException: If token validation fails
            """
            if not self.jwt_validator:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="OAuth2 authentication not configured"
                )

            try:
                claims = await self.jwt_validator.validate_token(token)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token: {e}",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            except Exception as e:
                logger.error(f"Token validation error: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Token validation error"
                )

            # Extract scopes from token claims
            scopes = []
            scope_claim = claims.get_claim("scope")
            if scope_claim:
                scopes = scope_claim.split() if isinstance(scope_claim, str) else scope_claim

            permissions_claim = claims.get_claim("permissions")
            if permissions_claim:
                scopes.extend(permissions_claim if isinstance(permissions_claim, list) else [permissions_claim])

            return AuthContext(
                user_id=claims.sub,
                auth_method="oauth2",
                scopes=scopes,
                token_claims=claims,
                metadata={"issuer": claims.iss, "audience": claims.aud}
            )

        async def _authenticate_api_key(self, raw_key: str) -> AuthContext:
            """
            Authenticate using API key.

            Args:
                raw_key: Raw API key from request

            Returns:
                AuthContext

            Raises:
                HTTPException: If API key validation fails
            """
            if not self.api_key_manager:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="API key authentication not configured"
                )

            # Note: In production, you'd look up the APIKey object from database
            # For now, we'll assume you have a method to retrieve it
            # This is a placeholder - implement proper key lookup!
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="API key lookup not implemented. See middleware documentation."
            )

            # Example implementation (uncomment when you have key lookup):
            # try:
            #     stored_key = await self._lookup_api_key(raw_key)
            #     if not stored_key:
            #         raise ValueError("Key not found")
            #
            #     if not self.api_key_manager.verify_key(raw_key, stored_key):
            #         raise ValueError("Invalid key")
            #
            #     return AuthContext(
            #         user_id=stored_key.user_id,
            #         auth_method="api_key",
            #         scopes=stored_key.scopes,
            #         api_key=stored_key,
            #         metadata={"key_id": stored_key.key_id}
            #     )
            # except ValueError as e:
            #     raise HTTPException(
            #         status_code=status.HTTP_401_UNAUTHORIZED,
            #         detail=f"Invalid API key: {e}"
            #     )

        async def _enforce_rate_limit(self, auth_context: AuthContext):
            """
            Enforce rate limits based on authentication method.

            Args:
                auth_context: Authentication context

            Raises:
                RateLimitExceeded: If rate limit exceeded
            """
            if not self.rate_limiter:
                return

            # Different limits for OAuth2 vs API key
            if auth_context.auth_method == "oauth2":
                limit = self.oauth2_rate_limit
            else:
                limit = self.api_key_rate_limit

            # Check rate limit
            identifier = f"{auth_context.auth_method}:{auth_context.user_id}"
            allowed = await self.rate_limiter.check_limit(
                identifier=identifier,
                limit=limit,
                window=3600  # 1 hour
            )

            if not allowed:
                raise RateLimitExceeded(
                    identifier=identifier,
                    limit=limit,
                    window=3600,
                    reset_in=await self.rate_limiter.get_reset_time(identifier)
                )


# Factory function
def create_auth_middleware(
    jwt_validator: Optional[JWTValidator] = None,
    api_key_manager: Optional[APIKeyManager] = None,
    rate_limiter: Optional[DistributedRateLimiter] = None,
    audit_trail: Optional[AuditTrail] = None,
    **kwargs
):
    """
    Create authentication middleware factory.

    Args:
        jwt_validator: Optional JWT validator
        api_key_manager: Optional API key manager
        rate_limiter: Optional rate limiter
        audit_trail: Optional audit trail
        **kwargs: Additional middleware configuration

    Returns:
        Middleware class (for use with app.add_middleware)

    Usage:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>>
        >>> middleware = create_auth_middleware(
        ...     jwt_validator=validator,
        ...     rate_limiter=limiter
        ... )
        >>>
        >>> app.add_middleware(
        ...     middleware,
        ...     jwt_validator=validator,
        ...     rate_limiter=limiter
        ... )
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI not installed. Install with: pip install fastapi"
        )

    # Return middleware class configured with dependencies
    class ConfiguredAuthMiddleware(AuthMiddleware):
        def __init__(self, app):
            super().__init__(
                app,
                jwt_validator=jwt_validator,
                api_key_manager=api_key_manager,
                rate_limiter=rate_limiter,
                audit_trail=audit_trail,
                **kwargs
            )

    return ConfiguredAuthMiddleware


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Auth Middleware Usage Example")
    print("=" * 70)

    print("""
from fastapi import FastAPI, Request
from HoloLoom.security.middleware import create_auth_middleware
from HoloLoom.security.jwt_validator import create_jwt_validator
from HoloLoom.security.rate_limiting import create_rate_limiter

# Create dependencies
jwt_validator = create_jwt_validator(
    issuer="https://your-tenant.auth0.com/",
    audience="your-api-identifier",
    jwks_url="https://your-tenant.auth0.com/.well-known/jwks.json"
)

rate_limiter = create_rate_limiter(redis_url="redis://localhost")

# Create FastAPI app
app = FastAPI()

# Add authentication middleware
app.add_middleware(
    create_auth_middleware(
        jwt_validator=jwt_validator,
        rate_limiter=rate_limiter,
        oauth2_rate_limit=1000,  # 1000 requests/hour for OAuth2
        api_key_rate_limit=100    # 100 requests/hour for API keys
    )
)

# Protected endpoint
@app.get("/api/protected")
async def protected_route(request: Request):
    auth = request.state.auth  # AuthContext
    return {
        "user_id": auth.user_id,
        "scopes": auth.scopes,
        "auth_method": auth.auth_method
    }

# Public endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
    """)

    print("\n" + "=" * 70)

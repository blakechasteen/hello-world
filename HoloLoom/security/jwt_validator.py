"""
JWT Token Validation

Secure JWT validation with signature verification, expiry checks, and claims validation.

Features:
- RSA/ECDSA signature verification
- Token expiry and not-before checks
- Issuer and audience validation
- JWKS (JSON Web Key Set) support
- Token caching (reduces validation overhead)
- Graceful fallback if python-jose unavailable

References:
- RFC 7519 (JSON Web Token)
- RFC 7517 (JSON Web Key)
- OpenID Connect Core 1.0

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json

try:
    from jose import jwt, jwk, JWTError
    from jose.backends import RSAKey, ECKey
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False
    logging.warning(
        "python-jose not installed. JWT validation will use fallback mode. "
        "Install with: pip install python-jose[cryptography]"
    )

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

logger = logging.getLogger(__name__)


@dataclass
class JWTClaims:
    """
    Parsed JWT claims.

    Standard claims (RFC 7519):
        sub: Subject (user ID)
        iss: Issuer
        aud: Audience
        exp: Expiration time
        nbf: Not before
        iat: Issued at
        jti: JWT ID

    Custom claims can be accessed via `custom_claims` dict.
    """
    sub: str  # Subject (user ID)
    iss: Optional[str] = None  # Issuer
    aud: Optional[str] = None  # Audience
    exp: Optional[int] = None  # Expiration time
    nbf: Optional[int] = None  # Not before
    iat: Optional[int] = None  # Issued at
    jti: Optional[str] = None  # JWT ID
    custom_claims: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, claims: Dict[str, Any]) -> "JWTClaims":
        """Parse claims from JWT payload."""
        standard_claims = {
            "sub", "iss", "aud", "exp", "nbf", "iat", "jti"
        }

        custom = {
            k: v for k, v in claims.items()
            if k not in standard_claims
        }

        return cls(
            sub=claims["sub"],
            iss=claims.get("iss"),
            aud=claims.get("aud"),
            exp=claims.get("exp"),
            nbf=claims.get("nbf"),
            iat=claims.get("iat"),
            jti=claims.get("jti"),
            custom_claims=custom
        )

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.exp is None:
            return False
        return time.time() >= self.exp

    def is_not_yet_valid(self) -> bool:
        """Check if token is not yet valid (nbf)."""
        if self.nbf is None:
            return False
        return time.time() < self.nbf

    def get_claim(self, claim_name: str, default: Any = None) -> Any:
        """Get custom claim value."""
        if self.custom_claims is None:
            return default
        return self.custom_claims.get(claim_name, default)


class JWTValidator:
    """
    JWT token validator with signature verification and caching.

    Supports:
    - RSA and ECDSA signature verification
    - JWKS (JSON Web Key Set) fetching
    - Token expiry and not-before validation
    - Issuer and audience validation
    - Token caching (reduces validation overhead)

    Usage:
        >>> validator = JWTValidator(
        ...     issuer="https://your-tenant.auth0.com/",
        ...     audience="your-api-identifier",
        ...     jwks_url="https://your-tenant.auth0.com/.well-known/jwks.json"
        ... )
        >>>
        >>> async with validator:
        ...     claims = await validator.validate_token(access_token)
        ...     print(f"User: {claims.sub}")
    """

    def __init__(
        self,
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        jwks_url: Optional[str] = None,
        algorithms: Optional[List[str]] = None,
        redis_url: Optional[str] = None,
        cache_ttl: int = 300,  # 5 minutes
        leeway: int = 0  # Clock skew tolerance in seconds
    ):
        """
        Initialize JWT validator.

        Args:
            issuer: Expected token issuer (validated if provided)
            audience: Expected token audience (validated if provided)
            jwks_url: URL to fetch JWKS (public keys)
            algorithms: Allowed signing algorithms (default: ["RS256", "ES256"])
            redis_url: Optional Redis URL for caching
            cache_ttl: Cache TTL in seconds
            leeway: Clock skew tolerance in seconds
        """
        self.issuer = issuer
        self.audience = audience
        self.jwks_url = jwks_url
        self.algorithms = algorithms or ["RS256", "ES256"]
        self.cache_ttl = cache_ttl
        self.leeway = leeway

        self.session: Optional[aiohttp.ClientSession] = None
        self.redis: Optional[aioredis.Redis] = None
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_cache_time: Optional[float] = None

        if redis_url and aioredis:
            self.redis_url = redis_url
        else:
            self.redis_url = None

        if not JOSE_AVAILABLE:
            logger.warning(
                "⚠️  JWT validation requires python-jose. "
                "Install with: pip install python-jose[cryptography]"
            )

    async def __aenter__(self):
        """Async context manager entry."""
        if aiohttp:
            self.session = aiohttp.ClientSession()

        if self.redis_url:
            try:
                self.redis = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis = None

        # Pre-fetch JWKS if URL provided
        if self.jwks_url:
            await self._fetch_jwks()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

        if self.redis:
            await self.redis.close()

    async def validate_token(
        self,
        token: str,
        verify_signature: bool = True,
        verify_expiry: bool = True,
        verify_nbf: bool = True
    ) -> JWTClaims:
        """
        Validate JWT token.

        Args:
            token: JWT token string
            verify_signature: Whether to verify signature (default: True)
            verify_expiry: Whether to check expiration (default: True)
            verify_nbf: Whether to check not-before (default: True)

        Returns:
            JWTClaims object

        Raises:
            ValueError: If token is invalid
            RuntimeError: If validation fails
        """
        if not JOSE_AVAILABLE:
            # Fallback: basic validation without signature verification
            logger.warning("JWT signature verification disabled (python-jose not available)")
            return await self._validate_token_fallback(token, verify_expiry, verify_nbf)

        # Check cache first
        if self.redis:
            cache_key = f"jwt:validated:{token[:32]}"
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    logger.debug("JWT validation cache hit")
                    claims_dict = json.loads(cached)
                    return JWTClaims.from_dict(claims_dict)
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        # Decode header to get key ID (kid)
        try:
            header = jwt.get_unverified_header(token)
        except JWTError as e:
            raise ValueError(f"Invalid JWT header: {e}")

        # Get signing key
        if verify_signature and self.jwks_url:
            signing_key = await self._get_signing_key(header.get("kid"))
        else:
            signing_key = None

        # Validate token
        try:
            options = {
                "verify_signature": verify_signature and signing_key is not None,
                "verify_exp": verify_expiry,
                "verify_nbf": verify_nbf,
                "verify_iss": self.issuer is not None,
                "verify_aud": self.audience is not None,
                "leeway": self.leeway
            }

            claims = jwt.decode(
                token,
                signing_key,
                algorithms=self.algorithms,
                issuer=self.issuer,
                audience=self.audience,
                options=options
            )
        except JWTError as e:
            raise ValueError(f"JWT validation failed: {e}")

        jwt_claims = JWTClaims.from_dict(claims)

        # Cache result
        if self.redis:
            cache_key = f"jwt:validated:{token[:32]}"
            try:
                # Calculate remaining TTL based on token expiry
                if jwt_claims.exp:
                    remaining_ttl = min(
                        self.cache_ttl,
                        int(jwt_claims.exp - time.time())
                    )
                else:
                    remaining_ttl = self.cache_ttl

                if remaining_ttl > 0:
                    await self.redis.setex(
                        cache_key,
                        remaining_ttl,
                        json.dumps(claims)
                    )
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")

        logger.info(f"JWT validated successfully (sub: {jwt_claims.sub})")

        return jwt_claims

    async def _validate_token_fallback(
        self,
        token: str,
        verify_expiry: bool,
        verify_nbf: bool
    ) -> JWTClaims:
        """
        Fallback validation without signature verification.

        WARNING: Only use for development! Does not verify signature.
        """
        import base64

        # Split token
        try:
            _, payload, _ = token.split(".")
        except ValueError:
            raise ValueError("Invalid JWT format")

        # Decode payload (add padding if needed)
        payload += "=" * (4 - len(payload) % 4)
        try:
            claims = json.loads(base64.urlsafe_b64decode(payload))
        except Exception as e:
            raise ValueError(f"Failed to decode JWT payload: {e}")

        jwt_claims = JWTClaims.from_dict(claims)

        # Validate expiry
        if verify_expiry and jwt_claims.is_expired():
            raise ValueError("Token expired")

        # Validate not-before
        if verify_nbf and jwt_claims.is_not_yet_valid():
            raise ValueError("Token not yet valid")

        logger.warning(
            "⚠️  JWT validated WITHOUT signature verification! "
            "Install python-jose for production use."
        )

        return jwt_claims

    async def _fetch_jwks(self) -> Dict[str, Any]:
        """
        Fetch JWKS (JSON Web Key Set) from provider.

        Returns:
            JWKS dictionary

        Caches result for 1 hour.
        """
        # Check cache (1 hour TTL)
        if self._jwks_cache and self._jwks_cache_time:
            if time.time() - self._jwks_cache_time < 3600:
                return self._jwks_cache

        if not self.session:
            raise RuntimeError("Session not initialized")

        if not self.jwks_url:
            raise ValueError("JWKS URL not configured")

        async with self.session.get(self.jwks_url) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch JWKS: {response.status}")

            jwks = await response.json()

        # Cache result
        self._jwks_cache = jwks
        self._jwks_cache_time = time.time()

        logger.info(f"Fetched JWKS from {self.jwks_url} ({len(jwks.get('keys', []))} keys)")

        return jwks

    async def _get_signing_key(self, kid: Optional[str] = None) -> Optional[str]:
        """
        Get signing key from JWKS.

        Args:
            kid: Key ID from JWT header

        Returns:
            Signing key (PEM format)
        """
        if not JOSE_AVAILABLE:
            return None

        jwks = await self._fetch_jwks()

        # Find matching key
        for key in jwks.get("keys", []):
            if kid is None or key.get("kid") == kid:
                # Convert JWK to PEM
                try:
                    if key.get("kty") == "RSA":
                        rsa_key = RSAKey(key, algorithm="RS256")
                        return rsa_key
                    elif key.get("kty") == "EC":
                        ec_key = ECKey(key, algorithm="ES256")
                        return ec_key
                except Exception as e:
                    logger.warning(f"Failed to parse key {kid}: {e}")
                    continue

        raise ValueError(f"Signing key not found (kid: {kid})")

    def decode_token_unverified(self, token: str) -> Dict[str, Any]:
        """
        Decode JWT without verification (for debugging).

        WARNING: Do not use for authentication! Only for inspection.

        Args:
            token: JWT token string

        Returns:
            Decoded claims (unverified)
        """
        if JOSE_AVAILABLE:
            return jwt.get_unverified_claims(token)
        else:
            # Fallback
            import base64
            _, payload, _ = token.split(".")
            payload += "=" * (4 - len(payload) % 4)
            return json.loads(base64.urlsafe_b64decode(payload))


# Factory function
def create_jwt_validator(
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    jwks_url: Optional[str] = None,
    **kwargs
) -> JWTValidator:
    """
    Create JWT validator from environment or provided configuration.

    Args:
        issuer: Expected token issuer
        audience: Expected token audience
        jwks_url: URL to fetch JWKS
        **kwargs: Additional configuration

    Returns:
        JWTValidator instance

    Usage:
        >>> validator = create_jwt_validator(
        ...     issuer="https://your-tenant.auth0.com/",
        ...     audience="your-api-identifier",
        ...     jwks_url="https://your-tenant.auth0.com/.well-known/jwks.json"
        ... )
    """
    return JWTValidator(
        issuer=issuer,
        audience=audience,
        jwks_url=jwks_url,
        algorithms=kwargs.get("algorithms"),
        redis_url=kwargs.get("redis_url"),
        cache_ttl=kwargs.get("cache_ttl", 300),
        leeway=kwargs.get("leeway", 0)
    )


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demonstrate JWT validation."""
        print("=" * 70)
        print("JWT Validator Demo")
        print("=" * 70)

        # Example JWT (Auth0)
        # NOTE: This is just a demo. Use real tokens in production!
        validator = JWTValidator(
            issuer="https://dev-example.auth0.com/",
            audience="https://api.example.com",
            jwks_url="https://dev-example.auth0.com/.well-known/jwks.json"
        )

        # Demo token (will fail validation without real JWKS)
        demo_token = (
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImRlbW8ta2lkIn0."
            "eyJzdWIiOiJhdXRoMHwxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiZW1haWwiOiJqb2huQGV4YW1wbGUuY29tIiwiaWF0IjoxNTE2MjM5MDIyLCJleHAiOjE1MTYyNDI2MjIsImF1ZCI6Imh0dHBzOi8vYXBpLmV4YW1wbGUuY29tIiwiaXNzIjoiaHR0cHM6Ly9kZXYtZXhhbXBsZS5hdXRoMC5jb20vIn0."
            "signature-here"
        )

        print("\n📋 Decoding token (unverified):")
        claims = validator.decode_token_unverified(demo_token)
        print(f"   Subject: {claims.get('sub')}")
        print(f"   Name: {claims.get('name')}")
        print(f"   Email: {claims.get('email')}")
        print(f"   Issuer: {claims.get('iss')}")
        print(f"   Audience: {claims.get('aud')}")

        print("\n⚠️  Full validation requires:")
        print("   1. Valid JWKS endpoint")
        print("   2. Real JWT with valid signature")
        print("   3. python-jose installed")

        print("\n" + "=" * 70)

    asyncio.run(demo())

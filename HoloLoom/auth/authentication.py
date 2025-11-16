"""
HoloLoom JWT Authentication
============================

Core JWT token generation and verification.

Features:
- Access tokens (1 hour default)
- Refresh tokens (7 days default)
- Token blacklist for logout
- Configurable expiry times
- Graceful degradation without dependencies

Created: 2025-11-16
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import JWT dependencies
try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    JWTError = Exception
    logger.warning(
        "python-jose not available. Install with: pip install python-jose[cryptography]"
    )

# Configuration from environment
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production-PLEASE")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "60"))
REFRESH_EXPIRY_DAYS = int(os.getenv("REFRESH_EXPIRY_DAYS", "7"))

# Token blacklist (in-memory set)
# In production, use Redis with TTL = token expiry
token_blacklist: Set[str] = set()


@dataclass
class TokenData:
    """Token payload data."""
    username: str
    role: str
    exp: Optional[datetime] = None


@dataclass
class Token:
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRY_MINUTES * 60


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.

    Args:
        data: Payload data (must include 'sub' for username)
        expires_delta: Optional custom expiry time

    Returns:
        JWT token string

    Raises:
        RuntimeError: If JWT dependencies not available
    """
    if not JWT_AVAILABLE:
        raise RuntimeError(
            "JWT not available. Install python-jose: "
            "pip install python-jose[cryptography]"
        )

    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT refresh token.

    Args:
        data: Payload data (must include 'sub' for username)
        expires_delta: Optional custom expiry time

    Returns:
        JWT refresh token string

    Raises:
        RuntimeError: If JWT dependencies not available
    """
    if not JWT_AVAILABLE:
        raise RuntimeError(
            "JWT not available. Install python-jose: "
            "pip install python-jose[cryptography]"
        )

    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_EXPIRY_DAYS)

    to_encode.update({"exp": expire, "type": "refresh"})

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
    """
    Verify JWT token and extract payload.

    Args:
        token: JWT token string
        token_type: Expected token type ("access" or "refresh")

    Returns:
        TokenData if valid, None if invalid

    Raises:
        RuntimeError: If JWT dependencies not available
    """
    if not JWT_AVAILABLE:
        raise RuntimeError(
            "JWT not available. Install python-jose: "
            "pip install python-jose[cryptography]"
        )

    # Check blacklist
    if token in token_blacklist:
        logger.warning(f"Token in blacklist: {token[:20]}...")
        return None

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        username: str = payload.get("sub")
        role: str = payload.get("role", "user")
        exp_timestamp: Optional[int] = payload.get("exp")
        actual_type: str = payload.get("type", "access")

        if username is None:
            logger.warning("Token missing 'sub' claim")
            return None

        if actual_type != token_type:
            logger.warning(
                f"Token type mismatch: expected {token_type}, got {actual_type}"
            )
            return None

        exp = datetime.fromtimestamp(exp_timestamp) if exp_timestamp else None

        return TokenData(username=username, role=role, exp=exp)

    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None


def revoke_token(token: str) -> None:
    """
    Add token to blacklist (logout).

    Args:
        token: JWT token to revoke

    Note:
        In production, use Redis with TTL = token expiry time.
        This in-memory implementation loses blacklist on restart.
    """
    token_blacklist.add(token)
    logger.info(f"Token revoked: {token[:20]}... (total blacklisted: {len(token_blacklist)})")


def clear_expired_tokens() -> int:
    """
    Clear expired tokens from blacklist.

    Returns:
        Number of tokens removed

    Note:
        This is a manual cleanup. In production with Redis, use TTL instead.
    """
    if not JWT_AVAILABLE:
        return 0

    expired = set()

    for token in token_blacklist:
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            exp_timestamp = payload.get("exp")

            if exp_timestamp and datetime.fromtimestamp(exp_timestamp) < datetime.utcnow():
                expired.add(token)
        except JWTError:
            # Token is invalid/expired, remove it
            expired.add(token)

    token_blacklist.difference_update(expired)

    logger.info(f"Cleared {len(expired)} expired tokens from blacklist")
    return len(expired)


def is_jwt_available() -> bool:
    """Check if JWT dependencies are available."""
    return JWT_AVAILABLE


# Example usage
if __name__ == "__main__":
    if not JWT_AVAILABLE:
        print("JWT not available. Install with: pip install python-jose[cryptography]")
    else:
        # Create token
        token = create_access_token({"sub": "admin", "role": "admin"})
        print(f"Access token: {token[:50]}...")

        # Verify token
        data = verify_token(token)
        if data:
            print(f"Verified: {data.username} ({data.role})")

        # Create refresh token
        refresh = create_refresh_token({"sub": "admin", "role": "admin"})
        print(f"Refresh token: {refresh[:50]}...")

        # Revoke token
        revoke_token(token)
        print("Token revoked")

        # Verify revoked token (should fail)
        data = verify_token(token)
        print(f"Revoked token verification: {data}")

"""
Test JWT authentication module.

Tests:
- Token generation
- Token verification
- Token blacklist (logout)
- Expiry handling
"""

import pytest
from datetime import timedelta

# Skip all tests if JWT not available
try:
    from HoloLoom.auth.authentication import (
        create_access_token,
        create_refresh_token,
        verify_token,
        revoke_token,
        clear_expired_tokens,
        is_jwt_available,
        token_blacklist,
    )

    JWT_AVAILABLE = is_jwt_available()
except ImportError:
    JWT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not JWT_AVAILABLE,
    reason="JWT dependencies not available"
)


def test_create_access_token():
    """Test access token creation."""
    token = create_access_token(data={"sub": "testuser", "role": "user"})

    assert isinstance(token, str)
    assert len(token) > 50  # JWT tokens are long


def test_verify_access_token():
    """Test access token verification."""
    token = create_access_token(data={"sub": "testuser", "role": "admin"})

    token_data = verify_token(token, token_type="access")

    assert token_data is not None
    assert token_data.username == "testuser"
    assert token_data.role == "admin"
    assert token_data.exp is not None


def test_create_refresh_token():
    """Test refresh token creation."""
    token = create_refresh_token(data={"sub": "testuser", "role": "user"})

    assert isinstance(token, str)
    assert len(token) > 50


def test_verify_refresh_token():
    """Test refresh token verification."""
    token = create_refresh_token(data={"sub": "testuser", "role": "user"})

    token_data = verify_token(token, token_type="refresh")

    assert token_data is not None
    assert token_data.username == "testuser"


def test_token_type_mismatch():
    """Test that access token cannot be used as refresh token."""
    access_token = create_access_token(data={"sub": "testuser"})

    # Try to verify as refresh token (should fail)
    token_data = verify_token(access_token, token_type="refresh")

    assert token_data is None


def test_invalid_token():
    """Test invalid token verification."""
    token_data = verify_token("invalid.token.here", token_type="access")

    assert token_data is None


def test_token_revocation():
    """Test token blacklist (logout)."""
    token = create_access_token(data={"sub": "testuser"})

    # Verify token works
    token_data = verify_token(token)
    assert token_data is not None

    # Revoke token
    revoke_token(token)

    # Verify token no longer works
    token_data = verify_token(token)
    assert token_data is None


def test_expired_token():
    """Test expired token verification."""
    # Create token that expires in -1 second (already expired)
    token = create_access_token(
        data={"sub": "testuser"},
        expires_delta=timedelta(seconds=-1)
    )

    # Should fail verification
    token_data = verify_token(token)
    assert token_data is None


def test_clear_expired_tokens():
    """Test clearing expired tokens from blacklist."""
    # Clear existing blacklist
    token_blacklist.clear()

    # Create and revoke expired token
    expired_token = create_access_token(
        data={"sub": "testuser"},
        expires_delta=timedelta(seconds=-1)
    )
    revoke_token(expired_token)

    # Create and revoke valid token
    valid_token = create_access_token(data={"sub": "testuser"})
    revoke_token(valid_token)

    assert len(token_blacklist) == 2

    # Clear expired tokens
    cleared = clear_expired_tokens()

    assert cleared == 1  # Only expired token removed
    assert len(token_blacklist) == 1
    assert valid_token in token_blacklist
    assert expired_token not in token_blacklist


def test_token_payload_data():
    """Test token contains correct payload data."""
    from jose import jwt
    from HoloLoom.auth.authentication import JWT_SECRET_KEY, JWT_ALGORITHM

    token = create_access_token(data={"sub": "testuser", "role": "admin"})

    # Decode without verification to inspect payload
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

    assert payload["sub"] == "testuser"
    assert payload["role"] == "admin"
    assert payload["type"] == "access"
    assert "exp" in payload

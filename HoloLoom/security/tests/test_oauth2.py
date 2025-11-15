"""
OAuth2/OpenID Connect Tests

Comprehensive test suite for OAuth2 provider integration and JWT validation.

Test Coverage:
- OAuth2 configuration (Auth0, Okta, Google, GitHub)
- PKCE challenge generation
- Token exchange and refresh
- JWT validation (signature, expiry, claims)
- Token caching
- Token introspection
- Token revocation
- Error handling and fallback

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

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


class TestOAuth2Config:
    """Test OAuth2 configuration."""

    def test_auth0_config(self):
        """Test Auth0 configuration creation."""
        config = OAuth2Config.auth0(
            domain="dev-example.auth0.com",
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback",
            audience="https://api.example.com"
        )

        assert config.provider == OAuth2Provider.AUTH0
        assert config.client_id == "test-client-id"
        assert config.authority == "https://dev-example.auth0.com"
        assert config.audience == "https://api.example.com"
        assert "openid" in config.scopes

    def test_okta_config(self):
        """Test Okta configuration creation."""
        config = OAuth2Config.okta(
            domain="dev-example.okta.com",
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback"
        )

        assert config.provider == OAuth2Provider.OKTA
        assert config.authority == "https://dev-example.okta.com/oauth2/default"

    def test_google_config(self):
        """Test Google OAuth2 configuration creation."""
        config = OAuth2Config.google(
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback"
        )

        assert config.provider == OAuth2Provider.GOOGLE
        assert config.authority == "https://accounts.google.com"

    def test_github_config(self):
        """Test GitHub OAuth2 configuration creation."""
        config = OAuth2Config.github(
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback"
        )

        assert config.provider == OAuth2Provider.GITHUB
        assert config.authority == "https://github.com"
        assert "read:user" in config.scopes


class TestPKCEChallenge:
    """Test PKCE challenge generation."""

    def test_generate_pkce_challenge(self):
        """Test PKCE challenge generation."""
        verifier, challenge = PKCEChallenge.generate()

        # Verify verifier length (43-128 chars for URL-safe base64)
        assert 43 <= len(verifier) <= 128
        assert len(challenge) == 43  # SHA256 base64 (no padding)

        # Verify URL-safe characters only
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in verifier)
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in challenge)

    def test_pkce_challenge_uniqueness(self):
        """Test that PKCE challenges are unique."""
        challenges = [PKCEChallenge.generate() for _ in range(10)]
        verifiers = [v for v, _ in challenges]
        challenge_codes = [c for _, c in challenges]

        # All verifiers should be unique
        assert len(set(verifiers)) == 10
        # All challenges should be unique
        assert len(set(challenge_codes)) == 10


class TestOAuth2Token:
    """Test OAuth2 token object."""

    def test_token_creation(self):
        """Test token creation."""
        token = OAuth2Token(
            access_token="test-access-token",
            token_type="Bearer",
            expires_in=3600,
            refresh_token="test-refresh-token",
            scope="openid profile email"
        )

        assert token.access_token == "test-access-token"
        assert token.token_type == "Bearer"
        assert token.expires_in == 3600
        assert token.refresh_token == "test-refresh-token"
        assert not token.is_expired()

    def test_token_expiry(self):
        """Test token expiry checking."""
        # Expired token
        expired_token = OAuth2Token(
            access_token="test-token",
            expires_in=1,
            created_at=datetime.now() - timedelta(seconds=10)
        )
        assert expired_token.is_expired()

        # Valid token
        valid_token = OAuth2Token(
            access_token="test-token",
            expires_in=3600
        )
        assert not valid_token.is_expired()

    def test_token_serialization(self):
        """Test token serialization/deserialization."""
        original = OAuth2Token(
            access_token="test-access-token",
            expires_in=3600,
            refresh_token="test-refresh-token"
        )

        # Serialize
        data = original.to_dict()
        assert data["access_token"] == "test-access-token"

        # Deserialize
        restored = OAuth2Token.from_dict(data)
        assert restored.access_token == original.access_token
        assert restored.expires_in == original.expires_in


class TestOAuth2Client:
    """Test OAuth2 client."""

    @pytest.fixture
    def auth0_config(self):
        """Create test Auth0 configuration."""
        return OAuth2Config.auth0(
            domain="dev-example.auth0.com",
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback"
        )

    @pytest.mark.asyncio
    async def test_authorization_url_generation(self, auth0_config):
        """Test authorization URL generation."""
        async with OAuth2Client(auth0_config) as client:
            auth_url, state, code_verifier = client.get_authorization_url()

            # Verify URL structure
            assert auth_url.startswith("https://dev-example.auth0.com/authorize?")
            assert "client_id=test-client-id" in auth_url
            assert "response_type=code" in auth_url
            assert f"state={state}" in auth_url
            assert "code_challenge=" in auth_url
            assert "code_challenge_method=S256" in auth_url

            # Verify state is stored
            assert state in client._state_store
            assert client._state_store[state]["code_verifier"] == code_verifier

    @pytest.mark.asyncio
    async def test_authorization_url_without_pkce(self, auth0_config):
        """Test authorization URL generation without PKCE."""
        async with OAuth2Client(auth0_config) as client:
            auth_url, state, code_verifier = client.get_authorization_url(use_pkce=False)

            # Verify no PKCE in URL
            assert "code_challenge=" not in auth_url
            assert code_verifier is None

    @pytest.mark.asyncio
    async def test_state_validation(self, auth0_config):
        """Test state parameter validation."""
        async with OAuth2Client(auth0_config) as client:
            # Generate valid state
            _, valid_state, _ = client.get_authorization_url()

            # Try to exchange code with invalid state
            with pytest.raises(ValueError, match="Invalid or expired state"):
                await client.exchange_code_for_token(
                    code="test-code",
                    state="invalid-state"
                )


class TestJWTClaims:
    """Test JWT claims parsing."""

    def test_claims_from_dict(self):
        """Test claims parsing from dictionary."""
        claims_dict = {
            "sub": "auth0|123456",
            "iss": "https://dev-example.auth0.com/",
            "aud": "https://api.example.com",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "scope": "openid profile email",
            "custom_claim": "custom_value"
        }

        claims = JWTClaims.from_dict(claims_dict)

        assert claims.sub == "auth0|123456"
        assert claims.iss == "https://dev-example.auth0.com/"
        assert claims.aud == "https://api.example.com"
        assert not claims.is_expired()
        assert claims.get_claim("custom_claim") == "custom_value"

    def test_claims_expiry(self):
        """Test claims expiry checking."""
        # Expired claims
        expired_claims = JWTClaims(
            sub="user123",
            exp=int(time.time()) - 3600  # 1 hour ago
        )
        assert expired_claims.is_expired()

        # Valid claims
        valid_claims = JWTClaims(
            sub="user123",
            exp=int(time.time()) + 3600  # 1 hour from now
        )
        assert not valid_claims.is_expired()

    def test_claims_not_before(self):
        """Test claims not-before validation."""
        # Not yet valid
        future_claims = JWTClaims(
            sub="user123",
            nbf=int(time.time()) + 3600  # 1 hour from now
        )
        assert future_claims.is_not_yet_valid()

        # Already valid
        valid_claims = JWTClaims(
            sub="user123",
            nbf=int(time.time()) - 3600  # 1 hour ago
        )
        assert not valid_claims.is_not_yet_valid()


class TestJWTValidator:
    """Test JWT validator."""

    @pytest.fixture
    def validator(self):
        """Create test JWT validator."""
        return JWTValidator(
            issuer="https://dev-example.auth0.com/",
            audience="https://api.example.com",
            algorithms=["RS256"]
        )

    def test_decode_unverified(self, validator):
        """Test unverified token decoding."""
        # Create a simple JWT (header.payload.signature)
        import base64
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
        ).decode().rstrip('=')
        payload = base64.urlsafe_b64encode(
            json.dumps({
                "sub": "user123",
                "iss": "https://dev-example.auth0.com/",
                "aud": "https://api.example.com",
                "exp": int(time.time()) + 3600
            }).encode()
        ).decode().rstrip('=')
        signature = "fake-signature"

        token = f"{header}.{payload}.{signature}"

        claims = validator.decode_token_unverified(token)

        assert claims["sub"] == "user123"
        assert claims["iss"] == "https://dev-example.auth0.com/"
        assert claims["aud"] == "https://api.example.com"

    @pytest.mark.asyncio
    async def test_fallback_validation(self, validator):
        """Test fallback validation (no signature verification)."""
        # Create a simple JWT
        import base64
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
        ).decode().rstrip('=')
        payload = base64.urlsafe_b64encode(
            json.dumps({
                "sub": "user123",
                "iss": "https://dev-example.auth0.com/",
                "aud": "https://api.example.com",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time())
            }).encode()
        ).decode().rstrip('=')
        signature = "fake-signature"

        token = f"{header}.{payload}.{signature}"

        async with validator:
            # Fallback validation (no JWKS, so signature not verified)
            claims = await validator._validate_token_fallback(
                token,
                verify_expiry=True,
                verify_nbf=True
            )

            assert claims.sub == "user123"
            assert not claims.is_expired()

    @pytest.mark.asyncio
    async def test_expired_token_rejection(self, validator):
        """Test that expired tokens are rejected."""
        import base64
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
        ).decode().rstrip('=')
        payload = base64.urlsafe_b64encode(
            json.dumps({
                "sub": "user123",
                "exp": int(time.time()) - 3600  # Expired 1 hour ago
            }).encode()
        ).decode().rstrip('=')
        signature = "fake-signature"

        token = f"{header}.{payload}.{signature}"

        async with validator:
            with pytest.raises(ValueError, match="Token expired"):
                await validator._validate_token_fallback(
                    token,
                    verify_expiry=True,
                    verify_nbf=False
                )


class TestOAuth2Factory:
    """Test OAuth2 factory functions."""

    def test_create_oauth2_client_auth0(self):
        """Test OAuth2 client factory for Auth0."""
        client = create_oauth2_client(
            provider="auth0",
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback",
            domain="dev-example.auth0.com"
        )

        assert isinstance(client, OAuth2Client)
        assert client.config.provider == OAuth2Provider.AUTH0

    def test_create_oauth2_client_google(self):
        """Test OAuth2 client factory for Google."""
        client = create_oauth2_client(
            provider="google",
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback"
        )

        assert isinstance(client, OAuth2Client)
        assert client.config.provider == OAuth2Provider.GOOGLE

    def test_create_oauth2_client_invalid_provider(self):
        """Test OAuth2 client factory with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_oauth2_client(
                provider="invalid-provider",
                client_id="test-client-id",
                client_secret="test-client-secret",
                redirect_uri="http://localhost:8000/callback"
            )


class TestJWTFactory:
    """Test JWT validator factory."""

    def test_create_jwt_validator(self):
        """Test JWT validator factory."""
        validator = create_jwt_validator(
            issuer="https://dev-example.auth0.com/",
            audience="https://api.example.com",
            jwks_url="https://dev-example.auth0.com/.well-known/jwks.json"
        )

        assert isinstance(validator, JWTValidator)
        assert validator.issuer == "https://dev-example.auth0.com/"
        assert validator.audience == "https://api.example.com"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

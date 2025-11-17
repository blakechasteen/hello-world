"""
OAuth2 Authentication Manager

Handles OAuth2 flows for Canvas, Google Classroom, and Microsoft.

Author: Agent D
Date: 2025-11-15
"""

import os
import json
import time
import secrets
import hashlib
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
from cryptography.fernet import Fernet


class OAuth2Provider(Enum):
    """Supported OAuth2 providers"""
    CANVAS = "canvas"
    GOOGLE = "google"
    MICROSOFT = "microsoft"


@dataclass
class OAuth2Token:
    """OAuth2 access token with metadata"""
    access_token: str
    refresh_token: Optional[str]
    expires_at: datetime
    token_type: str = "Bearer"
    scope: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now() >= self.expires_at

    def expires_soon(self, buffer_seconds: int = 300) -> bool:
        """Check if token expires within buffer_seconds"""
        return datetime.now() >= (self.expires_at - timedelta(seconds=buffer_seconds))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at.isoformat(),
            "token_type": self.token_type,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuth2Token":
        """Create from dictionary"""
        return cls(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            token_type=data.get("token_type", "Bearer"),
            scope=data.get("scope"),
        )


class OAuth2Manager:
    """
    OAuth2 authentication manager supporting multiple providers.

    Features:
    - Authorization code flow
    - Automatic token refresh
    - Secure token storage (encrypted)
    - Multiple provider support
    """

    # Provider-specific endpoints
    PROVIDER_CONFIGS = {
        OAuth2Provider.CANVAS: {
            "auth_endpoint": "/login/oauth2/auth",
            "token_endpoint": "/login/oauth2/token",
            "revoke_endpoint": "/login/oauth2/token",
        },
        OAuth2Provider.GOOGLE: {
            "auth_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
            "revoke_endpoint": "https://oauth2.googleapis.com/revoke",
        },
        OAuth2Provider.MICROSOFT: {
            "auth_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            "token_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
            "revoke_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/logout",
        },
    }

    def __init__(self, token_storage_path: str = "./.oauth_tokens"):
        """
        Initialize OAuth2 manager.

        Args:
            token_storage_path: Path to store encrypted tokens
        """
        self.token_storage_path = token_storage_path
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        self.tokens: Dict[str, OAuth2Token] = {}
        self._load_tokens()

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for token storage"""
        key_path = f"{self.token_storage_path}.key"

        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_path) if os.path.dirname(key_path) else ".", exist_ok=True)
            with open(key_path, "wb") as f:
                f.write(key)
            return key

    def _load_tokens(self):
        """Load encrypted tokens from disk"""
        if not os.path.exists(self.token_storage_path):
            return

        with open(self.token_storage_path, "rb") as f:
            encrypted_data = f.read()

        try:
            decrypted_data = self.fernet.decrypt(encrypted_data)
            tokens_dict = json.loads(decrypted_data)

            for provider, token_data in tokens_dict.items():
                self.tokens[provider] = OAuth2Token.from_dict(token_data)
        except Exception as e:
            print(f"Error loading tokens: {e}")

    def _save_tokens(self):
        """Save encrypted tokens to disk"""
        tokens_dict = {
            provider: token.to_dict()
            for provider, token in self.tokens.items()
        }

        plaintext_data = json.dumps(tokens_dict).encode()
        encrypted_data = self.fernet.encrypt(plaintext_data)

        os.makedirs(os.path.dirname(self.token_storage_path) if os.path.dirname(self.token_storage_path) else ".", exist_ok=True)
        with open(self.token_storage_path, "wb") as f:
            f.write(encrypted_data)

    def generate_auth_url(
        self,
        provider: OAuth2Provider,
        base_url: str,
        client_id: str,
        redirect_uri: str,
        scopes: List[str],
        state: Optional[str] = None,
    ) -> str:
        """
        Generate OAuth2 authorization URL.

        Args:
            provider: OAuth2 provider
            base_url: Base URL for provider (Canvas only)
            client_id: OAuth2 client ID
            redirect_uri: Redirect URI
            scopes: List of OAuth2 scopes
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        config = self.PROVIDER_CONFIGS[provider]

        if provider == OAuth2Provider.CANVAS:
            auth_url = f"{base_url}{config['auth_endpoint']}"
        else:
            auth_url = config["auth_endpoint"]

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
        }

        if provider == OAuth2Provider.GOOGLE:
            params["scope"] = " ".join(scopes)
            params["access_type"] = "offline"  # Get refresh token
            params["prompt"] = "consent"  # Force consent to get refresh token
        elif provider == OAuth2Provider.MICROSOFT:
            params["scope"] = " ".join(scopes)
            params["response_mode"] = "query"
        else:  # Canvas
            params["scope"] = " ".join(scopes)

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{auth_url}?{query_string}"

    async def exchange_code_for_token(
        self,
        provider: OAuth2Provider,
        base_url: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        code: str,
    ) -> OAuth2Token:
        """
        Exchange authorization code for access token.

        Args:
            provider: OAuth2 provider
            base_url: Base URL for provider (Canvas only)
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            redirect_uri: Redirect URI
            code: Authorization code

        Returns:
            OAuth2Token
        """
        config = self.PROVIDER_CONFIGS[provider]

        if provider == OAuth2Provider.CANVAS:
            token_url = f"{base_url}{config['token_endpoint']}"
        else:
            token_url = config["token_endpoint"]

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Token exchange failed: {error_text}")

                token_data = await response.json()

        # Calculate expiration time
        expires_in = token_data.get("expires_in", 3600)
        expires_at = datetime.now() + timedelta(seconds=expires_in)

        token = OAuth2Token(
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token"),
            expires_at=expires_at,
            token_type=token_data.get("token_type", "Bearer"),
            scope=token_data.get("scope"),
        )

        # Store token
        self.tokens[provider.value] = token
        self._save_tokens()

        return token

    async def refresh_token(
        self,
        provider: OAuth2Provider,
        base_url: str,
        client_id: str,
        client_secret: str,
    ) -> OAuth2Token:
        """
        Refresh access token using refresh token.

        Args:
            provider: OAuth2 provider
            base_url: Base URL for provider (Canvas only)
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret

        Returns:
            New OAuth2Token
        """
        current_token = self.tokens.get(provider.value)
        if not current_token or not current_token.refresh_token:
            raise Exception("No refresh token available")

        config = self.PROVIDER_CONFIGS[provider]

        if provider == OAuth2Provider.CANVAS:
            token_url = f"{base_url}{config['token_endpoint']}"
        else:
            token_url = config["token_endpoint"]

        data = {
            "grant_type": "refresh_token",
            "refresh_token": current_token.refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Token refresh failed: {error_text}")

                token_data = await response.json()

        # Calculate expiration time
        expires_in = token_data.get("expires_in", 3600)
        expires_at = datetime.now() + timedelta(seconds=expires_in)

        token = OAuth2Token(
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token", current_token.refresh_token),
            expires_at=expires_at,
            token_type=token_data.get("token_type", "Bearer"),
            scope=token_data.get("scope"),
        )

        # Store token
        self.tokens[provider.value] = token
        self._save_tokens()

        return token

    def get_token(self, provider: OAuth2Provider) -> Optional[OAuth2Token]:
        """Get token for provider"""
        return self.tokens.get(provider.value)

    async def get_valid_token(
        self,
        provider: OAuth2Provider,
        base_url: str,
        client_id: str,
        client_secret: str,
    ) -> Optional[OAuth2Token]:
        """
        Get valid token, refreshing if necessary.

        Args:
            provider: OAuth2 provider
            base_url: Base URL for provider (Canvas only)
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret

        Returns:
            Valid OAuth2Token or None if unavailable
        """
        token = self.get_token(provider)

        if not token:
            return None

        # Refresh if expired or expiring soon
        if token.is_expired() or token.expires_soon():
            try:
                token = await self.refresh_token(
                    provider, base_url, client_id, client_secret
                )
            except Exception as e:
                print(f"Token refresh failed: {e}")
                return None

        return token

    def revoke_token(self, provider: OAuth2Provider):
        """Revoke and remove token"""
        if provider.value in self.tokens:
            del self.tokens[provider.value]
            self._save_tokens()

    def clear_all_tokens(self):
        """Clear all stored tokens"""
        self.tokens.clear()
        self._save_tokens()

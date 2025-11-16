"""
HoloLoom API Key Management
============================

API key generation, storage, and verification.

Features:
- Prefixed keys (hololoom_live_, hololoom_test_)
- Secure random generation
- In-memory storage (demo)
- Last used tracking
- Expiry support (optional)

Format: hololoom_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6 (32 chars)

Created: 2025-11-16
"""

import os
import secrets
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class APIKeyType(str, Enum):
    """API key type enumeration."""
    LIVE = "live"
    TEST = "test"


@dataclass
class APIKey:
    """API key model."""
    key_id: str
    key: str
    username: str
    key_type: APIKeyType = APIKeyType.LIVE
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    expires_at: Optional[str] = None
    last_used: Optional[str] = None
    active: bool = True

    def to_dict(self, include_key: bool = False) -> Dict:
        """
        Convert to dictionary.

        Args:
            include_key: Include full key (only show on creation)

        Returns:
            Dictionary representation
        """
        data = {
            "key_id": self.key_id,
            "username": self.username,
            "key_type": self.key_type.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used": self.last_used,
            "active": self.active,
        }

        if include_key:
            data["key"] = self.key
        else:
            # Show only prefix for security
            data["key_prefix"] = self.key[:20] + "..."

        return data

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if not self.expires_at:
            return False

        expiry = datetime.fromisoformat(self.expires_at.replace("Z", ""))
        return datetime.utcnow() > expiry

    def mark_used(self) -> None:
        """Mark key as used (update last_used timestamp)."""
        self.last_used = datetime.utcnow().isoformat() + "Z"


class APIKeyManager:
    """
    API key manager.

    Stores keys in memory (demo).
    In production, use PostgreSQL/SQLite.
    """

    def __init__(self):
        self.keys: Dict[str, APIKey] = {}
        self.prefix = os.getenv("API_KEY_PREFIX", "hololoom")

    def generate_key(
        self,
        username: str,
        key_type: APIKeyType = APIKeyType.LIVE,
        expires_in_days: Optional[int] = None,
    ) -> APIKey:
        """
        Generate new API key.

        Args:
            username: Username to associate with key
            key_type: Key type (live or test)
            expires_in_days: Optional expiry in days

        Returns:
            Generated API key
        """
        # Generate secure random key
        random_part = secrets.token_urlsafe(24)  # ~32 chars after encoding

        # Format: hololoom_live_xxxxx or hololoom_test_xxxxx
        key = f"{self.prefix}_{key_type.value}_{random_part}"

        # Generate key ID
        key_id = f"ak_{secrets.token_hex(4)}"

        # Calculate expiry
        expires_at = None
        if expires_in_days:
            expiry = datetime.utcnow() + timedelta(days=expires_in_days)
            expires_at = expiry.isoformat() + "Z"

        api_key = APIKey(
            key_id=key_id,
            key=key,
            username=username,
            key_type=key_type,
            expires_at=expires_at,
        )

        self.keys[key] = api_key
        logger.info(f"API key generated: {key_id} for {username} ({key_type.value})")

        return api_key

    def verify_key(self, key: str) -> Optional[APIKey]:
        """
        Verify API key.

        Args:
            key: API key to verify

        Returns:
            APIKey if valid, None if invalid
        """
        api_key = self.keys.get(key)

        if not api_key:
            logger.warning(f"API key not found: {key[:20]}...")
            return None

        if not api_key.active:
            logger.warning(f"API key inactive: {api_key.key_id}")
            return None

        if api_key.is_expired():
            logger.warning(f"API key expired: {api_key.key_id}")
            return None

        # Update last used
        api_key.mark_used()
        logger.info(f"API key verified: {api_key.key_id} for {api_key.username}")

        return api_key

    def list_keys(self, username: Optional[str] = None) -> List[APIKey]:
        """
        List API keys.

        Args:
            username: Optional filter by username

        Returns:
            List of API keys
        """
        if username:
            return [k for k in self.keys.values() if k.username == username]

        return list(self.keys.values())

    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke API key.

        Args:
            key_id: Key ID to revoke

        Returns:
            True if revoked, False if not found
        """
        for key, api_key in self.keys.items():
            if api_key.key_id == key_id:
                api_key.active = False
                logger.info(f"API key revoked: {key_id}")
                return True

        logger.warning(f"API key not found for revocation: {key_id}")
        return False

    def delete_key(self, key_id: str) -> bool:
        """
        Delete API key permanently.

        Args:
            key_id: Key ID to delete

        Returns:
            True if deleted, False if not found
        """
        for key, api_key in self.keys.items():
            if api_key.key_id == key_id:
                del self.keys[key]
                logger.info(f"API key deleted: {key_id}")
                return True

        logger.warning(f"API key not found for deletion: {key_id}")
        return False

    def cleanup_expired(self) -> int:
        """
        Remove expired keys.

        Returns:
            Number of keys removed
        """
        expired = [
            key for key, api_key in self.keys.items()
            if api_key.is_expired()
        ]

        for key in expired:
            del self.keys[key]

        logger.info(f"Cleaned up {len(expired)} expired API keys")
        return len(expired)


# Global API key manager instance
api_key_manager = APIKeyManager()


def generate_api_key(
    username: str,
    key_type: APIKeyType = APIKeyType.LIVE,
    expires_in_days: Optional[int] = None,
) -> APIKey:
    """
    Generate new API key (convenience wrapper).

    Args:
        username: Username to associate with key
        key_type: Key type (live or test)
        expires_in_days: Optional expiry in days

    Returns:
        Generated API key
    """
    return api_key_manager.generate_key(username, key_type, expires_in_days)


def verify_api_key(key: str) -> Optional[APIKey]:
    """
    Verify API key (convenience wrapper).

    Args:
        key: API key to verify

    Returns:
        APIKey if valid, None if invalid
    """
    return api_key_manager.verify_key(key)


# Example usage
if __name__ == "__main__":
    # Generate key
    key = generate_api_key("admin", APIKeyType.LIVE, expires_in_days=30)
    print(f"Generated: {key.to_dict(include_key=True)}")

    # Verify key
    verified = verify_api_key(key.key)
    if verified:
        print(f"Verified: {verified.to_dict()}")

    # List keys
    keys = api_key_manager.list_keys("admin")
    print(f"\nKeys for admin: {len(keys)}")
    for k in keys:
        print(f"  - {k.to_dict()}")

    # Revoke key
    revoked = api_key_manager.revoke_key(key.key_id)
    print(f"\nRevoked: {revoked}")

    # Verify revoked key (should fail)
    verified = verify_api_key(key.key)
    print(f"Revoked key verification: {verified}")

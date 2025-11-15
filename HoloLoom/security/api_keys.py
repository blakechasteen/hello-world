"""
API Key Management

Secure API key generation, validation, and rotation for HoloLoom.

Features:
- Cryptographically secure key generation (256-bit)
- PBKDF2-HMAC-SHA256 hashing (never store raw keys!)
- Scoped permissions (read, write, admin)
- Key expiration and rotation
- Constant-time comparison (prevents timing attacks)

References:
- OWASP API Security Top 10
- NIST SP 800-132 (Password-Based Key Derivation)

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import secrets
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class APIKeyScope(Enum):
    """API key permission scopes."""
    READ = "read"  # Read-only access (queries, statistics)
    WRITE = "write"  # Read + write access (ingest, update)
    ADMIN = "admin"  # Full access (manage keys, users, system)


@dataclass
class APIKey:
    """
    API key with scoped permissions and expiration.

    Attributes:
        key_id: Public identifier (safe to log)
        key_hash: PBKDF2-HMAC-SHA256 hash of raw key
        user_id: Associated user identifier
        scopes: List of permission scopes
        created_at: Key creation timestamp
        expires_at: Optional expiration timestamp
        is_active: Whether key is active (for soft delete)
        name: Optional friendly name
        metadata: Optional additional metadata
    """
    key_id: str
    key_hash: str
    user_id: str
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    name: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for storage)."""
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "user_id": self.user_id,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "name": self.name,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKey":
        """Deserialize from dictionary."""
        return cls(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            user_id=data["user_id"],
            scopes=data["scopes"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at") else None
            ),
            is_active=data.get("is_active", True),
            name=data.get("name"),
            metadata=data.get("metadata")
        )

    def has_scope(self, scope: str) -> bool:
        """Check if key has specific scope."""
        return scope in self.scopes or "admin" in self.scopes

    def is_valid(self) -> bool:
        """Check if key is valid (active + not expired)."""
        if not self.is_active:
            return False

        if self.expires_at and datetime.now() > self.expires_at:
            return False

        return True


class APIKeyManager:
    """
    Manage API keys securely.

    Uses PBKDF2-HMAC-SHA256 for key hashing with 100,000 iterations.
    Never stores raw keys (shown only once on generation).

    Usage:
        >>> manager = APIKeyManager(secret="your-master-secret")
        >>> raw_key, api_key = manager.generate_key(
        ...     user_id="user_123",
        ...     scopes=["read", "write"],
        ...     name="Production API Key"
        ... )
        >>> # IMPORTANT: Save raw_key now! Never shown again!
        >>> print(f"Your API key: {raw_key}")
        >>>
        >>> # Later: Verify key
        >>> if manager.verify_key(raw_key, api_key):
        ...     print("Valid key!")
    """

    # Security constants
    HASH_ITERATIONS = 100000  # PBKDF2 iterations (100k recommended by NIST)
    KEY_LENGTH = 32  # 256 bits
    HASH_ALGORITHM = 'sha256'

    def __init__(self, secret: str, iterations: int = HASH_ITERATIONS):
        """
        Initialize API key manager.

        Args:
            secret: Master secret for PBKDF2 salt (keep secure!)
            iterations: PBKDF2 iterations (higher = more secure, slower)
        """
        self.secret = secret.encode()
        self.iterations = iterations

        if len(secret) < 32:
            logger.warning(
                "⚠️  API key master secret should be at least 32 characters! "
                "Generate with: openssl rand -hex 32"
            )

    def generate_key(
        self,
        user_id: str,
        scopes: List[str],
        ttl_days: int = 365,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, APIKey]:
        """
        Generate new API key.

        Args:
            user_id: User identifier
            scopes: List of scopes (["read"], ["read", "write"], ["admin"])
            ttl_days: Time to live in days (default: 365)
            name: Optional friendly name
            metadata: Optional additional metadata

        Returns:
            (raw_key, api_key_obj)

        IMPORTANT: raw_key is shown only once! Store it securely.
        """
        # Generate cryptographically secure random key
        raw_key = secrets.token_urlsafe(self.KEY_LENGTH)  # 256 bits

        # Hash for storage (PBKDF2-HMAC-SHA256)
        key_hash = self._hash_key(raw_key)

        # Generate public key ID
        key_id = secrets.token_urlsafe(16)  # 128 bits

        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            scopes=scopes,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=ttl_days),
            is_active=True,
            name=name,
            metadata=metadata
        )

        logger.info(
            f"✅ Generated API key: {key_id} for user {user_id} "
            f"(scopes: {scopes}, expires: {api_key.expires_at.strftime('%Y-%m-%d')})"
        )

        return raw_key, api_key

    def verify_key(self, raw_key: str, stored_key: APIKey) -> bool:
        """
        Verify API key against stored hash.

        Args:
            raw_key: Raw API key from user
            stored_key: Stored APIKey object

        Returns:
            True if key matches, False otherwise

        Note: Uses constant-time comparison to prevent timing attacks.
        """
        # Check if key is valid (active + not expired)
        if not stored_key.is_valid():
            logger.warning(
                f"❌ Key verification failed: {stored_key.key_id} "
                f"(active: {stored_key.is_active}, "
                f"expired: {stored_key.expires_at < datetime.now() if stored_key.expires_at else False})"
            )
            return False

        # Re-hash provided key
        key_hash = self._hash_key(raw_key)

        # Compare hashes (constant-time to prevent timing attacks)
        valid = secrets.compare_digest(key_hash, stored_key.key_hash)

        if valid:
            logger.debug(f"✅ Key verified: {stored_key.key_id}")
        else:
            logger.warning(f"❌ Invalid key provided for: {stored_key.key_id}")

        return valid

    def rotate_key(
        self,
        old_key: APIKey,
        ttl_days: int = 365
    ) -> tuple[str, APIKey]:
        """
        Rotate API key (generate new, invalidate old).

        Args:
            old_key: Old APIKey object
            ttl_days: Time to live for new key

        Returns:
            (raw_key, new_api_key)

        Usage:
            >>> raw_key, new_key = manager.rotate_key(old_key)
            >>> # Update database: deactivate old_key, store new_key
            >>> print(f"New API key: {raw_key}")
        """
        # Generate new key with same permissions
        raw_key, new_key = self.generate_key(
            user_id=old_key.user_id,
            scopes=old_key.scopes,
            ttl_days=ttl_days,
            name=f"{old_key.name} (rotated)" if old_key.name else None,
            metadata=old_key.metadata
        )

        # Invalidate old key
        old_key.is_active = False

        logger.info(
            f"🔄 Rotated key: {old_key.key_id} → {new_key.key_id}"
        )

        return raw_key, new_key

    def revoke_key(self, api_key: APIKey):
        """
        Revoke API key (soft delete).

        Args:
            api_key: APIKey object to revoke
        """
        api_key.is_active = False
        logger.info(f"🗑️  Revoked key: {api_key.key_id}")

    def _hash_key(self, raw_key: str) -> str:
        """
        Hash API key using PBKDF2-HMAC-SHA256.

        Args:
            raw_key: Raw API key

        Returns:
            Hex-encoded hash
        """
        return hashlib.pbkdf2_hmac(
            self.HASH_ALGORITHM,
            raw_key.encode(),
            self.secret,
            self.iterations
        ).hex()


# Factory function
def create_api_key_manager(
    secret: Optional[str] = None,
    iterations: int = APIKeyManager.HASH_ITERATIONS
) -> APIKeyManager:
    """
    Create API key manager with secret from environment or provided.

    Args:
        secret: Optional master secret (defaults to API_KEY_SECRET env var)
        iterations: PBKDF2 iterations

    Returns:
        APIKeyManager instance

    Usage:
        >>> import os
        >>> os.environ["API_KEY_SECRET"] = "your-secret-here"
        >>> manager = create_api_key_manager()
    """
    import os

    if secret is None:
        secret = os.environ.get("API_KEY_SECRET")

        if secret is None:
            raise ValueError(
                "API_KEY_SECRET environment variable not set! "
                "Set with: export API_KEY_SECRET=$(openssl rand -hex 32)"
            )

    return APIKeyManager(secret=secret, iterations=iterations)


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = APIKeyManager(secret="demo-secret-change-me-in-production")

    # Generate API key
    print("=" * 70)
    print("API Key Generation Demo")
    print("=" * 70)

    raw_key, api_key = manager.generate_key(
        user_id="user_123",
        scopes=["read", "write"],
        name="Development Key",
        metadata={"environment": "dev"}
    )

    print(f"\n✅ Generated API Key:")
    print(f"   Key ID: {api_key.key_id}")
    print(f"   Raw Key: {raw_key}")
    print(f"   ⚠️  SAVE THIS KEY! It will never be shown again!")
    print(f"\n   User: {api_key.user_id}")
    print(f"   Scopes: {', '.join(api_key.scopes)}")
    print(f"   Name: {api_key.name}")
    print(f"   Created: {api_key.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Expires: {api_key.expires_at.strftime('%Y-%m-%d')}")

    # Verify key
    print(f"\n\n{'=' * 70}")
    print("API Key Verification Demo")
    print("=" * 70)

    is_valid = manager.verify_key(raw_key, api_key)
    print(f"\n✅ Key valid: {is_valid}")

    # Test invalid key
    is_valid = manager.verify_key("wrong-key", api_key)
    print(f"❌ Invalid key rejected: {not is_valid}")

    # Test scope checking
    print(f"\n\n{'=' * 70}")
    print("Scope Checking Demo")
    print("=" * 70)

    print(f"\n   Has 'read' scope: {api_key.has_scope('read')}")
    print(f"   Has 'write' scope: {api_key.has_scope('write')}")
    print(f"   Has 'admin' scope: {api_key.has_scope('admin')}")

    # Rotate key
    print(f"\n\n{'=' * 70}")
    print("Key Rotation Demo")
    print("=" * 70)

    new_raw_key, new_api_key = manager.rotate_key(api_key)

    print(f"\n🔄 Rotated Key:")
    print(f"   Old Key ID: {api_key.key_id} (active: {api_key.is_active})")
    print(f"   New Key ID: {new_api_key.key_id}")
    print(f"   New Raw Key: {new_raw_key}")

    # Verify old key is invalid
    is_valid = manager.verify_key(raw_key, api_key)
    print(f"\n   Old key still valid: {is_valid} (should be False)")

    # Verify new key is valid
    is_valid = manager.verify_key(new_raw_key, new_api_key)
    print(f"   New key valid: {is_valid} (should be True)")

    print(f"\n{'=' * 70}\n")

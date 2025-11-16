"""
Test API key management module.

Tests:
- API key generation
- API key verification
- API key revocation
- Key expiry
- Key listing
"""

import pytest
from datetime import datetime

from HoloLoom.auth.api_keys import (
    APIKeyManager,
    APIKeyType,
    generate_api_key,
    verify_api_key,
    api_key_manager,
)


@pytest.fixture
def key_manager():
    """Create fresh API key manager for each test."""
    manager = APIKeyManager()
    yield manager
    # Cleanup
    manager.keys.clear()


def test_generate_live_key(key_manager):
    """Test generating live API key."""
    key = key_manager.generate_key("testuser", APIKeyType.LIVE)

    assert key.key.startswith("hololoom_live_")
    assert key.username == "testuser"
    assert key.key_type == APIKeyType.LIVE
    assert key.active is True
    assert key.expires_at is None


def test_generate_test_key(key_manager):
    """Test generating test API key."""
    key = key_manager.generate_key("testuser", APIKeyType.TEST)

    assert key.key.startswith("hololoom_test_")
    assert key.key_type == APIKeyType.TEST


def test_generate_key_with_expiry(key_manager):
    """Test generating key with expiry."""
    key = key_manager.generate_key("testuser", APIKeyType.LIVE, expires_in_days=30)

    assert key.expires_at is not None

    # Parse expiry date
    expiry = datetime.fromisoformat(key.expires_at.replace("Z", ""))
    now = datetime.utcnow()

    # Should be approximately 30 days in future
    days_until_expiry = (expiry - now).days
    assert 29 <= days_until_expiry <= 30


def test_verify_valid_key(key_manager):
    """Test verifying valid API key."""
    generated_key = key_manager.generate_key("testuser", APIKeyType.LIVE)

    # Verify key
    verified = key_manager.verify_key(generated_key.key)

    assert verified is not None
    assert verified.key_id == generated_key.key_id
    assert verified.username == "testuser"
    assert verified.last_used is not None  # Should be updated


def test_verify_invalid_key(key_manager):
    """Test verifying invalid API key."""
    verified = key_manager.verify_key("hololoom_live_invalid")

    assert verified is None


def test_verify_revoked_key(key_manager):
    """Test verifying revoked key."""
    key = key_manager.generate_key("testuser", APIKeyType.LIVE)

    # Revoke key
    key_manager.revoke_key(key.key_id)

    # Verify key fails
    verified = key_manager.verify_key(key.key)
    assert verified is None


def test_verify_expired_key(key_manager):
    """Test verifying expired key."""
    # Create key that expires immediately
    key = key_manager.generate_key("testuser", APIKeyType.LIVE, expires_in_days=-1)

    # Verify key fails
    verified = key_manager.verify_key(key.key)
    assert verified is None


def test_list_keys_all(key_manager):
    """Test listing all API keys."""
    # Generate multiple keys
    key_manager.generate_key("user1", APIKeyType.LIVE)
    key_manager.generate_key("user2", APIKeyType.LIVE)
    key_manager.generate_key("user1", APIKeyType.TEST)

    # List all keys
    keys = key_manager.list_keys()

    assert len(keys) == 3


def test_list_keys_by_username(key_manager):
    """Test listing API keys for specific user."""
    # Generate keys for different users
    key_manager.generate_key("user1", APIKeyType.LIVE)
    key_manager.generate_key("user2", APIKeyType.LIVE)
    key_manager.generate_key("user1", APIKeyType.TEST)

    # List keys for user1
    keys = key_manager.list_keys(username="user1")

    assert len(keys) == 2
    for key in keys:
        assert key.username == "user1"


def test_revoke_key(key_manager):
    """Test revoking API key."""
    key = key_manager.generate_key("testuser", APIKeyType.LIVE)

    # Revoke key
    revoked = key_manager.revoke_key(key.key_id)
    assert revoked is True

    # Verify key is inactive
    keys = key_manager.list_keys()
    revoked_key = next(k for k in keys if k.key_id == key.key_id)
    assert revoked_key.active is False


def test_revoke_nonexistent_key(key_manager):
    """Test revoking nonexistent key."""
    revoked = key_manager.revoke_key("ak_nonexistent")
    assert revoked is False


def test_delete_key(key_manager):
    """Test permanently deleting key."""
    key = key_manager.generate_key("testuser", APIKeyType.LIVE)

    # Delete key
    deleted = key_manager.delete_key(key.key_id)
    assert deleted is True

    # Verify key is gone
    keys = key_manager.list_keys()
    assert len(keys) == 0


def test_delete_nonexistent_key(key_manager):
    """Test deleting nonexistent key."""
    deleted = key_manager.delete_key("ak_nonexistent")
    assert deleted is False


def test_cleanup_expired_keys(key_manager):
    """Test cleaning up expired keys."""
    # Create valid key
    valid_key = key_manager.generate_key("user1", APIKeyType.LIVE, expires_in_days=30)

    # Create expired key
    expired_key = key_manager.generate_key("user2", APIKeyType.LIVE, expires_in_days=-1)

    assert len(key_manager.keys) == 2

    # Cleanup expired
    cleaned = key_manager.cleanup_expired()

    assert cleaned == 1
    assert len(key_manager.keys) == 1
    assert valid_key.key in key_manager.keys


def test_key_to_dict_without_key(key_manager):
    """Test converting key to dict without full key."""
    key = key_manager.generate_key("testuser", APIKeyType.LIVE)

    key_dict = key.to_dict(include_key=False)

    assert "key" not in key_dict
    assert "key_prefix" in key_dict
    assert key_dict["key_prefix"].endswith("...")


def test_key_to_dict_with_key(key_manager):
    """Test converting key to dict with full key."""
    key = key_manager.generate_key("testuser", APIKeyType.LIVE)

    key_dict = key.to_dict(include_key=True)

    assert "key" in key_dict
    assert key_dict["key"] == key.key


def test_global_api_key_manager():
    """Test global API key manager instance."""
    # Generate key using global instance
    key = generate_api_key("testuser", APIKeyType.LIVE)

    assert key is not None
    assert key.key.startswith("hololoom_live_")

    # Verify using global instance
    verified = verify_api_key(key.key)
    assert verified is not None
    assert verified.username == "testuser"

    # Cleanup
    api_key_manager.delete_key(key.key_id)


def test_last_used_tracking(key_manager):
    """Test that last_used is updated on verification."""
    key = key_manager.generate_key("testuser", APIKeyType.LIVE)

    assert key.last_used is None

    # Verify key
    key_manager.verify_key(key.key)

    # Check last_used was updated
    assert key.last_used is not None


def test_key_id_uniqueness(key_manager):
    """Test that key IDs are unique."""
    key1 = key_manager.generate_key("user1", APIKeyType.LIVE)
    key2 = key_manager.generate_key("user2", APIKeyType.LIVE)

    assert key1.key_id != key2.key_id
    assert key1.key != key2.key

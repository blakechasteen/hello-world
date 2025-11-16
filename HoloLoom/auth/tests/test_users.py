"""
Test user management module.

Tests:
- User authentication
- Password hashing
- User CRUD operations
- Role management
"""

import pytest

from HoloLoom.auth.users import (
    authenticate_user,
    get_user,
    create_user,
    delete_user,
    list_users,
    update_user_role,
    deactivate_user,
    activate_user,
    UserRole,
    is_bcrypt_available,
)


def test_authenticate_admin():
    """Test authenticating default admin user."""
    user = authenticate_user("admin", "admin")

    assert user is not None
    assert user.username == "admin"
    assert user.role == UserRole.ADMIN
    assert user.active is True


def test_authenticate_invalid_password():
    """Test authentication with invalid password."""
    user = authenticate_user("admin", "wrongpassword")

    assert user is None


def test_authenticate_nonexistent_user():
    """Test authentication with nonexistent user."""
    user = authenticate_user("nonexistent", "password")

    assert user is None


def test_get_user():
    """Test getting user by username."""
    user = get_user("admin")

    assert user is not None
    assert user.username == "admin"


def test_get_nonexistent_user():
    """Test getting nonexistent user."""
    user = get_user("nonexistent")

    assert user is None


def test_create_user():
    """Test creating new user."""
    # Create user
    user = create_user("testuser", "testpass", UserRole.USER)

    assert user.username == "testuser"
    assert user.role == UserRole.USER
    assert user.verify_password("testpass") is True

    # Verify user can be retrieved
    retrieved = get_user("testuser")
    assert retrieved is not None
    assert retrieved.username == "testuser"

    # Cleanup
    delete_user("testuser")


def test_create_duplicate_user():
    """Test creating user with existing username."""
    # Create first user
    user = create_user("duplicatetest", "password", UserRole.USER)

    # Try to create duplicate
    with pytest.raises(ValueError, match="User already exists"):
        create_user("duplicatetest", "password", UserRole.USER)

    # Cleanup
    delete_user("duplicatetest")


def test_delete_user():
    """Test deleting user."""
    # Create user
    create_user("deletetest", "password", UserRole.USER)

    # Delete user
    deleted = delete_user("deletetest")
    assert deleted is True

    # Verify user is gone
    user = get_user("deletetest")
    assert user is None


def test_delete_nonexistent_user():
    """Test deleting nonexistent user."""
    deleted = delete_user("nonexistent")
    assert deleted is False


def test_list_users():
    """Test listing all users."""
    users = list_users()

    assert len(users) >= 2  # At least admin and demo
    usernames = [u.username for u in users]
    assert "admin" in usernames
    assert "demo" in usernames


def test_update_user_role():
    """Test updating user role."""
    # Create user
    create_user("roletest", "password", UserRole.USER)

    # Update role
    updated = update_user_role("roletest", UserRole.ADMIN)
    assert updated is True

    # Verify role changed
    user = get_user("roletest")
    assert user.role == UserRole.ADMIN

    # Cleanup
    delete_user("roletest")


def test_update_nonexistent_user_role():
    """Test updating role for nonexistent user."""
    updated = update_user_role("nonexistent", UserRole.ADMIN)
    assert updated is False


def test_deactivate_user():
    """Test deactivating user."""
    # Create user
    create_user("deactivatetest", "password", UserRole.USER)

    # Deactivate
    deactivated = deactivate_user("deactivatetest")
    assert deactivated is True

    # Verify user is inactive
    user = get_user("deactivatetest")
    assert user.active is False

    # Verify authentication fails for inactive user
    auth_user = authenticate_user("deactivatetest", "password")
    assert auth_user is None

    # Cleanup
    delete_user("deactivatetest")


def test_activate_user():
    """Test activating user."""
    # Create and deactivate user
    create_user("activatetest", "password", UserRole.USER)
    deactivate_user("activatetest")

    # Activate
    activated = activate_user("activatetest")
    assert activated is True

    # Verify user is active
    user = get_user("activatetest")
    assert user.active is True

    # Verify authentication works
    auth_user = authenticate_user("activatetest", "password")
    assert auth_user is not None

    # Cleanup
    delete_user("activatetest")


def test_user_to_dict():
    """Test converting user to dictionary."""
    user = get_user("admin")
    user_dict = user.to_dict()

    assert "username" in user_dict
    assert "role" in user_dict
    assert "active" in user_dict
    assert "password_hash" not in user_dict  # Should not include password


def test_password_verification():
    """Test password verification."""
    user = get_user("admin")

    assert user.verify_password("admin") is True
    assert user.verify_password("wrongpassword") is False


@pytest.mark.skipif(
    not is_bcrypt_available(),
    reason="bcrypt not available"
)
def test_password_hashing_with_bcrypt():
    """Test password hashing uses bcrypt."""
    import bcrypt

    # Create user
    user = create_user("bcrypttest", "testpassword", UserRole.USER)

    # Verify password hash is bcrypt format
    assert user.password_hash.startswith(b"$2b$")

    # Verify bcrypt can verify it
    assert bcrypt.checkpw(b"testpassword", user.password_hash)

    # Cleanup
    delete_user("bcrypttest")

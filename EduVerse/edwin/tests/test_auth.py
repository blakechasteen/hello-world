"""
EdWIN Authentication Tests

Tests for JWT authentication, role-based access control, and user management.

Implementation Date: November 15, 2025

Usage:
    pytest EduVerse/edwin/tests/test_auth.py -v
"""

import pytest
from datetime import datetime, timedelta
from fastapi import HTTPException

from EduVerse.edwin.auth import (
    hash_password,
    verify_password,
    validate_password_strength,
    create_access_token,
    decode_token,
    User,
    UserRole,
    authenticate_user,
    create_user,
    _users_db
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def clear_users():
    """Clear users database before each test"""
    _users_db.clear()
    yield
    _users_db.clear()


@pytest.fixture
async def sample_user():
    """Create a sample user for testing"""
    user = await create_user(
        username="testuser",
        email="test@example.com",
        password="TestPass123!",
        full_name="Test User",
        role=UserRole.STUDENT,
        grade=8
    )
    return user


# ==============================================================================
# Password Tests
# ==============================================================================

def test_password_hashing():
    """Test password hashing and verification"""
    password = "MySecurePassword123!"

    # Hash password
    hashed = hash_password(password)

    # Verify it's hashed (not plaintext)
    assert hashed != password
    assert len(hashed) > 50  # Bcrypt hashes are long

    # Verify correct password
    assert verify_password(password, hashed) is True

    # Verify wrong password
    assert verify_password("WrongPassword", hashed) is False


def test_password_strength_validation():
    """Test password strength requirements"""

    # Valid passwords
    valid_passwords = [
        "TestPass123!",
        "Abcd1234$",
        "MyP@ssw0rd",
    ]

    for pwd in valid_passwords:
        is_valid, error = validate_password_strength(pwd)
        assert is_valid is True, f"Password '{pwd}' should be valid"
        assert error == ""

    # Invalid passwords
    invalid_cases = [
        ("short", "Password must be at least 8 characters long"),
        ("nouppercase123!", "Password must contain at least one uppercase letter"),
        ("NOLOWERCASE123!", "Password must contain at least one lowercase letter"),
        ("NoDigits!", "Password must contain at least one digit"),
        ("NoSpecial123", "Password must contain at least one special character"),
    ]

    for pwd, expected_error in invalid_cases:
        is_valid, error = validate_password_strength(pwd)
        assert is_valid is False, f"Password '{pwd}' should be invalid"
        assert expected_error in error


# ==============================================================================
# JWT Token Tests
# ==============================================================================

def test_create_and_decode_token():
    """Test JWT token creation and decoding"""
    user_id = "user_123"
    username = "testuser"
    role = UserRole.STUDENT

    # Create token
    token = create_access_token(user_id, username, role)

    assert isinstance(token, str)
    assert len(token) > 50  # JWT tokens are long

    # Decode token
    token_data = decode_token(token)

    assert token_data.user_id == user_id
    assert token_data.username == username
    assert token_data.role == role


def test_expired_token():
    """Test that expired tokens are rejected"""
    user_id = "user_123"
    username = "testuser"
    role = UserRole.STUDENT

    # Create token that expires immediately
    token = create_access_token(
        user_id,
        username,
        role,
        expires_delta=timedelta(seconds=-1)  # Already expired
    )

    # Decoding should raise exception
    with pytest.raises(HTTPException) as exc_info:
        decode_token(token)

    assert exc_info.value.status_code == 401


def test_invalid_token():
    """Test that invalid tokens are rejected"""
    invalid_token = "not.a.valid.jwt.token"

    with pytest.raises(HTTPException) as exc_info:
        decode_token(invalid_token)

    assert exc_info.value.status_code == 401


# ==============================================================================
# User Management Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_create_user():
    """Test user creation"""
    user = await create_user(
        username="newuser",
        email="new@example.com",
        password="NewPass123!",
        full_name="New User",
        role=UserRole.TEACHER,
        subjects=["math", "science"]
    )

    assert user.username == "newuser"
    assert user.email == "new@example.com"
    assert user.full_name == "New User"
    assert user.role == UserRole.TEACHER
    assert user.is_active is True
    assert isinstance(user.created_at, datetime)


@pytest.mark.asyncio
async def test_create_user_duplicate_username():
    """Test that duplicate usernames are rejected"""
    # Create first user
    await create_user(
        username="duplicate",
        email="user1@example.com",
        password="Pass123!",
        full_name="User One",
        role=UserRole.STUDENT,
        grade=8
    )

    # Try to create second user with same username
    with pytest.raises(HTTPException) as exc_info:
        await create_user(
            username="duplicate",
            email="user2@example.com",
            password="Pass456!",
            full_name="User Two",
            role=UserRole.STUDENT,
            grade=9
        )

    assert exc_info.value.status_code == 400
    assert "already exists" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_create_user_weak_password():
    """Test that weak passwords are rejected"""
    with pytest.raises(HTTPException) as exc_info:
        await create_user(
            username="weakpass",
            email="weak@example.com",
            password="weak",  # Too weak
            full_name="Weak Password User",
            role=UserRole.STUDENT,
            grade=8
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_authenticate_user(sample_user):
    """Test user authentication"""
    # Successful authentication
    user = await authenticate_user("testuser", "TestPass123!")

    assert user is not None
    assert user.username == "testuser"
    assert user.email == "test@example.com"

    # Wrong password
    user = await authenticate_user("testuser", "WrongPassword")
    assert user is None

    # Non-existent user
    user = await authenticate_user("nonexistent", "Pass123!")
    assert user is None


# ==============================================================================
# Role-Based Access Control Tests
# ==============================================================================

def test_role_hierarchy():
    """Test role hierarchy comparison"""
    assert UserRole.STUDENT < UserRole.PARENT
    assert UserRole.PARENT < UserRole.TEACHER
    assert UserRole.TEACHER < UserRole.ADMIN

    assert UserRole.ADMIN > UserRole.TEACHER
    assert UserRole.TEACHER > UserRole.PARENT
    assert UserRole.PARENT > UserRole.STUDENT


# ==============================================================================
# Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_full_authentication_flow():
    """Test complete authentication flow"""

    # 1. Create user
    user = await create_user(
        username="flowuser",
        email="flow@example.com",
        password="FlowPass123!",
        full_name="Flow User",
        role=UserRole.STUDENT,
        grade=10
    )

    assert user.username == "flowuser"

    # 2. Authenticate
    auth_user = await authenticate_user("flowuser", "FlowPass123!")
    assert auth_user is not None
    assert auth_user.user_id == user.user_id

    # 3. Create token
    token = create_access_token(
        auth_user.user_id,
        auth_user.username,
        auth_user.role
    )

    assert isinstance(token, str)

    # 4. Decode token
    token_data = decode_token(token)

    assert token_data.user_id == user.user_id
    assert token_data.username == "flowuser"
    assert token_data.role == UserRole.STUDENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

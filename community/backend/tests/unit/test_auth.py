"""
Unit tests for authentication endpoints
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.user import User
from auth.security import verify_password, decode_token


@pytest.mark.unit
@pytest.mark.auth
class TestUserRegistration:
    """Tests for user registration endpoint"""

    async def test_register_success(self, client: AsyncClient, test_db: AsyncSession):
        """Test successful user registration"""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "securepassword123",
                "display_name": "New User",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["username"] == "newuser"
        assert data["user"]["email"] == "newuser@example.com"

        # Verify user was created in database
        result = await test_db.execute(
            select(User).where(User.username == "newuser")
        )
        user = result.scalar_one_or_none()
        assert user is not None
        assert user.email == "newuser@example.com"
        assert user.status == "active"
        assert user.role == "member"

    async def test_register_duplicate_username(self, client: AsyncClient, test_user: User):
        """Test registration with duplicate username"""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",  # Already exists
                "email": "different@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == 400
        assert "username already registered" in response.json()["detail"].lower()

    async def test_register_duplicate_email(self, client: AsyncClient, test_user: User):
        """Test registration with duplicate email"""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "differentuser",
                "email": "test@example.com",  # Already exists
                "password": "password123",
            },
        )

        assert response.status_code == 400
        assert "email already registered" in response.json()["detail"].lower()

    async def test_register_weak_password(self, client: AsyncClient):
        """Test registration with weak password"""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "123",  # Too short
            },
        )

        assert response.status_code == 422  # Validation error

    async def test_register_invalid_email(self, client: AsyncClient):
        """Test registration with invalid email format"""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "not-an-email",
                "password": "securepassword123",
            },
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.unit
@pytest.mark.auth
class TestUserLogin:
    """Tests for user login endpoint"""

    async def test_login_success(self, client: AsyncClient, test_user: User):
        """Test successful login with username"""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "username": "testuser",
                "password": "testpassword123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "testuser"

    async def test_login_with_email(self, client: AsyncClient, test_user: User):
        """Test successful login with email"""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "username": "test@example.com",  # Email instead of username
                "password": "testpassword123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user"]["username"] == "testuser"

    async def test_login_wrong_password(self, client: AsyncClient, test_user: User):
        """Test login with incorrect password"""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "username": "testuser",
                "password": "wrongpassword",
            },
        )

        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    async def test_login_nonexistent_user(self, client: AsyncClient):
        """Test login with non-existent username"""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "username": "nonexistent",
                "password": "password123",
            },
        )

        assert response.status_code == 401

    async def test_login_inactive_user(
        self, client: AsyncClient, test_db: AsyncSession, test_user: User
    ):
        """Test login with inactive user account"""
        # Deactivate user
        test_user.status = "suspended"
        await test_db.commit()

        response = await client.post(
            "/api/v1/auth/login",
            json={
                "username": "testuser",
                "password": "testpassword123",
            },
        )

        assert response.status_code == 403
        assert "suspended" in response.json()["detail"].lower()


@pytest.mark.unit
@pytest.mark.auth
class TestTokenRefresh:
    """Tests for token refresh endpoint"""

    async def test_refresh_token_success(
        self, client: AsyncClient, test_user: User
    ):
        """Test successful token refresh"""
        # First login to get refresh token
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "username": "testuser",
                "password": "testpassword123",
            },
        )
        refresh_token = login_response.json()["refresh_token"]

        # Refresh token
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_refresh_with_invalid_token(self, client: AsyncClient):
        """Test token refresh with invalid token"""
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )

        assert response.status_code == 401

    async def test_refresh_with_access_token(
        self, client: AsyncClient, test_user_token: str
    ):
        """Test token refresh with access token instead of refresh token"""
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": test_user_token},  # Wrong token type
        )

        assert response.status_code == 401


@pytest.mark.unit
@pytest.mark.auth
class TestGetCurrentUser:
    """Tests for get current user endpoint"""

    async def test_get_me_success(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test getting current user info"""
        response = await client.get("/api/v1/auth/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert "password_hash" not in data  # Password should not be exposed

    async def test_get_me_without_auth(self, client: AsyncClient):
        """Test getting current user without authentication"""
        response = await client.get("/api/v1/auth/me")

        assert response.status_code == 403  # No Authorization header

    async def test_get_me_with_invalid_token(self, client: AsyncClient):
        """Test getting current user with invalid token"""
        response = await client.get(
            "/api/v1/auth/me", headers={"Authorization": "Bearer invalid.token"}
        )

        assert response.status_code == 401


@pytest.mark.unit
@pytest.mark.auth
class TestLogout:
    """Tests for logout endpoint"""

    async def test_logout_success(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test successful logout"""
        response = await client.post("/api/v1/auth/logout", headers=auth_headers)

        assert response.status_code == 200
        assert "logged out" in response.json()["message"].lower()

    async def test_logout_without_auth(self, client: AsyncClient):
        """Test logout without authentication"""
        response = await client.post("/api/v1/auth/logout")

        assert response.status_code == 403


@pytest.mark.unit
@pytest.mark.auth
class TestPasswordSecurity:
    """Tests for password hashing and verification"""

    def test_password_hashing(self):
        """Test password hashing produces different hashes"""
        from auth.security import get_password_hash

        password = "testpassword123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # Same password should produce different hashes (bcrypt salts)
        assert hash1 != hash2
        assert hash1 != password

    def test_password_verification(self):
        """Test password verification"""
        from auth.security import get_password_hash

        password = "testpassword123"
        hashed = get_password_hash(password)

        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False

    def test_token_decode(self, test_user_token: str, test_user: User):
        """Test JWT token decoding"""
        payload = decode_token(test_user_token)

        assert payload is not None
        assert payload["sub"] == str(test_user.user_id)
        assert payload["type"] == "access"
        assert "exp" in payload

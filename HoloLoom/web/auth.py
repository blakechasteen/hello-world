"""
Authentication Module
=====================
Secure authentication for HoloLoom web interface.

Features:
- JWT-based token authentication
- Bcrypt password hashing
- Session management
- User database (simple in-memory for demo, replace with DB in production)
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# HTTP Bearer token extractor
security = HTTPBearer()


@dataclass
class User:
    """User model."""
    username: str
    hashed_password: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self, include_password: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "disabled": self.disabled,
            "created_at": self.created_at.isoformat()
        }
        if include_password:
            data["hashed_password"] = self.hashed_password
        return data


class UserDatabase:
    """
    Simple in-memory user database.

    PRODUCTION: Replace with real database (PostgreSQL, MongoDB, etc.)
    """

    def __init__(self):
        self.users: Dict[str, User] = {}
        self._init_demo_users()

    def _init_demo_users(self):
        """Initialize demo users for testing."""
        # Demo user: admin / admin123
        self.create_user(
            username="admin",
            password="admin123",
            email="admin@hololoom.ai",
            full_name="Admin User"
        )

        # Demo user: demo / demo123
        self.create_user(
            username="demo",
            password="demo123",
            email="demo@hololoom.ai",
            full_name="Demo User"
        )

        logger.info(f"Initialized {len(self.users)} demo users")

    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> User:
        """Create new user."""
        if username in self.users:
            raise ValueError(f"User {username} already exists")

        hashed_password = pwd_context.hash(password)
        user = User(
            username=username,
            hashed_password=hashed_password,
            email=email,
            full_name=full_name
        )

        self.users[username] = user
        logger.info(f"Created user: {username}")
        return user

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.users.get(username)

    def verify_password(self, username: str, password: str) -> bool:
        """Verify user password."""
        user = self.get_user(username)
        if not user:
            return False
        return pwd_context.verify(password, user.hashed_password)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials."""
        if not self.verify_password(username, password):
            return None
        user = self.get_user(username)
        if user and user.disabled:
            return None
        return user

    def list_users(self) -> list[User]:
        """List all users (excluding passwords)."""
        return list(self.users.values())


# Global user database instance
user_db = UserDatabase()


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Payload data to encode
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and verify JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded payload or None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        return None


def get_current_user_from_token(token: str) -> Optional[User]:
    """
    Get current user from JWT token.

    Args:
        token: JWT token string

    Returns:
        User object or None
    """
    payload = decode_access_token(token)
    if not payload:
        return None

    username = payload.get("sub")
    if not username:
        return None

    user = user_db.get_user(username)
    return user


# FastAPI dependency for protected endpoints
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    FastAPI dependency: Extract and validate current user from Authorization header.

    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"message": f"Hello {user.username}"}
    """
    token = credentials.credentials
    user = get_current_user_from_token(token)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# Cookie-based authentication (alternative to Bearer token)
async def get_current_user_from_cookie(
    access_token: Optional[str] = Cookie(None)
) -> Optional[User]:
    """
    FastAPI dependency: Extract user from cookie.

    Returns None instead of raising exception (for optional auth).
    """
    if not access_token:
        return None

    return get_current_user_from_token(access_token)


# Session management
class SessionManager:
    """Manage active user sessions."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, username: str, token: str) -> str:
        """Create new session."""
        session_id = f"{username}_{datetime.now(timezone.utc).timestamp()}"

        self.sessions[session_id] = {
            "username": username,
            "token": token,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc)
        }

        logger.info(f"Created session for {username}: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def update_activity(self, session_id: str):
        """Update last activity timestamp."""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = datetime.now(timezone.utc)

    def delete_session(self, session_id: str):
        """Delete session."""
        if session_id in self.sessions:
            username = self.sessions[session_id]["username"]
            del self.sessions[session_id]
            logger.info(f"Deleted session for {username}: {session_id}")

    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Remove expired sessions."""
        now = datetime.now(timezone.utc)
        expired = []

        for session_id, session in self.sessions.items():
            age = now - session["last_activity"]
            if age > timedelta(hours=max_age_hours):
                expired.append(session_id)

        for session_id in expired:
            self.delete_session(session_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")


# Global session manager
session_manager = SessionManager()


# ============================================================================
# Helper Functions
# ============================================================================

def verify_token_for_websocket(token: str) -> Optional[str]:
    """
    Verify JWT token for WebSocket connection.

    Args:
        token: JWT token from query parameter or header

    Returns:
        Username if valid, None otherwise
    """
    user = get_current_user_from_token(token)
    if user:
        return user.username
    return None


def get_demo_credentials() -> Dict[str, str]:
    """Get demo user credentials for development."""
    return {
        "admin": "admin123",
        "demo": "demo123"
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("HoloLoom Authentication Module")
    print("="*80 + "\n")

    # Create user
    print("1. Creating user...")
    user = user_db.create_user(
        username="testuser",
        password="testpass123",
        email="test@example.com",
        full_name="Test User"
    )
    print(f"   ✓ Created: {user.username}")

    # Authenticate
    print("\n2. Authenticating...")
    auth_user = user_db.authenticate_user("testuser", "testpass123")
    if auth_user:
        print(f"   ✓ Authenticated: {auth_user.username}")
    else:
        print("   ✗ Authentication failed")

    # Create token
    print("\n3. Creating JWT token...")
    token = create_access_token({"sub": user.username})
    print(f"   ✓ Token: {token[:50]}...")

    # Verify token
    print("\n4. Verifying token...")
    verified_user = get_current_user_from_token(token)
    if verified_user:
        print(f"   ✓ Verified: {verified_user.username}")
    else:
        print("   ✗ Verification failed")

    # List users
    print("\n5. Demo users available:")
    for username, password in get_demo_credentials().items():
        print(f"   • {username} / {password}")

    print("\n✓ Authentication module ready!")

"""
Authentication Middleware
"""

from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import secrets

from ..config import get_settings
from ..models.auth import TokenData, User

settings = get_settings()

# Security schemes
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# In-memory storage (use Redis/database in production)
API_KEYS = {}
USERS = {}


def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"pk_{secrets.token_urlsafe(32)}"


def verify_api_key(api_key: str) -> bool:
    """Verify API key"""
    return api_key in API_KEYS


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)

    return encoded_jwt


def decode_access_token(token: str) -> TokenData:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        scopes: list = payload.get("scopes", [])

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

        return TokenData(username=username, scopes=scopes)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def api_key_auth(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """API Key authentication dependency"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required",
            headers={settings.api_key_header: "Required"},
        )

    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

    return True


async def jwt_auth(credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)) -> TokenData:
    """JWT authentication dependency"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_access_token(credentials.credentials)


async def get_current_user(token_data: TokenData = Depends(jwt_auth)) -> User:
    """Get current authenticated user"""
    user = USERS.get(token_data.username)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    return user


# Helper function to add default API key for development
def init_default_auth():
    """Initialize default API key and user for development"""
    default_key = "pk_dev_key_12345"
    API_KEYS[default_key] = {
        "key_id": "default",
        "name": "Development Key",
        "scopes": ["*"],
        "created_at": datetime.utcnow()
    }

    USERS["admin"] = User(
        username="admin",
        email="admin@example.com",
        full_name="Admin User",
        scopes=["*"]
    )

    return default_key

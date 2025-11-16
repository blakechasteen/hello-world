"""
HoloLoom Authentication Middleware
===================================

FastAPI dependency injection for authentication.

Features:
- JWT token verification
- API key verification
- Role-based access control
- Rate limiting on auth endpoints

Created: 2025-11-16
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from HoloLoom.auth.authentication import verify_token, TokenData
from HoloLoom.auth.users import User, UserRole, get_user
from HoloLoom.auth.api_keys import verify_api_key

logger = logging.getLogger(__name__)

# Security scheme for Bearer tokens
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> User:
    """
    Get current authenticated user from JWT or API key.

    Args:
        credentials: HTTP Authorization credentials

    Returns:
        Authenticated user

    Raises:
        HTTPException: 401 if invalid/expired token
    """
    token = credentials.credentials

    # Try JWT first
    try:
        token_data = verify_token(token, token_type="access")

        if token_data:
            user = get_user(token_data.username)

            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return user
    except Exception as e:
        logger.debug(f"JWT verification failed: {e}")

    # Try API key
    api_key = verify_api_key(token)

    if api_key:
        user = get_user(api_key.username)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        logger.info(f"Authenticated via API key: {api_key.key_id}")
        return user

    # Both failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.

    Args:
        current_user: Current user from token

    Returns:
        Active user

    Raises:
        HTTPException: 403 if user inactive
    """
    if not current_user.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return current_user


def require_role(required_role: UserRole):
    """
    Create dependency that requires specific role.

    Args:
        required_role: Required user role

    Returns:
        Dependency function

    Example:
        @app.get("/admin")
        async def admin_only(user: User = Depends(require_role(UserRole.ADMIN))):
            return {"message": "Admin access granted"}
    """
    async def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}"
            )

        return current_user

    return role_checker


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security, auto_error=False)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise None.

    Useful for endpoints that are public but enhance with user data if authenticated.

    Args:
        credentials: Optional HTTP Authorization credentials

    Returns:
        User if authenticated, None otherwise
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# Example usage in FastAPI endpoint:
"""
from fastapi import FastAPI, Depends
from HoloLoom.auth import get_current_user, require_role, UserRole, User

app = FastAPI()

# Protected endpoint (any authenticated user)
@app.get("/protected")
async def protected_route(user: User = Depends(get_current_user)):
    return {"message": f"Hello {user.username}!"}

# Admin-only endpoint
@app.get("/admin")
async def admin_only(user: User = Depends(require_role(UserRole.ADMIN))):
    return {"message": "Admin access granted"}

# Public endpoint with optional user data
@app.get("/public")
async def public_route(user: Optional[User] = Depends(get_optional_user)):
    if user:
        return {"message": f"Hello {user.username}!", "authenticated": True}
    else:
        return {"message": "Hello guest!", "authenticated": False}
"""

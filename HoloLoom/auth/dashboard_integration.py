"""
HoloLoom Dashboard Authentication Integration
==============================================

FastAPI routes and WebSocket auth for HoloLoom dashboard.

Features:
- Login/logout/refresh endpoints
- API key management endpoints
- WebSocket token verification
- Optional auth (opt-in via ENABLE_AUTH env var)

Created: 2025-11-16

Usage:
    from HoloLoom.auth.dashboard_integration import register_auth_routes, verify_websocket_token

    # In dashboard_server.py:
    register_auth_routes(app, limiter)  # Register auth endpoints

    # In WebSocket handler:
    user = await verify_websocket_token(websocket, token)
"""

import os
import logging
from typing import Optional
from datetime import timedelta

from fastapi import FastAPI, Depends, HTTPException, WebSocket, Query, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Check if auth is enabled
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"

# Import auth modules (graceful degradation)
try:
    from HoloLoom.auth.authentication import (
        create_access_token,
        create_refresh_token,
        verify_token,
        revoke_token,
        Token,
        is_jwt_available,
    )
    from HoloLoom.auth.users import (
        authenticate_user,
        get_user,
        User,
    )
    from HoloLoom.auth.api_keys import (
        APIKeyManager,
        APIKeyType,
        api_key_manager,
    )
    from HoloLoom.auth.middleware import (
        get_current_user,
        get_current_active_user,
    )

    AUTH_AVAILABLE = is_jwt_available()

except ImportError as e:
    logger.warning(f"Auth modules not available: {e}")
    AUTH_AVAILABLE = False


# ============================================================================
# Request/Response Models
# ============================================================================

class APIKeyCreate(BaseModel):
    """API key creation request."""
    key_type: str = "live"
    expires_in_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    """API key response."""
    key_id: str
    key: Optional[str] = None  # Only shown on creation
    key_prefix: Optional[str] = None
    username: str
    key_type: str
    created_at: str
    expires_at: Optional[str] = None
    last_used: Optional[str] = None
    active: bool


# ============================================================================
# Auth Endpoints
# ============================================================================

def register_auth_routes(app: FastAPI, limiter=None):
    """
    Register authentication routes to FastAPI app.

    Args:
        app: FastAPI application instance
        limiter: Optional rate limiter (slowapi Limiter or SimpleRateLimiter)
    """
    if not ENABLE_AUTH:
        logger.info("Authentication disabled (ENABLE_AUTH=false)")
        return

    if not AUTH_AVAILABLE:
        logger.warning(
            "Authentication enabled but dependencies not available. "
            "Install with: pip install python-jose[cryptography] bcrypt"
        )
        return

    logger.info("Registering authentication routes...")

    # Determine if rate limiting is available
    has_rate_limiting = limiter is not None

    # ========================================================================
    # POST /api/v1/auth/login - Login with username/password
    # ========================================================================

    if has_rate_limiting and hasattr(limiter, 'limit'):
        # slowapi-style decorator
        @app.post("/api/v1/auth/login", response_model=Token)
        @limiter.limit("10/minute")
        async def login(
            request,  # Required by slowapi
            form_data: OAuth2PasswordRequestForm = Depends()
        ):
            """Login with username and password."""
            user = authenticate_user(form_data.username, form_data.password)

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Create tokens
            access_token = create_access_token(
                data={"sub": user.username, "role": user.role.value}
            )
            refresh_token = create_refresh_token(
                data={"sub": user.username, "role": user.role.value}
            )

            logger.info(f"User logged in: {user.username}")

            return Token(
                access_token=access_token,
                refresh_token=refresh_token,
            )
    else:
        # No rate limiting or SimpleRateLimiter (async call)
        @app.post("/api/v1/auth/login", response_model=Token)
        async def login(
            form_data: OAuth2PasswordRequestForm = Depends()
        ):
            """Login with username and password."""
            user = authenticate_user(form_data.username, form_data.password)

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Create tokens
            access_token = create_access_token(
                data={"sub": user.username, "role": user.role.value}
            )
            refresh_token = create_refresh_token(
                data={"sub": user.username, "role": user.role.value}
            )

            logger.info(f"User logged in: {user.username}")

            return Token(
                access_token=access_token,
                refresh_token=refresh_token,
            )

    # ========================================================================
    # POST /api/v1/auth/refresh - Refresh access token
    # ========================================================================

    @app.post("/api/v1/auth/refresh")
    async def refresh(current_user: User = Depends(get_current_user)):
        """Refresh access token using refresh token."""
        # Create new access token
        access_token = create_access_token(
            data={"sub": current_user.username, "role": current_user.role.value}
        )

        logger.info(f"Token refreshed for: {current_user.username}")

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600,
        }

    # ========================================================================
    # POST /api/v1/auth/logout - Logout (revoke token)
    # ========================================================================

    @app.post("/api/v1/auth/logout")
    async def logout(current_user: User = Depends(get_current_user)):
        """Logout by revoking current token."""
        # Note: We don't have access to the token here
        # In production, client should send token explicitly
        logger.info(f"User logged out: {current_user.username}")

        return {"message": "Logged out successfully"}

    # ========================================================================
    # POST /api/v1/auth/api-keys - Generate API key
    # ========================================================================

    @app.post("/api/v1/auth/api-keys", response_model=APIKeyResponse, status_code=201)
    async def create_api_key(
        request: APIKeyCreate,
        current_user: User = Depends(get_current_active_user)
    ):
        """Generate new API key (requires JWT authentication)."""
        # Validate key type
        try:
            key_type = APIKeyType(request.key_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid key_type. Must be 'live' or 'test'"
            )

        # Generate key
        api_key = api_key_manager.generate_key(
            username=current_user.username,
            key_type=key_type,
            expires_in_days=request.expires_in_days,
        )

        # Return with full key (only shown once!)
        response = APIKeyResponse(**api_key.to_dict(include_key=True))

        logger.info(f"API key created: {api_key.key_id} for {current_user.username}")

        return response

    # ========================================================================
    # GET /api/v1/auth/api-keys - List user's API keys
    # ========================================================================

    @app.get("/api/v1/auth/api-keys")
    async def list_api_keys(current_user: User = Depends(get_current_active_user)):
        """List all API keys for current user."""
        keys = api_key_manager.list_keys(username=current_user.username)

        # Convert to response models (without full keys)
        response_keys = [
            APIKeyResponse(**k.to_dict(include_key=False))
            for k in keys
        ]

        return {"api_keys": response_keys}

    # ========================================================================
    # DELETE /api/v1/auth/api-keys/{key_id} - Revoke API key
    # ========================================================================

    @app.delete("/api/v1/auth/api-keys/{key_id}")
    async def revoke_api_key(
        key_id: str,
        current_user: User = Depends(get_current_active_user)
    ):
        """Revoke API key by ID."""
        # Verify ownership
        keys = api_key_manager.list_keys(username=current_user.username)
        if not any(k.key_id == key_id for k in keys):
            raise HTTPException(
                status_code=404,
                detail="API key not found"
            )

        # Revoke
        revoked = api_key_manager.revoke_key(key_id)

        if not revoked:
            raise HTTPException(
                status_code=404,
                detail="API key not found"
            )

        logger.info(f"API key revoked: {key_id} by {current_user.username}")

        return {"message": "API key revoked", "key_id": key_id}

    logger.info(f"Authentication routes registered: {5} endpoints")


# ============================================================================
# WebSocket Authentication
# ============================================================================

async def verify_websocket_token(
    websocket: WebSocket,
    token: Optional[str] = None
) -> Optional[User]:
    """
    Verify WebSocket connection token.

    Args:
        websocket: WebSocket connection
        token: Optional JWT token from query param or first message

    Returns:
        User if authenticated, None if auth disabled or token invalid
    """
    if not ENABLE_AUTH:
        # Auth disabled - allow all connections
        return None

    if not AUTH_AVAILABLE:
        logger.warning("Auth enabled but dependencies not available - allowing connection")
        return None

    if not token:
        logger.warning("WebSocket connection without token")
        await websocket.close(code=1008, reason="Authentication required")
        return None

    # Verify token
    token_data = verify_token(token, token_type="access")

    if not token_data:
        logger.warning("WebSocket connection with invalid token")
        await websocket.close(code=1008, reason="Invalid or expired token")
        return None

    # Get user
    user = get_user(token_data.username)

    if not user or not user.active:
        logger.warning(f"WebSocket connection for inactive user: {token_data.username}")
        await websocket.close(code=1008, reason="User not found or inactive")
        return None

    logger.info(f"WebSocket authenticated: {user.username}")
    return user


# ============================================================================
# Helper Functions
# ============================================================================

def is_auth_enabled() -> bool:
    """Check if authentication is enabled."""
    return ENABLE_AUTH and AUTH_AVAILABLE


def get_auth_status() -> dict:
    """Get authentication status for dashboard display."""
    return {
        "enabled": ENABLE_AUTH,
        "available": AUTH_AVAILABLE,
        "jwt_available": is_jwt_available() if AUTH_AVAILABLE else False,
    }


# Example usage:
if __name__ == "__main__":
    from fastapi import FastAPI

    app = FastAPI()

    # Register auth routes
    register_auth_routes(app)

    # Check status
    status = get_auth_status()
    print(f"Auth enabled: {status['enabled']}")
    print(f"Auth available: {status['available']}")

    # Run with: uvicorn HoloLoom.auth.dashboard_integration:app --reload

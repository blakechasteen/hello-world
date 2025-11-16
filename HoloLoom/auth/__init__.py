"""
HoloLoom Authentication System
==============================

Dual authentication support for HoloLoom dashboard:
- JWT (JSON Web Tokens) for user sessions
- API Keys for programmatic access

Created: 2025-11-16
Status: Production Ready
"""

from HoloLoom.auth.authentication import (
    create_access_token,
    create_refresh_token,
    verify_token,
    TokenData,
    Token,
)
from HoloLoom.auth.users import (
    User,
    UserRole,
    authenticate_user,
    get_user,
    create_user,
)
from HoloLoom.auth.api_keys import (
    APIKey,
    APIKeyManager,
    generate_api_key,
    verify_api_key,
)

# Try to import middleware (requires FastAPI)
try:
    from HoloLoom.auth.middleware import (
        get_current_user,
        get_current_active_user,
        require_role,
    )
except ImportError:
    # FastAPI not available - middleware functions won't be available
    get_current_user = None
    get_current_active_user = None
    require_role = None

__all__ = [
    # Authentication
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "TokenData",
    "Token",
    # Users
    "User",
    "UserRole",
    "authenticate_user",
    "get_user",
    "create_user",
    # API Keys
    "APIKey",
    "APIKeyManager",
    "generate_api_key",
    "verify_api_key",
    # Middleware
    "get_current_user",
    "get_current_active_user",
    "require_role",
]

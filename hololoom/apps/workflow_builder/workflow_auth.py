"""Authentication for Workflow Builder.

Single-user mode by default, multi-user enabled via WORKFLOW_MULTI_USER=true.
Implements HMAC-SHA256 request signing with nonce replay protection.

Created: 2025-12-23
Part of HoloLoom Workflow Builder production integration.
"""

import os
import hmac
import hashlib
import time
import uuid
from typing import Optional, Dict, Set, Callable
from dataclasses import dataclass, field
from functools import wraps

# Configuration from environment
MULTI_USER_ENABLED = os.environ.get("WORKFLOW_MULTI_USER", "false").lower() == "true"
API_SECRET = os.environ.get("WORKFLOW_API_SECRET", "dev-secret-change-in-production")
NONCE_EXPIRY = 300  # 5 minutes


@dataclass
class AuthContext:
    """Authentication context for a request.

    Attributes:
        user_id: Unique identifier for the user
        is_authenticated: Whether the user is authenticated
        permissions: Set of permission strings (read, write, delete, execute)
    """
    user_id: str
    is_authenticated: bool
    permissions: Set[str] = field(default_factory=set)

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def has_all_permissions(self, permissions: Set[str]) -> bool:
        """Check if user has all specified permissions."""
        return permissions.issubset(self.permissions)


class NonceTracker:
    """Track used nonces to prevent replay attacks.

    Maintains a sliding window of used nonces with automatic cleanup
    of expired entries.
    """

    def __init__(self, expiry_seconds: int = NONCE_EXPIRY):
        """Initialize nonce tracker.

        Args:
            expiry_seconds: How long to remember used nonces
        """
        self._used: Dict[str, float] = {}
        self._expiry = expiry_seconds

    def check_and_mark(self, nonce: str) -> bool:
        """Check if nonce is valid (not used) and mark as used.

        Args:
            nonce: Nonce string to check

        Returns:
            True if nonce is valid and was marked, False if already used
        """
        now = time.time()

        # Clean expired nonces
        self._used = {
            k: v for k, v in self._used.items()
            if now - v < self._expiry
        }

        if nonce in self._used:
            return False

        self._used[nonce] = now
        return True

    def clear(self):
        """Clear all tracked nonces."""
        self._used.clear()


# Global nonce tracker instance
_nonce_tracker = NonceTracker()


def compute_signature(message: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature.

    Args:
        message: Message to sign
        secret: Secret key for signing

    Returns:
        Hex-encoded signature string
    """
    return hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def verify_signature(
    api_key: str,
    timestamp: str,
    nonce: str,
    signature: str,
    secret: str = API_SECRET
) -> bool:
    """Verify request signature.

    Args:
        api_key: API key from request header
        timestamp: Unix timestamp from request header
        nonce: Unique nonce from request header
        signature: Signature to verify from request header
        secret: Secret key for verification

    Returns:
        True if signature is valid
    """
    message = f"{api_key}:{timestamp}:{nonce}"
    expected_sig = compute_signature(message, secret)
    return hmac.compare_digest(signature, expected_sig)


def create_signed_headers(
    api_key: str,
    secret: str = API_SECRET
) -> Dict[str, str]:
    """Create signed authentication headers for a request.

    Useful for clients making authenticated requests.

    Args:
        api_key: API key to use
        secret: Secret key for signing

    Returns:
        Dict of headers to include in request
    """
    timestamp = str(time.time())
    nonce = str(uuid.uuid4())
    message = f"{api_key}:{timestamp}:{nonce}"
    signature = compute_signature(message, secret)

    return {
        "X-API-Key": api_key,
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature
    }


class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class PermissionError(AuthenticationError):
    """Exception for permission denied errors."""
    def __init__(self, permission: str):
        super().__init__(
            f"Permission '{permission}' required",
            status_code=403
        )


def get_auth_context_sync(
    api_key: Optional[str] = None,
    timestamp: Optional[str] = None,
    nonce: Optional[str] = None,
    signature: Optional[str] = None
) -> AuthContext:
    """Get authentication context from request headers (synchronous).

    Args:
        api_key: X-API-Key header value
        timestamp: X-Timestamp header value
        nonce: X-Nonce header value
        signature: X-Signature header value

    Returns:
        AuthContext with user info and permissions

    Raises:
        AuthenticationError: If authentication fails
    """
    # Single-user mode: always authenticated as default user
    if not MULTI_USER_ENABLED:
        return AuthContext(
            user_id="default",
            is_authenticated=True,
            permissions={"read", "write", "delete", "execute"}
        )

    # Multi-user mode: require API key with signature
    if not api_key:
        raise AuthenticationError("API key required")

    if not all([timestamp, nonce, signature]):
        raise AuthenticationError("Missing auth headers (X-Timestamp, X-Nonce, X-Signature)")

    # Check timestamp freshness (5 minute window)
    try:
        ts = float(timestamp)
        if abs(time.time() - ts) > NONCE_EXPIRY:
            raise AuthenticationError("Request expired")
    except ValueError:
        raise AuthenticationError("Invalid timestamp format")

    # Check nonce hasn't been used
    if not _nonce_tracker.check_and_mark(nonce):
        raise AuthenticationError("Nonce already used (replay attack prevention)")

    # Verify signature
    if not verify_signature(api_key, timestamp, nonce, signature):
        raise AuthenticationError("Invalid signature")

    # In production, look up user from api_key in database
    # For now, use prefix as user ID
    return AuthContext(
        user_id=api_key[:8],
        is_authenticated=True,
        permissions={"read", "write", "execute"}  # Default permissions
    )


# ==================== FastAPI Integration ====================

try:
    from fastapi import Request, HTTPException, Depends
    from fastapi.security import APIKeyHeader

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def get_auth_context(
        request: Request,
        api_key: Optional[str] = Depends(api_key_header)
    ) -> AuthContext:
        """FastAPI dependency for authentication.

        Usage:
            @app.get("/api/workflows")
            async def list_workflows(auth: AuthContext = Depends(get_auth_context)):
                return workflows_for_user(auth.user_id)
        """
        try:
            return get_auth_context_sync(
                api_key=api_key,
                timestamp=request.headers.get("X-Timestamp"),
                nonce=request.headers.get("X-Nonce"),
                signature=request.headers.get("X-Signature")
            )
        except AuthenticationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)

    def require_permission(permission: str):
        """FastAPI dependency to require specific permission.

        Usage:
            @app.post("/api/workflow/save")
            async def save_workflow(
                workflow: Workflow,
                auth: AuthContext = Depends(require_permission("write"))
            ):
                # Only called if user has "write" permission
                ...
        """
        async def check_permission(auth: AuthContext = Depends(get_auth_context)):
            if permission not in auth.permissions:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission}' required"
                )
            return auth
        return check_permission

    def require_permissions(*permissions: str):
        """FastAPI dependency to require multiple permissions.

        Usage:
            @app.delete("/api/workflow/{id}")
            async def delete_workflow(
                id: str,
                auth: AuthContext = Depends(require_permissions("write", "delete"))
            ):
                ...
        """
        async def check_permissions(auth: AuthContext = Depends(get_auth_context)):
            missing = set(permissions) - auth.permissions
            if missing:
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing permissions: {', '.join(missing)}"
                )
            return auth
        return check_permissions

    FASTAPI_AVAILABLE = True

except ImportError:
    # FastAPI not available - provide stubs
    FASTAPI_AVAILABLE = False

    async def get_auth_context(*args, **kwargs) -> AuthContext:
        """Stub when FastAPI not available."""
        return get_auth_context_sync()

    def require_permission(permission: str):
        """Stub when FastAPI not available."""
        async def check(*args, **kwargs):
            return get_auth_context_sync()
        return check

    def require_permissions(*permissions: str):
        """Stub when FastAPI not available."""
        async def check(*args, **kwargs):
            return get_auth_context_sync()
        return check


# ==================== User Management (for multi-user mode) ====================

@dataclass
class User:
    """User record for multi-user mode."""
    id: str
    api_key: str
    name: str
    email: Optional[str]
    permissions: Set[str]
    created_at: float
    is_active: bool = True


class UserStore:
    """In-memory user store for development/testing.

    In production, replace with database-backed implementation.
    """

    def __init__(self):
        self._users: Dict[str, User] = {}
        self._api_key_index: Dict[str, str] = {}  # api_key -> user_id

    def create_user(
        self,
        name: str,
        email: Optional[str] = None,
        permissions: Optional[Set[str]] = None
    ) -> User:
        """Create a new user with generated API key."""
        user_id = str(uuid.uuid4())
        api_key = f"wf_{uuid.uuid4().hex}"

        user = User(
            id=user_id,
            api_key=api_key,
            name=name,
            email=email,
            permissions=permissions or {"read", "write", "execute"},
            created_at=time.time()
        )

        self._users[user_id] = user
        self._api_key_index[api_key] = user_id
        return user

    def get_by_api_key(self, api_key: str) -> Optional[User]:
        """Look up user by API key."""
        user_id = self._api_key_index.get(api_key)
        if user_id:
            return self._users.get(user_id)
        return None

    def get_by_id(self, user_id: str) -> Optional[User]:
        """Look up user by ID."""
        return self._users.get(user_id)

    def update_permissions(self, user_id: str, permissions: Set[str]) -> bool:
        """Update user permissions."""
        user = self._users.get(user_id)
        if user:
            user.permissions = permissions
            return True
        return False

    def deactivate(self, user_id: str) -> bool:
        """Deactivate a user (soft delete)."""
        user = self._users.get(user_id)
        if user:
            user.is_active = False
            return True
        return False


# Global user store instance (for multi-user mode)
_user_store = UserStore()


def get_user_store() -> UserStore:
    """Get the global user store instance."""
    return _user_store


# ==================== Utility Functions ====================

def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage.

    Args:
        api_key: Plain text API key

    Returns:
        SHA256 hash of the key
    """
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()


def generate_api_key(prefix: str = "wf") -> str:
    """Generate a new API key.

    Args:
        prefix: Prefix for the key (default: "wf" for workflow)

    Returns:
        Generated API key string
    """
    return f"{prefix}_{uuid.uuid4().hex}"


def is_multi_user_enabled() -> bool:
    """Check if multi-user mode is enabled."""
    return MULTI_USER_ENABLED


def get_current_mode() -> str:
    """Get current authentication mode as string."""
    return "multi-user" if MULTI_USER_ENABLED else "single-user"

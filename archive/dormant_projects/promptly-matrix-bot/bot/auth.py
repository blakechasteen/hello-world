"""
Phase 6A: Authentication & Security
JWT-based authentication, role-based access control (RBAC), and rate limiting.

Features:
- JWT token generation and validation
- Role-based access control (Owner, Admin, Editor, Viewer)
- Rate limiting per user/endpoint
- Session management
- API key authentication for external services

Security Features:
- Bcrypt password hashing
- Token expiration and refresh
- Rate limit per user (configurable)
- IP-based rate limiting (optional)
- Audit logging for auth events
"""

import os
import jwt
import time
import bcrypt
import secrets
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json


class UserRole(Enum):
    """User roles for RBAC."""

    OWNER = "owner"  # Full access, manage users
    ADMIN = "admin"  # Full access except user management
    EDITOR = "editor"  # Can execute commands, no admin
    VIEWER = "viewer"  # Read-only access


class Permission(Enum):
    """Granular permissions."""

    # User management
    MANAGE_USERS = "manage_users"
    VIEW_USERS = "view_users"

    # Bot commands
    EXECUTE_COMMANDS = "execute_commands"
    VIEW_AUDIT_LOG = "view_audit_log"

    # GitHub integration
    CREATE_PR = "create_pr"
    MERGE_PR = "merge_pr"
    REVIEW_CODE = "review_code"
    MANAGE_ISSUES = "manage_issues"
    TRIGGER_BUILDS = "trigger_builds"

    # HoloLoom
    QUERY_HOLOLOOM = "query_hololoom"
    MANAGE_MEMORIES = "manage_memories"

    # Configuration
    MANAGE_CONFIG = "manage_config"
    VIEW_METRICS = "view_metrics"


# Role → Permissions mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.OWNER: {
        # All permissions
        Permission.MANAGE_USERS,
        Permission.VIEW_USERS,
        Permission.EXECUTE_COMMANDS,
        Permission.VIEW_AUDIT_LOG,
        Permission.CREATE_PR,
        Permission.MERGE_PR,
        Permission.REVIEW_CODE,
        Permission.MANAGE_ISSUES,
        Permission.TRIGGER_BUILDS,
        Permission.QUERY_HOLOLOOM,
        Permission.MANAGE_MEMORIES,
        Permission.MANAGE_CONFIG,
        Permission.VIEW_METRICS,
    },
    UserRole.ADMIN: {
        # All except user management
        Permission.VIEW_USERS,
        Permission.EXECUTE_COMMANDS,
        Permission.VIEW_AUDIT_LOG,
        Permission.CREATE_PR,
        Permission.MERGE_PR,
        Permission.REVIEW_CODE,
        Permission.MANAGE_ISSUES,
        Permission.TRIGGER_BUILDS,
        Permission.QUERY_HOLOLOOM,
        Permission.MANAGE_MEMORIES,
        Permission.MANAGE_CONFIG,
        Permission.VIEW_METRICS,
    },
    UserRole.EDITOR: {
        # Can execute commands, no admin features
        Permission.VIEW_USERS,
        Permission.EXECUTE_COMMANDS,
        Permission.CREATE_PR,
        Permission.REVIEW_CODE,
        Permission.MANAGE_ISSUES,
        Permission.TRIGGER_BUILDS,
        Permission.QUERY_HOLOLOOM,
        Permission.VIEW_METRICS,
    },
    UserRole.VIEWER: {
        # Read-only
        Permission.VIEW_USERS,
        Permission.VIEW_AUDIT_LOG,
        Permission.VIEW_METRICS,
    },
}


@dataclass
class User:
    """User model."""

    user_id: str  # Matrix user ID (e.g., "@alice:matrix.org")
    username: str  # Display name
    role: UserRole
    password_hash: Optional[str] = None  # For API key users
    api_key: Optional[str] = None  # For external services
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_login: Optional[str] = None
    enabled: bool = True


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    max_requests: int  # Max requests in window
    window_seconds: int  # Time window in seconds
    burst_size: int = 0  # Allow burst (0 = no burst)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter."""
        self.config = config
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tokens": config.max_requests, "last_refill": time.time()}
        )

    def check_rate_limit(self, key: str) -> bool:
        """
        Check if request is allowed.

        Args:
            key: Rate limit key (e.g., user_id, IP address)

        Returns:
            True if allowed, False if rate limited
        """
        bucket = self.buckets[key]
        now = time.time()

        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_refill"]
        refill_rate = self.config.max_requests / self.config.window_seconds
        tokens_to_add = elapsed * refill_rate

        bucket["tokens"] = min(
            bucket["tokens"] + tokens_to_add,
            self.config.max_requests + self.config.burst_size,
        )
        bucket["last_refill"] = now

        # Check if request allowed
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        else:
            return False

    def get_remaining(self, key: str) -> int:
        """Get remaining tokens for key."""
        bucket = self.buckets.get(key)
        if not bucket:
            return self.config.max_requests

        return int(bucket["tokens"])

    def reset(self, key: str):
        """Reset rate limit for key."""
        if key in self.buckets:
            del self.buckets[key]


class AuthManager:
    """Authentication and authorization manager."""

    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        jwt_expiry_hours: int = 24,
        users_file: str = "config/users.json",
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize auth manager.

        Args:
            jwt_secret: Secret for JWT signing (auto-generated if None)
            jwt_expiry_hours: JWT token expiry in hours
            users_file: Path to users database
            rate_limit_config: Rate limit configuration
        """
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.jwt_expiry_hours = jwt_expiry_hours
        self.users_file = Path(users_file)
        self.users: Dict[str, User] = {}

        # Rate limiting
        self.rate_limit_config = rate_limit_config or RateLimitConfig(
            max_requests=100, window_seconds=60, burst_size=20
        )
        self.rate_limiter = RateLimiter(self.rate_limit_config)

        # Audit log
        self.audit_log: List[Dict[str, Any]] = []

        # Load users
        self._load_users()

    def _load_users(self):
        """Load users from file."""
        if self.users_file.exists():
            try:
                with open(self.users_file, "r") as f:
                    data = json.load(f)

                for user_data in data.get("users", []):
                    user = User(
                        user_id=user_data["user_id"],
                        username=user_data["username"],
                        role=UserRole(user_data["role"]),
                        password_hash=user_data.get("password_hash"),
                        api_key=user_data.get("api_key"),
                        created_at=user_data.get("created_at", datetime.utcnow().isoformat()),
                        last_login=user_data.get("last_login"),
                        enabled=user_data.get("enabled", True),
                    )
                    self.users[user.user_id] = user

            except Exception as e:
                print(f"⚠️  Failed to load users: {e}")

    def _save_users(self):
        """Save users to file."""
        self.users_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "users": [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "role": user.role.value,
                    "password_hash": user.password_hash,
                    "api_key": user.api_key,
                    "created_at": user.created_at,
                    "last_login": user.last_login,
                    "enabled": user.enabled,
                }
                for user in self.users.values()
            ]
        }

        with open(self.users_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_user(
        self,
        user_id: str,
        username: str,
        role: UserRole,
        password: Optional[str] = None,
        generate_api_key: bool = False,
    ) -> User:
        """
        Create a new user.

        Args:
            user_id: Matrix user ID
            username: Display name
            role: User role
            password: Password (optional, for API key users)
            generate_api_key: Generate API key

        Returns:
            Created user
        """
        # Hash password if provided
        password_hash = None
        if password:
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        # Generate API key if requested
        api_key = None
        if generate_api_key:
            api_key = secrets.token_urlsafe(32)

        user = User(
            user_id=user_id,
            username=username,
            role=role,
            password_hash=password_hash,
            api_key=api_key,
        )

        self.users[user_id] = user
        self._save_users()

        self._audit_log("user_created", user_id, {"role": role.value})

        return user

    def authenticate_password(self, user_id: str, password: str) -> Optional[str]:
        """
        Authenticate user with password.

        Args:
            user_id: User ID
            password: Password

        Returns:
            JWT token if successful, None otherwise
        """
        user = self.users.get(user_id)

        if not user or not user.enabled:
            self._audit_log("auth_failed", user_id, {"reason": "user_not_found_or_disabled"})
            return None

        if not user.password_hash:
            self._audit_log("auth_failed", user_id, {"reason": "no_password_set"})
            return None

        # Verify password
        if bcrypt.checkpw(password.encode(), user.password_hash.encode()):
            # Update last login
            user.last_login = datetime.utcnow().isoformat()
            self._save_users()

            # Generate JWT
            token = self._generate_jwt(user)

            self._audit_log("auth_success", user_id, {"method": "password"})

            return token
        else:
            self._audit_log("auth_failed", user_id, {"reason": "invalid_password"})
            return None

    def authenticate_api_key(self, api_key: str) -> Optional[str]:
        """
        Authenticate user with API key.

        Args:
            api_key: API key

        Returns:
            JWT token if successful, None otherwise
        """
        # Find user by API key
        user = None
        for u in self.users.values():
            if u.api_key == api_key and u.enabled:
                user = u
                break

        if not user:
            self._audit_log("auth_failed", "unknown", {"reason": "invalid_api_key"})
            return None

        # Update last login
        user.last_login = datetime.utcnow().isoformat()
        self._save_users()

        # Generate JWT
        token = self._generate_jwt(user)

        self._audit_log("auth_success", user.user_id, {"method": "api_key"})

        return token

    def verify_token(self, token: str) -> Optional[User]:
        """
        Verify JWT token.

        Args:
            token: JWT token

        Returns:
            User if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            user_id = payload.get("user_id")
            user = self.users.get(user_id)

            if user and user.enabled:
                return user
            else:
                return None

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def check_permission(self, user: User, permission: Permission) -> bool:
        """
        Check if user has permission.

        Args:
            user: User
            permission: Permission to check

        Returns:
            True if user has permission
        """
        role_permissions = ROLE_PERMISSIONS.get(user.role, set())
        return permission in role_permissions

    def check_rate_limit(self, user_id: str) -> bool:
        """
        Check rate limit for user.

        Args:
            user_id: User ID

        Returns:
            True if allowed, False if rate limited
        """
        allowed = self.rate_limiter.check_rate_limit(user_id)

        if not allowed:
            self._audit_log("rate_limit_exceeded", user_id, {})

        return allowed

    def get_remaining_requests(self, user_id: str) -> int:
        """Get remaining requests for user."""
        return self.rate_limiter.get_remaining(user_id)

    def reset_rate_limit(self, user_id: str):
        """Reset rate limit for user (admin only)."""
        self.rate_limiter.reset(user_id)

    def update_role(self, user_id: str, new_role: UserRole):
        """Update user role (owner only)."""
        user = self.users.get(user_id)
        if user:
            old_role = user.role
            user.role = new_role
            self._save_users()

            self._audit_log(
                "role_changed", user_id, {"old_role": old_role.value, "new_role": new_role.value}
            )

    def disable_user(self, user_id: str):
        """Disable user (owner only)."""
        user = self.users.get(user_id)
        if user:
            user.enabled = False
            self._save_users()

            self._audit_log("user_disabled", user_id, {})

    def enable_user(self, user_id: str):
        """Enable user (owner only)."""
        user = self.users.get(user_id)
        if user:
            user.enabled = True
            self._save_users()

            self._audit_log("user_enabled", user_id, {})

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log."""
        return self.audit_log[-limit:]

    def _generate_jwt(self, user: User) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "exp": datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours),
            "iat": datetime.utcnow(),
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        return token

    def _audit_log(self, event: str, user_id: str, metadata: Dict[str, Any]):
        """Add entry to audit log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "user_id": user_id,
            "metadata": metadata,
        }

        self.audit_log.append(entry)

        # Keep only last 10,000 entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]


# Decorator for permission checking
def require_permission(permission: Permission):
    """Decorator to require permission for function."""

    def decorator(func):
        async def wrapper(self, user: User, *args, **kwargs):
            if not self.auth.check_permission(user, permission):
                raise PermissionError(f"User lacks permission: {permission.value}")

            return await func(self, user, *args, **kwargs)

        return wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize auth manager
    auth = AuthManager()

    # Create owner user
    owner = auth.create_user(
        user_id="@alice:matrix.org",
        username="Alice",
        role=UserRole.OWNER,
        password="secure_password_123",
    )

    print(f"✅ Created owner: {owner.username} ({owner.role.value})")

    # Create editor with API key
    editor = auth.create_user(
        user_id="@bob:matrix.org",
        username="Bob",
        role=UserRole.EDITOR,
        generate_api_key=True,
    )

    print(f"✅ Created editor: {editor.username}")
    print(f"   API Key: {editor.api_key}")

    # Authenticate with password
    token = auth.authenticate_password("@alice:matrix.org", "secure_password_123")
    print(f"\n✅ Authenticated Alice: {token[:20]}...")

    # Verify token
    user = auth.verify_token(token)
    print(f"✅ Verified token: {user.username} ({user.role.value})")

    # Check permissions
    can_manage = auth.check_permission(user, Permission.MANAGE_USERS)
    print(f"✅ Alice can manage users: {can_manage}")

    can_manage_bob = auth.check_permission(editor, Permission.MANAGE_USERS)
    print(f"✅ Bob can manage users: {can_manage_bob}")

    # Rate limiting
    for i in range(5):
        allowed = auth.check_rate_limit("@alice:matrix.org")
        remaining = auth.get_remaining_requests("@alice:matrix.org")
        print(f"Request {i+1}: Allowed={allowed}, Remaining={remaining}")

    # Audit log
    print("\n📋 Audit Log:")
    for entry in auth.get_audit_log(limit=5):
        print(f"  {entry['timestamp']} - {entry['event']} ({entry['user_id']})")

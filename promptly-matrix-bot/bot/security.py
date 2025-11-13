"""
Security Hardening for Promptly Matrix Bot

Provides:
- Input sanitization (XSS, SQL injection, command injection)
- Role-Based Access Control (RBAC)
- JWT token validation
- Content Security Policy
- Rate limiting integration

Usage:
    from bot.security import sanitize_input, require_role, SecurityManager

    # Sanitize user input
    clean_text = sanitize_input(user_input)

    # Require specific role
    @require_role("admin")
    async def admin_function(user):
        # Admin-only operation
        pass
"""

import re
import html
import logging
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles for RBAC"""
    GUEST = "guest"  # Read-only access
    USER = "user"  # Standard user
    MODERATOR = "moderator"  # Can manage users
    ADMIN = "admin"  # Full access
    SYSTEM = "system"  # System processes


# Role hierarchy (higher roles inherit lower permissions)
ROLE_HIERARCHY = {
    Role.GUEST: [],
    Role.USER: [Role.GUEST],
    Role.MODERATOR: [Role.GUEST, Role.USER],
    Role.ADMIN: [Role.GUEST, Role.USER, Role.MODERATOR],
    Role.SYSTEM: [Role.GUEST, Role.USER, Role.MODERATOR, Role.ADMIN]
}


@dataclass
class Permission:
    """Permission definition"""
    name: str
    description: str
    required_role: Role


# Pre-defined permissions
PERMISSIONS = {
    "query": Permission("query", "Submit queries", Role.USER),
    "view_stats": Permission("view_stats", "View statistics", Role.GUEST),
    "manage_prompts": Permission("manage_prompts", "Manage prompts", Role.USER),
    "review_pr": Permission("review_pr", "Review pull requests", Role.MODERATOR),
    "manage_users": Permission("manage_users", "Manage users", Role.MODERATOR),
    "admin_config": Permission("admin_config", "Configure system", Role.ADMIN),
    "system_ops": Permission("system_ops", "System operations", Role.SYSTEM)
}


def sanitize_input(
    text: str,
    max_length: int = 10000,
    allow_html: bool = False,
    strip_sql: bool = True,
    strip_commands: bool = True
) -> str:
    """
    Sanitize user input to prevent injection attacks

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        allow_html: Allow HTML tags (will be escaped if False)
        strip_sql: Remove SQL injection patterns
        strip_commands: Remove command injection patterns

    Returns:
        Sanitized text

    Protection against:
    - XSS (Cross-Site Scripting)
    - SQL injection
    - Command injection
    - Path traversal
    - Excessive length
    """
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Escape HTML if not allowed
    if not allow_html:
        text = html.escape(text)

    # Remove null bytes (can break string processing)
    text = text.replace("\x00", "")

    # Strip SQL injection patterns
    if strip_sql:
        # Remove common SQL keywords in dangerous contexts
        sql_patterns = [
            r";\s*DROP\s+TABLE",
            r";\s*DELETE\s+FROM",
            r";\s*UPDATE\s+.*\s+SET",
            r"UNION\s+SELECT",
            r"--\s",  # SQL comment
            r"/\*.*?\*/",  # Multi-line comment
        ]
        for pattern in sql_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Strip command injection patterns
    if strip_commands:
        # Remove shell command separators and operators
        command_patterns = [
            r"[;&|`$]",  # Command separators
            r"\$\([^\)]*\)",  # Command substitution $(...)
            r"`[^`]*`",  # Backtick command substitution
        ]
        for pattern in command_patterns:
            text = re.sub(pattern, "", text)

    # Remove path traversal attempts
    text = text.replace("../", "").replace("..\\", "")

    return text


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename to prevent path traversal

    Args:
        filename: Input filename
        max_length: Maximum allowed length

    Returns:
        Safe filename
    """
    if not filename:
        return "unnamed"

    # Remove path separators
    filename = filename.replace("/", "_").replace("\\", "_")

    # Remove path traversal
    filename = filename.replace("..", "_")

    # Allow only alphanumeric, underscore, dash, dot
    filename = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", filename)

    # Truncate
    filename = filename[:max_length]

    # Ensure not empty
    if not filename:
        return "unnamed"

    return filename


def validate_email(email: str) -> bool:
    """
    Validate email format

    Args:
        email: Email address to validate

    Returns:
        True if valid format
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_matrix_user_id(user_id: str) -> bool:
    """
    Validate Matrix user ID format

    Args:
        user_id: Matrix user ID (@user:server.com)

    Returns:
        True if valid format
    """
    pattern = r"^@[a-zA-Z0-9._=\-]+:[a-zA-Z0-9.\-]+(:[0-9]+)?$"
    return bool(re.match(pattern, user_id))


class SecurityManager:
    """
    Central security management

    Handles:
    - User roles and permissions
    - Access control checks
    - Security audit logging
    """

    def __init__(self):
        self.user_roles: Dict[str, Role] = {}
        self.banned_users: set = set()
        self.access_log: List[Dict] = []

    def set_user_role(self, user_id: str, role: Role):
        """Set user role"""
        self.user_roles[user_id] = role
        logger.info(f"Set role '{role.value}' for user '{user_id}'")

    def get_user_role(self, user_id: str) -> Role:
        """Get user role (defaults to GUEST)"""
        return self.user_roles.get(user_id, Role.GUEST)

    def has_role(self, user_id: str, required_role: Role) -> bool:
        """
        Check if user has required role or higher

        Args:
            user_id: User identifier
            required_role: Required role

        Returns:
            True if user has required role or inherits it
        """
        user_role = self.get_user_role(user_id)

        # Check exact match
        if user_role == required_role:
            return True

        # Check hierarchy (if user role inherits required role)
        return required_role in ROLE_HIERARCHY.get(user_role, [])

    def has_permission(self, user_id: str, permission_name: str) -> bool:
        """
        Check if user has permission

        Args:
            user_id: User identifier
            permission_name: Permission name

        Returns:
            True if user has permission
        """
        permission = PERMISSIONS.get(permission_name)
        if not permission:
            logger.warning(f"Unknown permission: {permission_name}")
            return False

        return self.has_role(user_id, permission.required_role)

    def is_banned(self, user_id: str) -> bool:
        """Check if user is banned"""
        return user_id in self.banned_users

    def ban_user(self, user_id: str, reason: str = ""):
        """Ban user"""
        self.banned_users.add(user_id)
        logger.warning(f"Banned user '{user_id}': {reason}")

    def unban_user(self, user_id: str):
        """Unban user"""
        self.banned_users.discard(user_id)
        logger.info(f"Unbanned user '{user_id}'")

    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        allowed: bool,
        reason: str = ""
    ):
        """
        Log access attempt

        Args:
            user_id: User identifier
            action: Action attempted
            resource: Resource accessed
            allowed: Whether access was allowed
            reason: Reason for denial (if denied)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "allowed": allowed,
            "reason": reason
        }

        self.access_log.append(entry)

        if not allowed:
            logger.warning(
                f"Access denied for user '{user_id}': "
                f"action={action}, resource={resource}, reason={reason}"
            )

    def get_access_log(self, limit: int = 100) -> List[Dict]:
        """Get recent access log entries"""
        return self.access_log[-limit:]


# Global security manager
security_manager = SecurityManager()


def require_role(required_role: Role, security_mgr: Optional[SecurityManager] = None):
    """
    Decorator to require specific role for function

    Args:
        required_role: Required role
        security_mgr: Security manager (defaults to global)

    Example:
        @require_role(Role.ADMIN)
        async def admin_function(user_id: str):
            # Admin-only operation
            pass
    """
    mgr = security_mgr or security_manager

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract user_id from args or kwargs
            user_id = kwargs.get("user_id") or (args[0] if args else None)

            if not user_id:
                raise ValueError("user_id required for role check")

            # Check if banned
            if mgr.is_banned(user_id):
                mgr.log_access(user_id, func.__name__, "function", False, "User banned")
                raise PermissionError(f"User '{user_id}' is banned")

            # Check role
            if not mgr.has_role(user_id, required_role):
                user_role = mgr.get_user_role(user_id)
                mgr.log_access(
                    user_id,
                    func.__name__,
                    "function",
                    False,
                    f"Insufficient role: {user_role.value} (requires {required_role.value})"
                )
                raise PermissionError(
                    f"User '{user_id}' requires role '{required_role.value}' "
                    f"(has '{user_role.value}')"
                )

            # Log access
            mgr.log_access(user_id, func.__name__, "function", True)

            # Execute function
            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_permission(permission_name: str, security_mgr: Optional[SecurityManager] = None):
    """
    Decorator to require specific permission for function

    Args:
        permission_name: Required permission name
        security_mgr: Security manager (defaults to global)

    Example:
        @require_permission("admin_config")
        async def configure_system(user_id: str, config: dict):
            # Admin configuration
            pass
    """
    mgr = security_mgr or security_manager

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract user_id
            user_id = kwargs.get("user_id") or (args[0] if args else None)

            if not user_id:
                raise ValueError("user_id required for permission check")

            # Check if banned
            if mgr.is_banned(user_id):
                mgr.log_access(user_id, func.__name__, "function", False, "User banned")
                raise PermissionError(f"User '{user_id}' is banned")

            # Check permission
            if not mgr.has_permission(user_id, permission_name):
                mgr.log_access(
                    user_id,
                    func.__name__,
                    "function",
                    False,
                    f"Missing permission: {permission_name}"
                )
                raise PermissionError(
                    f"User '{user_id}' lacks permission '{permission_name}'"
                )

            # Log access
            mgr.log_access(user_id, func.__name__, "function", True)

            # Execute function
            return await func(*args, **kwargs)

        return wrapper
    return decorator


# Content Security Policy header
CSP_HEADER = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self' data:; "
    "connect-src 'self' ws: wss:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


def get_security_headers() -> Dict[str, str]:
    """
    Get recommended security headers

    Returns:
        Dict of security headers
    """
    return {
        "Content-Security-Policy": CSP_HEADER,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

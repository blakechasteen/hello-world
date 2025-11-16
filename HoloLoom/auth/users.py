"""
HoloLoom User Management
========================

Simple in-memory user store with password hashing.

Features:
- Bcrypt password hashing
- Role-based access control (admin, user)
- In-memory storage (demo)
- Production migration path documented

Created: 2025-11-16
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import bcrypt
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning(
        "bcrypt not available. Install with: pip install bcrypt"
    )


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"


@dataclass
class User:
    """User model."""
    username: str
    password_hash: bytes
    role: UserRole = UserRole.USER
    active: bool = True
    api_keys: List[str] = field(default_factory=list)
    created_at: Optional[str] = None

    def verify_password(self, password: str) -> bool:
        """
        Verify password against hash.

        Args:
            password: Plain text password

        Returns:
            True if password matches
        """
        if not BCRYPT_AVAILABLE:
            # Insecure fallback for development
            logger.warning("bcrypt not available - using insecure comparison")
            return password.encode() == self.password_hash

        return bcrypt.checkpw(password.encode(), self.password_hash)

    def to_dict(self) -> Dict:
        """Convert to dictionary (no password)."""
        return {
            "username": self.username,
            "role": self.role.value,
            "active": self.active,
            "api_keys": len(self.api_keys),  # Count only
            "created_at": self.created_at,
        }


def hash_password(password: str) -> bytes:
    """
    Hash password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Password hash

    Raises:
        RuntimeError: If bcrypt not available
    """
    if not BCRYPT_AVAILABLE:
        # Insecure fallback for development
        logger.warning(
            "bcrypt not available - using INSECURE plain text storage. "
            "Install bcrypt: pip install bcrypt"
        )
        return password.encode()

    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


# In-memory user database
# In production, replace with PostgreSQL/SQLite
users_db: Dict[str, User] = {
    "admin": User(
        username="admin",
        password_hash=hash_password("admin"),
        role=UserRole.ADMIN,
        created_at="2025-11-16T00:00:00Z",
    ),
    "demo": User(
        username="demo",
        password_hash=hash_password("demo"),
        role=UserRole.USER,
        created_at="2025-11-16T00:00:00Z",
    ),
}


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate user with username and password.

    Args:
        username: Username
        password: Plain text password

    Returns:
        User if authenticated, None if invalid
    """
    user = users_db.get(username)

    if not user:
        logger.warning(f"Authentication failed: user not found: {username}")
        return None

    if not user.active:
        logger.warning(f"Authentication failed: user inactive: {username}")
        return None

    if not user.verify_password(password):
        logger.warning(f"Authentication failed: invalid password: {username}")
        return None

    logger.info(f"User authenticated: {username} ({user.role.value})")
    return user


def get_user(username: str) -> Optional[User]:
    """
    Get user by username.

    Args:
        username: Username

    Returns:
        User if found, None otherwise
    """
    return users_db.get(username)


def create_user(
    username: str,
    password: str,
    role: UserRole = UserRole.USER
) -> User:
    """
    Create new user.

    Args:
        username: Username
        password: Plain text password
        role: User role

    Returns:
        Created user

    Raises:
        ValueError: If user already exists
    """
    if username in users_db:
        raise ValueError(f"User already exists: {username}")

    from datetime import datetime

    user = User(
        username=username,
        password_hash=hash_password(password),
        role=role,
        created_at=datetime.utcnow().isoformat() + "Z",
    )

    users_db[username] = user
    logger.info(f"User created: {username} ({role.value})")

    return user


def delete_user(username: str) -> bool:
    """
    Delete user.

    Args:
        username: Username

    Returns:
        True if deleted, False if not found
    """
    if username in users_db:
        del users_db[username]
        logger.info(f"User deleted: {username}")
        return True

    return False


def list_users() -> List[User]:
    """
    List all users.

    Returns:
        List of all users
    """
    return list(users_db.values())


def update_user_role(username: str, role: UserRole) -> bool:
    """
    Update user role.

    Args:
        username: Username
        role: New role

    Returns:
        True if updated, False if user not found
    """
    user = users_db.get(username)

    if not user:
        return False

    user.role = role
    logger.info(f"User role updated: {username} → {role.value}")

    return True


def deactivate_user(username: str) -> bool:
    """
    Deactivate user (soft delete).

    Args:
        username: Username

    Returns:
        True if deactivated, False if user not found
    """
    user = users_db.get(username)

    if not user:
        return False

    user.active = False
    logger.info(f"User deactivated: {username}")

    return True


def activate_user(username: str) -> bool:
    """
    Activate user.

    Args:
        username: Username

    Returns:
        True if activated, False if user not found
    """
    user = users_db.get(username)

    if not user:
        return False

    user.active = True
    logger.info(f"User activated: {username}")

    return True


def is_bcrypt_available() -> bool:
    """Check if bcrypt is available."""
    return BCRYPT_AVAILABLE


# Example usage
if __name__ == "__main__":
    # Test authentication
    user = authenticate_user("admin", "admin")
    if user:
        print(f"Authenticated: {user.username} ({user.role.value})")

    # Test invalid password
    user = authenticate_user("admin", "wrong")
    print(f"Invalid password: {user}")

    # Create new user
    try:
        new_user = create_user("testuser", "testpass", UserRole.USER)
        print(f"Created: {new_user.username}")
    except ValueError as e:
        print(f"Error: {e}")

    # List users
    users = list_users()
    print(f"\nAll users ({len(users)}):")
    for u in users:
        print(f"  - {u.username} ({u.role.value}, active={u.active})")

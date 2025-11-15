"""
Core RBAC System - Roles, Permissions, and Management

Implements hierarchical role-based access control with:
- Role hierarchy (admin inherits all permissions)
- Permission matrix (role → allowed operations)
- Dynamic role assignment
- Permission checking with caching

References:
- NIST RBAC Model (RBAC0-RBAC3)
- OWASP Access Control Cheat Sheet

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Set, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# Role and Permission Definitions
# ============================================================================

class Role(str, Enum):
    """
    Role hierarchy (higher roles inherit permissions from lower roles).

    Hierarchy: ADMIN > WRITE > READ > GUEST
    """
    ADMIN = "admin"      # Full access (manage users, keys, system config)
    WRITE = "write"      # Read + write data (queries, ingestion, updates)
    READ = "read"        # Read-only access (queries, statistics)
    GUEST = "guest"      # Very limited access (health checks only)


class Permission(str, Enum):
    """
    Fine-grained permissions for operations.

    Format: resource:action (e.g., query:read, user:write)
    """
    # Query permissions
    QUERY_READ = "query:read"       # Execute queries
    QUERY_WRITE = "query:write"     # Ingest data
    QUERY_DELETE = "query:delete"   # Delete data

    # User permissions
    USER_READ = "user:read"         # View user profiles
    USER_WRITE = "user:write"       # Edit users
    USER_DELETE = "user:delete"     # Delete users

    # System permissions
    SYSTEM_CONFIG = "system:config"     # Modify system configuration
    SYSTEM_METRICS = "system:metrics"   # View system metrics
    SYSTEM_HEALTH = "system:health"     # Health check endpoint

    # API Key permissions
    KEY_CREATE = "key:create"       # Create API keys
    KEY_ROTATE = "key:rotate"       # Rotate API keys
    KEY_REVOKE = "key:revoke"       # Revoke API keys
    KEY_LIST = "key:list"           # List API keys

    # Memory permissions
    MEMORY_READ = "memory:read"     # Read from memory
    MEMORY_WRITE = "memory:write"   # Write to memory
    MEMORY_DELETE = "memory:delete" # Delete from memory

    # Alignment permissions
    ALIGNMENT_VIEW = "alignment:view"       # View alignment logs
    ALIGNMENT_OVERRIDE = "alignment:override"  # Override safety checks


# Role hierarchy (higher level inherits from lower levels)
ROLE_HIERARCHY: Dict[Role, List[Role]] = {
    Role.ADMIN: [Role.WRITE, Role.READ, Role.GUEST],  # Admin inherits all
    Role.WRITE: [Role.READ, Role.GUEST],              # Write inherits read
    Role.READ: [Role.GUEST],                          # Read inherits guest
    Role.GUEST: [],                                   # Guest is base level
}


# Permission matrix (role → allowed permissions)
PERMISSION_MATRIX: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        # Admin has ALL permissions
        Permission.QUERY_READ, Permission.QUERY_WRITE, Permission.QUERY_DELETE,
        Permission.USER_READ, Permission.USER_WRITE, Permission.USER_DELETE,
        Permission.SYSTEM_CONFIG, Permission.SYSTEM_METRICS, Permission.SYSTEM_HEALTH,
        Permission.KEY_CREATE, Permission.KEY_ROTATE, Permission.KEY_REVOKE, Permission.KEY_LIST,
        Permission.MEMORY_READ, Permission.MEMORY_WRITE, Permission.MEMORY_DELETE,
        Permission.ALIGNMENT_VIEW, Permission.ALIGNMENT_OVERRIDE,
    },
    Role.WRITE: {
        # Write: read + write operations (no delete, no admin)
        Permission.QUERY_READ, Permission.QUERY_WRITE,
        Permission.USER_READ,
        Permission.SYSTEM_METRICS, Permission.SYSTEM_HEALTH,
        Permission.KEY_LIST,
        Permission.MEMORY_READ, Permission.MEMORY_WRITE,
        Permission.ALIGNMENT_VIEW,
    },
    Role.READ: {
        # Read-only: queries and statistics
        Permission.QUERY_READ,
        Permission.USER_READ,
        Permission.SYSTEM_METRICS, Permission.SYSTEM_HEALTH,
        Permission.KEY_LIST,
        Permission.MEMORY_READ,
        Permission.ALIGNMENT_VIEW,
    },
    Role.GUEST: {
        # Guest: very limited (health checks only)
        Permission.SYSTEM_HEALTH,
    },
}


# ============================================================================
# User Role Assignment
# ============================================================================

@dataclass
class UserRole:
    """
    User role assignment with metadata.

    Attributes:
        user_id: User identifier
        role: Assigned role
        assigned_at: When role was assigned
        assigned_by: Who assigned the role (user_id or system)
        expires_at: Optional expiration timestamp
        metadata: Optional additional metadata
    """
    user_id: str
    role: Role
    assigned_at: datetime
    assigned_by: str
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if role assignment has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "role": self.role.value,
            "assigned_at": self.assigned_at.isoformat(),
            "assigned_by": self.assigned_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserRole":
        """Deserialize from dictionary."""
        return cls(
            user_id=data["user_id"],
            role=Role(data["role"]),
            assigned_at=datetime.fromisoformat(data["assigned_at"]),
            assigned_by=data["assigned_by"],
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at") else None
            ),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# RBAC Manager
# ============================================================================

class RBACManager:
    """
    Role-Based Access Control manager.

    Features:
    - Hierarchical role system (admin inherits all permissions)
    - Permission caching for performance
    - Dynamic role assignment
    - Wildcard permission support (query:*)
    - Audit logging

    Usage:
        >>> rbac = RBACManager()
        >>> await rbac.assign_role("user_123", Role.WRITE)
        >>> has_perm = await rbac.check_permission("user_123", Permission.QUERY_WRITE)
        >>> print(has_perm)  # True
    """

    def __init__(
        self,
        storage: Optional['RBACStorage'] = None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300
    ):
        """
        Initialize RBAC manager.

        Args:
            storage: Storage backend (defaults to in-memory)
            enable_caching: Enable permission caching
            cache_ttl_seconds: Cache TTL in seconds (default: 5 minutes)
        """
        from HoloLoom.security.rbac.storage import InMemoryRBACStorage

        self.storage = storage or InMemoryRBACStorage()
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds

        # Permission cache: user_id → (permissions, expires_at)
        self._permission_cache: Dict[str, tuple[Set[Permission], datetime]] = {}

        # Statistics
        self._stats = {
            "role_assignments": 0,
            "permission_checks": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def assign_role(
        self,
        user_id: str,
        role: Role,
        assigned_by: str = "system",
        ttl_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserRole:
        """
        Assign role to user.

        Args:
            user_id: User identifier
            role: Role to assign
            assigned_by: Who is assigning (user_id or "system")
            ttl_days: Optional time-to-live in days
            metadata: Optional metadata

        Returns:
            UserRole object
        """
        user_role = UserRole(
            user_id=user_id,
            role=role,
            assigned_at=datetime.now(),
            assigned_by=assigned_by,
            expires_at=(
                datetime.now() + timedelta(days=ttl_days)
                if ttl_days else None
            ),
            metadata=metadata or {}
        )

        # Store in backend
        await self.storage.store_user_role(user_role)

        # Invalidate cache
        self._invalidate_cache(user_id)

        # Update stats
        self._stats["role_assignments"] += 1

        logger.info(
            f"✅ Assigned role {role.value} to user {user_id} "
            f"(by: {assigned_by}, expires: {user_role.expires_at})"
        )

        return user_role

    async def get_user_role(self, user_id: str) -> Optional[Role]:
        """
        Get user's current role.

        Args:
            user_id: User identifier

        Returns:
            Role or None if no role assigned
        """
        user_role = await self.storage.get_user_role(user_id)

        if user_role is None:
            return None

        # Check if expired
        if user_role.is_expired():
            logger.warning(
                f"⚠️  Role expired for user {user_id} "
                f"(expired: {user_role.expires_at})"
            )
            return None

        return user_role.role

    async def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        Get all permissions for user (including inherited).

        Args:
            user_id: User identifier

        Returns:
            Set of permissions
        """
        self._stats["permission_checks"] += 1

        # Check cache
        if self.enable_caching and user_id in self._permission_cache:
            permissions, expires_at = self._permission_cache[user_id]
            if datetime.now() < expires_at:
                self._stats["cache_hits"] += 1
                return permissions

        self._stats["cache_misses"] += 1

        # Get user role
        role = await self.get_user_role(user_id)
        if role is None:
            return set()

        # Get permissions (including inherited)
        permissions = self._get_role_permissions(role)

        # Cache result
        if self.enable_caching:
            expires_at = datetime.now() + timedelta(seconds=self.cache_ttl_seconds)
            self._permission_cache[user_id] = (permissions, expires_at)

        return permissions

    async def check_permission(
        self,
        user_id: str,
        permission: Permission
    ) -> bool:
        """
        Check if user has specific permission.

        Args:
            user_id: User identifier
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        permissions = await self.get_user_permissions(user_id)
        return permission in permissions

    async def check_any_permission(
        self,
        user_id: str,
        permissions: List[Permission]
    ) -> bool:
        """
        Check if user has ANY of the specified permissions.

        Args:
            user_id: User identifier
            permissions: List of permissions to check

        Returns:
            True if user has at least one permission
        """
        user_permissions = await self.get_user_permissions(user_id)
        return any(perm in user_permissions for perm in permissions)

    async def check_all_permissions(
        self,
        user_id: str,
        permissions: List[Permission]
    ) -> bool:
        """
        Check if user has ALL of the specified permissions.

        Args:
            user_id: User identifier
            permissions: List of permissions to check

        Returns:
            True if user has all permissions
        """
        user_permissions = await self.get_user_permissions(user_id)
        return all(perm in user_permissions for perm in permissions)

    async def revoke_role(self, user_id: str) -> bool:
        """
        Revoke user's role.

        Args:
            user_id: User identifier

        Returns:
            True if role was revoked, False if no role to revoke
        """
        success = await self.storage.delete_user_role(user_id)

        if success:
            self._invalidate_cache(user_id)
            logger.info(f"🗑️  Revoked role for user {user_id}")

        return success

    async def list_users_with_role(self, role: Role) -> List[UserRole]:
        """
        List all users with specific role.

        Args:
            role: Role to search for

        Returns:
            List of UserRole objects
        """
        return await self.storage.list_users_with_role(role)

    def get_statistics(self) -> Dict[str, Any]:
        """Get RBAC statistics."""
        return {
            **self._stats,
            "cache_size": len(self._permission_cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] / self._stats["permission_checks"]
                if self._stats["permission_checks"] > 0 else 0.0
            ),
        }

    def _get_role_permissions(self, role: Role) -> Set[Permission]:
        """
        Get all permissions for role (including inherited).

        Args:
            role: Role to get permissions for

        Returns:
            Set of permissions
        """
        permissions = set(PERMISSION_MATRIX[role])

        # Add permissions from inherited roles
        for inherited_role in ROLE_HIERARCHY[role]:
            permissions.update(PERMISSION_MATRIX[inherited_role])

        return permissions

    def _invalidate_cache(self, user_id: str):
        """Invalidate permission cache for user."""
        if user_id in self._permission_cache:
            del self._permission_cache[user_id]


# ============================================================================
# Factory Function
# ============================================================================

def create_rbac_manager(
    storage_type: str = "memory",
    redis_url: Optional[str] = None,
    enable_caching: bool = True
) -> RBACManager:
    """
    Create RBAC manager with specified storage backend.

    Args:
        storage_type: "memory" or "redis"
        redis_url: Redis URL (if storage_type="redis")
        enable_caching: Enable permission caching

    Returns:
        RBACManager instance

    Usage:
        >>> # In-memory (development)
        >>> rbac = create_rbac_manager(storage_type="memory")
        >>>
        >>> # Redis (production)
        >>> rbac = create_rbac_manager(
        ...     storage_type="redis",
        ...     redis_url="redis://localhost:6379/0"
        ... )
    """
    from HoloLoom.security.rbac.storage import create_rbac_storage

    storage = create_rbac_storage(storage_type, redis_url=redis_url)
    return RBACManager(storage=storage, enable_caching=enable_caching)


# ============================================================================
# Helper Functions
# ============================================================================

def get_role_level(role: Role) -> int:
    """
    Get numeric level for role (higher = more privileged).

    Args:
        role: Role to get level for

    Returns:
        Numeric level (0=guest, 1=read, 2=write, 3=admin)
    """
    levels = {
        Role.GUEST: 0,
        Role.READ: 1,
        Role.WRITE: 2,
        Role.ADMIN: 3,
    }
    return levels[role]


def role_inherits_from(role: Role, other: Role) -> bool:
    """
    Check if role inherits from other role.

    Args:
        role: Role to check
        other: Other role

    Returns:
        True if role inherits from other

    Example:
        >>> role_inherits_from(Role.ADMIN, Role.READ)  # True
        >>> role_inherits_from(Role.READ, Role.ADMIN)  # False
    """
    return other in ROLE_HIERARCHY[role]


def get_permission_resource(permission: Permission) -> str:
    """
    Extract resource from permission.

    Args:
        permission: Permission (e.g., Permission.QUERY_READ)

    Returns:
        Resource name (e.g., "query")
    """
    return permission.value.split(":")[0]


def get_permission_action(permission: Permission) -> str:
    """
    Extract action from permission.

    Args:
        permission: Permission (e.g., Permission.QUERY_READ)

    Returns:
        Action name (e.g., "read")
    """
    return permission.value.split(":")[1]

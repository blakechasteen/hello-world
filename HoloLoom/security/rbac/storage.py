"""
RBAC Storage Layer - Redis and In-Memory Backends

Provides storage backends for user role assignments:
- InMemoryRBACStorage: Fast in-memory storage (development)
- RedisRBACStorage: Distributed storage with TTL support (production)

Both implement the RBACStorage protocol for easy swapping.

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Protocol

from HoloLoom.security.rbac.core import Role, UserRole

logger = logging.getLogger(__name__)


# ============================================================================
# Storage Protocol
# ============================================================================

class RBACStorage(Protocol):
    """
    Protocol for RBAC storage backends.

    Implementations must provide:
    - store_user_role: Store/update user role
    - get_user_role: Retrieve user role
    - delete_user_role: Delete user role
    - list_users_with_role: List users with specific role
    """

    async def store_user_role(self, user_role: UserRole) -> None:
        """Store or update user role."""
        ...

    async def get_user_role(self, user_id: str) -> Optional[UserRole]:
        """Get user role by user_id."""
        ...

    async def delete_user_role(self, user_id: str) -> bool:
        """Delete user role. Returns True if deleted, False if not found."""
        ...

    async def list_users_with_role(self, role: Role) -> List[UserRole]:
        """List all users with specific role."""
        ...


# ============================================================================
# In-Memory Storage (Development)
# ============================================================================

class InMemoryRBACStorage:
    """
    In-memory RBAC storage (fast, non-persistent).

    Good for:
    - Development and testing
    - Single-server deployments
    - Prototyping

    Not suitable for:
    - Multi-server deployments (no shared state)
    - Production (data lost on restart)
    """

    def __init__(self):
        """Initialize in-memory storage."""
        # user_id → UserRole
        self._roles: Dict[str, UserRole] = {}

        # role → List[user_id] (index for fast lookup)
        self._role_index: Dict[Role, List[str]] = {
            role: [] for role in Role
        }

    async def store_user_role(self, user_role: UserRole) -> None:
        """
        Store or update user role.

        Args:
            user_role: UserRole object to store
        """
        user_id = user_role.user_id

        # Remove from old role index if exists
        if user_id in self._roles:
            old_role = self._roles[user_id].role
            if user_id in self._role_index[old_role]:
                self._role_index[old_role].remove(user_id)

        # Store user role
        self._roles[user_id] = user_role

        # Update role index
        if user_id not in self._role_index[user_role.role]:
            self._role_index[user_role.role].append(user_id)

        logger.debug(f"Stored role {user_role.role.value} for user {user_id}")

    async def get_user_role(self, user_id: str) -> Optional[UserRole]:
        """
        Get user role by user_id.

        Args:
            user_id: User identifier

        Returns:
            UserRole object or None if not found
        """
        return self._roles.get(user_id)

    async def delete_user_role(self, user_id: str) -> bool:
        """
        Delete user role.

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        if user_id not in self._roles:
            return False

        # Remove from role index
        role = self._roles[user_id].role
        if user_id in self._role_index[role]:
            self._role_index[role].remove(user_id)

        # Delete user role
        del self._roles[user_id]

        logger.debug(f"Deleted role for user {user_id}")
        return True

    async def list_users_with_role(self, role: Role) -> List[UserRole]:
        """
        List all users with specific role.

        Args:
            role: Role to search for

        Returns:
            List of UserRole objects
        """
        user_ids = self._role_index[role]
        return [self._roles[user_id] for user_id in user_ids]


# ============================================================================
# Redis Storage (Production)
# ============================================================================

class RedisRBACStorage:
    """
    Redis-backed RBAC storage (distributed, persistent).

    Good for:
    - Production deployments
    - Multi-server deployments (shared state)
    - Automatic TTL support

    Features:
    - Distributed storage (all servers see same roles)
    - Automatic expiration (via Redis TTL)
    - Fast permission lookups (<1ms)
    - Persistence (data survives restarts)

    Usage:
        >>> storage = RedisRBACStorage(redis_url="redis://localhost:6379/0")
        >>> await storage.store_user_role(user_role)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize Redis storage.

        Args:
            redis_url: Redis connection URL

        Raises:
            ImportError: If redis package not installed
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "Redis storage requires 'redis' package. "
                "Install with: pip install redis"
            )

        self.redis_url = redis_url
        self._redis = redis.from_url(redis_url, decode_responses=True)

        # Key prefixes
        self.USER_ROLE_PREFIX = "rbac:user:"
        self.ROLE_INDEX_PREFIX = "rbac:role:"

    async def store_user_role(self, user_role: UserRole) -> None:
        """
        Store or update user role in Redis.

        Args:
            user_role: UserRole object to store
        """
        user_id = user_role.user_id

        # Remove from old role index if exists
        old_user_role = await self.get_user_role(user_id)
        if old_user_role:
            old_role_key = f"{self.ROLE_INDEX_PREFIX}{old_user_role.role.value}"
            await self._redis.srem(old_role_key, user_id)

        # Store user role
        user_role_key = f"{self.USER_ROLE_PREFIX}{user_id}"
        await self._redis.set(
            user_role_key,
            json.dumps(user_role.to_dict())
        )

        # Set TTL if expires_at is set
        if user_role.expires_at:
            from datetime import datetime
            ttl_seconds = int((user_role.expires_at - datetime.now()).total_seconds())
            if ttl_seconds > 0:
                await self._redis.expire(user_role_key, ttl_seconds)

        # Update role index
        role_index_key = f"{self.ROLE_INDEX_PREFIX}{user_role.role.value}"
        await self._redis.sadd(role_index_key, user_id)

        logger.debug(f"Stored role {user_role.role.value} for user {user_id} in Redis")

    async def get_user_role(self, user_id: str) -> Optional[UserRole]:
        """
        Get user role from Redis.

        Args:
            user_id: User identifier

        Returns:
            UserRole object or None if not found
        """
        user_role_key = f"{self.USER_ROLE_PREFIX}{user_id}"
        data = await self._redis.get(user_role_key)

        if data is None:
            return None

        return UserRole.from_dict(json.loads(data))

    async def delete_user_role(self, user_id: str) -> bool:
        """
        Delete user role from Redis.

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        # Get current role to update index
        user_role = await self.get_user_role(user_id)
        if user_role is None:
            return False

        # Remove from role index
        role_index_key = f"{self.ROLE_INDEX_PREFIX}{user_role.role.value}"
        await self._redis.srem(role_index_key, user_id)

        # Delete user role
        user_role_key = f"{self.USER_ROLE_PREFIX}{user_id}"
        deleted = await self._redis.delete(user_role_key)

        logger.debug(f"Deleted role for user {user_id} from Redis")
        return deleted > 0

    async def list_users_with_role(self, role: Role) -> List[UserRole]:
        """
        List all users with specific role from Redis.

        Args:
            role: Role to search for

        Returns:
            List of UserRole objects
        """
        role_index_key = f"{self.ROLE_INDEX_PREFIX}{role.value}"
        user_ids = await self._redis.smembers(role_index_key)

        user_roles = []
        for user_id in user_ids:
            user_role = await self.get_user_role(user_id)
            if user_role:  # Could be None if expired
                user_roles.append(user_role)

        return user_roles

    async def close(self):
        """Close Redis connection."""
        await self._redis.close()


# ============================================================================
# Factory Function
# ============================================================================

def create_rbac_storage(
    storage_type: str = "memory",
    redis_url: Optional[str] = None
) -> RBACStorage:
    """
    Create RBAC storage backend.

    Args:
        storage_type: "memory" or "redis"
        redis_url: Redis URL (required if storage_type="redis")

    Returns:
        RBACStorage implementation

    Raises:
        ValueError: If invalid storage_type or missing redis_url

    Usage:
        >>> # In-memory (development)
        >>> storage = create_rbac_storage("memory")
        >>>
        >>> # Redis (production)
        >>> storage = create_rbac_storage("redis", redis_url="redis://localhost:6379/0")
    """
    if storage_type == "memory":
        return InMemoryRBACStorage()

    elif storage_type == "redis":
        if redis_url is None:
            raise ValueError(
                "redis_url required when storage_type='redis'. "
                "Example: redis://localhost:6379/0"
            )
        return RedisRBACStorage(redis_url=redis_url)

    else:
        raise ValueError(
            f"Invalid storage_type: {storage_type}. "
            f"Must be 'memory' or 'redis'"
        )

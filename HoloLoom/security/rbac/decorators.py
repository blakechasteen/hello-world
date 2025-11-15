"""
FastAPI RBAC Decorators - Endpoint Protection

Provides decorators for protecting FastAPI endpoints with role/permission checks:
- @require_role: Require specific role
- @require_permission: Require specific permission
- @require_any_role: Require any of specified roles
- @require_all_permissions: Require all specified permissions

Integrates with FastAPI dependency injection and JWT/API key authentication.

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
from functools import wraps
from typing import Optional, List, Callable, Any

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from HoloLoom.security.rbac.core import Role, Permission, RBACManager

logger = logging.getLogger(__name__)

# FastAPI security scheme
security = HTTPBearer()

# Global RBAC manager instance (set via set_rbac_manager)
_rbac_manager: Optional[RBACManager] = None


# ============================================================================
# Configuration
# ============================================================================

def set_rbac_manager(rbac: RBACManager):
    """
    Set global RBAC manager instance.

    Call this during app startup:
        >>> rbac = create_rbac_manager()
        >>> set_rbac_manager(rbac)

    Args:
        rbac: RBACManager instance
    """
    global _rbac_manager
    _rbac_manager = rbac
    logger.info("✅ RBAC manager configured for decorators")


def get_rbac_manager() -> RBACManager:
    """
    Get global RBAC manager instance.

    Returns:
        RBACManager instance

    Raises:
        RuntimeError: If RBAC manager not configured
    """
    if _rbac_manager is None:
        raise RuntimeError(
            "RBAC manager not configured! "
            "Call set_rbac_manager(rbac) during app startup."
        )
    return _rbac_manager


# ============================================================================
# User Extraction from Request
# ============================================================================

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
) -> str:
    """
    Extract user_id from JWT token or API key.

    This is a simplified version. In production, you would:
    1. Verify JWT signature
    2. Decode JWT payload
    3. Extract user_id from claims
    4. Or verify API key and get user_id from key metadata

    Args:
        credentials: HTTP Bearer credentials (JWT or API key)
        request: FastAPI request object

    Returns:
        User identifier

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials

    # TODO: Implement proper JWT verification
    # For now, we'll extract user_id from a simple format: "user_<id>"
    # In production, decode JWT and verify signature

    try:
        # Example JWT payload: {"sub": "user_123", "role": "admin"}
        # For demo, we'll parse a simple format
        if token.startswith("user_"):
            return token
        else:
            # In production, decode JWT here
            raise ValueError("Invalid token format")

    except Exception as e:
        logger.warning(f"❌ Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_role(
    user_id: str = Depends(get_current_user_id),
    rbac: RBACManager = Depends(get_rbac_manager)
) -> Role:
    """
    Get current user's role.

    Args:
        user_id: User identifier (from get_current_user_id)
        rbac: RBAC manager instance

    Returns:
        User's role

    Raises:
        HTTPException: If user has no role assigned
    """
    role = await rbac.get_user_role(user_id)

    if role is None:
        logger.warning(f"❌ No role assigned to user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No role assigned. Contact administrator.",
        )

    return role


# ============================================================================
# Role-Based Decorators
# ============================================================================

def require_role(required_role: Role):
    """
    Decorator to require specific role for endpoint.

    Checks if user has required role (or higher in hierarchy).

    Args:
        required_role: Minimum required role

    Returns:
        Decorated function

    Usage:
        >>> @app.get("/admin/users")
        >>> @require_role(Role.ADMIN)
        >>> async def list_users():
        ...     return {"users": [...]}

    Example:
        - User with ADMIN role can access WRITE, READ, GUEST endpoints
        - User with WRITE role can access READ, GUEST endpoints
        - User with READ role can access GUEST endpoints
    """
    async def role_checker(
        user_id: str = Depends(get_current_user_id),
        rbac: RBACManager = Depends(get_rbac_manager)
    ):
        user_role = await rbac.get_user_role(user_id)

        if user_role is None:
            logger.warning(f"❌ No role for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No role assigned",
            )

        # Check if user role is sufficient (hierarchy check)
        from HoloLoom.security.rbac.core import get_role_level

        if get_role_level(user_role) < get_role_level(required_role):
            logger.warning(
                f"❌ Insufficient role for user {user_id}: "
                f"has {user_role.value}, needs {required_role.value}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_role.value} role",
            )

        logger.debug(
            f"✅ Role check passed for user {user_id}: "
            f"{user_role.value} >= {required_role.value}"
        )
        return user_id

    return Depends(role_checker)


def require_permission(required_permission: Permission):
    """
    Decorator to require specific permission for endpoint.

    Args:
        required_permission: Required permission

    Returns:
        Decorated function

    Usage:
        >>> @app.post("/query")
        >>> @require_permission(Permission.QUERY_WRITE)
        >>> async def execute_query(query: str):
        ...     return {"result": "..."}
    """
    async def permission_checker(
        user_id: str = Depends(get_current_user_id),
        rbac: RBACManager = Depends(get_rbac_manager)
    ):
        has_permission = await rbac.check_permission(user_id, required_permission)

        if not has_permission:
            logger.warning(
                f"❌ Permission denied for user {user_id}: "
                f"missing {required_permission.value}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {required_permission.value}",
            )

        logger.debug(
            f"✅ Permission check passed for user {user_id}: "
            f"{required_permission.value}"
        )
        return user_id

    return Depends(permission_checker)


def require_any_role(required_roles: List[Role]):
    """
    Decorator to require ANY of the specified roles.

    Args:
        required_roles: List of acceptable roles

    Returns:
        Decorated function

    Usage:
        >>> @app.get("/stats")
        >>> @require_any_role([Role.ADMIN, Role.WRITE, Role.READ])
        >>> async def get_stats():
        ...     return {"stats": {...}}
    """
    async def any_role_checker(
        user_id: str = Depends(get_current_user_id),
        rbac: RBACManager = Depends(get_rbac_manager)
    ):
        user_role = await rbac.get_user_role(user_id)

        if user_role is None or user_role not in required_roles:
            logger.warning(
                f"❌ User {user_id} missing required roles: "
                f"{[r.value for r in required_roles]}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {', '.join(r.value for r in required_roles)}",
            )

        logger.debug(
            f"✅ Any-role check passed for user {user_id}: "
            f"{user_role.value} in {[r.value for r in required_roles]}"
        )
        return user_id

    return Depends(any_role_checker)


def require_all_permissions(required_permissions: List[Permission]):
    """
    Decorator to require ALL of the specified permissions.

    Args:
        required_permissions: List of required permissions

    Returns:
        Decorated function

    Usage:
        >>> @app.delete("/user/{user_id}")
        >>> @require_all_permissions([
        ...     Permission.USER_READ,
        ...     Permission.USER_DELETE
        ... ])
        >>> async def delete_user(user_id: str):
        ...     return {"deleted": user_id}
    """
    async def all_permissions_checker(
        user_id: str = Depends(get_current_user_id),
        rbac: RBACManager = Depends(get_rbac_manager)
    ):
        has_all = await rbac.check_all_permissions(user_id, required_permissions)

        if not has_all:
            missing = []
            for perm in required_permissions:
                if not await rbac.check_permission(user_id, perm):
                    missing.append(perm.value)

            logger.warning(
                f"❌ User {user_id} missing permissions: {missing}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(missing)}",
            )

        logger.debug(
            f"✅ All-permissions check passed for user {user_id}"
        )
        return user_id

    return Depends(all_permissions_checker)


# ============================================================================
# Convenience Dependencies
# ============================================================================

async def require_admin(
    user_id: str = Depends(require_role(Role.ADMIN))
) -> str:
    """
    Convenience dependency for admin-only endpoints.

    Usage:
        >>> @app.get("/admin/config")
        >>> async def get_config(user_id: str = Depends(require_admin)):
        ...     return {"config": {...}}
    """
    return user_id


async def require_write(
    user_id: str = Depends(require_role(Role.WRITE))
) -> str:
    """
    Convenience dependency for write-access endpoints.

    Usage:
        >>> @app.post("/ingest")
        >>> async def ingest_data(
        ...     data: dict,
        ...     user_id: str = Depends(require_write)
        ... ):
        ...     return {"ingested": True}
    """
    return user_id


async def require_read(
    user_id: str = Depends(require_role(Role.READ))
) -> str:
    """
    Convenience dependency for read-access endpoints.

    Usage:
        >>> @app.get("/query")
        >>> async def query(q: str, user_id: str = Depends(require_read)):
        ...     return {"result": [...]}
    """
    return user_id


# ============================================================================
# Resource-Based Access Control Helper
# ============================================================================

async def check_resource_access(
    user_id: str,
    resource_owner_id: str,
    required_permission: Permission,
    rbac: RBACManager
) -> bool:
    """
    Check resource-based access control.

    Users can access their own resources OR have the required permission.

    Args:
        user_id: Current user ID
        resource_owner_id: Resource owner ID
        required_permission: Permission needed to access others' resources
        rbac: RBAC manager

    Returns:
        True if access allowed

    Usage:
        >>> # User can edit own profile, or admin can edit any profile
        >>> can_edit = await check_resource_access(
        ...     user_id="user_123",
        ...     resource_owner_id="user_456",
        ...     required_permission=Permission.USER_WRITE,
        ...     rbac=rbac
        ... )
    """
    # Allow if user owns the resource
    if user_id == resource_owner_id:
        return True

    # Allow if user has required permission
    return await rbac.check_permission(user_id, required_permission)


def require_resource_access(
    resource_owner_id: str,
    required_permission: Permission
):
    """
    Decorator for resource-based access control.

    Args:
        resource_owner_id: Owner of the resource
        required_permission: Permission needed to access others' resources

    Returns:
        Decorated function

    Usage:
        >>> @app.put("/profile/{user_id}")
        >>> async def update_profile(
        ...     user_id: str,
        ...     data: dict,
        ...     _: str = require_resource_access(user_id, Permission.USER_WRITE)
        ... ):
        ...     return {"updated": user_id}
    """
    async def resource_access_checker(
        current_user_id: str = Depends(get_current_user_id),
        rbac: RBACManager = Depends(get_rbac_manager)
    ):
        has_access = await check_resource_access(
            current_user_id,
            resource_owner_id,
            required_permission,
            rbac
        )

        if not has_access:
            logger.warning(
                f"❌ Resource access denied for user {current_user_id}: "
                f"resource owned by {resource_owner_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: not resource owner and insufficient permissions",
            )

        return current_user_id

    return Depends(resource_access_checker)

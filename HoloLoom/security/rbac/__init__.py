"""
Role-Based Access Control (RBAC) for HoloLoom

Comprehensive RBAC system with:
- Hierarchical role system (admin > write > read > guest)
- Fine-grained permissions (query:read, user:write, etc.)
- Resource-based access control (user can edit own profile)
- Policy engine for complex rules
- FastAPI decorator integration
- Redis/PostgreSQL storage support

Usage:
    >>> from HoloLoom.security.rbac import RBACManager, Role, Permission
    >>> from HoloLoom.security.rbac.decorators import require_role, require_permission
    >>>
    >>> # Initialize RBAC
    >>> rbac = RBACManager()
    >>>
    >>> # Assign role to user
    >>> await rbac.assign_role("user_123", Role.WRITE)
    >>>
    >>> # Check permissions
    >>> has_perm = await rbac.check_permission("user_123", Permission.QUERY_WRITE)
    >>>
    >>> # FastAPI endpoint protection
    >>> @require_role(Role.ADMIN)
    >>> async def admin_endpoint():
    ...     return {"message": "Admin only"}

Author: HoloLoom Security Team
Created: 2025-11-15
"""

from HoloLoom.security.rbac.core import (
    Role,
    Permission,
    RBACManager,
    create_rbac_manager,
    ROLE_HIERARCHY,
    PERMISSION_MATRIX,
)

from HoloLoom.security.rbac.decorators import (
    require_role,
    require_permission,
    require_any_role,
    require_all_permissions,
    get_current_user_role,
)

from HoloLoom.security.rbac.policy_engine import (
    PolicyEngine,
    PolicyRule,
    PolicyContext,
    PolicyDecision,
    create_policy_engine,
)

from HoloLoom.security.rbac.storage import (
    RBACStorage,
    InMemoryRBACStorage,
    RedisRBACStorage,
    create_rbac_storage,
)

__all__ = [
    # Core
    "Role",
    "Permission",
    "RBACManager",
    "create_rbac_manager",
    "ROLE_HIERARCHY",
    "PERMISSION_MATRIX",
    # Decorators
    "require_role",
    "require_permission",
    "require_any_role",
    "require_all_permissions",
    "get_current_user_role",
    # Policy Engine
    "PolicyEngine",
    "PolicyRule",
    "PolicyContext",
    "PolicyDecision",
    "create_policy_engine",
    # Storage
    "RBACStorage",
    "InMemoryRBACStorage",
    "RedisRBACStorage",
    "create_rbac_storage",
]

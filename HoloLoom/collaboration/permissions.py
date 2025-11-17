"""
Permission System
=================
Role-based access control for collaborative sessions.

Philosophy:
Not all shuttles have equal access to the loom. Some may only observe,
while others can weave new threads or even reconfigure the loom itself.
Permissions ensure orderly collaboration.

Roles (ascending privilege):
- VIEWER: Read-only access
- EDITOR: Can make changes
- ADMIN: Can manage users and settings
- OWNER: Full control including deletion
"""

import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles in order of privilege."""
    VIEWER = "viewer"      # Read-only
    EDITOR = "editor"      # Can edit
    ADMIN = "admin"        # Can manage users
    OWNER = "owner"        # Full control

    def __lt__(self, other):
        """Enable role comparison."""
        if not isinstance(other, Role):
            return NotImplemented

        role_order = {
            Role.VIEWER: 0,
            Role.EDITOR: 1,
            Role.ADMIN: 2,
            Role.OWNER: 3
        }

        return role_order[self] < role_order[other]

    def __le__(self, other):
        """Enable role comparison."""
        return self == other or self < other

    def __gt__(self, other):
        """Enable role comparison."""
        if not isinstance(other, Role):
            return NotImplemented
        return not self <= other

    def __ge__(self, other):
        """Enable role comparison."""
        return self == other or self > other


class Permission(str, Enum):
    """Specific permissions."""
    # Read permissions
    READ_SESSION = "read_session"
    READ_HISTORY = "read_history"
    READ_USERS = "read_users"

    # Write permissions
    SEND_MESSAGE = "send_message"
    EDIT_MESSAGE = "edit_message"
    DELETE_MESSAGE = "delete_message"

    # State permissions
    UPDATE_STATE = "update_state"
    CLEAR_HISTORY = "clear_history"

    # User management
    INVITE_USER = "invite_user"
    REMOVE_USER = "remove_user"
    CHANGE_USER_ROLE = "change_user_role"

    # Session management
    EDIT_SESSION = "edit_session"
    DELETE_SESSION = "delete_session"
    TRANSFER_OWNERSHIP = "transfer_ownership"


# Define permissions for each role
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_SESSION,
        Permission.READ_HISTORY,
        Permission.READ_USERS,
    },
    Role.EDITOR: {
        # All viewer permissions
        Permission.READ_SESSION,
        Permission.READ_HISTORY,
        Permission.READ_USERS,
        # Plus editing
        Permission.SEND_MESSAGE,
        Permission.EDIT_MESSAGE,
        Permission.UPDATE_STATE,
    },
    Role.ADMIN: {
        # All editor permissions
        Permission.READ_SESSION,
        Permission.READ_HISTORY,
        Permission.READ_USERS,
        Permission.SEND_MESSAGE,
        Permission.EDIT_MESSAGE,
        Permission.UPDATE_STATE,
        # Plus user management
        Permission.INVITE_USER,
        Permission.REMOVE_USER,
        Permission.CHANGE_USER_ROLE,
        Permission.DELETE_MESSAGE,
        Permission.CLEAR_HISTORY,
        Permission.EDIT_SESSION,
    },
    Role.OWNER: {
        # All permissions
        Permission.READ_SESSION,
        Permission.READ_HISTORY,
        Permission.READ_USERS,
        Permission.SEND_MESSAGE,
        Permission.EDIT_MESSAGE,
        Permission.UPDATE_STATE,
        Permission.INVITE_USER,
        Permission.REMOVE_USER,
        Permission.CHANGE_USER_ROLE,
        Permission.DELETE_MESSAGE,
        Permission.CLEAR_HISTORY,
        Permission.EDIT_SESSION,
        Permission.DELETE_SESSION,
        Permission.TRANSFER_OWNERSHIP,
    }
}


@dataclass
class PermissionCheck:
    """Result of permission check."""
    allowed: bool
    reason: Optional[str] = None

    @staticmethod
    def allow() -> 'PermissionCheck':
        """Create allowed result."""
        return PermissionCheck(allowed=True)

    @staticmethod
    def deny(reason: str) -> 'PermissionCheck':
        """Create denied result."""
        return PermissionCheck(allowed=False, reason=reason)


class PermissionManager:
    """
    Manages permissions for collaborative sessions.

    Features:
    - Role-based access control
    - Permission checking
    - Role elevation/demotion
    - Custom permission rules
    """

    def __init__(self):
        """Initialize permission manager."""
        self.custom_permissions: Dict[str, Dict[str, Set[Permission]]] = {}  # session_id -> user_id -> permissions
        logger.info("PermissionManager initialized")

    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get permissions for a role."""
        return ROLE_PERMISSIONS.get(role, set())

    def has_permission(
        self,
        user_id: str,
        role: Role,
        permission: Permission,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Check if user has permission.

        Args:
            user_id: User ID
            role: User's role
            permission: Permission to check
            session_id: Optional session for custom permissions

        Returns:
            True if user has permission
        """
        # Check role-based permissions
        role_perms = self.get_role_permissions(role)
        if permission in role_perms:
            return True

        # Check custom permissions for session
        if session_id and session_id in self.custom_permissions:
            user_perms = self.custom_permissions[session_id].get(user_id, set())
            if permission in user_perms:
                return True

        return False

    def check_permission(
        self,
        user_id: str,
        role: Role,
        permission: Permission,
        session_id: Optional[str] = None
    ) -> PermissionCheck:
        """
        Check permission with detailed result.

        Args:
            user_id: User ID
            role: User's role
            permission: Permission to check
            session_id: Optional session for custom permissions

        Returns:
            PermissionCheck result
        """
        if self.has_permission(user_id, role, permission, session_id):
            return PermissionCheck.allow()
        else:
            return PermissionCheck.deny(
                f"User {user_id} with role {role.value} does not have permission: {permission.value}"
            )

    def can_perform_action(
        self,
        user_id: str,
        role: Role,
        action: str,
        session_id: Optional[str] = None
    ) -> PermissionCheck:
        """
        Check if user can perform an action.

        Maps actions to permissions.

        Args:
            user_id: User ID
            role: User's role
            action: Action name
            session_id: Optional session ID

        Returns:
            PermissionCheck result
        """
        # Map actions to permissions
        action_permission_map = {
            "send_message": Permission.SEND_MESSAGE,
            "edit_message": Permission.EDIT_MESSAGE,
            "delete_message": Permission.DELETE_MESSAGE,
            "update_state": Permission.UPDATE_STATE,
            "clear_history": Permission.CLEAR_HISTORY,
            "invite_user": Permission.INVITE_USER,
            "remove_user": Permission.REMOVE_USER,
            "change_role": Permission.CHANGE_USER_ROLE,
            "edit_session": Permission.EDIT_SESSION,
            "delete_session": Permission.DELETE_SESSION,
            "transfer_ownership": Permission.TRANSFER_OWNERSHIP,
        }

        permission = action_permission_map.get(action)
        if permission is None:
            return PermissionCheck.deny(f"Unknown action: {action}")

        return self.check_permission(user_id, role, permission, session_id)

    def can_modify_user(
        self,
        actor_id: str,
        actor_role: Role,
        target_id: str,
        target_role: Role
    ) -> PermissionCheck:
        """
        Check if actor can modify target user.

        Rules:
        - Can only modify users with lower role
        - Cannot modify yourself (except owner can transfer ownership)
        - Must have CHANGE_USER_ROLE permission

        Args:
            actor_id: ID of user performing action
            actor_role: Role of actor
            target_id: ID of target user
            target_role: Role of target user

        Returns:
            PermissionCheck result
        """
        # Check base permission
        if not self.has_permission(actor_id, actor_role, Permission.CHANGE_USER_ROLE):
            return PermissionCheck.deny(
                f"User {actor_id} does not have permission to change user roles"
            )

        # Cannot modify users with equal or higher role
        if target_role >= actor_role:
            return PermissionCheck.deny(
                f"Cannot modify user with role {target_role.value} "
                f"(actor role: {actor_role.value})"
            )

        return PermissionCheck.allow()

    def grant_permission(
        self,
        session_id: str,
        user_id: str,
        permission: Permission
    ):
        """
        Grant custom permission to user in session.

        Args:
            session_id: Session ID
            user_id: User ID
            permission: Permission to grant
        """
        if session_id not in self.custom_permissions:
            self.custom_permissions[session_id] = {}

        if user_id not in self.custom_permissions[session_id]:
            self.custom_permissions[session_id][user_id] = set()

        self.custom_permissions[session_id][user_id].add(permission)

        logger.info(
            f"Granted {permission.value} to user {user_id} in session {session_id}"
        )

    def revoke_permission(
        self,
        session_id: str,
        user_id: str,
        permission: Permission
    ):
        """
        Revoke custom permission from user in session.

        Args:
            session_id: Session ID
            user_id: User ID
            permission: Permission to revoke
        """
        if session_id not in self.custom_permissions:
            return

        if user_id not in self.custom_permissions[session_id]:
            return

        self.custom_permissions[session_id][user_id].discard(permission)

        logger.info(
            f"Revoked {permission.value} from user {user_id} in session {session_id}"
        )

    def get_user_permissions(
        self,
        user_id: str,
        role: Role,
        session_id: Optional[str] = None
    ) -> Set[Permission]:
        """
        Get all permissions for user.

        Args:
            user_id: User ID
            role: User's role
            session_id: Optional session ID

        Returns:
            Set of permissions
        """
        # Start with role permissions
        permissions = self.get_role_permissions(role).copy()

        # Add custom permissions
        if session_id and session_id in self.custom_permissions:
            user_perms = self.custom_permissions[session_id].get(user_id, set())
            permissions.update(user_perms)

        return permissions

    def validate_role_change(
        self,
        actor_id: str,
        actor_role: Role,
        target_id: str,
        current_role: Role,
        new_role: Role
    ) -> PermissionCheck:
        """
        Validate a role change request.

        Args:
            actor_id: User making the change
            actor_role: Actor's role
            target_id: User being changed
            current_role: Target's current role
            new_role: Requested new role

        Returns:
            PermissionCheck result
        """
        # Check if actor can modify target
        can_modify = self.can_modify_user(actor_id, actor_role, target_id, current_role)
        if not can_modify.allowed:
            return can_modify

        # Cannot promote to role >= actor's role (except owner)
        if new_role >= actor_role and actor_role != Role.OWNER:
            return PermissionCheck.deny(
                f"Cannot promote user to role {new_role.value} "
                f"(actor role: {actor_role.value})"
            )

        # Owner -> non-owner requires transfer ownership permission
        if current_role == Role.OWNER and new_role != Role.OWNER:
            if not self.has_permission(actor_id, actor_role, Permission.TRANSFER_OWNERSHIP):
                return PermissionCheck.deny(
                    "Changing owner role requires TRANSFER_OWNERSHIP permission"
                )

        return PermissionCheck.allow()


# ============================================================================
# Middleware/Decorators
# ============================================================================

def require_permission(permission: Permission):
    """
    Decorator for functions requiring specific permission.

    Usage:
        @require_permission(Permission.SEND_MESSAGE)
        async def send_message(user_id: str, role: Role, message: str):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user_id and role from args/kwargs
            user_id = kwargs.get('user_id') or (args[0] if len(args) > 0 else None)
            role = kwargs.get('role') or (args[1] if len(args) > 1 else None)

            if not user_id or not role:
                raise ValueError("Function must have user_id and role parameters")

            # Check permission
            manager = PermissionManager()
            if not manager.has_permission(user_id, role, permission):
                raise PermissionError(
                    f"User {user_id} with role {role.value} does not have permission: {permission.value}"
                )

            # Execute function
            return await func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Permission System")
    print("="*80 + "\n")

    manager = PermissionManager()

    # Define users
    print("1. User roles:")
    users = {
        "alice": Role.OWNER,
        "bob": Role.ADMIN,
        "charlie": Role.EDITOR,
        "diana": Role.VIEWER
    }

    for user, role in users.items():
        print(f"   • {user}: {role.value}")

    # Check permissions
    print("\n2. Permission checks:")

    checks = [
        ("alice", Role.OWNER, Permission.DELETE_SESSION, True),
        ("bob", Role.ADMIN, Permission.INVITE_USER, True),
        ("charlie", Role.EDITOR, Permission.SEND_MESSAGE, True),
        ("diana", Role.VIEWER, Permission.EDIT_MESSAGE, False),
        ("diana", Role.VIEWER, Permission.READ_HISTORY, True),
    ]

    for user, role, perm, expected in checks:
        result = manager.has_permission(user, role, perm)
        symbol = "✓" if result == expected else "✗"
        print(f"   {symbol} {user} can {perm.value}: {result}")

    # Role comparison
    print("\n3. Role hierarchy:")
    print(f"   • VIEWER < EDITOR: {Role.VIEWER < Role.EDITOR}")
    print(f"   • EDITOR < ADMIN: {Role.EDITOR < Role.ADMIN}")
    print(f"   • ADMIN < OWNER: {Role.ADMIN < Role.OWNER}")

    # Modify user
    print("\n4. Role change validation:")
    changes = [
        ("bob", Role.ADMIN, "charlie", Role.EDITOR, Role.VIEWER, True),   # Demote editor
        ("charlie", Role.EDITOR, "bob", Role.ADMIN, Role.VIEWER, False),  # Cannot modify higher role
        ("bob", Role.ADMIN, "diana", Role.VIEWER, Role.OWNER, False),     # Cannot promote to owner
    ]

    for actor, actor_role, target, current, new, expected in changes:
        result = manager.validate_role_change(actor, actor_role, target, current, new)
        symbol = "✓" if result.allowed == expected else "✗"
        print(f"   {symbol} {actor} change {target}: {current.value} -> {new.value}")
        if not result.allowed:
            print(f"      Reason: {result.reason}")

    # Custom permissions
    print("\n5. Custom permissions:")
    session_id = "test-session"
    manager.grant_permission(session_id, "diana", Permission.SEND_MESSAGE)
    result = manager.has_permission("diana", Role.VIEWER, Permission.SEND_MESSAGE, session_id)
    print(f"   ✓ Diana granted SEND_MESSAGE: {result}")

    # List permissions
    print("\n6. User permissions:")
    for user, role in users.items():
        perms = manager.get_user_permissions(user, role, session_id)
        print(f"   • {user} ({role.value}): {len(perms)} permissions")

    print("\n✓ Permission system ready!")

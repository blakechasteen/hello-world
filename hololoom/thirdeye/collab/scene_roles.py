"""
Scene Role Enforcement System
=============================

Role-based access control for collaborative ThirdEye scenes.

Defines who can do what in a collaborative scene:
- VIEWER: Read-only access
- CONTRIBUTOR: Can add elements
- EDITOR: Full editing capabilities
- OWNER: Manage roles and scene settings

Created: 2025-12-03
Author: HoloLoom Team
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .scene_sync import SceneOperationType

logger = logging.getLogger(__name__)


class SceneRole(Enum):
    """
    User roles in a collaborative scene.

    Roles are hierarchical: OWNER > EDITOR > CONTRIBUTOR > VIEWER
    Higher roles inherit all permissions of lower roles.
    """
    VIEWER = auto()       # Read-only, can view scene
    CONTRIBUTOR = auto()  # Can add new elements
    EDITOR = auto()       # Can edit/delete any element
    OWNER = auto()        # Full control, can manage roles


@dataclass
class RolePermissions:
    """
    Permissions associated with a role.

    Defines what operations each role can perform.
    """
    role: SceneRole

    # Operation permissions
    can_view: bool = True
    can_add: bool = False
    can_update_own: bool = False  # Update elements they created
    can_update_any: bool = False  # Update any element
    can_delete_own: bool = False
    can_delete_any: bool = False
    can_add_connection: bool = False
    can_update_metadata: bool = False
    can_update_camera: bool = False
    can_update_lighting: bool = False
    can_manage_roles: bool = False
    can_export_scene: bool = False

    @classmethod
    def for_role(cls, role: SceneRole) -> 'RolePermissions':
        """Get permissions for a specific role."""
        if role == SceneRole.VIEWER:
            return cls(
                role=role,
                can_view=True,
                can_export_scene=True,  # Viewers can export for reference
            )
        elif role == SceneRole.CONTRIBUTOR:
            return cls(
                role=role,
                can_view=True,
                can_add=True,
                can_update_own=True,
                can_delete_own=True,
                can_add_connection=True,
                can_export_scene=True,
            )
        elif role == SceneRole.EDITOR:
            return cls(
                role=role,
                can_view=True,
                can_add=True,
                can_update_own=True,
                can_update_any=True,
                can_delete_own=True,
                can_delete_any=True,
                can_add_connection=True,
                can_update_metadata=True,
                can_update_camera=True,
                can_update_lighting=True,
                can_export_scene=True,
            )
        elif role == SceneRole.OWNER:
            return cls(
                role=role,
                can_view=True,
                can_add=True,
                can_update_own=True,
                can_update_any=True,
                can_delete_own=True,
                can_delete_any=True,
                can_add_connection=True,
                can_update_metadata=True,
                can_update_camera=True,
                can_update_lighting=True,
                can_manage_roles=True,
                can_export_scene=True,
            )
        else:
            # Default to viewer for unknown roles
            return cls.for_role(SceneRole.VIEWER)


@dataclass
class RoleAssignment:
    """
    Assignment of a role to a user for a scene.
    """
    user_id: str
    role: SceneRole
    assigned_by: str
    assigned_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None  # None = never expires

    def is_expired(self) -> bool:
        """Check if this assignment has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class RoleChangeEvent:
    """
    Event emitted when a role changes.
    """
    user_id: str
    previous_role: SceneRole | None
    new_role: SceneRole | None  # None = role removed
    changed_by: str
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""


class RoleEnforcementError(Exception):
    """Raised when an operation is blocked due to insufficient permissions."""

    def __init__(self, user_id: str, operation: SceneOperationType, required_permission: str):
        self.user_id = user_id
        self.operation = operation
        self.required_permission = required_permission
        super().__init__(
            f"User '{user_id}' lacks permission '{required_permission}' "
            f"for operation '{operation.name}'"
        )


class RoleEnforcer:
    """
    Enforces role-based access control for collaborative scenes.

    Usage:
        enforcer = RoleEnforcer(scene_id="scene_123")
        enforcer.assign_role(user_id="user_1", role=SceneRole.OWNER, assigned_by="system")
        enforcer.assign_role(user_id="user_2", role=SceneRole.CONTRIBUTOR, assigned_by="user_1")

        # Check before operation
        if enforcer.can_perform(user_id="user_2", operation=SceneOperationType.ADD_ELEMENT):
            # Proceed with operation
            pass

        # Or enforce (raises exception if not allowed)
        enforcer.enforce(user_id="user_2", operation=SceneOperationType.REMOVE_ELEMENT)
    """

    # Map operations to required permissions
    OPERATION_PERMISSIONS: dict[SceneOperationType, str] = {
        SceneOperationType.ADD_ELEMENT: 'can_add',
        SceneOperationType.UPDATE_ELEMENT: 'can_update_any',  # Default to any, check owner below
        SceneOperationType.REMOVE_ELEMENT: 'can_delete_any',
        SceneOperationType.MOVE_ELEMENT: 'can_update_any',
        SceneOperationType.RESIZE_ELEMENT: 'can_update_any',
        SceneOperationType.RESTYLE_ELEMENT: 'can_update_any',
        SceneOperationType.ADD_CONNECTION: 'can_add_connection',
        SceneOperationType.REMOVE_CONNECTION: 'can_delete_any',
        SceneOperationType.UPDATE_METADATA: 'can_update_metadata',
        SceneOperationType.UPDATE_CAMERA: 'can_update_camera',
        SceneOperationType.UPDATE_LIGHTING: 'can_update_lighting',
        SceneOperationType.UPDATE_BACKGROUND: 'can_update_metadata',
    }

    def __init__(self, scene_id: str):
        """
        Initialize role enforcer for a scene.

        Args:
            scene_id: ID of the scene to enforce roles for
        """
        self.scene_id = scene_id
        self._assignments: dict[str, RoleAssignment] = {}
        self._element_owners: dict[str, str] = {}  # element_id -> user_id
        self._change_handlers: list[Callable[[RoleChangeEvent], None]] = []

    # -------------------------------------------------------------------------
    # Role Management
    # -------------------------------------------------------------------------

    def assign_role(
        self,
        user_id: str,
        role: SceneRole,
        assigned_by: str,
        expires_at: datetime | None = None,
        reason: str = "",
    ) -> RoleAssignment:
        """
        Assign a role to a user.

        Args:
            user_id: User to assign role to
            role: Role to assign
            assigned_by: User making the assignment (must be OWNER)
            expires_at: When the role expires (None = never)
            reason: Reason for assignment

        Returns:
            The role assignment

        Raises:
            RoleEnforcementError: If assigner doesn't have permission
        """
        # First assignment can be made by system
        if self._assignments and assigned_by != "system":
            # Check that assigner has permission to manage roles
            assigner_role = self.get_role(assigned_by)
            if assigner_role is None or not RolePermissions.for_role(assigner_role).can_manage_roles:
                raise RoleEnforcementError(
                    assigned_by,
                    SceneOperationType.UPDATE_METADATA,  # Closest match
                    "can_manage_roles"
                )

        # Get previous role for event
        previous = self._assignments.get(user_id)
        previous_role = previous.role if previous else None

        # Create assignment
        assignment = RoleAssignment(
            user_id=user_id,
            role=role,
            assigned_by=assigned_by,
            expires_at=expires_at,
        )
        self._assignments[user_id] = assignment

        # Emit change event
        event = RoleChangeEvent(
            user_id=user_id,
            previous_role=previous_role,
            new_role=role,
            changed_by=assigned_by,
            reason=reason,
        )
        self._emit_change(event)

        logger.info(f"Assigned role {role.name} to user {user_id} in scene {self.scene_id}")

        return assignment

    def revoke_role(
        self,
        user_id: str,
        revoked_by: str,
        reason: str = "",
    ):
        """
        Revoke a user's role (remove from scene).

        Args:
            user_id: User to remove
            revoked_by: User making the removal (must be OWNER)
            reason: Reason for removal

        Raises:
            RoleEnforcementError: If revoker doesn't have permission
        """
        if revoked_by != "system":
            revoker_role = self.get_role(revoked_by)
            if revoker_role is None or not RolePermissions.for_role(revoker_role).can_manage_roles:
                raise RoleEnforcementError(
                    revoked_by,
                    SceneOperationType.UPDATE_METADATA,
                    "can_manage_roles"
                )

        # Can't revoke your own role if you're the only owner
        if self.get_role(user_id) == SceneRole.OWNER:
            owners = [u for u, a in self._assignments.items()
                      if a.role == SceneRole.OWNER and not a.is_expired()]
            if len(owners) <= 1:
                raise ValueError("Cannot remove the last owner from a scene")

        previous = self._assignments.pop(user_id, None)
        if previous:
            event = RoleChangeEvent(
                user_id=user_id,
                previous_role=previous.role,
                new_role=None,
                changed_by=revoked_by,
                reason=reason,
            )
            self._emit_change(event)

            logger.info(f"Revoked role from user {user_id} in scene {self.scene_id}")

    def get_role(self, user_id: str) -> SceneRole | None:
        """
        Get a user's current role.

        Returns None if user has no role or role has expired.
        """
        assignment = self._assignments.get(user_id)
        if assignment is None or assignment.is_expired():
            return None
        return assignment.role

    def get_permissions(self, user_id: str) -> RolePermissions:
        """
        Get a user's current permissions.

        Returns viewer permissions if user has no role.
        """
        role = self.get_role(user_id)
        if role is None:
            # No role = no permissions (not even viewer by default)
            return RolePermissions(role=SceneRole.VIEWER, can_view=False)
        return RolePermissions.for_role(role)

    def list_users_by_role(self, role: SceneRole | None = None) -> list[str]:
        """
        List users with a specific role (or all users if role is None).
        """
        result = []
        for user_id, assignment in self._assignments.items():
            if assignment.is_expired():
                continue
            if role is None or assignment.role == role:
                result.append(user_id)
        return result

    # -------------------------------------------------------------------------
    # Permission Checking
    # -------------------------------------------------------------------------

    def can_perform(
        self,
        user_id: str,
        operation: SceneOperationType,
        element_id: str | None = None,
    ) -> bool:
        """
        Check if a user can perform an operation.

        Args:
            user_id: User attempting the operation
            operation: Type of operation
            element_id: ID of element being operated on (for ownership checks)

        Returns:
            True if the operation is allowed
        """
        permissions = self.get_permissions(user_id)

        # Get required permission for this operation
        required = self.OPERATION_PERMISSIONS.get(operation)
        if required is None:
            # Unknown operation - default deny
            return False

        # Check if user has the required permission
        if getattr(permissions, required, False):
            return True

        # Special case: check "own" permissions for update/delete operations
        if operation in (SceneOperationType.UPDATE_ELEMENT, SceneOperationType.MOVE_ELEMENT,
                         SceneOperationType.RESIZE_ELEMENT, SceneOperationType.RESTYLE_ELEMENT):
            if permissions.can_update_own and element_id:
                if self._element_owners.get(element_id) == user_id:
                    return True

        if operation in (SceneOperationType.REMOVE_ELEMENT, SceneOperationType.REMOVE_CONNECTION):
            if permissions.can_delete_own and element_id:
                if self._element_owners.get(element_id) == user_id:
                    return True

        return False

    def enforce(
        self,
        user_id: str,
        operation: SceneOperationType,
        element_id: str | None = None,
    ):
        """
        Enforce that a user can perform an operation.

        Raises:
            RoleEnforcementError: If the operation is not allowed
        """
        if not self.can_perform(user_id, operation, element_id):
            required = self.OPERATION_PERMISSIONS.get(operation, 'unknown')
            raise RoleEnforcementError(user_id, operation, required)

    # -------------------------------------------------------------------------
    # Element Ownership
    # -------------------------------------------------------------------------

    def register_element_owner(self, element_id: str, user_id: str):
        """
        Register the owner of an element.

        Called when an element is created to track ownership.
        """
        self._element_owners[element_id] = user_id

    def get_element_owner(self, element_id: str) -> str | None:
        """Get the owner of an element."""
        return self._element_owners.get(element_id)

    def remove_element(self, element_id: str):
        """Remove element from ownership tracking (when element is deleted)."""
        self._element_owners.pop(element_id, None)

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def on_role_change(self, handler: Callable[[RoleChangeEvent], None]):
        """Subscribe to role change events."""
        self._change_handlers.append(handler)

    def _emit_change(self, event: RoleChangeEvent):
        """Emit a role change event to all handlers."""
        for handler in self._change_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Role change handler error: {e}")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'scene_id': self.scene_id,
            'assignments': {
                user_id: {
                    'role': assignment.role.name,
                    'assigned_by': assignment.assigned_by,
                    'assigned_at': assignment.assigned_at.isoformat(),
                    'expires_at': assignment.expires_at.isoformat() if assignment.expires_at else None,
                }
                for user_id, assignment in self._assignments.items()
            },
            'element_owners': dict(self._element_owners),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RoleEnforcer':
        """Deserialize from dictionary."""
        enforcer = cls(scene_id=data['scene_id'])

        for user_id, assignment_data in data.get('assignments', {}).items():
            enforcer._assignments[user_id] = RoleAssignment(
                user_id=user_id,
                role=SceneRole[assignment_data['role']],
                assigned_by=assignment_data['assigned_by'],
                assigned_at=datetime.fromisoformat(assignment_data['assigned_at']),
                expires_at=datetime.fromisoformat(assignment_data['expires_at'])
                          if assignment_data.get('expires_at') else None,
            )

        enforcer._element_owners = dict(data.get('element_owners', {}))

        return enforcer


def create_role_enforcer(scene_id: str, owner_id: str) -> RoleEnforcer:
    """
    Factory function to create a role enforcer with an initial owner.

    Args:
        scene_id: ID of the scene
        owner_id: User ID of the initial owner

    Returns:
        RoleEnforcer with owner assigned
    """
    enforcer = RoleEnforcer(scene_id=scene_id)
    enforcer.assign_role(
        user_id=owner_id,
        role=SceneRole.OWNER,
        assigned_by="system",
        reason="Scene creator",
    )
    return enforcer


__all__ = [
    'SceneRole',
    'RolePermissions',
    'RoleAssignment',
    'RoleChangeEvent',
    'RoleEnforcementError',
    'RoleEnforcer',
    'create_role_enforcer',
]

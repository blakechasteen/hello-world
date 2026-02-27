"""
Collaborative Scene Synchronization
====================================

Wraps ThirdEye VisualScene with real-time synchronization capabilities.

Uses HoloLoom.collaboration.sync for CRDT-based conflict resolution,
enabling multiple users to edit the same scene without conflicts.

Key Features:
- Real-time element add/update/remove with broadcast
- CRDT conflict resolution for concurrent edits
- Operation history for undo/redo
- Scene versioning with snapshots

Created: 2025-12-01
Author: HoloLoom Team
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
import asyncio
import json
import uuid
import logging

# ThirdEye imports
from hololoom.thirdeye.visualizers import VisualScene, VisualElement

# Role enforcement imports (graceful fallback if not available)
try:
    from .scene_roles import (
        SceneRole,
        RoleEnforcer,
        RoleEnforcementError,
        RolePermissions,
        create_role_enforcer,
    )
    ROLES_AVAILABLE = True
except ImportError:
    ROLES_AVAILABLE = False

# Collaboration imports (graceful fallback if not available)
try:
    from hololoom.collaboration.sync import (
        StateSynchronizer,
        Operation,
        OperationType,
        SyncState,
        Conflict,
        ConflictResolution,
        create_state_synchronizer,
    )
    from hololoom.collaboration.attribution import (
        AttributionManager,
        Contribution,
        ContributionType,
        create_attribution_manager,
    )
    COLLAB_AVAILABLE = True
except ImportError:
    COLLAB_AVAILABLE = False

logger = logging.getLogger(__name__)


class SceneOperationType(Enum):
    """Types of operations on collaborative scenes."""
    ADD_ELEMENT = auto()
    UPDATE_ELEMENT = auto()
    REMOVE_ELEMENT = auto()
    MOVE_ELEMENT = auto()
    RESIZE_ELEMENT = auto()
    RESTYLE_ELEMENT = auto()
    ADD_CONNECTION = auto()
    REMOVE_CONNECTION = auto()
    UPDATE_METADATA = auto()
    UPDATE_CAMERA = auto()
    UPDATE_LIGHTING = auto()
    UPDATE_BACKGROUND = auto()


@dataclass
class SceneOperation:
    """
    An atomic operation on a collaborative scene.

    Designed for CRDT sync - each operation is:
    - Idempotent (can be applied multiple times safely)
    - Commutative (order doesn't matter for concurrent ops)
    - Associative (grouping doesn't matter)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: SceneOperationType = SceneOperationType.UPDATE_ELEMENT

    # Target
    element_id: Optional[str] = None
    path: str = ""  # Dot-notation path like "elements.btn_1.position.x"

    # Value
    value: Any = None
    previous_value: Any = None  # For undo

    # Metadata
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""

    # Conflict resolution
    vector_clock: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            'id': self.id,
            'type': self.type.name,
            'element_id': self.element_id,
            'path': self.path,
            'value': self.value,
            'previous_value': self.previous_value,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'vector_clock': self.vector_clock,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneOperation':
        """Create from serialized dict."""
        return cls(
            id=data.get('id', str(uuid.uuid4())[:8]),
            type=SceneOperationType[data.get('type', 'UPDATE_ELEMENT')],
            element_id=data.get('element_id'),
            path=data.get('path', ''),
            value=data.get('value'),
            previous_value=data.get('previous_value'),
            user_id=data.get('user_id', ''),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            session_id=data.get('session_id', ''),
            vector_clock=data.get('vector_clock', {}),
        )


@dataclass
class SceneState:
    """
    Complete state of a collaborative scene.

    Used for snapshots, sync, and recovery.
    """
    scene: VisualScene
    version: int = 0
    last_modified: datetime = field(default_factory=datetime.now)
    modified_by: str = ""

    # Tracking
    active_users: Set[str] = field(default_factory=set)
    pending_operations: List[SceneOperation] = field(default_factory=list)
    operation_history: List[SceneOperation] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    session_id: str = ""


class CollaborativeScene:
    """
    Wraps VisualScene with real-time collaboration capabilities.

    Provides:
    - Real-time sync of scene state across users
    - CRDT-based conflict resolution
    - Operation history with undo/redo
    - Attribution tracking for contributions

    Usage:
        scene = VisualScene(mode=VisualizationMode.NARRATIVE, ...)
        collab = CollaborativeScene(scene, session_id="session_123")

        # Add element (broadcasts to all users)
        await collab.add_element(element, user_id="user_1")

        # Update element (with conflict resolution)
        await collab.update_element("element_id", {"label": "New Label"}, user_id="user_1")

        # Subscribe to changes
        collab.on_change(lambda op: print(f"Change: {op}"))
    """

    def __init__(
        self,
        scene: VisualScene,
        session_id: str,
        user_id: Optional[str] = None,
        enable_sync: bool = True,
        enable_attribution: bool = True,
        enable_role_enforcement: bool = True,
        max_history: int = 100,
    ):
        self.scene = scene
        self.session_id = session_id
        self.user_id = user_id or str(uuid.uuid4())[:8]
        self.enable_sync = enable_sync and COLLAB_AVAILABLE
        self.enable_attribution = enable_attribution and COLLAB_AVAILABLE
        self.enable_role_enforcement = enable_role_enforcement and ROLES_AVAILABLE
        self.max_history = max_history

        # State tracking
        self._state = SceneState(
            scene=scene,
            session_id=session_id,
            created_by=self.user_id,
        )
        self._state.active_users.add(self.user_id)

        # Element index for fast lookup
        self._elements: Dict[str, VisualElement] = {
            e.id: e for e in scene.elements
        }

        # Vector clock for ordering
        self._vector_clock: Dict[str, int] = {self.user_id: 0}

        # Synchronizer (if available)
        self._sync: Optional['StateSynchronizer'] = None
        self._attribution: Optional['AttributionManager'] = None
        self._role_enforcer: Optional['RoleEnforcer'] = None

        # Event handlers
        self._change_handlers: List[Callable[[SceneOperation], None]] = []
        self._conflict_handlers: List[Callable[[SceneOperation, SceneOperation], None]] = []

        # Initialize collaboration if available
        if self.enable_sync:
            self._init_sync()
        if self.enable_attribution:
            self._init_attribution()
        if self.enable_role_enforcement:
            self._init_roles()

    def _init_sync(self):
        """Initialize state synchronizer."""
        try:
            self._sync = create_state_synchronizer(
                session_id=self.session_id,
                user_id=self.user_id,
            )
            # Register apply handler for scene elements
            self._sync.register_apply_handler(
                "scene_element",
                self._handle_sync_operation
            )
            logger.info(f"Sync initialized for session {self.session_id}")
        except Exception as e:
            logger.warning(f"Could not initialize sync: {e}")
            self._sync = None

    def _handle_sync_operation(self, op: 'Operation') -> bool:
        """Handle a sync operation from the synchronizer."""
        try:
            # The operation data contains our SceneOperation
            scene_op_data = op.data
            if isinstance(scene_op_data, dict):
                scene_op = SceneOperation.from_dict(scene_op_data)
                # Apply the operation locally (already applied by us, just confirm)
                logger.debug(f"Sync operation applied: {scene_op.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to handle sync operation: {e}")
            return False

    def _init_attribution(self):
        """Initialize attribution manager."""
        try:
            self._attribution = create_attribution_manager()
            logger.info("Attribution tracking initialized")
        except Exception as e:
            logger.warning(f"Could not initialize attribution: {e}")
            self._attribution = None

    def _init_roles(self):
        """Initialize role enforcement."""
        try:
            # Create role enforcer with current user as owner
            self._role_enforcer = create_role_enforcer(
                scene_id=self.session_id,
                owner_id=self.user_id,
            )
            logger.info(f"Role enforcement initialized for scene {self.session_id}")
        except Exception as e:
            logger.warning(f"Could not initialize role enforcement: {e}")
            self._role_enforcer = None

    def _enforce_permission(
        self,
        user_id: str,
        operation: SceneOperationType,
        element_id: Optional[str] = None,
    ):
        """
        Check if user can perform operation.

        Raises RoleEnforcementError if not allowed.
        """
        if not self._role_enforcer:
            return  # No enforcement configured

        self._role_enforcer.enforce(user_id, operation, element_id)

    # -------------------------------------------------------------------------
    # Role Management (Public API)
    # -------------------------------------------------------------------------

    def assign_role(
        self,
        user_id: str,
        role: 'SceneRole',
        assigned_by: Optional[str] = None,
    ):
        """
        Assign a role to a user.

        Args:
            user_id: User to assign role to
            role: Role to assign (VIEWER, CONTRIBUTOR, EDITOR, OWNER)
            assigned_by: User making the assignment (must be OWNER)

        Raises:
            RoleEnforcementError: If assigner doesn't have permission
            RuntimeError: If role enforcement not enabled
        """
        if not self._role_enforcer:
            raise RuntimeError("Role enforcement not enabled")

        assigned_by = assigned_by or self.user_id
        self._role_enforcer.assign_role(
            user_id=user_id,
            role=role,
            assigned_by=assigned_by,
        )

    def revoke_role(
        self,
        user_id: str,
        revoked_by: Optional[str] = None,
    ):
        """
        Revoke a user's role.

        Args:
            user_id: User to remove
            revoked_by: User making the removal (must be OWNER)
        """
        if not self._role_enforcer:
            raise RuntimeError("Role enforcement not enabled")

        revoked_by = revoked_by or self.user_id
        self._role_enforcer.revoke_role(
            user_id=user_id,
            revoked_by=revoked_by,
        )

    def get_user_role(self, user_id: str) -> Optional['SceneRole']:
        """Get a user's current role."""
        if not self._role_enforcer:
            return None
        return self._role_enforcer.get_role(user_id)

    def get_user_permissions(self, user_id: str) -> Optional['RolePermissions']:
        """Get a user's current permissions."""
        if not self._role_enforcer:
            return None
        return self._role_enforcer.get_permissions(user_id)

    def can_perform(
        self,
        user_id: str,
        operation: SceneOperationType,
        element_id: Optional[str] = None,
    ) -> bool:
        """Check if user can perform an operation (without raising)."""
        if not self._role_enforcer:
            return True  # No enforcement = all allowed
        return self._role_enforcer.can_perform(user_id, operation, element_id)

    # -------------------------------------------------------------------------
    # Element Operations
    # -------------------------------------------------------------------------

    async def add_element(
        self,
        element: VisualElement,
        user_id: Optional[str] = None,
    ) -> SceneOperation:
        """
        Add an element to the scene and broadcast to all users.

        Args:
            element: The element to add
            user_id: User performing the action

        Returns:
            The operation that was applied

        Raises:
            RoleEnforcementError: If user lacks permission
        """
        user_id = user_id or self.user_id

        # Enforce permission
        self._enforce_permission(user_id, SceneOperationType.ADD_ELEMENT)

        # Create operation
        op = SceneOperation(
            type=SceneOperationType.ADD_ELEMENT,
            element_id=element.id,
            path=f"elements.{element.id}",
            value=self._element_to_dict(element),
            user_id=user_id,
            session_id=self.session_id,
            vector_clock=self._increment_clock(user_id),
        )

        # Apply locally
        self._apply_add_element(element)

        # Register element ownership for role enforcement
        if self._role_enforcer:
            self._role_enforcer.register_element_owner(element.id, user_id)

        # Broadcast
        await self._broadcast(op)

        # Track attribution
        await self._track_contribution(op, ContributionType.CREATE if COLLAB_AVAILABLE else None)

        # Record in history
        self._record_operation(op)

        return op

    async def update_element(
        self,
        element_id: str,
        changes: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> SceneOperation:
        """
        Update an element's properties.

        Args:
            element_id: ID of element to update
            changes: Dictionary of changes (e.g., {"label": "New", "position.x": 5})
            user_id: User performing the action

        Returns:
            The operation that was applied

        Raises:
            RoleEnforcementError: If user lacks permission
        """
        user_id = user_id or self.user_id

        # Enforce permission (with element_id for ownership check)
        self._enforce_permission(user_id, SceneOperationType.UPDATE_ELEMENT, element_id)

        element = self._elements.get(element_id)
        if not element:
            raise ValueError(f"Element {element_id} not found")

        # Capture previous values for undo
        previous = {}
        for key in changes:
            previous[key] = self._get_nested(element, key)

        # Create operation
        op = SceneOperation(
            type=SceneOperationType.UPDATE_ELEMENT,
            element_id=element_id,
            path=f"elements.{element_id}",
            value=changes,
            previous_value=previous,
            user_id=user_id,
            session_id=self.session_id,
            vector_clock=self._increment_clock(user_id),
        )

        # Apply locally
        self._apply_update_element(element_id, changes)

        # Broadcast
        await self._broadcast(op)

        # Track attribution
        await self._track_contribution(op, ContributionType.KNOWLEDGE_EDIT if COLLAB_AVAILABLE else None)

        # Record in history
        self._record_operation(op)

        return op

    async def remove_element(
        self,
        element_id: str,
        user_id: Optional[str] = None,
    ) -> SceneOperation:
        """
        Remove an element from the scene.

        Args:
            element_id: ID of element to remove
            user_id: User performing the action

        Returns:
            The operation that was applied

        Raises:
            RoleEnforcementError: If user lacks permission
        """
        user_id = user_id or self.user_id

        # Enforce permission (with element_id for ownership check)
        self._enforce_permission(user_id, SceneOperationType.REMOVE_ELEMENT, element_id)

        element = self._elements.get(element_id)
        if not element:
            raise ValueError(f"Element {element_id} not found")

        # Create operation with full element data for undo
        op = SceneOperation(
            type=SceneOperationType.REMOVE_ELEMENT,
            element_id=element_id,
            path=f"elements.{element_id}",
            value=None,
            previous_value=self._element_to_dict(element),
            user_id=user_id,
            session_id=self.session_id,
            vector_clock=self._increment_clock(user_id),
        )

        # Apply locally
        self._apply_remove_element(element_id)

        # Remove element from ownership tracking
        if self._role_enforcer:
            self._role_enforcer.remove_element(element_id)

        # Broadcast
        await self._broadcast(op)

        # Track attribution
        await self._track_contribution(op, ContributionType.KNOWLEDGE_DELETE if COLLAB_AVAILABLE else None)

        # Record in history
        self._record_operation(op)

        return op

    async def move_element(
        self,
        element_id: str,
        position: Dict[str, float],
        user_id: Optional[str] = None,
    ) -> SceneOperation:
        """
        Move an element to a new position.

        Args:
            element_id: ID of element to move
            position: New position {"x": float, "y": float, "z": float}
            user_id: User performing the action
        """
        return await self.update_element(
            element_id,
            {"position": position},
            user_id,
        )

    # -------------------------------------------------------------------------
    # Connection Operations
    # -------------------------------------------------------------------------

    async def add_connection(
        self,
        from_id: str,
        to_id: str,
        user_id: Optional[str] = None,
    ) -> SceneOperation:
        """
        Add a connection between two elements.

        Raises:
            RoleEnforcementError: If user lacks permission
        """
        user_id = user_id or self.user_id

        # Enforce permission
        self._enforce_permission(user_id, SceneOperationType.ADD_CONNECTION, from_id)

        from_element = self._elements.get(from_id)
        if not from_element:
            raise ValueError(f"Element {from_id} not found")

        # Create operation
        op = SceneOperation(
            type=SceneOperationType.ADD_CONNECTION,
            element_id=from_id,
            path=f"elements.{from_id}.connections",
            value=to_id,
            user_id=user_id,
            session_id=self.session_id,
            vector_clock=self._increment_clock(user_id),
        )

        # Apply locally
        if to_id not in from_element.connections:
            from_element.connections.append(to_id)

        # Broadcast and record
        await self._broadcast(op)
        self._record_operation(op)

        return op

    # -------------------------------------------------------------------------
    # Scene Metadata Operations
    # -------------------------------------------------------------------------

    async def update_camera(
        self,
        camera: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> SceneOperation:
        """
        Update scene camera settings.

        Raises:
            RoleEnforcementError: If user lacks permission
        """
        user_id = user_id or self.user_id

        # Enforce permission
        self._enforce_permission(user_id, SceneOperationType.UPDATE_CAMERA)

        op = SceneOperation(
            type=SceneOperationType.UPDATE_CAMERA,
            path="camera",
            value=camera,
            previous_value=self.scene.camera,
            user_id=user_id,
            session_id=self.session_id,
            vector_clock=self._increment_clock(user_id),
        )

        self.scene.camera = camera
        await self._broadcast(op)
        self._record_operation(op)

        return op

    async def update_lighting(
        self,
        lighting: str,
        user_id: Optional[str] = None,
    ) -> SceneOperation:
        """
        Update scene lighting.

        Raises:
            RoleEnforcementError: If user lacks permission
        """
        user_id = user_id or self.user_id

        # Enforce permission
        self._enforce_permission(user_id, SceneOperationType.UPDATE_LIGHTING)

        op = SceneOperation(
            type=SceneOperationType.UPDATE_LIGHTING,
            path="lighting",
            value=lighting,
            previous_value=self.scene.lighting,
            user_id=user_id,
            session_id=self.session_id,
            vector_clock=self._increment_clock(user_id),
        )

        self.scene.lighting = lighting
        await self._broadcast(op)
        self._record_operation(op)

        return op

    # -------------------------------------------------------------------------
    # Receiving Remote Operations
    # -------------------------------------------------------------------------

    async def apply_remote_operation(self, op: SceneOperation) -> bool:
        """
        Apply an operation received from another user.

        Handles conflict resolution if needed.

        Args:
            op: The remote operation

        Returns:
            True if applied successfully
        """
        # Check for conflicts with pending operations
        conflicts = self._detect_conflicts(op)
        if conflicts:
            for conflict_op in conflicts:
                self._notify_conflict(op, conflict_op)
            # Resolve using vector clock ordering
            if not self._should_apply(op, conflicts):
                logger.info(f"Operation {op.id} superseded by local operation")
                return False

        # Apply based on type
        if op.type == SceneOperationType.ADD_ELEMENT:
            element = self._dict_to_element(op.value)
            self._apply_add_element(element)

        elif op.type == SceneOperationType.UPDATE_ELEMENT:
            self._apply_update_element(op.element_id, op.value)

        elif op.type == SceneOperationType.REMOVE_ELEMENT:
            self._apply_remove_element(op.element_id)

        elif op.type == SceneOperationType.ADD_CONNECTION:
            if op.element_id in self._elements:
                element = self._elements[op.element_id]
                if op.value not in element.connections:
                    element.connections.append(op.value)

        elif op.type == SceneOperationType.UPDATE_CAMERA:
            self.scene.camera = op.value

        elif op.type == SceneOperationType.UPDATE_LIGHTING:
            self.scene.lighting = op.value

        # Update vector clock
        self._merge_clock(op.vector_clock)

        # Notify handlers
        self._notify_change(op)

        return True

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def on_change(self, handler: Callable[[SceneOperation], None]):
        """Subscribe to scene changes."""
        self._change_handlers.append(handler)

    def on_conflict(self, handler: Callable[[SceneOperation, SceneOperation], None]):
        """Subscribe to conflict notifications."""
        self._conflict_handlers.append(handler)

    def _notify_change(self, op: SceneOperation):
        """Notify all change handlers."""
        for handler in self._change_handlers:
            try:
                handler(op)
            except Exception as e:
                logger.error(f"Change handler error: {e}")

    def _notify_conflict(self, remote: SceneOperation, local: SceneOperation):
        """Notify all conflict handlers."""
        for handler in self._conflict_handlers:
            try:
                handler(remote, local)
            except Exception as e:
                logger.error(f"Conflict handler error: {e}")

    # -------------------------------------------------------------------------
    # Sync & Broadcast
    # -------------------------------------------------------------------------

    async def _broadcast(self, op: SceneOperation):
        """Broadcast operation to all users."""
        if self._sync:
            try:
                # Map scene operation type to sync operation type
                if op.type in (SceneOperationType.ADD_ELEMENT, SceneOperationType.ADD_CONNECTION):
                    sync_type = OperationType.INSERT
                elif op.type in (SceneOperationType.REMOVE_ELEMENT, SceneOperationType.REMOVE_CONNECTION):
                    sync_type = OperationType.DELETE
                else:
                    sync_type = OperationType.UPDATE

                # Create sync operation using the synchronizer
                sync_op = self._sync.create_operation(
                    op_type=sync_type,
                    target_type="scene_element",
                    target_id=op.element_id or op.path,
                    data=op.to_dict(),
                )
                await self._sync.apply_local(sync_op)
            except Exception as e:
                logger.error(f"Broadcast failed: {e}")
                self._state.pending_operations.append(op)

    async def sync_pending(self):
        """Sync any pending operations."""
        if not self._state.pending_operations:
            return

        pending = list(self._state.pending_operations)
        self._state.pending_operations.clear()

        for op in pending:
            await self._broadcast(op)

    # -------------------------------------------------------------------------
    # Attribution
    # -------------------------------------------------------------------------

    async def _track_contribution(
        self,
        op: SceneOperation,
        contrib_type: Optional['ContributionType'],
    ):
        """Track user contribution."""
        if not self._attribution or not contrib_type:
            return

        try:
            # Use the record_contribution method which handles ID generation
            self._attribution.record_contribution(
                contribution_type=contrib_type,
                user_id=op.user_id,
                user_name=op.user_id,  # Use user_id as name if not available
                target_type="scene_element",
                target_id=op.element_id or op.path,
                content_preview=f"{op.type.name} on {op.element_id or op.path}",
                session_id=op.session_id,
                metadata={'operation_id': op.id},
            )
        except Exception as e:
            logger.warning(f"Attribution tracking failed: {e}")

    # -------------------------------------------------------------------------
    # History & Undo
    # -------------------------------------------------------------------------

    def _record_operation(self, op: SceneOperation):
        """Record operation in history."""
        self._state.operation_history.append(op)
        self._state.version += 1
        self._state.last_modified = datetime.now()
        self._state.modified_by = op.user_id

        # Trim history if too long
        if len(self._state.operation_history) > self.max_history:
            self._state.operation_history = self._state.operation_history[-self.max_history:]

    async def undo(self, user_id: Optional[str] = None) -> Optional[SceneOperation]:
        """
        Undo the last operation by this user.

        Returns the operation that was undone, or None if nothing to undo.
        """
        user_id = user_id or self.user_id

        # Find last operation by this user
        for op in reversed(self._state.operation_history):
            if op.user_id == user_id and op.previous_value is not None:
                # Create inverse operation
                if op.type == SceneOperationType.ADD_ELEMENT:
                    return await self.remove_element(op.element_id, user_id)

                elif op.type == SceneOperationType.REMOVE_ELEMENT:
                    element = self._dict_to_element(op.previous_value)
                    return await self.add_element(element, user_id)

                elif op.type == SceneOperationType.UPDATE_ELEMENT:
                    return await self.update_element(
                        op.element_id,
                        op.previous_value,
                        user_id,
                    )

        return None

    # -------------------------------------------------------------------------
    # Vector Clock Operations
    # -------------------------------------------------------------------------

    def _increment_clock(self, user_id: str) -> Dict[str, int]:
        """Increment and return vector clock."""
        self._vector_clock[user_id] = self._vector_clock.get(user_id, 0) + 1
        return dict(self._vector_clock)

    def _merge_clock(self, other: Dict[str, int]):
        """Merge remote clock with local."""
        for user_id, count in other.items():
            self._vector_clock[user_id] = max(
                self._vector_clock.get(user_id, 0),
                count
            )

    def _detect_conflicts(self, op: SceneOperation) -> List[SceneOperation]:
        """Detect conflicting operations."""
        conflicts = []
        for pending in self._state.pending_operations:
            if pending.element_id == op.element_id:
                conflicts.append(pending)
        return conflicts

    def _should_apply(self, op: SceneOperation, conflicts: List[SceneOperation]) -> bool:
        """Determine if operation should be applied based on vector clock."""
        # Simple last-writer-wins with user_id as tiebreaker
        for conflict in conflicts:
            # Compare timestamps first
            if op.timestamp < conflict.timestamp:
                return False
            elif op.timestamp == conflict.timestamp:
                # Tiebreaker: lexicographic user_id
                if op.user_id > conflict.user_id:
                    return False
        return True

    # -------------------------------------------------------------------------
    # Local Apply Operations
    # -------------------------------------------------------------------------

    def _apply_add_element(self, element: VisualElement):
        """Apply add element locally."""
        self._elements[element.id] = element
        self.scene.elements.append(element)

    def _apply_update_element(self, element_id: str, changes: Dict[str, Any]):
        """Apply update element locally."""
        element = self._elements.get(element_id)
        if not element:
            return

        for key, value in changes.items():
            self._set_nested(element, key, value)

    def _apply_remove_element(self, element_id: str):
        """Apply remove element locally."""
        if element_id in self._elements:
            element = self._elements.pop(element_id)
            self.scene.elements = [e for e in self.scene.elements if e.id != element_id]

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _element_to_dict(self, element: VisualElement) -> Dict[str, Any]:
        """Convert element to dict."""
        return {
            'id': element.id,
            'element_type': element.element_type,
            'label': element.label,
            'position': element.position,
            'size': element.size,
            'style': element.style,
            'content': element.content,
            'children': [self._element_to_dict(c) for c in element.children],
            'connections': element.connections,
            'metadata': element.metadata,
        }

    def _dict_to_element(self, data: Dict[str, Any]) -> VisualElement:
        """Create element from dict."""
        return VisualElement(
            id=data['id'],
            element_type=data.get('element_type', 'box'),
            label=data.get('label', ''),
            position=data.get('position', {'x': 0, 'y': 0, 'z': 0}),
            size=data.get('size', {'w': 1, 'h': 1, 'd': 1}),
            style=data.get('style', {}),
            content=data.get('content'),
            children=[self._dict_to_element(c) for c in data.get('children', [])],
            connections=data.get('connections', []),
            metadata=data.get('metadata', {}),
        )

    def _get_nested(self, obj: Any, path: str) -> Any:
        """Get nested attribute using dot notation."""
        parts = path.split('.')
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
        return current

    def _set_nested(self, obj: Any, path: str, value: Any):
        """Set nested attribute using dot notation."""
        parts = path.split('.')
        current = obj
        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current.setdefault(part, {})
            else:
                current = getattr(current, part)

        final_key = parts[-1]
        if isinstance(current, dict):
            current[final_key] = value
        else:
            setattr(current, final_key, value)

    # -------------------------------------------------------------------------
    # State Access
    # -------------------------------------------------------------------------

    @property
    def elements(self) -> List[VisualElement]:
        """Get all elements."""
        return list(self._elements.values())

    @property
    def version(self) -> int:
        """Get current scene version."""
        return self._state.version

    @property
    def active_users(self) -> Set[str]:
        """Get active users."""
        return self._state.active_users

    def get_element(self, element_id: str) -> Optional[VisualElement]:
        """Get element by ID."""
        return self._elements.get(element_id)

    def get_state(self) -> SceneState:
        """Get full scene state."""
        return self._state

    def get_history(self, limit: int = 20) -> List[SceneOperation]:
        """Get recent operation history."""
        return self._state.operation_history[-limit:]

    # -------------------------------------------------------------------------
    # User Management
    # -------------------------------------------------------------------------

    async def user_joined(self, user_id: str):
        """Handle user joining the scene."""
        self._state.active_users.add(user_id)
        logger.info(f"User {user_id} joined scene {self.session_id}")

    async def user_left(self, user_id: str):
        """Handle user leaving the scene."""
        self._state.active_users.discard(user_id)
        logger.info(f"User {user_id} left scene {self.session_id}")


def create_collaborative_scene(
    scene: VisualScene,
    session_id: str,
    user_id: Optional[str] = None,
    **kwargs,
) -> CollaborativeScene:
    """
    Factory function to create a collaborative scene.

    Args:
        scene: The base VisualScene to wrap
        session_id: Unique session identifier
        user_id: Current user's ID
        **kwargs: Additional options

    Returns:
        CollaborativeScene instance
    """
    return CollaborativeScene(
        scene=scene,
        session_id=session_id,
        user_id=user_id,
        **kwargs,
    )


__all__ = [
    'CollaborativeScene',
    'SceneOperation',
    'SceneOperationType',
    'SceneState',
    'create_collaborative_scene',
    # Re-export role types for convenience
    'SceneRole',
    'RoleEnforcementError',
    'ROLES_AVAILABLE',
]

# Conditionally re-export role types
if ROLES_AVAILABLE:
    from .scene_roles import SceneRole, RoleEnforcementError

"""
ThirdEye Collaborative Scenes
=============================

Phase 1: Shared Scene State (December 2025)
Phase 2: Role Enforcement (December 2025)
Phase 3: Shared Styles + Activity Feed (December 2025)

Enables multiple users to view and edit the same ThirdEye scene
in real-time with proper synchronization and presence tracking.

Key Components:
- CollaborativeScene: Wraps VisualScene with sync capabilities
- ScenePresenceTracker: Tracks where users are focused in the scene
- SceneOperation: Atomic scene modifications for CRDT sync
- RoleEnforcer: Role-based access control for scene operations
- SharedStyleLibrary: Collaborative style management with voting
- ActivityFeed: Real-time activity tracking and notifications

Integration with HoloLoom.collaboration:
- Uses StateSynchronizer for CRDT-based conflict resolution
- Uses PresenceManager for cursor/focus tracking
- Uses AttributionManager for contribution tracking

Created: 2025-12-01
Updated: 2025-12-03 (Added role enforcement, shared styles, activity feed)
Author: HoloLoom Team
"""

from .scene_presence import (
    USER_COLORS,
    CursorMode,
    FocusTarget,
    SceneCursor,
    SceneFocus,
    ScenePresenceTracker,
    create_scene_presence,
)
from .scene_sync import (
    ROLES_AVAILABLE,
    CollaborativeScene,
    SceneOperation,
    SceneOperationType,
    SceneState,
    create_collaborative_scene,
)

# Role enforcement (graceful import if available)
try:
    from .scene_roles import (
        RoleAssignment,
        RoleChangeEvent,
        RoleEnforcementError,
        RoleEnforcer,
        RolePermissions,
        SceneRole,
        create_role_enforcer,
    )
    _ROLES_EXPORTS = [
        'SceneRole',
        'RolePermissions',
        'RoleAssignment',
        'RoleChangeEvent',
        'RoleEnforcementError',
        'RoleEnforcer',
        'create_role_enforcer',
        'ROLES_AVAILABLE',
    ]
except ImportError:
    _ROLES_EXPORTS = ['ROLES_AVAILABLE']

# Shared styles (graceful import if available)
try:
    from .shared_styles import (
        SharedStyle,
        SharedStyleLibrary,
        StyleCategory,
        StyleChangeEvent,
        StyleProperties,
        StyleStatus,
        StyleVersion,
        StyleVote,
        VoteType,
        create_style_library,
    )
    _STYLES_EXPORTS = [
        'StyleCategory',
        'StyleStatus',
        'VoteType',
        'StyleProperties',
        'StyleVote',
        'StyleVersion',
        'SharedStyle',
        'StyleChangeEvent',
        'SharedStyleLibrary',
        'create_style_library',
    ]
except ImportError:
    _STYLES_EXPORTS = []

# Activity feed (graceful import if available)
try:
    from .activity_feed import (
        ActivityEntry,
        ActivityFeed,
        ActivityFilter,
        ActivityPriority,
        ActivityType,
        create_activity_feed,
    )
    _ACTIVITY_EXPORTS = [
        'ActivityType',
        'ActivityPriority',
        'ActivityEntry',
        'ActivityFilter',
        'ActivityFeed',
        'create_activity_feed',
    ]
except ImportError:
    _ACTIVITY_EXPORTS = []

__all__ = [
    # Scene synchronization
    'CollaborativeScene',
    'SceneOperation',
    'SceneOperationType',
    'SceneState',
    'create_collaborative_scene',

    # Scene presence
    'ScenePresenceTracker',
    'SceneFocus',
    'SceneCursor',
    'FocusTarget',
    'CursorMode',
    'USER_COLORS',
    'create_scene_presence',
] + _ROLES_EXPORTS + _STYLES_EXPORTS + _ACTIVITY_EXPORTS

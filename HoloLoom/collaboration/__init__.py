"""
Collaboration Module
====================
Real-time collaborative editing with CRDT-based conflict resolution.

Components:
- Session: Multi-user session management
- Presence: User presence and activity tracking
- Sync: CRDT-based state synchronization
- Permissions: Role-based access control
- Activity: Version history and activity logging

Philosophy:
Collaboration is like weaving with multiple hands - each user is a shuttle
moving through the shared warp space. CRDTs ensure the threads don't tangle.
"""

from .session import CollaborativeSession, SessionManager
from .presence import PresenceTracker, UserPresence, PresenceStatus
from .sync import CRDTEngine, Delta, StateVector
from .permissions import PermissionManager, Role, Permission
from .activity import ActivityLogger, ActivityEvent, EventType

__all__ = [
    # Session management
    "CollaborativeSession",
    "SessionManager",

    # Presence tracking
    "PresenceTracker",
    "UserPresence",
    "PresenceStatus",

    # Synchronization
    "CRDTEngine",
    "Delta",
    "StateVector",

    # Permissions
    "PermissionManager",
    "Role",
    "Permission",

    # Activity logging
    "ActivityLogger",
    "ActivityEvent",
    "EventType",
]

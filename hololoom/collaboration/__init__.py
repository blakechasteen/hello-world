"""
HoloLoom Phase 3: Multi-User Collaboration
November 2025

Team knowledge management with:
- User identification and profiles
- Contribution tracking and attribution
- Access controls and permissions
- Knowledge sharing and export
- Real-time collaboration sessions
- Presence tracking and cursors
- State synchronization with conflict resolution
- WebRTC voice/video communication
"""

# Original modules
from hololoom.collaboration.user_manager import UserManager, User, UserRole
from hololoom.collaboration.contribution_tracker import ContributionTracker
from hololoom.collaboration.access_control import AccessController, Permission, AccessLevel
from hololoom.collaboration.knowledge_sharing import KnowledgeSharing, SharedKnowledge

# New session management (November 2025)
from hololoom.collaboration.session import (
    Session,
    SessionManager,
    SessionState,
    SessionType,
    SessionSettings,
    Participant,
    ParticipantRole,
    JoinRequest,
    create_session_manager,
)

# Real-time presence tracking
from hololoom.collaboration.presence import (
    UserPresence,
    PresenceManager,
    ActivityStatus,
    FocusType,
    CursorPosition,
    SelectionState,
    TypingIndicator,
    create_presence_manager,
)

# State synchronization with CRDT-inspired conflict resolution
from hololoom.collaboration.sync import (
    StateSynchronizer,
    Operation,
    OperationType,
    SyncState,
    Conflict,
    ConflictResolution,
    OperationBuffer,
    create_state_synchronizer,
)

# Enhanced attribution tracking
from hololoom.collaboration.attribution import (
    AttributionManager,
    Contribution,
    ContributionType,
    QualityRating,
    UserContributionStats,
    AttributionContext,
    create_attribution_manager,
)

# WebRTC voice/video communication
from hololoom.collaboration.voice import (
    VoiceRoom,
    VoiceManager,
    VoiceRoomSettings,
    VoiceRoomParticipant,
    PeerConnection,
    MediaTrack,
    MediaType,
    StreamQuality,
    ConnectionState,
    SignalingMessage,
    SignalingType,
    create_voice_room,
    create_voice_manager,
)

__all__ = [
    # Original - User management
    'UserManager',
    'User',
    'UserRole',

    # Original - Contribution tracking (legacy)
    'ContributionTracker',

    # Original - Access control
    'AccessController',
    'Permission',
    'AccessLevel',

    # Original - Knowledge sharing
    'KnowledgeSharing',
    'SharedKnowledge',

    # New - Session management
    'Session',
    'SessionManager',
    'SessionState',
    'SessionType',
    'SessionSettings',
    'Participant',
    'ParticipantRole',
    'JoinRequest',
    'create_session_manager',

    # New - Presence tracking
    'UserPresence',
    'PresenceManager',
    'ActivityStatus',
    'FocusType',
    'CursorPosition',
    'SelectionState',
    'TypingIndicator',
    'create_presence_manager',

    # New - State synchronization
    'StateSynchronizer',
    'Operation',
    'OperationType',
    'SyncState',
    'Conflict',
    'ConflictResolution',
    'OperationBuffer',
    'create_state_synchronizer',

    # New - Attribution (enhanced)
    'AttributionManager',
    'Contribution',
    'ContributionType',
    'QualityRating',
    'UserContributionStats',
    'AttributionContext',
    'create_attribution_manager',

    # New - Voice/Video
    'VoiceRoom',
    'VoiceManager',
    'VoiceRoomSettings',
    'VoiceRoomParticipant',
    'PeerConnection',
    'MediaTrack',
    'MediaType',
    'StreamQuality',
    'ConnectionState',
    'SignalingMessage',
    'SignalingType',
    'create_voice_room',
    'create_voice_manager',
]

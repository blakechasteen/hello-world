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
from hololoom.collaboration.access_control import AccessController, AccessLevel, Permission

# Enhanced attribution tracking
from hololoom.collaboration.attribution import (
    AttributionContext,
    AttributionManager,
    Contribution,
    ContributionType,
    QualityRating,
    UserContributionStats,
    create_attribution_manager,
)
from hololoom.collaboration.contribution_tracker import ContributionTracker
from hololoom.collaboration.knowledge_sharing import KnowledgeSharing, SharedKnowledge

# Real-time presence tracking
from hololoom.collaboration.presence import (
    ActivityStatus,
    CursorPosition,
    FocusType,
    PresenceManager,
    SelectionState,
    TypingIndicator,
    UserPresence,
    create_presence_manager,
)

# New session management (November 2025)
from hololoom.collaboration.session import (
    JoinRequest,
    Participant,
    ParticipantRole,
    Session,
    SessionManager,
    SessionSettings,
    SessionState,
    SessionType,
    create_session_manager,
)

# State synchronization with CRDT-inspired conflict resolution
from hololoom.collaboration.sync import (
    Conflict,
    ConflictResolution,
    Operation,
    OperationBuffer,
    OperationType,
    StateSynchronizer,
    SyncState,
    create_state_synchronizer,
)
from hololoom.collaboration.user_manager import User, UserManager, UserRole

# WebRTC voice/video communication
from hololoom.collaboration.voice import (
    ConnectionState,
    MediaTrack,
    MediaType,
    PeerConnection,
    SignalingMessage,
    SignalingType,
    StreamQuality,
    VoiceManager,
    VoiceRoom,
    VoiceRoomParticipant,
    VoiceRoomSettings,
    create_voice_manager,
    create_voice_room,
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

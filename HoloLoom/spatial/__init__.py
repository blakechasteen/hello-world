# HoloLoom Phase 4: Spatial Computing (Expanded & Enhanced)
# November 2025
#
# WebXR integration for immersive knowledge visualization
# - AR Knowledge Overlay
# - VR Memory Palace
# - Mixed Reality Collaboration
# - Spatial Audio (3D soundscapes)
# - Gaze Tracking (attention-based interaction)
# - Spatial UI (3D panels, menus, widgets)
# - Session Recording (capture and playback XR sessions)
# - Haptic Feedback (controller/hand haptics)
# - Physics Objects (physics-based manipulation)
# - 3D Whiteboard (collaborative drawing)
# - Spatial Notifications (3D alerts and messages)
# - Environment Mapping (scene persistence)
# - Avatar System (customizable user avatars)

from HoloLoom.spatial.webxr_graph import WebXRKnowledgeGraph, XRNode, XRAnchor
from HoloLoom.spatial.spatial_anchors import SpatialAnchorManager, SpatialAnchor
from HoloLoom.spatial.hand_tracking import HandTracker, HandGesture, GestureType
from HoloLoom.spatial.voice_commands import VoiceCommandSystem, VoiceCommand
from HoloLoom.spatial.presence import SpatialPresence, UserPresence, PresenceEvent
from HoloLoom.spatial.spatial_audio import SpatialAudioManager, AudioSource, AudioListener
from HoloLoom.spatial.gaze_tracking import GazeTracker, GazeRay, AttentionHeatMap, GazeHighlighter
from HoloLoom.spatial.spatial_ui import SpatialUIManager, Panel, Button, RadialMenu, InfoCard, TagCloud
from HoloLoom.spatial.session_recording import SessionRecorder, SessionPlayer, Recording, RecordingStorage
from HoloLoom.spatial.haptic_feedback import (
    HapticFeedbackManager, HapticPattern, HapticPulse, HapticEnvelope,
    HapticDevice, HapticWaveform, SpatialHapticZone, HapticPatternLibrary
)
from HoloLoom.spatial.physics_objects import (
    PhysicsWorld, PhysicsBody, PhysicsConstraint, PhysicsMaterial,
    CollisionShape, BodyType as PhysicsBodyType, PhysicsShape, CollisionLayer, ConstraintType
)
from HoloLoom.spatial.whiteboard_3d import (
    Whiteboard3D, WhiteboardManager, Stroke, Shape3D, TextAnnotation, StickyNote,
    WhiteboardLayer, DrawingTool, BrushStyle, BrushSettings
)
from HoloLoom.spatial.spatial_notifications import (
    SpatialNotificationManager, Notification, NotificationQueue,
    Badge, ProgressIndicator, AttentionGuide,
    NotificationType, NotificationPriority, NotificationPosition, AnimationType
)
from HoloLoom.spatial.environment_mapping import (
    EnvironmentMapper, Plane, EnvironmentMesh, PersistentAnchor, SceneSnapshot,
    SurfaceType, MeshType, ScanQuality, ScanProgress
)
from HoloLoom.spatial.avatar_system import (
    AvatarManager, Avatar, AvatarAnimator, AvatarIKSolver, AvatarPresets,
    BodySettings, HeadSettings, HandSettings, Expression, IKState, NameTag, StatusIndicator,
    AvatarStyle, BodyType as AvatarBodyType, HeadShape, HandStyle, ExpressionType,
    AnimationState, StatusType, IKTarget, AnimationClip
)

__all__ = [
    # WebXR Graph Visualization
    'WebXRKnowledgeGraph',
    'XRNode',
    'XRAnchor',
    # Spatial Anchors
    'SpatialAnchorManager',
    'SpatialAnchor',
    # Hand Tracking
    'HandTracker',
    'HandGesture',
    'GestureType',
    # Voice Commands
    'VoiceCommandSystem',
    'VoiceCommand',
    # Multi-user Presence
    'SpatialPresence',
    'UserPresence',
    'PresenceEvent',
    # Spatial Audio
    'SpatialAudioManager',
    'AudioSource',
    'AudioListener',
    # Gaze Tracking
    'GazeTracker',
    'GazeRay',
    'AttentionHeatMap',
    'GazeHighlighter',
    # Spatial UI
    'SpatialUIManager',
    'Panel',
    'Button',
    'RadialMenu',
    'InfoCard',
    'TagCloud',
    # Session Recording
    'SessionRecorder',
    'SessionPlayer',
    'Recording',
    'RecordingStorage',
    # Haptic Feedback
    'HapticFeedbackManager',
    'HapticPattern',
    'HapticPulse',
    'HapticEnvelope',
    'HapticDevice',
    'HapticWaveform',
    'SpatialHapticZone',
    'HapticPatternLibrary',
    # Physics Objects
    'PhysicsWorld',
    'PhysicsBody',
    'PhysicsConstraint',
    'PhysicsMaterial',
    'CollisionShape',
    'PhysicsBodyType',
    'PhysicsShape',
    'CollisionLayer',
    'ConstraintType',
    # 3D Whiteboard
    'Whiteboard3D',
    'WhiteboardManager',
    'Stroke',
    'Shape3D',
    'TextAnnotation',
    'StickyNote',
    'WhiteboardLayer',
    'DrawingTool',
    'BrushStyle',
    'BrushSettings',
    # Spatial Notifications
    'SpatialNotificationManager',
    'Notification',
    'NotificationQueue',
    'Badge',
    'ProgressIndicator',
    'AttentionGuide',
    'NotificationType',
    'NotificationPriority',
    'NotificationPosition',
    'AnimationType',
    # Environment Mapping
    'EnvironmentMapper',
    'Plane',
    'EnvironmentMesh',
    'PersistentAnchor',
    'SceneSnapshot',
    'SurfaceType',
    'MeshType',
    'ScanQuality',
    'ScanProgress',
    # Avatar System
    'AvatarManager',
    'Avatar',
    'AvatarAnimator',
    'AvatarIKSolver',
    'AvatarPresets',
    'BodySettings',
    'HeadSettings',
    'HandSettings',
    'Expression',
    'IKState',
    'NameTag',
    'StatusIndicator',
    'AvatarStyle',
    'AvatarBodyType',
    'HeadShape',
    'HandStyle',
    'ExpressionType',
    'AnimationState',
    'StatusType',
    'IKTarget',
    'AnimationClip',
]

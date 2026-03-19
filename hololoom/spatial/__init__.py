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
# - Math Types (shared Vector3, Quaternion, Color, Transform)
#
# Phase 4 Enhancement (November 2025):
# - Collaborative Spatial Session (bridge spatial + collaboration)
# - AR Knowledge Overlay (knowledge graph visualization)
# - Mobile-Responsive Spatial UI (touch gestures, adaptive layouts)

# Shared math types (used across all spatial modules)
from hololoom.spatial.avatar_system import (
    AnimationClip,
    AnimationState,
    Avatar,
    AvatarAnimator,
    AvatarIKSolver,
    AvatarManager,
    AvatarPresets,
    AvatarStyle,
    BodySettings,
    Expression,
    ExpressionType,
    HandSettings,
    HandStyle,
    HeadSettings,
    HeadShape,
    IKState,
    IKTarget,
    NameTag,
    StatusIndicator,
    StatusType,
)
from hololoom.spatial.avatar_system import BodyType as AvatarBodyType

# Phase 4 Enhancement: Collaborative Spatial Session (November 2025)
from hololoom.spatial.collaborative_session import (
    ARContextProvider,
    CollaborativeSpatialSession,
    GestureAction,
    GestureCollaborationBridge,
    GestureMapping,
    SharedSpatialObject,
    SpatialActivityType,
    SpatialContext,
    SpatialZone,
    SpatialZoneType,
    create_collaborative_spatial_session,
)
from hololoom.spatial.environment_mapping import (
    EnvironmentMapper,
    EnvironmentMesh,
    MeshType,
    PersistentAnchor,
    Plane,
    ScanProgress,
    ScanQuality,
    SceneSnapshot,
    SurfaceType,
)
from hololoom.spatial.gaze_tracking import AttentionHeatMap, GazeHighlighter, GazeRay, GazeTracker
from hololoom.spatial.hand_tracking import GestureType, HandGesture, HandTracker
from hololoom.spatial.haptic_feedback import (
    HapticDevice,
    HapticEnvelope,
    HapticFeedbackManager,
    HapticPattern,
    HapticPatternLibrary,
    HapticPulse,
    HapticWaveform,
    SpatialHapticZone,
)

# Phase 4 Enhancement: AR Knowledge Overlay (November 2025)
from hololoom.spatial.knowledge_overlay import (
    EdgeStyle,
    KnowledgeEdgeOverlay,
    KnowledgeNodeOverlay,
    KnowledgeOverlayManager,
    LayoutAlgorithm,
    LayoutEngine,
    MemoryPalaceRoom,
    OverlayCluster,
    OverlayStyle,
    VisibilityMode,
    create_knowledge_overlay_manager,
)
from hololoom.spatial.math_types import BoundingBox, Color, Quaternion, Transform, Vector3

# Phase 4 Enhancement: Mobile-Responsive Spatial UI (November 2025)
from hololoom.spatial.mobile_spatial_ui import (
    BottomSheet,
    DeviceCapabilities,
    DeviceType,
    FloatingActionButton,
    InputMode,
    MobileSpatialUIManager,
    ScreenOrientation,
    SpatialHUD,
    TouchGesture,
    TouchGestureRecognizer,
    TouchGestureType,
    UIAnchor,
    UIComponent,
    UIScale,
    create_default_spatial_ui,
    create_mobile_ui_manager,
)
from hololoom.spatial.mobile_spatial_ui import RadialMenu as MobileRadialMenu
from hololoom.spatial.physics_objects import BodyType as PhysicsBodyType
from hololoom.spatial.physics_objects import (
    CollisionLayer,
    CollisionShape,
    ConstraintType,
    PhysicsBody,
    PhysicsConstraint,
    PhysicsMaterial,
    PhysicsShape,
    PhysicsWorld,
)
from hololoom.spatial.presence import PresenceEvent, SpatialPresence, UserPresence
from hololoom.spatial.session_recording import (
    Recording,
    RecordingStorage,
    SessionPlayer,
    SessionRecorder,
)
from hololoom.spatial.spatial_anchors import SpatialAnchor, SpatialAnchorManager
from hololoom.spatial.spatial_audio import AudioListener, AudioSource, SpatialAudioManager
from hololoom.spatial.spatial_notifications import (
    AnimationType,
    AttentionGuide,
    Badge,
    Notification,
    NotificationPosition,
    NotificationPriority,
    NotificationQueue,
    NotificationType,
    ProgressIndicator,
    SpatialNotificationManager,
)
from hololoom.spatial.spatial_ui import (
    Button,
    InfoCard,
    Panel,
    RadialMenu,
    SpatialUIManager,
    TagCloud,
)
from hololoom.spatial.voice_commands import VoiceCommand, VoiceCommandSystem
from hololoom.spatial.webxr_graph import WebXRKnowledgeGraph, XRAnchor, XRNode
from hololoom.spatial.whiteboard_3d import (
    BrushSettings,
    BrushStyle,
    DrawingTool,
    Shape3D,
    StickyNote,
    Stroke,
    TextAnnotation,
    Whiteboard3D,
    WhiteboardLayer,
    WhiteboardManager,
)

__all__ = [
    # Shared Math Types
    'Vector3',
    'Quaternion',
    'Color',
    'Transform',
    'BoundingBox',
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
    # Phase 4 Enhancement: Collaborative Spatial Session
    'CollaborativeSpatialSession',
    'ARContextProvider',
    'GestureCollaborationBridge',
    'SharedSpatialObject',
    'SpatialZone',
    'SpatialContext',
    'SpatialActivityType',
    'SpatialZoneType',
    'GestureAction',
    'GestureMapping',
    'create_collaborative_spatial_session',
    # Phase 4 Enhancement: AR Knowledge Overlay
    'KnowledgeOverlayManager',
    'KnowledgeNodeOverlay',
    'KnowledgeEdgeOverlay',
    'OverlayCluster',
    'MemoryPalaceRoom',
    'LayoutEngine',
    'OverlayStyle',
    'VisibilityMode',
    'LayoutAlgorithm',
    'EdgeStyle',
    'create_knowledge_overlay_manager',
    # Phase 4 Enhancement: Mobile-Responsive Spatial UI
    'MobileSpatialUIManager',
    'TouchGestureRecognizer',
    'SpatialHUD',
    'UIComponent',
    'FloatingActionButton',
    'BottomSheet',
    'MobileRadialMenu',
    'DeviceCapabilities',
    'TouchGesture',
    'TouchGestureType',
    'DeviceType',
    'ScreenOrientation',
    'InputMode',
    'UIScale',
    'UIAnchor',
    'create_mobile_ui_manager',
    'create_default_spatial_ui',
]

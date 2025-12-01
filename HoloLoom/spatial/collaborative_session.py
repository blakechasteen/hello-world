# HoloLoom Phase 4: Collaborative Spatial Session
# November 2025
#
# Bridge module integrating spatial computing with multi-user collaboration
# - Unified session management (spatial + collaboration)
# - Synchronized spatial state across users
# - AR context provider (location/activity awareness)
# - Gesture-to-collaboration mappings
# - Spatial knowledge overlay system

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from datetime import datetime
import asyncio
import logging
import uuid

# Import spatial modules
from HoloLoom.spatial.math_types import Vector3, Quaternion, Color, Transform, BoundingBox
from HoloLoom.spatial.avatar_system import (
    AvatarManager, Avatar, AvatarPresets, StatusType as AvatarStatusType,
    Expression, ExpressionType, AnimationState, IKState
)
from HoloLoom.spatial.whiteboard_3d import (
    WhiteboardManager, Whiteboard3D, Stroke, Shape3D, TextAnnotation, StickyNote
)
from HoloLoom.spatial.physics_objects import (
    PhysicsWorld, PhysicsBody, CollisionShape, BodyType as PhysicsBodyType
)
from HoloLoom.spatial.environment_mapping import (
    EnvironmentMapper, Plane, SceneSnapshot, SurfaceType
)
from HoloLoom.spatial.hand_tracking import HandTracker, HandGesture, GestureType
from HoloLoom.spatial.spatial_audio import SpatialAudioManager, AudioSource
from HoloLoom.spatial.spatial_notifications import (
    SpatialNotificationManager, Notification, NotificationType, NotificationPriority
)
from HoloLoom.spatial.gaze_tracking import GazeTracker, GazeRay

# Import collaboration modules
from HoloLoom.collaboration.session import (
    Session, SessionManager, SessionState, SessionType, Participant, ParticipantRole
)
from HoloLoom.collaboration.presence import (
    PresenceManager, UserPresence, ActivityStatus, FocusType, CursorPosition
)
from HoloLoom.collaboration.sync import (
    StateSynchronizer, Operation, OperationType, ConflictResolution
)
from HoloLoom.collaboration.attribution import (
    AttributionManager, Contribution, ContributionType
)
from HoloLoom.collaboration.voice import (
    VoiceManager, VoiceRoom, VoiceRoomSettings
)

logger = logging.getLogger(__name__)


class SpatialActivityType(Enum):
    """Types of spatial activities for context awareness."""
    IDLE = auto()
    VIEWING = auto()           # Looking at content
    MANIPULATING = auto()      # Moving/rotating objects
    DRAWING = auto()           # Using whiteboard
    ANNOTATING = auto()        # Adding text/sticky notes
    PRESENTING = auto()        # Active presentation mode
    EXPLORING = auto()         # Walking/teleporting around
    COLLABORATING = auto()     # Interacting with another user
    SEARCHING = auto()         # Looking for specific content
    ORGANIZING = auto()        # Arranging spatial layout


class SpatialZoneType(Enum):
    """Types of spatial zones for context."""
    MAIN_STAGE = auto()        # Presentation area
    DISCUSSION_AREA = auto()   # Group discussion zone
    PRIVATE_BOOTH = auto()     # Private 1:1 space
    WHITEBOARD_ZONE = auto()   # Collaborative drawing area
    KNOWLEDGE_GARDEN = auto()  # Knowledge graph visualization
    ARCHIVE_VAULT = auto()     # Historical content
    CREATION_STUDIO = auto()   # Content creation space
    OBSERVATION_DECK = auto()  # Spectator area


class GestureAction(Enum):
    """Gesture-to-collaboration action mappings."""
    WAVE = auto()              # Greet/get attention
    POINT = auto()             # Highlight/indicate
    THUMBS_UP = auto()         # Approve/agree
    THUMBS_DOWN = auto()       # Disapprove/disagree
    RAISE_HAND = auto()        # Request to speak
    CLAP = auto()              # Applause
    PINCH = auto()             # Select/grab
    SPREAD = auto()            # Scale up/expand
    SWIPE = auto()             # Navigate/dismiss
    PUSH = auto()              # Share with others
    PULL = auto()              # Retrieve/take ownership
    CIRCLE = auto()            # Group selection


@dataclass
class SpatialContext:
    """Current spatial context for a user."""
    user_id: str

    # Location
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    current_zone: Optional[SpatialZoneType] = None
    nearby_users: List[str] = field(default_factory=list)

    # Activity
    activity: SpatialActivityType = SpatialActivityType.IDLE
    activity_target: Optional[str] = None  # Object/user being interacted with
    activity_duration: float = 0.0  # Seconds in current activity

    # Gaze
    gaze_target: Optional[str] = None
    gaze_duration: float = 0.0
    attention_history: List[str] = field(default_factory=list)  # Last 10 targets

    # Interaction
    held_objects: List[str] = field(default_factory=list)
    selected_objects: List[str] = field(default_factory=list)
    active_tools: List[str] = field(default_factory=list)

    # Temporal
    last_updated: datetime = field(default_factory=datetime.now)
    session_start: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "current_zone": self.current_zone.name if self.current_zone else None,
            "nearby_users": self.nearby_users,
            "activity": self.activity.name,
            "activity_target": self.activity_target,
            "activity_duration": self.activity_duration,
            "gaze_target": self.gaze_target,
            "gaze_duration": self.gaze_duration,
            "attention_history": self.attention_history[-10:],
            "held_objects": self.held_objects,
            "selected_objects": self.selected_objects,
            "active_tools": self.active_tools,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class SpatialZone:
    """Defines a spatial zone in the environment."""
    zone_id: str
    zone_type: SpatialZoneType
    name: str

    # Bounds
    center: Vector3 = field(default_factory=Vector3)
    size: Vector3 = field(default_factory=lambda: Vector3(5.0, 3.0, 5.0))

    # Access
    is_public: bool = True
    allowed_users: List[str] = field(default_factory=list)
    max_occupants: int = 10

    # Current state
    occupants: List[str] = field(default_factory=list)

    # Visual
    color: Color = field(default_factory=lambda: Color(0.3, 0.5, 0.8, 0.2))
    show_boundary: bool = True

    def contains_point(self, point: Vector3) -> bool:
        """Check if a point is within this zone."""
        half_size = Vector3(self.size.x / 2, self.size.y / 2, self.size.z / 2)
        return (
            self.center.x - half_size.x <= point.x <= self.center.x + half_size.x and
            self.center.y - half_size.y <= point.y <= self.center.y + half_size.y and
            self.center.z - half_size.z <= point.z <= self.center.z + half_size.z
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "zone_type": self.zone_type.name,
            "name": self.name,
            "center": self.center.to_dict(),
            "size": self.size.to_dict(),
            "is_public": self.is_public,
            "allowed_users": self.allowed_users,
            "max_occupants": self.max_occupants,
            "occupants": self.occupants,
            "color": self.color.to_dict(),
            "show_boundary": self.show_boundary
        }


@dataclass
class SharedSpatialObject:
    """A shared object in the spatial environment."""
    object_id: str
    name: str
    object_type: str  # "physics", "knowledge_node", "annotation", "media"

    # Transform
    transform: Transform = field(default_factory=Transform)

    # Ownership
    owner_id: Optional[str] = None
    is_locked: bool = False
    locked_by: Optional[str] = None

    # Synchronization
    version: int = 0
    last_modifier: Optional[str] = None
    last_modified: datetime = field(default_factory=datetime.now)

    # Physics (optional)
    physics_body_id: Optional[str] = None
    is_grabbable: bool = True
    is_throwable: bool = True

    # Knowledge (optional - links to HoloLoom memory)
    memory_id: Optional[str] = None
    knowledge_node_id: Optional[str] = None

    # Visual
    color: Optional[Color] = None
    scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "object_type": self.object_type,
            "transform": self.transform.to_dict(),
            "owner_id": self.owner_id,
            "is_locked": self.is_locked,
            "locked_by": self.locked_by,
            "version": self.version,
            "last_modifier": self.last_modifier,
            "last_modified": self.last_modified.isoformat(),
            "is_grabbable": self.is_grabbable,
            "memory_id": self.memory_id,
            "knowledge_node_id": self.knowledge_node_id,
            "color": self.color.to_dict() if self.color else None,
            "scale": self.scale.to_dict()
        }


@dataclass
class GestureMapping:
    """Maps a gesture to a collaboration action."""
    gesture_type: GestureType
    action: GestureAction
    requires_target: bool = False
    cooldown_seconds: float = 0.5
    last_triggered: Optional[datetime] = None

    def can_trigger(self) -> bool:
        """Check if gesture can be triggered (cooldown passed)."""
        if self.last_triggered is None:
            return True
        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown_seconds


class ARContextProvider:
    """
    Provides context-aware information based on spatial location and activity.

    Bridges spatial awareness with knowledge retrieval.
    """

    def __init__(self, session: 'CollaborativeSpatialSession'):
        self.session = session
        self.user_contexts: Dict[str, SpatialContext] = {}
        self.context_update_interval = 0.1  # 100ms

        # Activity detection thresholds
        self.idle_threshold = 3.0  # seconds without movement
        self.gaze_threshold = 2.0  # seconds looking at same target

        # Proximity settings
        self.nearby_distance = 3.0  # meters
        self.interaction_distance = 1.5  # meters

    def get_context(self, user_id: str) -> Optional[SpatialContext]:
        """Get current context for a user."""
        return self.user_contexts.get(user_id)

    def update_context(
        self,
        user_id: str,
        position: Vector3,
        rotation: Quaternion,
        gaze_ray: Optional[GazeRay] = None
    ) -> SpatialContext:
        """Update spatial context for a user."""
        now = datetime.now()

        # Get or create context
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = SpatialContext(user_id=user_id)

        ctx = self.user_contexts[user_id]
        old_position = ctx.position
        old_activity = ctx.activity

        # Update position
        ctx.position = position
        ctx.rotation = rotation

        # Detect zone
        ctx.current_zone = self._detect_zone(position)

        # Find nearby users
        ctx.nearby_users = self._find_nearby_users(user_id, position)

        # Update gaze tracking
        if gaze_ray:
            ctx.gaze_target = self._raycast_target(gaze_ray)
            if ctx.gaze_target:
                if len(ctx.attention_history) == 0 or ctx.attention_history[-1] != ctx.gaze_target:
                    ctx.attention_history.append(ctx.gaze_target)
                    ctx.attention_history = ctx.attention_history[-10:]
                    ctx.gaze_duration = 0.0
                else:
                    ctx.gaze_duration += self.context_update_interval

        # Detect activity
        ctx.activity = self._detect_activity(ctx, old_position, now)

        # Update duration
        if ctx.activity == old_activity:
            ctx.activity_duration += (now - ctx.last_updated).total_seconds()
        else:
            ctx.activity_duration = 0.0

        ctx.last_updated = now
        return ctx

    def _detect_zone(self, position: Vector3) -> Optional[SpatialZoneType]:
        """Detect which zone a position is in."""
        for zone in self.session.zones.values():
            if zone.contains_point(position):
                return zone.zone_type
        return None

    def _find_nearby_users(self, user_id: str, position: Vector3) -> List[str]:
        """Find users within nearby distance."""
        nearby = []
        for other_id, ctx in self.user_contexts.items():
            if other_id == user_id:
                continue

            dx = position.x - ctx.position.x
            dy = position.y - ctx.position.y
            dz = position.z - ctx.position.z
            distance = (dx*dx + dy*dy + dz*dz) ** 0.5

            if distance <= self.nearby_distance:
                nearby.append(other_id)

        return nearby

    def _raycast_target(self, gaze_ray: GazeRay) -> Optional[str]:
        """Find what user is looking at (simplified raycast)."""
        # In production, this would do proper spatial queries
        # For now, check against shared objects and users

        # Check shared objects
        for obj in self.session.shared_objects.values():
            if self._ray_intersects_object(gaze_ray, obj):
                return f"object:{obj.object_id}"

        # Check other users (avatars)
        for user_id, ctx in self.user_contexts.items():
            if self._ray_intersects_point(gaze_ray, ctx.position, 0.5):
                return f"user:{user_id}"

        return None

    def _ray_intersects_object(self, ray: GazeRay, obj: SharedSpatialObject) -> bool:
        """Simple sphere intersection test."""
        return self._ray_intersects_point(
            ray,
            Vector3(obj.transform.position.x, obj.transform.position.y, obj.transform.position.z),
            obj.scale.x * 0.5
        )

    def _ray_intersects_point(self, ray: GazeRay, point: Vector3, radius: float) -> bool:
        """Check if ray intersects sphere at point with radius."""
        # Simplified check using distance from ray
        dx = point.x - ray.origin.x
        dy = point.y - ray.origin.y
        dz = point.z - ray.origin.z

        # Project onto ray direction
        dot = dx * ray.direction.x + dy * ray.direction.y + dz * ray.direction.z
        if dot < 0:
            return False  # Behind ray

        # Distance from ray
        closest_x = ray.origin.x + ray.direction.x * dot
        closest_y = ray.origin.y + ray.direction.y * dot
        closest_z = ray.origin.z + ray.direction.z * dot

        dist_sq = (point.x - closest_x)**2 + (point.y - closest_y)**2 + (point.z - closest_z)**2
        return dist_sq <= radius * radius

    def _detect_activity(
        self,
        ctx: SpatialContext,
        old_position: Vector3,
        now: datetime
    ) -> SpatialActivityType:
        """Detect current activity based on context."""
        # Check for movement
        dx = ctx.position.x - old_position.x
        dy = ctx.position.y - old_position.y
        dz = ctx.position.z - old_position.z
        movement = (dx*dx + dy*dy + dz*dz) ** 0.5

        # Check held objects
        if ctx.held_objects:
            return SpatialActivityType.MANIPULATING

        # Check active tools
        if "whiteboard_brush" in ctx.active_tools:
            return SpatialActivityType.DRAWING
        if "annotation_tool" in ctx.active_tools:
            return SpatialActivityType.ANNOTATING

        # Check movement
        if movement > 0.1:  # Moving
            return SpatialActivityType.EXPLORING

        # Check gaze
        if ctx.gaze_target and ctx.gaze_duration > self.gaze_threshold:
            return SpatialActivityType.VIEWING

        # Check nearby users for collaboration
        if ctx.nearby_users:
            for nearby_id in ctx.nearby_users:
                nearby_ctx = self.user_contexts.get(nearby_id)
                if nearby_ctx and nearby_id in (nearby_ctx.attention_history[-3:] if nearby_ctx.attention_history else []):
                    return SpatialActivityType.COLLABORATING

        # Default to idle if no movement for threshold
        elapsed = (now - ctx.last_updated).total_seconds()
        if movement < 0.01 and elapsed > self.idle_threshold:
            return SpatialActivityType.IDLE

        return SpatialActivityType.VIEWING

    def get_relevant_knowledge_query(self, user_id: str) -> Optional[str]:
        """Generate a knowledge query based on spatial context."""
        ctx = self.get_context(user_id)
        if not ctx:
            return None

        # Build query based on context
        query_parts = []

        # Zone context
        if ctx.current_zone:
            zone_queries = {
                SpatialZoneType.KNOWLEDGE_GARDEN: "Show related concepts",
                SpatialZoneType.ARCHIVE_VAULT: "Show historical context",
                SpatialZoneType.CREATION_STUDIO: "Show similar creations",
                SpatialZoneType.DISCUSSION_AREA: "Show discussion topics"
            }
            if ctx.current_zone in zone_queries:
                query_parts.append(zone_queries[ctx.current_zone])

        # Gaze target context
        if ctx.gaze_target:
            if ctx.gaze_target.startswith("object:"):
                obj_id = ctx.gaze_target.split(":")[1]
                obj = self.session.shared_objects.get(obj_id)
                if obj and obj.knowledge_node_id:
                    query_parts.append(f"Related to {obj.name}")

        # Activity context
        if ctx.activity == SpatialActivityType.SEARCHING:
            query_parts.append("Search results")
        elif ctx.activity == SpatialActivityType.COLLABORATING:
            query_parts.append("Shared knowledge")

        return " | ".join(query_parts) if query_parts else None

    def remove_user(self, user_id: str):
        """Remove user context."""
        if user_id in self.user_contexts:
            del self.user_contexts[user_id]

    def to_state(self) -> Dict[str, Any]:
        """Export context provider state."""
        return {
            "user_contexts": {
                user_id: ctx.to_dict()
                for user_id, ctx in self.user_contexts.items()
            },
            "context_update_interval": self.context_update_interval
        }


class GestureCollaborationBridge:
    """
    Maps spatial gestures to collaboration actions.

    Enables natural gesture-based collaboration commands.
    """

    def __init__(self, session: 'CollaborativeSpatialSession'):
        self.session = session
        self.mappings: Dict[GestureType, GestureMapping] = {}
        self.gesture_handlers: Dict[GestureAction, Callable] = {}

        # Register default mappings
        self._register_default_mappings()

    def _register_default_mappings(self):
        """Register default gesture-to-action mappings."""
        defaults = [
            GestureMapping(GestureType.WAVE, GestureAction.WAVE),
            GestureMapping(GestureType.POINT, GestureAction.POINT, requires_target=True),
            GestureMapping(GestureType.THUMBS_UP, GestureAction.THUMBS_UP),
            GestureMapping(GestureType.THUMBS_DOWN, GestureAction.THUMBS_DOWN),
            GestureMapping(GestureType.OPEN_PALM, GestureAction.RAISE_HAND),
            GestureMapping(GestureType.PINCH, GestureAction.PINCH, requires_target=True),
            GestureMapping(GestureType.SPREAD, GestureAction.SPREAD, requires_target=True),
            GestureMapping(GestureType.SWIPE_LEFT, GestureAction.SWIPE),
            GestureMapping(GestureType.SWIPE_RIGHT, GestureAction.SWIPE),
        ]

        for mapping in defaults:
            self.mappings[mapping.gesture_type] = mapping

    def register_handler(self, action: GestureAction, handler: Callable):
        """Register a handler for a gesture action."""
        self.gesture_handlers[action] = handler

    async def handle_gesture(
        self,
        user_id: str,
        gesture: HandGesture,
        target: Optional[str] = None
    ) -> bool:
        """
        Handle a gesture and trigger collaboration action.

        Returns True if action was triggered.
        """
        mapping = self.mappings.get(gesture.gesture_type)
        if not mapping:
            return False

        if not mapping.can_trigger():
            return False

        if mapping.requires_target and not target:
            return False

        # Mark as triggered
        mapping.last_triggered = datetime.now()

        # Execute action
        action = mapping.action

        # Call registered handler
        handler = self.gesture_handlers.get(action)
        if handler:
            try:
                await handler(user_id, target)
            except Exception as e:
                logger.error(f"Gesture handler error: {e}")
                return False

        # Default actions
        await self._execute_default_action(user_id, action, target)

        return True

    async def _execute_default_action(
        self,
        user_id: str,
        action: GestureAction,
        target: Optional[str]
    ):
        """Execute default action for gesture."""
        if action == GestureAction.WAVE:
            # Trigger wave animation and notification
            await self.session.notify_users(
                f"User waved at everyone",
                user_id,
                NotificationType.INFO
            )

        elif action == GestureAction.POINT and target:
            # Highlight target for all users
            await self.session.highlight_object(target, user_id, duration=3.0)

        elif action == GestureAction.THUMBS_UP and target:
            # Record positive contribution
            if target.startswith("object:"):
                obj_id = target.split(":")[1]
                await self.session.record_contribution(
                    user_id,
                    ContributionType.COMMENT,
                    obj_id,
                    {"action": "approved"}
                )

        elif action == GestureAction.RAISE_HAND:
            # Request to speak in voice room
            if self.session.voice_room:
                await self.session.voice_room.request_floor(user_id)

        elif action == GestureAction.PUSH and target:
            # Share object with nearby users
            ctx = self.session.context_provider.get_context(user_id)
            if ctx and ctx.nearby_users:
                await self.session.share_object(target, ctx.nearby_users)

    def add_mapping(
        self,
        gesture_type: GestureType,
        action: GestureAction,
        requires_target: bool = False,
        cooldown: float = 0.5
    ):
        """Add or update a gesture mapping."""
        self.mappings[gesture_type] = GestureMapping(
            gesture_type=gesture_type,
            action=action,
            requires_target=requires_target,
            cooldown_seconds=cooldown
        )

    def to_state(self) -> Dict[str, Any]:
        """Export gesture mappings state."""
        return {
            "mappings": {
                gt.name: {
                    "action": m.action.name,
                    "requires_target": m.requires_target,
                    "cooldown": m.cooldown_seconds
                }
                for gt, m in self.mappings.items()
            }
        }


class CollaborativeSpatialSession:
    """
    Unified session for collaborative spatial experiences.

    Bridges HoloLoom's collaboration system with spatial computing modules:
    - Session + Presence management (collaboration)
    - Avatar + Physics + Whiteboard (spatial)
    - Context-aware knowledge retrieval
    - Gesture-based collaboration
    - Synchronized state across users
    """

    def __init__(
        self,
        session_id: str,
        name: str,
        owner_id: str,
        owner_name: str,
        session_type: SessionType = SessionType.KNOWLEDGE_BASE
    ):
        self.session_id = session_id
        self.name = name
        self.owner_id = owner_id
        self.session_type = session_type
        self.created_at = datetime.now()

        # Collaboration layer
        self.session_manager = SessionManager()
        self.presence_manager = PresenceManager(session_id)
        self.sync = StateSynchronizer(session_id)
        self.attribution = AttributionManager()
        self.voice_manager = VoiceManager()
        self.voice_room: Optional[VoiceRoom] = None

        # Spatial layer
        self.avatar_manager = AvatarManager()
        self.whiteboard_manager = WhiteboardManager()
        self.physics_world = PhysicsWorld()
        self.environment_mapper = EnvironmentMapper()
        self.audio_manager = SpatialAudioManager()
        self.notification_manager = SpatialNotificationManager()
        self.gaze_tracker = GazeTracker()
        self.hand_tracker = HandTracker()

        # Bridge components
        self.context_provider = ARContextProvider(self)
        self.gesture_bridge = GestureCollaborationBridge(self)

        # Shared state
        self.zones: Dict[str, SpatialZone] = {}
        self.shared_objects: Dict[str, SharedSpatialObject] = {}

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Background tasks
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None

        # Initialize session
        self._init_session(owner_id, owner_name)

    def _init_session(self, owner_id: str, owner_name: str):
        """Initialize session with owner."""
        # Create default zones
        self._create_default_zones()

    def _create_default_zones(self):
        """Create default spatial zones."""
        defaults = [
            SpatialZone(
                zone_id="main_stage",
                zone_type=SpatialZoneType.MAIN_STAGE,
                name="Main Stage",
                center=Vector3(0, 0, 0),
                size=Vector3(10, 4, 10)
            ),
            SpatialZone(
                zone_id="whiteboard_zone",
                zone_type=SpatialZoneType.WHITEBOARD_ZONE,
                name="Whiteboard Area",
                center=Vector3(8, 0, 0),
                size=Vector3(5, 3, 5),
                color=Color(0.8, 0.4, 0.2, 0.2)
            ),
            SpatialZone(
                zone_id="knowledge_garden",
                zone_type=SpatialZoneType.KNOWLEDGE_GARDEN,
                name="Knowledge Garden",
                center=Vector3(-8, 0, 0),
                size=Vector3(8, 4, 8),
                color=Color(0.2, 0.8, 0.4, 0.2)
            ),
            SpatialZone(
                zone_id="discussion_area",
                zone_type=SpatialZoneType.DISCUSSION_AREA,
                name="Discussion Circle",
                center=Vector3(0, 0, 8),
                size=Vector3(6, 3, 6),
                color=Color(0.4, 0.4, 0.8, 0.2)
            )
        ]

        for zone in defaults:
            self.zones[zone.zone_id] = zone

    async def start(self):
        """Start the collaborative spatial session."""
        self._running = True

        # Start presence tracking
        await self.presence_manager.start()

        # Start synchronization
        await self.sync.start()

        # Start background sync
        self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info(f"Collaborative spatial session {self.session_id} started")

    async def stop(self):
        """Stop the session."""
        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        await self.presence_manager.stop()
        await self.sync.stop()

        # Close voice room
        if self.voice_room:
            await self.voice_manager.close_room(self.voice_room.room_id)

        logger.info(f"Collaborative spatial session {self.session_id} stopped")

    async def _sync_loop(self):
        """Background synchronization loop."""
        while self._running:
            await asyncio.sleep(0.05)  # 20 Hz sync rate

            try:
                # Sync avatar positions
                await self._sync_avatars()

                # Update physics
                self.physics_world.step(0.05)

                # Check zone transitions
                await self._check_zone_transitions()

            except Exception as e:
                logger.error(f"Sync loop error: {e}")

    async def _sync_avatars(self):
        """Synchronize avatar states across users."""
        for avatar_id, avatar in self.avatar_manager.avatars.items():
            # Create sync operation for avatar state
            op = self.sync.create_operation(
                op_type=OperationType.UPDATE,
                target_type="avatar",
                target_id=avatar_id,
                data=avatar.to_dict(),
                parent_ops=[]
            )
            await self.sync.apply_local(op)

    async def _check_zone_transitions(self):
        """Check and handle zone transitions for users."""
        for user_id, ctx in self.context_provider.user_contexts.items():
            current_zone = ctx.current_zone

            for zone in self.zones.values():
                if zone.contains_point(ctx.position):
                    if current_zone != zone.zone_type:
                        # Zone transition
                        if user_id in zone.occupants:
                            continue

                        # Leave old zone
                        for z in self.zones.values():
                            if user_id in z.occupants:
                                z.occupants.remove(user_id)

                        # Enter new zone
                        zone.occupants.append(user_id)

                        # Emit event
                        self._emit_event("zone_transition", {
                            "user_id": user_id,
                            "from_zone": current_zone.name if current_zone else None,
                            "to_zone": zone.zone_type.name
                        })
                    break

    # === User Management ===

    async def join(
        self,
        user_id: str,
        display_name: str,
        avatar_preset: str = "casual"
    ) -> Tuple[Avatar, UserPresence]:
        """
        User joins the collaborative spatial session.

        Returns (Avatar, UserPresence) for the user.
        """
        # Add to presence
        presence = self.presence_manager.add_user(user_id, display_name)

        # Create avatar based on preset
        presets = {
            "professional": AvatarPresets.professional,
            "casual": AvatarPresets.casual,
            "robot": AvatarPresets.robot,
            "hologram": AvatarPresets.hologram,
            "minimal": AvatarPresets.minimal
        }

        preset_fn = presets.get(avatar_preset, AvatarPresets.casual)
        avatar = preset_fn(user_id, display_name)

        # Register avatar
        self.avatar_manager.avatars[avatar.avatar_id] = avatar
        self.avatar_manager.animators[avatar.avatar_id] = self.avatar_manager.animators.get(
            avatar.avatar_id,
            __import__('HoloLoom.spatial.avatar_system', fromlist=['AvatarAnimator']).AvatarAnimator()
        )

        # Initialize context
        self.context_provider.user_contexts[user_id] = SpatialContext(user_id=user_id)

        # Place in main stage
        avatar.position = Vector3(0, 0, 0)

        # Notify others
        await self.notify_users(
            f"{display_name} joined",
            user_id,
            NotificationType.INFO
        )

        # Emit event
        self._emit_event("user_joined", {
            "user_id": user_id,
            "display_name": display_name,
            "avatar_id": avatar.avatar_id
        })

        return avatar, presence

    async def leave(self, user_id: str):
        """User leaves the session."""
        # Get avatar for name
        avatar = None
        for a in self.avatar_manager.avatars.values():
            if a.user_id == user_id:
                avatar = a
                break

        display_name = avatar.name_tag.display_name if avatar and avatar.name_tag else user_id

        # Remove from presence
        self.presence_manager.remove_user(user_id)

        # Remove avatar
        if avatar:
            self.avatar_manager.remove_avatar(avatar.avatar_id)

        # Remove context
        self.context_provider.remove_user(user_id)

        # Remove from zones
        for zone in self.zones.values():
            if user_id in zone.occupants:
                zone.occupants.remove(user_id)

        # Notify others
        await self.notify_users(
            f"{display_name} left",
            user_id,
            NotificationType.INFO
        )

        # Emit event
        self._emit_event("user_left", {"user_id": user_id})

    # === Spatial Updates ===

    async def update_user_transform(
        self,
        user_id: str,
        position: Vector3,
        rotation: Quaternion,
        ik_state: Optional[IKState] = None,
        gaze_ray: Optional[GazeRay] = None
    ):
        """Update user's spatial transform."""
        # Find avatar
        avatar = None
        for a in self.avatar_manager.avatars.values():
            if a.user_id == user_id:
                avatar = a
                break

        if avatar:
            # Update avatar transform
            self.avatar_manager.update_position(avatar.avatar_id, position, rotation)

            # Update IK if provided
            if ik_state:
                self.avatar_manager.update_ik(avatar.avatar_id, ik_state)

        # Update presence cursor (as 3D position)
        self.presence_manager.update_cursor(user_id, position.x, position.y, position.z)

        # Update context
        self.context_provider.update_context(user_id, position, rotation, gaze_ray)

    async def handle_gesture(
        self,
        user_id: str,
        gesture: HandGesture,
        hand: str = "right"
    ) -> bool:
        """Handle a user gesture."""
        # Get gaze target for gesture targeting
        ctx = self.context_provider.get_context(user_id)
        target = ctx.gaze_target if ctx else None

        return await self.gesture_bridge.handle_gesture(user_id, gesture, target)

    # === Shared Objects ===

    async def create_shared_object(
        self,
        creator_id: str,
        name: str,
        object_type: str,
        position: Vector3,
        rotation: Quaternion = None,
        knowledge_node_id: Optional[str] = None
    ) -> SharedSpatialObject:
        """Create a new shared spatial object."""
        obj_id = str(uuid.uuid4())[:8]

        obj = SharedSpatialObject(
            object_id=obj_id,
            name=name,
            object_type=object_type,
            transform=Transform(
                position=position,
                rotation=rotation or Quaternion.identity(),
                scale=Vector3(1, 1, 1)
            ),
            owner_id=creator_id,
            knowledge_node_id=knowledge_node_id
        )

        self.shared_objects[obj_id] = obj

        # Record contribution
        await self.record_contribution(
            creator_id,
            ContributionType.CREATE,
            obj_id,
            {"object_type": object_type, "name": name}
        )

        # Sync operation
        op = self.sync.create_operation(
            op_type=OperationType.INSERT,
            target_type="shared_object",
            target_id=obj_id,
            data=obj.to_dict(),
            parent_ops=[]
        )
        await self.sync.apply_local(op)

        return obj

    async def grab_object(self, user_id: str, object_id: str) -> bool:
        """User grabs a shared object."""
        obj = self.shared_objects.get(object_id)
        if not obj or obj.is_locked:
            return False

        # Acquire lock
        if not self.sync.acquire_lock(object_id, "shared_object"):
            return False

        obj.is_locked = True
        obj.locked_by = user_id

        # Update user context
        ctx = self.context_provider.get_context(user_id)
        if ctx:
            ctx.held_objects.append(object_id)

        return True

    async def release_object(self, user_id: str, object_id: str) -> bool:
        """User releases a grabbed object."""
        obj = self.shared_objects.get(object_id)
        if not obj or obj.locked_by != user_id:
            return False

        # Release lock
        self.sync.release_lock(object_id)

        obj.is_locked = False
        obj.locked_by = None
        obj.last_modifier = user_id
        obj.last_modified = datetime.now()
        obj.version += 1

        # Update user context
        ctx = self.context_provider.get_context(user_id)
        if ctx and object_id in ctx.held_objects:
            ctx.held_objects.remove(object_id)

        return True

    async def update_object_transform(
        self,
        user_id: str,
        object_id: str,
        position: Vector3,
        rotation: Optional[Quaternion] = None,
        scale: Optional[Vector3] = None
    ) -> bool:
        """Update a shared object's transform."""
        obj = self.shared_objects.get(object_id)
        if not obj:
            return False

        # Check lock
        if obj.is_locked and obj.locked_by != user_id:
            return False

        obj.transform.position = position
        if rotation:
            obj.transform.rotation = rotation
        if scale:
            obj.scale = scale

        obj.last_modifier = user_id
        obj.last_modified = datetime.now()
        obj.version += 1

        # Sync
        op = self.sync.create_operation(
            op_type=OperationType.UPDATE,
            target_type="shared_object",
            target_id=object_id,
            data={"transform": obj.transform.to_dict(), "scale": obj.scale.to_dict()},
            parent_ops=[]
        )
        await self.sync.apply_local(op)

        return True

    async def highlight_object(
        self,
        object_id: str,
        highlighter_id: str,
        duration: float = 3.0
    ):
        """Highlight an object for all users."""
        obj = self.shared_objects.get(object_id)
        if not obj:
            return

        # Create highlight notification
        notification = Notification(
            notification_id=str(uuid.uuid4())[:8],
            notification_type=NotificationType.HIGHLIGHT,
            title="Highlighted",
            message=f"User pointed at {obj.name}",
            priority=NotificationPriority.MEDIUM,
            duration=duration,
            metadata={"object_id": object_id, "highlighter": highlighter_id}
        )

        # Broadcast to all users
        for user_id in self.presence_manager.presences.keys():
            self.notification_manager.show(notification, user_id)

    async def share_object(self, object_id: str, target_user_ids: List[str]):
        """Share an object with specific users."""
        obj = self.shared_objects.get(object_id)
        if not obj:
            return

        for user_id in target_user_ids:
            notification = Notification(
                notification_id=str(uuid.uuid4())[:8],
                notification_type=NotificationType.SHARE,
                title="Shared with you",
                message=f"Someone shared {obj.name}",
                priority=NotificationPriority.HIGH,
                metadata={"object_id": object_id}
            )
            self.notification_manager.show(notification, user_id)

    # === Whiteboard Integration ===

    async def create_whiteboard(
        self,
        creator_id: str,
        position: Vector3,
        size: Tuple[float, float] = (3.0, 2.0)
    ) -> Whiteboard3D:
        """Create a collaborative whiteboard."""
        whiteboard = self.whiteboard_manager.create_whiteboard(
            width=size[0],
            height=size[1]
        )

        # Create as shared object
        await self.create_shared_object(
            creator_id=creator_id,
            name=f"Whiteboard {whiteboard.whiteboard_id[:4]}",
            object_type="whiteboard",
            position=position
        )

        return whiteboard

    async def whiteboard_stroke(
        self,
        user_id: str,
        whiteboard_id: str,
        stroke: Stroke
    ) -> bool:
        """Add a stroke to a whiteboard."""
        whiteboard = self.whiteboard_manager.whiteboards.get(whiteboard_id)
        if not whiteboard:
            return False

        # Add stroke
        whiteboard.add_stroke(stroke)

        # Record contribution
        await self.record_contribution(
            user_id,
            ContributionType.CREATE,
            f"stroke:{whiteboard_id}",
            {"stroke_points": len(stroke.points)}
        )

        # Sync
        op = self.sync.create_operation(
            op_type=OperationType.INSERT,
            target_type="whiteboard_stroke",
            target_id=stroke.stroke_id,
            data=stroke.to_dict(),
            parent_ops=[]
        )
        await self.sync.apply_local(op)

        return True

    # === Voice Integration ===

    async def enable_voice(self, settings: Optional[VoiceRoomSettings] = None) -> VoiceRoom:
        """Enable voice chat for the session."""
        if self.voice_room:
            return self.voice_room

        settings = settings or VoiceRoomSettings(
            spatial_audio=True,
            echo_cancellation=True,
            noise_suppression=True
        )

        self.voice_room = await self.voice_manager.create_room(
            session_id=self.session_id,
            settings=settings
        )

        return self.voice_room

    async def join_voice(self, user_id: str, display_name: str) -> bool:
        """User joins voice chat."""
        if not self.voice_room:
            return False

        participant = await self.voice_manager.join_room(
            self.voice_room.room_id,
            user_id,
            display_name
        )

        return participant is not None

    # === Notifications ===

    async def notify_users(
        self,
        message: str,
        sender_id: Optional[str] = None,
        notification_type: NotificationType = NotificationType.INFO,
        target_users: Optional[List[str]] = None
    ):
        """Send notification to users."""
        notification = Notification(
            notification_id=str(uuid.uuid4())[:8],
            notification_type=notification_type,
            title="Notification",
            message=message,
            priority=NotificationPriority.MEDIUM,
            metadata={"sender": sender_id}
        )

        targets = target_users or list(self.presence_manager.presences.keys())

        for user_id in targets:
            if user_id != sender_id:  # Don't notify sender
                self.notification_manager.show(notification, user_id)

    # === Attribution ===

    async def record_contribution(
        self,
        user_id: str,
        contribution_type: ContributionType,
        target_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Contribution:
        """Record a user contribution."""
        # Get user name
        presence = self.presence_manager.get_presence(user_id)
        user_name = presence.display_name if presence else user_id

        return self.attribution.record_contribution(
            contribution_type=contribution_type,
            user_id=user_id,
            user_name=user_name,
            target_type="spatial_object",
            target_id=target_id,
            data=data
        )

    # === Events ===

    def on(self, event: str, handler: Callable):
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, data: Dict[str, Any]):
        """Emit an event to handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(event, data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    # === State Export ===

    def to_state(self) -> Dict[str, Any]:
        """Export complete session state for WebXR client."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "session_type": self.session_type.name,
            "created_at": self.created_at.isoformat(),

            # Users
            "presence": self.presence_manager.to_state(),
            "avatars": self.avatar_manager.to_state(),

            # Spatial
            "zones": {zid: z.to_dict() for zid, z in self.zones.items()},
            "shared_objects": {oid: o.to_dict() for oid, o in self.shared_objects.items()},
            "whiteboards": self.whiteboard_manager.to_state(),
            "physics": self.physics_world.to_state(),

            # Context
            "context": self.context_provider.to_state(),

            # Voice
            "voice_enabled": self.voice_room is not None,

            # Stats
            "user_count": len(self.presence_manager.presences),
            "object_count": len(self.shared_objects)
        }


# Factory functions

def create_collaborative_spatial_session(
    name: str,
    owner_id: str,
    owner_name: str,
    session_type: SessionType = SessionType.KNOWLEDGE_BASE
) -> CollaborativeSpatialSession:
    """Create a new collaborative spatial session."""
    session_id = str(uuid.uuid4())[:12]
    return CollaborativeSpatialSession(
        session_id=session_id,
        name=name,
        owner_id=owner_id,
        owner_name=owner_name,
        session_type=session_type
    )


async def create_and_start_session(
    name: str,
    owner_id: str,
    owner_name: str,
    session_type: SessionType = SessionType.KNOWLEDGE_BASE
) -> CollaborativeSpatialSession:
    """Create and start a collaborative spatial session."""
    session = create_collaborative_spatial_session(
        name=name,
        owner_id=owner_id,
        owner_name=owner_name,
        session_type=session_type
    )
    await session.start()
    return session

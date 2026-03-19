"""
HoloLoom Phase 4: Multi-User Spatial Presence
November 2025

Shared spatial sessions for collaborative
knowledge exploration in XR.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PresenceState(Enum):
    """User presence state."""
    OFFLINE = "offline"
    ONLINE = "online"
    ACTIVE = "active"           # Actively interacting
    IDLE = "idle"               # Online but inactive
    AWAY = "away"               # Temporarily away


class InteractionType(Enum):
    """Types of user interactions."""
    VIEW = "view"               # Viewing content
    SELECT = "select"           # Selected a node
    EDIT = "edit"               # Editing content
    ANNOTATE = "annotate"       # Adding annotations
    NAVIGATE = "navigate"       # Moving through space
    GESTURE = "gesture"         # Hand gesture
    VOICE = "voice"             # Voice command


class SessionRole(Enum):
    """User roles in a session."""
    HOST = "host"               # Session creator
    COLLABORATOR = "collaborator"  # Full edit access
    VIEWER = "viewer"           # Read-only access
    PRESENTER = "presenter"     # Presentation mode


@dataclass
class AvatarPose:
    """3D pose of user's avatar in XR space."""
    # Head position
    head_x: float = 0.0
    head_y: float = 1.6         # Default standing height
    head_z: float = 0.0

    # Head rotation (quaternion)
    head_qx: float = 0.0
    head_qy: float = 0.0
    head_qz: float = 0.0
    head_qw: float = 1.0

    # Hand positions (optional)
    left_hand_x: float | None = None
    left_hand_y: float | None = None
    left_hand_z: float | None = None

    right_hand_x: float | None = None
    right_hand_y: float | None = None
    right_hand_z: float | None = None

    # Controller ray direction (for pointing)
    ray_origin_x: float | None = None
    ray_origin_y: float | None = None
    ray_origin_z: float | None = None
    ray_direction_x: float | None = None
    ray_direction_y: float | None = None
    ray_direction_z: float | None = None

    def to_dict(self) -> dict:
        return {
            "head": {
                "position": {"x": self.head_x, "y": self.head_y, "z": self.head_z},
                "rotation": {"x": self.head_qx, "y": self.head_qy, "z": self.head_qz, "w": self.head_qw}
            },
            "left_hand": {
                "x": self.left_hand_x, "y": self.left_hand_y, "z": self.left_hand_z
            } if self.left_hand_x is not None else None,
            "right_hand": {
                "x": self.right_hand_x, "y": self.right_hand_y, "z": self.right_hand_z
            } if self.right_hand_x is not None else None,
            "ray": {
                "origin": {"x": self.ray_origin_x, "y": self.ray_origin_y, "z": self.ray_origin_z},
                "direction": {"x": self.ray_direction_x, "y": self.ray_direction_y, "z": self.ray_direction_z}
            } if self.ray_origin_x is not None else None
        }


@dataclass
class UserPresence:
    """Represents a user's presence in shared XR space."""
    user_id: str
    display_name: str
    state: PresenceState = PresenceState.ONLINE
    role: SessionRole = SessionRole.COLLABORATOR

    # Avatar
    avatar_color: str = "#00ff88"
    avatar_pose: AvatarPose = field(default_factory=AvatarPose)

    # Current focus
    focused_node_id: str | None = None
    selected_node_ids: set[str] = field(default_factory=set)
    current_interaction: InteractionType | None = None

    # Activity tracking
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    last_position_update: datetime = field(default_factory=datetime.now)

    # Session-specific
    session_id: str | None = None
    is_muted: bool = False
    can_edit: bool = True

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "state": self.state.value,
            "role": self.role.value,
            "avatar_color": self.avatar_color,
            "avatar_pose": self.avatar_pose.to_dict(),
            "focused_node_id": self.focused_node_id,
            "selected_node_ids": list(self.selected_node_ids),
            "current_interaction": self.current_interaction.value if self.current_interaction else None,
            "joined_at": self.joined_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "is_muted": self.is_muted,
            "can_edit": self.can_edit
        }

    def update_activity(self):
        """Mark user as active."""
        self.last_active = datetime.now()
        if self.state == PresenceState.IDLE:
            self.state = PresenceState.ACTIVE


@dataclass
class PresenceEvent:
    """An event in the presence system."""
    event_id: str
    event_type: str
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }


@dataclass
class SharedSession:
    """A shared XR session for multiple users."""
    session_id: str
    name: str
    host_id: str
    created_at: datetime = field(default_factory=datetime.now)

    # Participants
    participants: dict[str, UserPresence] = field(default_factory=dict)
    max_participants: int = 10

    # Session state
    is_active: bool = True
    is_locked: bool = False        # Locked = no new participants
    require_approval: bool = False  # Host must approve joins

    # Shared state
    shared_view_target: str | None = None  # Sync everyone's view
    presentation_mode: bool = False
    presenter_id: str | None = None

    # Access control
    password_hash: str | None = None
    allowed_domains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "host_id": self.host_id,
            "created_at": self.created_at.isoformat(),
            "participant_count": len(self.participants),
            "max_participants": self.max_participants,
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "presentation_mode": self.presentation_mode,
            "presenter_id": self.presenter_id
        }


class SpatialPresence:
    """
    Manages multi-user presence in shared XR space.

    Features:
    - Session management (create, join, leave)
    - Real-time pose synchronization
    - Focus and selection sharing
    - Presentation mode
    - Event broadcasting
    """

    # Avatar colors for participants
    AVATAR_COLORS = [
        "#00ff88", "#ff6b6b", "#4ecdc4", "#ffe66d",
        "#95e1d3", "#f38181", "#a8d8ea", "#ffaaa5",
        "#fcbad3", "#aa96da", "#c9b1ff", "#6eb5ff"
    ]

    def __init__(self):
        self.sessions: dict[str, SharedSession] = {}
        self.user_to_session: dict[str, str] = {}  # user_id -> session_id
        self._session_counter = 0
        self._event_counter = 0

        # Event callbacks
        self.on_user_joined: Callable[[str, UserPresence], None] | None = None
        self.on_user_left: Callable[[str, str], None] | None = None
        self.on_pose_update: Callable[[str, str, AvatarPose], None] | None = None
        self.on_focus_change: Callable[[str, str, str | None], None] | None = None
        self.on_event: Callable[[PresenceEvent], None] | None = None

    def _generate_session_id(self) -> str:
        self._session_counter += 1
        return f"session_{self._session_counter:06d}"

    def _generate_event_id(self) -> str:
        self._event_counter += 1
        return f"evt_{self._event_counter:06d}"

    def _assign_color(self, session: SharedSession) -> str:
        """Assign an unused avatar color."""
        used_colors = {p.avatar_color for p in session.participants.values()}
        for color in self.AVATAR_COLORS:
            if color not in used_colors:
                return color
        return self.AVATAR_COLORS[len(session.participants) % len(self.AVATAR_COLORS)]

    # === Session Management ===

    def create_session(
        self,
        host_id: str,
        host_name: str,
        session_name: str = "Shared Session",
        password: str = None,
        max_participants: int = 10
    ) -> SharedSession:
        """
        Create a new shared session.

        Args:
            host_id: User ID of the host
            host_name: Display name of the host
            session_name: Name of the session
            password: Optional password for joining
            max_participants: Maximum number of participants
        """
        session_id = self._generate_session_id()

        # Hash password if provided
        password_hash = None
        if password:
            password_hash = hashlib.sha256(password.encode()).hexdigest()

        session = SharedSession(
            session_id=session_id,
            name=session_name,
            host_id=host_id,
            max_participants=max_participants,
            password_hash=password_hash
        )

        # Add host as first participant
        host_presence = UserPresence(
            user_id=host_id,
            display_name=host_name,
            state=PresenceState.ACTIVE,
            role=SessionRole.HOST,
            avatar_color=self.AVATAR_COLORS[0],
            session_id=session_id,
            can_edit=True
        )

        session.participants[host_id] = host_presence
        self.sessions[session_id] = session
        self.user_to_session[host_id] = session_id

        self._emit_event("session_created", host_id, session_id, {
            "session_name": session_name
        })

        return session

    def join_session(
        self,
        session_id: str,
        user_id: str,
        display_name: str,
        password: str = None
    ) -> UserPresence | None:
        """
        Join an existing session.

        Returns:
            UserPresence if joined successfully, None otherwise
        """
        session = self.sessions.get(session_id)
        if not session:
            return None

        if not session.is_active:
            return None

        if session.is_locked:
            return None

        if len(session.participants) >= session.max_participants:
            return None

        # Check password
        if session.password_hash:
            if not password:
                return None
            if hashlib.sha256(password.encode()).hexdigest() != session.password_hash:
                return None

        # Check if user is already in another session
        if user_id in self.user_to_session:
            self.leave_session(user_id)

        # Create presence
        presence = UserPresence(
            user_id=user_id,
            display_name=display_name,
            state=PresenceState.ACTIVE,
            role=SessionRole.COLLABORATOR,
            avatar_color=self._assign_color(session),
            session_id=session_id
        )

        session.participants[user_id] = presence
        self.user_to_session[user_id] = session_id

        if self.on_user_joined:
            self.on_user_joined(session_id, presence)

        self._emit_event("user_joined", user_id, session_id, {
            "display_name": display_name
        })

        return presence

    def leave_session(self, user_id: str) -> bool:
        """Leave current session."""
        session_id = self.user_to_session.get(user_id)
        if not session_id:
            return False

        session = self.sessions.get(session_id)
        if not session:
            return False

        # Remove from session
        if user_id in session.participants:
            del session.participants[user_id]

        del self.user_to_session[user_id]

        if self.on_user_left:
            self.on_user_left(session_id, user_id)

        self._emit_event("user_left", user_id, session_id, {})

        # If host left and session is empty, close it
        if user_id == session.host_id:
            if not session.participants:
                self._close_session(session_id)
            else:
                # Transfer host to next participant
                new_host = next(iter(session.participants.keys()))
                session.host_id = new_host
                session.participants[new_host].role = SessionRole.HOST

        return True

    def _close_session(self, session_id: str):
        """Close a session."""
        session = self.sessions.get(session_id)
        if not session:
            return

        session.is_active = False

        # Remove all participants from session mapping
        for user_id in list(session.participants.keys()):
            if user_id in self.user_to_session:
                del self.user_to_session[user_id]

        self._emit_event("session_closed", session.host_id, session_id, {})

    # === Pose Updates ===

    def update_pose(self, user_id: str, pose: AvatarPose):
        """Update a user's avatar pose."""
        session_id = self.user_to_session.get(user_id)
        if not session_id:
            return

        session = self.sessions.get(session_id)
        if not session or user_id not in session.participants:
            return

        presence = session.participants[user_id]
        presence.avatar_pose = pose
        presence.last_position_update = datetime.now()
        presence.update_activity()

        if self.on_pose_update:
            self.on_pose_update(session_id, user_id, pose)

    def update_focus(self, user_id: str, node_id: str | None):
        """Update what node a user is focusing on."""
        session_id = self.user_to_session.get(user_id)
        if not session_id:
            return

        session = self.sessions.get(session_id)
        if not session or user_id not in session.participants:
            return

        presence = session.participants[user_id]
        old_focus = presence.focused_node_id
        presence.focused_node_id = node_id
        presence.update_activity()

        if self.on_focus_change:
            self.on_focus_change(session_id, user_id, node_id)

        if old_focus != node_id:
            self._emit_event("focus_changed", user_id, session_id, {
                "old_node_id": old_focus,
                "new_node_id": node_id
            })

    def update_selection(self, user_id: str, node_ids: set[str]):
        """Update a user's selection."""
        session_id = self.user_to_session.get(user_id)
        if not session_id:
            return

        session = self.sessions.get(session_id)
        if not session or user_id not in session.participants:
            return

        presence = session.participants[user_id]
        presence.selected_node_ids = node_ids
        presence.update_activity()

        self._emit_event("selection_changed", user_id, session_id, {
            "node_ids": list(node_ids)
        })

    def update_interaction(self, user_id: str, interaction: InteractionType | None):
        """Update what type of interaction user is performing."""
        session_id = self.user_to_session.get(user_id)
        if not session_id:
            return

        session = self.sessions.get(session_id)
        if not session or user_id not in session.participants:
            return

        presence = session.participants[user_id]
        presence.current_interaction = interaction
        presence.update_activity()

    # === Presentation Mode ===

    def start_presentation(self, presenter_id: str) -> bool:
        """Start presentation mode (sync everyone's view to presenter)."""
        session_id = self.user_to_session.get(presenter_id)
        if not session_id:
            return False

        session = self.sessions.get(session_id)
        if not session:
            return False

        # Check if user can present (host or collaborator)
        presence = session.participants.get(presenter_id)
        if not presence or presence.role == SessionRole.VIEWER:
            return False

        session.presentation_mode = True
        session.presenter_id = presenter_id
        presence.role = SessionRole.PRESENTER

        self._emit_event("presentation_started", presenter_id, session_id, {})
        return True

    def end_presentation(self, user_id: str) -> bool:
        """End presentation mode."""
        session_id = self.user_to_session.get(user_id)
        if not session_id:
            return False

        session = self.sessions.get(session_id)
        if not session:
            return False

        # Only host or presenter can end
        if user_id != session.host_id and user_id != session.presenter_id:
            return False

        # Restore presenter's previous role
        if session.presenter_id and session.presenter_id in session.participants:
            if session.presenter_id == session.host_id:
                session.participants[session.presenter_id].role = SessionRole.HOST
            else:
                session.participants[session.presenter_id].role = SessionRole.COLLABORATOR

        session.presentation_mode = False
        session.presenter_id = None

        self._emit_event("presentation_ended", user_id, session_id, {})
        return True

    def set_shared_view(self, presenter_id: str, node_id: str) -> bool:
        """Set the shared view target during presentation."""
        session_id = self.user_to_session.get(presenter_id)
        if not session_id:
            return False

        session = self.sessions.get(session_id)
        if not session or not session.presentation_mode:
            return False

        if presenter_id != session.presenter_id:
            return False

        session.shared_view_target = node_id

        self._emit_event("view_sync", presenter_id, session_id, {
            "node_id": node_id
        })
        return True

    # === Queries ===

    def get_session(self, session_id: str) -> SharedSession | None:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def get_user_session(self, user_id: str) -> SharedSession | None:
        """Get the session a user is in."""
        session_id = self.user_to_session.get(user_id)
        if session_id:
            return self.sessions.get(session_id)
        return None

    def get_presence(self, user_id: str) -> UserPresence | None:
        """Get a user's presence."""
        session = self.get_user_session(user_id)
        if session:
            return session.participants.get(user_id)
        return None

    def get_session_participants(self, session_id: str) -> list[UserPresence]:
        """Get all participants in a session."""
        session = self.sessions.get(session_id)
        if not session:
            return []
        return list(session.participants.values())

    def get_active_sessions(self) -> list[SharedSession]:
        """Get all active sessions."""
        return [s for s in self.sessions.values() if s.is_active]

    def get_users_focused_on(self, session_id: str, node_id: str) -> list[UserPresence]:
        """Get all users focused on a specific node."""
        session = self.sessions.get(session_id)
        if not session:
            return []

        return [
            p for p in session.participants.values()
            if p.focused_node_id == node_id
        ]

    # === Events ===

    def _emit_event(self, event_type: str, user_id: str, session_id: str, data: dict):
        """Emit a presence event."""
        event = PresenceEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            user_id=user_id,
            data={"session_id": session_id, **data}
        )

        if self.on_event:
            self.on_event(event)

    # === State Export ===

    def export_session_state(self, session_id: str) -> str:
        """Export session state as JSON for network sync."""
        session = self.sessions.get(session_id)
        if not session:
            return "{}"

        data = {
            "session": session.to_dict(),
            "participants": [p.to_dict() for p in session.participants.values()]
        }
        return json.dumps(data, indent=2)

    def get_statistics(self) -> dict:
        """Get presence system statistics."""
        active_sessions = [s for s in self.sessions.values() if s.is_active]
        total_users = sum(len(s.participants) for s in active_sessions)

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "total_users": total_users,
            "users_by_state": self._count_users_by_state()
        }

    def _count_users_by_state(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for session in self.sessions.values():
            if not session.is_active:
                continue
            for presence in session.participants.values():
                state = presence.state.value
                counts[state] = counts.get(state, 0) + 1
        return counts


# Quick test
if __name__ == "__main__":
    print("Testing Spatial Presence...")

    presence = SpatialPresence()

    # Event callback
    def on_event(evt):
        print(f"Event: {evt.event_type} from {evt.user_id}")

    presence.on_event = on_event

    # Create session
    session = presence.create_session(
        host_id="alice",
        host_name="Alice",
        session_name="Knowledge Exploration"
    )
    print(f"Created session: {session.session_id}")

    # Join session
    bob = presence.join_session(
        session.session_id,
        user_id="bob",
        display_name="Bob"
    )
    print(f"Bob joined with color: {bob.avatar_color}")

    charlie = presence.join_session(
        session.session_id,
        user_id="charlie",
        display_name="Charlie"
    )
    print(f"Charlie joined with color: {charlie.avatar_color}")

    # Update poses
    presence.update_pose("alice", AvatarPose(head_x=1.0, head_y=1.6, head_z=2.0))
    presence.update_pose("bob", AvatarPose(head_x=-1.0, head_y=1.6, head_z=2.0))

    # Update focus
    presence.update_focus("alice", "thompson_sampling")
    presence.update_focus("bob", "thompson_sampling")

    # Who's looking at thompson_sampling?
    viewers = presence.get_users_focused_on(session.session_id, "thompson_sampling")
    print(f"Users focused on Thompson Sampling: {[v.display_name for v in viewers]}")

    # Start presentation
    presence.start_presentation("alice")
    presence.set_shared_view("alice", "bayesian_methods")

    # Statistics
    stats = presence.get_statistics()
    print(f"\nStats: {stats}")

    # Export state
    state_json = presence.export_session_state(session.session_id)
    print(f"\nSession state (first 500 chars): {state_json[:500]}")

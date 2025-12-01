"""
Real-Time Presence System for Multi-User Collaboration.

Tracks cursor positions, focus states, and activity status across participants.

Phase 3: Multi-User Collaboration - Presence infrastructure.

Created: November 2025
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Tuple
import asyncio
import logging

logger = logging.getLogger(__name__)


class ActivityStatus(Enum):
    """User activity status."""
    ONLINE = "online"
    ACTIVE = "active"          # Recently interacted
    IDLE = "idle"              # No activity for a while
    AWAY = "away"              # User set as away
    DO_NOT_DISTURB = "dnd"     # User set DND
    OFFLINE = "offline"


class FocusType(Enum):
    """What element the user is focused on."""
    NONE = "none"
    QUERY_INPUT = "query_input"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    WHITEBOARD = "whiteboard"
    RESPONSE_VIEW = "response_view"
    METRICS_PANEL = "metrics_panel"
    SPATIAL_OBJECT = "spatial_object"
    MENU = "menu"
    SETTINGS = "settings"


@dataclass
class CursorPosition:
    """User cursor/pointer position."""
    x: float
    y: float
    z: float = 0.0  # For 3D/spatial contexts
    viewport_id: Optional[str] = None  # Which viewport
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "viewport_id": self.viewport_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SelectionState:
    """User's current selection."""
    element_type: str  # "node", "edge", "text", "region", etc.
    element_ids: List[str] = field(default_factory=list)
    text_range: Optional[Tuple[int, int]] = None  # For text selections
    bounds: Optional[Dict[str, float]] = None  # Selection bounding box
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "element_ids": self.element_ids,
            "text_range": self.text_range,
            "bounds": self.bounds,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TypingIndicator:
    """Tracks if user is typing in a specific context."""
    context: str  # "query", "comment", "whiteboard_text", etc.
    started_at: datetime = field(default_factory=datetime.now)
    last_keystroke: datetime = field(default_factory=datetime.now)

    def is_active(self, timeout_seconds: float = 3.0) -> bool:
        """Check if typing is still considered active."""
        elapsed = (datetime.now() - self.last_keystroke).total_seconds()
        return elapsed < timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "started_at": self.started_at.isoformat(),
            "is_active": self.is_active()
        }


@dataclass
class UserPresence:
    """Complete presence state for a single user."""
    user_id: str
    display_name: str
    status: ActivityStatus = ActivityStatus.ONLINE
    custom_status: Optional[str] = None

    # Cursor and focus
    cursor: Optional[CursorPosition] = None
    focus: FocusType = FocusType.NONE
    focus_element_id: Optional[str] = None  # Specific element ID

    # Activity tracking
    selection: Optional[SelectionState] = None
    typing: Optional[TypingIndicator] = None

    # Timestamps
    last_seen: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    session_start: datetime = field(default_factory=datetime.now)

    # Visual appearance
    cursor_color: str = "#4F46E5"  # Indigo default
    avatar_url: Optional[str] = None

    def update_cursor(self, x: float, y: float, z: float = 0.0, viewport_id: Optional[str] = None):
        """Update cursor position."""
        self.cursor = CursorPosition(x=x, y=y, z=z, viewport_id=viewport_id)
        self.last_activity = datetime.now()
        self._update_status_from_activity()

    def update_focus(self, focus_type: FocusType, element_id: Optional[str] = None):
        """Update what the user is focused on."""
        self.focus = focus_type
        self.focus_element_id = element_id
        self.last_activity = datetime.now()
        self._update_status_from_activity()

    def update_selection(
        self,
        element_type: str,
        element_ids: List[str],
        text_range: Optional[Tuple[int, int]] = None,
        bounds: Optional[Dict[str, float]] = None
    ):
        """Update user's selection."""
        self.selection = SelectionState(
            element_type=element_type,
            element_ids=element_ids,
            text_range=text_range,
            bounds=bounds
        )
        self.last_activity = datetime.now()
        self._update_status_from_activity()

    def start_typing(self, context: str):
        """Mark user as typing."""
        now = datetime.now()
        if self.typing and self.typing.context == context:
            self.typing.last_keystroke = now
        else:
            self.typing = TypingIndicator(context=context)
        self.last_activity = now
        self._update_status_from_activity()

    def stop_typing(self):
        """Clear typing indicator."""
        self.typing = None

    def set_status(self, status: ActivityStatus, custom_message: Optional[str] = None):
        """Manually set status."""
        self.status = status
        self.custom_status = custom_message

    def heartbeat(self):
        """Update last seen timestamp."""
        self.last_seen = datetime.now()

    def _update_status_from_activity(self):
        """Auto-update status based on activity."""
        if self.status not in (ActivityStatus.AWAY, ActivityStatus.DO_NOT_DISTURB):
            self.status = ActivityStatus.ACTIVE

    def check_idle(self, idle_threshold: timedelta = timedelta(minutes=5)) -> bool:
        """Check if user should be marked idle."""
        if self.status in (ActivityStatus.AWAY, ActivityStatus.DO_NOT_DISTURB, ActivityStatus.OFFLINE):
            return False

        elapsed = datetime.now() - self.last_activity
        if elapsed > idle_threshold:
            self.status = ActivityStatus.IDLE
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "status": self.status.value,
            "custom_status": self.custom_status,
            "cursor": self.cursor.to_dict() if self.cursor else None,
            "focus": self.focus.value,
            "focus_element_id": self.focus_element_id,
            "selection": self.selection.to_dict() if self.selection else None,
            "typing": self.typing.to_dict() if self.typing else None,
            "last_seen": self.last_seen.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "cursor_color": self.cursor_color,
            "avatar_url": self.avatar_url
        }

    def to_cursor_overlay(self) -> Dict[str, Any]:
        """Get minimal data for cursor overlay rendering."""
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "cursor": self.cursor.to_dict() if self.cursor else None,
            "color": self.cursor_color,
            "is_typing": self.typing.is_active() if self.typing else False
        }


class PresenceManager:
    """
    Manages real-time presence for a collaboration session.

    Handles cursor tracking, status updates, and activity monitoring.
    """

    def __init__(
        self,
        session_id: str,
        idle_threshold: timedelta = timedelta(minutes=5),
        broadcast_interval: float = 0.05  # 50ms for smooth cursor updates
    ):
        self.session_id = session_id
        self.idle_threshold = idle_threshold
        self.broadcast_interval = broadcast_interval

        self.presences: Dict[str, UserPresence] = {}
        self._cursor_colors = self._generate_color_palette()
        self._color_index = 0

        self._event_handlers: Dict[str, List[Callable]] = {}
        self._idle_check_task: Optional[asyncio.Task] = None
        self._running = False

    def _generate_color_palette(self) -> List[str]:
        """Generate distinct colors for user cursors."""
        return [
            "#4F46E5",  # Indigo
            "#10B981",  # Emerald
            "#F59E0B",  # Amber
            "#EF4444",  # Red
            "#8B5CF6",  # Violet
            "#06B6D4",  # Cyan
            "#EC4899",  # Pink
            "#F97316",  # Orange
            "#14B8A6",  # Teal
            "#6366F1",  # Indigo-500
            "#84CC16",  # Lime
            "#A855F7",  # Purple
        ]

    def _get_next_color(self) -> str:
        """Get next color from palette."""
        color = self._cursor_colors[self._color_index % len(self._cursor_colors)]
        self._color_index += 1
        return color

    async def start(self):
        """Start presence monitoring."""
        self._running = True
        self._idle_check_task = asyncio.create_task(self._idle_check_loop())
        logger.info(f"Presence manager started for session {self.session_id}")

    async def stop(self):
        """Stop presence monitoring."""
        self._running = False
        if self._idle_check_task:
            self._idle_check_task.cancel()
            try:
                await self._idle_check_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Presence manager stopped for session {self.session_id}")

    async def _idle_check_loop(self):
        """Periodically check for idle users."""
        while self._running:
            await asyncio.sleep(30)  # Check every 30 seconds

            now = datetime.now()
            for user_id, presence in self.presences.items():
                old_status = presence.status
                if presence.check_idle(self.idle_threshold):
                    if old_status != presence.status:
                        self._emit_event("status_changed", {
                            "user_id": user_id,
                            "old_status": old_status.value,
                            "new_status": presence.status.value
                        })

    def add_user(
        self,
        user_id: str,
        display_name: str,
        avatar_url: Optional[str] = None
    ) -> UserPresence:
        """Add a user to presence tracking."""
        presence = UserPresence(
            user_id=user_id,
            display_name=display_name,
            cursor_color=self._get_next_color(),
            avatar_url=avatar_url
        )
        self.presences[user_id] = presence
        self._emit_event("user_joined", {"presence": presence.to_dict()})
        return presence

    def remove_user(self, user_id: str) -> bool:
        """Remove a user from presence tracking."""
        if user_id not in self.presences:
            return False

        presence = self.presences.pop(user_id)
        self._emit_event("user_left", {"user_id": user_id})
        return True

    def update_cursor(
        self,
        user_id: str,
        x: float,
        y: float,
        z: float = 0.0,
        viewport_id: Optional[str] = None
    ) -> bool:
        """Update user's cursor position."""
        presence = self.presences.get(user_id)
        if not presence:
            return False

        presence.update_cursor(x, y, z, viewport_id)
        self._emit_event("cursor_moved", {
            "user_id": user_id,
            "cursor": presence.cursor.to_dict()
        })
        return True

    def update_focus(
        self,
        user_id: str,
        focus_type: FocusType,
        element_id: Optional[str] = None
    ) -> bool:
        """Update user's focus target."""
        presence = self.presences.get(user_id)
        if not presence:
            return False

        old_focus = presence.focus
        presence.update_focus(focus_type, element_id)

        if old_focus != focus_type:
            self._emit_event("focus_changed", {
                "user_id": user_id,
                "focus": focus_type.value,
                "element_id": element_id
            })
        return True

    def update_selection(
        self,
        user_id: str,
        element_type: str,
        element_ids: List[str],
        text_range: Optional[Tuple[int, int]] = None,
        bounds: Optional[Dict[str, float]] = None
    ) -> bool:
        """Update user's selection."""
        presence = self.presences.get(user_id)
        if not presence:
            return False

        presence.update_selection(element_type, element_ids, text_range, bounds)
        self._emit_event("selection_changed", {
            "user_id": user_id,
            "selection": presence.selection.to_dict()
        })
        return True

    def set_typing(self, user_id: str, context: str, is_typing: bool) -> bool:
        """Update typing indicator."""
        presence = self.presences.get(user_id)
        if not presence:
            return False

        if is_typing:
            presence.start_typing(context)
            self._emit_event("typing_started", {
                "user_id": user_id,
                "context": context
            })
        else:
            presence.stop_typing()
            self._emit_event("typing_stopped", {
                "user_id": user_id,
                "context": context
            })
        return True

    def set_status(
        self,
        user_id: str,
        status: ActivityStatus,
        custom_message: Optional[str] = None
    ) -> bool:
        """Set user's status."""
        presence = self.presences.get(user_id)
        if not presence:
            return False

        old_status = presence.status
        presence.set_status(status, custom_message)

        if old_status != status:
            self._emit_event("status_changed", {
                "user_id": user_id,
                "old_status": old_status.value,
                "new_status": status.value,
                "custom_message": custom_message
            })
        return True

    def heartbeat(self, user_id: str) -> bool:
        """Process heartbeat from user."""
        presence = self.presences.get(user_id)
        if not presence:
            return False

        presence.heartbeat()
        return True

    def get_presence(self, user_id: str) -> Optional[UserPresence]:
        """Get presence for a specific user."""
        return self.presences.get(user_id)

    def get_all_presences(self) -> List[UserPresence]:
        """Get all active presences."""
        return list(self.presences.values())

    def get_cursor_overlays(self) -> List[Dict[str, Any]]:
        """Get cursor data for overlay rendering."""
        return [p.to_cursor_overlay() for p in self.presences.values()]

    def get_users_at_focus(self, focus_type: FocusType) -> List[UserPresence]:
        """Get users focused on a specific element type."""
        return [p for p in self.presences.values() if p.focus == focus_type]

    def get_users_viewing_element(self, element_id: str) -> List[UserPresence]:
        """Get users focused on a specific element."""
        return [
            p for p in self.presences.values()
            if p.focus_element_id == element_id
        ]

    def get_active_typists(self) -> List[Tuple[str, str]]:
        """Get list of (user_id, context) for active typists."""
        result = []
        for presence in self.presences.values():
            if presence.typing and presence.typing.is_active():
                result.append((presence.user_id, presence.typing.context))
        return result

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
                logger.error(f"Presence event handler error: {e}")

    def to_state(self) -> Dict[str, Any]:
        """Export presence state."""
        return {
            "session_id": self.session_id,
            "presences": {k: v.to_dict() for k, v in self.presences.items()},
            "total_users": len(self.presences),
            "active_users": len([p for p in self.presences.values() if p.status == ActivityStatus.ACTIVE]),
            "typing_users": len(self.get_active_typists())
        }


# Factory function
def create_presence_manager(
    session_id: str,
    idle_threshold_minutes: int = 5
) -> PresenceManager:
    """Create a presence manager for a session."""
    return PresenceManager(
        session_id=session_id,
        idle_threshold=timedelta(minutes=idle_threshold_minutes)
    )

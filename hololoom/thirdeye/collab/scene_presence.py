"""
Scene Presence Tracking
=======================

Tracks where users are focused within a ThirdEye scene.

Extends HoloLoom.collaboration.presence with scene-specific focus tracking:
- Which element a user is hovering over
- Where their cursor is in 3D scene space
- What action they're currently performing (select, draw, annotate)
- Their assigned collaboration color

Created: 2025-12-01
Author: HoloLoom Team
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
import asyncio
import logging
import uuid

# Try to import collaboration presence (graceful fallback)
try:
    from hololoom.collaboration.presence import (
        UserPresence,
        PresenceManager,
        ActivityStatus,
        FocusType,
        CursorPosition,
        create_presence_manager,
    )
    PRESENCE_AVAILABLE = True
except ImportError:
    PRESENCE_AVAILABLE = False

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Color Palette for Users
# -------------------------------------------------------------------------

USER_COLORS = [
    {"name": "blue", "hex": "#3B82F6", "rgb": (59, 130, 246)},
    {"name": "green", "hex": "#22C55E", "rgb": (34, 197, 94)},
    {"name": "purple", "hex": "#A855F7", "rgb": (168, 85, 247)},
    {"name": "orange", "hex": "#F97316", "rgb": (249, 115, 22)},
    {"name": "pink", "hex": "#EC4899", "rgb": (236, 72, 153)},
    {"name": "cyan", "hex": "#06B6D4", "rgb": (6, 182, 212)},
    {"name": "yellow", "hex": "#EAB308", "rgb": (234, 179, 8)},
    {"name": "red", "hex": "#EF4444", "rgb": (239, 68, 68)},
]


class CursorMode(Enum):
    """What action the user is performing."""
    SELECT = "select"           # Default - selecting elements
    MOVE = "move"               # Moving an element
    RESIZE = "resize"           # Resizing an element
    DRAW = "draw"               # Drawing new element
    CONNECT = "connect"         # Creating connections
    ANNOTATE = "annotate"       # Adding annotations
    PAN = "pan"                 # Panning the view
    ZOOM = "zoom"               # Zooming
    ROTATE_VIEW = "rotate_view" # Rotating view (3D)


class FocusTarget(Enum):
    """What the user is focused on."""
    NONE = "none"               # No focus
    ELEMENT = "element"         # Focused on specific element
    CONNECTION = "connection"   # Focused on a connection
    CANVAS = "canvas"           # Focused on empty canvas
    UI = "ui"                   # Focused on UI controls
    MENU = "menu"               # In a menu


@dataclass
class SceneCursor:
    """
    User's cursor position in scene space.
    """
    # 3D position in scene coordinates
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # 2D screen position (for 2D views)
    screen_x: float = 0.0
    screen_y: float = 0.0

    # Cursor state
    mode: CursorMode = CursorMode.SELECT
    is_pressed: bool = False
    drag_start: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'screen_x': self.screen_x,
            'screen_y': self.screen_y,
            'mode': self.mode.value,
            'is_pressed': self.is_pressed,
            'drag_start': self.drag_start,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneCursor':
        return cls(
            x=data.get('x', 0.0),
            y=data.get('y', 0.0),
            z=data.get('z', 0.0),
            screen_x=data.get('screen_x', 0.0),
            screen_y=data.get('screen_y', 0.0),
            mode=CursorMode(data.get('mode', 'select')),
            is_pressed=data.get('is_pressed', False),
            drag_start=data.get('drag_start'),
        )


@dataclass
class SceneFocus:
    """
    Complete focus state for a user in the scene.
    """
    user_id: str
    user_name: str = ""

    # Focus target
    target_type: FocusTarget = FocusTarget.NONE
    element_id: Optional[str] = None  # If focused on element
    connection_id: Optional[str] = None  # If focused on connection

    # Cursor
    cursor: SceneCursor = field(default_factory=SceneCursor)

    # Visual identity
    color: Dict[str, Any] = field(default_factory=lambda: USER_COLORS[0])
    avatar_url: Optional[str] = None

    # Selection
    selected_elements: Set[str] = field(default_factory=set)

    # Activity
    is_active: bool = True
    last_activity: datetime = field(default_factory=datetime.now)
    idle_seconds: float = 0.0

    # Typing/editing indicator
    is_typing: bool = False
    typing_in_element: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'user_name': self.user_name,
            'target_type': self.target_type.value,
            'element_id': self.element_id,
            'connection_id': self.connection_id,
            'cursor': self.cursor.to_dict(),
            'color': self.color,
            'avatar_url': self.avatar_url,
            'selected_elements': list(self.selected_elements),
            'is_active': self.is_active,
            'last_activity': self.last_activity.isoformat(),
            'idle_seconds': self.idle_seconds,
            'is_typing': self.is_typing,
            'typing_in_element': self.typing_in_element,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneFocus':
        return cls(
            user_id=data['user_id'],
            user_name=data.get('user_name', ''),
            target_type=FocusTarget(data.get('target_type', 'none')),
            element_id=data.get('element_id'),
            connection_id=data.get('connection_id'),
            cursor=SceneCursor.from_dict(data.get('cursor', {})),
            color=data.get('color', USER_COLORS[0]),
            avatar_url=data.get('avatar_url'),
            selected_elements=set(data.get('selected_elements', [])),
            is_active=data.get('is_active', True),
            last_activity=datetime.fromisoformat(data['last_activity']) if 'last_activity' in data else datetime.now(),
            idle_seconds=data.get('idle_seconds', 0.0),
            is_typing=data.get('is_typing', False),
            typing_in_element=data.get('typing_in_element'),
        )


class ScenePresenceTracker:
    """
    Tracks presence and focus of all users in a collaborative scene.

    Provides:
    - Real-time cursor positions for all users
    - Focus indicators (which element each user is on)
    - Selection state (multi-select tracking)
    - Typing/editing indicators
    - Idle detection

    Usage:
        tracker = ScenePresenceTracker(session_id="session_123", user_id="user_1")

        # Update local cursor
        await tracker.update_cursor(x=100, y=50, z=0)

        # Focus on element
        await tracker.focus_element("element_id")

        # Select elements
        await tracker.select_elements(["elem_1", "elem_2"])

        # Get all users' focus states
        all_focus = tracker.get_all_focus()
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        user_name: str = "",
        broadcast_interval: float = 0.05,  # 20 FPS cursor updates
        idle_timeout: float = 30.0,  # Seconds before marked idle
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.user_name = user_name or f"User {user_id[:4]}"
        self.broadcast_interval = broadcast_interval
        self.idle_timeout = idle_timeout

        # Assign color based on user_id hash
        color_index = hash(user_id) % len(USER_COLORS)
        self._my_color = USER_COLORS[color_index]

        # Local focus state
        self._my_focus = SceneFocus(
            user_id=user_id,
            user_name=self.user_name,
            color=self._my_color,
        )

        # All users' focus states
        self._focus_states: Dict[str, SceneFocus] = {
            user_id: self._my_focus
        }

        # Presence manager (if available)
        self._presence: Optional['PresenceManager'] = None
        if PRESENCE_AVAILABLE:
            self._init_presence()

        # Event handlers
        self._focus_handlers: List[Callable[[str, SceneFocus], None]] = []
        self._cursor_handlers: List[Callable[[str, SceneCursor], None]] = []

        # Background tasks
        self._broadcast_task: Optional[asyncio.Task] = None
        self._idle_check_task: Optional[asyncio.Task] = None

        # Cursor throttling
        self._cursor_dirty = False
        self._last_broadcast = datetime.now()

    def _init_presence(self):
        """Initialize presence manager."""
        try:
            self._presence = create_presence_manager(
                session_id=self.session_id,
            )
            logger.info(f"Presence initialized for {self.user_id}")
        except Exception as e:
            logger.warning(f"Could not initialize presence: {e}")
            self._presence = None

    # -------------------------------------------------------------------------
    # Cursor Updates
    # -------------------------------------------------------------------------

    async def update_cursor(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        screen_x: Optional[float] = None,
        screen_y: Optional[float] = None,
        mode: Optional[CursorMode] = None,
        is_pressed: bool = False,
    ):
        """
        Update local cursor position.

        Throttled to broadcast_interval to avoid flooding.
        """
        cursor = self._my_focus.cursor
        cursor.x = x
        cursor.y = y
        cursor.z = z
        if screen_x is not None:
            cursor.screen_x = screen_x
        if screen_y is not None:
            cursor.screen_y = screen_y
        if mode is not None:
            cursor.mode = mode
        cursor.is_pressed = is_pressed

        # Mark activity
        self._my_focus.last_activity = datetime.now()
        self._my_focus.is_active = True
        self._my_focus.idle_seconds = 0.0

        # Mark dirty for broadcast
        self._cursor_dirty = True

        # Notify local handlers immediately
        for handler in self._cursor_handlers:
            try:
                handler(self.user_id, cursor)
            except Exception as e:
                logger.error(f"Cursor handler error: {e}")

    async def start_drag(self, x: float, y: float, z: float = 0.0):
        """Mark start of drag operation."""
        self._my_focus.cursor.drag_start = {'x': x, 'y': y, 'z': z}
        self._my_focus.cursor.is_pressed = True

    async def end_drag(self):
        """Mark end of drag operation."""
        self._my_focus.cursor.drag_start = None
        self._my_focus.cursor.is_pressed = False

    # -------------------------------------------------------------------------
    # Focus Updates
    # -------------------------------------------------------------------------

    async def focus_element(self, element_id: str):
        """Focus on a specific element."""
        self._my_focus.target_type = FocusTarget.ELEMENT
        self._my_focus.element_id = element_id
        self._my_focus.connection_id = None
        self._my_focus.last_activity = datetime.now()

        await self._broadcast_focus()
        self._notify_focus_change()

    async def focus_connection(self, connection_id: str):
        """Focus on a connection."""
        self._my_focus.target_type = FocusTarget.CONNECTION
        self._my_focus.element_id = None
        self._my_focus.connection_id = connection_id
        self._my_focus.last_activity = datetime.now()

        await self._broadcast_focus()
        self._notify_focus_change()

    async def focus_canvas(self):
        """Focus on empty canvas."""
        self._my_focus.target_type = FocusTarget.CANVAS
        self._my_focus.element_id = None
        self._my_focus.connection_id = None
        self._my_focus.last_activity = datetime.now()

        await self._broadcast_focus()
        self._notify_focus_change()

    async def clear_focus(self):
        """Clear focus."""
        self._my_focus.target_type = FocusTarget.NONE
        self._my_focus.element_id = None
        self._my_focus.connection_id = None

        await self._broadcast_focus()
        self._notify_focus_change()

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    async def select_elements(self, element_ids: List[str], add: bool = False):
        """
        Select elements.

        Args:
            element_ids: Elements to select
            add: If True, add to selection. If False, replace selection.
        """
        if add:
            self._my_focus.selected_elements.update(element_ids)
        else:
            self._my_focus.selected_elements = set(element_ids)

        self._my_focus.last_activity = datetime.now()
        await self._broadcast_focus()
        self._notify_focus_change()

    async def deselect_elements(self, element_ids: List[str]):
        """Remove elements from selection."""
        self._my_focus.selected_elements -= set(element_ids)
        await self._broadcast_focus()
        self._notify_focus_change()

    async def clear_selection(self):
        """Clear all selection."""
        self._my_focus.selected_elements.clear()
        await self._broadcast_focus()
        self._notify_focus_change()

    # -------------------------------------------------------------------------
    # Typing Indicator
    # -------------------------------------------------------------------------

    async def start_typing(self, element_id: Optional[str] = None):
        """Mark user as typing."""
        self._my_focus.is_typing = True
        self._my_focus.typing_in_element = element_id
        self._my_focus.last_activity = datetime.now()
        await self._broadcast_focus()

    async def stop_typing(self):
        """Mark user as not typing."""
        self._my_focus.is_typing = False
        self._my_focus.typing_in_element = None
        await self._broadcast_focus()

    # -------------------------------------------------------------------------
    # Cursor Mode
    # -------------------------------------------------------------------------

    async def set_cursor_mode(self, mode: CursorMode):
        """Set cursor mode."""
        self._my_focus.cursor.mode = mode
        self._cursor_dirty = True

    # -------------------------------------------------------------------------
    # Remote Updates
    # -------------------------------------------------------------------------

    async def apply_remote_focus(self, user_id: str, focus_data: Dict[str, Any]):
        """Apply focus update from remote user."""
        if user_id == self.user_id:
            return  # Ignore own updates

        focus = SceneFocus.from_dict(focus_data)
        self._focus_states[user_id] = focus

        # Notify handlers
        for handler in self._focus_handlers:
            try:
                handler(user_id, focus)
            except Exception as e:
                logger.error(f"Focus handler error: {e}")

    async def apply_remote_cursor(self, user_id: str, cursor_data: Dict[str, Any]):
        """Apply cursor update from remote user."""
        if user_id == self.user_id:
            return

        if user_id not in self._focus_states:
            # Create new focus state for unknown user
            color_index = hash(user_id) % len(USER_COLORS)
            self._focus_states[user_id] = SceneFocus(
                user_id=user_id,
                color=USER_COLORS[color_index],
            )

        cursor = SceneCursor.from_dict(cursor_data)
        self._focus_states[user_id].cursor = cursor
        self._focus_states[user_id].last_activity = datetime.now()
        self._focus_states[user_id].is_active = True

        # Notify handlers
        for handler in self._cursor_handlers:
            try:
                handler(user_id, cursor)
            except Exception as e:
                logger.error(f"Cursor handler error: {e}")

    async def user_left(self, user_id: str):
        """Handle user leaving."""
        if user_id in self._focus_states:
            del self._focus_states[user_id]
            logger.info(f"User {user_id} presence removed")

    # -------------------------------------------------------------------------
    # Broadcasting
    # -------------------------------------------------------------------------

    async def _broadcast_focus(self):
        """Broadcast focus state to all users."""
        if self._presence:
            try:
                await self._presence.broadcast_state({
                    'type': 'scene_focus',
                    'user_id': self.user_id,
                    'focus': self._my_focus.to_dict(),
                })
            except Exception as e:
                logger.error(f"Focus broadcast failed: {e}")

    async def _broadcast_cursor(self):
        """Broadcast cursor position to all users."""
        if not self._cursor_dirty:
            return

        self._cursor_dirty = False

        if self._presence:
            try:
                await self._presence.broadcast_state({
                    'type': 'scene_cursor',
                    'user_id': self.user_id,
                    'cursor': self._my_focus.cursor.to_dict(),
                })
            except Exception as e:
                logger.error(f"Cursor broadcast failed: {e}")

    async def _broadcast_loop(self):
        """Background loop for cursor broadcasts."""
        while True:
            await asyncio.sleep(self.broadcast_interval)
            await self._broadcast_cursor()

    async def _idle_check_loop(self):
        """Background loop for idle detection."""
        while True:
            await asyncio.sleep(1.0)

            now = datetime.now()

            # Check local idle
            idle = (now - self._my_focus.last_activity).total_seconds()
            self._my_focus.idle_seconds = idle
            if idle > self.idle_timeout:
                self._my_focus.is_active = False

            # Check remote idle
            for user_id, focus in list(self._focus_states.items()):
                if user_id == self.user_id:
                    continue

                idle = (now - focus.last_activity).total_seconds()
                focus.idle_seconds = idle
                if idle > self.idle_timeout:
                    focus.is_active = False

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def on_focus_change(self, handler: Callable[[str, SceneFocus], None]):
        """Subscribe to focus changes (any user)."""
        self._focus_handlers.append(handler)

    def on_cursor_move(self, handler: Callable[[str, SceneCursor], None]):
        """Subscribe to cursor movements (any user)."""
        self._cursor_handlers.append(handler)

    def _notify_focus_change(self):
        """Notify local handlers of focus change."""
        for handler in self._focus_handlers:
            try:
                handler(self.user_id, self._my_focus)
            except Exception as e:
                logger.error(f"Focus handler error: {e}")

    # -------------------------------------------------------------------------
    # State Access
    # -------------------------------------------------------------------------

    def get_my_focus(self) -> SceneFocus:
        """Get local user's focus state."""
        return self._my_focus

    def get_focus(self, user_id: str) -> Optional[SceneFocus]:
        """Get specific user's focus state."""
        return self._focus_states.get(user_id)

    def get_all_focus(self) -> Dict[str, SceneFocus]:
        """Get all users' focus states."""
        return dict(self._focus_states)

    def get_active_users(self) -> List[str]:
        """Get list of active user IDs."""
        return [
            user_id for user_id, focus in self._focus_states.items()
            if focus.is_active
        ]

    def get_users_on_element(self, element_id: str) -> List[str]:
        """Get users focused on specific element."""
        return [
            user_id for user_id, focus in self._focus_states.items()
            if focus.element_id == element_id
        ]

    def get_users_selecting_element(self, element_id: str) -> List[str]:
        """Get users who have element selected."""
        return [
            user_id for user_id, focus in self._focus_states.items()
            if element_id in focus.selected_elements
        ]

    def is_element_locked(self, element_id: str) -> Optional[str]:
        """
        Check if element is being edited by another user.

        Returns user_id if locked, None otherwise.
        """
        for user_id, focus in self._focus_states.items():
            if user_id == self.user_id:
                continue

            # Locked if user is typing in this element
            if focus.is_typing and focus.typing_in_element == element_id:
                return user_id

            # Locked if user is dragging this element
            if focus.cursor.is_pressed and focus.element_id == element_id:
                return user_id

        return None

    @property
    def my_color(self) -> Dict[str, Any]:
        """Get local user's assigned color."""
        return self._my_color

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self):
        """Start background tasks."""
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        self._idle_check_task = asyncio.create_task(self._idle_check_loop())
        logger.info(f"Scene presence started for {self.user_id}")

    async def stop(self):
        """Stop background tasks."""
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        if self._idle_check_task:
            self._idle_check_task.cancel()
            try:
                await self._idle_check_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Scene presence stopped for {self.user_id}")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


def create_scene_presence(
    session_id: str,
    user_id: str,
    user_name: str = "",
    **kwargs,
) -> ScenePresenceTracker:
    """
    Factory function to create scene presence tracker.

    Args:
        session_id: Unique session identifier
        user_id: Current user's ID
        user_name: Display name
        **kwargs: Additional options

    Returns:
        ScenePresenceTracker instance
    """
    return ScenePresenceTracker(
        session_id=session_id,
        user_id=user_id,
        user_name=user_name,
        **kwargs,
    )


__all__ = [
    'ScenePresenceTracker',
    'SceneFocus',
    'SceneCursor',
    'FocusTarget',
    'CursorMode',
    'USER_COLORS',
    'create_scene_presence',
]

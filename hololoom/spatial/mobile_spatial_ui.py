"""
Mobile-Responsive Spatial UI for AR/VR Experiences.

Provides adaptive UI components that work across mobile, tablet, and desktop:
- Touch-optimized controls for mobile AR
- Responsive layouts for different screen sizes
- Mobile gesture recognition (pinch, swipe, tap, drag)
- Portrait/landscape adaptation
- Performance optimization for mobile devices

Phase 4: Spatial Computing Enhancement - Mobile UI
Created: November 2025
"""

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Import spatial math types
from hololoom.spatial.math_types import Color, Quaternion, Vector3

logger = logging.getLogger(__name__)


# ============================================================================
# Device & Screen Enums
# ============================================================================

class DeviceType(Enum):
    """Device categories for responsive design."""
    MOBILE_PHONE = "mobile_phone"      # <768px, touch primary
    TABLET = "tablet"                   # 768-1024px, touch or stylus
    DESKTOP = "desktop"                 # >1024px, mouse + keyboard
    VR_HEADSET = "vr_headset"          # Immersive, controllers
    AR_GLASSES = "ar_glasses"          # See-through, gaze + gesture


class ScreenOrientation(Enum):
    """Screen orientation modes."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SQUARE = "square"  # For certain tablets/foldables


class InputMode(Enum):
    """Primary input mode for the device."""
    TOUCH = "touch"           # Touch screen
    MOUSE = "mouse"           # Mouse + keyboard
    CONTROLLER = "controller"  # VR/game controllers
    GAZE = "gaze"             # Eye tracking + dwell
    GESTURE = "gesture"       # Hand tracking
    VOICE = "voice"           # Voice commands


class UIScale(Enum):
    """UI density/scale modes."""
    COMPACT = "compact"    # Dense UI for power users
    NORMAL = "normal"      # Standard balanced UI
    COMFORTABLE = "comfortable"  # Larger touch targets
    ACCESSIBLE = "accessible"    # Maximum accessibility


# ============================================================================
# Touch Gesture Types
# ============================================================================

class TouchGestureType(Enum):
    """Touch gesture categories."""
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH_IN = "pinch_in"      # Zoom out
    PINCH_OUT = "pinch_out"    # Zoom in
    ROTATE = "rotate"          # Two-finger rotation
    PAN = "pan"                # Drag/pan
    EDGE_SWIPE_LEFT = "edge_swipe_left"
    EDGE_SWIPE_RIGHT = "edge_swipe_right"


@dataclass
class TouchGesture:
    """Detected touch gesture event."""
    gesture_type: TouchGestureType
    position: tuple[float, float]  # Screen position (0-1 normalized)
    delta: tuple[float, float] = (0.0, 0.0)  # Movement delta
    scale: float = 1.0  # For pinch gestures
    rotation: float = 0.0  # For rotate gestures (radians)
    velocity: float = 0.0  # Gesture velocity
    finger_count: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gesture_type": self.gesture_type.value,
            "position": self.position,
            "delta": self.delta,
            "scale": self.scale,
            "rotation": self.rotation,
            "velocity": self.velocity,
            "finger_count": self.finger_count,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# Device Capabilities
# ============================================================================

@dataclass
class DeviceCapabilities:
    """Detected device capabilities."""
    device_type: DeviceType
    screen_width: int
    screen_height: int
    pixel_ratio: float = 1.0
    orientation: ScreenOrientation = ScreenOrientation.PORTRAIT

    # Input capabilities
    has_touch: bool = False
    has_multi_touch: bool = False
    has_stylus: bool = False
    has_mouse: bool = False
    has_keyboard: bool = False
    has_accelerometer: bool = False
    has_gyroscope: bool = False
    has_camera: bool = False
    has_ar_support: bool = False
    has_vr_support: bool = False

    # Performance hints
    gpu_tier: int = 2  # 1=low, 2=mid, 3=high
    memory_gb: float = 4.0
    is_low_power_mode: bool = False
    preferred_frame_rate: int = 60

    # Network
    connection_type: str = "wifi"  # wifi, cellular, ethernet
    estimated_bandwidth_mbps: float = 10.0

    def get_primary_input(self) -> InputMode:
        """Determine primary input mode."""
        if self.device_type == DeviceType.VR_HEADSET:
            return InputMode.CONTROLLER
        elif self.device_type == DeviceType.AR_GLASSES:
            return InputMode.GAZE
        elif self.has_touch:
            return InputMode.TOUCH
        else:
            return InputMode.MOUSE

    def get_recommended_scale(self) -> UIScale:
        """Recommend UI scale based on device."""
        if self.device_type == DeviceType.MOBILE_PHONE:
            return UIScale.COMFORTABLE
        elif self.device_type == DeviceType.TABLET:
            return UIScale.NORMAL
        elif self.is_low_power_mode:
            return UIScale.COMPACT
        else:
            return UIScale.NORMAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type.value,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "pixel_ratio": self.pixel_ratio,
            "orientation": self.orientation.value,
            "has_touch": self.has_touch,
            "has_ar_support": self.has_ar_support,
            "gpu_tier": self.gpu_tier,
            "primary_input": self.get_primary_input().value
        }


# ============================================================================
# Responsive Layout
# ============================================================================

class LayoutBreakpoint(Enum):
    """Responsive layout breakpoints."""
    XS = 0      # 0-479px (small phones)
    SM = 480    # 480-767px (large phones)
    MD = 768    # 768-1023px (tablets)
    LG = 1024   # 1024-1439px (small desktop)
    XL = 1440   # 1440px+ (large desktop)


@dataclass
class ResponsiveValue:
    """Value that adapts to breakpoints."""
    xs: Any = None
    sm: Any = None
    md: Any = None
    lg: Any = None
    xl: Any = None

    def get_for_width(self, width: int) -> Any:
        """Get value for screen width."""
        if width >= 1440 and self.xl is not None:
            return self.xl
        elif width >= 1024 and self.lg is not None:
            return self.lg
        elif width >= 768 and self.md is not None:
            return self.md
        elif width >= 480 and self.sm is not None:
            return self.sm
        elif self.xs is not None:
            return self.xs
        # Fallback to first non-None value
        for val in [self.xs, self.sm, self.md, self.lg, self.xl]:
            if val is not None:
                return val
        return None


@dataclass
class LayoutConstraints:
    """Constraints for UI layout."""
    min_width: float = 0.0
    max_width: float = float('inf')
    min_height: float = 0.0
    max_height: float = float('inf')
    aspect_ratio: float | None = None  # width/height
    margin: float = 8.0
    padding: float = 8.0
    safe_area_insets: tuple[float, float, float, float] = (0, 0, 0, 0)  # top, right, bottom, left


# ============================================================================
# Mobile UI Components
# ============================================================================

class UIComponentType(Enum):
    """Types of UI components."""
    BUTTON = "button"
    FLOATING_BUTTON = "floating_button"
    PANEL = "panel"
    CARD = "card"
    MENU = "menu"
    RADIAL_MENU = "radial_menu"
    TOOLBAR = "toolbar"
    BOTTOM_SHEET = "bottom_sheet"
    MODAL = "modal"
    TOAST = "toast"
    TOOLTIP = "tooltip"
    HUD = "hud"
    MINIMAP = "minimap"
    COMPASS = "compass"


class UIAnchor(Enum):
    """UI anchoring positions."""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class UIComponent:
    """Base UI component."""
    component_id: str
    component_type: UIComponentType
    anchor: UIAnchor = UIAnchor.CENTER

    # Position (relative to anchor, in normalized 0-1 coords)
    offset_x: float = 0.0
    offset_y: float = 0.0

    # Size
    width: ResponsiveValue = field(default_factory=lambda: ResponsiveValue(xs=1.0, md=0.5))
    height: ResponsiveValue = field(default_factory=lambda: ResponsiveValue(xs=0.1, md=0.1))

    # Appearance
    visible: bool = True
    opacity: float = 1.0
    z_index: int = 0

    # Behavior
    interactive: bool = True
    touch_target_padding: float = 8.0  # Extra touch area

    # State
    is_pressed: bool = False
    is_focused: bool = False
    is_disabled: bool = False

    # Content
    label: str | None = None
    icon: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_computed_position(self, screen_width: int, screen_height: int,
                             safe_area: tuple[float, float, float, float]) -> tuple[float, float]:
        """Calculate actual screen position."""
        top, right, bottom, left = safe_area

        # Anchor base positions
        anchor_positions = {
            UIAnchor.TOP_LEFT: (left, top),
            UIAnchor.TOP_CENTER: (screen_width / 2, top),
            UIAnchor.TOP_RIGHT: (screen_width - right, top),
            UIAnchor.CENTER_LEFT: (left, screen_height / 2),
            UIAnchor.CENTER: (screen_width / 2, screen_height / 2),
            UIAnchor.CENTER_RIGHT: (screen_width - right, screen_height / 2),
            UIAnchor.BOTTOM_LEFT: (left, screen_height - bottom),
            UIAnchor.BOTTOM_CENTER: (screen_width / 2, screen_height - bottom),
            UIAnchor.BOTTOM_RIGHT: (screen_width - right, screen_height - bottom),
        }

        base_x, base_y = anchor_positions[self.anchor]
        return (
            base_x + self.offset_x * screen_width,
            base_y + self.offset_y * screen_height
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "anchor": self.anchor.value,
            "visible": self.visible,
            "interactive": self.interactive,
            "label": self.label,
            "icon": self.icon
        }


@dataclass
class FloatingActionButton(UIComponent):
    """Mobile-style floating action button."""
    size: float = 56.0  # Standard FAB size
    mini: bool = False  # Mini FAB (40px)
    extended: bool = False  # Extended FAB with label
    color: Color = field(default_factory=lambda: Color(0.3, 0.5, 1.0, 1.0))
    elevation: float = 6.0

    def __post_init__(self):
        self.component_type = UIComponentType.FLOATING_BUTTON
        if self.mini:
            self.size = 40.0
        self.touch_target_padding = max(8.0, (48.0 - self.size) / 2)


@dataclass
class BottomSheet(UIComponent):
    """Mobile bottom sheet / drawer."""
    height_collapsed: float = 0.1  # 10% of screen
    height_expanded: float = 0.8   # 80% of screen
    height_half: float = 0.5       # Half expansion

    current_state: str = "collapsed"  # collapsed, half, expanded
    draggable: bool = True
    snap_points: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.8])

    # Content
    header_height: float = 32.0
    content_scrollable: bool = True

    def __post_init__(self):
        self.component_type = UIComponentType.BOTTOM_SHEET
        self.anchor = UIAnchor.BOTTOM_CENTER
        self.width = ResponsiveValue(xs=1.0, md=0.8, lg=0.6)


@dataclass
class RadialMenu(UIComponent):
    """Radial/pie menu for spatial interaction."""
    radius: float = 100.0
    inner_radius: float = 30.0
    items: list[dict[str, Any]] = field(default_factory=list)
    open_gesture: TouchGestureType = TouchGestureType.LONG_PRESS

    is_open: bool = False
    selected_index: int | None = None

    def __post_init__(self):
        self.component_type = UIComponentType.RADIAL_MENU

    def get_item_angle(self, index: int) -> float:
        """Get angle for item (radians)."""
        if not self.items:
            return 0.0
        angle_per_item = 2 * math.pi / len(self.items)
        return index * angle_per_item - math.pi / 2  # Start from top

    def get_item_position(self, index: int) -> tuple[float, float]:
        """Get position offset for item."""
        angle = self.get_item_angle(index)
        mid_radius = (self.radius + self.inner_radius) / 2
        return (
            math.cos(angle) * mid_radius,
            math.sin(angle) * mid_radius
        )


@dataclass
class SpatialHUD:
    """Head-up display for AR/VR."""
    hud_id: str
    visible: bool = True
    follow_gaze: bool = True
    follow_distance: float = 2.0  # Meters from viewer

    # Elements
    elements: dict[str, UIComponent] = field(default_factory=dict)

    # Layout zones
    top_bar_height: float = 0.08    # 8% of view
    bottom_bar_height: float = 0.1  # 10% of view
    side_margin: float = 0.05       # 5% margin

    # Performance
    update_rate_hz: float = 30.0
    lazy_update: bool = True  # Only update when changed

    def add_element(self, component: UIComponent):
        """Add HUD element."""
        self.elements[component.component_id] = component

    def remove_element(self, component_id: str):
        """Remove HUD element."""
        self.elements.pop(component_id, None)

    def get_world_position(self, viewer_position: Vector3,
                          viewer_rotation: Quaternion) -> Vector3:
        """Get HUD world position based on viewer."""
        # HUD stays at fixed distance in front of viewer
        forward = viewer_rotation.rotate_vector(Vector3(0, 0, -1))
        return Vector3(
            viewer_position.x + forward.x * self.follow_distance,
            viewer_position.y + forward.y * self.follow_distance,
            viewer_position.z + forward.z * self.follow_distance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hud_id": self.hud_id,
            "visible": self.visible,
            "follow_gaze": self.follow_gaze,
            "element_count": len(self.elements)
        }


# ============================================================================
# Touch Gesture Recognition
# ============================================================================

@dataclass
class TouchPoint:
    """Single touch point tracking."""
    touch_id: int
    position: tuple[float, float]
    start_position: tuple[float, float]
    start_time: datetime
    last_position: tuple[float, float]
    last_time: datetime
    is_active: bool = True


class TouchGestureRecognizer:
    """
    Recognizes touch gestures from raw touch events.

    Handles single-touch (tap, swipe, long press) and
    multi-touch (pinch, rotate) gestures.
    """

    def __init__(
        self,
        tap_threshold_ms: float = 200.0,
        long_press_threshold_ms: float = 500.0,
        swipe_min_distance: float = 0.1,  # 10% of screen
        swipe_max_time_ms: float = 300.0,
        edge_threshold: float = 0.05  # 5% from edge
    ):
        self.tap_threshold_ms = tap_threshold_ms
        self.long_press_threshold_ms = long_press_threshold_ms
        self.swipe_min_distance = swipe_min_distance
        self.swipe_max_time_ms = swipe_max_time_ms
        self.edge_threshold = edge_threshold

        self.active_touches: dict[int, TouchPoint] = {}
        self.gesture_handlers: dict[TouchGestureType, list[Callable]] = {}

        # Multi-touch state
        self._initial_pinch_distance: float | None = None
        self._initial_rotation: float | None = None

        # Long press detection
        self._long_press_task: asyncio.Task | None = None

    def on_touch_start(self, touch_id: int, x: float, y: float) -> TouchGesture | None:
        """Handle touch start event."""
        now = datetime.now()
        self.active_touches[touch_id] = TouchPoint(
            touch_id=touch_id,
            position=(x, y),
            start_position=(x, y),
            start_time=now,
            last_position=(x, y),
            last_time=now
        )

        # Start long press detection
        if len(self.active_touches) == 1:
            self._start_long_press_detection(touch_id, x, y)

        # Initialize multi-touch tracking
        if len(self.active_touches) == 2:
            self._cancel_long_press()
            self._init_multi_touch()

        return None

    def on_touch_move(self, touch_id: int, x: float, y: float) -> TouchGesture | None:
        """Handle touch move event."""
        if touch_id not in self.active_touches:
            return None

        touch = self.active_touches[touch_id]
        touch.last_position = touch.position
        touch.last_time = datetime.now()
        touch.position = (x, y)

        # Cancel long press if moved too much
        dx = abs(x - touch.start_position[0])
        dy = abs(y - touch.start_position[1])
        if dx > 0.02 or dy > 0.02:  # 2% movement
            self._cancel_long_press()

        # Multi-touch gestures
        if len(self.active_touches) == 2:
            return self._detect_multi_touch_gesture()

        # Pan gesture
        if len(self.active_touches) == 1:
            delta = (x - touch.last_position[0], y - touch.last_position[1])
            return TouchGesture(
                gesture_type=TouchGestureType.PAN,
                position=(x, y),
                delta=delta,
                finger_count=1
            )

        return None

    def on_touch_end(self, touch_id: int) -> TouchGesture | None:
        """Handle touch end event."""
        if touch_id not in self.active_touches:
            return None

        touch = self.active_touches.pop(touch_id)
        self._cancel_long_press()

        now = datetime.now()
        duration_ms = (now - touch.start_time).total_seconds() * 1000

        dx = touch.position[0] - touch.start_position[0]
        dy = touch.position[1] - touch.start_position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Detect gesture type
        if distance < 0.02 and duration_ms < self.tap_threshold_ms:
            # Check for double tap
            return TouchGesture(
                gesture_type=TouchGestureType.TAP,
                position=touch.position,
                finger_count=1
            )

        elif distance >= self.swipe_min_distance and duration_ms < self.swipe_max_time_ms:
            # Swipe detection
            velocity = distance / (duration_ms / 1000) if duration_ms > 0 else 0
            swipe_type = self._classify_swipe(dx, dy, touch.start_position)

            return TouchGesture(
                gesture_type=swipe_type,
                position=touch.position,
                delta=(dx, dy),
                velocity=velocity,
                finger_count=1
            )

        return None

    def _start_long_press_detection(self, touch_id: int, x: float, y: float):
        """Start async long press detection."""
        async def detect_long_press():
            await asyncio.sleep(self.long_press_threshold_ms / 1000)
            if touch_id in self.active_touches:
                touch = self.active_touches[touch_id]
                dx = abs(touch.position[0] - touch.start_position[0])
                dy = abs(touch.position[1] - touch.start_position[1])
                if dx < 0.02 and dy < 0.02:  # Hasn't moved much
                    gesture = TouchGesture(
                        gesture_type=TouchGestureType.LONG_PRESS,
                        position=touch.position,
                        finger_count=1
                    )
                    self._emit_gesture(gesture)

        self._long_press_task = asyncio.create_task(detect_long_press())

    def _cancel_long_press(self):
        """Cancel pending long press."""
        if self._long_press_task:
            self._long_press_task.cancel()
            self._long_press_task = None

    def _init_multi_touch(self):
        """Initialize multi-touch tracking."""
        if len(self.active_touches) != 2:
            return

        touches = list(self.active_touches.values())
        p1, p2 = touches[0].position, touches[1].position

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        self._initial_pinch_distance = math.sqrt(dx * dx + dy * dy)
        self._initial_rotation = math.atan2(dy, dx)

    def _detect_multi_touch_gesture(self) -> TouchGesture | None:
        """Detect pinch/rotate gesture."""
        if len(self.active_touches) != 2:
            return None

        touches = list(self.active_touches.values())
        p1, p2 = touches[0].position, touches[1].position

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        current_distance = math.sqrt(dx * dx + dy * dy)
        current_rotation = math.atan2(dy, dx)

        center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        # Calculate scale (pinch)
        scale = 1.0
        if self._initial_pinch_distance and self._initial_pinch_distance > 0:
            scale = current_distance / self._initial_pinch_distance

        # Calculate rotation
        rotation = 0.0
        if self._initial_rotation is not None:
            rotation = current_rotation - self._initial_rotation

        # Determine gesture type
        scale_threshold = 0.1  # 10% change
        rotation_threshold = 0.1  # ~6 degrees

        if abs(scale - 1.0) > scale_threshold:
            gesture_type = TouchGestureType.PINCH_OUT if scale > 1.0 else TouchGestureType.PINCH_IN
            return TouchGesture(
                gesture_type=gesture_type,
                position=center,
                scale=scale,
                rotation=rotation,
                finger_count=2
            )
        elif abs(rotation) > rotation_threshold:
            return TouchGesture(
                gesture_type=TouchGestureType.ROTATE,
                position=center,
                scale=scale,
                rotation=rotation,
                finger_count=2
            )

        return None

    def _classify_swipe(self, dx: float, dy: float,
                       start_pos: tuple[float, float]) -> TouchGestureType:
        """Classify swipe direction."""
        # Check for edge swipes
        is_left_edge = start_pos[0] < self.edge_threshold
        is_right_edge = start_pos[0] > 1.0 - self.edge_threshold

        if abs(dx) > abs(dy):
            # Horizontal swipe
            if dx > 0:
                return TouchGestureType.EDGE_SWIPE_LEFT if is_left_edge else TouchGestureType.SWIPE_RIGHT
            else:
                return TouchGestureType.EDGE_SWIPE_RIGHT if is_right_edge else TouchGestureType.SWIPE_LEFT
        else:
            # Vertical swipe
            return TouchGestureType.SWIPE_UP if dy < 0 else TouchGestureType.SWIPE_DOWN

    def on(self, gesture_type: TouchGestureType, handler: Callable[[TouchGesture], None]):
        """Register gesture handler."""
        if gesture_type not in self.gesture_handlers:
            self.gesture_handlers[gesture_type] = []
        self.gesture_handlers[gesture_type].append(handler)

    def _emit_gesture(self, gesture: TouchGesture):
        """Emit gesture to handlers."""
        handlers = self.gesture_handlers.get(gesture.gesture_type, [])
        for handler in handlers:
            try:
                handler(gesture)
            except Exception as e:
                logger.error(f"Gesture handler error: {e}")


# ============================================================================
# Responsive UI Manager
# ============================================================================

class MobileSpatialUIManager:
    """
    Manages mobile-responsive spatial UI.

    Handles:
    - Device capability detection
    - Responsive layout management
    - Touch gesture routing
    - Performance optimization
    """

    def __init__(self, initial_capabilities: DeviceCapabilities | None = None):
        self.capabilities = initial_capabilities or DeviceCapabilities(
            device_type=DeviceType.MOBILE_PHONE,
            screen_width=375,
            screen_height=812,
            has_touch=True
        )

        self.components: dict[str, UIComponent] = {}
        self.hud: SpatialHUD | None = None
        self.gesture_recognizer = TouchGestureRecognizer()

        # UI state
        self.active_modal: str | None = None
        self.active_bottom_sheet: str | None = None
        self.ui_scale = UIScale.NORMAL

        # Performance
        self.render_quality: float = 1.0
        self.reduce_motion: bool = False

        # Event handlers
        self._ui_event_handlers: dict[str, list[Callable]] = {}

    def update_capabilities(self, capabilities: DeviceCapabilities):
        """Update device capabilities and adapt UI."""
        self.capabilities = capabilities
        self.ui_scale = capabilities.get_recommended_scale()

        # Adjust render quality based on GPU tier
        if capabilities.gpu_tier == 1:
            self.render_quality = 0.5
        elif capabilities.gpu_tier == 2:
            self.render_quality = 0.75
        else:
            self.render_quality = 1.0

        # Adapt all components
        self._adapt_components_to_screen()

        logger.info(f"UI adapted for {capabilities.device_type.value}, "
                   f"scale: {self.ui_scale.value}")

    def add_component(self, component: UIComponent):
        """Add UI component."""
        self.components[component.component_id] = component

    def remove_component(self, component_id: str):
        """Remove UI component."""
        self.components.pop(component_id, None)

    def create_floating_button(
        self,
        component_id: str,
        icon: str,
        anchor: UIAnchor = UIAnchor.BOTTOM_RIGHT,
        action: Callable | None = None
    ) -> FloatingActionButton:
        """Create and add floating action button."""
        fab = FloatingActionButton(
            component_id=component_id,
            component_type=UIComponentType.FLOATING_BUTTON,
            anchor=anchor,
            offset_x=-0.05 if "right" in anchor.value else 0.05,
            offset_y=-0.05 if "bottom" in anchor.value else 0.05,
            icon=icon
        )

        self.add_component(fab)

        if action:
            self.on_component_action(component_id, action)

        return fab

    def create_bottom_sheet(
        self,
        component_id: str,
        header_label: str = ""
    ) -> BottomSheet:
        """Create and add bottom sheet."""
        sheet = BottomSheet(
            component_id=component_id,
            component_type=UIComponentType.BOTTOM_SHEET,
            label=header_label
        )

        self.add_component(sheet)
        return sheet

    def create_radial_menu(
        self,
        component_id: str,
        items: list[dict[str, Any]]
    ) -> RadialMenu:
        """Create radial menu."""
        menu = RadialMenu(
            component_id=component_id,
            component_type=UIComponentType.RADIAL_MENU,
            items=items
        )

        self.add_component(menu)
        return menu

    def create_hud(self, hud_id: str = "main_hud") -> SpatialHUD:
        """Create spatial HUD."""
        self.hud = SpatialHUD(hud_id=hud_id)
        return self.hud

    def setup_gesture_handlers(self):
        """Setup default gesture handlers."""
        # Pinch to zoom
        self.gesture_recognizer.on(TouchGestureType.PINCH_OUT, self._on_zoom_in)
        self.gesture_recognizer.on(TouchGestureType.PINCH_IN, self._on_zoom_out)

        # Swipe for navigation
        self.gesture_recognizer.on(TouchGestureType.SWIPE_LEFT, self._on_swipe_left)
        self.gesture_recognizer.on(TouchGestureType.SWIPE_RIGHT, self._on_swipe_right)

        # Long press for context menu
        self.gesture_recognizer.on(TouchGestureType.LONG_PRESS, self._on_long_press)

        # Edge swipes for drawers
        self.gesture_recognizer.on(TouchGestureType.EDGE_SWIPE_LEFT, self._on_open_left_drawer)
        self.gesture_recognizer.on(TouchGestureType.EDGE_SWIPE_RIGHT, self._on_open_right_drawer)

    def _on_zoom_in(self, gesture: TouchGesture):
        """Handle zoom in gesture."""
        self._emit_ui_event("zoom", {"scale": gesture.scale, "direction": "in"})

    def _on_zoom_out(self, gesture: TouchGesture):
        """Handle zoom out gesture."""
        self._emit_ui_event("zoom", {"scale": gesture.scale, "direction": "out"})

    def _on_swipe_left(self, gesture: TouchGesture):
        """Handle swipe left."""
        self._emit_ui_event("navigate", {"direction": "left"})

    def _on_swipe_right(self, gesture: TouchGesture):
        """Handle swipe right."""
        self._emit_ui_event("navigate", {"direction": "right"})

    def _on_long_press(self, gesture: TouchGesture):
        """Handle long press."""
        # Find component under touch
        component = self._hit_test(gesture.position)
        if component:
            self._emit_ui_event("context_menu", {
                "component_id": component.component_id,
                "position": gesture.position
            })

    def _on_open_left_drawer(self, gesture: TouchGesture):
        """Handle left edge swipe."""
        self._emit_ui_event("drawer", {"side": "left", "action": "open"})

    def _on_open_right_drawer(self, gesture: TouchGesture):
        """Handle right edge swipe."""
        self._emit_ui_event("drawer", {"side": "right", "action": "open"})

    def _hit_test(self, position: tuple[float, float]) -> UIComponent | None:
        """Find component at screen position."""
        screen_x = position[0] * self.capabilities.screen_width
        screen_y = position[1] * self.capabilities.screen_height

        # Sort by z-index (highest first)
        sorted_components = sorted(
            self.components.values(),
            key=lambda c: c.z_index,
            reverse=True
        )

        for component in sorted_components:
            if not component.visible or not component.interactive:
                continue

            comp_pos = component.get_computed_position(
                self.capabilities.screen_width,
                self.capabilities.screen_height,
                (0, 0, 0, 0)  # TODO: Real safe area
            )

            # Simple rectangular hit test
            width = component.width.get_for_width(self.capabilities.screen_width)
            height = component.height.get_for_width(self.capabilities.screen_width)

            if isinstance(width, float):
                width *= self.capabilities.screen_width
            if isinstance(height, float):
                height *= self.capabilities.screen_height

            # Include touch padding
            padding = component.touch_target_padding

            if (comp_pos[0] - padding <= screen_x <= comp_pos[0] + width + padding and
                comp_pos[1] - padding <= screen_y <= comp_pos[1] + height + padding):
                return component

        return None

    def _adapt_components_to_screen(self):
        """Adapt all components to current screen."""
        for component in self.components.values():
            # Adjust touch targets for mobile
            if self.capabilities.device_type == DeviceType.MOBILE_PHONE:
                component.touch_target_padding = max(8.0, component.touch_target_padding)

            # Increase sizes for accessible mode
            if self.ui_scale == UIScale.ACCESSIBLE:
                if isinstance(component, FloatingActionButton):
                    component.size *= 1.25

    def on_component_action(self, component_id: str, handler: Callable):
        """Register component action handler."""
        key = f"action:{component_id}"
        if key not in self._ui_event_handlers:
            self._ui_event_handlers[key] = []
        self._ui_event_handlers[key].append(handler)

    def on(self, event_type: str, handler: Callable):
        """Register UI event handler."""
        if event_type not in self._ui_event_handlers:
            self._ui_event_handlers[event_type] = []
        self._ui_event_handlers[event_type].append(handler)

    def _emit_ui_event(self, event_type: str, data: dict[str, Any]):
        """Emit UI event."""
        handlers = self._ui_event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"UI event handler error: {e}")

    def get_performance_hints(self) -> dict[str, Any]:
        """Get performance recommendations for current device."""
        return {
            "render_quality": self.render_quality,
            "reduce_motion": self.reduce_motion or self.capabilities.is_low_power_mode,
            "max_visible_overlays": 50 if self.capabilities.gpu_tier >= 2 else 20,
            "physics_enabled": self.capabilities.gpu_tier >= 2,
            "shadow_quality": "high" if self.capabilities.gpu_tier >= 3 else "low",
            "particle_count": 100 if self.capabilities.gpu_tier >= 2 else 20,
            "frame_rate_target": self.capabilities.preferred_frame_rate
        }

    def to_state(self) -> dict[str, Any]:
        """Export UI state."""
        return {
            "device": self.capabilities.to_dict(),
            "ui_scale": self.ui_scale.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "hud": self.hud.to_dict() if self.hud else None,
            "active_modal": self.active_modal,
            "performance": self.get_performance_hints()
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_mobile_ui_manager(
    screen_width: int = 375,
    screen_height: int = 812,
    device_type: DeviceType = DeviceType.MOBILE_PHONE,
    has_ar: bool = False
) -> MobileSpatialUIManager:
    """Create mobile UI manager with detected capabilities."""
    capabilities = DeviceCapabilities(
        device_type=device_type,
        screen_width=screen_width,
        screen_height=screen_height,
        has_touch=device_type in [DeviceType.MOBILE_PHONE, DeviceType.TABLET],
        has_multi_touch=device_type in [DeviceType.MOBILE_PHONE, DeviceType.TABLET],
        has_ar_support=has_ar,
        orientation=ScreenOrientation.PORTRAIT if screen_height > screen_width else ScreenOrientation.LANDSCAPE
    )

    manager = MobileSpatialUIManager(capabilities)
    manager.setup_gesture_handlers()

    return manager


def create_default_spatial_ui(manager: MobileSpatialUIManager) -> dict[str, UIComponent]:
    """Create default spatial UI components."""
    components = {}

    # Main action button (bottom right)
    components["main_fab"] = manager.create_floating_button(
        "main_fab",
        icon="add",
        anchor=UIAnchor.BOTTOM_RIGHT
    )

    # Knowledge panel (bottom sheet)
    components["knowledge_sheet"] = manager.create_bottom_sheet(
        "knowledge_sheet",
        header_label="Knowledge"
    )

    # Quick actions radial menu
    components["quick_menu"] = manager.create_radial_menu(
        "quick_menu",
        items=[
            {"icon": "search", "label": "Search", "action": "search"},
            {"icon": "add", "label": "Add", "action": "add"},
            {"icon": "share", "label": "Share", "action": "share"},
            {"icon": "settings", "label": "Settings", "action": "settings"}
        ]
    )

    # Create HUD
    hud = manager.create_hud()

    # Add HUD elements
    hud.add_element(UIComponent(
        component_id="compass",
        component_type=UIComponentType.COMPASS,
        anchor=UIAnchor.TOP_RIGHT
    ))

    hud.add_element(UIComponent(
        component_id="minimap",
        component_type=UIComponentType.MINIMAP,
        anchor=UIAnchor.TOP_LEFT,
        width=ResponsiveValue(xs=0.25, md=0.15),
        height=ResponsiveValue(xs=0.15, md=0.1)
    ))

    return components

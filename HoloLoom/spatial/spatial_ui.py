"""
Spatial UI System for HoloLoom Spatial Computing.

3D UI elements that exist in XR space: panels, menus, buttons,
sliders, and other interactive widgets.

Features:
- 3D panels with billboarding options
- Floating menus attached to nodes or world space
- Touch/gaze-interactive buttons
- Radial menus for quick actions
- Tag clouds around nodes
- Info cards with rich content

Created: November 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Union
from enum import Enum
from datetime import datetime
import math


class UIAnchorMode(Enum):
    """How UI element is anchored in space."""
    WORLD = "world"              # Fixed world position
    NODE = "node"                # Attached to a node
    HEAD_LOCKED = "head_locked"  # Follows user head
    HAND_LOCKED = "hand_locked"  # Attached to hand
    BILLBOARD = "billboard"      # Always faces user


class UIInteractionMode(Enum):
    """How UI element is interacted with."""
    GAZE_DWELL = "gaze_dwell"    # Look to select
    TOUCH = "touch"              # Direct touch
    POINTER = "pointer"          # Controller pointer
    PINCH = "pinch"              # Hand pinch gesture
    VOICE = "voice"              # Voice command


class UIElementState(Enum):
    """State of a UI element."""
    NORMAL = "normal"
    HOVERED = "hovered"
    PRESSED = "pressed"
    DISABLED = "disabled"
    SELECTED = "selected"


@dataclass
class UIPosition:
    """3D position for UI element."""
    x: float
    y: float
    z: float
    rotation_x: float = 0.0  # Euler angles in radians
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    scale: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "rotation_x": self.rotation_x,
            "rotation_y": self.rotation_y,
            "rotation_z": self.rotation_z,
            "scale": self.scale
        }


@dataclass
class UISize:
    """Size of a UI element in 3D space (meters)."""
    width: float
    height: float
    depth: float = 0.01  # Thin by default

    def to_dict(self) -> Dict[str, float]:
        return {"width": self.width, "height": self.height, "depth": self.depth}


@dataclass
class UIStyle:
    """Visual style for UI elements."""
    background_color: str = "#1a1a2e"
    foreground_color: str = "#ffffff"
    accent_color: str = "#0ea5e9"
    border_color: str = "#3b82f6"
    border_width: float = 0.002
    border_radius: float = 0.02
    opacity: float = 0.9
    glow: bool = False
    glow_color: str = "#3b82f6"
    glow_intensity: float = 0.5
    font_size: float = 0.02  # Meters
    font_family: str = "sans-serif"
    padding: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backgroundColor": self.background_color,
            "foregroundColor": self.foreground_color,
            "accentColor": self.accent_color,
            "borderColor": self.border_color,
            "borderWidth": self.border_width,
            "borderRadius": self.border_radius,
            "opacity": self.opacity,
            "glow": self.glow,
            "glowColor": self.glow_color,
            "glowIntensity": self.glow_intensity,
            "fontSize": self.font_size,
            "fontFamily": self.font_family,
            "padding": self.padding
        }


@dataclass
class UIElement:
    """Base UI element in 3D space."""
    element_id: str
    element_type: str
    position: UIPosition
    size: UISize
    style: UIStyle = field(default_factory=UIStyle)
    anchor_mode: UIAnchorMode = UIAnchorMode.WORLD
    anchor_node_id: Optional[str] = None
    interaction_mode: UIInteractionMode = UIInteractionMode.GAZE_DWELL
    state: UIElementState = UIElementState.NORMAL
    visible: bool = True
    enabled: bool = True
    tooltip: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.element_id,
            "type": self.element_type,
            "position": self.position.to_dict(),
            "size": self.size.to_dict(),
            "style": self.style.to_dict(),
            "anchorMode": self.anchor_mode.value,
            "anchorNodeId": self.anchor_node_id,
            "interactionMode": self.interaction_mode.value,
            "state": self.state.value,
            "visible": self.visible,
            "enabled": self.enabled,
            "tooltip": self.tooltip,
            "metadata": self.metadata
        }


@dataclass
class Panel(UIElement):
    """3D panel that can contain other elements."""
    title: Optional[str] = None
    content: str = ""
    children: List[str] = field(default_factory=list)  # Child element IDs

    def __post_init__(self):
        self.element_type = "panel"

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["title"] = self.title
        data["content"] = self.content
        data["children"] = self.children
        return data


@dataclass
class Button(UIElement):
    """Interactive 3D button."""
    label: str = "Button"
    icon: Optional[str] = None  # Icon name/URL
    callback_action: Optional[str] = None  # Action to trigger

    def __post_init__(self):
        self.element_type = "button"

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["label"] = self.label
        data["icon"] = self.icon
        data["callbackAction"] = self.callback_action
        return data


@dataclass
class Slider(UIElement):
    """3D slider for value selection."""
    value: float = 0.5
    min_value: float = 0.0
    max_value: float = 1.0
    step: float = 0.01
    label: Optional[str] = None
    show_value: bool = True
    callback_action: Optional[str] = None

    def __post_init__(self):
        self.element_type = "slider"

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "value": self.value,
            "minValue": self.min_value,
            "maxValue": self.max_value,
            "step": self.step,
            "label": self.label,
            "showValue": self.show_value,
            "callbackAction": self.callback_action
        })
        return data


@dataclass
class Toggle(UIElement):
    """3D toggle switch."""
    checked: bool = False
    label: Optional[str] = None
    callback_action: Optional[str] = None

    def __post_init__(self):
        self.element_type = "toggle"

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "checked": self.checked,
            "label": self.label,
            "callbackAction": self.callback_action
        })
        return data


@dataclass
class MenuItem:
    """Item in a radial or linear menu."""
    item_id: str
    label: str
    icon: Optional[str] = None
    action: Optional[str] = None
    enabled: bool = True
    submenu: Optional[List["MenuItem"]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.item_id,
            "label": self.label,
            "icon": self.icon,
            "action": self.action,
            "enabled": self.enabled
        }
        if self.submenu:
            data["submenu"] = [item.to_dict() for item in self.submenu]
        return data


@dataclass
class RadialMenu(UIElement):
    """Radial menu for quick actions."""
    items: List[MenuItem] = field(default_factory=list)
    inner_radius: float = 0.05  # Meters
    outer_radius: float = 0.15
    open: bool = False
    selected_item_id: Optional[str] = None

    def __post_init__(self):
        self.element_type = "radial_menu"

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "items": [item.to_dict() for item in self.items],
            "innerRadius": self.inner_radius,
            "outerRadius": self.outer_radius,
            "open": self.open,
            "selectedItemId": self.selected_item_id
        })
        return data


@dataclass
class InfoCard(UIElement):
    """Information card displaying node details."""
    title: str = ""
    subtitle: Optional[str] = None
    body: str = ""
    image_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    actions: List[MenuItem] = field(default_factory=list)
    linked_node_id: Optional[str] = None

    def __post_init__(self):
        self.element_type = "info_card"

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "title": self.title,
            "subtitle": self.subtitle,
            "body": self.body,
            "imageUrl": self.image_url,
            "tags": self.tags,
            "actions": [a.to_dict() for a in self.actions],
            "linkedNodeId": self.linked_node_id
        })
        return data


@dataclass
class TagCloud(UIElement):
    """3D tag cloud around a node."""
    tags: List[Dict[str, Any]] = field(default_factory=list)  # {label, weight, color, action}
    radius: float = 0.3  # Distribution radius
    linked_node_id: Optional[str] = None

    def __post_init__(self):
        self.element_type = "tag_cloud"

    def add_tag(self, label: str, weight: float = 1.0, color: str = "#3b82f6", action: Optional[str] = None):
        """Add a tag to the cloud."""
        self.tags.append({
            "label": label,
            "weight": weight,
            "color": color,
            "action": action
        })

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "tags": self.tags,
            "radius": self.radius,
            "linkedNodeId": self.linked_node_id
        })
        return data


@dataclass
class ProgressRing(UIElement):
    """3D progress ring indicator."""
    progress: float = 0.0  # 0.0 to 1.0
    label: Optional[str] = None
    ring_width: float = 0.01
    radius: float = 0.05

    def __post_init__(self):
        self.element_type = "progress_ring"

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "progress": self.progress,
            "label": self.label,
            "ringWidth": self.ring_width,
            "radius": self.radius
        })
        return data


class SpatialUIManager:
    """
    Manages all spatial UI elements.

    Handles creation, updates, layout, and interaction
    for 3D UI elements in XR space.
    """

    def __init__(
        self,
        default_style: Optional[UIStyle] = None,
        billboard_all: bool = False
    ):
        self.elements: Dict[str, UIElement] = {}
        self.default_style = default_style or UIStyle()
        self.billboard_all = billboard_all

        # Element groups for batch operations
        self.groups: Dict[str, List[str]] = {}

        # Layout helpers
        self.head_position: Optional[tuple] = None
        self.head_direction: Optional[tuple] = None

        # Interaction callbacks
        self.action_handlers: Dict[str, Callable[[str, Dict], None]] = {}

        # Event callbacks
        self.on_element_click: List[Callable[[str], None]] = []
        self.on_element_hover: List[Callable[[str, bool], None]] = []
        self.on_value_change: List[Callable[[str, Any], None]] = []

    def create_panel(
        self,
        panel_id: str,
        position: tuple,
        size: tuple = (0.4, 0.3),
        title: Optional[str] = None,
        content: str = "",
        anchor_mode: UIAnchorMode = UIAnchorMode.WORLD,
        anchor_node_id: Optional[str] = None,
        style: Optional[UIStyle] = None
    ) -> Panel:
        """Create a 3D panel."""
        panel = Panel(
            element_id=panel_id,
            element_type="panel",
            position=UIPosition(position[0], position[1], position[2]),
            size=UISize(size[0], size[1]),
            style=style or self.default_style,
            anchor_mode=anchor_mode,
            anchor_node_id=anchor_node_id,
            title=title,
            content=content
        )
        self.elements[panel_id] = panel
        return panel

    def create_button(
        self,
        button_id: str,
        position: tuple,
        label: str,
        size: tuple = (0.1, 0.05),
        icon: Optional[str] = None,
        action: Optional[str] = None,
        style: Optional[UIStyle] = None
    ) -> Button:
        """Create a 3D button."""
        button = Button(
            element_id=button_id,
            element_type="button",
            position=UIPosition(position[0], position[1], position[2]),
            size=UISize(size[0], size[1]),
            style=style or self.default_style,
            label=label,
            icon=icon,
            callback_action=action
        )
        self.elements[button_id] = button
        return button

    def create_radial_menu(
        self,
        menu_id: str,
        position: tuple,
        items: List[Dict[str, str]],
        inner_radius: float = 0.05,
        outer_radius: float = 0.15
    ) -> RadialMenu:
        """Create a radial menu."""
        menu_items = [
            MenuItem(
                item_id=f"{menu_id}_{i}",
                label=item.get("label", ""),
                icon=item.get("icon"),
                action=item.get("action")
            )
            for i, item in enumerate(items)
        ]

        menu = RadialMenu(
            element_id=menu_id,
            element_type="radial_menu",
            position=UIPosition(position[0], position[1], position[2]),
            size=UISize(outer_radius * 2, outer_radius * 2),
            style=self.default_style,
            items=menu_items,
            inner_radius=inner_radius,
            outer_radius=outer_radius
        )
        self.elements[menu_id] = menu
        return menu

    def create_info_card(
        self,
        card_id: str,
        position: tuple,
        title: str,
        body: str,
        subtitle: Optional[str] = None,
        tags: Optional[List[str]] = None,
        linked_node_id: Optional[str] = None,
        size: tuple = (0.3, 0.2)
    ) -> InfoCard:
        """Create an info card."""
        card = InfoCard(
            element_id=card_id,
            element_type="info_card",
            position=UIPosition(position[0], position[1], position[2]),
            size=UISize(size[0], size[1]),
            style=self.default_style,
            title=title,
            subtitle=subtitle,
            body=body,
            tags=tags or [],
            linked_node_id=linked_node_id
        )

        if linked_node_id:
            card.anchor_mode = UIAnchorMode.NODE
            card.anchor_node_id = linked_node_id

        self.elements[card_id] = card
        return card

    def create_tag_cloud(
        self,
        cloud_id: str,
        position: tuple,
        tags: List[Dict[str, Any]],
        radius: float = 0.3,
        linked_node_id: Optional[str] = None
    ) -> TagCloud:
        """Create a 3D tag cloud."""
        cloud = TagCloud(
            element_id=cloud_id,
            element_type="tag_cloud",
            position=UIPosition(position[0], position[1], position[2]),
            size=UISize(radius * 2, radius * 2),
            style=self.default_style,
            tags=tags,
            radius=radius,
            linked_node_id=linked_node_id
        )

        if linked_node_id:
            cloud.anchor_mode = UIAnchorMode.NODE
            cloud.anchor_node_id = linked_node_id

        self.elements[cloud_id] = cloud
        return cloud

    def create_slider(
        self,
        slider_id: str,
        position: tuple,
        value: float = 0.5,
        min_value: float = 0.0,
        max_value: float = 1.0,
        label: Optional[str] = None,
        action: Optional[str] = None,
        size: tuple = (0.15, 0.02)
    ) -> Slider:
        """Create a 3D slider."""
        slider = Slider(
            element_id=slider_id,
            element_type="slider",
            position=UIPosition(position[0], position[1], position[2]),
            size=UISize(size[0], size[1]),
            style=self.default_style,
            value=value,
            min_value=min_value,
            max_value=max_value,
            label=label,
            callback_action=action
        )
        self.elements[slider_id] = slider
        return slider

    def create_progress_ring(
        self,
        ring_id: str,
        position: tuple,
        progress: float = 0.0,
        label: Optional[str] = None,
        radius: float = 0.05
    ) -> ProgressRing:
        """Create a progress ring indicator."""
        ring = ProgressRing(
            element_id=ring_id,
            element_type="progress_ring",
            position=UIPosition(position[0], position[1], position[2]),
            size=UISize(radius * 2, radius * 2),
            style=self.default_style,
            progress=progress,
            label=label,
            radius=radius
        )
        self.elements[ring_id] = ring
        return ring

    def get_element(self, element_id: str) -> Optional[UIElement]:
        """Get an element by ID."""
        return self.elements.get(element_id)

    def remove_element(self, element_id: str):
        """Remove an element."""
        if element_id in self.elements:
            del self.elements[element_id]

            # Remove from groups
            for group_elements in self.groups.values():
                if element_id in group_elements:
                    group_elements.remove(element_id)

    def set_element_state(self, element_id: str, state: UIElementState):
        """Set element interaction state."""
        if element_id in self.elements:
            self.elements[element_id].state = state

    def set_element_visible(self, element_id: str, visible: bool):
        """Set element visibility."""
        if element_id in self.elements:
            self.elements[element_id].visible = visible

    def update_element_position(self, element_id: str, position: tuple):
        """Update element position."""
        if element_id in self.elements:
            elem = self.elements[element_id]
            elem.position.x = position[0]
            elem.position.y = position[1]
            elem.position.z = position[2]

    def update_slider_value(self, slider_id: str, value: float):
        """Update slider value."""
        elem = self.elements.get(slider_id)
        if elem and isinstance(elem, Slider):
            elem.value = max(elem.min_value, min(elem.max_value, value))

            # Fire value change callback
            for callback in self.on_value_change:
                callback(slider_id, elem.value)

    def update_progress(self, ring_id: str, progress: float):
        """Update progress ring."""
        elem = self.elements.get(ring_id)
        if elem and isinstance(elem, ProgressRing):
            elem.progress = max(0.0, min(1.0, progress))

    def add_to_group(self, element_id: str, group_name: str):
        """Add element to a group."""
        if group_name not in self.groups:
            self.groups[group_name] = []

        if element_id not in self.groups[group_name]:
            self.groups[group_name].append(element_id)

    def set_group_visible(self, group_name: str, visible: bool):
        """Set visibility for all elements in a group."""
        if group_name in self.groups:
            for element_id in self.groups[group_name]:
                self.set_element_visible(element_id, visible)

    def update_head_pose(self, position: tuple, direction: tuple):
        """Update head pose for billboarding."""
        self.head_position = position
        self.head_direction = direction

    def apply_billboarding(self):
        """Apply billboarding to elements that need it."""
        if not self.head_position:
            return

        for element in self.elements.values():
            if element.anchor_mode == UIAnchorMode.BILLBOARD or self.billboard_all:
                self._billboard_element(element)

    def _billboard_element(self, element: UIElement):
        """Make element face the user."""
        if not self.head_position:
            return

        # Calculate direction from element to head
        dx = self.head_position[0] - element.position.x
        dy = self.head_position[1] - element.position.y
        dz = self.head_position[2] - element.position.z

        # Calculate rotation (simplified - Y-axis only for now)
        element.position.rotation_y = math.atan2(dx, dz)

    def handle_interaction(self, element_id: str, interaction_type: str, data: Dict[str, Any] = None):
        """Handle interaction with an element."""
        element = self.elements.get(element_id)
        if not element or not element.enabled:
            return

        if interaction_type == "click":
            self._handle_click(element, data or {})
        elif interaction_type == "hover_enter":
            element.state = UIElementState.HOVERED
            for callback in self.on_element_hover:
                callback(element_id, True)
        elif interaction_type == "hover_exit":
            element.state = UIElementState.NORMAL
            for callback in self.on_element_hover:
                callback(element_id, False)
        elif interaction_type == "press":
            element.state = UIElementState.PRESSED
        elif interaction_type == "release":
            element.state = UIElementState.HOVERED
        elif interaction_type == "value_change":
            self._handle_value_change(element, data or {})

    def _handle_click(self, element: UIElement, data: Dict[str, Any]):
        """Handle click on element."""
        # Fire click callbacks
        for callback in self.on_element_click:
            callback(element.element_id)

        # Handle specific element types
        if isinstance(element, Button) and element.callback_action:
            self._trigger_action(element.callback_action, element.element_id, data)

        elif isinstance(element, Toggle):
            element.checked = not element.checked
            if element.callback_action:
                self._trigger_action(element.callback_action, element.element_id, {"checked": element.checked})

        elif isinstance(element, RadialMenu):
            if data.get("item_id"):
                element.selected_item_id = data["item_id"]
                for item in element.items:
                    if item.item_id == data["item_id"] and item.action:
                        self._trigger_action(item.action, element.element_id, {"item": item.label})

    def _handle_value_change(self, element: UIElement, data: Dict[str, Any]):
        """Handle value change on element."""
        if isinstance(element, Slider) and "value" in data:
            self.update_slider_value(element.element_id, data["value"])
            if element.callback_action:
                self._trigger_action(element.callback_action, element.element_id, {"value": element.value})

    def _trigger_action(self, action: str, element_id: str, data: Dict[str, Any]):
        """Trigger an action handler."""
        if action in self.action_handlers:
            self.action_handlers[action](element_id, data)

    def register_action_handler(self, action: str, handler: Callable[[str, Dict], None]):
        """Register a handler for an action."""
        self.action_handlers[action] = handler

    def create_node_context_menu(
        self,
        node_id: str,
        position: tuple,
        node_label: str,
        actions: List[Dict[str, str]] = None
    ) -> RadialMenu:
        """Create a context menu for a node."""
        default_actions = [
            {"label": "Expand", "icon": "expand", "action": f"expand_{node_id}"},
            {"label": "Focus", "icon": "target", "action": f"focus_{node_id}"},
            {"label": "Details", "icon": "info", "action": f"details_{node_id}"},
            {"label": "Hide", "icon": "eye-off", "action": f"hide_{node_id}"},
            {"label": "Connect", "icon": "link", "action": f"connect_{node_id}"},
        ]

        menu = self.create_radial_menu(
            menu_id=f"context_menu_{node_id}",
            position=position,
            items=actions or default_actions
        )
        menu.anchor_mode = UIAnchorMode.NODE
        menu.anchor_node_id = node_id

        return menu

    def create_query_input_panel(
        self,
        position: tuple,
        on_submit_action: str = "submit_query"
    ) -> Panel:
        """Create a floating query input panel."""
        panel = self.create_panel(
            panel_id="query_input_panel",
            position=position,
            size=(0.5, 0.15),
            title="Ask HoloLoom"
        )
        panel.anchor_mode = UIAnchorMode.HEAD_LOCKED

        # Add submit button
        self.create_button(
            button_id="query_submit",
            position=(position[0] + 0.15, position[1] - 0.03, position[2]),
            label="Ask",
            action=on_submit_action
        )
        panel.children.append("query_submit")

        return panel

    def to_state(self) -> Dict[str, Any]:
        """Export complete UI state for WebXR client."""
        return {
            "type": "spatial_ui_state",
            "elements": [elem.to_dict() for elem in self.elements.values()],
            "groups": dict(self.groups),
            "timestamp": datetime.now().isoformat()
        }

    def get_elements_at_position(
        self,
        position: tuple,
        radius: float = 0.1
    ) -> List[UIElement]:
        """Get elements near a position."""
        nearby = []

        for element in self.elements.values():
            dx = element.position.x - position[0]
            dy = element.position.y - position[1]
            dz = element.position.z - position[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist <= radius:
                nearby.append(element)

        return nearby

    def layout_around_node(
        self,
        node_position: tuple,
        elements: List[str],
        radius: float = 0.3,
        start_angle: float = 0.0
    ):
        """Layout elements in a circle around a node."""
        angle_step = (2 * math.pi) / len(elements) if elements else 0

        for i, element_id in enumerate(elements):
            element = self.elements.get(element_id)
            if element:
                angle = start_angle + i * angle_step
                x = node_position[0] + radius * math.cos(angle)
                y = node_position[1]
                z = node_position[2] + radius * math.sin(angle)

                element.position.x = x
                element.position.y = y
                element.position.z = z
                element.position.rotation_y = angle + math.pi  # Face center

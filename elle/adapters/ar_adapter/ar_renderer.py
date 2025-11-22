"""
AR Renderer - Converts Elle actions to AR visualizations

Defines visualization types (overlays, highlights, paths, panels) that can
be rendered by the AR client (Three.js, ARCore, ARKit).

Created: 2025-11-22
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from elle.adapters.ar_adapter.ar_events import Vector3


# ============================================================================
# Visualization Types
# ============================================================================

class ARVisualizationType(str, Enum):
    """Types of AR visualizations"""
    OVERLAY = "overlay"        # Text/icon overlay on object
    HIGHLIGHT = "highlight"    # Bounding box/glow around object
    PATH = "path"              # 3D path/arrow for navigation
    PANEL = "panel"            # Info panel (world-locked or head-locked)
    ANNOTATION = "annotation"  # Persistent label
    ANIMATION = "animation"    # Animated instruction
    SPATIAL_AUDIO = "spatial_audio"  # 3D positioned audio


# ============================================================================
# Style & Appearance
# ============================================================================

@dataclass
class VisualStyle:
    """Visual appearance parameters"""
    color: str = "#ffffff"           # Hex color
    opacity: float = 0.9             # 0.0 - 1.0
    size: float = 1.0                # Scale multiplier
    font_size: Optional[int] = None  # For text (px)
    animation: Optional[str] = None  # "pulse", "fade", "grow", etc.
    duration_sec: Optional[float] = None  # Auto-fade duration


# ============================================================================
# Base Visualization
# ============================================================================

@dataclass
class ARVisualization:
    """
    Base AR visualization specification.

    Converted to platform-specific rendering calls by AR client.
    """
    id: str  # Unique visualization ID
    viz_type: ARVisualizationType
    style: VisualStyle = field(default_factory=VisualStyle)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Lifecycle
    auto_dismiss_sec: Optional[float] = None  # Auto-hide after duration
    persist: bool = False  # Keep across sessions


# ============================================================================
# Specific Visualization Types
# ============================================================================

@dataclass
class AROverlay(ARVisualization):
    """
    Text/icon overlay attached to an object.

    Example: Label showing "Stanley Toolbox | Last used: Nov 21"
    """
    target_object_id: str
    content: str  # Text or icon name
    position_offset: Vector3 = field(default_factory=lambda: Vector3(0, 0.2, 0))  # Offset from object
    face_user: bool = True  # Billboard effect (always face camera)

    def __post_init__(self):
        self.viz_type = ARVisualizationType.OVERLAY


@dataclass
class ARHighlight(ARVisualization):
    """
    Visual highlight around an object (bounding box, glow, outline).

    Example: Pulsing box around drill when user asks "where's my drill?"
    """
    target_object_id: str
    highlight_type: str = "bounding_box"  # bounding_box, glow, outline, pulse
    intensity: float = 1.0  # 0.0 - 1.0

    def __post_init__(self):
        self.viz_type = ARVisualizationType.HIGHLIGHT


@dataclass
class ARPath(ARVisualization):
    """
    3D path visualization (arrows, line, particles).

    Example: Arrow pointing to shelf where drill is located
    """
    waypoints: List[Vector3]  # Path vertices
    path_type: str = "arrow"  # arrow, line, particles, dotted_line
    width: float = 0.05  # meters
    end_marker: bool = True  # Show marker at destination

    def __post_init__(self):
        self.viz_type = ARVisualizationType.PATH


@dataclass
class ARPanel(ARVisualization):
    """
    Info panel (floating UI element).

    Can be world-locked (fixed in space) or head-locked (HUD style).
    """
    content: Dict[str, Any]  # Structured content (title, body, buttons, etc.)
    position: Vector3 = field(default_factory=lambda: Vector3(0, 1.5, -1.0))
    lock_mode: str = "world"  # world, head, object
    target_object_id: Optional[str] = None  # For object-locked panels
    size: tuple[float, float] = (0.4, 0.3)  # width, height in meters

    def __post_init__(self):
        self.viz_type = ARVisualizationType.PANEL


@dataclass
class ARAnnotation(ARVisualization):
    """
    Persistent annotation (label that stays across sessions).

    Example: User-created labels like "Tool storage" on a cabinet
    """
    text: str
    position: Vector3
    anchor_id: Optional[str] = None  # World anchor for persistence
    target_object_id: Optional[str] = None

    def __post_init__(self):
        self.viz_type = ARVisualizationType.ANNOTATION
        self.persist = True  # Annotations persist by default


@dataclass
class ARAnimation(ARVisualization):
    """
    Animated instruction overlay.

    Example: Animated arrows showing "pull back chuck sleeve, insert bit"
    """
    target_object_id: str
    animation_type: str  # "arrows", "highlight_sequence", "ghost_hands"
    steps: List[Dict[str, Any]]  # Animation steps with timing
    loop: bool = False

    def __post_init__(self):
        self.viz_type = ARVisualizationType.ANIMATION


# ============================================================================
# Visualization Collection
# ============================================================================

@dataclass
class ARVisualizationSet:
    """
    Collection of visualizations to render together.

    Elle actions can produce multiple visualizations (e.g., highlight + overlay + path).
    """
    visualizations: List[ARVisualization] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, viz: ARVisualization) -> "ARVisualizationSet":
        """Add visualization (builder pattern)"""
        self.visualizations.append(viz)
        return self

    def get_by_type(self, viz_type: ARVisualizationType) -> List[ARVisualization]:
        """Filter visualizations by type"""
        return [v for v in self.visualizations if v.viz_type == viz_type]

    def clear_non_persistent(self) -> None:
        """Remove all non-persistent visualizations"""
        self.visualizations = [v for v in self.visualizations if v.persist]

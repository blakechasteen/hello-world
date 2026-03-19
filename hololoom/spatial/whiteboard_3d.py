"""
Collaborative 3D Whiteboard System for HoloLoom Spatial Computing.

Enables shared 3D drawing and annotation in XR space:
- 3D freehand drawing
- Multiple brush types and colors
- Shape tools (lines, rectangles, circles, arrows)
- Text annotations
- Sticky notes
- Multi-user collaboration
- Undo/redo history
- Save/load boards

Created: November 2025
"""

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DrawingTool(Enum):
    """Available drawing tools."""
    FREEHAND = "freehand"
    LINE = "line"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ARROW = "arrow"
    TEXT = "text"
    STICKY_NOTE = "sticky_note"
    HIGHLIGHTER = "highlighter"
    ERASER = "eraser"
    LASER_POINTER = "laser_pointer"
    STAMP = "stamp"


class BrushStyle(Enum):
    """Brush rendering styles."""
    SOLID = "solid"
    DOTTED = "dotted"
    DASHED = "dashed"
    GLOW = "glow"
    NEON = "neon"
    SKETCH = "sketch"
    RIBBON = "ribbon"
    TUBE = "tube"


@dataclass
class Color:
    """RGBA color."""
    r: float = 1.0  # 0-1
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

    def to_hex(self) -> str:
        return f"#{int(self.r * 255):02x}{int(self.g * 255):02x}{int(self.b * 255):02x}"

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    @classmethod
    def from_hex(cls, hex_color: str) -> "Color":
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return cls(r, g, b)


# Preset colors
class Colors:
    RED = Color(1.0, 0.2, 0.2)
    GREEN = Color(0.2, 0.8, 0.2)
    BLUE = Color(0.2, 0.4, 1.0)
    YELLOW = Color(1.0, 0.9, 0.2)
    ORANGE = Color(1.0, 0.5, 0.0)
    PURPLE = Color(0.7, 0.3, 0.9)
    CYAN = Color(0.0, 0.8, 0.9)
    PINK = Color(1.0, 0.4, 0.7)
    WHITE = Color(1.0, 1.0, 1.0)
    BLACK = Color(0.1, 0.1, 0.1)
    GRAY = Color(0.5, 0.5, 0.5)


@dataclass
class Point3D:
    """3D point with optional pressure/tilt."""
    x: float
    y: float
    z: float
    pressure: float = 1.0  # 0-1, for pressure-sensitive input
    tilt_x: float = 0.0    # Controller/pen tilt
    tilt_y: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_to(self, other: "Point3D") -> float:
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


@dataclass
class BrushSettings:
    """Brush configuration."""
    tool: DrawingTool = DrawingTool.FREEHAND
    color: Color = field(default_factory=lambda: Colors.WHITE)
    size: float = 0.01  # Meters
    style: BrushStyle = BrushStyle.SOLID
    opacity: float = 1.0
    smoothing: float = 0.5  # 0-1, stroke smoothing
    pressure_sensitivity: bool = True
    taper_start: float = 0.0  # Taper at stroke start
    taper_end: float = 0.0    # Taper at stroke end


@dataclass
class Stroke:
    """A single drawing stroke."""
    stroke_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    points: list[Point3D] = field(default_factory=list)
    brush: BrushSettings = field(default_factory=BrushSettings)
    user_id: str = ""
    board_id: str = ""
    layer: int = 0
    visible: bool = True
    locked: bool = False
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_point(self, x: float, y: float, z: float, pressure: float = 1.0):
        """Add point to stroke."""
        self.points.append(Point3D(x, y, z, pressure))

    def get_bounds(self) -> tuple[Point3D, Point3D]:
        """Get bounding box."""
        if not self.points:
            return Point3D(), Point3D()

        min_p = Point3D(
            min(p.x for p in self.points),
            min(p.y for p in self.points),
            min(p.z for p in self.points)
        )
        max_p = Point3D(
            max(p.x for p in self.points),
            max(p.y for p in self.points),
            max(p.z for p in self.points)
        )
        return min_p, max_p

    def get_length(self) -> float:
        """Get total stroke length."""
        if len(self.points) < 2:
            return 0.0
        return sum(
            self.points[i].distance_to(self.points[i+1])
            for i in range(len(self.points) - 1)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.stroke_id,
            "points": [
                {"x": p.x, "y": p.y, "z": p.z, "pressure": p.pressure}
                for p in self.points
            ],
            "brush": {
                "tool": self.brush.tool.value,
                "color": self.brush.color.to_tuple(),
                "size": self.brush.size,
                "style": self.brush.style.value
            },
            "userId": self.user_id,
            "layer": self.layer,
            "visible": self.visible,
            "locked": self.locked,
            "createdAt": self.created_at
        }


@dataclass
class Shape3D:
    """3D shape annotation."""
    shape_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    shape_type: str = "line"  # line, rectangle, circle, arrow
    start_point: Point3D = field(default_factory=Point3D)
    end_point: Point3D = field(default_factory=Point3D)
    color: Color = field(default_factory=lambda: Colors.WHITE)
    line_width: float = 0.005
    filled: bool = False
    fill_color: Color | None = None
    user_id: str = ""
    board_id: str = ""
    layer: int = 0
    visible: bool = True
    locked: bool = False
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.shape_id,
            "type": self.shape_type,
            "start": self.start_point.to_tuple(),
            "end": self.end_point.to_tuple(),
            "color": self.color.to_tuple(),
            "lineWidth": self.line_width,
            "filled": self.filled,
            "userId": self.user_id,
            "layer": self.layer
        }


@dataclass
class TextAnnotation:
    """3D text annotation."""
    text_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    position: Point3D = field(default_factory=Point3D)
    rotation: tuple[float, float, float] = (0, 0, 0)
    font_size: float = 0.05  # Meters
    color: Color = field(default_factory=lambda: Colors.WHITE)
    background_color: Color | None = None
    font_family: str = "sans-serif"
    bold: bool = False
    italic: bool = False
    user_id: str = ""
    board_id: str = ""
    layer: int = 0
    visible: bool = True
    locked: bool = False
    billboard: bool = True  # Face viewer
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.text_id,
            "text": self.text,
            "position": self.position.to_tuple(),
            "rotation": self.rotation,
            "fontSize": self.font_size,
            "color": self.color.to_tuple(),
            "backgroundColor": self.background_color.to_tuple() if self.background_color else None,
            "billboard": self.billboard,
            "userId": self.user_id,
            "layer": self.layer
        }


@dataclass
class StickyNote:
    """3D sticky note."""
    note_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    position: Point3D = field(default_factory=Point3D)
    rotation: tuple[float, float, float] = (0, 0, 0)
    size: tuple[float, float] = (0.15, 0.15)  # Width, height in meters
    color: Color = field(default_factory=lambda: Colors.YELLOW)
    text_color: Color = field(default_factory=lambda: Colors.BLACK)
    pinned: bool = False
    linked_node_id: str | None = None  # Link to knowledge node
    user_id: str = ""
    board_id: str = ""
    layer: int = 1
    visible: bool = True
    locked: bool = False
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.note_id,
            "text": self.text,
            "position": self.position.to_tuple(),
            "rotation": self.rotation,
            "size": self.size,
            "color": self.color.to_tuple(),
            "textColor": self.text_color.to_tuple(),
            "pinned": self.pinned,
            "linkedNodeId": self.linked_node_id,
            "userId": self.user_id,
            "layer": self.layer
        }


@dataclass
class WhiteboardLayer:
    """Layer within a whiteboard."""
    layer_id: int
    name: str = "Layer"
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0


@dataclass
class UndoAction:
    """Undo/redo action."""
    action_type: str  # add, remove, modify
    element_type: str  # stroke, shape, text, note
    element_id: str
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class Whiteboard3D:
    """
    A 3D whiteboard/canvas for collaborative drawing.
    """

    def __init__(
        self,
        board_id: str,
        name: str = "Whiteboard",
        position: tuple[float, float, float] = (0, 1.5, -1),
        size: tuple[float, float] = (2.0, 1.5),  # Width, height
        owner_id: str = ""
    ):
        self.board_id = board_id
        self.name = name
        self.position = Point3D(*position)
        self.size = size
        self.owner_id = owner_id

        # Content
        self.strokes: dict[str, Stroke] = {}
        self.shapes: dict[str, Shape3D] = {}
        self.texts: dict[str, TextAnnotation] = {}
        self.sticky_notes: dict[str, StickyNote] = {}

        # Layers
        self.layers: list[WhiteboardLayer] = [
            WhiteboardLayer(0, "Background"),
            WhiteboardLayer(1, "Main"),
            WhiteboardLayer(2, "Annotations")
        ]
        self.active_layer: int = 1

        # Undo/redo
        self.undo_stack: list[UndoAction] = []
        self.redo_stack: list[UndoAction] = []
        self.max_undo: int = 50

        # Collaboration
        self.collaborators: dict[str, dict[str, Any]] = {}
        self.is_public: bool = False

        # Active drawing (per user)
        self.active_strokes: dict[str, Stroke] = {}  # user_id -> current stroke

        # Metadata
        self.created_at: float = datetime.now().timestamp()
        self.modified_at: float = self.created_at
        self.metadata: dict[str, Any] = {}

        # Callbacks
        self.on_stroke_complete: list[Callable[[Stroke], None]] = []
        self.on_element_added: list[Callable[[str, str], None]] = []
        self.on_element_removed: list[Callable[[str, str], None]] = []

    def start_stroke(
        self,
        user_id: str,
        x: float, y: float, z: float,
        brush: BrushSettings | None = None,
        pressure: float = 1.0
    ) -> str:
        """Start a new stroke."""
        stroke = Stroke(
            user_id=user_id,
            board_id=self.board_id,
            brush=brush or BrushSettings(),
            layer=self.active_layer
        )
        stroke.add_point(x, y, z, pressure)

        self.active_strokes[user_id] = stroke
        return stroke.stroke_id

    def continue_stroke(
        self,
        user_id: str,
        x: float, y: float, z: float,
        pressure: float = 1.0
    ) -> bool:
        """Add point to active stroke."""
        if user_id not in self.active_strokes:
            return False

        stroke = self.active_strokes[user_id]
        stroke.add_point(x, y, z, pressure)
        return True

    def end_stroke(self, user_id: str) -> Stroke | None:
        """Complete and save stroke."""
        if user_id not in self.active_strokes:
            return None

        stroke = self.active_strokes.pop(user_id)

        # Apply smoothing if enabled
        if stroke.brush.smoothing > 0 and len(stroke.points) > 2:
            stroke.points = self._smooth_points(stroke.points, stroke.brush.smoothing)

        # Save stroke
        self.strokes[stroke.stroke_id] = stroke
        self.modified_at = datetime.now().timestamp()

        # Add to undo stack
        self._add_undo_action("add", "stroke", stroke.stroke_id, after_state=stroke.to_dict())

        # Fire callbacks
        for callback in self.on_stroke_complete:
            callback(stroke)
        for callback in self.on_element_added:
            callback("stroke", stroke.stroke_id)

        return stroke

    def cancel_stroke(self, user_id: str):
        """Cancel active stroke."""
        self.active_strokes.pop(user_id, None)

    def _smooth_points(self, points: list[Point3D], factor: float) -> list[Point3D]:
        """Apply smoothing to points."""
        if len(points) < 3:
            return points

        smoothed = [points[0]]  # Keep first point

        for i in range(1, len(points) - 1):
            prev = points[i - 1]
            curr = points[i]
            next_p = points[i + 1]

            # Weighted average
            w = factor * 0.5
            smooth_x = curr.x * (1 - w) + (prev.x + next_p.x) * w * 0.5
            smooth_y = curr.y * (1 - w) + (prev.y + next_p.y) * w * 0.5
            smooth_z = curr.z * (1 - w) + (prev.z + next_p.z) * w * 0.5

            smoothed.append(Point3D(smooth_x, smooth_y, smooth_z, curr.pressure))

        smoothed.append(points[-1])  # Keep last point
        return smoothed

    def add_shape(
        self,
        shape_type: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        user_id: str,
        color: Color | None = None,
        line_width: float = 0.005,
        filled: bool = False
    ) -> Shape3D:
        """Add a shape annotation."""
        shape = Shape3D(
            shape_type=shape_type,
            start_point=Point3D(*start),
            end_point=Point3D(*end),
            color=color or Colors.WHITE,
            line_width=line_width,
            filled=filled,
            user_id=user_id,
            board_id=self.board_id,
            layer=self.active_layer
        )

        self.shapes[shape.shape_id] = shape
        self.modified_at = datetime.now().timestamp()

        self._add_undo_action("add", "shape", shape.shape_id, after_state=shape.to_dict())

        for callback in self.on_element_added:
            callback("shape", shape.shape_id)

        return shape

    def add_text(
        self,
        text: str,
        position: tuple[float, float, float],
        user_id: str,
        font_size: float = 0.05,
        color: Color | None = None,
        billboard: bool = True
    ) -> TextAnnotation:
        """Add text annotation."""
        annotation = TextAnnotation(
            text=text,
            position=Point3D(*position),
            font_size=font_size,
            color=color or Colors.WHITE,
            billboard=billboard,
            user_id=user_id,
            board_id=self.board_id,
            layer=self.active_layer
        )

        self.texts[annotation.text_id] = annotation
        self.modified_at = datetime.now().timestamp()

        self._add_undo_action("add", "text", annotation.text_id, after_state=annotation.to_dict())

        for callback in self.on_element_added:
            callback("text", annotation.text_id)

        return annotation

    def add_sticky_note(
        self,
        text: str,
        position: tuple[float, float, float],
        user_id: str,
        color: Color | None = None,
        linked_node_id: str | None = None
    ) -> StickyNote:
        """Add sticky note."""
        note = StickyNote(
            text=text,
            position=Point3D(*position),
            color=color or Colors.YELLOW,
            linked_node_id=linked_node_id,
            user_id=user_id,
            board_id=self.board_id,
            layer=self.active_layer
        )

        self.sticky_notes[note.note_id] = note
        self.modified_at = datetime.now().timestamp()

        self._add_undo_action("add", "note", note.note_id, after_state=note.to_dict())

        for callback in self.on_element_added:
            callback("note", note.note_id)

        return note

    def remove_element(self, element_type: str, element_id: str) -> bool:
        """Remove an element from the board."""
        collection = self._get_collection(element_type)
        if not collection or element_id not in collection:
            return False

        element = collection[element_id]

        # Check if locked
        if hasattr(element, "locked") and element.locked:
            return False

        # Store for undo
        before_state = element.to_dict() if hasattr(element, "to_dict") else None
        self._add_undo_action("remove", element_type, element_id, before_state=before_state)

        del collection[element_id]
        self.modified_at = datetime.now().timestamp()

        for callback in self.on_element_removed:
            callback(element_type, element_id)

        return True

    def erase_at(
        self,
        x: float, y: float, z: float,
        radius: float = 0.05
    ) -> list[str]:
        """Erase strokes within radius of point."""
        erased = []
        point = Point3D(x, y, z)

        for stroke_id, stroke in list(self.strokes.items()):
            if stroke.locked:
                continue

            # Check if any point is within radius
            for p in stroke.points:
                if p.distance_to(point) < radius:
                    self.remove_element("stroke", stroke_id)
                    erased.append(stroke_id)
                    break

        return erased

    def clear(self, user_id: str):
        """Clear all content (undoable)."""
        # Store current state
        before_state = self.to_dict()

        self.strokes.clear()
        self.shapes.clear()
        self.texts.clear()
        self.sticky_notes.clear()
        self.modified_at = datetime.now().timestamp()

        self._add_undo_action("clear", "board", self.board_id, before_state=before_state)

    def _get_collection(self, element_type: str) -> dict | None:
        """Get collection by element type."""
        collections = {
            "stroke": self.strokes,
            "shape": self.shapes,
            "text": self.texts,
            "note": self.sticky_notes
        }
        return collections.get(element_type)

    def _add_undo_action(
        self,
        action_type: str,
        element_type: str,
        element_id: str,
        before_state: dict | None = None,
        after_state: dict | None = None
    ):
        """Add action to undo stack."""
        action = UndoAction(
            action_type=action_type,
            element_type=element_type,
            element_id=element_id,
            before_state=before_state,
            after_state=after_state
        )

        self.undo_stack.append(action)
        self.redo_stack.clear()  # Clear redo on new action

        # Trim undo stack
        while len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        """Undo last action."""
        if not self.undo_stack:
            return False

        action = self.undo_stack.pop()
        self.redo_stack.append(action)

        if action.action_type == "add":
            collection = self._get_collection(action.element_type)
            if collection and action.element_id in collection:
                del collection[action.element_id]

        elif action.action_type == "remove":
            # Restore element from before_state
            self._restore_element(action.element_type, action.before_state)

        elif action.action_type == "clear":
            # Restore full board state
            self._restore_board_state(action.before_state)

        self.modified_at = datetime.now().timestamp()
        return True

    def redo(self) -> bool:
        """Redo last undone action."""
        if not self.redo_stack:
            return False

        action = self.redo_stack.pop()
        self.undo_stack.append(action)

        if action.action_type == "add":
            # Re-add element from after_state
            self._restore_element(action.element_type, action.after_state)

        elif action.action_type == "remove":
            collection = self._get_collection(action.element_type)
            if collection and action.element_id in collection:
                del collection[action.element_id]

        elif action.action_type == "clear":
            self.strokes.clear()
            self.shapes.clear()
            self.texts.clear()
            self.sticky_notes.clear()

        self.modified_at = datetime.now().timestamp()
        return True

    def _restore_element(self, element_type: str, state: dict[str, Any]):
        """Restore element from state dict."""
        if not state:
            return

        # Simplified restoration - would need full deserialization in production
        pass

    def _restore_board_state(self, state: dict[str, Any]):
        """Restore full board state."""
        if not state:
            return

        # Simplified restoration
        pass

    def add_collaborator(self, user_id: str, name: str, color: Color):
        """Add collaborator to board."""
        self.collaborators[user_id] = {
            "name": name,
            "color": color.to_tuple(),
            "joined_at": datetime.now().timestamp(),
            "cursor_position": None,
            "is_drawing": False
        }

    def remove_collaborator(self, user_id: str):
        """Remove collaborator."""
        self.collaborators.pop(user_id, None)
        self.active_strokes.pop(user_id, None)

    def update_cursor(self, user_id: str, position: tuple[float, float, float]):
        """Update user cursor position."""
        if user_id in self.collaborators:
            self.collaborators[user_id]["cursor_position"] = position
            self.collaborators[user_id]["is_drawing"] = user_id in self.active_strokes

    def get_statistics(self) -> dict[str, Any]:
        """Get board statistics."""
        total_points = sum(len(s.points) for s in self.strokes.values())
        total_length = sum(s.get_length() for s in self.strokes.values())

        return {
            "strokes": len(self.strokes),
            "shapes": len(self.shapes),
            "texts": len(self.texts),
            "sticky_notes": len(self.sticky_notes),
            "total_points": total_points,
            "total_length_meters": round(total_length, 2),
            "collaborators": len(self.collaborators),
            "layers": len(self.layers),
            "undo_available": len(self.undo_stack),
            "redo_available": len(self.redo_stack)
        }

    def to_dict(self) -> dict[str, Any]:
        """Export full board state."""
        return {
            "id": self.board_id,
            "name": self.name,
            "position": self.position.to_tuple(),
            "size": self.size,
            "ownerId": self.owner_id,
            "strokes": {sid: s.to_dict() for sid, s in self.strokes.items()},
            "shapes": {sid: s.to_dict() for sid, s in self.shapes.items()},
            "texts": {tid: t.to_dict() for tid, t in self.texts.items()},
            "stickyNotes": {nid: n.to_dict() for nid, n in self.sticky_notes.items()},
            "layers": [{"id": l.layer_id, "name": l.name, "visible": l.visible} for l in self.layers],
            "activeLayer": self.active_layer,
            "collaborators": self.collaborators,
            "isPublic": self.is_public,
            "createdAt": self.created_at,
            "modifiedAt": self.modified_at,
            "statistics": self.get_statistics()
        }


class WhiteboardManager:
    """
    Manages multiple whiteboards in spatial environment.
    """

    def __init__(self):
        self.boards: dict[str, Whiteboard3D] = {}
        self.active_board: str | None = None

        # User states
        self.user_brushes: dict[str, BrushSettings] = {}

        # Callbacks
        self.on_board_created: list[Callable[[Whiteboard3D], None]] = []
        self.on_board_deleted: list[Callable[[str], None]] = []

    def create_board(
        self,
        board_id: str,
        name: str = "Whiteboard",
        position: tuple[float, float, float] = (0, 1.5, -1),
        size: tuple[float, float] = (2.0, 1.5),
        owner_id: str = ""
    ) -> Whiteboard3D:
        """Create new whiteboard."""
        board = Whiteboard3D(
            board_id=board_id,
            name=name,
            position=position,
            size=size,
            owner_id=owner_id
        )

        self.boards[board_id] = board

        if self.active_board is None:
            self.active_board = board_id

        for callback in self.on_board_created:
            callback(board)

        return board

    def delete_board(self, board_id: str) -> bool:
        """Delete whiteboard."""
        if board_id not in self.boards:
            return False

        del self.boards[board_id]

        if self.active_board == board_id:
            self.active_board = next(iter(self.boards), None)

        for callback in self.on_board_deleted:
            callback(board_id)

        return True

    def get_board(self, board_id: str) -> Whiteboard3D | None:
        """Get whiteboard by ID."""
        return self.boards.get(board_id)

    def set_active_board(self, board_id: str) -> bool:
        """Set active whiteboard."""
        if board_id not in self.boards:
            return False
        self.active_board = board_id
        return True

    def get_active_board(self) -> Whiteboard3D | None:
        """Get currently active board."""
        if self.active_board:
            return self.boards.get(self.active_board)
        return None

    def set_user_brush(self, user_id: str, brush: BrushSettings):
        """Set brush settings for user."""
        self.user_brushes[user_id] = brush

    def get_user_brush(self, user_id: str) -> BrushSettings:
        """Get brush settings for user."""
        return self.user_brushes.get(user_id, BrushSettings())

    def get_statistics(self) -> dict[str, Any]:
        """Get manager statistics."""
        total_strokes = sum(len(b.strokes) for b in self.boards.values())
        total_elements = sum(
            len(b.strokes) + len(b.shapes) + len(b.texts) + len(b.sticky_notes)
            for b in self.boards.values()
        )

        return {
            "total_boards": len(self.boards),
            "total_strokes": total_strokes,
            "total_elements": total_elements,
            "active_board": self.active_board,
            "active_users": len(self.user_brushes)
        }

    def to_state(self) -> dict[str, Any]:
        """Export full state for WebXR client."""
        return {
            "type": "whiteboard_state",
            "boards": {bid: b.to_dict() for bid, b in self.boards.items()},
            "activeBoard": self.active_board,
            "statistics": self.get_statistics(),
            "timestamp": datetime.now().isoformat()
        }

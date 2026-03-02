"""
Gaze Tracking System for HoloLoom Spatial Computing.

Eye/head gaze tracking for attention-based interaction in XR.

Features:
- Eye gaze ray casting (WebXR eye tracking)
- Head gaze fallback (when eye tracking unavailable)
- Gaze dwell selection (look to select)
- Attention heat maps
- Focus prediction
- Gaze-based node highlighting

Created: November 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set
from enum import Enum
from datetime import datetime
import math
import asyncio


class GazeSource(Enum):
    """Source of gaze data."""
    EYE_TRACKING = "eye_tracking"      # Native eye tracking (Quest Pro, etc.)
    HEAD_POSE = "head_pose"            # Head direction fallback
    CONTROLLER = "controller"          # Controller pointing
    HYBRID = "hybrid"                  # Combined sources


class DwellState(Enum):
    """State of dwell selection."""
    NONE = "none"
    DWELLING = "dwelling"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"


@dataclass
class GazeRay:
    """A gaze ray in 3D space."""
    origin_x: float
    origin_y: float
    origin_z: float
    direction_x: float
    direction_y: float
    direction_z: float
    source: GazeSource = GazeSource.EYE_TRACKING
    confidence: float = 1.0  # Eye tracking confidence
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    @property
    def origin(self) -> tuple:
        return (self.origin_x, self.origin_y, self.origin_z)

    @property
    def direction(self) -> tuple:
        return (self.direction_x, self.direction_y, self.direction_z)

    def point_at_distance(self, distance: float) -> tuple:
        """Get point along ray at given distance."""
        return (
            self.origin_x + self.direction_x * distance,
            self.origin_y + self.direction_y * distance,
            self.origin_z + self.direction_z * distance
        )


@dataclass
class GazeHit:
    """Result of gaze ray intersection."""
    hit: bool
    node_id: Optional[str] = None
    position: Optional[tuple] = None
    distance: float = 0.0
    normal: Optional[tuple] = None


@dataclass
class DwellSelection:
    """Dwell selection state for a target."""
    target_id: str
    start_time: float
    duration: float = 0.0
    progress: float = 0.0  # 0.0 to 1.0
    state: DwellState = DwellState.DWELLING
    trigger_duration: float = 1.0  # Seconds to trigger


@dataclass
class AttentionPoint:
    """A point of attention in the scene."""
    position: tuple
    duration: float  # Total time looking
    weight: float  # Attention weight
    node_id: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class GazeFocus:
    """Current gaze focus state."""
    focused_node_id: Optional[str] = None
    focus_duration: float = 0.0
    focus_position: Optional[tuple] = None
    is_stable: bool = False  # Gaze is stable (not saccading)
    stability_score: float = 0.0  # How stable the gaze is


class AttentionHeatMap:
    """Attention heat map tracking where user looks."""

    def __init__(
        self,
        resolution: int = 100,
        bounds: tuple = (-5, -5, -5, 5, 5, 5),  # min_x, min_y, min_z, max_x, max_y, max_z
        decay_rate: float = 0.01  # Heat decay per second
    ):
        self.resolution = resolution
        self.bounds = bounds
        self.decay_rate = decay_rate
        self.grid: Dict[tuple, float] = {}  # (grid_x, grid_y, grid_z) -> heat
        self.last_update = datetime.now().timestamp()

    def _world_to_grid(self, x: float, y: float, z: float) -> tuple:
        """Convert world position to grid coordinates."""
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds

        # Normalize to 0-1
        nx = (x - min_x) / (max_x - min_x) if max_x > min_x else 0.5
        ny = (y - min_y) / (max_y - min_y) if max_y > min_y else 0.5
        nz = (z - min_z) / (max_z - min_z) if max_z > min_z else 0.5

        # Clamp and convert to grid
        gx = max(0, min(self.resolution - 1, int(nx * self.resolution)))
        gy = max(0, min(self.resolution - 1, int(ny * self.resolution)))
        gz = max(0, min(self.resolution - 1, int(nz * self.resolution)))

        return (gx, gy, gz)

    def _grid_to_world(self, gx: int, gy: int, gz: int) -> tuple:
        """Convert grid coordinates to world position."""
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds

        x = min_x + (gx / self.resolution) * (max_x - min_x)
        y = min_y + (gy / self.resolution) * (max_y - min_y)
        z = min_z + (gz / self.resolution) * (max_z - min_z)

        return (x, y, z)

    def add_attention(
        self,
        position: tuple,
        intensity: float = 1.0,
        radius: int = 2
    ):
        """Add attention at a position with spatial spread."""
        center = self._world_to_grid(*position)

        # Add heat with gaussian-like falloff
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    gx = center[0] + dx
                    gy = center[1] + dy
                    gz = center[2] + dz

                    if 0 <= gx < self.resolution and 0 <= gy < self.resolution and 0 <= gz < self.resolution:
                        # Distance-based falloff
                        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                        falloff = math.exp(-dist * 0.5) if dist > 0 else 1.0

                        key = (gx, gy, gz)
                        self.grid[key] = self.grid.get(key, 0.0) + intensity * falloff

    def update(self, delta_time: float):
        """Apply decay to heat map."""
        decay = 1.0 - (self.decay_rate * delta_time)

        # Apply decay and remove cold cells
        cold_cells = []
        for key, heat in self.grid.items():
            new_heat = heat * decay
            if new_heat < 0.01:
                cold_cells.append(key)
            else:
                self.grid[key] = new_heat

        for key in cold_cells:
            del self.grid[key]

    def get_hotspots(self, threshold: float = 0.5, max_count: int = 10) -> List[tuple]:
        """Get world positions of hot spots."""
        # Sort cells by heat
        sorted_cells = sorted(
            self.grid.items(),
            key=lambda x: x[1],
            reverse=True
        )

        hotspots = []
        for (gx, gy, gz), heat in sorted_cells:
            if heat >= threshold and len(hotspots) < max_count:
                world_pos = self._grid_to_world(gx, gy, gz)
                hotspots.append((world_pos, heat))

        return hotspots

    def get_heat_at(self, position: tuple) -> float:
        """Get heat value at a world position."""
        grid_pos = self._world_to_grid(*position)
        return self.grid.get(grid_pos, 0.0)

    def to_visualization_data(self) -> Dict[str, Any]:
        """Export for 3D visualization."""
        points = []
        for (gx, gy, gz), heat in self.grid.items():
            if heat > 0.1:  # Only visible heat
                world_pos = self._grid_to_world(gx, gy, gz)
                points.append({
                    "position": world_pos,
                    "heat": heat,
                    "color": self._heat_to_color(heat)
                })

        return {
            "type": "attention_heatmap",
            "points": points,
            "bounds": self.bounds,
            "resolution": self.resolution,
            "timestamp": datetime.now().isoformat()
        }

    def _heat_to_color(self, heat: float) -> str:
        """Convert heat value to color (blue->red gradient)."""
        # Clamp heat to 0-1
        h = max(0.0, min(1.0, heat))

        # Blue (cold) to Red (hot) through yellow
        if h < 0.5:
            r = int(255 * (h * 2))
            g = int(255 * (h * 2))
            b = int(255 * (1 - h * 2))
        else:
            r = 255
            g = int(255 * (1 - (h - 0.5) * 2))
            b = 0

        return f"#{r:02x}{g:02x}{b:02x}"


class GazeTracker:
    """
    Main gaze tracking system.

    Handles eye/head gaze tracking, dwell selection,
    and attention analysis.
    """

    def __init__(
        self,
        dwell_duration: float = 1.0,  # Seconds to trigger selection
        stability_threshold: float = 0.02,  # Movement threshold for stability
        attention_decay: float = 0.01,
        node_positions: Optional[Dict[str, tuple]] = None,
        node_radii: Optional[Dict[str, float]] = None
    ):
        self.dwell_duration = dwell_duration
        self.stability_threshold = stability_threshold

        # Node geometry for hit testing
        self.node_positions = node_positions or {}
        self.node_radii = node_radii or {}  # Default radius if not specified
        self.default_node_radius = 0.1

        # Gaze state
        self.current_ray: Optional[GazeRay] = None
        self.previous_ray: Optional[GazeRay] = None
        self.gaze_source = GazeSource.HEAD_POSE
        self.eye_tracking_available = False

        # Focus tracking
        self.current_focus = GazeFocus()
        self.focus_history: List[tuple] = []  # (node_id, duration)
        self.max_focus_history = 100

        # Dwell selection
        self.active_dwells: Dict[str, DwellSelection] = {}
        self.dwell_callbacks: List[Callable[[str], None]] = []

        # Attention tracking
        self.attention_map = AttentionHeatMap(decay_rate=attention_decay)
        self.attention_points: List[AttentionPoint] = []
        self.max_attention_points = 1000

        # Gaze prediction (simple velocity-based)
        self.gaze_velocity: Optional[tuple] = None
        self.prediction_enabled = True

        # Callbacks
        self.on_focus_change: List[Callable[[Optional[str], Optional[str]], None]] = []
        self.on_dwell_select: List[Callable[[str], None]] = []
        self.on_attention_hotspot: List[Callable[[tuple, float], None]] = []

    def update_node_positions(self, positions: Dict[str, tuple]):
        """Update node positions for hit testing."""
        self.node_positions.update(positions)

    def update_node_radius(self, node_id: str, radius: float):
        """Set hit test radius for a node."""
        self.node_radii[node_id] = radius

    def set_eye_tracking_available(self, available: bool):
        """Set whether eye tracking hardware is available."""
        self.eye_tracking_available = available
        if available:
            self.gaze_source = GazeSource.EYE_TRACKING
        else:
            self.gaze_source = GazeSource.HEAD_POSE

    def update_gaze(self, ray: GazeRay, delta_time: float = 0.016):
        """Update gaze with new ray data."""
        self.previous_ray = self.current_ray
        self.current_ray = ray

        # Calculate gaze velocity for prediction
        if self.previous_ray:
            self._update_velocity(delta_time)

        # Perform hit test
        hit = self._ray_cast()

        # Update focus
        self._update_focus(hit, delta_time)

        # Update dwell selections
        self._update_dwells(hit, delta_time)

        # Update attention map
        if hit.hit and hit.position:
            self.attention_map.add_attention(hit.position, intensity=delta_time)
            self._add_attention_point(hit.position, hit.node_id)

        # Apply decay
        self.attention_map.update(delta_time)

        return hit

    def _update_velocity(self, delta_time: float):
        """Calculate gaze velocity for prediction."""
        if not self.previous_ray or not self.current_ray:
            return

        dx = self.current_ray.direction_x - self.previous_ray.direction_x
        dy = self.current_ray.direction_y - self.previous_ray.direction_y
        dz = self.current_ray.direction_z - self.previous_ray.direction_z

        if delta_time > 0:
            self.gaze_velocity = (dx / delta_time, dy / delta_time, dz / delta_time)

    def _ray_cast(self) -> GazeHit:
        """Cast gaze ray against scene nodes."""
        if not self.current_ray:
            return GazeHit(hit=False)

        closest_hit: Optional[GazeHit] = None
        closest_distance = float('inf')

        for node_id, position in self.node_positions.items():
            radius = self.node_radii.get(node_id, self.default_node_radius)

            # Ray-sphere intersection
            hit = self._ray_sphere_intersect(
                self.current_ray.origin,
                self.current_ray.direction,
                position,
                radius
            )

            if hit and hit < closest_distance:
                closest_distance = hit
                hit_position = self.current_ray.point_at_distance(hit)

                # Calculate surface normal
                nx = hit_position[0] - position[0]
                ny = hit_position[1] - position[1]
                nz = hit_position[2] - position[2]
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 0:
                    normal = (nx/length, ny/length, nz/length)
                else:
                    normal = (0, 1, 0)

                closest_hit = GazeHit(
                    hit=True,
                    node_id=node_id,
                    position=hit_position,
                    distance=hit,
                    normal=normal
                )

        return closest_hit or GazeHit(hit=False)

    def _ray_sphere_intersect(
        self,
        ray_origin: tuple,
        ray_direction: tuple,
        sphere_center: tuple,
        sphere_radius: float
    ) -> Optional[float]:
        """Ray-sphere intersection test. Returns distance or None."""
        # Vector from ray origin to sphere center
        oc_x = ray_origin[0] - sphere_center[0]
        oc_y = ray_origin[1] - sphere_center[1]
        oc_z = ray_origin[2] - sphere_center[2]

        # Quadratic formula coefficients
        a = (ray_direction[0]**2 + ray_direction[1]**2 + ray_direction[2]**2)
        b = 2.0 * (oc_x * ray_direction[0] + oc_y * ray_direction[1] + oc_z * ray_direction[2])
        c = oc_x**2 + oc_y**2 + oc_z**2 - sphere_radius**2

        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return None

        # Return closest positive intersection
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        return None

    def _update_focus(self, hit: GazeHit, delta_time: float):
        """Update focus state based on hit."""
        old_focus = self.current_focus.focused_node_id

        if hit.hit and hit.node_id:
            if hit.node_id == self.current_focus.focused_node_id:
                # Same node - increase duration
                self.current_focus.focus_duration += delta_time
                self.current_focus.focus_position = hit.position

                # Calculate stability
                if self.previous_ray and self.current_ray:
                    movement = self._calculate_gaze_movement()
                    self.current_focus.is_stable = movement < self.stability_threshold
                    self.current_focus.stability_score = max(0, 1.0 - movement / self.stability_threshold)
            else:
                # New focus target
                if old_focus:
                    self._record_focus_history(old_focus, self.current_focus.focus_duration)

                self.current_focus = GazeFocus(
                    focused_node_id=hit.node_id,
                    focus_duration=delta_time,
                    focus_position=hit.position,
                    is_stable=False,
                    stability_score=0.0
                )

                # Fire focus change callback
                for callback in self.on_focus_change:
                    callback(old_focus, hit.node_id)
        else:
            # No hit - clear focus
            if old_focus:
                self._record_focus_history(old_focus, self.current_focus.focus_duration)

                self.current_focus = GazeFocus()

                for callback in self.on_focus_change:
                    callback(old_focus, None)

    def _calculate_gaze_movement(self) -> float:
        """Calculate gaze movement magnitude."""
        if not self.previous_ray or not self.current_ray:
            return 0.0

        dx = self.current_ray.direction_x - self.previous_ray.direction_x
        dy = self.current_ray.direction_y - self.previous_ray.direction_y
        dz = self.current_ray.direction_z - self.previous_ray.direction_z

        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _record_focus_history(self, node_id: str, duration: float):
        """Record focus to history."""
        self.focus_history.append((node_id, duration))

        # Trim history
        if len(self.focus_history) > self.max_focus_history:
            self.focus_history = self.focus_history[-self.max_focus_history:]

    def _update_dwells(self, hit: GazeHit, delta_time: float):
        """Update dwell selection states."""
        # Get currently gazed node
        current_target = hit.node_id if hit.hit else None

        # Update existing dwells
        completed_dwells = []
        cancelled_dwells = []

        for target_id, dwell in self.active_dwells.items():
            if target_id == current_target:
                # Still looking at target
                dwell.duration += delta_time
                dwell.progress = min(1.0, dwell.duration / dwell.trigger_duration)

                if dwell.progress >= 1.0:
                    dwell.state = DwellState.TRIGGERED
                    completed_dwells.append(target_id)
            else:
                # Looked away - cancel dwell
                dwell.state = DwellState.CANCELLED
                cancelled_dwells.append(target_id)

        # Fire callbacks for completed dwells
        for target_id in completed_dwells:
            for callback in self.on_dwell_select:
                callback(target_id)
            del self.active_dwells[target_id]

        # Remove cancelled dwells
        for target_id in cancelled_dwells:
            del self.active_dwells[target_id]

        # Start new dwell if gazing at unhit target
        if current_target and current_target not in self.active_dwells:
            self.active_dwells[current_target] = DwellSelection(
                target_id=current_target,
                start_time=datetime.now().timestamp(),
                trigger_duration=self.dwell_duration
            )

    def _add_attention_point(self, position: tuple, node_id: Optional[str]):
        """Add attention point to history."""
        self.attention_points.append(AttentionPoint(
            position=position,
            duration=0.016,  # One frame
            weight=1.0,
            node_id=node_id
        ))

        # Trim history
        if len(self.attention_points) > self.max_attention_points:
            self.attention_points = self.attention_points[-self.max_attention_points:]

    def get_predicted_gaze(self, time_ahead: float = 0.1) -> Optional[GazeRay]:
        """Predict where user will look based on velocity."""
        if not self.prediction_enabled or not self.current_ray or not self.gaze_velocity:
            return self.current_ray

        # Simple linear prediction
        predicted = GazeRay(
            origin_x=self.current_ray.origin_x,
            origin_y=self.current_ray.origin_y,
            origin_z=self.current_ray.origin_z,
            direction_x=self.current_ray.direction_x + self.gaze_velocity[0] * time_ahead,
            direction_y=self.current_ray.direction_y + self.gaze_velocity[1] * time_ahead,
            direction_z=self.current_ray.direction_z + self.gaze_velocity[2] * time_ahead,
            source=self.current_ray.source,
            confidence=self.current_ray.confidence * 0.8  # Lower confidence for prediction
        )

        # Normalize direction
        length = math.sqrt(
            predicted.direction_x**2 +
            predicted.direction_y**2 +
            predicted.direction_z**2
        )
        if length > 0:
            predicted.direction_x /= length
            predicted.direction_y /= length
            predicted.direction_z /= length

        return predicted

    def get_focus_statistics(self) -> Dict[str, Any]:
        """Get focus duration statistics per node."""
        node_durations: Dict[str, float] = {}

        for node_id, duration in self.focus_history:
            node_durations[node_id] = node_durations.get(node_id, 0.0) + duration

        # Add current focus
        if self.current_focus.focused_node_id:
            node_id = self.current_focus.focused_node_id
            node_durations[node_id] = node_durations.get(node_id, 0.0) + self.current_focus.focus_duration

        total_duration = sum(node_durations.values())

        return {
            "node_durations": node_durations,
            "total_focus_time": total_duration,
            "most_focused": max(node_durations.items(), key=lambda x: x[1])[0] if node_durations else None,
            "current_focus": self.current_focus.focused_node_id,
            "focus_count": len(self.focus_history) + (1 if self.current_focus.focused_node_id else 0)
        }

    def get_attention_hotspots(self, count: int = 5) -> List[tuple]:
        """Get top attention hotspots."""
        return self.attention_map.get_hotspots(threshold=0.3, max_count=count)

    def register_dwell_callback(self, callback: Callable[[str], None]):
        """Register callback for dwell selection."""
        self.on_dwell_select.append(callback)

    def register_focus_callback(self, callback: Callable[[Optional[str], Optional[str]], None]):
        """Register callback for focus changes (old_id, new_id)."""
        self.on_focus_change.append(callback)

    def to_state(self) -> Dict[str, Any]:
        """Export current state for WebXR client."""
        dwell_states = {}
        for target_id, dwell in self.active_dwells.items():
            dwell_states[target_id] = {
                "progress": dwell.progress,
                "state": dwell.state.value,
                "duration": dwell.duration
            }

        return {
            "type": "gaze_state",
            "gaze_source": self.gaze_source.value,
            "eye_tracking_available": self.eye_tracking_available,
            "current_ray": {
                "origin": self.current_ray.origin if self.current_ray else None,
                "direction": self.current_ray.direction if self.current_ray else None,
                "confidence": self.current_ray.confidence if self.current_ray else 0
            },
            "focus": {
                "node_id": self.current_focus.focused_node_id,
                "duration": self.current_focus.focus_duration,
                "position": self.current_focus.focus_position,
                "is_stable": self.current_focus.is_stable,
                "stability_score": self.current_focus.stability_score
            },
            "dwells": dwell_states,
            "attention_hotspots": [
                {"position": pos, "heat": heat}
                for pos, heat in self.get_attention_hotspots()
            ],
            "timestamp": datetime.now().isoformat()
        }


class GazeHighlighter:
    """Manages gaze-based highlighting effects."""

    def __init__(self, tracker: GazeTracker):
        self.tracker = tracker

        # Highlight states
        self.highlighted_nodes: Set[str] = set()
        self.highlight_intensity: Dict[str, float] = {}

        # Settings
        self.hover_intensity = 0.5
        self.focus_intensity = 1.0
        self.dwell_intensity_multiplier = 1.5
        self.fade_speed = 3.0  # Intensity per second

        # Register callbacks
        tracker.register_focus_callback(self._on_focus_change)

    def _on_focus_change(self, old_id: Optional[str], new_id: Optional[str]):
        """Handle focus change events."""
        if old_id:
            # Start fading old highlight
            pass  # Will fade in update()

        if new_id:
            self.highlighted_nodes.add(new_id)
            self.highlight_intensity[new_id] = self.hover_intensity

    def update(self, delta_time: float):
        """Update highlight states."""
        # Update focused node
        focus = self.tracker.current_focus
        if focus.focused_node_id:
            # Increase intensity based on stability
            target_intensity = self.focus_intensity
            if focus.focused_node_id in self.tracker.active_dwells:
                dwell = self.tracker.active_dwells[focus.focused_node_id]
                target_intensity *= (1.0 + dwell.progress * (self.dwell_intensity_multiplier - 1))

            current = self.highlight_intensity.get(focus.focused_node_id, 0)
            self.highlight_intensity[focus.focused_node_id] = min(
                target_intensity,
                current + self.fade_speed * delta_time
            )

        # Fade non-focused nodes
        nodes_to_remove = []
        for node_id in self.highlighted_nodes:
            if node_id != focus.focused_node_id:
                current = self.highlight_intensity.get(node_id, 0)
                new_intensity = current - self.fade_speed * delta_time

                if new_intensity <= 0:
                    nodes_to_remove.append(node_id)
                else:
                    self.highlight_intensity[node_id] = new_intensity

        for node_id in nodes_to_remove:
            self.highlighted_nodes.discard(node_id)
            self.highlight_intensity.pop(node_id, None)

    def get_highlight_state(self) -> Dict[str, float]:
        """Get current highlight intensities."""
        return dict(self.highlight_intensity)

    def get_highlight_color(self, node_id: str, base_color: str = "#00ff00") -> str:
        """Get highlight color with intensity applied."""
        intensity = self.highlight_intensity.get(node_id, 0)
        if intensity <= 0:
            return base_color

        # Parse base color
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)

        # Blend with highlight (white glow)
        blend = min(1.0, intensity)
        r = int(r + (255 - r) * blend * 0.5)
        g = int(g + (255 - g) * blend * 0.5)
        b = int(b + (255 - b) * blend * 0.5)

        return f"#{r:02x}{g:02x}{b:02x}"

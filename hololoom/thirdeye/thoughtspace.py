"""
Thoughtspace - Full-screen expandable world for 3rdEye.

The Thoughtspace is where concepts become worlds. It manages the
transition from the compact sidebar panel to full-screen immersive
visualization, and handles semantic positioning of concept clusters.

Created: 2025-11-30
Author: HoloLoom Team
"""

import asyncio
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging

from hololoom.thirdeye.concept import (
    Concept,
    ConceptType,
    ConceptWorld,
    SemanticPosition,
)
from hololoom.thirdeye.transition import (
    Transition,
    TransitionType,
    TransitionConfig,
)

logger = logging.getLogger(__name__)


class ThoughtspaceMode(Enum):
    """Display modes for the Thoughtspace."""

    COMPACT = "compact"        # 300px sidebar panel
    EXPANDED = "expanded"      # 50% of screen
    FULLSCREEN = "fullscreen"  # Entire viewport
    OVERLAY = "overlay"        # Transparent overlay on chat


class CameraMode(Enum):
    """Camera behavior modes."""

    FOLLOW = "follow"          # Camera follows focal concept
    FREE = "free"              # User-controlled camera
    ORBIT = "orbit"            # Orbit around focal point
    JOURNEY = "journey"        # Animated path through concepts


@dataclass
class CameraState:
    """Current camera state in the Thoughtspace."""

    # Position
    x: float = 0.0
    y: float = 5.0
    z: float = 15.0

    # Target (look-at point)
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0

    # Orientation
    fov: float = 60.0          # Field of view in degrees
    zoom: float = 1.0          # Zoom level (1.0 = default)

    # Animation
    is_animating: bool = False
    animation_progress: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON for WebSocket streaming."""
        return {
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "target": {"x": self.target_x, "y": self.target_y, "z": self.target_z},
            "fov": self.fov,
            "zoom": self.zoom,
            "is_animating": self.is_animating,
            "animation_progress": self.animation_progress,
        }


@dataclass
class ThoughtspaceState:
    """Complete state of the Thoughtspace for streaming to client."""

    mode: ThoughtspaceMode
    camera: CameraState
    world: ConceptWorld
    camera_mode: CameraMode = CameraMode.FOLLOW

    # UI state
    show_connections: bool = True
    show_labels: bool = True
    background_color: str = "#f8fafc"  # Cream background
    ambient_light: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON for WebSocket streaming."""
        return {
            "mode": self.mode.value,
            "camera_mode": self.camera_mode.value,
            "camera": self.camera.to_dict(),
            "world": self.world.to_dict(),
            "ui": {
                "show_connections": self.show_connections,
                "show_labels": self.show_labels,
                "background_color": self.background_color,
                "ambient_light": self.ambient_light,
            }
        }


class SemanticPositioner:
    """
    Calculate semantic positions for concepts in 3D space.

    Uses embeddings to position related concepts near each other,
    creating natural clusters based on semantic similarity.
    """

    # Layout parameters
    CLUSTER_RADIUS = 8.0       # Radius of concept clusters
    VERTICAL_SPREAD = 5.0      # Y-axis spread
    SEPARATION_FORCE = 2.0     # Force keeping concepts apart

    def __init__(self):
        """Initialize the positioner."""
        self._positions: Dict[str, SemanticPosition] = {}

    def position_concept(
        self,
        concept: Concept,
        existing_concepts: List[Concept],
    ) -> SemanticPosition:
        """
        Calculate position for a new concept.

        Uses semantic similarity to position the concept near
        related concepts while maintaining separation.

        Args:
            concept: The concept to position
            existing_concepts: Concepts already in the space

        Returns:
            Calculated SemanticPosition
        """
        if not existing_concepts:
            # First concept - center of space
            return SemanticPosition(x=0.0, y=0.0, z=0.0)

        # If concept has an embedding, use it for initial position
        if concept.position.embedding:
            # Use first 3 dimensions as base position (scaled)
            emb = concept.position.embedding
            base_x = emb[0] * self.CLUSTER_RADIUS if len(emb) > 0 else 0.0
            base_y = emb[1] * self.VERTICAL_SPREAD if len(emb) > 1 else 0.0
            base_z = emb[2] * self.CLUSTER_RADIUS if len(emb) > 2 else 0.0
        else:
            # No embedding - position based on connections
            base_x, base_y, base_z = self._position_by_connections(
                concept, existing_concepts
            )

        # Apply separation to avoid overlaps
        final_x, final_y, final_z = self._apply_separation(
            base_x, base_y, base_z, existing_concepts
        )

        return SemanticPosition(
            x=final_x,
            y=final_y,
            z=final_z,
            embedding=concept.position.embedding,
        )

    def _position_by_connections(
        self,
        concept: Concept,
        existing: List[Concept],
    ) -> Tuple[float, float, float]:
        """Position concept near its connections."""
        if not concept.connections:
            # No connections - random position
            import random
            angle = random.random() * 2 * math.pi
            radius = self.CLUSTER_RADIUS
            return (
                math.cos(angle) * radius,
                random.random() * self.VERTICAL_SPREAD,
                math.sin(angle) * radius,
            )

        # Average position of connected concepts
        connected = [c for c in existing if c.id in concept.connections]
        if not connected:
            return (0.0, 0.0, 0.0)

        avg_x = sum(c.position.x for c in connected) / len(connected)
        avg_y = sum(c.position.y for c in connected) / len(connected)
        avg_z = sum(c.position.z for c in connected) / len(connected)

        # Offset slightly from the average
        offset = 3.0
        import random
        angle = random.random() * 2 * math.pi
        return (
            avg_x + math.cos(angle) * offset,
            avg_y + random.random() * 2 - 1,
            avg_z + math.sin(angle) * offset,
        )

    def _apply_separation(
        self,
        x: float, y: float, z: float,
        existing: List[Concept],
    ) -> Tuple[float, float, float]:
        """Apply separation force to avoid overlaps."""
        min_distance = 2.5  # Minimum distance between concepts

        for c in existing:
            dx = x - c.position.x
            dy = y - c.position.y
            dz = z - c.position.z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)

            if distance < min_distance and distance > 0.01:
                # Push apart
                force = (min_distance - distance) / distance * self.SEPARATION_FORCE
                x += dx * force
                y += dy * force
                z += dz * force

        return (x, y, z)

    def reposition_all(self, concepts: List[Concept]) -> Dict[str, SemanticPosition]:
        """
        Reposition all concepts using force-directed layout.

        Useful after adding many concepts or when layout needs refresh.

        Args:
            concepts: All concepts to position

        Returns:
            Dict mapping concept IDs to new positions
        """
        if not concepts:
            return {}

        # Initialize positions (use existing or random)
        positions: Dict[str, Tuple[float, float, float]] = {}
        for c in concepts:
            if c.position.x != 0 or c.position.y != 0 or c.position.z != 0:
                positions[c.id] = (c.position.x, c.position.y, c.position.z)
            else:
                import random
                angle = random.random() * 2 * math.pi
                radius = random.random() * self.CLUSTER_RADIUS
                positions[c.id] = (
                    math.cos(angle) * radius,
                    random.random() * self.VERTICAL_SPREAD,
                    math.sin(angle) * radius,
                )

        # Force-directed iterations
        for _ in range(50):  # 50 iterations
            forces: Dict[str, Tuple[float, float, float]] = {
                c.id: (0.0, 0.0, 0.0) for c in concepts
            }

            # Repulsion between all pairs
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    dx = positions[c1.id][0] - positions[c2.id][0]
                    dy = positions[c1.id][1] - positions[c2.id][1]
                    dz = positions[c1.id][2] - positions[c2.id][2]
                    dist_sq = dx*dx + dy*dy + dz*dz + 0.01
                    dist = math.sqrt(dist_sq)

                    repulsion = 5.0 / dist_sq
                    fx = dx / dist * repulsion
                    fy = dy / dist * repulsion
                    fz = dz / dist * repulsion

                    forces[c1.id] = (
                        forces[c1.id][0] + fx,
                        forces[c1.id][1] + fy,
                        forces[c1.id][2] + fz,
                    )
                    forces[c2.id] = (
                        forces[c2.id][0] - fx,
                        forces[c2.id][1] - fy,
                        forces[c2.id][2] - fz,
                    )

            # Attraction along edges
            for c in concepts:
                for conn_id in c.connections:
                    if conn_id in positions:
                        dx = positions[conn_id][0] - positions[c.id][0]
                        dy = positions[conn_id][1] - positions[c.id][1]
                        dz = positions[conn_id][2] - positions[c.id][2]
                        dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01

                        attraction = dist * 0.1
                        fx = dx / dist * attraction
                        fy = dy / dist * attraction
                        fz = dz / dist * attraction

                        forces[c.id] = (
                            forces[c.id][0] + fx,
                            forces[c.id][1] + fy,
                            forces[c.id][2] + fz,
                        )

            # Apply forces with damping
            damping = 0.1
            for c in concepts:
                x, y, z = positions[c.id]
                fx, fy, fz = forces[c.id]
                positions[c.id] = (
                    x + fx * damping,
                    y + fy * damping,
                    z + fz * damping,
                )

        # Convert to SemanticPosition objects
        result: Dict[str, SemanticPosition] = {}
        for c in concepts:
            x, y, z = positions[c.id]
            result[c.id] = SemanticPosition(
                x=x, y=y, z=z,
                embedding=c.position.embedding,
            )

        return result


class Thoughtspace:
    """
    Manager for the full-screen Thoughtspace visualization.

    Handles mode transitions (compact ↔ fullscreen), camera control,
    and coordination between the concept world and the renderer.
    """

    def __init__(
        self,
        on_state_change: Optional[Callable[[ThoughtspaceState], None]] = None,
    ):
        """
        Initialize the Thoughtspace.

        Args:
            on_state_change: Callback when state changes (for streaming)
        """
        self._world = ConceptWorld(name="Thoughtspace")
        self._camera = CameraState()
        self._mode = ThoughtspaceMode.COMPACT
        self._camera_mode = CameraMode.FOLLOW
        self._positioner = SemanticPositioner()
        self._on_state_change = on_state_change

    @property
    def mode(self) -> ThoughtspaceMode:
        """Current display mode."""
        return self._mode

    @property
    def world(self) -> ConceptWorld:
        """The concept world."""
        return self._world

    @property
    def camera(self) -> CameraState:
        """Current camera state."""
        return self._camera

    def get_state(self) -> ThoughtspaceState:
        """Get complete current state."""
        return ThoughtspaceState(
            mode=self._mode,
            camera=self._camera,
            world=self._world,
            camera_mode=self._camera_mode,
        )

    def _emit_state(self) -> None:
        """Emit state change to callback."""
        if self._on_state_change:
            self._on_state_change(self.get_state())

    # Mode transitions

    def expand(self) -> Transition:
        """Expand from compact to fullscreen mode."""
        old_mode = self._mode
        self._mode = ThoughtspaceMode.FULLSCREEN

        # Zoom camera out to show more of the world
        self._camera.z = 25.0
        self._camera.y = 10.0

        self._emit_state()

        return Transition(
            transition_type=TransitionType.ZOOM,
            config=TransitionConfig(duration_ms=600, easing="easeInOutQuad"),
            triggered_by="expand",
        )

    def collapse(self) -> Transition:
        """Collapse from fullscreen to compact mode."""
        self._mode = ThoughtspaceMode.COMPACT

        # Zoom camera in to focus on current concept
        if self._world.focal_concept_id:
            focal = self._world.get_concept(self._world.focal_concept_id)
            if focal:
                self._camera.target_x = focal.position.x
                self._camera.target_y = focal.position.y
                self._camera.target_z = focal.position.z
        self._camera.z = 15.0
        self._camera.y = 5.0

        self._emit_state()

        return Transition(
            transition_type=TransitionType.ZOOM,
            config=TransitionConfig(duration_ms=400, easing="easeInOutQuad"),
            triggered_by="collapse",
        )

    def toggle_mode(self) -> Transition:
        """Toggle between compact and fullscreen."""
        if self._mode == ThoughtspaceMode.COMPACT:
            return self.expand()
        else:
            return self.collapse()

    # Concept management

    def add_concept(self, concept: Concept) -> None:
        """
        Add a concept to the world.

        Positions the concept based on semantic similarity to existing
        concepts and updates the camera to track it.
        """
        # Calculate position
        existing = list(self._world.concepts.values())
        position = self._positioner.position_concept(concept, existing)
        concept.position = position

        # Add to world
        self._world.add_concept(concept)

        # Focus camera on new concept (if in follow mode)
        if self._camera_mode == CameraMode.FOLLOW:
            self._world.set_focus(concept.id)
            self._look_at(concept)

        self._emit_state()

    def focus_concept(self, concept_id: str) -> Optional[Transition]:
        """Focus camera on a specific concept."""
        concept = self._world.get_concept(concept_id)
        if not concept:
            return None

        self._world.set_focus(concept_id)
        transition = self._look_at(concept)
        self._emit_state()

        return transition

    def _look_at(self, concept: Concept) -> Transition:
        """Smoothly move camera to look at a concept."""
        # Calculate camera position (slightly offset from concept)
        distance = 12.0
        self._camera.target_x = concept.position.x
        self._camera.target_y = concept.position.y
        self._camera.target_z = concept.position.z
        self._camera.x = concept.position.x
        self._camera.y = concept.position.y + 3.0
        self._camera.z = concept.position.z + distance

        return Transition(
            transition_type=TransitionType.ZOOM,
            config=TransitionConfig(duration_ms=500, easing="easeInOutCubic"),
            to_concept_id=concept.id,
            triggered_by="focus",
        )

    def relayout(self) -> None:
        """Recompute positions for all concepts."""
        concepts = list(self._world.concepts.values())
        new_positions = self._positioner.reposition_all(concepts)

        for concept_id, position in new_positions.items():
            concept = self._world.get_concept(concept_id)
            if concept:
                concept.position = position

        self._emit_state()

    # Camera control

    def set_camera_mode(self, mode: CameraMode) -> None:
        """Set camera behavior mode."""
        self._camera_mode = mode
        self._emit_state()

    def orbit(self, angle_delta: float) -> None:
        """Rotate camera around the focal point."""
        if not self._world.focal_concept_id:
            return

        focal = self._world.get_concept(self._world.focal_concept_id)
        if not focal:
            return

        # Current distance from focal point
        dx = self._camera.x - focal.position.x
        dz = self._camera.z - focal.position.z
        distance = math.sqrt(dx*dx + dz*dz)

        # Current angle
        current_angle = math.atan2(dz, dx)
        new_angle = current_angle + angle_delta

        # New position
        self._camera.x = focal.position.x + math.cos(new_angle) * distance
        self._camera.z = focal.position.z + math.sin(new_angle) * distance

        self._emit_state()

    def zoom(self, factor: float) -> None:
        """Zoom camera in (factor < 1) or out (factor > 1)."""
        # Move camera along look direction
        dx = self._camera.x - self._camera.target_x
        dy = self._camera.y - self._camera.target_y
        dz = self._camera.z - self._camera.target_z

        self._camera.x = self._camera.target_x + dx * factor
        self._camera.y = self._camera.target_y + dy * factor
        self._camera.z = self._camera.target_z + dz * factor

        self._camera.zoom = 1.0 / factor

        self._emit_state()


# Factory function
def create_thoughtspace(
    on_state_change: Optional[Callable[[ThoughtspaceState], None]] = None,
) -> Thoughtspace:
    """
    Create a Thoughtspace instance.

    Args:
        on_state_change: Callback when state changes

    Returns:
        Configured Thoughtspace instance
    """
    return Thoughtspace(on_state_change=on_state_change)

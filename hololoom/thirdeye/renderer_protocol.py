"""
Renderer Protocol - Abstract rendering interface for 3rdEye.

Defines the ConceptRenderer protocol that all renderers must implement.
Separates WHAT to render from HOW to render it.

Created: 2025-11-30
Author: HoloLoom Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable

from hololoom.thirdeye.concept import Concept, ConceptWorld
from hololoom.thirdeye.transition import Transition
from hololoom.thirdeye.thoughtspace import ThoughtspaceState


class RenderDimension(Enum):
    """Rendering dimensionality."""

    ONE_D = "1d"       # Sparklines, inline visualizations
    TWO_D = "2d"       # Tufte-style SVG/CSS
    THREE_D = "3d"     # WebGL/Three.js
    AR = "ar"          # WebXR augmented reality


class RenderQuality(Enum):
    """Rendering quality presets."""

    LOW = "low"           # Mobile, low-power devices
    MEDIUM = "medium"     # Standard desktop
    HIGH = "high"         # High-end desktop
    ULTRA = "ultra"       # Maximum quality (research/demo)


@dataclass
class RenderConfig:
    """Configuration for rendering."""

    dimension: RenderDimension = RenderDimension.THREE_D
    quality: RenderQuality = RenderQuality.MEDIUM

    # Size
    width: int = 800
    height: int = 600

    # Visual settings
    background_color: str = "#f8fafc"    # Cream
    ambient_light: float = 0.6
    enable_shadows: bool = True
    enable_bloom: bool = False           # Glow effects
    enable_antialiasing: bool = True

    # Performance
    target_fps: int = 60
    max_concepts: int = 100              # Cull beyond this

    # Labels
    show_labels: bool = True
    label_font_size: int = 12
    label_font_family: str = "system-ui, sans-serif"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON for client configuration."""
        return {
            "dimension": self.dimension.value,
            "quality": self.quality.value,
            "width": self.width,
            "height": self.height,
            "background_color": self.background_color,
            "ambient_light": self.ambient_light,
            "enable_shadows": self.enable_shadows,
            "enable_bloom": self.enable_bloom,
            "enable_antialiasing": self.enable_antialiasing,
            "target_fps": self.target_fps,
            "max_concepts": self.max_concepts,
            "show_labels": self.show_labels,
            "label_font_size": self.label_font_size,
            "label_font_family": self.label_font_family,
        }


@dataclass
class RenderCommand:
    """
    A command to be executed by the renderer.

    Represents a single rendering operation that can be
    serialized and sent to the client.
    """

    command_type: str
    target_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON for WebSocket streaming."""
        return {
            "type": self.command_type,
            "target": self.target_id,
            "data": self.data,
        }


@dataclass
class RenderFrame:
    """
    A complete rendering frame.

    Contains all commands needed to render a single frame,
    enabling batched updates for performance.
    """

    frame_id: int
    commands: List[RenderCommand] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_command(self, command: RenderCommand) -> None:
        """Add a command to this frame."""
        self.commands.append(command)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON for WebSocket streaming."""
        return {
            "frame_id": self.frame_id,
            "commands": [cmd.to_dict() for cmd in self.commands],
            "metadata": self.metadata,
        }


@runtime_checkable
class ConceptRenderer(Protocol):
    """
    Protocol for concept visualization renderers.

    All renderers (2D, 3D, AR) must implement this interface.
    This enables swapping renderers without changing visualization logic.
    """

    def configure(self, config: RenderConfig) -> None:
        """Configure the renderer with settings."""
        ...

    def render_concept(self, concept: Concept) -> RenderCommand:
        """
        Generate render command for a single concept.

        Args:
            concept: The concept to render

        Returns:
            RenderCommand that creates the visual representation
        """
        ...

    def render_connection(
        self,
        from_concept: Concept,
        to_concept: Concept,
        edge_type: str = "default",
    ) -> RenderCommand:
        """
        Generate render command for a connection between concepts.

        Args:
            from_concept: Source concept
            to_concept: Target concept
            edge_type: Type of edge (affects styling)

        Returns:
            RenderCommand that creates the connection visual
        """
        ...

    def render_transition(
        self,
        transition: Transition,
        from_concept: Optional[Concept],
        to_concept: Concept,
    ) -> List[RenderCommand]:
        """
        Generate render commands for a concept transition.

        Args:
            transition: The transition to animate
            from_concept: Previous concept (None if first)
            to_concept: New concept

        Returns:
            List of RenderCommands for the animation
        """
        ...

    def render_world(self, world: ConceptWorld) -> RenderFrame:
        """
        Generate a complete frame for the entire world.

        Args:
            world: The concept world to render

        Returns:
            Complete RenderFrame with all commands
        """
        ...

    def render_state(self, state: ThoughtspaceState) -> RenderFrame:
        """
        Generate a complete frame from Thoughtspace state.

        Args:
            state: Complete Thoughtspace state

        Returns:
            Complete RenderFrame with all commands
        """
        ...

    def update_concept(self, concept: Concept) -> RenderCommand:
        """
        Generate command to update an existing concept visual.

        Args:
            concept: The concept with updated properties

        Returns:
            RenderCommand to update the visual
        """
        ...

    def remove_concept(self, concept_id: str) -> RenderCommand:
        """
        Generate command to remove a concept visual.

        Args:
            concept_id: ID of concept to remove

        Returns:
            RenderCommand to remove the visual
        """
        ...

    def set_camera(
        self,
        x: float, y: float, z: float,
        target_x: float, target_y: float, target_z: float,
    ) -> RenderCommand:
        """
        Generate command to set camera position.

        Args:
            x, y, z: Camera position
            target_x, target_y, target_z: Look-at point

        Returns:
            RenderCommand to set camera
        """
        ...


class BaseConceptRenderer(ABC):
    """
    Base class for concept renderers.

    Provides common functionality and default implementations
    that specific renderers can override.
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        """Initialize with optional config."""
        self._config = config or RenderConfig()
        self._frame_counter = 0

    def configure(self, config: RenderConfig) -> None:
        """Configure the renderer."""
        self._config = config

    @property
    def config(self) -> RenderConfig:
        """Current configuration."""
        return self._config

    def next_frame_id(self) -> int:
        """Get next frame ID."""
        self._frame_counter += 1
        return self._frame_counter

    # Default implementations that can be overridden

    def render_world(self, world: ConceptWorld) -> RenderFrame:
        """Default world rendering - render all concepts and connections."""
        frame = RenderFrame(frame_id=self.next_frame_id())

        # Render concepts
        for concept in world.concepts.values():
            frame.add_command(self.render_concept(concept))

        # Render connections
        for concept in world.concepts.values():
            for conn_id in concept.connections:
                if conn_id in world.concepts:
                    frame.add_command(
                        self.render_connection(
                            concept,
                            world.concepts[conn_id],
                        )
                    )

        # Set camera to focal concept if any
        if world.focal_concept_id and world.focal_concept_id in world.concepts:
            focal = world.concepts[world.focal_concept_id]
            frame.add_command(
                self.set_camera(
                    focal.position.x,
                    focal.position.y + 5.0,
                    focal.position.z + 15.0,
                    focal.position.x,
                    focal.position.y,
                    focal.position.z,
                )
            )

        return frame

    def render_state(self, state: ThoughtspaceState) -> RenderFrame:
        """Render from complete Thoughtspace state."""
        frame = self.render_world(state.world)

        # Add camera from state
        frame.add_command(
            self.set_camera(
                state.camera.x,
                state.camera.y,
                state.camera.z,
                state.camera.target_x,
                state.camera.target_y,
                state.camera.target_z,
            )
        )

        return frame

    @abstractmethod
    def render_concept(self, concept: Concept) -> RenderCommand:
        """Must be implemented by subclasses."""
        pass

    @abstractmethod
    def render_connection(
        self,
        from_concept: Concept,
        to_concept: Concept,
        edge_type: str = "default",
    ) -> RenderCommand:
        """Must be implemented by subclasses."""
        pass

    @abstractmethod
    def render_transition(
        self,
        transition: Transition,
        from_concept: Optional[Concept],
        to_concept: Concept,
    ) -> List[RenderCommand]:
        """Must be implemented by subclasses."""
        pass

    @abstractmethod
    def update_concept(self, concept: Concept) -> RenderCommand:
        """Must be implemented by subclasses."""
        pass

    @abstractmethod
    def remove_concept(self, concept_id: str) -> RenderCommand:
        """Must be implemented by subclasses."""
        pass

    @abstractmethod
    def set_camera(
        self,
        x: float, y: float, z: float,
        target_x: float, target_y: float, target_z: float,
    ) -> RenderCommand:
        """Must be implemented by subclasses."""
        pass

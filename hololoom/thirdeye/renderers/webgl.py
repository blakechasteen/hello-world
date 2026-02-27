"""
WebGL Renderer - 3D visualization for 3rdEye using Three.js.

Generates render commands that the TypeScript client executes
using Three.js for GPU-accelerated 3D visualization.

Created: 2025-11-30
Author: HoloLoom Team
"""

from typing import Dict, Any, List, Optional

from hololoom.thirdeye.concept import Concept, ConceptType
from hololoom.thirdeye.transition import Transition, TransitionType
from hololoom.thirdeye.renderer_protocol import (
    BaseConceptRenderer,
    RenderCommand,
    RenderConfig,
    RenderQuality,
)


# Color palette - soft, elegant colors on cream background
CONCEPT_COLORS = {
    ConceptType.ENTITY: "#3b82f6",        # Soft blue
    ConceptType.RELATIONSHIP: "#10b981",   # Soft green
    ConceptType.PROCESS: "#8b5cf6",        # Soft purple
    ConceptType.COMPARISON: "#f59e0b",     # Soft amber
    ConceptType.HIERARCHY: "#6366f1",      # Indigo
    ConceptType.TEMPORAL: "#06b6d4",       # Cyan
    ConceptType.PROBABILITY: "#ec4899",    # Pink
    ConceptType.DECISION: "#f97316",       # Orange
    ConceptType.ACTIVATION: "#eab308",     # Yellow
    ConceptType.MEMORY: "#14b8a6",         # Teal
    ConceptType.CLUSTER: "#a855f7",        # Purple
    ConceptType.THREAD: "#64748b",         # Slate
    ConceptType.WORLD: "#0ea5e9",          # Sky blue
}

# Edge colors by type
EDGE_COLORS = {
    "default": "#94a3b8",     # Slate-400
    "IS_A": "#3b82f6",        # Blue
    "USES": "#10b981",        # Green
    "MENTIONS": "#9ca3af",    # Gray
    "LEADS_TO": "#f59e0b",    # Amber
    "PART_OF": "#8b5cf6",     # Purple
}


class WebGLRenderer(BaseConceptRenderer):
    """
    Three.js-based 3D renderer for concepts.

    Generates JSON commands that the TypeScript client interprets
    to create Three.js scene objects.
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        """Initialize the WebGL renderer."""
        super().__init__(config)
        self._quality_settings = self._get_quality_settings()

    def _get_quality_settings(self) -> Dict[str, Any]:
        """Get quality-dependent settings."""
        quality = self._config.quality

        if quality == RenderQuality.LOW:
            return {
                "sphere_segments": 8,
                "line_segments": 4,
                "enable_shadows": False,
                "enable_glow": False,
                "max_lights": 2,
            }
        elif quality == RenderQuality.MEDIUM:
            return {
                "sphere_segments": 16,
                "line_segments": 8,
                "enable_shadows": True,
                "enable_glow": False,
                "max_lights": 4,
            }
        elif quality == RenderQuality.HIGH:
            return {
                "sphere_segments": 32,
                "line_segments": 16,
                "enable_shadows": True,
                "enable_glow": True,
                "max_lights": 8,
            }
        else:  # ULTRA
            return {
                "sphere_segments": 64,
                "line_segments": 32,
                "enable_shadows": True,
                "enable_glow": True,
                "max_lights": 16,
            }

    def render_concept(self, concept: Concept) -> RenderCommand:
        """
        Render a concept as a 3D sphere with label.

        The sphere size and color vary based on concept properties.
        """
        color = CONCEPT_COLORS.get(concept.concept_type, "#64748b")

        # Size based on importance (0.5 to 2.0)
        base_size = 0.5 + concept.importance * 1.5

        # Opacity based on confidence
        opacity = 0.7 + concept.confidence * 0.3

        return RenderCommand(
            command_type="create_concept",
            target_id=concept.id,
            data={
                "position": {
                    "x": concept.position.x,
                    "y": concept.position.y,
                    "z": concept.position.z,
                },
                "geometry": {
                    "type": "sphere",
                    "radius": base_size,
                    "segments": self._quality_settings["sphere_segments"],
                },
                "material": {
                    "type": "phong",  # MeshPhongMaterial
                    "color": color,
                    "opacity": opacity,
                    "transparent": opacity < 1.0,
                    "emissive": color,
                    "emissiveIntensity": 0.1,
                },
                "label": {
                    "text": concept.label,
                    "visible": self._config.show_labels,
                    "fontSize": self._config.label_font_size,
                    "fontFamily": self._config.label_font_family,
                    "color": "#1e293b",  # Slate-800
                    "offset": {"x": 0, "y": base_size + 0.5, "z": 0},
                },
                "metadata": {
                    "type": concept.concept_type.name,
                    "importance": concept.importance,
                    "confidence": concept.confidence,
                },
            },
        )

    def render_connection(
        self,
        from_concept: Concept,
        to_concept: Concept,
        edge_type: str = "default",
    ) -> RenderCommand:
        """
        Render a connection as a curved line between concepts.

        Uses quadratic bezier curves for elegant arcs.
        """
        color = EDGE_COLORS.get(edge_type, EDGE_COLORS["default"])

        # Calculate midpoint with offset for curve
        mid_x = (from_concept.position.x + to_concept.position.x) / 2
        mid_y = (from_concept.position.y + to_concept.position.y) / 2 + 1.0  # Lift up
        mid_z = (from_concept.position.z + to_concept.position.z) / 2

        return RenderCommand(
            command_type="create_connection",
            target_id=f"edge_{from_concept.id}_{to_concept.id}",
            data={
                "from": {
                    "x": from_concept.position.x,
                    "y": from_concept.position.y,
                    "z": from_concept.position.z,
                },
                "to": {
                    "x": to_concept.position.x,
                    "y": to_concept.position.y,
                    "z": to_concept.position.z,
                },
                "control": {
                    "x": mid_x,
                    "y": mid_y,
                    "z": mid_z,
                },
                "geometry": {
                    "type": "quadratic_curve",
                    "segments": self._quality_settings["line_segments"],
                },
                "material": {
                    "type": "line",
                    "color": color,
                    "opacity": 0.6,
                    "linewidth": 1.5,
                },
                "edge_type": edge_type,
            },
        )

    def render_transition(
        self,
        transition: Transition,
        from_concept: Optional[Concept],
        to_concept: Concept,
    ) -> List[RenderCommand]:
        """
        Generate animation commands for a concept transition.

        Returns multiple commands that the client executes in sequence.
        """
        commands: List[RenderCommand] = []
        duration = transition.config.duration_ms
        easing = transition.config.easing

        if transition.transition_type == TransitionType.MORPH:
            # Smooth morph between shapes
            if from_concept:
                commands.append(RenderCommand(
                    command_type="animate_morph",
                    target_id=from_concept.id,
                    data={
                        "to_id": to_concept.id,
                        "duration_ms": duration,
                        "easing": easing,
                        "scale_peak": transition.config.scale_factor,
                    },
                ))
            else:
                # No from concept - just fade in the new one
                commands.append(self.render_concept(to_concept))
                commands.append(RenderCommand(
                    command_type="animate_fade_in",
                    target_id=to_concept.id,
                    data={
                        "duration_ms": duration,
                        "easing": easing,
                    },
                ))

        elif transition.transition_type == TransitionType.DISSOLVE_EMERGE:
            # Fade out old, fade in new
            if from_concept:
                commands.append(RenderCommand(
                    command_type="animate_fade_out",
                    target_id=from_concept.id,
                    data={
                        "duration_ms": duration // 2,
                        "easing": "easeIn",
                        "blur_peak": transition.config.blur_peak,
                        "remove_after": True,
                    },
                ))

            # Create and fade in new concept
            commands.append(self.render_concept(to_concept))
            commands.append(RenderCommand(
                command_type="animate_fade_in",
                target_id=to_concept.id,
                data={
                    "duration_ms": duration // 2,
                    "delay_ms": duration // 3,  # Overlap
                    "easing": "easeOut",
                },
            ))

        elif transition.transition_type == TransitionType.COLLAPSE_EXPAND:
            # Shrink old, expand new
            if from_concept:
                commands.append(RenderCommand(
                    command_type="animate_scale",
                    target_id=from_concept.id,
                    data={
                        "to_scale": 0.0,
                        "duration_ms": duration // 2,
                        "easing": "easeIn",
                        "remove_after": True,
                    },
                ))

            commands.append(self.render_concept(to_concept))
            commands.append(RenderCommand(
                command_type="animate_scale",
                target_id=to_concept.id,
                data={
                    "from_scale": 0.0,
                    "to_scale": 1.0,
                    "duration_ms": duration // 2,
                    "delay_ms": duration // 3,
                    "easing": "easeOutBack",
                },
            ))

        elif transition.transition_type == TransitionType.PULSE:
            # Emphasis pulse (no change, just highlight)
            commands.append(RenderCommand(
                command_type="animate_pulse",
                target_id=to_concept.id,
                data={
                    "scale_peak": transition.config.scale_factor,
                    "duration_ms": duration,
                    "easing": "easeInOutQuad",
                    "glow_color": CONCEPT_COLORS.get(to_concept.concept_type, "#fff"),
                },
            ))

        elif transition.transition_type == TransitionType.BRANCH:
            # One splits into many (handled at higher level)
            commands.append(RenderCommand(
                command_type="animate_branch",
                target_id=from_concept.id if from_concept else None,
                data={
                    "duration_ms": duration,
                    "easing": easing,
                    "stagger_ms": transition.config.stagger_ms,
                },
            ))

        elif transition.transition_type == TransitionType.ZOOM:
            # Camera movement (not concept animation)
            commands.append(RenderCommand(
                command_type="animate_camera",
                target_id="camera",
                data={
                    "to_target": to_concept.id,
                    "duration_ms": duration,
                    "easing": easing,
                },
            ))

        else:
            # Default: just render the new concept
            commands.append(self.render_concept(to_concept))

        return commands

    def update_concept(self, concept: Concept) -> RenderCommand:
        """Update an existing concept's visual properties."""
        color = CONCEPT_COLORS.get(concept.concept_type, "#64748b")
        base_size = 0.5 + concept.importance * 1.5
        opacity = 0.7 + concept.confidence * 0.3

        return RenderCommand(
            command_type="update_concept",
            target_id=concept.id,
            data={
                "position": {
                    "x": concept.position.x,
                    "y": concept.position.y,
                    "z": concept.position.z,
                },
                "scale": base_size,
                "material": {
                    "color": color,
                    "opacity": opacity,
                },
                "label": {
                    "text": concept.label,
                    "visible": self._config.show_labels,
                },
            },
        )

    def remove_concept(self, concept_id: str) -> RenderCommand:
        """Remove a concept from the scene."""
        return RenderCommand(
            command_type="remove_concept",
            target_id=concept_id,
            data={
                "animate": True,
                "duration_ms": 300,
            },
        )

    def set_camera(
        self,
        x: float, y: float, z: float,
        target_x: float, target_y: float, target_z: float,
    ) -> RenderCommand:
        """Set camera position and look-at target."""
        return RenderCommand(
            command_type="set_camera",
            target_id="camera",
            data={
                "position": {"x": x, "y": y, "z": z},
                "target": {"x": target_x, "y": target_y, "z": target_z},
                "animate": True,
                "duration_ms": 500,
            },
        )

    # Additional WebGL-specific methods

    def set_lighting(
        self,
        ambient: float = 0.6,
        directional: float = 0.8,
        direction: tuple = (1, 1, 1),
    ) -> RenderCommand:
        """Configure scene lighting."""
        return RenderCommand(
            command_type="set_lighting",
            target_id="lights",
            data={
                "ambient": {
                    "intensity": ambient,
                    "color": "#ffffff",
                },
                "directional": {
                    "intensity": directional,
                    "color": "#ffffff",
                    "position": {"x": direction[0], "y": direction[1], "z": direction[2]},
                    "cast_shadows": self._quality_settings["enable_shadows"],
                },
            },
        )

    def set_background(self, color: str = "#f8fafc") -> RenderCommand:
        """Set scene background color."""
        return RenderCommand(
            command_type="set_background",
            target_id="scene",
            data={
                "color": color,
                "fog_enabled": True,
                "fog_color": color,
                "fog_near": 50,
                "fog_far": 100,
            },
        )

    def create_floor_grid(
        self,
        size: float = 100,
        divisions: int = 20,
    ) -> RenderCommand:
        """Create a subtle floor grid for spatial reference."""
        return RenderCommand(
            command_type="create_grid",
            target_id="floor_grid",
            data={
                "size": size,
                "divisions": divisions,
                "color": "#e2e8f0",      # Slate-200
                "opacity": 0.3,
                "position_y": -5.0,
            },
        )


# Factory function
def create_webgl_renderer(config: Optional[RenderConfig] = None) -> WebGLRenderer:
    """Create a WebGL renderer instance."""
    return WebGLRenderer(config)

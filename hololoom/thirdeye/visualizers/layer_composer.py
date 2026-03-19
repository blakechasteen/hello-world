"""
Layer Composer
==============

Combines multiple visualization modes into coherent layered scenes.

Layers stack naturally:
- Base layer: World/Environment (backdrop)
- Object layer: Physical objects, characters
- Overlay layer: UI elements, annotations
- HUD layer: Data, metrics, always-visible info

Philosophy: "Complex scenes are simple layers composed thoughtfully."

Created: 2025-12-01
Author: HoloLoom Team
"""

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LayerType(Enum):
    """Types of layers in the scene stack."""
    BACKGROUND = "background"  # Skybox, distant environment
    BASE = "base"              # Ground plane, room walls
    WORLD = "world"            # Environment props (trees, buildings)
    OBJECT = "object"          # Physical objects, furniture
    CHARACTER = "character"    # Characters, actors
    FOCUS = "focus"            # Primary subject of discussion
    OVERLAY = "overlay"        # UI elements, annotations
    HUD = "hud"                # Always-visible data, metrics


@dataclass
class LayerConfig:
    """Configuration for a layer."""
    layer_type: LayerType
    z_offset: float = 0.0          # Z position offset
    y_offset: float = 0.0          # Y position offset
    scale: float = 1.0             # Scale multiplier
    opacity: float = 1.0           # Layer opacity
    blur: float = 0.0              # Depth-of-field blur
    interactive: bool = True       # Can be clicked/hovered
    visible: bool = True
    blend_mode: str = "normal"     # normal, add, multiply


# Default layer configurations
DEFAULT_LAYER_CONFIGS: dict[LayerType, LayerConfig] = {
    LayerType.BACKGROUND: LayerConfig(
        layer_type=LayerType.BACKGROUND,
        z_offset=-50.0,
        scale=5.0,
        opacity=0.8,
        blur=2.0,
        interactive=False,
    ),
    LayerType.BASE: LayerConfig(
        layer_type=LayerType.BASE,
        z_offset=-20.0,
        y_offset=-2.0,
        scale=2.0,
        blur=0.5,
    ),
    LayerType.WORLD: LayerConfig(
        layer_type=LayerType.WORLD,
        z_offset=-10.0,
        blur=0.3,
    ),
    LayerType.OBJECT: LayerConfig(
        layer_type=LayerType.OBJECT,
        z_offset=-5.0,
    ),
    LayerType.CHARACTER: LayerConfig(
        layer_type=LayerType.CHARACTER,
        z_offset=0.0,
    ),
    LayerType.FOCUS: LayerConfig(
        layer_type=LayerType.FOCUS,
        z_offset=2.0,
        scale=1.1,  # Slightly larger for emphasis
    ),
    LayerType.OVERLAY: LayerConfig(
        layer_type=LayerType.OVERLAY,
        z_offset=5.0,
        opacity=0.95,
    ),
    LayerType.HUD: LayerConfig(
        layer_type=LayerType.HUD,
        z_offset=10.0,
        opacity=0.9,
        blur=0.0,
    ),
}


@dataclass
class LayeredElement:
    """An element assigned to a layer."""
    element: dict[str, Any]
    layer: LayerType
    priority: int = 0  # Within-layer ordering


@dataclass
class ComposedScene:
    """A scene composed of multiple layers."""
    layers: dict[LayerType, list[dict[str, Any]]]
    layer_configs: dict[LayerType, LayerConfig]
    camera: dict[str, Any]
    lighting: dict[str, Any]
    background: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_flat_elements(self) -> list[dict[str, Any]]:
        """Get all elements flattened with layer info applied."""
        elements = []

        for layer_type in LayerType:
            if layer_type not in self.layers:
                continue

            config = self.layer_configs.get(layer_type, DEFAULT_LAYER_CONFIGS[layer_type])
            if not config.visible:
                continue

            for elem in self.layers[layer_type]:
                # Apply layer transforms
                transformed = self._apply_layer_transform(elem, config)
                transformed['layer'] = layer_type.value
                elements.append(transformed)

        return elements

    def _apply_layer_transform(
        self,
        element: dict[str, Any],
        config: LayerConfig
    ) -> dict[str, Any]:
        """Apply layer configuration to element."""
        result = copy.deepcopy(element)

        # Apply position offset
        pos = result.get('position', {'x': 0, 'y': 0, 'z': 0})
        result['position'] = {
            'x': pos.get('x', 0),
            'y': pos.get('y', 0) + config.y_offset,
            'z': pos.get('z', 0) + config.z_offset,
        }

        # Apply scale
        if 'size' in result:
            size = result['size']
            result['size'] = {
                k: v * config.scale for k, v in size.items()
            }
        elif 'scale' in result:
            result['scale'] = result['scale'] * config.scale

        # Apply opacity
        result['opacity'] = result.get('opacity', 1.0) * config.opacity

        # Add layer metadata
        result['layerConfig'] = {
            'blur': config.blur,
            'interactive': config.interactive,
            'blendMode': config.blend_mode,
        }

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mode': 'layered',
            'elements': self.get_flat_elements(),
            'layers': {
                layer.value: len(elements)
                for layer, elements in self.layers.items()
            },
            'camera': self.camera,
            'lighting': self.lighting,
            'background': self.background,
            'metadata': self.metadata,
        }


class LayerComposer:
    """
    Composes multiple visualization modes into layered scenes.

    Handles:
    - Layer assignment based on element types
    - Z-ordering and depth management
    - Layer blending and transparency
    - Focus management (what's in foreground)
    """

    def __init__(self):
        self.layer_configs = copy.deepcopy(DEFAULT_LAYER_CONFIGS)

    def compose(
        self,
        scenes: list[dict[str, Any]],
        focus_mode: str | None = None,
    ) -> ComposedScene:
        """
        Compose multiple scenes into layers.

        Args:
            scenes: List of scene dictionaries from various visualizers
            focus_mode: Which mode should be in focus (foreground)

        Returns:
            ComposedScene with all elements layered
        """
        layers: dict[LayerType, list[dict[str, Any]]] = {lt: [] for lt in LayerType}

        # Process each scene
        for scene in scenes:
            mode = scene.get('mode', 'abstract')
            elements = scene.get('elements', [])

            for element in elements:
                layer = self._determine_layer(element, mode, focus_mode)
                layers[layer].append(element)

        # Get camera and lighting from focus scene or first scene
        focus_scene = next(
            (s for s in scenes if s.get('mode') == focus_mode),
            scenes[0] if scenes else {}
        )

        camera = focus_scene.get('camera', self._default_camera())
        lighting = focus_scene.get('lighting', 'neutral')
        background = self._compose_background(scenes)

        return ComposedScene(
            layers=layers,
            layer_configs=self.layer_configs,
            camera=camera,
            lighting=self._normalize_lighting(lighting),
            background=background,
            metadata={
                'scene_count': len(scenes),
                'modes': [s.get('mode', 'unknown') for s in scenes],
                'focus_mode': focus_mode,
            }
        )

    def _determine_layer(
        self,
        element: dict[str, Any],
        mode: str,
        focus_mode: str | None
    ) -> LayerType:
        """Determine which layer an element belongs to."""
        element_type = element.get('type', 'unknown')

        # Check if this mode is the focus
        is_focus = mode == focus_mode

        # Mode-based layer assignment
        if mode == 'world':
            if element_type in ['skybox', 'sky']:
                return LayerType.BACKGROUND
            elif element_type in ['terrain', 'ground', 'floor']:
                return LayerType.BASE
            elif element_type in ['environment_prop', 'tree', 'building']:
                return LayerType.WORLD
            else:
                return LayerType.FOCUS if is_focus else LayerType.WORLD

        elif mode == 'narrative':
            if element_type in ['backdrop', 'scene_bg']:
                return LayerType.BASE
            elif element_type in ['character', 'actor']:
                return LayerType.CHARACTER
            elif element_type in ['dialogue', 'speech_bubble']:
                return LayerType.OVERLAY
            else:
                return LayerType.FOCUS if is_focus else LayerType.OBJECT

        elif mode == 'architecture':
            if element_type in ['layer', 'tier']:
                return LayerType.BASE
            elif element_type in ['arch_component', 'service', 'database']:
                return LayerType.OBJECT if not is_focus else LayerType.FOCUS
            elif element_type in ['arch_connection', 'connection']:
                return LayerType.OVERLAY
            elif element_type in ['text_label', 'label']:
                return LayerType.HUD
            else:
                return LayerType.OBJECT

        elif mode == 'ui_design':
            if element_type in ['screen_bg', 'app_background']:
                return LayerType.BASE
            elif element_type in ['ui_panel', 'panel', 'form']:
                return LayerType.OBJECT if not is_focus else LayerType.FOCUS
            elif element_type in ['label', 'annotation']:
                return LayerType.OVERLAY
            else:
                return LayerType.OVERLAY

        elif mode == 'data':
            if element_type in ['chart', 'graph', 'bar_3d']:
                return LayerType.FOCUS if is_focus else LayerType.OBJECT
            elif element_type in ['axis', 'grid']:
                return LayerType.BASE
            elif element_type in ['label', 'annotation', 'tooltip']:
                return LayerType.HUD
            else:
                return LayerType.OBJECT

        elif mode == 'object':
            if element_type in ['physical_object', 'product']:
                return LayerType.FOCUS if is_focus else LayerType.OBJECT
            elif element_type in ['pedestal', 'platform']:
                return LayerType.BASE
            else:
                return LayerType.OBJECT

        else:  # abstract
            if element_type in ['concept_sphere']:
                return LayerType.FOCUS if is_focus else LayerType.OBJECT
            else:
                return LayerType.OBJECT

    def _compose_background(
        self,
        scenes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compose background from multiple scenes."""
        # Find the most "environmental" scene for background
        for scene in scenes:
            if scene.get('mode') == 'world':
                return scene.get('background', {'type': 'color', 'color': '#1a1a2e'})

        # Fall back to first scene's background or default
        for scene in scenes:
            if 'background' in scene:
                return scene['background']

        return {'type': 'gradient', 'colors': ['#1a1a2e', '#16213e']}

    def _normalize_lighting(self, lighting: Any) -> dict[str, Any]:
        """Normalize lighting to dict format."""
        if isinstance(lighting, str):
            presets = {
                'neutral': {
                    'ambient': {'intensity': 0.4, 'color': '#ffffff'},
                    'directional': {'intensity': 0.6, 'color': '#ffffff'},
                },
                'dramatic': {
                    'ambient': {'intensity': 0.2, 'color': '#1a1a2e'},
                    'directional': {'intensity': 0.8, 'color': '#ffd700'},
                },
                'soft': {
                    'ambient': {'intensity': 0.5, 'color': '#f0f0f0'},
                    'directional': {'intensity': 0.4, 'color': '#ffffff'},
                },
                'natural': {
                    'ambient': {'intensity': 0.3, 'color': '#87ceeb'},
                    'directional': {'intensity': 0.7, 'color': '#fff8dc'},
                },
            }
            return presets.get(lighting, presets['neutral'])

        return lighting if isinstance(lighting, dict) else {'preset': 'neutral'}

    def _default_camera(self) -> dict[str, Any]:
        """Get default camera configuration."""
        return {
            'position': {'x': 0, 'y': 5, 'z': 15},
            'target': {'x': 0, 'y': 0, 'z': 0},
            'fov': 50,
        }

    def set_layer_config(
        self,
        layer_type: LayerType,
        **kwargs
    ) -> None:
        """Update configuration for a layer."""
        if layer_type in self.layer_configs:
            config = self.layer_configs[layer_type]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    def set_focus(self, scene: ComposedScene, layer: LayerType) -> ComposedScene:
        """Adjust scene to focus on a specific layer."""
        # Increase blur on non-focus layers
        for lt in LayerType:
            config = scene.layer_configs.get(lt)
            if config:
                if lt == layer:
                    config.blur = 0.0
                    config.scale = 1.1
                else:
                    # Blur increases with distance from focus layer
                    distance = abs(lt.value - layer.value) if hasattr(lt, 'value') else 1
                    config.blur = min(distance * 0.5, 2.0)

        return scene


def compose_layered_scene(
    scenes: list[dict[str, Any]],
    focus_mode: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function to compose scenes into layers.

    Args:
        scenes: List of scene dicts from visualizers
        focus_mode: Optional mode to focus on

    Returns:
        Composed scene dict ready for rendering
    """
    composer = LayerComposer()
    composed = composer.compose(scenes, focus_mode)
    return composed.to_dict()


__all__ = [
    'LayerType',
    'LayerConfig',
    'LayeredElement',
    'ComposedScene',
    'LayerComposer',
    'compose_layered_scene',
    'DEFAULT_LAYER_CONFIGS',
]

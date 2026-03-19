"""
Enhanced Scene Composer
=======================

Integrates all visualization systems into a unified composer:
- Conversation Memory: Entity persistence across messages
- Semantic Extraction: 228D semantic space understanding
- Visual Styles: Coherent aesthetic systems
- Layer Composition: Multi-mode scene stacking

This is the production-ready scene composer that creates
rich, contextual, beautiful visualizations.

Philosophy: "Integrate thoughtfully, compose elegantly."

Created: 2025-12-01
Author: HoloLoom Team
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from . import VisualizationMode
from .conversation_memory import (
    EntityType,
    extract_entities_from_text,
    extract_relationships_from_text,
    get_conversation_memory,
)
from .layer_composer import (
    LayerComposer,
)
from .scene_composer import SceneComposer
from .semantic_extractor import (
    SemanticProfile,
    get_semantic_extractor,
)
from .visual_styles import (
    StyleName,
    VisualStyle,
    get_style,
    get_style_for_mode,
    get_style_for_mood,
)


@dataclass
class ComposerConfig:
    """Configuration for the enhanced composer."""
    # Feature flags
    enable_memory: bool = True
    enable_semantic: bool = True
    enable_styles: bool = True
    enable_layers: bool = True

    # Memory settings
    max_entities: int = 100
    entity_decay_hours: float = 24.0

    # Semantic settings
    semantic_threshold: float = 0.3

    # Style settings
    default_style: StyleName | None = None
    auto_style_from_mood: bool = True

    # Layer settings
    enable_focus_blur: bool = True
    layer_separation: float = 1.0


@dataclass
class EnhancedScene:
    """
    A fully enhanced scene with all systems integrated.

    Contains:
    - Base scene from visualizer
    - Persistent entities from memory
    - Semantic styling from analysis
    - Visual style configuration
    - Layer composition data
    """
    mode: str
    title: str
    elements: list[dict[str, Any]]
    camera: dict[str, Any]
    lighting: dict[str, Any]
    background: dict[str, Any]

    # Enhanced data
    style: VisualStyle | None = None
    semantic_profile: SemanticProfile | None = None
    persistent_entities: list[dict[str, Any]] = field(default_factory=list)
    layer_config: dict[str, Any] | None = None
    transition: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'mode': self.mode,
            'title': self.title,
            'elements': self.elements,
            'camera': self.camera,
            'lighting': self.lighting,
            'background': self.background,
            'metadata': self.metadata,
        }

        if self.style:
            result['style'] = {
                'name': self.style.name.value,
                'colors': self.style.colors.as_dict(),
                'animation': {
                    'duration': self.style.animation.duration,
                    'easing': self.style.animation.easing,
                    'entrance': self.style.animation.entrance,
                },
            }

        if self.semantic_profile:
            result['semantic'] = {
                'mood': self.semantic_profile.mood.value,
                'density': self.semantic_profile.density.value,
                'color_temperature': self.semantic_profile.color_temperature,
                'animation_speed': self.semantic_profile.animation_speed,
            }

        if self.persistent_entities:
            result['persistent_entities'] = self.persistent_entities

        if self.layer_config:
            result['layers'] = self.layer_config

        if self.transition:
            result['transition'] = self.transition

        return result


class EnhancedSceneComposer:
    """
    Production-ready scene composer integrating all visualization systems.

    Flow:
    1. Receive text → Extract entities to memory
    2. Analyze semantics → Derive visual parameters
    3. Detect mode → Route to visualizer
    4. Apply style → Consistent aesthetics
    5. Compose layers → Stack multiple modes
    6. Return enhanced scene
    """

    def __init__(
        self,
        config: ComposerConfig | None = None,
        hololoom=None,
    ):
        self.config = config or ComposerConfig()

        # Core components
        self.base_composer = SceneComposer()
        self.memory = get_conversation_memory() if self.config.enable_memory else None
        self.semantic = get_semantic_extractor(hololoom) if self.config.enable_semantic else None
        self.layer_composer = LayerComposer() if self.config.enable_layers else None

        # State
        self._current_style: VisualStyle | None = None
        self._active_layers: list[dict[str, Any]] = []

    async def compose_async(
        self,
        text: str,
        force_mode: str | None = None,
        force_style: str | None = None,
        additional_scenes: list[dict[str, Any]] | None = None,
    ) -> EnhancedScene:
        """
        Compose an enhanced scene asynchronously.

        Args:
            text: The message/content to visualize
            force_mode: Override automatic mode detection
            force_style: Override automatic style selection
            additional_scenes: Additional scenes to layer with main scene

        Returns:
            EnhancedScene with all systems integrated
        """
        # Step 1: Extract and remember entities
        if self.memory and self.config.enable_memory:
            entities = extract_entities_from_text(text, self.memory)
            relationships = extract_relationships_from_text(text, self.memory)

        # Step 2: Analyze semantics
        semantic_profile = None
        if self.semantic and self.config.enable_semantic:
            semantic_profile = await self.semantic.analyze(text)

        # Step 3: Get base scene from mode-specific visualizer
        mode_enum = None
        if force_mode:
            try:
                mode_enum = VisualizationMode(force_mode)
            except ValueError:
                pass

        base_scene = self.base_composer.compose(text, force_mode=mode_enum)
        mode = base_scene.get('mode', 'abstract')

        # Step 4: Select and apply style
        style = self._select_style(mode, semantic_profile, force_style)
        styled_elements = self._apply_style(base_scene.get('elements', []), style)

        # Step 5: Add persistent entities from memory
        persistent = []
        if self.memory and self.config.enable_memory:
            persistent = self._get_persistent_entities(mode)
            styled_elements = self._merge_persistent_entities(styled_elements, persistent)

        # Step 6: Compose layers if multiple scenes
        layer_config = None
        if self.layer_composer and self.config.enable_layers and additional_scenes:
            all_scenes = [base_scene] + additional_scenes
            composed = self.layer_composer.compose(all_scenes, focus_mode=mode)
            layer_config = {
                'layers': {lt.value: len(elems) for lt, elems in composed.layers.items()},
                'focus': mode,
            }
            # Use layered elements
            styled_elements = self._apply_style(composed.get_flat_elements(), style)

        # Step 7: Apply semantic effects
        if semantic_profile:
            styled_elements = self._apply_semantic_effects(styled_elements, semantic_profile)

        # Step 8: Generate enhanced lighting and background
        lighting = self._generate_lighting(style, semantic_profile, base_scene.get('lighting'))
        background = self._generate_background(style, semantic_profile, base_scene.get('background'))

        # Build enhanced scene
        return EnhancedScene(
            mode=mode,
            title=base_scene.get('title', 'Scene'),
            elements=styled_elements,
            camera=base_scene.get('camera', {'position': {'x': 0, 'y': 5, 'z': 15}}),
            lighting=lighting,
            background=background,
            style=style,
            semantic_profile=semantic_profile,
            persistent_entities=[{'name': e['name'], 'id': e['id']} for e in persistent],
            layer_config=layer_config,
            transition=base_scene.get('transition'),
            metadata={
                'memory_entity_count': len(self.memory.entities) if self.memory else 0,
                'semantic_mood': semantic_profile.mood.value if semantic_profile else None,
                'style_name': style.name.value if style else None,
            }
        )

    def compose(
        self,
        text: str,
        force_mode: str | None = None,
        force_style: str | None = None,
    ) -> dict[str, Any]:
        """
        Compose an enhanced scene synchronously.

        For async usage (recommended), use compose_async().
        """
        # Run async version in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new loop for sync call
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.compose_async(text, force_mode, force_style)
                    )
                    scene = future.result()
            else:
                scene = loop.run_until_complete(
                    self.compose_async(text, force_mode, force_style)
                )
        except RuntimeError:
            # No event loop, create one
            scene = asyncio.run(self.compose_async(text, force_mode, force_style))

        return scene.to_dict()

    def _select_style(
        self,
        mode: str,
        profile: SemanticProfile | None,
        force_style: str | None,
    ) -> VisualStyle | None:
        """Select appropriate visual style."""
        if not self.config.enable_styles:
            return None

        # Forced style takes precedence
        if force_style:
            try:
                return get_style(StyleName(force_style))
            except ValueError:
                pass

        # Use configured default
        if self.config.default_style:
            return get_style(self.config.default_style)

        # Auto-select from mood if enabled
        if self.config.auto_style_from_mood and profile:
            return get_style_for_mood(profile.mood.value)

        # Fall back to mode-based style
        return get_style_for_mode(mode)

    def _apply_style(
        self,
        elements: list[dict[str, Any]],
        style: VisualStyle | None,
    ) -> list[dict[str, Any]]:
        """Apply visual style to elements."""
        if not style:
            return elements

        return [style.apply_to_element(elem) for elem in elements]

    def _get_persistent_entities(self, current_mode: str) -> list[dict[str, Any]]:
        """Get persistent entities relevant to current mode."""
        if not self.memory:
            return []

        # Get active entities
        active = self.memory.get_active_entities(min_importance=0.2)

        # Filter by mode relevance
        mode_entity_types = {
            'narrative': [EntityType.CHARACTER, EntityType.LOCATION, EntityType.OBJECT],
            'architecture': [EntityType.SERVICE, EntityType.DATABASE, EntityType.API, EntityType.COMPONENT],
            'ui_design': [EntityType.SCREEN, EntityType.PANEL, EntityType.FORM, EntityType.BUTTON],
            'data': [EntityType.METRIC, EntityType.DATASET],
        }

        relevant_types = mode_entity_types.get(current_mode, list(EntityType))

        return [
            {
                'id': e.id,
                'name': e.name,
                'type': e.entity_type.value,
                'position': e.position,
                'importance': e.importance,
                'attributes': e.attributes,
            }
            for e in active
            if e.entity_type in relevant_types
        ]

    def _merge_persistent_entities(
        self,
        elements: list[dict[str, Any]],
        persistent: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge persistent entities with new elements."""
        # Create lookup of existing elements by name
        existing_names = {
            elem.get('name', '').lower()
            for elem in elements
        }

        # Add persistent entities not already in scene
        result = list(elements)
        for entity in persistent:
            if entity['name'].lower() not in existing_names:
                # Convert entity to scene element
                result.append({
                    'id': entity['id'],
                    'type': f"persistent_{entity['type']}",
                    'name': entity['name'],
                    'label': entity['name'],
                    'position': entity['position'],
                    'persistent': True,
                    'importance': entity['importance'],
                    'opacity': 0.7 + entity['importance'] * 0.3,  # More important = more visible
                })

        return result

    def _apply_semantic_effects(
        self,
        elements: list[dict[str, Any]],
        profile: SemanticProfile,
    ) -> list[dict[str, Any]]:
        """Apply semantic-derived effects to elements."""
        result = []

        for elem in elements:
            enhanced = elem.copy()

            # Add mood-based effects
            if profile.mood.value == 'urgent':
                enhanced.setdefault('effects', []).append('pulse')
            elif profile.mood.value == 'mysterious':
                enhanced.setdefault('effects', []).append('glow')
                enhanced['opacity'] = enhanced.get('opacity', 1.0) * 0.9
            elif profile.mood.value == 'dramatic':
                enhanced.setdefault('effects', []).append('shadow')

            # Add animation speed
            enhanced['animationSpeed'] = profile.animation_speed

            result.append(enhanced)

        return result

    def _generate_lighting(
        self,
        style: VisualStyle | None,
        profile: SemanticProfile | None,
        base_lighting: Any,
    ) -> dict[str, Any]:
        """Generate lighting configuration."""
        if style:
            lighting = {
                'ambient': {
                    'color': style.lighting.ambient_color,
                    'intensity': style.lighting.ambient_intensity,
                },
                'directional': {
                    'color': style.lighting.directional_color,
                    'intensity': style.lighting.directional_intensity,
                    'position': style.lighting.directional_position,
                    'castShadow': style.lighting.shadows,
                },
            }

            if style.lighting.fog_enabled:
                lighting['fog'] = {
                    'color': style.lighting.fog_color,
                    'near': style.lighting.fog_near,
                    'far': style.lighting.fog_far,
                }

            # Adjust for semantic profile
            if profile:
                intensity_mod = profile.lighting_intensity
                lighting['ambient']['intensity'] *= intensity_mod
                lighting['directional']['intensity'] *= intensity_mod

            return lighting

        # Fall back to preset
        if isinstance(base_lighting, str):
            presets = {
                'neutral': {'ambient': {'intensity': 0.4}, 'directional': {'intensity': 0.6}},
                'soft': {'ambient': {'intensity': 0.5}, 'directional': {'intensity': 0.4}},
                'dramatic': {'ambient': {'intensity': 0.2}, 'directional': {'intensity': 0.8}},
                'natural': {'ambient': {'intensity': 0.3}, 'directional': {'intensity': 0.7}},
            }
            return presets.get(base_lighting, presets['neutral'])

        return base_lighting or {'ambient': {'intensity': 0.4}, 'directional': {'intensity': 0.6}}

    def _generate_background(
        self,
        style: VisualStyle | None,
        profile: SemanticProfile | None,
        base_background: Any,
    ) -> dict[str, Any]:
        """Generate background configuration."""
        if style:
            return {
                'type': 'color',
                'color': style.colors.background,
            }

        if isinstance(base_background, dict):
            return base_background

        return {'type': 'gradient', 'colors': ['#1a1a2e', '#16213e']}

    def get_memory_state(self) -> dict[str, Any]:
        """Get current memory state for debugging."""
        if not self.memory:
            return {'enabled': False}

        return {
            'enabled': True,
            'entity_count': len(self.memory.entities),
            'relationship_count': len(self.memory.relationships),
            'timeline_length': len(self.memory.timeline),
            'entities': [
                {'name': e.name, 'type': e.entity_type.value, 'importance': e.importance}
                for e in self.memory.get_active_entities()
            ],
        }

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()


# Singleton instance
_enhanced_composer: EnhancedSceneComposer | None = None


def get_enhanced_composer(
    config: ComposerConfig | None = None,
    hololoom=None,
) -> EnhancedSceneComposer:
    """Get or create enhanced scene composer singleton."""
    global _enhanced_composer
    if _enhanced_composer is None or config is not None:
        _enhanced_composer = EnhancedSceneComposer(config, hololoom)
    return _enhanced_composer


def compose_enhanced_scene(
    text: str,
    force_mode: str | None = None,
    force_style: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function to compose an enhanced scene.

    Args:
        text: The message/content to visualize
        force_mode: Override automatic mode detection
        force_style: Override automatic style selection

    Returns:
        Enhanced scene data ready for rendering
    """
    composer = get_enhanced_composer()
    return composer.compose(text, force_mode, force_style)


__all__ = [
    'ComposerConfig',
    'EnhancedScene',
    'EnhancedSceneComposer',
    'get_enhanced_composer',
    'compose_enhanced_scene',
]

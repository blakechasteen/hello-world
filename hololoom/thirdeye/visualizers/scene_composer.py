"""
Scene Composer
==============

Master orchestrator that:
1. Detects what you're talking about
2. Selects the appropriate visualizer
3. Composes the final 3D scene

This is the brain that decides whether to show UI mockups,
movie scenes, architecture diagrams, or data visualizations.

Created: 2025-12-01
Author: HoloLoom Team
"""

from dataclasses import dataclass, field
from typing import Any

from . import VisualizationMode, detect_visualization_mode
from .architecture_visualizer import extract_architecture_from_text
from .narrative_visualizer import extract_narrative_from_text
from .ui_visualizer import extract_ui_from_text


@dataclass
class ConversationContext:
    """Tracks conversation context for better visualization decisions."""
    recent_messages: list[str] = field(default_factory=list)
    current_mode: VisualizationMode = VisualizationMode.ABSTRACT
    mode_history: list[VisualizationMode] = field(default_factory=list)
    entities_mentioned: dict[str, int] = field(default_factory=dict)
    scene_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_message(self, text: str) -> None:
        """Add a message to context, keep last 10."""
        self.recent_messages.append(text)
        if len(self.recent_messages) > 10:
            self.recent_messages.pop(0)

    def get_full_context(self) -> str:
        """Get concatenated recent context."""
        return ' '.join(self.recent_messages[-5:])


class SceneComposer:
    """
    Master scene composer that orchestrates visualization.

    Flow:
    1. Receive text from conversation
    2. Analyze context + detect visualization mode
    3. Route to appropriate visualizer
    4. Compose final scene with transitions
    5. Return render-ready scene data
    """

    def __init__(self):
        self.context = ConversationContext()
        self._last_scene: dict[str, Any] | None = None

    def compose(self, text: str, force_mode: VisualizationMode | None = None) -> dict[str, Any]:
        """
        Compose a visualization scene from text.

        Args:
            text: The message/content to visualize
            force_mode: Override automatic mode detection

        Returns:
            Scene data ready for rendering
        """
        # Update context
        self.context.add_message(text)

        # Detect or use forced mode
        if force_mode:
            mode = force_mode
        else:
            mode = detect_visualization_mode(self.context.get_full_context())

        # Track mode changes for transitions
        old_mode = self.context.current_mode
        self.context.current_mode = mode
        self.context.mode_history.append(mode)

        # Route to appropriate visualizer
        scene = self._route_to_visualizer(text, mode)

        # If no scene from specific visualizer, create abstract visualization
        if not scene:
            scene = self._create_abstract_scene(text)

        # Add transition if mode changed
        if old_mode != mode and self._last_scene:
            scene = self._add_mode_transition(scene, old_mode, mode)

        self._last_scene = scene

        return scene

    def _route_to_visualizer(self, text: str, mode: VisualizationMode) -> dict[str, Any] | None:
        """Route to the appropriate specialized visualizer."""
        if mode == VisualizationMode.UI_DESIGN:
            return extract_ui_from_text(text)

        elif mode == VisualizationMode.NARRATIVE:
            return extract_narrative_from_text(text)

        elif mode == VisualizationMode.ARCHITECTURE:
            return extract_architecture_from_text(text)

        elif mode == VisualizationMode.DATA:
            return self._extract_data_visualization(text)

        elif mode == VisualizationMode.OBJECT:
            return self._extract_object_visualization(text)

        elif mode == VisualizationMode.WORLD:
            return self._extract_world_visualization(text)

        return None

    def _extract_data_visualization(self, text: str) -> dict[str, Any] | None:
        """Extract data visualization from text."""
        import re

        # Look for numbers and percentages
        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%?', text)
        labels = re.findall(r'([a-z]+)\s*(?::|is|was|at)\s*\d', text.lower())

        if not numbers:
            return None

        # Create a simple bar chart visualization
        data_points = []
        for i, num in enumerate(numbers[:6]):  # Limit to 6 bars
            label = labels[i] if i < len(labels) else f'Item {i+1}'
            data_points.append({
                'label': label.title(),
                'value': float(num)
            })

        return {
            'mode': 'data',
            'title': 'Data Visualization',
            'chart_type': 'bar',
            'elements': [{
                'id': 'chart_0',
                'type': 'chart',
                'chart_type': 'bar_3d',
                'data': data_points,
                'position': {'x': 0, 'y': 0, 'z': 0},
                'size': {'w': 10, 'h': 5, 'd': 5}
            }],
            'camera': {
                'position': {'x': 8, 'y': 6, 'z': 12},
                'target': {'x': 0, 'y': 1, 'z': 0}
            },
            'lighting': 'soft',
            'background': {'type': 'gradient', 'colors': ['#f8fafc', '#e2e8f0']}
        }

    def _extract_object_visualization(self, text: str) -> dict[str, Any] | None:
        """Extract physical object visualization from text."""

        # Common object keywords
        object_patterns = {
            'phone': {'shape': 'rounded_box', 'size': {'w': 0.3, 'h': 0.6, 'd': 0.03}, 'color': '#1e293b'},
            'laptop': {'shape': 'box', 'size': {'w': 1.4, 'h': 0.03, 'd': 0.9}, 'color': '#374151'},
            'car': {'shape': 'car', 'size': {'w': 1.8, 'h': 0.6, 'd': 0.8}, 'color': '#ef4444'},
            'chair': {'shape': 'chair', 'size': {'w': 0.5, 'h': 1.0, 'd': 0.5}, 'color': '#78350f'},
            'table': {'shape': 'table', 'size': {'w': 1.5, 'h': 0.75, 'd': 0.8}, 'color': '#78350f'},
            'book': {'shape': 'box', 'size': {'w': 0.15, 'h': 0.22, 'd': 0.03}, 'color': '#1e40af'},
            'cup': {'shape': 'cylinder', 'size': {'w': 0.08, 'h': 0.12, 'd': 0.08}, 'color': '#ffffff'},
            'bottle': {'shape': 'cylinder', 'size': {'w': 0.07, 'h': 0.25, 'd': 0.07}, 'color': '#22c55e'},
            'box': {'shape': 'box', 'size': {'w': 0.4, 'h': 0.4, 'd': 0.4}, 'color': '#d4a574'},
        }

        text_lower = text.lower()
        elements = []

        for obj_name, obj_props in object_patterns.items():
            if obj_name in text_lower:
                elements.append({
                    'id': f'object_{len(elements)}',
                    'type': 'physical_object',
                    'object_type': obj_name,
                    'label': obj_name.title(),
                    'shape': obj_props['shape'],
                    'size': obj_props['size'],
                    'color': obj_props['color'],
                    'position': {'x': len(elements) * 2 - 2, 'y': 0, 'z': 0}
                })

        if not elements:
            return None

        return {
            'mode': 'object',
            'title': 'Object Visualization',
            'elements': elements,
            'camera': {
                'position': {'x': 0, 'y': 2, 'z': 6},
                'target': {'x': 0, 'y': 0.5, 'z': 0}
            },
            'lighting': 'soft',
            'background': {'type': 'gradient', 'colors': ['#fafafa', '#e5e5e5']}
        }

    def _extract_world_visualization(self, text: str) -> dict[str, Any] | None:
        """Extract environment/world visualization from text."""

        # Environment keywords to terrain types
        terrain_types = {
            'forest': {'ground': '#228b22', 'sky': ['#87ceeb', '#4169e1'], 'props': ['trees', 'bushes']},
            'desert': {'ground': '#deb887', 'sky': ['#87ceeb', '#f0e68c'], 'props': ['dunes', 'cactus']},
            'ocean': {'ground': '#006994', 'sky': ['#87ceeb', '#4169e1'], 'props': ['waves', 'islands']},
            'mountain': {'ground': '#808080', 'sky': ['#b0c4de', '#778899'], 'props': ['peaks', 'snow']},
            'city': {'ground': '#696969', 'sky': ['#708090', '#4a5568'], 'props': ['buildings', 'roads']},
            'space': {'ground': '#0a0a0a', 'sky': ['#0a0a2e', '#1a1a4e'], 'props': ['stars', 'nebula']},
            'room': {'ground': '#8b4513', 'sky': ['#f5f5f5', '#e0e0e0'], 'props': ['walls', 'floor']},
        }

        text_lower = text.lower()
        terrain = None

        for terrain_name, props in terrain_types.items():
            if terrain_name in text_lower:
                terrain = {'name': terrain_name, **props}
                break

        if not terrain:
            return None

        return {
            'mode': 'world',
            'title': f"{terrain['name'].title()} Environment",
            'elements': [
                {
                    'id': 'ground',
                    'type': 'terrain',
                    'terrain_type': terrain['name'],
                    'color': terrain['ground'],
                    'position': {'x': 0, 'y': -0.5, 'z': 0},
                    'size': {'w': 100, 'h': 1, 'd': 100}
                },
                {
                    'id': 'sky',
                    'type': 'skybox',
                    'colors': terrain['sky'],
                }
            ] + [
                {
                    'id': f'prop_{i}',
                    'type': 'environment_prop',
                    'prop_type': prop,
                    'position': {'x': (i - 1) * 8, 'y': 0, 'z': -10}
                }
                for i, prop in enumerate(terrain['props'])
            ],
            'camera': {
                'position': {'x': 0, 'y': 5, 'z': 20},
                'target': {'x': 0, 'y': 0, 'z': 0}
            },
            'lighting': 'natural',
            'background': {
                'type': 'gradient',
                'colors': terrain['sky']
            }
        }

    def _create_abstract_scene(self, text: str) -> dict[str, Any]:
        """Create abstract concept visualization as fallback."""
        import re

        # Extract key terms
        words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        if not words:
            words = re.findall(r'\b([a-z]{4,})\b', text.lower())

        # Deduplicate and limit
        seen = set()
        concepts = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen and len(concepts) < 5:
                seen.add(word_lower)
                concepts.append(word)

        # Create concept spheres
        elements = []
        for i, concept in enumerate(concepts):
            angle = (i / max(len(concepts), 1)) * 6.28  # Distribute in circle
            x = 4 * (i - len(concepts)/2)

            elements.append({
                'id': f'concept_{i}',
                'type': 'concept_sphere',
                'label': concept.title(),
                'position': {'x': x, 'y': 0, 'z': 0},
                'importance': 0.8 - i * 0.1,
                'color': self._get_concept_color(i)
            })

        return {
            'mode': 'abstract',
            'title': 'Concept Map',
            'elements': elements,
            'camera': {
                'position': {'x': 0, 'y': 3, 'z': 12},
                'target': {'x': 0, 'y': 0, 'z': 0}
            },
            'lighting': 'soft',
            'background': {'type': 'gradient', 'colors': ['#f8fafc', '#e2e8f0']}
        }

    def _get_concept_color(self, index: int) -> str:
        """Get color for concept based on index."""
        colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#ec4899']
        return colors[index % len(colors)]

    def _add_mode_transition(self, new_scene: dict[str, Any],
                            old_mode: VisualizationMode,
                            new_mode: VisualizationMode) -> dict[str, Any]:
        """Add smooth transition animation between modes."""
        new_scene['transition'] = {
            'from_mode': old_mode.value,
            'to_mode': new_mode.value,
            'type': 'dissolve_emerge',
            'duration_ms': 800,
            'easing': 'easeInOutCubic'
        }
        return new_scene

    def get_context_summary(self) -> dict[str, Any]:
        """Get summary of current conversation context."""
        return {
            'current_mode': self.context.current_mode.value,
            'mode_history': [m.value for m in self.context.mode_history[-5:]],
            'message_count': len(self.context.recent_messages)
        }


# Singleton instance
_composer: SceneComposer | None = None


def get_composer() -> SceneComposer:
    """Get or create the scene composer singleton."""
    global _composer
    if _composer is None:
        _composer = SceneComposer()
    return _composer


def compose_scene(text: str, force_mode: str | None = None) -> dict[str, Any]:
    """
    Convenience function to compose a scene from text.

    Args:
        text: The message/content to visualize
        force_mode: Optional mode override ('ui_design', 'narrative', 'architecture', etc.)

    Returns:
        Scene data ready for rendering
    """
    composer = get_composer()

    mode = None
    if force_mode:
        try:
            mode = VisualizationMode(force_mode)
        except ValueError:
            pass

    return composer.compose(text, force_mode=mode)


__all__ = [
    'SceneComposer',
    'ConversationContext',
    'get_composer',
    'compose_scene'
]

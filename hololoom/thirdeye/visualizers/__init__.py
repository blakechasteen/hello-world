"""
ThirdEye Visualizers
====================

Context-aware visualization that renders what you're actually talking about.

Visualization Modes:
- UI_DESIGN: Renders actual UI mockups, wireframes, components
- NARRATIVE: Cinematic scenes with characters, environments, dialogue
- ARCHITECTURE: System diagrams, code flow, component relationships
- DATA: Charts, graphs, network visualizations
- OBJECT: Physical objects, 3D representations
- WORLD: Environments, landscapes, spatial concepts
- ABSTRACT: Fallback - concept spheres for abstract ideas

Created: 2025-12-01
Author: HoloLoom Team
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VisualizationMode(Enum):
    """What kind of thing are we visualizing?"""
    UI_DESIGN = "ui_design"        # UI mockups, wireframes, components
    NARRATIVE = "narrative"        # Stories, scenes, dialogue
    ARCHITECTURE = "architecture"  # System diagrams, code structure
    DATA = "data"                  # Charts, graphs, metrics
    OBJECT = "object"              # Physical things, products
    WORLD = "world"                # Environments, spaces, maps
    ABSTRACT = "abstract"          # Fallback - concept spheres


@dataclass
class VisualElement:
    """A single visual element to render."""
    id: str
    element_type: str  # 'panel', 'character', 'box', 'chart', 'object', etc.
    label: str
    position: dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    size: dict[str, float] = field(default_factory=lambda: {"w": 1, "h": 1, "d": 1})
    style: dict[str, Any] = field(default_factory=dict)
    content: str | None = None  # HTML, text, or data
    children: list['VisualElement'] = field(default_factory=list)
    connections: list[str] = field(default_factory=list)  # IDs of connected elements
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualScene:
    """A complete scene to render."""
    mode: VisualizationMode
    title: str
    elements: list[VisualElement]
    camera: dict[str, Any] = field(default_factory=lambda: {
        "position": {"x": 0, "y": 5, "z": 15},
        "target": {"x": 0, "y": 0, "z": 0}
    })
    lighting: str = "soft"  # 'soft', 'dramatic', 'neutral', 'warm', 'cool'
    background: dict[str, Any] = field(default_factory=lambda: {"type": "gradient", "colors": ["#f8fafc", "#e2e8f0"]})
    animation: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Detection keywords for each mode
MODE_KEYWORDS = {
    VisualizationMode.UI_DESIGN: [
        'button', 'form', 'input', 'navbar', 'sidebar', 'header', 'footer',
        'modal', 'dialog', 'card', 'panel', 'layout', 'grid', 'flexbox',
        'component', 'widget', 'ui', 'ux', 'interface', 'design', 'mockup',
        'wireframe', 'prototype', 'screen', 'page', 'menu', 'dropdown',
        'checkbox', 'radio', 'toggle', 'slider', 'tab', 'accordion'
    ],
    VisualizationMode.NARRATIVE: [
        'story', 'scene', 'character', 'plot', 'dialogue', 'narrative',
        'chapter', 'act', 'protagonist', 'antagonist', 'setting', 'conflict',
        'climax', 'resolution', 'movie', 'film', 'screenplay', 'script',
        'she said', 'he said', 'then', 'suddenly', 'meanwhile', 'finally',
        'once upon', 'the end', 'adventure', 'journey', 'hero', 'villain'
    ],
    VisualizationMode.ARCHITECTURE: [
        'system', 'architecture', 'component', 'module', 'service', 'api',
        'database', 'server', 'client', 'microservice', 'pipeline', 'flow',
        'diagram', 'class', 'function', 'method', 'interface', 'protocol',
        'layer', 'tier', 'backend', 'frontend', 'middleware', 'gateway',
        'queue', 'cache', 'storage', 'network', 'endpoint', 'route'
    ],
    VisualizationMode.DATA: [
        'chart', 'graph', 'metric', 'data', 'analytics', 'statistics',
        'percentage', 'trend', 'growth', 'decline', 'comparison', 'breakdown',
        'distribution', 'histogram', 'pie', 'bar', 'line', 'scatter',
        'visualization', 'dashboard', 'kpi', 'report', 'table', 'spreadsheet'
    ],
    VisualizationMode.OBJECT: [
        'product', 'device', 'machine', 'tool', 'furniture', 'vehicle',
        'building', 'structure', 'equipment', 'appliance', 'gadget',
        'physical', 'tangible', 'real-world', 'object', 'thing', 'item'
    ],
    VisualizationMode.WORLD: [
        'environment', 'landscape', 'world', 'map', 'terrain', 'space',
        'room', 'building', 'city', 'forest', 'ocean', 'mountain',
        'indoor', 'outdoor', 'virtual', 'metaverse', '3d space', 'location'
    ]
}


def detect_visualization_mode(text: str) -> VisualizationMode:
    """Detect the best visualization mode based on text content."""
    text_lower = text.lower()

    # Count keyword matches for each mode
    scores = {}
    for mode, keywords in MODE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[mode] = score

    # Get mode with highest score
    best_mode = max(scores, key=scores.get)

    # If no strong signal, default to abstract
    if scores[best_mode] < 2:
        return VisualizationMode.ABSTRACT

    return best_mode


__all__ = [
    'VisualizationMode',
    'VisualElement',
    'VisualScene',
    'detect_visualization_mode',
    'MODE_KEYWORDS'
]

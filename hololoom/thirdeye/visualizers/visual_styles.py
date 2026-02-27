"""
Visual Styles
=============

Coherent visual languages for different contexts.

Each style is a complete aesthetic system:
- Color palette
- Material properties
- Lighting setup
- Typography
- Animation characteristics
- Spatial layout rules

Philosophy: "Style is the consistent application of aesthetic decisions."

Created: 2025-12-01
Author: HoloLoom Team
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class StyleName(Enum):
    """Available visual styles."""
    BLUEPRINT = "blueprint"        # Technical, architectural
    STORYBOARD = "storyboard"      # Narrative, hand-drawn feel
    DASHBOARD = "dashboard"        # Data, minimal, Tufte-inspired
    SHOWROOM = "showroom"          # Products, studio lighting
    CINEMATIC = "cinematic"        # Dramatic, depth of field
    SKETCH = "sketch"              # Rough, ideation
    NEON = "neon"                  # Cyberpunk, glowing
    PAPER = "paper"                # Minimal, document-like


@dataclass
class ColorPalette:
    """A coherent color palette for a style."""
    primary: str           # Main UI/element color
    secondary: str         # Supporting color
    accent: str           # Highlight color
    background: str       # Scene background
    surface: str          # Panel/card surfaces
    text: str             # Text/labels
    muted: str            # Less important elements
    success: str = "#22c55e"
    warning: str = "#f59e0b"
    error: str = "#ef4444"

    def as_dict(self) -> Dict[str, str]:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "accent": self.accent,
            "background": self.background,
            "surface": self.surface,
            "text": self.text,
            "muted": self.muted,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
        }


@dataclass
class MaterialProperties:
    """Material properties for 3D elements."""
    roughness: float = 0.5         # 0=shiny, 1=matte
    metalness: float = 0.0         # 0=plastic, 1=metal
    opacity: float = 1.0
    emissive_intensity: float = 0.0
    wireframe: bool = False
    flat_shading: bool = False
    side: str = "front"  # front, back, double


@dataclass
class LightingSetup:
    """Lighting configuration for a style."""
    ambient_color: str = "#ffffff"
    ambient_intensity: float = 0.4
    directional_color: str = "#ffffff"
    directional_intensity: float = 0.6
    directional_position: Dict[str, float] = field(
        default_factory=lambda: {"x": 5, "y": 10, "z": 5}
    )
    shadows: bool = False
    fog_enabled: bool = False
    fog_color: str = "#ffffff"
    fog_near: float = 10
    fog_far: float = 50


@dataclass
class TypographyStyle:
    """Typography settings for labels and text."""
    font_family: str = "Inter, system-ui, sans-serif"
    font_weight: int = 400
    letter_spacing: float = 0
    text_transform: str = "none"  # none, uppercase, lowercase
    label_size: float = 14
    title_size: float = 24
    caption_size: float = 11
    line_height: float = 1.4


@dataclass
class AnimationStyle:
    """Animation characteristics for a style."""
    duration: float = 0.6          # Default transition duration
    easing: str = "easeOutCubic"
    stagger: float = 0.05          # Delay between elements
    entrance: str = "fadeIn"       # fadeIn, slideUp, scaleIn, none
    hover_effect: str = "lift"     # lift, glow, none
    idle_animation: str = "none"   # float, pulse, rotate, none


@dataclass
class LayoutRules:
    """Spatial layout rules for a style."""
    grid_size: float = 1.0         # Base grid unit
    spacing: float = 2.0           # Space between elements
    margin: float = 1.0            # Edge margin
    max_elements_per_row: int = 5
    vertical_layers: bool = True   # Stack in Y axis
    depth_layers: bool = False     # Stack in Z axis
    alignment: str = "center"      # left, center, right


@dataclass
class VisualStyle:
    """Complete visual style definition."""
    name: StyleName
    display_name: str
    description: str

    colors: ColorPalette
    materials: MaterialProperties
    lighting: LightingSetup
    typography: TypographyStyle
    animation: AnimationStyle
    layout: LayoutRules

    # Style-specific element overrides
    element_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def apply_to_element(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this style to an element."""
        styled = element.copy()

        # Apply base material properties
        styled['material'] = {
            'roughness': self.materials.roughness,
            'metalness': self.materials.metalness,
            'opacity': self.materials.opacity,
            'emissiveIntensity': self.materials.emissive_intensity,
            'wireframe': self.materials.wireframe,
            'flatShading': self.materials.flat_shading,
        }

        # Apply color from palette if not specified
        if 'color' not in styled:
            element_type = styled.get('type', 'default')
            if element_type in ['service', 'component', 'panel']:
                styled['color'] = self.colors.primary
            elif element_type in ['database', 'storage']:
                styled['color'] = self.colors.secondary
            elif element_type in ['connection', 'edge']:
                styled['color'] = self.colors.muted
            else:
                styled['color'] = self.colors.surface

        # Apply element-specific overrides
        element_type = styled.get('type', 'default')
        if element_type in self.element_overrides:
            styled.update(self.element_overrides[element_type])

        # Add animation settings
        styled['animation'] = {
            'entrance': self.animation.entrance,
            'duration': self.animation.duration,
            'easing': self.animation.easing,
            'idle': self.animation.idle_animation,
        }

        return styled

    def get_scene_config(self) -> Dict[str, Any]:
        """Get scene-wide configuration for this style."""
        return {
            'background': {
                'color': self.colors.background,
            },
            'lighting': {
                'ambient': {
                    'color': self.lighting.ambient_color,
                    'intensity': self.lighting.ambient_intensity,
                },
                'directional': {
                    'color': self.lighting.directional_color,
                    'intensity': self.lighting.directional_intensity,
                    'position': self.lighting.directional_position,
                    'castShadow': self.lighting.shadows,
                },
            },
            'fog': {
                'enabled': self.lighting.fog_enabled,
                'color': self.lighting.fog_color,
                'near': self.lighting.fog_near,
                'far': self.lighting.fog_far,
            } if self.lighting.fog_enabled else None,
            'layout': {
                'gridSize': self.layout.grid_size,
                'spacing': self.layout.spacing,
                'margin': self.layout.margin,
            },
        }


# =============================================================================
# Style Definitions
# =============================================================================

BLUEPRINT_STYLE = VisualStyle(
    name=StyleName.BLUEPRINT,
    display_name="Blueprint",
    description="Technical, architectural style with blue lines on dark background",

    colors=ColorPalette(
        primary="#3b82f6",      # Blue lines
        secondary="#60a5fa",    # Light blue
        accent="#fbbf24",       # Yellow highlights
        background="#0f172a",   # Dark navy
        surface="#1e293b",      # Dark surface
        text="#e2e8f0",         # Light text
        muted="#64748b",        # Gray
    ),

    materials=MaterialProperties(
        roughness=1.0,
        metalness=0.0,
        wireframe=True,         # Wireframe for blueprint look
        flat_shading=True,
    ),

    lighting=LightingSetup(
        ambient_color="#3b82f6",
        ambient_intensity=0.3,
        directional_intensity=0.4,
    ),

    typography=TypographyStyle(
        font_family="'JetBrains Mono', 'Fira Code', monospace",
        font_weight=400,
        letter_spacing=0.5,
        text_transform="uppercase",
    ),

    animation=AnimationStyle(
        duration=0.4,
        easing="easeOutQuart",
        entrance="fadeIn",
        hover_effect="glow",
        idle_animation="none",
    ),

    layout=LayoutRules(
        grid_size=1.0,
        spacing=3.0,
        vertical_layers=True,
    ),

    element_overrides={
        'connection': {'lineWidth': 1, 'dashed': True, 'dashSize': 0.2},
        'label': {'background': 'transparent', 'borderWidth': 1},
    }
)


STORYBOARD_STYLE = VisualStyle(
    name=StyleName.STORYBOARD,
    display_name="Storyboard",
    description="Narrative style with warm tones and sketch-like quality",

    colors=ColorPalette(
        primary="#78350f",      # Warm brown
        secondary="#a16207",    # Golden
        accent="#dc2626",       # Red accent
        background="#fef3c7",   # Cream paper
        surface="#fef9c3",      # Light yellow
        text="#451a03",         # Dark brown
        muted="#92400e",        # Medium brown
    ),

    materials=MaterialProperties(
        roughness=0.9,
        metalness=0.0,
        flat_shading=True,
    ),

    lighting=LightingSetup(
        ambient_color="#fef3c7",
        ambient_intensity=0.6,
        directional_color="#fbbf24",
        directional_intensity=0.4,
    ),

    typography=TypographyStyle(
        font_family="'Caveat', 'Patrick Hand', cursive",
        font_weight=500,
        letter_spacing=0,
    ),

    animation=AnimationStyle(
        duration=0.8,
        easing="easeInOutQuad",
        entrance="slideUp",
        hover_effect="lift",
        idle_animation="none",
    ),

    layout=LayoutRules(
        grid_size=1.5,
        spacing=2.5,
        vertical_layers=False,
        depth_layers=True,  # Scenes in depth
    ),

    element_overrides={
        'character': {'outline': True, 'outlineWidth': 2, 'outlineColor': '#78350f'},
        'backdrop': {'opacity': 0.8},
    }
)


DASHBOARD_STYLE = VisualStyle(
    name=StyleName.DASHBOARD,
    display_name="Dashboard",
    description="Clean, minimal, data-focused (Tufte-inspired)",

    colors=ColorPalette(
        primary="#1f2937",      # Dark gray
        secondary="#374151",    # Medium gray
        accent="#3b82f6",       # Blue accent
        background="#ffffff",   # White
        surface="#f9fafb",      # Light gray
        text="#111827",         # Near black
        muted="#9ca3af",        # Light gray
    ),

    materials=MaterialProperties(
        roughness=1.0,
        metalness=0.0,
        opacity=1.0,
    ),

    lighting=LightingSetup(
        ambient_color="#ffffff",
        ambient_intensity=0.8,
        directional_intensity=0.3,
    ),

    typography=TypographyStyle(
        font_family="'Inter', -apple-system, system-ui, sans-serif",
        font_weight=400,
        letter_spacing=-0.2,
        label_size=12,
    ),

    animation=AnimationStyle(
        duration=0.3,
        easing="easeOutQuart",
        entrance="fadeIn",
        stagger=0.02,
        hover_effect="none",
        idle_animation="none",
    ),

    layout=LayoutRules(
        grid_size=0.5,
        spacing=1.5,
        margin=0.5,
        alignment="left",
    ),

    element_overrides={
        'chart': {'strokeWidth': 1.5, 'fillOpacity': 0.1},
        'axis': {'color': '#e5e7eb', 'tickLength': 4},
        'label': {'background': 'transparent'},
    }
)


SHOWROOM_STYLE = VisualStyle(
    name=StyleName.SHOWROOM,
    display_name="Showroom",
    description="Product display with studio lighting and reflections",

    colors=ColorPalette(
        primary="#18181b",      # Near black
        secondary="#3f3f46",    # Dark gray
        accent="#f59e0b",       # Warm accent
        background="#18181b",   # Dark
        surface="#27272a",      # Dark surface
        text="#fafafa",         # White text
        muted="#71717a",        # Gray
    ),

    materials=MaterialProperties(
        roughness=0.3,
        metalness=0.8,
        opacity=1.0,
    ),

    lighting=LightingSetup(
        ambient_color="#ffffff",
        ambient_intensity=0.2,
        directional_color="#ffffff",
        directional_intensity=0.9,
        shadows=True,
    ),

    typography=TypographyStyle(
        font_family="'Helvetica Neue', Helvetica, Arial, sans-serif",
        font_weight=300,
        letter_spacing=2,
        text_transform="uppercase",
    ),

    animation=AnimationStyle(
        duration=0.8,
        easing="easeInOutCubic",
        entrance="scaleIn",
        hover_effect="lift",
        idle_animation="rotate",
    ),

    layout=LayoutRules(
        grid_size=2.0,
        spacing=4.0,
        max_elements_per_row=3,
    ),

    element_overrides={
        'product': {'castShadow': True, 'receiveShadow': True},
        'pedestal': {'color': '#27272a', 'height': 0.1},
    }
)


CINEMATIC_STYLE = VisualStyle(
    name=StyleName.CINEMATIC,
    display_name="Cinematic",
    description="Dramatic, film-like with depth of field and atmosphere",

    colors=ColorPalette(
        primary="#1e293b",      # Blue-gray
        secondary="#334155",    # Lighter blue-gray
        accent="#eab308",       # Golden
        background="#0c0a09",   # Near black
        surface="#1c1917",      # Dark surface
        text="#fafaf9",         # Warm white
        muted="#57534e",        # Brown-gray
    ),

    materials=MaterialProperties(
        roughness=0.6,
        metalness=0.2,
    ),

    lighting=LightingSetup(
        ambient_color="#1e293b",
        ambient_intensity=0.15,
        directional_color="#fef3c7",
        directional_intensity=0.8,
        shadows=True,
        fog_enabled=True,
        fog_color="#1e293b",
        fog_near=15,
        fog_far=40,
    ),

    typography=TypographyStyle(
        font_family="'Playfair Display', Georgia, serif",
        font_weight=400,
        letter_spacing=0.5,
    ),

    animation=AnimationStyle(
        duration=1.2,
        easing="easeInOutQuint",
        entrance="fadeIn",
        stagger=0.1,
        hover_effect="glow",
        idle_animation="float",
    ),

    layout=LayoutRules(
        grid_size=1.5,
        spacing=3.0,
        depth_layers=True,
    ),

    element_overrides={
        'character': {'rimLight': True, 'rimColor': '#fbbf24'},
        'backdrop': {'depthWrite': False, 'fog': True},
    }
)


NEON_STYLE = VisualStyle(
    name=StyleName.NEON,
    display_name="Neon",
    description="Cyberpunk aesthetic with glowing elements",

    colors=ColorPalette(
        primary="#06b6d4",      # Cyan
        secondary="#8b5cf6",    # Purple
        accent="#f43f5e",       # Pink
        background="#0a0a0a",   # Black
        surface="#18181b",      # Dark
        text="#fafafa",         # White
        muted="#52525b",        # Gray
    ),

    materials=MaterialProperties(
        roughness=0.2,
        metalness=0.5,
        emissive_intensity=0.8,
    ),

    lighting=LightingSetup(
        ambient_color="#06b6d4",
        ambient_intensity=0.2,
        directional_color="#8b5cf6",
        directional_intensity=0.4,
    ),

    typography=TypographyStyle(
        font_family="'Orbitron', 'Audiowide', sans-serif",
        font_weight=700,
        letter_spacing=2,
        text_transform="uppercase",
    ),

    animation=AnimationStyle(
        duration=0.5,
        easing="easeOutBack",
        entrance="scaleIn",
        hover_effect="glow",
        idle_animation="pulse",
    ),

    layout=LayoutRules(
        grid_size=1.0,
        spacing=2.0,
    ),

    element_overrides={
        'glow': {'bloom': True, 'bloomStrength': 0.8},
        'connection': {'color': '#06b6d4', 'glow': True, 'animated': True},
    }
)


# =============================================================================
# Style Registry and Selection
# =============================================================================

STYLES: Dict[StyleName, VisualStyle] = {
    StyleName.BLUEPRINT: BLUEPRINT_STYLE,
    StyleName.STORYBOARD: STORYBOARD_STYLE,
    StyleName.DASHBOARD: DASHBOARD_STYLE,
    StyleName.SHOWROOM: SHOWROOM_STYLE,
    StyleName.CINEMATIC: CINEMATIC_STYLE,
    StyleName.NEON: NEON_STYLE,
}


def get_style(name: StyleName) -> VisualStyle:
    """Get a visual style by name."""
    return STYLES.get(name, DASHBOARD_STYLE)


def get_style_for_mode(visualization_mode: str) -> VisualStyle:
    """Get recommended style for a visualization mode."""
    mode_to_style = {
        'ui_design': StyleName.DASHBOARD,
        'narrative': StyleName.STORYBOARD,
        'architecture': StyleName.BLUEPRINT,
        'data': StyleName.DASHBOARD,
        'object': StyleName.SHOWROOM,
        'world': StyleName.CINEMATIC,
        'abstract': StyleName.NEON,
    }

    style_name = mode_to_style.get(visualization_mode, StyleName.DASHBOARD)
    return STYLES[style_name]


def get_style_for_mood(mood: str) -> VisualStyle:
    """Get style based on semantic mood."""
    mood_to_style = {
        'neutral': StyleName.DASHBOARD,
        'dramatic': StyleName.CINEMATIC,
        'calm': StyleName.STORYBOARD,
        'urgent': StyleName.NEON,
        'technical': StyleName.BLUEPRINT,
        'playful': StyleName.STORYBOARD,
        'mysterious': StyleName.CINEMATIC,
    }

    style_name = mood_to_style.get(mood, StyleName.DASHBOARD)
    return STYLES[style_name]


def list_styles() -> List[Dict[str, str]]:
    """List all available styles."""
    return [
        {
            'name': style.name.value,
            'display_name': style.display_name,
            'description': style.description,
        }
        for style in STYLES.values()
    ]


__all__ = [
    'StyleName',
    'ColorPalette',
    'MaterialProperties',
    'LightingSetup',
    'TypographyStyle',
    'AnimationStyle',
    'LayoutRules',
    'VisualStyle',
    'STYLES',
    'get_style',
    'get_style_for_mode',
    'get_style_for_mood',
    'list_styles',
    # Individual styles
    'BLUEPRINT_STYLE',
    'STORYBOARD_STYLE',
    'DASHBOARD_STYLE',
    'SHOWROOM_STYLE',
    'CINEMATIC_STYLE',
    'NEON_STYLE',
]

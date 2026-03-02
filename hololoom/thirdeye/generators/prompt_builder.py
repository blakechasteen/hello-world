"""
Scene to Prompt Builder
=======================

Converts ThirdEye scene descriptions into optimized prompts
for image generation models (Stable Diffusion, Flux, etc).

Philosophy: "The scene knows what it wants to be."

Created: 2025-12-01
Author: HoloLoom Team
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PromptComponents:
    """Structured prompt components for image generation."""
    subject: str = ""
    setting: str = ""
    style: str = ""
    mood: str = ""
    lighting: str = ""
    composition: str = ""
    details: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)

    def to_positive_prompt(self) -> str:
        """Build the positive prompt string."""
        parts = []

        if self.subject:
            parts.append(self.subject)
        if self.setting:
            parts.append(self.setting)
        if self.mood:
            parts.append(f"{self.mood} atmosphere")
        if self.lighting:
            parts.append(f"{self.lighting} lighting")
        if self.style:
            parts.append(self.style)
        if self.composition:
            parts.append(self.composition)
        if self.details:
            parts.extend(self.details)

        return ", ".join(parts)

    def to_negative_prompt(self) -> str:
        """Build the negative prompt string."""
        defaults = [
            "blurry", "low quality", "distorted", "deformed",
            "ugly", "bad anatomy", "watermark", "text"
        ]
        return ", ".join(defaults + self.negative)


class SceneToPrompt:
    """
    Converts ThirdEye scene JSON to image generation prompts.

    Understands:
    - Visualization modes (narrative, ui_design, architecture, etc.)
    - Visual styles (cinematic, blueprint, neon, etc.)
    - Semantic profiles (mood, density, color temperature)
    - Scene elements (characters, objects, environments)
    """

    # Mode-specific prompt templates
    MODE_STYLES = {
        'narrative': {
            'base_style': "cinematic scene, dramatic composition, storytelling",
            'quality': "detailed, high quality, professional illustration",
        },
        'ui_design': {
            'base_style': "UI design mockup, clean interface, modern app design",
            'quality': "crisp, vector-like, professional UI/UX",
        },
        'architecture': {
            'base_style': "technical diagram, system architecture, flowchart",
            'quality': "clean lines, professional, technical illustration",
        },
        'data': {
            'base_style': "data visualization, infographic, chart design",
            'quality': "clean, minimalist, information design",
        },
        'world': {
            'base_style': "environment concept art, landscape, world building",
            'quality': "detailed environment, atmospheric, concept art",
        },
        'object': {
            'base_style': "product visualization, 3D render, object study",
            'quality': "detailed, studio lighting, product photography style",
        },
        'abstract': {
            'base_style': "abstract visualization, concept art, symbolic",
            'quality': "artistic, evocative, thoughtful composition",
        },
    }

    # Visual style to prompt mapping
    STYLE_PROMPTS = {
        'blueprint': "technical blueprint style, blue and white, schematic, engineering drawing",
        'storyboard': "storyboard art, pencil sketch, sequential frames, cinematic",
        'dashboard': "data dashboard, dark theme, glowing metrics, modern UI",
        'showroom': "product showroom, studio lighting, clean background, presentation",
        'cinematic': "cinematic composition, dramatic lighting, film still, anamorphic",
        'neon': "neon cyberpunk, glowing lights, dark background, synthwave colors",
    }

    # Mood to atmosphere mapping
    MOOD_PROMPTS = {
        'neutral': "balanced, clear",
        'urgent': "intense, dynamic, high energy, dramatic tension",
        'calm': "serene, peaceful, soft, tranquil",
        'mysterious': "enigmatic, shadowy, atmospheric, intriguing",
        'dramatic': "bold, striking, powerful, cinematic drama",
        'playful': "whimsical, fun, colorful, lighthearted",
        'technical': "precise, analytical, structured, professional",
    }

    # Color temperature to lighting
    TEMPERATURE_LIGHTING = {
        'warm': "warm golden lighting, sunset tones, amber glow",
        'cool': "cool blue lighting, moonlight, crisp shadows",
        'neutral': "balanced natural lighting, soft white",
    }

    def __init__(self):
        self.element_extractors = {
            'character': self._extract_character,
            'persistent_character': self._extract_character,
            'screen': self._extract_ui_element,
            'ui_panel': self._extract_ui_element,
            'persistent_screen': self._extract_ui_element,
            'arch_component': self._extract_architecture,
            'service': self._extract_architecture,
            'database': self._extract_architecture,
            'location': self._extract_location,
            'persistent_location': self._extract_location,
            'concept_sphere': self._extract_concept,
            'terrain': self._extract_environment,
            'skybox': self._extract_environment,
        }

    def convert(self, scene: Dict[str, Any]) -> PromptComponents:
        """
        Convert a ThirdEye scene to prompt components.

        Args:
            scene: Scene dictionary from enhanced composer

        Returns:
            PromptComponents ready for image generation
        """
        components = PromptComponents()

        mode = scene.get('mode', 'abstract')

        # Get base style for mode
        mode_config = self.MODE_STYLES.get(mode, self.MODE_STYLES['abstract'])
        components.style = mode_config['base_style']
        components.details.append(mode_config['quality'])

        # Apply visual style if present
        if 'style' in scene:
            style_name = scene['style'].get('name', '')
            if style_name in self.STYLE_PROMPTS:
                components.details.append(self.STYLE_PROMPTS[style_name])

        # Apply semantic profile
        if 'semantic' in scene:
            semantic = scene['semantic']

            # Mood
            mood = semantic.get('mood', 'neutral')
            components.mood = self.MOOD_PROMPTS.get(mood, mood)

            # Color temperature -> lighting
            temp = semantic.get('color_temperature', 'neutral')
            components.lighting = self.TEMPERATURE_LIGHTING.get(temp, temp)

            # Density affects detail level
            density = semantic.get('density', 'balanced')
            if density == 'detailed':
                components.details.append("highly detailed, intricate")
            elif density == 'sparse':
                components.details.append("minimalist, clean, simple")

        # Extract subject from elements
        elements = scene.get('elements', [])
        subjects = []
        settings = []

        for elem in elements:
            elem_type = elem.get('type', '')

            # Try specific extractor
            for type_prefix, extractor in self.element_extractors.items():
                if type_prefix in elem_type:
                    subj, sett = extractor(elem)
                    if subj:
                        subjects.append(subj)
                    if sett:
                        settings.append(sett)
                    break

        # Build subject and setting
        if subjects:
            # Prioritize: take first 3 most important subjects
            components.subject = ", ".join(subjects[:3])
        else:
            components.subject = self._subject_from_title(scene.get('title', ''))

        if settings:
            components.setting = ", ".join(set(settings[:2]))

        # Add composition hints from camera
        if 'camera' in scene:
            cam = scene['camera']
            pos = cam.get('position', {})
            y = pos.get('y', 0)
            z = pos.get('z', 10)

            if y > 5:
                components.composition = "aerial view, bird's eye perspective"
            elif y < 1:
                components.composition = "low angle, dramatic perspective"
            elif z > 15:
                components.composition = "wide shot, establishing shot"
            elif z < 8:
                components.composition = "close-up, detailed view"
            else:
                components.composition = "medium shot, balanced composition"

        # Add mode-specific negative prompts
        self._add_mode_negatives(mode, components)

        return components

    def _extract_character(self, elem: Dict[str, Any]) -> Tuple[str, str]:
        """Extract character description."""
        name = elem.get('label', elem.get('name', ''))
        return f"character: {name}", ""

    def _extract_ui_element(self, elem: Dict[str, Any]) -> Tuple[str, str]:
        """Extract UI element description."""
        name = elem.get('label', elem.get('name', ''))
        return f"{name} interface", "digital screen"

    def _extract_architecture(self, elem: Dict[str, Any]) -> Tuple[str, str]:
        """Extract architecture component."""
        name = elem.get('label', elem.get('name', ''))
        elem_type = elem.get('type', '')
        if 'database' in elem_type:
            return f"database: {name}", "technical diagram"
        return f"service: {name}", "system architecture"

    def _extract_location(self, elem: Dict[str, Any]) -> Tuple[str, str]:
        """Extract location description."""
        name = elem.get('label', elem.get('name', ''))
        return "", f"location: {name}"

    def _extract_concept(self, elem: Dict[str, Any]) -> Tuple[str, str]:
        """Extract concept sphere."""
        label = elem.get('label', '')
        return f"concept: {label}", ""

    def _extract_environment(self, elem: Dict[str, Any]) -> Tuple[str, str]:
        """Extract environment element."""
        terrain = elem.get('terrain_type', elem.get('type', ''))
        return "", f"{terrain} environment"

    def _subject_from_title(self, title: str) -> str:
        """Derive subject from scene title."""
        if not title:
            return "abstract visualization"
        return title.lower()

    def _add_mode_negatives(self, mode: str, components: PromptComponents):
        """Add mode-specific negative prompts."""
        if mode == 'ui_design':
            components.negative.extend([
                "3D", "realistic photo", "natural scene"
            ])
        elif mode == 'architecture':
            components.negative.extend([
                "realistic", "photo", "natural", "organic"
            ])
        elif mode == 'narrative':
            components.negative.extend([
                "diagram", "UI", "interface", "technical"
            ])


def build_prompt_from_scene(
    scene: Dict[str, Any],
    additional_style: Optional[str] = None,
    additional_negative: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Convenience function to build prompts from scene.

    Args:
        scene: ThirdEye scene dictionary
        additional_style: Extra style keywords to add
        additional_negative: Extra negative prompts

    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    builder = SceneToPrompt()
    components = builder.convert(scene)

    if additional_style:
        components.details.append(additional_style)

    if additional_negative:
        components.negative.extend(additional_negative)

    return components.to_positive_prompt(), components.to_negative_prompt()


__all__ = ['SceneToPrompt', 'PromptComponents', 'build_prompt_from_scene']

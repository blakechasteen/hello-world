"""
Narrative Visualizer
====================

Renders stories, scenes, and dialogue as cinematic visualizations.
Creates character silhouettes, environments, and dramatic camera movements.

When discussing narratives, this creates visual story representations
with characters, settings, and action.

Created: 2025-12-01
Author: HoloLoom Team
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CharacterMood(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    THOUGHTFUL = "thoughtful"
    DETERMINED = "determined"
    FEARFUL = "fearful"


class SceneType(Enum):
    DIALOGUE = "dialogue"       # Characters talking
    ACTION = "action"          # Movement, events
    DESCRIPTION = "description" # Setting, atmosphere
    TRANSITION = "transition"   # Scene change


@dataclass
class Character:
    """A character in the narrative."""
    id: str
    name: str
    role: str = "character"  # 'protagonist', 'antagonist', 'supporting', 'background'
    mood: CharacterMood = CharacterMood.NEUTRAL
    position: dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    color: str = "#3b82f6"  # Silhouette color
    speaking: bool = False
    current_line: str | None = None


@dataclass
class Environment:
    """The setting/environment of a scene."""
    name: str
    type: str  # 'indoor', 'outdoor', 'abstract'
    time_of_day: str = "day"  # 'dawn', 'day', 'dusk', 'night'
    weather: str = "clear"  # 'clear', 'cloudy', 'rain', 'storm', 'snow', 'fog'
    mood: str = "neutral"  # 'peaceful', 'tense', 'mysterious', 'dramatic', 'romantic'
    colors: list[str] = field(default_factory=lambda: ["#87ceeb", "#4a90d9"])  # Sky gradient
    props: list[str] = field(default_factory=list)  # Objects in scene


@dataclass
class NarrativeScene:
    """A single scene in the narrative."""
    id: str
    scene_type: SceneType
    characters: list[Character]
    environment: Environment | None
    dialogue: list[dict[str, str]] = field(default_factory=list)  # [{"character": "name", "line": "text"}]
    action: str | None = None
    narration: str | None = None
    camera_angle: str = "medium"  # 'wide', 'medium', 'close', 'extreme_close'
    camera_movement: str | None = None  # 'pan_left', 'pan_right', 'dolly_in', 'dolly_out', 'track'


# Environment presets
ENVIRONMENT_PRESETS = {
    'office': Environment(
        name="Office",
        type="indoor",
        mood="neutral",
        colors=["#f5f5f5", "#e0e0e0"],
        props=["desk", "chair", "computer", "window"]
    ),
    'forest': Environment(
        name="Forest",
        type="outdoor",
        mood="mysterious",
        colors=["#228b22", "#006400"],
        props=["trees", "path", "bushes", "rocks"]
    ),
    'city': Environment(
        name="City Street",
        type="outdoor",
        mood="busy",
        colors=["#708090", "#4a5568"],
        props=["buildings", "streetlight", "cars", "sidewalk"]
    ),
    'home': Environment(
        name="Living Room",
        type="indoor",
        mood="cozy",
        colors=["#f5deb3", "#deb887"],
        props=["sofa", "lamp", "bookshelf", "window"]
    ),
    'space': Environment(
        name="Space",
        type="abstract",
        mood="vast",
        colors=["#0a0a2e", "#1a1a4e"],
        props=["stars", "nebula", "planets"]
    ),
    'beach': Environment(
        name="Beach",
        type="outdoor",
        time_of_day="day",
        weather="clear",
        mood="peaceful",
        colors=["#87ceeb", "#00bfff"],
        props=["sand", "waves", "palm_trees", "sun"]
    ),
    'mountains': Environment(
        name="Mountains",
        type="outdoor",
        mood="majestic",
        colors=["#b0c4de", "#778899"],
        props=["peaks", "snow", "clouds", "valley"]
    )
}

# Character role colors
ROLE_COLORS = {
    'protagonist': '#3b82f6',   # Blue
    'antagonist': '#ef4444',    # Red
    'supporting': '#10b981',    # Green
    'mentor': '#8b5cf6',        # Purple
    'love_interest': '#ec4899', # Pink
    'comic_relief': '#f59e0b',  # Amber
    'background': '#6b7280'     # Gray
}


class NarrativeExtractor:
    """Extract narrative elements from text."""

    # Patterns to detect characters
    CHARACTER_PATTERNS = [
        r'(?:^|\s)([A-Z][a-z]+)(?:\s+said|\s+asked|\s+replied|\s+whispered|\s+shouted)',
        r'"[^"]+"\s+said\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+looked\s+at',
        r'([A-Z][a-z]+)\s+turned\s+to',
        r'([A-Z][a-z]+)\s+walked',
        r'([A-Z][a-z]+)\s+ran',
    ]

    # Dialogue pattern
    DIALOGUE_PATTERN = r'"([^"]+)"(?:\s+(?:said|asked|replied|whispered|shouted))?\s*([A-Z][a-z]+)?'

    # Environment keywords
    ENVIRONMENT_KEYWORDS = {
        'office': ['office', 'desk', 'computer', 'workplace', 'cubicle'],
        'forest': ['forest', 'woods', 'trees', 'path', 'nature', 'trail'],
        'city': ['city', 'street', 'building', 'urban', 'downtown', 'traffic'],
        'home': ['home', 'house', 'living room', 'kitchen', 'bedroom', 'apartment'],
        'space': ['space', 'stars', 'galaxy', 'cosmos', 'spaceship', 'orbit'],
        'beach': ['beach', 'ocean', 'sand', 'waves', 'shore', 'sea'],
        'mountains': ['mountain', 'peak', 'cliff', 'valley', 'hiking', 'summit']
    }

    # Mood keywords
    MOOD_KEYWORDS = {
        CharacterMood.HAPPY: ['smiled', 'laughed', 'grinned', 'joyful', 'happy', 'excited', 'delighted'],
        CharacterMood.SAD: ['cried', 'wept', 'sighed', 'sad', 'depressed', 'tearful', 'mourned'],
        CharacterMood.ANGRY: ['shouted', 'yelled', 'furious', 'angry', 'enraged', 'frustrated'],
        CharacterMood.SURPRISED: ['gasped', 'shocked', 'surprised', 'amazed', 'stunned', 'startled'],
        CharacterMood.THOUGHTFUL: ['pondered', 'considered', 'thoughtful', 'contemplated', 'wondered'],
        CharacterMood.DETERMINED: ['resolved', 'determined', 'decided', 'committed', 'focused'],
        CharacterMood.FEARFUL: ['trembled', 'feared', 'scared', 'terrified', 'frightened', 'anxious']
    }

    def extract_narrative(self, text: str) -> NarrativeScene:
        """Extract narrative elements from text."""
        characters = self._extract_characters(text)
        dialogue = self._extract_dialogue(text)
        environment = self._detect_environment(text)
        scene_type = self._detect_scene_type(text)

        # Assign dialogue to characters and update moods
        for d in dialogue:
            char_name = d.get('character')
            if char_name:
                for char in characters:
                    if char.name.lower() == char_name.lower():
                        char.speaking = True
                        char.current_line = d['line']
                        char.mood = self._detect_mood(d['line'])
                        break

        # Position characters in scene
        self._position_characters(characters)

        return NarrativeScene(
            id="scene_0",
            scene_type=scene_type,
            characters=characters,
            environment=environment,
            dialogue=dialogue,
            action=self._extract_action(text),
            narration=self._extract_narration(text),
            camera_angle=self._suggest_camera_angle(scene_type, len(characters)),
            camera_movement=self._suggest_camera_movement(scene_type)
        )

    def _extract_characters(self, text: str) -> list[Character]:
        """Extract character names from text."""
        characters = {}

        for pattern in self.CHARACTER_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                name = match if isinstance(match, str) else match[0]
                if name and name not in characters and len(name) > 1:
                    # Assign role based on appearance order
                    role = 'protagonist' if len(characters) == 0 else \
                           'antagonist' if len(characters) == 1 else 'supporting'

                    characters[name] = Character(
                        id=f"char_{len(characters)}",
                        name=name,
                        role=role,
                        color=ROLE_COLORS.get(role, '#6b7280')
                    )

        return list(characters.values())

    def _extract_dialogue(self, text: str) -> list[dict[str, str]]:
        """Extract dialogue from text."""
        dialogue = []
        matches = re.findall(self.DIALOGUE_PATTERN, text)

        for match in matches:
            line, speaker = match
            dialogue.append({
                'line': line,
                'character': speaker if speaker else None
            })

        return dialogue

    def _detect_environment(self, text: str) -> Environment | None:
        """Detect environment from context."""
        text_lower = text.lower()

        for env_key, keywords in self.ENVIRONMENT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                env = ENVIRONMENT_PRESETS.get(env_key)
                if env:
                    # Detect time of day
                    if any(w in text_lower for w in ['night', 'dark', 'midnight', 'moon']):
                        env.time_of_day = 'night'
                        env.colors = ['#1a1a2e', '#16213e']
                    elif any(w in text_lower for w in ['dawn', 'sunrise', 'morning']):
                        env.time_of_day = 'dawn'
                        env.colors = ['#ff7b54', '#ffb26b']
                    elif any(w in text_lower for w in ['dusk', 'sunset', 'evening']):
                        env.time_of_day = 'dusk'
                        env.colors = ['#ff6b6b', '#feca57']

                    # Detect weather
                    if any(w in text_lower for w in ['rain', 'raining', 'storm', 'stormy']):
                        env.weather = 'rain'
                        env.mood = 'dramatic'
                    elif any(w in text_lower for w in ['fog', 'foggy', 'mist', 'misty']):
                        env.weather = 'fog'
                        env.mood = 'mysterious'

                    return env

        return None

    def _detect_scene_type(self, text: str) -> SceneType:
        """Detect the type of scene."""
        if '"' in text:  # Has dialogue
            return SceneType.DIALOGUE
        elif any(w in text.lower() for w in ['ran', 'jumped', 'fought', 'chased', 'attacked']):
            return SceneType.ACTION
        else:
            return SceneType.DESCRIPTION

    def _detect_mood(self, text: str) -> CharacterMood:
        """Detect character mood from their dialogue/action."""
        text_lower = text.lower()

        for mood, keywords in self.MOOD_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return mood

        return CharacterMood.NEUTRAL

    def _extract_action(self, text: str) -> str | None:
        """Extract action descriptions."""
        action_patterns = [
            r'([A-Z][a-z]+\s+(?:ran|walked|jumped|fought|grabbed|threw|opened|closed)[^.]+\.)',
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]

        return None

    def _extract_narration(self, text: str) -> str | None:
        """Extract narration (non-dialogue, non-action)."""
        # Remove dialogue
        narration = re.sub(r'"[^"]+"', '', text)
        # Remove character actions
        narration = re.sub(r'[A-Z][a-z]+\s+(?:said|asked|replied)', '', narration)
        narration = narration.strip()

        if len(narration) > 20:
            return narration[:200] + ('...' if len(narration) > 200 else '')

        return None

    def _position_characters(self, characters: list[Character]) -> None:
        """Position characters in scene for visual layout."""
        n = len(characters)
        if n == 0:
            return

        # Spread characters horizontally
        spacing = 4.0
        start_x = -spacing * (n - 1) / 2

        for i, char in enumerate(characters):
            char.position = {
                'x': start_x + i * spacing,
                'y': 0,
                'z': 0 if char.speaking else -2  # Speaking characters in front
            }


class NarrativeSceneBuilder:
    """Build 3D scene data for narrative visualization."""

    def build_scene(self, narrative: NarrativeScene) -> dict[str, Any]:
        """Convert narrative scene to 3D scene representation."""
        elements = []

        # Add environment backdrop
        if narrative.environment:
            elements.append(self._build_environment(narrative.environment))

        # Add characters
        for char in narrative.characters:
            elements.append(self._build_character(char))

        # Add dialogue bubbles
        for i, d in enumerate(narrative.dialogue[:3]):  # Limit to 3 recent lines
            elements.append(self._build_dialogue_bubble(d, i, narrative.characters))

        # Add narration overlay if present
        if narrative.narration:
            elements.append({
                'id': 'narration',
                'type': 'text_overlay',
                'position': {'x': 0, 'y': 3, 'z': 0},
                'content': narrative.narration,
                'style': {
                    'color': '#ffffff',
                    'background': 'rgba(0,0,0,0.7)',
                    'font_style': 'italic',
                    'max_width': 600
                }
            })

        return {
            'mode': 'narrative',
            'title': f"Scene: {narrative.environment.name if narrative.environment else 'Story'}",
            'scene_type': narrative.scene_type.value,
            'elements': elements,
            'camera': self._get_camera_settings(narrative),
            'lighting': self._get_lighting(narrative),
            'background': self._get_background(narrative.environment),
            'animation': self._get_animation(narrative)
        }

    def _build_environment(self, env: Environment) -> dict[str, Any]:
        """Build environment backdrop element."""
        return {
            'id': 'environment',
            'type': 'environment',
            'environment_type': env.type,
            'name': env.name,
            'time_of_day': env.time_of_day,
            'weather': env.weather,
            'mood': env.mood,
            'colors': env.colors,
            'props': env.props,
            'position': {'x': 0, 'y': 0, 'z': -20},
            'size': {'w': 60, 'h': 30, 'd': 1}
        }

    def _build_character(self, char: Character) -> dict[str, Any]:
        """Build character element."""
        return {
            'id': char.id,
            'type': 'character',
            'name': char.name,
            'role': char.role,
            'mood': char.mood.value,
            'speaking': char.speaking,
            'position': char.position,
            'color': char.color,
            'style': {
                'glow': char.speaking,  # Glow if speaking
                'scale': 1.2 if char.speaking else 1.0
            }
        }

    def _build_dialogue_bubble(self, dialogue: dict[str, str], index: int,
                               characters: list[Character]) -> dict[str, Any]:
        """Build dialogue bubble element."""
        # Find character position
        char_name = dialogue.get('character', '')
        position = {'x': 0, 'y': 2 + index * 0.8, 'z': 2}

        for char in characters:
            if char.name.lower() == char_name.lower():
                position['x'] = char.position['x']
                break

        return {
            'id': f'dialogue_{index}',
            'type': 'dialogue_bubble',
            'speaker': char_name,
            'line': dialogue['line'],
            'position': position,
            'style': {
                'background': '#ffffff',
                'color': '#1e293b',
                'tail_direction': 'down'
            }
        }

    def _get_camera_settings(self, narrative: NarrativeScene) -> dict[str, Any]:
        """Get camera settings based on scene type."""
        if narrative.camera_angle == 'wide':
            return {
                'position': {'x': 0, 'y': 5, 'z': 20},
                'target': {'x': 0, 'y': 0, 'z': 0},
                'fov': 75
            }
        elif narrative.camera_angle == 'close':
            return {
                'position': {'x': 0, 'y': 1, 'z': 6},
                'target': {'x': 0, 'y': 1, 'z': 0},
                'fov': 45
            }
        else:  # medium
            return {
                'position': {'x': 0, 'y': 2, 'z': 12},
                'target': {'x': 0, 'y': 0, 'z': 0},
                'fov': 60
            }

    def _get_lighting(self, narrative: NarrativeScene) -> str:
        """Get lighting based on environment mood."""
        if narrative.environment:
            if narrative.environment.mood == 'dramatic':
                return 'dramatic'
            elif narrative.environment.mood == 'mysterious':
                return 'low'
            elif narrative.environment.time_of_day == 'night':
                return 'moonlight'
            elif narrative.environment.time_of_day in ['dawn', 'dusk']:
                return 'warm'

        return 'soft'

    def _get_background(self, env: Environment | None) -> dict[str, Any]:
        """Get background settings."""
        if env:
            return {
                'type': 'gradient',
                'colors': env.colors
            }
        return {
            'type': 'gradient',
            'colors': ['#e2e8f0', '#94a3b8']
        }

    def _get_animation(self, narrative: NarrativeScene) -> dict[str, Any] | None:
        """Get animation settings based on scene."""
        if narrative.scene_type == SceneType.ACTION:
            return {
                'camera_shake': True,
                'speed': 1.5
            }
        elif narrative.camera_movement:
            return {
                'camera_movement': narrative.camera_movement,
                'duration': 2.0
            }
        return None

    def _suggest_camera_angle(self, scene_type: SceneType, num_characters: int) -> str:
        """Suggest appropriate camera angle."""
        if scene_type == SceneType.DIALOGUE and num_characters <= 2:
            return 'medium'
        elif scene_type == SceneType.ACTION:
            return 'wide'
        elif num_characters > 3:
            return 'wide'
        return 'medium'

    def _suggest_camera_movement(self, scene_type: SceneType) -> str | None:
        """Suggest camera movement if appropriate."""
        if scene_type == SceneType.ACTION:
            return 'track'
        elif scene_type == SceneType.TRANSITION:
            return 'pan_right'
        return None


def extract_narrative_from_text(text: str) -> dict[str, Any]:
    """Main entry point: extract narrative and build scene from text."""
    extractor = NarrativeExtractor()
    builder = NarrativeSceneBuilder()

    narrative = extractor.extract_narrative(text)

    if not narrative.characters and not narrative.environment:
        return None

    return builder.build_scene(narrative)


__all__ = [
    'Character',
    'CharacterMood',
    'Environment',
    'NarrativeScene',
    'SceneType',
    'NarrativeExtractor',
    'NarrativeSceneBuilder',
    'ENVIRONMENT_PRESETS',
    'extract_narrative_from_text'
]

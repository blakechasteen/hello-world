"""
Semantic Extractor
==================

Uses HoloLoom's 228D semantic space to enrich scene extraction.

Instead of keyword matching, we understand the meaning:
- Formality axis → Technical vs casual visual style
- Emotionality axis → Dramatic vs neutral lighting
- Urgency axis → Animation speed
- Complexity axis → Level of detail

The 16 interpretable axes become visual parameters.

Philosophy: "Meaning drives visualization, not keywords."

Created: 2025-12-01
Author: HoloLoom Team
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class VisualMood(Enum):
    """Mood derived from semantic analysis."""
    NEUTRAL = "neutral"
    DRAMATIC = "dramatic"
    CALM = "calm"
    URGENT = "urgent"
    TECHNICAL = "technical"
    PLAYFUL = "playful"
    MYSTERIOUS = "mysterious"


class VisualDensity(Enum):
    """How much detail to show."""
    MINIMAL = "minimal"      # Essential elements only
    STANDARD = "standard"    # Normal level of detail
    DETAILED = "detailed"    # Rich, complex scenes
    MAXIMAL = "maximal"      # Everything visible


@dataclass
class SemanticProfile:
    """
    Semantic analysis of text mapped to visual parameters.

    Each dimension is 0.0-1.0, derived from hololoom's 228D space.
    """
    # Core semantic dimensions (from hololoom's 16 interpretable axes)
    sentiment: float = 0.5       # negative ↔ positive
    formality: float = 0.5       # casual ↔ formal
    technicality: float = 0.5    # general ↔ technical
    certainty: float = 0.5       # uncertain ↔ certain
    urgency: float = 0.5         # relaxed ↔ urgent
    abstraction: float = 0.5     # concrete ↔ abstract
    specificity: float = 0.5     # vague ↔ specific
    temporality: float = 0.5     # timeless ↔ time-bound
    objectivity: float = 0.5     # subjective ↔ objective
    complexity: float = 0.5      # simple ↔ complex
    scope: float = 0.5           # narrow ↔ broad
    directness: float = 0.5      # indirect ↔ direct
    emotionality: float = 0.5    # neutral ↔ emotional
    actionability: float = 0.5   # informational ↔ actionable
    novelty: float = 0.5         # familiar ↔ novel
    controversy: float = 0.5     # consensus ↔ controversial

    # Derived visual parameters
    @property
    def mood(self) -> VisualMood:
        """Derive visual mood from semantic dimensions."""
        if self.urgency > 0.7:
            return VisualMood.URGENT
        if self.emotionality > 0.7:
            return VisualMood.DRAMATIC
        if self.technicality > 0.7:
            return VisualMood.TECHNICAL
        if self.sentiment > 0.7 and self.emotionality < 0.4:
            return VisualMood.PLAYFUL
        if self.certainty < 0.3 or self.abstraction > 0.7:
            return VisualMood.MYSTERIOUS
        if self.urgency < 0.3 and self.emotionality < 0.4:
            return VisualMood.CALM
        return VisualMood.NEUTRAL

    @property
    def density(self) -> VisualDensity:
        """Derive visual density from complexity and scope."""
        score = (self.complexity + self.scope + self.specificity) / 3
        if score > 0.75:
            return VisualDensity.MAXIMAL
        if score > 0.5:
            return VisualDensity.DETAILED
        if score > 0.25:
            return VisualDensity.STANDARD
        return VisualDensity.MINIMAL

    @property
    def animation_speed(self) -> float:
        """Derive animation speed from urgency and emotionality."""
        return 0.5 + (self.urgency * 0.3) + (self.emotionality * 0.2)

    @property
    def color_temperature(self) -> str:
        """Derive color temperature from sentiment and emotionality."""
        warmth = (self.sentiment + self.emotionality) / 2
        if warmth > 0.65:
            return "warm"
        if warmth < 0.35:
            return "cool"
        return "neutral"

    @property
    def lighting_intensity(self) -> float:
        """Derive lighting intensity from certainty and formality."""
        return 0.5 + (self.certainty * 0.3) + (self.formality * 0.2)


@dataclass
class EnrichedScene:
    """A scene enriched with semantic understanding."""
    elements: List[Dict[str, Any]]
    profile: SemanticProfile
    style_hints: Dict[str, Any] = field(default_factory=dict)
    related_memories: List[Dict[str, Any]] = field(default_factory=list)

    def get_lighting_config(self) -> Dict[str, Any]:
        """Get lighting configuration based on semantic profile."""
        mood = self.profile.mood
        intensity = self.profile.lighting_intensity
        temp = self.profile.color_temperature

        configs = {
            VisualMood.NEUTRAL: {
                "ambient": 0.4 * intensity,
                "directional": 0.6 * intensity,
                "color": "#ffffff",
            },
            VisualMood.DRAMATIC: {
                "ambient": 0.2 * intensity,
                "directional": 0.9 * intensity,
                "color": "#ffd700" if temp == "warm" else "#4169e1",
                "shadows": True,
            },
            VisualMood.CALM: {
                "ambient": 0.6 * intensity,
                "directional": 0.4 * intensity,
                "color": "#e6f2ff",
            },
            VisualMood.URGENT: {
                "ambient": 0.3 * intensity,
                "directional": 0.8 * intensity,
                "color": "#ff6b6b",
                "pulse": True,
            },
            VisualMood.TECHNICAL: {
                "ambient": 0.5 * intensity,
                "directional": 0.5 * intensity,
                "color": "#e0e0e0",
            },
            VisualMood.PLAYFUL: {
                "ambient": 0.5 * intensity,
                "directional": 0.6 * intensity,
                "color": "#ffeb3b",
            },
            VisualMood.MYSTERIOUS: {
                "ambient": 0.2 * intensity,
                "directional": 0.4 * intensity,
                "color": "#9c27b0",
                "fog": True,
            },
        }

        return configs.get(mood, configs[VisualMood.NEUTRAL])

    def get_animation_config(self) -> Dict[str, Any]:
        """Get animation configuration based on semantic profile."""
        speed = self.profile.animation_speed
        mood = self.profile.mood

        return {
            "transition_duration": max(0.2, 1.0 - speed * 0.5),  # 0.5-0.8s
            "easing": "easeInOutCubic" if mood == VisualMood.CALM else "easeOutQuart",
            "stagger_delay": 0.05 + (1.0 - speed) * 0.1,
            "pulse_enabled": mood == VisualMood.URGENT,
            "float_enabled": mood in [VisualMood.MYSTERIOUS, VisualMood.PLAYFUL],
        }


class SemanticExtractor:
    """
    Extracts semantic understanding from text using HoloLoom's 228D space.

    Connects to HoloLoom's semantic calculus for rich understanding,
    or falls back to heuristic analysis when HoloLoom is unavailable.
    """

    def __init__(self, hololoom=None):
        """
        Initialize with optional HoloLoom instance.

        Args:
            hololoom: Optional HoloLoom instance for semantic projection
        """
        self.hololoom = hololoom
        self._cache: Dict[str, SemanticProfile] = {}

    async def analyze(self, text: str) -> SemanticProfile:
        """
        Analyze text and return semantic profile.

        Uses HoloLoom's semantic calculus if available,
        falls back to heuristic analysis otherwise.
        """
        # Check cache
        cache_key = hash(text[:100])  # Hash first 100 chars
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try HoloLoom semantic projection
        if self.hololoom is not None:
            try:
                profile = await self._analyze_with_hololoom(text)
            except Exception:
                profile = self._analyze_heuristic(text)
        else:
            profile = self._analyze_heuristic(text)

        self._cache[cache_key] = profile
        return profile

    async def _analyze_with_hololoom(self, text: str) -> SemanticProfile:
        """Analyze using HoloLoom's 228D semantic space."""
        # Project text onto semantic space
        # The first 16 dimensions are our interpretable axes
        projection = await self.hololoom.project_semantic(text)

        # Map projection to profile
        # (Assuming projection returns dict with axis names)
        return SemanticProfile(
            sentiment=projection.get('sentiment', 0.5),
            formality=projection.get('formality', 0.5),
            technicality=projection.get('technicality', 0.5),
            certainty=projection.get('certainty', 0.5),
            urgency=projection.get('urgency', 0.5),
            abstraction=projection.get('abstraction', 0.5),
            specificity=projection.get('specificity', 0.5),
            temporality=projection.get('temporality', 0.5),
            objectivity=projection.get('objectivity', 0.5),
            complexity=projection.get('complexity', 0.5),
            scope=projection.get('scope', 0.5),
            directness=projection.get('directness', 0.5),
            emotionality=projection.get('emotionality', 0.5),
            actionability=projection.get('actionability', 0.5),
            novelty=projection.get('novelty', 0.5),
            controversy=projection.get('controversy', 0.5),
        )

    def _analyze_heuristic(self, text: str) -> SemanticProfile:
        """Fallback heuristic analysis when HoloLoom unavailable."""
        text_lower = text.lower()
        word_count = len(text.split())

        # Sentiment indicators
        positive_words = {'good', 'great', 'excellent', 'happy', 'success', 'beautiful', 'amazing'}
        negative_words = {'bad', 'terrible', 'fail', 'error', 'problem', 'ugly', 'wrong'}
        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)
        sentiment = 0.5 + (positive_count - negative_count) * 0.1

        # Formality indicators
        formal_words = {'therefore', 'however', 'furthermore', 'regarding', 'pursuant'}
        informal_words = {'hey', 'cool', 'awesome', 'yeah', 'gonna', 'wanna'}
        formal_count = sum(1 for w in formal_words if w in text_lower)
        informal_count = sum(1 for w in informal_words if w in text_lower)
        formality = 0.5 + (formal_count - informal_count) * 0.15

        # Technicality indicators
        tech_words = {'api', 'database', 'function', 'class', 'service', 'component',
                      'algorithm', 'interface', 'protocol', 'implementation'}
        tech_count = sum(1 for w in tech_words if w in text_lower)
        technicality = min(0.3 + tech_count * 0.15, 1.0)

        # Urgency indicators
        urgent_words = {'urgent', 'immediately', 'asap', 'now', 'critical', 'emergency'}
        urgent_count = sum(1 for w in urgent_words if w in text_lower)
        has_exclamation = '!' in text
        urgency = min(0.3 + urgent_count * 0.2 + (0.15 if has_exclamation else 0), 1.0)

        # Emotionality indicators
        emotion_words = {'love', 'hate', 'fear', 'joy', 'anger', 'sad', 'excited', 'worried'}
        emotion_count = sum(1 for w in emotion_words if w in text_lower)
        emotionality = min(0.3 + emotion_count * 0.2 + (0.1 if has_exclamation else 0), 1.0)

        # Complexity (based on sentence structure and length)
        avg_word_length = sum(len(w) for w in text.split()) / max(word_count, 1)
        complexity = min(0.3 + (avg_word_length - 4) * 0.1 + (word_count / 50) * 0.2, 1.0)

        # Certainty indicators
        uncertain_words = {'maybe', 'perhaps', 'might', 'could', 'possibly', 'uncertain'}
        certain_words = {'definitely', 'certainly', 'absolutely', 'must', 'always', 'never'}
        uncertain_count = sum(1 for w in uncertain_words if w in text_lower)
        certain_count = sum(1 for w in certain_words if w in text_lower)
        certainty = 0.5 + (certain_count - uncertain_count) * 0.15

        # Abstraction (concrete vs abstract)
        concrete_words = {'table', 'button', 'screen', 'user', 'file', 'image', 'room'}
        abstract_words = {'concept', 'idea', 'theory', 'philosophy', 'meaning', 'essence'}
        concrete_count = sum(1 for w in concrete_words if w in text_lower)
        abstract_count = sum(1 for w in abstract_words if w in text_lower)
        abstraction = 0.5 + (abstract_count - concrete_count) * 0.1

        # Actionability
        action_words = {'click', 'create', 'build', 'implement', 'deploy', 'run', 'execute'}
        action_count = sum(1 for w in action_words if w in text_lower)
        actionability = min(0.3 + action_count * 0.15, 1.0)

        return SemanticProfile(
            sentiment=max(0, min(sentiment, 1)),
            formality=max(0, min(formality, 1)),
            technicality=max(0, min(technicality, 1)),
            certainty=max(0, min(certainty, 1)),
            urgency=max(0, min(urgency, 1)),
            abstraction=max(0, min(abstraction, 1)),
            specificity=0.5,  # Hard to determine heuristically
            temporality=0.5,
            objectivity=0.5,
            complexity=max(0, min(complexity, 1)),
            scope=min(0.3 + word_count / 100, 1.0),
            directness=0.5,
            emotionality=max(0, min(emotionality, 1)),
            actionability=max(0, min(actionability, 1)),
            novelty=0.5,
            controversy=0.5,
        )

    async def enrich_scene(
        self,
        text: str,
        base_elements: List[Dict[str, Any]],
    ) -> EnrichedScene:
        """
        Enrich a scene with semantic understanding.

        Args:
            text: Original text being visualized
            base_elements: Elements extracted by other visualizers

        Returns:
            EnrichedScene with semantic styling applied
        """
        profile = await self.analyze(text)

        # Recall related memories from hololoom
        related = []
        if self.hololoom is not None:
            try:
                memories = await self.hololoom.recall(text, k=5)
                related = [{"text": m.content, "relevance": m.relevance} for m in memories]
            except Exception:
                pass

        # Apply semantic styling to elements
        enriched_elements = self._apply_semantic_styling(base_elements, profile)

        # Generate style hints
        style_hints = {
            "mood": profile.mood.value,
            "density": profile.density.value,
            "color_temperature": profile.color_temperature,
            "animation_speed": profile.animation_speed,
            "lighting": {
                "intensity": profile.lighting_intensity,
                "ambient": 0.4 * profile.lighting_intensity,
            },
        }

        return EnrichedScene(
            elements=enriched_elements,
            profile=profile,
            style_hints=style_hints,
            related_memories=related,
        )

    def _apply_semantic_styling(
        self,
        elements: List[Dict[str, Any]],
        profile: SemanticProfile,
    ) -> List[Dict[str, Any]]:
        """Apply semantic-derived styling to elements."""
        styled = []

        for elem in elements:
            styled_elem = elem.copy()

            # Add mood-based effects
            if profile.mood == VisualMood.DRAMATIC:
                styled_elem.setdefault('effects', []).append('shadow')
                styled_elem.setdefault('effects', []).append('rim_light')

            elif profile.mood == VisualMood.MYSTERIOUS:
                styled_elem.setdefault('effects', []).append('fog')
                styled_elem.setdefault('effects', []).append('glow')
                styled_elem['opacity'] = elem.get('opacity', 1.0) * 0.85

            elif profile.mood == VisualMood.URGENT:
                styled_elem.setdefault('effects', []).append('pulse')
                styled_elem['animation'] = {
                    'type': 'pulse',
                    'speed': profile.animation_speed * 1.5,
                }

            elif profile.mood == VisualMood.PLAYFUL:
                styled_elem.setdefault('effects', []).append('bounce')
                styled_elem['animation'] = {
                    'type': 'float',
                    'speed': profile.animation_speed,
                }

            # Adjust detail level based on density
            if profile.density == VisualDensity.MINIMAL:
                styled_elem['detail_level'] = 'low'
            elif profile.density == VisualDensity.MAXIMAL:
                styled_elem['detail_level'] = 'high'
                styled_elem.setdefault('effects', []).append('detail_lines')

            # Temperature affects colors
            if 'color' in styled_elem:
                styled_elem['color'] = self._adjust_color_temperature(
                    styled_elem['color'],
                    profile.color_temperature
                )

            styled.append(styled_elem)

        return styled

    def _adjust_color_temperature(self, color: str, temperature: str) -> str:
        """Adjust color based on temperature setting."""
        if not color.startswith('#') or len(color) != 7:
            return color

        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)

            if temperature == "warm":
                r = min(255, int(r * 1.1))
                b = int(b * 0.9)
            elif temperature == "cool":
                r = int(r * 0.9)
                b = min(255, int(b * 1.1))

            return f"#{r:02x}{g:02x}{b:02x}"
        except ValueError:
            return color


# Singleton instance
_semantic_extractor: Optional[SemanticExtractor] = None


def get_semantic_extractor(hololoom=None) -> SemanticExtractor:
    """Get or create semantic extractor singleton."""
    global _semantic_extractor
    if _semantic_extractor is None or hololoom is not None:
        _semantic_extractor = SemanticExtractor(hololoom)
    return _semantic_extractor


__all__ = [
    'VisualMood',
    'VisualDensity',
    'SemanticProfile',
    'EnrichedScene',
    'SemanticExtractor',
    'get_semantic_extractor',
]

"""
Consciousness Slider: Individual → Shared → Universal Dream Spectrum
=====================================================================

Version: 1.0.0
Date: 2025-11-24
Status: Production Ready

The Consciousness Slider enables smooth transitions across dream consciousness levels,
from deeply personal (individual) through collaborative (shared) to collective (universal).

Core Philosophy:
    "Each consciousness level reveals different facets of truth. Individual dreams
     show personal meaning. Shared dreams reveal relational patterns. Universal
     dreams dissolve the ego to show archetypal truth."

Architecture:
    Slider Value (0.0-1.0)
         ↓
    [ConsciousnessLevel Determination]
         ↓
    [Parameter Interpolation]
         ↓
    ConsciousnessSettings
         ↓
    [Dream Adjustment]
         ↓
    Privacy-Preserving Dream

Three Consciousness Levels:
    1. INDIVIDUAL (0-0.33): Private dreams, personal symbols, ego prominence
    2. SHARED (0.33-0.66): Collaborative dreams, relational symbols, ego softening
    3. UNIVERSAL (0.66-1.0): Collective dreams, archetypal symbols, ego dissolution

Integration Points:
    - symbolic_encoder.py: Privacy gradient controls symbol selection
    - dream_matching.py: Max participants filters dream recipients
    - narrative generator: Ego dissolution affects narrative voice
    - symbol database: Symbol universality filters available symbols
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, List, Any
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class ConsciousnessLevel(Enum):
    """Three consciousness levels in the dream spectrum."""
    INDIVIDUAL = "individual"  # 0-0.33: Personal, private, ego-centered
    SHARED = "shared"  # 0.33-0.66: Relational, symbolic, ego-softening
    UNIVERSAL = "universal"  # 0.66-1.0: Archetypal, collective, ego-dissolving


class PrivacyMode(Enum):
    """Privacy preservation mode."""
    NONE = "none"  # No privacy (0.0): Specific facts allowed
    SYMBOLIC = "symbolic"  # Medium (0.5): Emotional essence only
    ARCHETYPAL = "archetypal"  # Full (1.0): Pure archetype, ego dissolved


# Parameters controlling consciousness level behavior
CONSCIOUSNESS_LEVEL_BOUNDARIES = {
    ConsciousnessLevel.INDIVIDUAL: (0.0, 0.33),
    ConsciousnessLevel.SHARED: (0.33, 0.66),
    ConsciousnessLevel.UNIVERSAL: (0.66, 1.0),
}

# Slider value ranges for each parameter across consciousness levels
PARAMETER_RANGES = {
    # Maximum number of dream participants
    "max_participants": {
        ConsciousnessLevel.INDIVIDUAL: (1, 1),  # Solo dreams
        ConsciousnessLevel.SHARED: (2, 5),  # Small group
        ConsciousnessLevel.UNIVERSAL: (6, 999),  # Everyone
    },
    # Privacy gradient: 0 = expose details, 1 = obscure everything
    "privacy_gradient": {
        ConsciousnessLevel.INDIVIDUAL: (0.0, 0.3),
        ConsciousnessLevel.SHARED: (0.5, 0.8),
        ConsciousnessLevel.UNIVERSAL: (0.9, 1.0),
    },
    # Archetypal strength: 0 = personal, 1 = pure archetype
    "archetypal_strength": {
        ConsciousnessLevel.INDIVIDUAL: (0.2, 0.5),
        ConsciousnessLevel.SHARED: (0.6, 0.8),
        ConsciousnessLevel.UNIVERSAL: (0.9, 1.0),
    },
    # Ego dissolution: 0 = strong ego, 1 = ego dissolves
    "ego_dissolution": {
        ConsciousnessLevel.INDIVIDUAL: (0.0, 0.2),
        ConsciousnessLevel.SHARED: (0.3, 0.6),
        ConsciousnessLevel.UNIVERSAL: (0.7, 1.0),
    },
    # Symbol universality: 0 = personal symbols, 1 = universal symbols
    "symbol_universality": {
        ConsciousnessLevel.INDIVIDUAL: (0.3, 0.6),
        ConsciousnessLevel.SHARED: (0.6, 0.9),
        ConsciousnessLevel.UNIVERSAL: (0.9, 1.0),
    },
}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ConsciousnessSettings:
    """
    Complete settings derived from slider value.

    Attributes:
        slider_value: Consciousness slider position (0.0-1.0)
        level: Consciousness level (individual/shared/universal)
        max_participants: Maximum number of dream participants (1-999)
        privacy_gradient: Privacy preservation (0.0=none, 1.0=maximum)
        archetypal_strength: Archetype prominence (0.0=personal, 1.0=archetypal)
        ego_dissolution: Ego dissolution level (0.0=present, 1.0=dissolved)
        symbol_universality: Universal symbol filter (0.0=personal, 1.0=universal)
        privacy_mode: Privacy mode category (none/symbolic/archetypal)
        level_position: Position within consciousness level (0.0-1.0)
        narrative_voice: Narrative voice type ("first_person"/"third_person"/"collective")
        symbol_filter_threshold: Minimum universality score for symbols (0.0-1.0)
        detail_obscuration: How much detail to obscure (0.0-1.0)
        emotional_universalization: How much to universalize emotion (0.0-1.0)
    """

    # Core parameters
    slider_value: float
    level: ConsciousnessLevel
    max_participants: int
    privacy_gradient: float
    archetypal_strength: float
    ego_dissolution: float
    symbol_universality: float

    # Derived parameters
    privacy_mode: PrivacyMode = field(init=False)
    level_position: float = field(init=False)
    narrative_voice: str = field(init=False)
    symbol_filter_threshold: float = field(init=False)
    detail_obscuration: float = field(init=False)
    emotional_universalization: float = field(init=False)

    def __post_init__(self):
        """Derive secondary parameters from primary ones."""
        # Determine privacy mode
        if self.privacy_gradient < 0.4:
            self.privacy_mode = PrivacyMode.NONE
        elif self.privacy_gradient < 0.7:
            self.privacy_mode = PrivacyMode.SYMBOLIC
        else:
            self.privacy_mode = PrivacyMode.ARCHETYPAL

        # Calculate position within level (0.0-1.0)
        level_start, level_end = CONSCIOUSNESS_LEVEL_BOUNDARIES[self.level]
        level_range = level_end - level_start
        self.level_position = (self.slider_value - level_start) / level_range

        # Determine narrative voice based on ego dissolution
        if self.ego_dissolution < 0.3:
            self.narrative_voice = "first_person"  # "I am trapped..."
        elif self.ego_dissolution < 0.7:
            self.narrative_voice = "third_person"  # "Someone is trapped..."
        else:
            self.narrative_voice = "collective"  # "We are all trapped..."

        # Symbol filter: only allow symbols with universality >= threshold
        self.symbol_filter_threshold = self.symbol_universality - 0.2  # Allow 20% range below

        # Detail obscuration: complement of privacy gradient (higher = more obscure)
        self.detail_obscuration = self.privacy_gradient

        # Emotional universalization: how much to shift emotion to archetypal form
        self.emotional_universalization = self.archetypal_strength

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "slider_value": self.slider_value,
            "level": self.level.value,
            "max_participants": self.max_participants,
            "privacy_gradient": self.privacy_gradient,
            "archetypal_strength": self.archetypal_strength,
            "ego_dissolution": self.ego_dissolution,
            "symbol_universality": self.symbol_universality,
            "privacy_mode": self.privacy_mode.value,
            "level_position": self.level_position,
            "narrative_voice": self.narrative_voice,
            "symbol_filter_threshold": self.symbol_filter_threshold,
            "detail_obscuration": self.detail_obscuration,
            "emotional_universalization": self.emotional_universalization,
        }


@dataclass
class DreamAdjustment:
    """
    Adjustments applied to dream based on consciousness settings.

    Attributes:
        participant_filter: Recipient list (specific IDs or "all")
        privacy_instructions: Instructions for symbolic_encoder
        symbol_requirements: Symbol universality constraints
        narrative_adjustments: Voice and perspective changes
        ego_handling: How to handle ego/self references
        emotional_translation: How to translate emotions to universal forms
    """

    participant_filter: List[str] | str  # List of IDs or "all"
    privacy_instructions: Dict[str, float]  # privacy_gradient, etc.
    symbol_requirements: Dict[str, float]  # symbol_universality, archetype_score, etc.
    narrative_adjustments: Dict[str, str]  # voice, perspective, tense
    ego_handling: Dict[str, Any]  # ego_visibility, identity_obscuration, etc.
    emotional_translation: Dict[str, float]  # universalization scores


# ============================================================================
# Main ConsciousnessSlider Class
# ============================================================================


class ConsciousnessSlider:
    """
    Consciousness slider system for dream consciousness transitions.

    The slider enables smooth parameter interpolation across three consciousness
    levels, from individual (personal, private) through shared (relational, symbolic)
    to universal (archetypal, collective).

    Example:
        >>> slider = ConsciousnessSlider()
        >>> settings = slider.get_settings(0.5)  # Mid-range (shared consciousness)
        >>> print(settings.level)  # ConsciousnessLevel.SHARED
        >>> print(settings.privacy_gradient)  # ~0.65

        >>> adjustment = slider.adjust_dream_parameters(base_dream, settings)
        >>> # Dream now has max 2-5 participants, symbolic privacy, etc.
    """

    def __init__(self):
        """Initialize consciousness slider."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_settings(self, slider_value: float) -> ConsciousnessSettings:
        """
        Convert slider value to consciousness settings.

        Args:
            slider_value: Position on consciousness slider (0.0-1.0)
                0.0 = individual, 0.5 = shared, 1.0 = universal

        Returns:
            ConsciousnessSettings with all parameters computed

        Raises:
            ValueError: If slider_value not in [0.0, 1.0]
        """
        if not 0.0 <= slider_value <= 1.0:
            raise ValueError(
                f"Slider value must be in [0.0, 1.0], got {slider_value}"
            )

        # Determine consciousness level
        level = self._determine_level(slider_value)

        # Interpolate parameters within the level
        params = self._interpolate_parameters(slider_value, level)

        settings = ConsciousnessSettings(
            slider_value=slider_value,
            level=level,
            max_participants=params["max_participants"],
            privacy_gradient=params["privacy_gradient"],
            archetypal_strength=params["archetypal_strength"],
            ego_dissolution=params["ego_dissolution"],
            symbol_universality=params["symbol_universality"],
        )

        self.logger.debug(
            f"Consciousness settings for slider {slider_value:.2f}: "
            f"level={level.value}, privacy={settings.privacy_gradient:.2f}, "
            f"archetype={settings.archetypal_strength:.2f}"
        )

        return settings

    def _determine_level(self, slider_value: float) -> ConsciousnessLevel:
        """Determine consciousness level from slider value."""
        if slider_value < 0.33:
            return ConsciousnessLevel.INDIVIDUAL
        elif slider_value < 0.66:
            return ConsciousnessLevel.SHARED
        else:
            return ConsciousnessLevel.UNIVERSAL

    def _interpolate_parameters(
        self, slider_value: float, level: ConsciousnessLevel
    ) -> Dict[str, float]:
        """
        Interpolate all parameters for given slider value and level.

        Uses linear interpolation within level boundaries for smooth transitions.
        """
        level_start, level_end = CONSCIOUSNESS_LEVEL_BOUNDARIES[level]
        t = (slider_value - level_start) / (level_end - level_start)  # 0.0-1.0 within level

        params = {}

        for param_name, level_ranges in PARAMETER_RANGES.items():
            min_val, max_val = level_ranges[level]
            # Linear interpolation within level
            params[param_name] = min_val + t * (max_val - min_val)

            # Special handling for integer parameters
            if param_name == "max_participants":
                params[param_name] = int(
                    min_val + t * (max_val - min_val)
                )

        return params

    def adjust_dream_parameters(
        self, base_dream: Dict[str, Any], settings: ConsciousnessSettings
    ) -> Dict[str, Any]:
        """
        Apply consciousness settings to dream parameters.

        Modifies dream to match consciousness level:
        - Filters participants by max_participants
        - Sets privacy parameters for symbolic_encoder
        - Adjusts narrative voice and perspective
        - Filters available symbols by universality
        - Handles ego references

        Args:
            base_dream: Base dream dict with structure:
                {
                    "content": str,
                    "symbols": List[Dict],
                    "participants": List[str],
                    "emotional_essence": Dict,
                    ...
                }
            settings: ConsciousnessSettings from slider

        Returns:
            Modified dream dict with consciousness adjustments applied
        """
        adjusted = base_dream.copy()

        # 1. Filter participants by max_participants
        adjusted["consciousness_settings"] = {
            "max_participants": settings.max_participants,
            "level": settings.level.value,
            "slider_value": settings.slider_value,
        }

        if "participants" in base_dream:
            # If more participants than allowed, sample randomly
            if len(base_dream["participants"]) > settings.max_participants:
                original_participants = base_dream["participants"]
                selected_participants = np.random.choice(
                    original_participants,
                    size=settings.max_participants,
                    replace=False,
                ).tolist()
                adjusted["participants"] = selected_participants
                adjusted["consciousness_filtering"] = {
                    "original_count": len(original_participants),
                    "selected_count": settings.max_participants,
                    "reason": f"Consciousness level {settings.level.value} allows max {settings.max_participants}",
                }

        # 2. Add privacy instructions for symbolic_encoder
        adjusted["privacy_instructions"] = {
            "privacy_gradient": settings.privacy_gradient,
            "privacy_mode": settings.privacy_mode.value,
            "detail_obscuration": settings.detail_obscuration,
        }

        # 3. Add symbol requirements (filters for symbol selection)
        adjusted["symbol_requirements"] = {
            "minimum_universality": settings.symbol_filter_threshold,
            "archetypal_strength": settings.archetypal_strength,
            "universality_boost": settings.symbol_universality,
        }

        # 4. Add narrative adjustments
        adjusted["narrative_adjustments"] = {
            "narrative_voice": settings.narrative_voice,
            "perspective": self._perspective_from_ego(settings.ego_dissolution),
            "ego_prominence": 1.0 - settings.ego_dissolution,  # Inverse: high ego = high prominence
            "emotional_universalization": settings.emotional_universalization,
        }

        # 5. Handle ego references based on ego_dissolution
        if "content" in base_dream:
            adjusted["content"] = self._adjust_ego_references(
                base_dream["content"], settings.ego_dissolution, settings.narrative_voice
            )

        # 6. Adjust emotional translation
        if "emotional_essence" in base_dream:
            adjusted["emotional_translation"] = {
                "universalization_level": settings.emotional_universalization,
                "archetypal_mapping": settings.archetypal_strength,
            }

        logger.info(
            f"Applied consciousness settings (level={settings.level.value}, "
            f"slider={settings.slider_value:.2f}) to dream"
        )

        return adjusted

    def _perspective_from_ego(self, ego_dissolution: float) -> str:
        """Map ego dissolution to perspective."""
        if ego_dissolution < 0.3:
            return "first_person"  # "I" perspective
        elif ego_dissolution < 0.7:
            return "third_person"  # "Someone" perspective
        else:
            return "collective"  # "We all" perspective

    def _adjust_ego_references(
        self, content: str, ego_dissolution: float, narrative_voice: str
    ) -> str:
        """
        Adjust ego references in dream content based on dissolution level.

        This is a basic implementation. In production, use NLP for better results.
        """
        if narrative_voice == "first_person":
            # Keep first person, no changes needed
            return content
        elif narrative_voice == "third_person":
            # Simple replacement of first person pronouns
            replacements = {
                "I am": "Someone is",
                "I feel": "A presence feels",
                "I see": "One sees",
                "I know": "It is known",
                "my ": "their ",
                "me ": "them ",
            }
            adjusted = content
            for orig, replacement in replacements.items():
                adjusted = adjusted.replace(orig, replacement)
            return adjusted
        else:  # collective
            # Replace with collective pronouns
            replacements = {
                "I am": "We are",
                "I feel": "We feel",
                "I see": "We see",
                "I know": "We know",
                "my ": "our ",
                "me ": "us ",
            }
            adjusted = content
            for orig, replacement in replacements.items():
                adjusted = adjusted.replace(orig, replacement)
            return adjusted

    def get_slider_description(self, slider_value: float) -> str:
        """
        Get human-readable description of consciousness level.

        Args:
            slider_value: Position on slider (0.0-1.0)

        Returns:
            Descriptive text about consciousness level
        """
        if slider_value < 0.33:
            return (
                f"Individual Consciousness ({slider_value:.0%}): "
                "Private dream reflecting personal meaning. "
                "Specific memories and personal symbols. Strong ego presence."
            )
        elif slider_value < 0.66:
            return (
                f"Shared Consciousness ({slider_value:.0%}): "
                "Relational dream between small group. "
                "Symbolic language preserves privacy while enabling empathy. "
                "Ego softening begins."
            )
        else:
            return (
                f"Universal Consciousness ({slider_value:.0%}): "
                "Collective dream for all residents. "
                "Archetypal symbols dissolve individual ego. "
                "Pure essence of shared human experience."
            )

    def create_dream_adjustment(
        self, settings: ConsciousnessSettings, base_dream: Dict[str, Any]
    ) -> DreamAdjustment:
        """
        Create detailed dream adjustment object.

        Args:
            settings: Consciousness settings
            base_dream: Base dream to adjust

        Returns:
            DreamAdjustment with all necessary modifications
        """
        # Determine participant filter
        if settings.max_participants >= 6:
            participant_filter = "all"
        else:
            # Sample from base dream participants
            if "participants" in base_dream:
                participant_filter = np.random.choice(
                    base_dream["participants"],
                    size=min(settings.max_participants, len(base_dream["participants"])),
                    replace=False,
                ).tolist()
            else:
                participant_filter = "all"

        return DreamAdjustment(
            participant_filter=participant_filter,
            privacy_instructions={
                "privacy_gradient": settings.privacy_gradient,
                "privacy_mode": settings.privacy_mode.value,
                "detail_level": 1.0 - settings.detail_obscuration,
            },
            symbol_requirements={
                "minimum_universality": settings.symbol_filter_threshold,
                "archetypal_strength": settings.archetypal_strength,
            },
            narrative_adjustments={
                "narrative_voice": settings.narrative_voice,
                "perspective": self._perspective_from_ego(settings.ego_dissolution),
                "ego_visibility": 1.0 - settings.ego_dissolution,
            },
            ego_handling={
                "ego_dissolution": settings.ego_dissolution,
                "identity_obscuration": settings.detail_obscuration,
            },
            emotional_translation={
                "universalization_level": settings.emotional_universalization,
            },
        )


# ============================================================================
# Utility Functions
# ============================================================================


def slider_value_to_percentage(slider_value: float) -> str:
    """Convert slider value to percentage string."""
    return f"{slider_value * 100:.1f}%"


def get_consciousness_level_name(slider_value: float) -> str:
    """Get consciousness level name from slider value."""
    level = ConsciousnessSlider()._determine_level(slider_value)
    return level.value


def interpolate_between_levels(
    value1: float, value2: float, t: float
) -> float:
    """
    Smooth interpolation between two parameter values.

    Args:
        value1: Start value
        value2: End value
        t: Interpolation factor (0.0-1.0)

    Returns:
        Interpolated value using cubic Hermite spline
    """
    # Cubic Hermite for smooth transition
    t_smooth = t * t * (3 - 2 * t)
    return value1 + t_smooth * (value2 - value1)

"""
Emotion modeling system for Elle Game Engine.

Implements dimensional emotion model (valence-arousal-dominance) with trust mechanics.
NPCs have rich emotional states that update based on player actions and decay over time.

References:
- Russell's Circumplex Model of Affect (valence × arousal)
- PAD Emotional State Model (Mehrabian, 1996)
- Trust dynamics from social psychology

Created: 2025-11-16
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ============================================================================
# Emotion Dimensions and Mappings
# ============================================================================

class EmotionDimension(str, Enum):
    """Emotional dimensions in PAD model."""
    VALENCE = "valence"  # Positive/negative (-1.0 to 1.0)
    AROUSAL = "arousal"  # Calm/excited (0.0 to 1.0)
    DOMINANCE = "dominance"  # Submissive/dominant (0.0 to 1.0)
    TRUST = "trust"  # Trust toward player (0.0 to 1.0)


# Emotion labels mapped to PAD coordinates
# Format: (valence, arousal, dominance)
EMOTION_LABELS: Dict[str, Tuple[float, float, float]] = {
    # Positive emotions
    "happy": (0.8, 0.6, 0.5),
    "excited": (0.7, 0.9, 0.6),
    "content": (0.6, 0.3, 0.5),
    "grateful": (0.7, 0.4, 0.4),
    "proud": (0.6, 0.5, 0.7),
    "relieved": (0.5, 0.2, 0.4),

    # Negative emotions
    "angry": (-0.6, 0.8, 0.7),
    "fearful": (-0.7, 0.7, 0.2),
    "sad": (-0.6, 0.3, 0.3),
    "disgusted": (-0.7, 0.5, 0.6),
    "ashamed": (-0.5, 0.4, 0.2),
    "anxious": (-0.4, 0.7, 0.3),

    # Neutral/complex
    "neutral": (0.0, 0.4, 0.5),
    "curious": (0.2, 0.6, 0.5),
    "confused": (-0.2, 0.5, 0.3),
    "suspicious": (-0.3, 0.6, 0.6),
    "annoyed": (-0.4, 0.6, 0.5),
}


# ============================================================================
# Emotional State
# ============================================================================

@dataclass
class EmotionalState:
    """
    PAD (Pleasure-Arousal-Dominance) emotional state model.

    Dimensions:
    - valence: Positive/negative emotion (-1.0 to 1.0)
    - arousal: Energy level (0.0 = calm, 1.0 = excited)
    - dominance: Control/power (0.0 = submissive, 1.0 = dominant)
    - trust: Trust toward player (0.0 = no trust, 1.0 = complete trust)

    All dimensions decay toward baseline over time (emotional homeostasis).
    """

    valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.4  # 0.0 (calm) to 1.0 (excited), baseline 0.4
    dominance: float = 0.5  # 0.0 (submissive) to 1.0 (dominant), baseline 0.5
    trust: float = 0.5  # 0.0 (distrust) to 1.0 (trust), baseline 0.5

    # Metadata
    last_update: float = field(default_factory=time.time)
    baseline_valence: float = 0.0
    baseline_arousal: float = 0.4
    baseline_dominance: float = 0.5
    baseline_trust: float = 0.5

    def __post_init__(self):
        """Validate emotional dimensions."""
        self._clamp()

    def _clamp(self):
        """Clamp all dimensions to valid ranges."""
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))
        self.trust = max(0.0, min(1.0, self.trust))

    def update(
        self,
        valence_delta: float = 0.0,
        arousal_delta: float = 0.0,
        dominance_delta: float = 0.0,
        trust_delta: float = 0.0,
    ):
        """
        Update emotional state with deltas.

        Args:
            valence_delta: Change in valence (-1.0 to 1.0)
            arousal_delta: Change in arousal (-1.0 to 1.0)
            dominance_delta: Change in dominance (-1.0 to 1.0)
            trust_delta: Change in trust (-1.0 to 1.0)
        """
        self.valence += valence_delta
        self.arousal += arousal_delta
        self.dominance += dominance_delta
        self.trust += trust_delta

        self._clamp()
        self.last_update = time.time()

    def decay(self, time_elapsed_seconds: float, decay_rate: float = 0.1):
        """
        Decay emotions toward baseline (emotional homeostasis).

        Uses exponential decay: x(t) = baseline + (x0 - baseline) * e^(-decay_rate * t)

        Args:
            time_elapsed_seconds: Time since last update
            decay_rate: Decay constant (default: 0.1 = ~10 seconds to half-life)
        """
        decay_factor = math.exp(-decay_rate * time_elapsed_seconds)

        # Decay toward baseline
        self.valence = self.baseline_valence + (self.valence - self.baseline_valence) * decay_factor
        self.arousal = self.baseline_arousal + (self.arousal - self.baseline_arousal) * decay_factor
        self.dominance = self.baseline_dominance + (self.dominance - self.baseline_dominance) * decay_factor
        self.trust = self.baseline_trust + (self.trust - self.baseline_trust) * decay_factor

        self._clamp()
        self.last_update = time.time()

    def get_emotion_label(self) -> str:
        """
        Get closest emotion label based on current PAD coordinates.

        Uses Euclidean distance in PAD space.

        Returns:
            Closest emotion label (e.g., "happy", "angry", "curious")
        """
        min_distance = float('inf')
        closest_label = "neutral"

        for label, (v, a, d) in EMOTION_LABELS.items():
            distance = math.sqrt(
                (self.valence - v) ** 2 +
                (self.arousal - a) ** 2 +
                (self.dominance - d) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_label = label

        return closest_label

    def get_tone(self) -> str:
        """
        Get dialogue tone based on emotional state.

        Returns:
            Tone descriptor for dialogue (e.g., "warm", "stern", "anxious")
        """
        emotion = self.get_emotion_label()

        # Map emotions to tones
        tone_map = {
            "happy": "warm",
            "excited": "enthusiastic",
            "content": "calm",
            "grateful": "appreciative",
            "proud": "confident",
            "relieved": "relieved",
            "angry": "stern",
            "fearful": "nervous",
            "sad": "melancholic",
            "disgusted": "dismissive",
            "ashamed": "apologetic",
            "anxious": "anxious",
            "neutral": "neutral",
            "curious": "curious",
            "confused": "puzzled",
            "suspicious": "wary",
            "annoyed": "irritated",
        }

        return tone_map.get(emotion, "neutral")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "dominance": round(self.dominance, 3),
            "trust": round(self.trust, 3),
            "emotion_label": self.get_emotion_label(),
            "tone": self.get_tone(),
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EmotionalState":
        """Create from dictionary."""
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.4),
            dominance=data.get("dominance", 0.5),
            trust=data.get("trust", 0.5),
            last_update=data.get("last_update", time.time()),
        )

    @classmethod
    def from_emotion_label(cls, label: str, trust: float = 0.5) -> "EmotionalState":
        """
        Create emotional state from emotion label.

        Args:
            label: Emotion label (e.g., "happy", "angry", "curious")
            trust: Initial trust level (0.0 to 1.0)

        Returns:
            EmotionalState initialized to that emotion
        """
        if label not in EMOTION_LABELS:
            label = "neutral"

        v, a, d = EMOTION_LABELS[label]
        return cls(valence=v, arousal=a, dominance=d, trust=trust)


# ============================================================================
# Emotion Engine
# ============================================================================

class EmotionEngine:
    """
    Engine for updating NPC emotions based on player actions.

    Responsibilities:
    1. Apply emotion deltas based on action types
    2. Handle decay over time (emotional homeostasis)
    3. Maintain emotion history for learning
    4. Generate emotion-aware prompt modifications
    """

    def __init__(
        self,
        decay_rate: float = 0.1,
        decay_check_interval: float = 10.0,
    ):
        """
        Initialize emotion engine.

        Args:
            decay_rate: Exponential decay constant (default: 0.1)
            decay_check_interval: Seconds between automatic decay checks (default: 10)
        """
        self.decay_rate = decay_rate
        self.decay_check_interval = decay_check_interval

        # Emotion history: {npc_id: List[(timestamp, emotion_label)]}
        self.emotion_history: Dict[str, List[Tuple[float, str]]] = {}

    def apply_decay(self, emotional_state: EmotionalState) -> EmotionalState:
        """
        Apply time-based decay to emotional state.

        Args:
            emotional_state: Current emotional state

        Returns:
            Updated emotional state after decay
        """
        time_elapsed = time.time() - emotional_state.last_update

        if time_elapsed > 0:
            emotional_state.decay(time_elapsed, self.decay_rate)

        return emotional_state

    def process_player_action(
        self,
        emotional_state: EmotionalState,
        action_type: str,
        npc_id: Optional[str] = None,
        intensity: float = 1.0,
    ) -> EmotionalState:
        """
        Update emotional state based on player action.

        Args:
            emotional_state: Current emotional state
            action_type: Type of player action (e.g., "help", "insult", "gift")
            npc_id: NPC ID for history tracking
            intensity: Action intensity multiplier (0.0 to 2.0)

        Returns:
            Updated emotional state
        """
        # First apply decay
        self.apply_decay(emotional_state)

        # Get action deltas
        deltas = self._get_action_deltas(action_type, intensity)

        # Apply deltas
        emotional_state.update(**deltas)

        # Track history
        if npc_id:
            self._track_emotion(npc_id, emotional_state.get_emotion_label())

        return emotional_state

    def _get_action_deltas(
        self,
        action_type: str,
        intensity: float = 1.0,
    ) -> Dict[str, float]:
        """
        Get emotion deltas for a specific action type.

        Args:
            action_type: Type of action
            intensity: Intensity multiplier

        Returns:
            Dictionary of deltas for each dimension
        """
        # Define action effects (base values)
        action_effects = {
            # Positive actions
            "help": {"valence_delta": 0.3, "trust_delta": 0.2, "arousal_delta": 0.1},
            "gift": {"valence_delta": 0.4, "trust_delta": 0.3, "arousal_delta": 0.2},
            "compliment": {"valence_delta": 0.2, "trust_delta": 0.1, "arousal_delta": 0.1},
            "agree": {"valence_delta": 0.1, "trust_delta": 0.1},
            "defend": {"valence_delta": 0.3, "trust_delta": 0.4, "arousal_delta": 0.3},

            # Negative actions
            "insult": {"valence_delta": -0.4, "trust_delta": -0.3, "arousal_delta": 0.3, "dominance_delta": -0.2},
            "threaten": {"valence_delta": -0.5, "trust_delta": -0.5, "arousal_delta": 0.4, "dominance_delta": -0.3},
            "steal": {"valence_delta": -0.6, "trust_delta": -0.6, "arousal_delta": 0.4, "dominance_delta": -0.1},
            "ignore": {"valence_delta": -0.2, "trust_delta": -0.1, "arousal_delta": -0.1},
            "lie_detected": {"valence_delta": -0.4, "trust_delta": -0.5, "arousal_delta": 0.2},

            # Neutral/mixed
            "question": {"arousal_delta": 0.1},
            "trade": {"valence_delta": 0.1, "trust_delta": 0.05},
            "negotiate": {"arousal_delta": 0.2, "dominance_delta": 0.1},
        }

        # Get base deltas
        deltas = action_effects.get(action_type, {})

        # Apply intensity multiplier
        return {k: v * intensity for k, v in deltas.items()}

    def _track_emotion(self, npc_id: str, emotion_label: str):
        """Track emotion in history."""
        if npc_id not in self.emotion_history:
            self.emotion_history[npc_id] = []

        self.emotion_history[npc_id].append((time.time(), emotion_label))

        # Keep only last 100 entries
        if len(self.emotion_history[npc_id]) > 100:
            self.emotion_history[npc_id] = self.emotion_history[npc_id][-100:]

    def get_emotion_history(self, npc_id: str, limit: int = 10) -> List[Tuple[float, str]]:
        """
        Get recent emotion history for NPC.

        Args:
            npc_id: NPC ID
            limit: Maximum number of entries to return

        Returns:
            List of (timestamp, emotion_label) tuples
        """
        return self.emotion_history.get(npc_id, [])[-limit:]

    def generate_emotion_context(self, emotional_state: EmotionalState) -> str:
        """
        Generate natural language context about emotional state.

        Used to inject into LLM prompts for emotion-aware responses.

        Args:
            emotional_state: Current emotional state

        Returns:
            Natural language description of emotional state
        """
        emotion = emotional_state.get_emotion_label()
        tone = emotional_state.get_tone()
        trust_level = emotional_state.trust

        # Trust descriptors
        if trust_level < 0.2:
            trust_desc = "deeply distrusts the player"
        elif trust_level < 0.4:
            trust_desc = "is wary of the player"
        elif trust_level < 0.6:
            trust_desc = "is neutral toward the player"
        elif trust_level < 0.8:
            trust_desc = "trusts the player somewhat"
        else:
            trust_desc = "deeply trusts the player"

        # Arousal descriptors
        if emotional_state.arousal < 0.3:
            arousal_desc = "calm"
        elif emotional_state.arousal < 0.6:
            arousal_desc = "moderately energized"
        else:
            arousal_desc = "highly energized"

        # Build context
        context = f"The NPC is feeling {emotion} ({arousal_desc}). "
        context += f"They {trust_desc}. "
        context += f"Their dialogue should have a {tone} tone."

        return context

    def get_emotion_modifiers(self, emotional_state: EmotionalState) -> Dict[str, any]:
        """
        Get emotion-based modifiers for game mechanics.

        Args:
            emotional_state: Current emotional state

        Returns:
            Dictionary of modifiers (e.g., price multipliers, quest difficulty)
        """
        trust = emotional_state.trust
        valence = emotional_state.valence

        # Price modifier (merchants)
        # Low trust = higher prices, high trust = lower prices
        price_multiplier = 1.0 + (0.5 - trust) * 0.5  # Range: 0.75x to 1.25x

        # Quest difficulty modifier
        # Positive emotion + high trust = easier quests
        # Negative emotion + low trust = harder quests
        difficulty_modifier = 1.0 - (valence * 0.2) - (trust - 0.5) * 0.3  # Range: 0.5x to 1.5x

        # Hint generosity (how helpful NPC is)
        hint_generosity = trust * 0.5 + (valence + 1.0) * 0.25  # Range: 0.0 to 1.0

        return {
            "price_multiplier": round(price_multiplier, 2),
            "quest_difficulty_modifier": round(difficulty_modifier, 2),
            "hint_generosity": round(hint_generosity, 2),
        }

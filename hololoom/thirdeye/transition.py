"""
Transition - Smooth concept transitions for 3rdEye.

Handles the visual morphing between concepts as conversation topics evolve.
Transitions are chosen based on semantic distance and concept types.

Created: 2025-11-30
Author: HoloLoom Team
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TransitionType(Enum):
    """Types of visual transitions between concepts."""

    # Smooth morphs (close semantic distance)
    MORPH = "morph"                  # 400-800ms, shape interpolation
    BLEND = "blend"                  # 300-500ms, color/opacity blend

    # Discrete transitions (far semantic distance)
    DISSOLVE_EMERGE = "dissolve_emerge"  # 600-1000ms, fade out/fade in
    COLLAPSE_EXPAND = "collapse_expand"  # 500-800ms, shrink/grow

    # Special transitions
    PULSE = "pulse"                  # 200ms, emphasis without change
    BRANCH = "branch"                # 500ms, split into multiple
    MERGE = "merge"                  # 500ms, combine multiple into one
    ZOOM = "zoom"                    # 400ms, camera moves, not concept

    # Instant (no animation)
    CUT = "cut"                      # 0ms, immediate switch


@dataclass
class TransitionConfig:
    """Configuration for transition timing and easing."""

    duration_ms: int = 500
    easing: str = "easeInOutQuad"    # CSS/GSAP easing function
    delay_ms: int = 0
    stagger_ms: int = 50             # For multi-element transitions

    # Visual parameters
    fade_overlap: float = 0.3        # How much fade in/out overlap (0-1)
    scale_factor: float = 1.2        # Peak scale during morph
    blur_peak: float = 2.0           # Peak blur pixels during dissolve


@dataclass
class Transition:
    """
    A transition between concepts.

    Encapsulates the visual change from one concept to another,
    including timing, easing, and any intermediate states.
    """

    transition_type: TransitionType
    config: TransitionConfig = field(default_factory=TransitionConfig)

    # Source and target concepts (by ID)
    from_concept_id: str | None = None
    to_concept_id: str = ""

    # Metadata
    semantic_distance: float = 0.0   # 0.0-1.0, how different the concepts are
    triggered_by: str = "message"    # What caused this transition

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON for WebSocket streaming."""
        return {
            "type": self.transition_type.value,
            "duration_ms": self.config.duration_ms,
            "easing": self.config.easing,
            "delay_ms": self.config.delay_ms,
            "stagger_ms": self.config.stagger_ms,
            "from_concept_id": self.from_concept_id,
            "to_concept_id": self.to_concept_id,
            "semantic_distance": self.semantic_distance,
            "triggered_by": self.triggered_by,
            "visual": {
                "fade_overlap": self.config.fade_overlap,
                "scale_factor": self.config.scale_factor,
                "blur_peak": self.config.blur_peak,
            }
        }


class TransitionCalculator:
    """
    Calculate optimal transitions between concepts.

    Uses semantic distance and concept types to choose the most
    appropriate visual transition that conveys meaning.
    """

    # Thresholds for transition type selection
    MORPH_THRESHOLD = 0.3      # Below this = morph
    BLEND_THRESHOLD = 0.5      # Below this = blend
    DISSOLVE_THRESHOLD = 0.7   # Below this = dissolve, above = cut

    # Duration scaling
    MIN_DURATION_MS = 200
    MAX_DURATION_MS = 1200
    BASE_DURATION_MS = 500

    def __init__(self):
        """Initialize the transition calculator."""
        self._embedder = None  # Lazy load embedder if needed

    def calculate(
        self,
        current_embedding: list[float] | None,
        new_embedding: list[float] | None,
        current_type: str | None = None,
        new_type: str | None = None,
    ) -> Transition:
        """
        Calculate the optimal transition based on semantic distance.

        Args:
            current_embedding: Embedding of current concept (or None if first)
            new_embedding: Embedding of new concept
            current_type: Type of current concept
            new_type: Type of new concept

        Returns:
            Transition with appropriate type and timing
        """
        # First concept - fade in
        if current_embedding is None:
            return Transition(
                transition_type=TransitionType.DISSOLVE_EMERGE,
                config=TransitionConfig(duration_ms=600, easing="easeOut"),
                semantic_distance=1.0,
                triggered_by="initial",
            )

        # Calculate semantic distance
        distance = self._semantic_distance(current_embedding, new_embedding)

        # Choose transition type based on distance
        transition_type, base_duration = self._choose_transition_type(
            distance, current_type, new_type
        )

        # Scale duration by distance (closer = faster)
        duration_ms = self._scale_duration(distance, base_duration)

        # Choose easing based on transition type
        easing = self._choose_easing(transition_type, distance)

        config = TransitionConfig(
            duration_ms=duration_ms,
            easing=easing,
            fade_overlap=0.3 if distance < 0.5 else 0.1,
            scale_factor=1.1 if distance < 0.3 else 1.0,
        )

        return Transition(
            transition_type=transition_type,
            config=config,
            semantic_distance=distance,
            triggered_by="message",
        )

    def _semantic_distance(
        self,
        embedding_a: list[float] | None,
        embedding_b: list[float] | None,
    ) -> float:
        """Calculate cosine distance between embeddings (0=identical, 1=orthogonal)."""
        if embedding_a is None or embedding_b is None:
            return 1.0

        if len(embedding_a) != len(embedding_b):
            # Different dimensions - use first N that match
            min_len = min(len(embedding_a), len(embedding_b))
            embedding_a = embedding_a[:min_len]
            embedding_b = embedding_b[:min_len]

        if not embedding_a:
            return 1.0

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding_a, embedding_b))
        norm_a = math.sqrt(sum(a * a for a in embedding_a))
        norm_b = math.sqrt(sum(b * b for b in embedding_b))

        if norm_a == 0 or norm_b == 0:
            return 1.0

        similarity = dot_product / (norm_a * norm_b)

        # Convert similarity (-1 to 1) to distance (0 to 1)
        distance = (1.0 - similarity) / 2.0
        return max(0.0, min(1.0, distance))

    def _choose_transition_type(
        self,
        distance: float,
        current_type: str | None,
        new_type: str | None,
    ) -> tuple[TransitionType, int]:
        """Choose transition type based on distance and types."""

        # Special case: same type, close distance = morph
        if current_type == new_type and distance < self.MORPH_THRESHOLD:
            return TransitionType.MORPH, 400

        # Very close concepts = smooth morph
        if distance < self.MORPH_THRESHOLD:
            return TransitionType.MORPH, 500

        # Moderately close = blend
        if distance < self.BLEND_THRESHOLD:
            return TransitionType.BLEND, 400

        # Moderate distance = dissolve/emerge
        if distance < self.DISSOLVE_THRESHOLD:
            return TransitionType.DISSOLVE_EMERGE, 800

        # Far distance = collapse/expand (more dramatic)
        return TransitionType.COLLAPSE_EXPAND, 700

    def _scale_duration(self, distance: float, base_duration: int) -> int:
        """Scale duration based on semantic distance."""
        # Closer = faster, farther = slower
        scale = 0.6 + (distance * 0.8)  # 0.6 to 1.4
        duration = int(base_duration * scale)
        return max(self.MIN_DURATION_MS, min(self.MAX_DURATION_MS, duration))

    def _choose_easing(self, transition_type: TransitionType, distance: float) -> str:
        """Choose easing function based on transition type."""
        if transition_type == TransitionType.MORPH:
            return "easeInOutQuad" if distance < 0.2 else "easeInOutCubic"
        elif transition_type == TransitionType.BLEND:
            return "easeInOutSine"
        elif transition_type == TransitionType.DISSOLVE_EMERGE:
            return "easeInOutQuart"
        elif transition_type == TransitionType.COLLAPSE_EXPAND:
            return "easeOutBack"  # Slight overshoot for drama
        else:
            return "easeInOutQuad"


# Singleton calculator for convenience
_calculator = TransitionCalculator()


def calculate_transition(
    current_embedding: list[float] | None,
    new_embedding: list[float] | None,
    current_type: str | None = None,
    new_type: str | None = None,
) -> Transition:
    """
    Calculate the optimal transition between concepts.

    Convenience function using the singleton calculator.
    """
    return _calculator.calculate(
        current_embedding, new_embedding, current_type, new_type
    )


def create_emphasis_transition(duration_ms: int = 200) -> Transition:
    """Create a pulse transition for emphasis (no concept change)."""
    return Transition(
        transition_type=TransitionType.PULSE,
        config=TransitionConfig(
            duration_ms=duration_ms,
            easing="easeInOutQuad",
            scale_factor=1.15,
        ),
        semantic_distance=0.0,
        triggered_by="emphasis",
    )


def create_branch_transition(
    parent_id: str,
    child_ids: list[str],
    duration_ms: int = 500,
) -> Transition:
    """Create a branch transition (one concept splits into many)."""
    return Transition(
        transition_type=TransitionType.BRANCH,
        config=TransitionConfig(
            duration_ms=duration_ms,
            easing="easeOutCubic",
            stagger_ms=100,
        ),
        from_concept_id=parent_id,
        to_concept_id=child_ids[0] if child_ids else "",
        semantic_distance=0.5,
        triggered_by="branch",
    )


def create_merge_transition(
    source_ids: list[str],
    target_id: str,
    duration_ms: int = 500,
) -> Transition:
    """Create a merge transition (many concepts combine into one)."""
    return Transition(
        transition_type=TransitionType.MERGE,
        config=TransitionConfig(
            duration_ms=duration_ms,
            easing="easeInCubic",
            stagger_ms=50,
        ),
        from_concept_id=source_ids[0] if source_ids else None,
        to_concept_id=target_id,
        semantic_distance=0.5,
        triggered_by="merge",
    )

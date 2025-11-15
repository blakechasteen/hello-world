"""
Elle Core - User Models

Defines user state and intent representation.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EnergyLevel(Enum):
    """User's current energy level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FocusState(Enum):
    """User's current focus/flow state."""
    SCATTERED = "scattered"   # Distracted, no clear direction
    BLOCKED = "blocked"       # Wants to focus but something's preventing it
    SHALLOW = "shallow"       # Light attention
    DEEP = "deep"            # Flow state, deep work


class MoodState(Enum):
    """User's current mood."""
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    CALM = "calm"
    ENGAGED = "engaged"


@dataclass
class UserState:
    """
    Observed state of the user.

    This is inferred from sensors, context, or explicitly set.
    """
    energy: EnergyLevel = EnergyLevel.MEDIUM
    focus: FocusState = FocusState.SHALLOW
    mood: MoodState = MoodState.NEUTRAL

    def in_flow(self) -> bool:
        """Check if user is in a flow state that should be protected."""
        return self.focus == FocusState.DEEP

    def needs_help(self) -> bool:
        """Check if user might benefit from intervention."""
        return (
            self.focus == FocusState.BLOCKED
            or (self.energy == EnergyLevel.LOW and self.mood == MoodState.FRUSTRATED)
        )

    def __str__(self) -> str:
        return f"Energy: {self.energy.value}, Focus: {self.focus.value}, Mood: {self.mood.value}"


@dataclass
class UserIntent:
    """
    User's stated or inferred intent.

    This can come from:
    - Explicit voice command
    - Gesture
    - Contextual inference (e.g., walking to shed with tools)
    """
    description: str  # Natural language description
    explicit: bool = False  # True if user stated this, False if inferred
    confidence: float = 1.0  # 0.0-1.0, how certain we are about this intent

    def __str__(self) -> str:
        prefix = "Intent (explicit)" if self.explicit else f"Intent (inferred, {self.confidence:.0%})"
        return f"{prefix}: {self.description}"

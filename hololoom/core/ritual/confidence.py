"""
Confidence Model
================

Tracks agent confidence as a value in [0.0, 1.0] that responds to signals,
decays over time with domain-specific rates, and weights cross-domain transfers.

Signals are additive (clamped). Decay is exponential with domain-tuned lambda.
Domain adjacency provides transfer weights for cross-domain confidence.
"""

import math
from dataclasses import dataclass, field
from enum import Enum


class ConfidenceSignal(Enum):
    """
    Discrete events that move confidence up or down.
    Values are the delta applied to the confidence score.
    """
    TOOL_SUCCESS = +0.08
    TOOL_FAILURE = -0.15
    RETRIEVAL_HIT = +0.05
    RETRIEVAL_MISS = -0.08
    CONTRADICTION = -0.20
    REFLECTION_PASS = +0.10
    ESCALATION_USED = -0.25
    HUMAN_CORRECTION = -0.30


# Domain-specific decay rates (lambda for exponential decay).
# Higher = faster decay = knowledge goes stale quickly.
DOMAIN_DECAY_RATES: dict[str, float] = {
    "coding": 0.001,       # slow — code state is stable
    "beekeeping": 0.003,   # medium — seasonal context shifts
    "farming": 0.003,
    "weather": 0.020,      # fast — rapidly stale
    "universal": 0.002,
}

# Cross-domain adjacency weights.
# When an agent's home domain differs from the task domain,
# its confidence is weighted by this factor.
DOMAIN_ADJACENCY: dict[tuple[str, str], float] = {
    ("farming", "beekeeping"): 0.7,
    ("beekeeping", "farming"): 0.7,
    ("farming", "weather"): 0.8,
    ("weather", "farming"): 0.5,
}

# Default weight when domains have no adjacency entry
DEFAULT_ADJACENCY_WEIGHT = 0.4


@dataclass
class ConfidenceModel:
    """
    Tracks and updates agent confidence with signal-based deltas,
    time-based decay, and domain-aware weighting.
    """
    value: float = 1.0
    domain: str = "universal"
    history: list[float] = field(default_factory=list)

    def update(self, signal: ConfidenceSignal) -> float:
        """Apply a discrete signal to confidence. Clamps to [0.0, 1.0]."""
        delta = signal.value
        self.value = max(0.0, min(1.0, self.value + delta))
        self.history.append(self.value)
        return self.value

    def decay(self, elapsed_ms: int) -> float:
        """
        Apply exponential time decay: value *= exp(-lambda * t).
        Lambda is domain-specific. Elapsed time in milliseconds.
        """
        lam = DOMAIN_DECAY_RATES.get(self.domain, 0.002)
        elapsed_s = elapsed_ms / 1000.0
        self.value *= math.exp(-lam * elapsed_s)
        self.history.append(self.value)
        return self.value

    def domain_weight(self, task_domain: str) -> float:
        """
        How much this model's confidence should be weighted when
        applied to a task in a different domain.
        """
        if task_domain == self.domain:
            return 1.0
        key = (self.domain, task_domain)
        return DOMAIN_ADJACENCY.get(key, DEFAULT_ADJACENCY_WEIGHT)

    @property
    def is_low(self) -> bool:
        """Below the threshold where caution is warranted."""
        return self.value < 0.6

    @property
    def is_critical(self) -> bool:
        """Below the threshold where escalation should be considered."""
        return self.value < 0.35

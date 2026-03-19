"""
Adaptive Budget — per-query math budget based on complexity.

Simple queries get BARE (20). Complex analytical queries get RESEARCH (200).
Budget is computed from query signals using additive scoring (Design Principle 4).

Tiers map to existing PatternCard: BARE, FAST, FUSED, plus new RESEARCH.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class BudgetTier(Enum):
    """Math budget tiers — maps to PatternCard."""
    BARE = 20       # Simple lookup/similarity
    FAST = 50       # Standard queries (current default)
    FUSED = 100     # Multi-domain analytical
    RESEARCH = 200  # Deep research, multi-hop reasoning


@dataclass
class BudgetAllocation:
    """Result of budget allocation for a query."""
    tier: BudgetTier
    total_budget: int
    complexity_score: float
    signals: dict[str, float]


# Scoring weights (Design Principle 4: additive scoring with named terms)
BUDGET_WEIGHTS = {
    "query_length": 15,
    "entity_count": 20,
    "attention_entropy": 25,
    "n_intents": 15,
    "mode": 25,
}

# Mode multipliers (map PatternCard → complexity boost)
MODE_MULTIPLIERS = {
    "bare": 0.0,
    "fast": 0.3,
    "fused": 0.7,
    "research": 1.0,
}

# Band thresholds (same pattern as SOUS GPS scoring)
BUDGET_BANDS = [
    (0.75, BudgetTier.RESEARCH),
    (0.50, BudgetTier.FUSED),
    (0.25, BudgetTier.FAST),
    (0.00, BudgetTier.BARE),
]


def _normalize(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a value to [0, 1]."""
    if max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


class BudgetAllocator:
    """Allocates math budget per query based on complexity signals."""

    def __init__(self, weights: dict[str, float] = None):
        self.weights = weights or BUDGET_WEIGHTS

    def allocate(
        self,
        query_length: int = 0,
        entity_count: int = 0,
        attention_entropy: float = 0.0,
        n_intents: int = 1,
        pattern_card: str = "fast",
    ) -> BudgetAllocation:
        """Compute budget tier from query complexity signals.

        Args:
            query_length: Number of characters in query
            entity_count: Number of named entities detected
            attention_entropy: Entropy from warp compute (0-1 normalized)
            n_intents: Number of detected intents
            pattern_card: Current pattern card mode (bare/fast/fused/research)

        Returns:
            BudgetAllocation with tier, total budget, and score breakdown
        """
        signals = {
            "query_length": _normalize(query_length, 0, 500),
            "entity_count": _normalize(entity_count, 0, 10),
            "attention_entropy": max(0.0, min(1.0, attention_entropy)),
            "n_intents": _normalize(n_intents, 1, 4),
            "mode": MODE_MULTIPLIERS.get(pattern_card.lower(), 0.3),
        }

        # Additive scoring (Design Principle 4)
        total_weight = sum(self.weights.values())
        score = sum(
            self.weights.get(name, 0) * value
            for name, value in signals.items()
        ) / total_weight

        # Band mapping
        tier = BudgetTier.BARE
        for threshold, budget_tier in BUDGET_BANDS:
            if score >= threshold:
                tier = budget_tier
                break

        return BudgetAllocation(
            tier=tier,
            total_budget=tier.value,
            complexity_score=round(score, 4),
            signals={k: round(v, 4) for k, v in signals.items()},
        )


# Singleton allocator
_allocator: BudgetAllocator | None = None


def get_allocator() -> BudgetAllocator:
    """Get or create the global budget allocator."""
    global _allocator
    if _allocator is None:
        _allocator = BudgetAllocator()
    return _allocator

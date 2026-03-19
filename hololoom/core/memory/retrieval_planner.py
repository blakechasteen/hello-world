"""
Retrieval Planner — Intent-aware query planning (borrowed from SimpleMem).

Classifies query complexity to determine how deep the 4-level cascade
should go. Simple lookups skip expensive Level 3/4 entirely.

Integration: optional parameter on LiteMemoryBus.__init__.
"""

from dataclasses import dataclass
from enum import Enum

from .bus import MemoryQuery

__all__ = ["RetrievalComplexity", "RetrievalPlan", "RetrievalPlanner"]


class RetrievalComplexity(Enum):
    """How complex the retrieval task is."""
    POINT = "point"                # Single fact → L1-2 only
    NEIGHBORHOOD = "neighborhood"  # Related facts → L1-3
    EXPLORATORY = "exploratory"    # Open-ended → L1-4 + graph expand


@dataclass
class RetrievalPlan:
    """What the cascade should do for this query."""
    complexity: RetrievalComplexity
    max_cascade_level: int     # 2, 3, or 4
    expand_graph: bool         # 1-hop expansion in L3
    max_results: int           # Adaptive based on complexity
    token_budget: int | None = None  # Override config budget if set


class RetrievalPlanner:
    """Classify query intent to determine retrieval scope.

    Heuristic-only — no LLM needed. Can be subclassed for smarter planning.
    """

    def plan(self, q: MemoryQuery) -> RetrievalPlan:
        """Produce a retrieval plan based on query shape.

        Rules:
        - entity_ids present → POINT (direct lookup, L1-2)
        - structured filters only (no free text) → POINT
        - intent + structured constraints → NEIGHBORHOOD (L1-3)
        - free-text only → EXPLORATORY (all 4 levels)
        """
        # POINT: entity_ids present = direct lookup
        if q.entity_ids:
            return RetrievalPlan(
                RetrievalComplexity.POINT, 2, False, q.max_results,
            )

        has_structured = bool(
            q.entity_type or q.memory_type or q.time_after or q.time_before
        )

        # POINT: structured filters only, no free-text intent
        if has_structured and not q.intent:
            return RetrievalPlan(
                RetrievalComplexity.POINT, 2, False, q.max_results,
            )

        # NEIGHBORHOOD: intent + structured constraints (narrowed search)
        if q.intent and has_structured:
            return RetrievalPlan(
                RetrievalComplexity.NEIGHBORHOOD, 3, True, q.max_results,
            )

        # EXPLORATORY: free-text only, no structural constraints
        return RetrievalPlan(
            RetrievalComplexity.EXPLORATORY, 4, True, q.max_results,
        )

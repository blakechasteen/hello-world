"""Hive scoring terms — pure functions for additive scoring.

Each term takes a context dict (from inspection_to_context) and returns
a float in [0, 1]. Weights determine maximum contribution to composite.
"""

from typing import Any, Dict, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hololoom.core.memory.scored_observation import ScoringTerm
from .models import HiveInspection


def inspection_to_context(inspection: HiveInspection) -> Dict[str, Any]:
    """Convert a HiveInspection to a scoring context dict."""
    return {
        "varroa_per_100": inspection.varroa_count,
        "queen_seen": inspection.queen_seen,
        "eggs_seen": inspection.eggs_seen,
        "brood_pattern": inspection.brood_pattern,
        "stores_level": inspection.honey_stores,
        "queen_cells": inspection.queen_cells,
        "disease_signs": inspection.disease_signs,
        "temperament": inspection.temperament,
    }


# =============================================================================
# Term Functions (pure, independently testable)
# =============================================================================

def varroa_score(ctx: Dict[str, Any]) -> float:
    """Varroa mite load. Higher count = higher urgency."""
    count = ctx.get("varroa_per_100", 0)
    if count >= 8:
        return 1.0
    elif count >= 5:
        return 0.7
    elif count >= 3:
        return 0.4
    return 0.0


def queenless_score(ctx: Dict[str, Any]) -> float:
    """Queen absence indicator. No queen + no eggs = emergency."""
    queen = ctx.get("queen_seen", True)
    eggs = ctx.get("eggs_seen", True)
    if not queen and not eggs:
        return 1.0
    elif not queen:
        return 0.5
    return 0.0


def queen_cell_score(ctx: Dict[str, Any]) -> float:
    """Swarm / supersedure risk from queen cells."""
    cells = ctx.get("queen_cells", 0)
    if cells >= 5:
        return 1.0
    elif cells >= 2:
        return 0.7
    elif cells >= 1:
        return 0.4
    return 0.0


def stores_score(ctx: Dict[str, Any]) -> float:
    """Honey/pollen stores. Low stores = needs feeding."""
    level = ctx.get("stores_level", "adequate")
    return {"light": 0.8, "adequate": 0.2, "heavy": 0.0}.get(level, 0.3)


def disease_score(ctx: Dict[str, Any]) -> float:
    """Disease indicators. Any signs = concern."""
    signs = ctx.get("disease_signs", [])
    if not signs:
        return 0.0
    # More diseases = more concern, cap at 1.0
    return min(1.0, len(signs) * 0.5)


def brood_score(ctx: Dict[str, Any]) -> float:
    """Brood pattern quality. Spotty/absent = issue."""
    pattern = ctx.get("brood_pattern", "solid")
    return {"solid": 0.0, "spotty": 0.6, "absent": 1.0}.get(pattern, 0.3)


# =============================================================================
# Term Registry
# =============================================================================

HIVE_TERMS: List[ScoringTerm] = [
    ScoringTerm("w_varroa", 30, varroa_score),
    ScoringTerm("w_queenless", 25, queenless_score),
    ScoringTerm("w_queen_cells", 20, queen_cell_score),
    ScoringTerm("w_stores", 15, stores_score),
    ScoringTerm("w_disease", 25, disease_score),
    ScoringTerm("w_brood", 10, brood_score),
]

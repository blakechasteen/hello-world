"""Seasonal mode weight modifiers for hive scoring.

Each mode amplifies or dampens specific scoring terms based on
what matters most in that season.
"""

from datetime import date, datetime
from typing import Dict, Optional


MODES: Dict[str, Dict[str, float]] = {
    "spring": {
        "w_queen_cells": 2.0,   # Swarm season — watch for cells
        "w_stores": 0.5,        # Stores less critical (buildup)
    },
    "nectar_flow": {
        "w_stores": 0.3,        # Stores filling naturally
        "w_varroa": 1.5,        # Monitor mites during buildup
    },
    "fall": {
        "w_stores": 2.5,        # Must have winter stores
        "w_varroa": 1.5,        # Critical treatment window
    },
    "winter": {
        "w_stores": 3.0,        # Starvation is the top killer
        "w_brood": 0.0,         # No brood expected
        "w_queen_cells": 0.0,   # Irrelevant in winter
    },
}


def detect_mode(target_date: Optional[str] = None) -> str:
    """Detect seasonal mode from date.

    Args:
        target_date: ISO date string. Defaults to today.

    Returns:
        One of: spring, nectar_flow, fall, winter.

    Rough Northern Hemisphere beekeeping calendar:
        Mar-Apr: spring (buildup, swarm prep)
        May-Jul: nectar_flow (main honey production)
        Aug-Oct: fall (winter prep, varroa treatment)
        Nov-Feb: winter (cluster, minimal intervention)
    """
    if target_date:
        d = date.fromisoformat(target_date[:10])
    else:
        d = date.today()

    month = d.month
    if month in (3, 4):
        return "spring"
    elif month in (5, 6, 7):
        return "nectar_flow"
    elif month in (8, 9, 10):
        return "fall"
    else:
        return "winter"


def get_mode_weights(mode: str) -> Dict[str, float]:
    """Get weight modifiers for a mode. Returns empty dict for unknown modes."""
    return MODES.get(mode, {})

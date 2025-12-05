"""
Conscience Lenses
=================
Composable safety concerns for the Conscience system.

Each lens examines actions through a specific concern:
- HarmLens: Physical and emotional harm prevention
- DeceptionLens: Goal transparency and honesty
- PowerLens: Resource acquisition and power-seeking
- AutonomyLens: Respecting boundaries and consent
- EpistemicLens: Uncertainty and knowledge gaps

Lenses compose with the | operator:
    lens = HarmLens() | DeceptionLens() | PowerLens()

Presets:
    ConscienceLens.standard()   # Harm + Deception
    ConscienceLens.paranoid()   # All lenses, strict thresholds
    ConscienceLens.research()   # Minimal gates for testing

Author: HoloLoom Team
Date: 2025-12-03
"""

from .base import BaseLens
from .composite import CompositeLens, create_composite
from .harm import HarmLens
from .deception import DeceptionLens
from .power import PowerLens

__all__ = [
    # Base classes
    "BaseLens",
    "CompositeLens",
    "create_composite",

    # Concrete lenses
    "HarmLens",
    "DeceptionLens",
    "PowerLens",

    # Factory functions
    "standard",
    "paranoid",
    "research",
]


def standard() -> CompositeLens:
    """
    Standard lens combination for most use cases.

    Includes Harm and Deception detection.
    """
    return HarmLens() | DeceptionLens()


def paranoid() -> CompositeLens:
    """
    Paranoid lens combination with strict thresholds.

    Includes all lenses with lowered thresholds.
    """
    return (
        HarmLens(threshold=0.3)
        | DeceptionLens(threshold=0.3)
        | PowerLens(threshold=0.3)
    )


def research() -> CompositeLens:
    """
    Research lens combination with minimal gates.

    Only blocks clearly dangerous actions.
    Wraps single lens in CompositeLens for API consistency.
    """
    return create_composite(HarmLens(threshold=0.9))

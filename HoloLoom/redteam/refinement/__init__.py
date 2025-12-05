"""Attack refinement and quality tracking for CARTS (Collaborative Attack Refinement Tracking System).

This module provides quality trajectory tracking for adversarial attack refinement,
enabling detection of plateaus, regressions, and successful patterns.

**Status**: Production Ready (November 2025)
**Location**: `HoloLoom/redteam/refinement/`
**Performance**: <5ms per quality record

Key Classes:
- QualityTrajectoryTracker: Main trajectory tracking orchestrator
- StrategyTrajectory: Quality evolution for a single attack strategy
- RefinementPattern: Discovered patterns for improving attacks
"""

from .quality_trajectory import (
    QualityTrajectoryTracker,
    StrategyTrajectory,
    RefinementPattern
)

__all__ = [
    'QualityTrajectoryTracker',
    'StrategyTrajectory',
    'RefinementPattern'
]

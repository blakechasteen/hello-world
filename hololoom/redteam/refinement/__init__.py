"""Attack refinement and quality tracking for CARTS (Collaborative Attack Refinement Tracking System).

This module provides quality trajectory tracking for adversarial attack refinement,
enabling detection of plateaus, regressions, and successful patterns.

**Status**: Production Ready (November 2025)
**Location**: `HoloLoom/redteam/refinement/`
**Performance**: <5ms per quality record

Key Classes (Quality Trajectory):
- QualityTrajectoryTracker: Main trajectory tracking orchestrator
- StrategyTrajectory: Quality evolution for a single attack strategy
- RefinementPattern: Discovered patterns for improving attacks

Key Classes (Attack Refinement - Wave 2):
- AttackRefiner: Main refinement orchestrator with async API
- AttackRefinementStrategy: Enum of 5 strategies (OBFUSCATE, MUTATE, VERIFY, ELEGANCE, RECURSIVE)
- AttackQualityMetrics: Multi-dimensional quality scoring (effectiveness, stealth, reliability, elegance)
- AttackRefinementResult: Complete result container with provenance
"""

from .quality_trajectory import (
    QualityTrajectoryTracker,
    StrategyTrajectory,
    RefinementPattern
)

from .attack_refinement import (
    AttackRefiner,
    AttackRefinementStrategy,
    AttackQualityMetrics,
    AttackRefinementResult
)

__all__ = [
    # Quality Trajectory
    'QualityTrajectoryTracker',
    'StrategyTrajectory',
    'RefinementPattern',
    # Attack Refinement (Wave 2)
    'AttackRefiner',
    'AttackRefinementStrategy',
    'AttackQualityMetrics',
    'AttackRefinementResult'
]

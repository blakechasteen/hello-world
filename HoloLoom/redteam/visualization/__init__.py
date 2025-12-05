"""Tufte-style visualizations for CARTS red team analytics.

Provides production-grade visualization of attack trajectories, strategy effectiveness,
and adversarial learning patterns using Tufte principles:
- Maximize data-ink ratio (show meaning first)
- Small multiples for comparison
- High information density
- Zero external dependencies (pure HTML/CSS/SVG)

Example:
    >>> from HoloLoom.redteam.visualization import render_attack_trajectory
    >>>
    >>> strategies = ["prompt_injection", "jailbreak", "context_overflow"]
    >>> success_rates = [0.65, 0.42, 0.28]
    >>> attack_counts = [10, 10, 10]
    >>> bypass_counts = [6, 4, 3]
    >>>
    >>> html = render_attack_trajectory(
    ...     strategies, success_rates, attack_counts, bypass_counts,
    ...     title="Attack Effectiveness Trajectory"
    ... )
"""

from .attack_trajectory import (
    AttackPoint,
    StrategyMetrics,
    AttackTrajectoryRenderer,
    render_attack_trajectory,
    AnomalyType,
)

__all__ = [
    "AttackPoint",
    "StrategyMetrics",
    "AttackTrajectoryRenderer",
    "render_attack_trajectory",
    "AnomalyType",
]

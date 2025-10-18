"""
Convergence - Decision Collapse
================================
The engine that collapses continuous probabilities to discrete decisions.

Exports:
- ConvergenceEngine: Main decision engine
- CollapseStrategy: Enum of collapse strategies
- CollapseResult: Result dataclass
- ThompsonBandit: Thompson Sampling bandit
- create_convergence_engine: Factory function
"""

from .engine import (
    ConvergenceEngine,
    CollapseStrategy,
    CollapseResult,
    ThompsonBandit,
    create_convergence_engine
)

__all__ = [
    "ConvergenceEngine",
    "CollapseStrategy",
    "CollapseResult",
    "ThompsonBandit",
    "create_convergence_engine"
]

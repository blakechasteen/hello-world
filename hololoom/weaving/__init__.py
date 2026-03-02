"""
HoloLoom Weaving Architecture
=============================
Modular weaving components for the HoloLoom orchestrator.

This package contains:
- stages/: Individual processing stages
- strategies/: Complexity-based execution strategies
- protocols.py: Protocol definitions
"""

from .protocols import (
    WeavingStageProtocol,
    ComplexityStrategyProtocol,
    StageResult,
)

__all__ = [
    "WeavingStageProtocol",
    "ComplexityStrategyProtocol",
    "StageResult",
]
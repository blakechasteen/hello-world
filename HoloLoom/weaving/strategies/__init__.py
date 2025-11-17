"""
Complexity Strategies - Progressive 3-5-7-9 Processing
========================================================

This module contains strategy classes for the progressive complexity modes
in HoloLoom's mythRL architecture (LITE/FAST/FULL/RESEARCH).

**Implemented: November 2025** - Extracted from weaving_orchestrator.py

**Progressive Complexity:**
- LITE (3 steps): <50ms - Simple lookups, cached queries
- FAST (5 steps): <150ms - Standard queries, real-time apps
- FULL (7 steps): <300ms - Complex queries, production systems
- RESEARCH (9 steps): No limit - Research, debugging, quality maximization

Author: Claude Code (Elegance Pass refactoring)
"""

from .complexity_strategy import (
    ComplexityStrategy,
    LiteStrategy,
    FastStrategy,
    FullStrategy,
    ResearchStrategy,
    create_complexity_strategy
)

__all__ = [
    'ComplexityStrategy',
    'LiteStrategy',
    'FastStrategy',
    'FullStrategy',
    'ResearchStrategy',
    'create_complexity_strategy',
]

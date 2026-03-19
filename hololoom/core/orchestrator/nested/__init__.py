"""
HoloLoom Phase 6: Nested Learning Meta-Architecture

This module implements Google Research's Nested Learning paradigm, treating
architecture and optimization as hierarchical learning problems at different frequencies.

Core Components:
- Ultra-fast optimizer: Sub-query routing, token-level decisions (1-10ms)
- Medium-frequency optimizer: Step selection, retrieval strategy learning (1-5s)
- Very slow optimizer: Architectural evolution (manual/rare)
- Meta-policy: Learnable step selection
- Recurrent orchestrator: Hope-style hidden state across queries
- Continuum Memory System: Unified memory spectrum
- Self-optimizing memory: Learned retrieval strategies

Philosophy:
"Architecture and optimization are not separate—they are nested learning problems
at different frequencies."

References:
- Google Research: Nested Learning (2024)
- HoloLoom Phase 5: Recursive Learning
"""

from hololoom.orchestrator.nested.hierarchy import (
    NestedLearningHierarchy,
    OptimizationLevel,
)
from hololoom.orchestrator.nested.ultra_fast import (
    SubQueryRouter,
    UltraFastOptimizer,
    WorkingMemory,
)

__all__ = [
    # Ultra-fast components
    "UltraFastOptimizer",
    "WorkingMemory",
    "SubQueryRouter",

    # Hierarchy
    "NestedLearningHierarchy",
    "OptimizationLevel",
]

__version__ = "6.0.0-alpha"

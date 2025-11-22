"""
Memory Symphony - Unified Memory Coordination
==============================================

Orchestrates across 7 memory systems for optimal performance:
- Knowledge Graph (symbolic)
- Vector Memory (semantic)
- Query Cache (100x speedup)
- Hot Pattern Feedback (adaptation)
- Awareness Graph (activation)
- Spring Dynamics (physics)
- Multi-Wave Engine (temporal)

Author: Claude Code
Date: 2025-11-22
"""

# Protocol exports (always available)
from .protocol import (
    MemoryStrategy,
    MemorySystem,
    MemoryQuery,
    MemoryResult,
    MemoryCoordinationResult,
    MemoryPerformanceMetrics,
    MemoryConductorProtocol,
    MemorySystemProtocol,
    StrategySelectionCriteria,
    CoordinationPlan,
)

__all__ = [
    # Protocols
    "MemoryStrategy",
    "MemorySystem",
    "MemoryQuery",
    "MemoryResult",
    "MemoryCoordinationResult",
    "MemoryPerformanceMetrics",
    "MemoryConductorProtocol",
    "MemorySystemProtocol",
    "StrategySelectionCriteria",
    "CoordinationPlan",
    # Implementations (lazy loaded)
    "MemoryConductor",
    "create_memory_conductor",
]


def __getattr__(name):
    """Lazy loading for implementations."""
    if name == "MemoryConductor":
        from .conductor import MemoryConductor
        globals()[name] = MemoryConductor
        return MemoryConductor
    elif name == "create_memory_conductor":
        from .conductor import create_memory_conductor
        globals()[name] = create_memory_conductor
        return create_memory_conductor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

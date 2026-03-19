"""
Memory Symphony - Re-export from canonical location.

This module re-exports HoloLoom.memory.symphony for backward compatibility.
The canonical location is HoloLoom/memory/symphony/.

Author: Claude Code
Date: 2026-01-22
"""

# Re-export everything from canonical location
from hololoom.memory.symphony import (
    CoordinationPlan,
    # Implementations
    MemoryConductor,
    MemoryConductorProtocol,
    MemoryCoordinationResult,
    MemoryPerformanceMetrics,
    MemoryQuery,
    MemoryResult,
    # Protocols
    MemoryStrategy,
    MemorySystem,
    MemorySystemProtocol,
    StrategySelectionCriteria,
    create_memory_conductor,
)

__all__ = [
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
    "MemoryConductor",
    "create_memory_conductor",
]

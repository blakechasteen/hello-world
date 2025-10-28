"""
HoloLoom Canonical Protocols
=============================
Single source of truth for all protocol definitions in HoloLoom.

This package provides:
- Core types (ComplexityLevel, ProvenceTrace, MythRLResult)
- Memory protocols (MemoryStore, MemoryNavigator, PatternDetector)
- Core feature protocols (Embedder, MotifDetector, PolicyEngine)
- Routing protocols (RoutingStrategy, ExecutionEngine)
- Tool protocols (ToolExecutor, ToolRegistry)
- mythRL Shuttle protocols (PatternSelectionProtocol, DecisionEngineProtocol, etc.)

Philosophy:
- Protocols define WHAT, not HOW (interfaces, not implementations)
- All implementations are swappable via dependency injection
- Protocol-based design enables clean architecture
- No business logic in protocol definitions

Usage:
    from HoloLoom.protocols import (
        ComplexityLevel,
        MemoryStore,
        PolicyEngine,
        PatternSelectionProtocol,
        DecisionEngineProtocol
    )

Author: mythRL Team
Date: 2025-10-27 (Phase 1 Protocol Standardization - Task 1.1)
"""

# ============================================================================
# Import Core Types
# ============================================================================

from .types import (
    ComplexityLevel,
    ProvenceTrace,
    MythRLResult,
)

# ============================================================================
# Import Core Feature Protocols
# ============================================================================

from .core_features import (
    Embedder,
    MotifDetector,
    PolicyEngine,
    RoutingStrategy,
    ExecutionEngine,
    ToolRegistry,
)

# ============================================================================
# Import Memory Protocols (from core.py)
# ============================================================================

from .core import (
    MemoryNavigator,
    PatternDetector,
)

# MemoryStore is in memory.protocol to avoid circular imports
from HoloLoom.memory.protocol import MemoryStore

# ============================================================================
# Import Shuttle Protocols (from shuttle.py)
# ============================================================================

from .shuttle import (
    PatternSelectionProtocol,
    FeatureExtractionProtocol,
    WarpSpaceProtocol,
    DecisionEngineProtocol,
    ToolExecutor,
)

# ============================================================================
# Re-exports from Documentation.types for convenience
# ============================================================================

try:
    from HoloLoom.documentation.types import (
        Query, Features, Context, Response, MemoryShard,
        PolicyAction, ActionPlan, ToolCall, ToolResult, Vector
    )
    _HAS_DOC_TYPES = True
except ImportError:
    _HAS_DOC_TYPES = False


# ============================================================================
# Compatibility Aliases
# ============================================================================

# For backward compatibility with code expecting different names
MemoryBackendProtocol = MemoryStore
ToolExecutionProtocol = ToolExecutor


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # ===== Core Types =====
    'ComplexityLevel',
    'ProvenceTrace',
    'MythRLResult',

    # ===== Core Feature Protocols =====
    'Embedder',
    'MotifDetector',
    'PolicyEngine',

    # ===== Memory Protocols =====
    'MemoryStore',
    'MemoryNavigator',
    'PatternDetector',

    # ===== Routing Protocols =====
    'RoutingStrategy',
    'ExecutionEngine',

    # ===== Tool Protocols =====
    'ToolExecutor',
    'ToolRegistry',

    # ===== mythRL Shuttle Protocols =====
    'PatternSelectionProtocol',
    'FeatureExtractionProtocol',
    'WarpSpaceProtocol',
    'DecisionEngineProtocol',

    # ===== Compatibility Aliases =====
    'MemoryBackendProtocol',
    'ToolExecutionProtocol',
]

# Add Documentation types to exports if available
if _HAS_DOC_TYPES:
    __all__.extend([
        'Query', 'Features', 'Context', 'Response', 'MemoryShard',
        'PolicyAction', 'ActionPlan', 'ToolCall', 'ToolResult', 'Vector'
    ])


# ============================================================================
# Version Info
# ============================================================================

__version__ = '1.0.0'
__author__ = 'mythRL Team'
__date__ = '2025-10-27'
__status__ = 'Production - Task 1.1 Complete'

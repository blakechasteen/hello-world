"""
HoloLoom Canonical Protocols
=============================
Single source of truth for all protocol definitions in HoloLoom.

This package provides:
- Core types (ComplexityLevel, ProvenanceTrace, MythRLResult)
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
    ProvenanceTrace,
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
# Import Memory Types and Protocols
# ============================================================================

from .memory_types import (
    Memory,
    MemoryQuery,
    MemoryRetrievalResult,
    Strategy,
    QueryMode,
    shards_to_memories,
)

from .memory_protocols import (
    MemoryStore,
    MemoryNavigator,
    PatternDetector,
)

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
# Import Retrieval Protocols (from retrieval.py)
# ============================================================================

from .retrieval import (
    RetrievalStrategy,
    RetrievalResult,
    SpringActivationMetadata,
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
# MemoryBackendProtocol = MemoryStore  # Disabled to avoid circular import
ToolExecutionProtocol = ToolExecutor


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # ===== Core Types =====
    'ComplexityLevel',
    'ProvenanceTrace',
    'MythRLResult',

    # ===== Memory Types =====
    'Memory',
    'MemoryQuery',
    'MemoryRetrievalResult',
    'Strategy',
    'QueryMode',
    'shards_to_memories',

    # ===== Memory Protocols =====
    'MemoryStore',
    'MemoryNavigator',
    'PatternDetector',

    # ===== Core Feature Protocols =====
    'Embedder',
    'MotifDetector',
    'PolicyEngine',

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

    # ===== Retrieval Protocols =====
    'RetrievalStrategy',
    'RetrievalResult',
    'SpringActivationMetadata',

    # ===== Compatibility Aliases =====
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

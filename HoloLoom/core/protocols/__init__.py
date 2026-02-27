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
- Conscience protocols (ConscienceProtocol, ConscienceDecision, StepType, RiskLevel)

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
        DecisionEngineProtocol,
        # Conscience protocols (December 2025)
        ConscienceProtocol,
        ConscienceDecision,
        StepType,
        RiskLevel,
    )

Author: mythRL Team
Date: 2025-10-27 (Phase 1 Protocol Standardization - Task 1.1)
Updated: 2025-12-03 (Phase 2A Conscience Integration)
Updated: 2026-02-26 (Department Protocol extracted from departments/)
"""

# ============================================================================
# Import Core Types
# ============================================================================

from .types import (
    ComplexityLevel,
    BanditStrategy,
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
# Import Jenny UI Protocols (from jenny.py)
# ============================================================================

from .jenny import (
    CompilationStrategy,
    RenderTarget,
    JennyCompilerProtocol,
    JennyRendererProtocol,
    JennyLifecycleProtocol,
    SpecLedgerProtocol,
)

# ============================================================================
# Import Conscience Protocols (December 2025 - Phase 2A)
# ============================================================================

from .conscience import (
    # Enums
    StepType,
    RiskLevel,
    # Core types
    ConscienceDecision,
    # Protocols
    ConscienceProtocol,
    ExtendedConscienceProtocol,
    # Implementations
    NullConscience,
    # Factory functions
    create_allowed_decision,
    create_blocked_decision,
    create_review_decision,
)

# ============================================================================
# Import Department Protocol (February 2026 — extracted from departments/)
# ============================================================================

from .department import (
    # Confidence
    ConfidenceLevel,
    ConfidenceMetadata,
    # Privacy
    PrivacyLevel,
    PrivacyEnvelope,
    # Requests and Responses
    DepartmentRequest,
    DepartmentResponse,
    # Verification
    VerificationStatus,
    VerificationCheck,
    DSStarCheck,
    VerificationResult,
    # Protocols
    Department,
    DepartmentProtocol,
    # Helpers
    create_simple_request,
    create_simple_response,
    # Type Aliases
    DepartmentFactory,
    VerificationFunction,
    # Configuration
    DepartmentConfig,
    DepartmentManifest,
    # Learning Functions
    compute_learning_rate,
    should_update_now,
)

# ============================================================================
# Re-exports from Documentation.types for convenience
# ============================================================================

try:
    from HoloLoom.protocols.types import (
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
    'BanditStrategy',
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

    # ===== Jenny UI Protocols =====
    'CompilationStrategy',
    'RenderTarget',
    'JennyCompilerProtocol',
    'JennyRendererProtocol',
    'JennyLifecycleProtocol',
    'SpecLedgerProtocol',

    # ===== Conscience Protocols (December 2025) =====
    'StepType',
    'RiskLevel',
    'ConscienceDecision',
    'ConscienceProtocol',
    'ExtendedConscienceProtocol',
    'NullConscience',
    'create_allowed_decision',
    'create_blocked_decision',
    'create_review_decision',

    # ===== Department Protocol (February 2026) =====
    'ConfidenceLevel',
    'ConfidenceMetadata',
    'PrivacyLevel',
    'PrivacyEnvelope',
    'DepartmentRequest',
    'DepartmentResponse',
    'VerificationStatus',
    'VerificationCheck',
    'DSStarCheck',
    'VerificationResult',
    'Department',
    'DepartmentProtocol',
    'create_simple_request',
    'create_simple_response',
    'DepartmentFactory',
    'VerificationFunction',
    'DepartmentConfig',
    'DepartmentManifest',
    'compute_learning_rate',
    'should_update_now',

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

__version__ = '1.1.0'
__author__ = 'mythRL Team'
__date__ = '2025-12-03'
__status__ = 'Production - Phase 2A Conscience Integration'

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
    from hololoom.protocols import (
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

# ============================================================================
# Import Conscience Protocols (December 2025 - Phase 2A)
# ============================================================================
from .conscience import (
    # Core types
    ConscienceDecision,
    # Protocols
    ConscienceProtocol,
    ExtendedConscienceProtocol,
    # Implementations
    NullConscience,
    RiskLevel,
    # Enums
    StepType,
    # Factory functions
    create_allowed_decision,
    create_blocked_decision,
    create_review_decision,
)

# ============================================================================
# Import Core Feature Protocols
# ============================================================================
from .core_features import (
    Embedder,
    ExecutionEngine,
    MotifDetector,
    PolicyEngine,
    RoutingStrategy,
    ToolRegistry,
)

# ============================================================================
# Import Feedback Protocol (March 2026 — multi-temporal feedback)
# ============================================================================
from .feedback import (
    FeedbackConsumer,
    FeedbackSignal,
    FeedbackSource,
    Timescale,
    TIMESCALE_WEIGHT,
)

# ============================================================================
# Import Department Protocol (February 2026 — extracted from departments/)
# ============================================================================
from .department import (
    # Confidence
    ConfidenceLevel,
    ConfidenceMetadata,
    # Protocols
    Department,
    # Configuration
    DepartmentConfig,
    # Type Aliases
    DepartmentFactory,
    DepartmentManifest,
    DepartmentProtocol,
    # Requests and Responses
    DepartmentRequest,
    DepartmentResponse,
    DSStarCheck,
    PrivacyEnvelope,
    # Privacy
    PrivacyLevel,
    VerificationCheck,
    VerificationFunction,
    VerificationResult,
    # Verification
    VerificationStatus,
    # Learning Functions
    compute_learning_rate,
    # Helpers
    create_simple_request,
    create_simple_response,
    should_update_now,
)

# ============================================================================
# Import Jenny UI Protocols (from jenny.py)
# ============================================================================
from .jenny import (
    CompilationStrategy,
    JennyCompilerProtocol,
    JennyLifecycleProtocol,
    JennyRendererProtocol,
    RenderTarget,
    SpecLedgerProtocol,
)
from .memory_protocols import (
    MemoryNavigator,
    MemoryStore,
    PatternDetector,
)

# ============================================================================
# Import Memory Types and Protocols
# ============================================================================
from .memory_types import (
    Memory,
    MemoryQuery,
    MemoryRetrievalResult,
    QueryMode,
    Strategy,
    shards_to_memories,
)

# ============================================================================
# Import Retrieval Protocols (from retrieval.py)
# ============================================================================
from .retrieval import (
    RetrievalResult,
    RetrievalStrategy,
    SpringActivationMetadata,
)

# ============================================================================
# Import Shuttle Protocols (from shuttle.py)
# ============================================================================
from .shuttle import (
    DecisionEngineProtocol,
    FeatureExtractionProtocol,
    PatternSelectionProtocol,
    ToolExecutor,
    WarpSpaceProtocol,
)
from .types import (
    BanditStrategy,
    ComplexityLevel,
    MythRLResult,
    ProvenanceTrace,
)

# ============================================================================
# Re-exports from Documentation.types for convenience
# ============================================================================

try:
    from hololoom.protocols.types import (
        ActionPlan,
        Context,
        Features,
        MemoryShard,
        PolicyAction,
        Query,
        Response,
        ToolCall,
        ToolResult,
        Vector,
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

    # ===== Feedback Protocol (March 2026) =====
    'FeedbackSignal',
    'FeedbackSource',
    'FeedbackConsumer',
    'Timescale',
    'TIMESCALE_WEIGHT',

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

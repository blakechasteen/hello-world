"""
Ritual Package - 5-Phase Coding Ritual System
==============================================

Created: December 30, 2025
Purpose: Unified exports for the coding ritual workflow system

The ritual system orchestrates a 5-phase coding workflow:
    AWAKEN -> PLAN -> IMPLEMENT -> REVIEW -> REFLECT

This package provides:
- Event-driven workflow orchestration (ritual_events)
- Agent capability registration (agent_registration)
- Thompson Sampling phase selection (thompson_router)
- Priority-queued inter-phase messaging (message_bus)
- MI-aware context handoff (context_handoff)
- Autonomous ritual coordination (coordinator)
- Phase implementations (phases subpackage)

Example Usage:
    from ritual import (
        RitualContext,
        RitualAgentCoordinator,
        create_coordinator,
        HandoffStrategy,
    )

    # Create coordinator
    coordinator = create_coordinator()

    # Execute ritual
    result = await coordinator.delegate_ritual(
        feature_name="Add new feature",
        context_type="new_feature",
        auto_decisions=True
    )
"""

# ============================================================================
# Ritual Events - EventBus foundation
# ============================================================================

from .ritual_events import (
    # Enums
    RitualPhase,
    RitualEventType,
    DecisionType,

    # Data structures
    RitualContext,
    RitualDecision,

    # Decision constants
    PHASE_DECISION_OPTIONS,

    # Event creation
    create_ritual_event,

    # Event factories - Ritual lifecycle
    ritual_started,
    ritual_completed,
    ritual_cancelled,
    ritual_paused,
    ritual_resumed,

    # Event factories - Phase lifecycle
    phase_started,
    phase_completed,
    phase_skipped,
    phase_failed,

    # Event factories - Decisions
    decision_required,
    decision_made,

    # Event factories - Memory
    memory_stored,
    context_restored,
    context_handoff as context_handoff_event,  # Renamed to avoid conflict

    # Event factories - Learning
    learning_update,
)


# ============================================================================
# Agent Registration - Capability-based agent registry
# ============================================================================

from .agent_registration import (
    # Enums
    AgentCapability,

    # Data structures
    RitualAgentInfo,

    # Main class
    RitualAgentRegistry,

    # Constants
    PHASE_CAPABILITY_MAP,
    DEFAULT_RITUAL_AGENTS,

    # Factory functions
    create_registry,

    # Utility functions
    get_capability_for_phase,
    get_phase_for_capability,
)


# ============================================================================
# Thompson Router - Phase selection with exploration/exploitation
# ============================================================================

from .thompson_router import (
    # Enums
    SelectionStrategy,

    # Data structures
    ThompsonPrior,

    # Main class
    RitualPhaseRouter,

    # Factory functions
    create_router,
    create_thompson_router,
    create_adaptive_router,
)


# ============================================================================
# Message Bus - Priority-queued inter-phase communication
# ============================================================================

from .message_bus import (
    # Enums
    MessageType,
    MessagePriority,

    # Data structures
    PhaseMessage,

    # Type aliases
    MessageHandler,

    # Main class
    RitualMessageBus,

    # Factory
    create_message_bus,
)


# ============================================================================
# Context Handoff - MI-aware filtering between phases
# ============================================================================

from .context_handoff import (
    # Enums
    HandoffStrategy,

    # Data structures
    ContextItem,
    HandoffResult,

    # Main class
    RitualContextHandoff,

    # Factory
    create_context_handoff,

    # Constants
    PHASE_KEYWORDS,
    CAPABILITY_KEYWORDS,
)


# ============================================================================
# Coordinator - Autonomous ritual orchestration
# ============================================================================

from .coordinator import (
    # Enums
    CoordinatorState,

    # Data structures
    RitualExecutionResult,
    PhaseExecutionResult,

    # Constants
    DEFAULT_PHASE_ORDER,

    # Main class
    RitualAgentCoordinator,

    # Factory functions
    create_coordinator,
    create_minimal_coordinator,
)


# ============================================================================
# Phases Subpackage - Phase implementations
# ============================================================================

from .phases import (
    # Base classes and protocols
    PhasePreflightResult,
    PhaseResult,
    PhaseExecutionContext,
    RitualPhaseSkill,
    AbstractRitualPhase,

    # AWAKEN phase
    RestoredContext,
    AwakenPhase,
    create_awaken_phase,

    # PLAN phase
    ImplementationStep,
    SimilarPattern,
    ImplementationPlan,
    PlanPhase,
    create_plan_phase,

    # REVIEW phase
    QualityIssue,
    ImplementationDecision,
    Learning,
    ReviewResult,
    ReviewPhase,
    create_review_phase,

    # REFLECT phase
    SessionSummary,
    ReflectPhase,
    create_reflect_phase,
)


# ============================================================================
# Convenience Functions
# ============================================================================

async def quick_ritual(
    feature_name: str,
    auto_decisions: bool = True,
    context_type: str = "default"
) -> RitualExecutionResult:
    """
    Simplest way to run a ritual - one async function call.

    Example:
        result = await quick_ritual("Add user authentication")

    Args:
        feature_name: Feature being worked on
        auto_decisions: Auto-accept suggested decisions (default: True)
        context_type: Context for Thompson Sampling (default: "default")

    Returns:
        RitualExecutionResult with phases completed, decisions, etc.
    """
    coordinator = create_minimal_coordinator()
    return await coordinator.delegate_ritual(
        feature_name=feature_name,
        context_type=context_type,
        auto_decisions=auto_decisions
    )


# ============================================================================
# Package Metadata
# ============================================================================

__version__ = "1.0.0"
__author__ = "HoloLoom Team"
__created__ = "December 30, 2025"


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Package metadata
    '__version__',
    '__author__',
    '__created__',

    # ========== ritual_events ==========
    # Enums
    'RitualPhase',
    'RitualEventType',
    'DecisionType',
    # Data structures
    'RitualContext',
    'RitualDecision',
    # Decision constants
    'PHASE_DECISION_OPTIONS',
    # Event creation
    'create_ritual_event',
    # Event factories - Ritual lifecycle
    'ritual_started',
    'ritual_completed',
    'ritual_cancelled',
    'ritual_paused',
    'ritual_resumed',
    # Event factories - Phase lifecycle
    'phase_started',
    'phase_completed',
    'phase_skipped',
    'phase_failed',
    # Event factories - Decisions
    'decision_required',
    'decision_made',
    # Event factories - Memory
    'memory_stored',
    'context_restored',
    'context_handoff_event',
    # Event factories - Learning
    'learning_update',

    # ========== agent_registration ==========
    # Enums
    'AgentCapability',
    # Data structures
    'RitualAgentInfo',
    # Main class
    'RitualAgentRegistry',
    # Constants
    'PHASE_CAPABILITY_MAP',
    'DEFAULT_RITUAL_AGENTS',
    # Factory functions
    'create_registry',
    # Utility functions
    'get_capability_for_phase',
    'get_phase_for_capability',

    # ========== thompson_router ==========
    # Enums
    'SelectionStrategy',
    # Data structures
    'ThompsonPrior',
    # Main class
    'RitualPhaseRouter',
    # Factory functions
    'create_router',
    'create_thompson_router',
    'create_adaptive_router',

    # ========== message_bus ==========
    # Enums
    'MessageType',
    'MessagePriority',
    # Data structures
    'PhaseMessage',
    # Type aliases
    'MessageHandler',
    # Main class
    'RitualMessageBus',
    # Factory
    'create_message_bus',

    # ========== context_handoff ==========
    # Enums
    'HandoffStrategy',
    # Data structures
    'ContextItem',
    'HandoffResult',
    # Main class
    'RitualContextHandoff',
    # Factory
    'create_context_handoff',
    # Constants
    'PHASE_KEYWORDS',
    'CAPABILITY_KEYWORDS',

    # ========== coordinator ==========
    # Enums
    'CoordinatorState',
    # Data structures
    'RitualExecutionResult',
    'PhaseExecutionResult',
    # Constants
    'DEFAULT_PHASE_ORDER',
    # Main class
    'RitualAgentCoordinator',
    # Factory functions
    'create_coordinator',
    'create_minimal_coordinator',
    # Convenience functions
    'quick_ritual',

    # ========== phases ==========
    # Base classes and protocols
    'PhasePreflightResult',
    'PhaseResult',
    'PhaseExecutionContext',
    'RitualPhaseSkill',
    'AbstractRitualPhase',
    # AWAKEN phase
    'RestoredContext',
    'AwakenPhase',
    'create_awaken_phase',
    # PLAN phase
    'ImplementationStep',
    'SimilarPattern',
    'ImplementationPlan',
    'PlanPhase',
    'create_plan_phase',
    # REVIEW phase
    'QualityIssue',
    'ImplementationDecision',
    'Learning',
    'ReviewResult',
    'ReviewPhase',
    'create_review_phase',
    # REFLECT phase
    'SessionSummary',
    'ReflectPhase',
    'create_reflect_phase',
]
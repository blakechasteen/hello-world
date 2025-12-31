"""
Ritual Phases Package
=====================

Created: December 30, 2025
Purpose: Export all phase components for the 5-phase coding ritual

Phases:
- AWAKEN: Restore context, recall relevant knowledge
- PLAN: Capture requirements, create implementation plan
- IMPLEMENT: (Manual coding - no automation)
- REVIEW: Quality check, capture learnings
- REFLECT: End-of-session consolidation
"""

# Base classes and protocols
from .base import (
    AgentCapability,
    PhasePreflightResult,
    PhaseResult,
    PhaseExecutionContext,
    RitualPhaseSkill,
    AbstractRitualPhase,
)

# AWAKEN phase
from .awaken import (
    RestoredContext,
    AwakenPhase,
    create_awaken_phase,
)

# PLAN phase
from .plan import (
    ImplementationStep,
    SimilarPattern,
    ImplementationPlan,
    PlanPhase,
    create_plan_phase,
)

# REVIEW phase
from .review import (
    QualityIssue,
    ImplementationDecision,
    Learning,
    ReviewResult,
    ReviewPhase,
    create_review_phase,
)

# REFLECT phase
from .reflect import (
    SessionSummary,
    ReflectPhase,
    create_reflect_phase,
)


__all__ = [
    # Base classes and protocols
    'AgentCapability',
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
"""
HoloLoom Alignment Framework
============================
Comprehensive alignment, safety, and interpretability infrastructure.

Modules:
- safety_guardrails: Policy gating, risk escalation, adversarial defense
- deception_detection: Behavioral probes, goal transparency
- instrumental_convergence: Resource-seeking bounds, autonomy limits
- audit_trail: Complete decision logging and provenance
"""

from .safety_guardrails import (
    SafetyGuardrails,
    RiskLevel,
    ActionCategory,
    SafetyDecision,
)
from .deception_detection import (
    DeceptionDetector,
    BehavioralProbe,
    GoalTransparency,
)
from .instrumental_convergence import (
    InstrumentalConvergenceGuard,
    AutonomyLimiter,
    ResourceBounds,
)
from .audit_trail import (
    AuditTrail,
    DecisionLog,
    ProvenanceTracer,
)

__all__ = [
    "SafetyGuardrails",
    "RiskLevel",
    "ActionCategory",
    "SafetyDecision",
    "DeceptionDetector",
    "BehavioralProbe",
    "GoalTransparency",
    "InstrumentalConvergenceGuard",
    "AutonomyLimiter",
    "ResourceBounds",
    "AuditTrail",
    "DecisionLog",
    "ProvenanceTracer",
]
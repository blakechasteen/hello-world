"""
HoloLoom Alignment Framework
============================
Comprehensive alignment, safety, and interpretability infrastructure.

Phase 1 - Core Safety (Shipped):
- safety_guardrails: Policy gating, risk escalation, adversarial defense
- deception_detection: Behavioral probes, goal transparency
- instrumental_convergence: Resource-seeking bounds, autonomy limits
- audit_trail: Complete decision logging and provenance
- monitoring: Latency tracking, Prometheus export, alignment metrics (E2.1 Dec 2025)
- alerting: Webhook alerts for Slack/Discord/Email (E2.3 Dec 2025)

Phase 2 - Advanced Interpretability (In Progress):
- shap_lime_explainer: Model-agnostic feature attribution
- causal_explainer: Causal reasoning and intervention analysis
- counterfactual_generator: What-if analysis for decisions
- agentic_explainability: Interpretability for agentic reasoning modes
"""

# Phase 1: Core Safety
from .safety_guardrails import (
    SafetyGuardrails,
    RiskLevel,
    ActionCategory,
    ActionRequest,
    SafetyDecision,
    create_guardrails,  # Factory function
)
from .deception_detection import (
    DeceptionDetector,
    BehavioralProbe,
    GoalTransparency,
    create_detector,  # Factory function
)
from .instrumental_convergence import (
    InstrumentalConvergenceGuard,
    AutonomyLimit,
    ResourceBounds,
    create_guard,  # Factory function
)
from .audit_trail import (
    AuditTrail,
    DecisionLog,
    DecisionType,
    OutcomeType,
    ProvenanceTracer,
    create_audit_trail,  # Factory function
)

# E2.1: Monitoring & Prometheus Metrics (December 2025)
from .monitoring import (
    AlignmentMonitor,
    AlignmentMetrics,
    AlertLevel,
    Alert,
    LatencyMetrics,
    SafetyRiskLevel,
    DeceptionFlagType,
    ConvergenceViolationType,
    AutonomyStepType,
    ResourceMetricType,
    get_global_monitor,
    set_global_monitor,
)

# E2.3: Webhook Alerting (December 2025)
from .alerting import (
    AlertDispatcher,
    AlertConfig,
    AlertSeverity,
    AlertChannel,
    Alert as WebhookAlert,  # Rename to avoid conflict with monitoring.Alert
    get_alert_dispatcher,
    set_alert_dispatcher,
    dispatch_alignment_alert,
    alert_deception_detected,
    alert_convergence_violation,
    alert_high_risk_action,
)

# Phase 2: Advanced Interpretability
from .agentic_explainability import (
    AgenticExplainer,
    StepExplanation,
    ReasoningExplanation,
    ExplanationDepth,
    explain_agentic_result,
)

__all__ = [
    # Phase 1 - Classes
    "SafetyGuardrails",
    "RiskLevel",
    "ActionCategory",
    "ActionRequest",
    "SafetyDecision",
    "DeceptionDetector",
    "BehavioralProbe",
    "GoalTransparency",
    "InstrumentalConvergenceGuard",
    "AutonomyLimit",
    "ResourceBounds",
    "AuditTrail",
    "DecisionLog",
    "DecisionType",
    "OutcomeType",
    "ProvenanceTracer",
    # Phase 1 - Factory Functions
    "create_guardrails",
    "create_detector",
    "create_guard",
    "create_audit_trail",
    # E2.1 - Monitoring & Prometheus Metrics (December 2025)
    "AlignmentMonitor",
    "AlignmentMetrics",
    "AlertLevel",
    "Alert",
    "LatencyMetrics",
    "SafetyRiskLevel",
    "DeceptionFlagType",
    "ConvergenceViolationType",
    "AutonomyStepType",
    "ResourceMetricType",
    "get_global_monitor",
    "set_global_monitor",
    # E2.3 - Webhook Alerting (December 2025)
    "AlertDispatcher",
    "AlertConfig",
    "AlertSeverity",
    "AlertChannel",
    "WebhookAlert",
    "get_alert_dispatcher",
    "set_alert_dispatcher",
    "dispatch_alignment_alert",
    "alert_deception_detected",
    "alert_convergence_violation",
    "alert_high_risk_action",
    # Phase 2
    "AgenticExplainer",
    "StepExplanation",
    "ReasoningExplanation",
    "ExplanationDepth",
    "explain_agentic_result",
]
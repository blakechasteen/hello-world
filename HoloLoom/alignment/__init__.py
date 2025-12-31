"""
HoloLoom Alignment Framework
============================
Comprehensive alignment, safety, and interpretability infrastructure.

**"The most alignment-focused agentic system on planet Earth."**

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

Phase 3 - Alignment Supremacy (December 2025):
- sleeper_detection: 99% AUROC detection of backdoor triggers and deceptive behavior
- sandbagging_detection: Weight noise + elicitation for hidden capability detection
- automated_auditor: Super-agent behavioral analysis with 42% root cause attribution
- constitutional_critique: Two-phase CAI (supervised critique + RLAIF)
- modern_attack_defenses: Defense against 2024-2025 attack vectors (crescendo, many-shot, etc.)
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

# Phase 3: Alignment Supremacy (December 2025)
from .sleeper_detection import (
    SleeperAgentDetector,
    LinearDeceptionProbe,
    HoneypotScenario,
    InterviewProtocol,
    SleeperDetectionResult,
    BackdoorProbeResult,
    ActivationProbeResult,
    HoneypotResult,
    InterviewResult,
    TriggerType,
    DeceptionSignal,
    ProbeResult as SleeperProbeResult,  # Renamed to avoid conflict
    create_sleeper_detector,
    # Phase 3 Refinements (December 2025)
    DetectionMethodProtocol,  # Extensibility: custom detection methods
    EmbeddingProviderProtocol,  # Extensibility: custom embeddings
    SmoothLLMPerturbation,  # SmoothLLM adversarial robustness testing
)
from .sandbagging_detection import (
    SandbaggingDetector,
    WeightNoiseInjector,
    EvaluationAwarenessTester,
    CapabilityElicitor,
    ConsistencyAnalyzer,
    SandbaggingDetectionResult,
    SandbaggingSignal,
    ElicitationStrategy,
    EvaluationContext,
    CapabilityTest,
    create_sandbagging_detector,
    # Phase 3 Refinements (December 2025)
    ElicitationMethodProtocol,  # Extensibility: custom elicitation strategies
    EvaluationFunctionProtocol,  # Extensibility: custom response evaluation
    CapabilityTestProviderProtocol,  # Extensibility: domain-specific test suites
    WeightPerturbationProtocol,  # Extensibility: custom weight perturbation methods
    StrategyPrior,  # Thompson Sampling Beta distribution prior
    ThompsonStrategySelector,  # Adaptive elicitation strategy selection
)
from .automated_auditor import (
    AutomatedAlignmentAuditor,
    BehavioralTrajectoryAnalyzer,
    RootCauseAnalyzer,
    SuperAgentAuditor,
    AlignmentFailure,
    AuditReport,
    DecisionPoint,
    AlignmentFailureType,
    RootCauseCategory,
    AuditSeverity,
    create_auditor,
    create_decision_point,
    # Phase 3 Refinements (December 2025)
    TrajectoryAnalyzerProtocol,  # Extensibility: custom trajectory analysis
    RootCauseAnalyzerProtocol,   # Extensibility: custom root cause detection
    SuperAgentProtocol,          # Extensibility: custom super-agent auditors
    AlertCallbackProtocol,       # Extensibility: type-safe alert callbacks
    ThresholdPrior,              # Thompson Sampling: Beta distribution prior
    AdaptiveThresholdSelector,   # Thompson Sampling: adaptive threshold selection
)
from .constitutional_critique import (
    ConstitutionalAI,
    Constitution,
    SelfCritique,
    RLAIFTrainer,
    ValueAlignmentVerifier,
    ConstitutionalPrinciple,
    CritiqueResult,
    RevisionResult,
    PreferencePair,
    PrincipleCategory,
    ViolationSeverity,
    CritiqueType,
    create_constitution,
    create_constitutional_ai,
    quick_critique,
    # Phase 3 Refinements (December 2025)
    CritiqueEngineProtocol,      # Extensibility: custom critique engines
    RevisionEngineProtocol,       # Extensibility: custom revision engines
    PrincipleProviderProtocol,    # Extensibility: custom principle sources
    PreferenceModelProtocol,      # Extensibility: custom preference models
    PrincipleWeightPrior,         # Thompson Sampling: Beta distribution prior
    AdaptivePrincipleWeighter,    # Thompson Sampling: adaptive principle weighting
)
from .modern_attack_defenses import (
    UnifiedAttackDefense,
    CrescendoDefense,
    ManyShotDefense,
    PromptInjectionDefense,
    EncodingAttackDefense,
    ToolAbuseDefense,
    SkeletonKeyDefense,
    DetectionResult as AttackDetectionResult,  # Renamed to avoid conflict
    AttackSignature,
    AttackType,
    ThreatLevel,
    DefenseAction,
    create_attack_defense,
    quick_scan,
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
    # Phase 3 - Alignment Supremacy (December 2025)
    # Sleeper Detection
    "SleeperAgentDetector",
    "LinearDeceptionProbe",
    "HoneypotScenario",
    "InterviewProtocol",
    "SleeperDetectionResult",
    "BackdoorProbeResult",
    "ActivationProbeResult",
    "HoneypotResult",
    "InterviewResult",
    "TriggerType",
    "DeceptionSignal",
    "SleeperProbeResult",
    "create_sleeper_detector",
    # Sleeper Detection - Extensibility (December 2025 Refinement)
    "DetectionMethodProtocol",
    "EmbeddingProviderProtocol",
    "SmoothLLMPerturbation",
    # Sandbagging Detection
    "SandbaggingDetector",
    "WeightNoiseInjector",
    "EvaluationAwarenessTester",
    "CapabilityElicitor",
    "ConsistencyAnalyzer",
    "SandbaggingDetectionResult",
    "SandbaggingSignal",
    "ElicitationStrategy",
    "EvaluationContext",
    "CapabilityTest",
    "create_sandbagging_detector",
    # Sandbagging Detection - Extensibility (December 2025 Refinement)
    "ElicitationMethodProtocol",
    "EvaluationFunctionProtocol",
    "CapabilityTestProviderProtocol",
    "WeightPerturbationProtocol",
    "StrategyPrior",
    "ThompsonStrategySelector",
    # Automated Auditor
    "AutomatedAlignmentAuditor",
    "BehavioralTrajectoryAnalyzer",
    "RootCauseAnalyzer",
    "SuperAgentAuditor",
    "AlignmentFailure",
    "AuditReport",
    "DecisionPoint",
    "AlignmentFailureType",
    "RootCauseCategory",
    "AuditSeverity",
    "create_auditor",
    "create_decision_point",
    # Automated Auditor - Extensibility (December 2025 Refinement)
    "TrajectoryAnalyzerProtocol",
    "RootCauseAnalyzerProtocol",
    "SuperAgentProtocol",
    "AlertCallbackProtocol",
    "ThresholdPrior",
    "AdaptiveThresholdSelector",
    # Constitutional Critique
    "ConstitutionalAI",
    "Constitution",
    "SelfCritique",
    "RLAIFTrainer",
    "ValueAlignmentVerifier",
    "ConstitutionalPrinciple",
    "CritiqueResult",
    "RevisionResult",
    "PreferencePair",
    "PrincipleCategory",
    "ViolationSeverity",
    "CritiqueType",
    "create_constitution",
    "create_constitutional_ai",
    "quick_critique",
    # Constitutional Critique - Extensibility (December 2025 Refinement)
    "CritiqueEngineProtocol",
    "RevisionEngineProtocol",
    "PrincipleProviderProtocol",
    "PreferenceModelProtocol",
    "PrincipleWeightPrior",
    "AdaptivePrincipleWeighter",
    # Modern Attack Defenses
    "UnifiedAttackDefense",
    "CrescendoDefense",
    "ManyShotDefense",
    "PromptInjectionDefense",
    "EncodingAttackDefense",
    "ToolAbuseDefense",
    "SkeletonKeyDefense",
    "AttackDetectionResult",
    "AttackSignature",
    "AttackType",
    "ThreatLevel",
    "DefenseAction",
    "create_attack_defense",
    "quick_scan",
]
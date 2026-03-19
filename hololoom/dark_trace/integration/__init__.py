"""
Dark Trace Integration: HoloLoom System Integration

This module provides deep integration between Dark Trace interpretability
and other HoloLoom systems:

- Weaving Orchestrator: Interpretability hooks in the weaving cycle
- Alignment Framework: Safety-interpretability bridge
- Monitoring: Production-time feature monitoring and drift detection

Key Components:
- DarkTraceOrchestrator: Interpretability-aware orchestration
- AlignmentBridge: Connect safety checks to feature analysis
- InterpretabilityMonitor: Production feature monitoring

Design Philosophy:
- Interpretability should be automatic, not optional
- Every decision should be traceable to features
- Safety and interpretability are two sides of the same coin
- Production monitoring detects feature drift early

Integration Points:
1. Weaving Cycle: Inject DarkTrace at extraction/decision points
2. Alignment: Use feature analysis to validate safety decisions
3. Monitoring: Track feature statistics over time

Usage:
    from hololoom.dark_trace.integration import (
        # Orchestrator integration
        DarkTraceOrchestrator,
        create_traced_orchestrator,
        # Alignment integration
        AlignmentBridge,
        create_alignment_bridge,
        # Monitoring
        InterpretabilityMonitor,
        create_monitor,
    )

    # Create interpretability-aware orchestrator
    orchestrator = create_traced_orchestrator(config, shards)

    # Process with full feature tracing
    spacetime = await orchestrator.weave(query)

    # Access interpretability analysis
    trace = spacetime.metadata.get('dark_trace')
    print(trace.active_features)
    print(trace.safety_relevant)
"""

from hololoom.dark_trace.integration.alignment import (
    AlignmentBridge,
    AlignmentConfig,
    SafetyFeatureMapping,
    create_alignment_bridge,
)
from hololoom.dark_trace.integration.monitoring import (
    DriftDetector,
    FeatureStatistics,
    InterpretabilityMonitor,
    MonitorConfig,
    create_monitor,
)
from hololoom.dark_trace.integration.orchestrator import (
    DarkTraceOrchestrator,
    OrchestratorConfig,
    TracedSpacetime,
    create_traced_orchestrator,
)
from hololoom.dark_trace.integration.orchestrator_hooks import (
    DarkTraceIntegrator,
    DarkTraceMiddleware,
    DecisionHook,
    HookContext,
    HookPoint,
    HookResult,
    IntegrationBlockedError,
    IntegrationMode,
    MiddlewareConfig,
    PostWeaveHook,
    PreWeaveHook,
    create_integrator,
    integrate_dark_trace,
)
from hololoom.dark_trace.integration.safety_gate import (
    ConcernType,
    DarkTraceSafetyGate,
    GateDecision,
    SafetyAction,
    SafetyConcern,
    SafetyGateConfig,
    SafetyPatternDetector,
    create_safety_gate,
)

__all__ = [
    # Orchestrator integration
    "DarkTraceOrchestrator",
    "TracedSpacetime",
    "OrchestratorConfig",
    "create_traced_orchestrator",
    # Alignment integration
    "AlignmentBridge",
    "SafetyFeatureMapping",
    "AlignmentConfig",
    "create_alignment_bridge",
    # Monitoring
    "InterpretabilityMonitor",
    "FeatureStatistics",
    "DriftDetector",
    "MonitorConfig",
    "create_monitor",
    # Safety Gate
    "SafetyAction",
    "ConcernType",
    "SafetyConcern",
    "GateDecision",
    "SafetyGateConfig",
    "SafetyPatternDetector",
    "DarkTraceSafetyGate",
    "create_safety_gate",
    # Orchestrator Hooks
    "IntegrationMode",
    "HookPoint",
    "HookContext",
    "HookResult",
    "PreWeaveHook",
    "PostWeaveHook",
    "DecisionHook",
    "MiddlewareConfig",
    "DarkTraceMiddleware",
    "DarkTraceIntegrator",
    "IntegrationBlockedError",
    "create_integrator",
    "integrate_dark_trace",
]

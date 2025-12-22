"""
Dark Trace: Responsibility & Interpretability Suite

HoloLoom's comprehensive interpretability system providing:
- Sparse Autoencoder (SAE) decomposition of neural activations
- Semantic Axes projection for interpretable dimensions
- Causal validation through ablation, injection, and patching
- Unified feature registry with SAE↔Semantic correlation tracking
- Safety monitoring and steering capabilities

Research Foundation:
- SAE: Anthropic's "Towards Monosemanticity" (2023)
- Scaling: "Scaling Monosemanticity" (2024)
- Causal: ACDC (NeurIPS 2023), Activation Patching

Dark Trace Mission:
- Responsibility: We take responsibility for understanding what our systems learn
- Interpretability: Every decision, activation, and representation is transparent
- Alignment: Safety-relevant features are identified and controlled

Quick Start:
    from HoloLoom.dark_trace import DarkTraceEngine, TraceConfig, create_engine

    # Create engine with standard preset
    config = TraceConfig.standard(input_dim=384)
    engine = create_engine(config)

    # Analyze activations
    result = engine.analyze(activations)
    print(result.explanation)

    # Steer toward goals
    steering = engine.steer({"semantic.Warmth": 0.8, "semantic.Formality": -0.5})
    steered = activations + steering.vector
"""

# =============================================================================
# Protocol Layer - Unified Interpretability Interface
# =============================================================================

from HoloLoom.dark_trace.protocol import (
    # Enums
    LensType,
    FeatureSource,
    SafetyFlag,
    # Core dataclasses
    Feature,
    FeatureActivation,
    SteeringVector,
    # Protocols
    TraceLens,
    CausalValidator,
    # Base class
    BaseLens,
    # Type aliases
    Activations,
    FeatureMap,
    GoalMap,
    ActiveFeatures,
)

# =============================================================================
# Result Layer - Analysis Output Structures
# =============================================================================

from HoloLoom.dark_trace.result import (
    # Core results
    LensResult,
    TraceResult,
    # Steering and causal results
    SteeringResult,
    AblationResult,
    InjectionResult,
    PatchResult,
    # Circuit discovery
    CircuitNode,
    CircuitEdge,
    CircuitTrace,
    # Evaluation
    LensEvaluationMetrics,
    DarkTraceEvaluationReport,
)

# =============================================================================
# Registry Layer - Unified Feature Namespace
# =============================================================================

from HoloLoom.dark_trace.registry import (
    CorrelationMatrix,
    FeatureRegistry,
    # Factory functions
    create_sae_features,
    create_semantic_features,
)

# =============================================================================
# Configuration Layer - Presets and Settings
# =============================================================================

from HoloLoom.dark_trace.trace_config import (
    # Mode enum
    TraceMode,
    # Component configs
    SAEConfig,
    SemanticConfig,
    DomainConfig,
    CausalConfig,
    SafetyConfig,
    CorrelationConfig,
    LoggingConfig,
    # Main config with presets
    TraceConfig,
)

# =============================================================================
# Engine Layer - Main API
# =============================================================================

from HoloLoom.dark_trace.engine import (
    DarkTraceEngine,
    create_engine,
)

# =============================================================================
# Existing Components - SAE, Probes, Steering
# =============================================================================

from HoloLoom.dark_trace.sae import (
    SparseAutoEncoder,
    DarkSaeTrainer,
)

from HoloLoom.dark_trace.probe import (
    MindProbe,
)

from HoloLoom.dark_trace.steering_policy import (
    PIDSteeringController,
    ConsistencyGuard,
)

from HoloLoom.dark_trace.auto_probe import (
    AutoProbe,
)

# =============================================================================
# Model Adapters - Multi-Model Support
# =============================================================================

from HoloLoom.dark_trace.models import (
    # Adapter protocol
    ModelAdapter,
    LayerInfo,
    LayerType,
    ModelCapabilities,
    SteeringConfig,
    HookHandle,
    ActivationHook,
    DummyAdapter,
    ActivationCache,
    create_hook,
    make_cache_key,
    # Policy adapter
    PolicyAdapter,
    # Transformer adapter
    TransformerAdapter,
    # Fingerprinting
    FeatureFingerprint,
    ModelFingerprinter,
    FingerprintConfig,
    FingerprintComparison,
    compare_fingerprints,
    find_universal_features,
    find_model_specific_features,
)

# =============================================================================
# Integration Layer - HoloLoom System Integration
# =============================================================================

from HoloLoom.dark_trace.integration import (
    # Orchestrator integration
    DarkTraceOrchestrator,
    TracedSpacetime,
    OrchestratorConfig,
    create_traced_orchestrator,
    # Alignment integration
    AlignmentBridge,
    SafetyFeatureMapping,
    AlignmentConfig,
    create_alignment_bridge,
    # Monitoring
    InterpretabilityMonitor,
    FeatureStatistics,
    DriftDetector,
    MonitorConfig,
    create_monitor,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Protocol layer
    "LensType",
    "FeatureSource",
    "SafetyFlag",
    "Feature",
    "FeatureActivation",
    "SteeringVector",
    "TraceLens",
    "CausalValidator",
    "BaseLens",
    "Activations",
    "FeatureMap",
    "GoalMap",
    "ActiveFeatures",
    # Result layer
    "LensResult",
    "TraceResult",
    "SteeringResult",
    "AblationResult",
    "InjectionResult",
    "PatchResult",
    "CircuitNode",
    "CircuitEdge",
    "CircuitTrace",
    "LensEvaluationMetrics",
    "DarkTraceEvaluationReport",
    # Registry layer
    "CorrelationMatrix",
    "FeatureRegistry",
    "create_sae_features",
    "create_semantic_features",
    # Configuration layer
    "TraceMode",
    "SAEConfig",
    "SemanticConfig",
    "DomainConfig",
    "CausalConfig",
    "SafetyConfig",
    "CorrelationConfig",
    "LoggingConfig",
    "TraceConfig",
    # Engine layer
    "DarkTraceEngine",
    "create_engine",
    # Existing components
    "SparseAutoEncoder",
    "DarkSaeTrainer",
    "MindProbe",
    "PIDSteeringController",
    "ConsistencyGuard",
    "AutoProbe",
    # Model adapters (Phase 9)
    "ModelAdapter",
    "LayerInfo",
    "LayerType",
    "ModelCapabilities",
    "SteeringConfig",
    "HookHandle",
    "ActivationHook",
    "DummyAdapter",
    "ActivationCache",
    "create_hook",
    "make_cache_key",
    "PolicyAdapter",
    "TransformerAdapter",
    "FeatureFingerprint",
    "ModelFingerprinter",
    "FingerprintConfig",
    "FingerprintComparison",
    "compare_fingerprints",
    "find_universal_features",
    "find_model_specific_features",
    # Integration layer (Phase 10)
    "DarkTraceOrchestrator",
    "TracedSpacetime",
    "OrchestratorConfig",
    "create_traced_orchestrator",
    "AlignmentBridge",
    "SafetyFeatureMapping",
    "AlignmentConfig",
    "create_alignment_bridge",
    "InterpretabilityMonitor",
    "FeatureStatistics",
    "DriftDetector",
    "MonitorConfig",
    "create_monitor",
]

# Version
__version__ = "1.0.0"

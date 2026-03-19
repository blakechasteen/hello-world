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
    from hololoom.dark_trace import DarkTraceEngine, TraceConfig, create_engine

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

from hololoom.dark_trace.auto_probe import (
    AutoProbe,
)

# =============================================================================
# Engine Layer - Main API
# =============================================================================
from hololoom.dark_trace.engine import (
    DarkTraceEngine,
    create_engine,
)

# =============================================================================
# Integration Layer - HoloLoom System Integration
# =============================================================================
from hololoom.dark_trace.integration import (
    # Alignment integration
    AlignmentBridge,
    AlignmentConfig,
    # Orchestrator integration
    DarkTraceOrchestrator,
    DriftDetector,
    FeatureStatistics,
    # Monitoring
    InterpretabilityMonitor,
    MonitorConfig,
    OrchestratorConfig,
    SafetyFeatureMapping,
    TracedSpacetime,
    create_alignment_bridge,
    create_monitor,
    create_traced_orchestrator,
)

# =============================================================================
# Model Adapters - Multi-Model Support
# =============================================================================
from hololoom.dark_trace.models import (
    ActivationCache,
    ActivationHook,
    DummyAdapter,
    # Fingerprinting
    FeatureFingerprint,
    FingerprintComparison,
    FingerprintConfig,
    HookHandle,
    LayerInfo,
    LayerType,
    # Adapter protocol
    ModelAdapter,
    ModelCapabilities,
    ModelFingerprinter,
    # Policy adapter
    PolicyAdapter,
    SteeringConfig,
    # Transformer adapter
    TransformerAdapter,
    compare_fingerprints,
    create_hook,
    find_model_specific_features,
    find_universal_features,
    make_cache_key,
)
from hololoom.dark_trace.probe import (
    MindProbe,
)
from hololoom.dark_trace.protocol import (
    # Type aliases
    Activations,
    ActiveFeatures,
    # Base class
    BaseLens,
    CausalValidator,
    # Core dataclasses
    Feature,
    FeatureActivation,
    FeatureMap,
    FeatureSource,
    GoalMap,
    # Enums
    LensType,
    SafetyFlag,
    SteeringVector,
    # Protocols
    TraceLens,
)

# =============================================================================
# Registry Layer - Unified Feature Namespace
# =============================================================================
from hololoom.dark_trace.registry import (
    CorrelationMatrix,
    FeatureRegistry,
    # Factory functions
    create_sae_features,
    create_semantic_features,
)

# =============================================================================
# Result Layer - Analysis Output Structures
# =============================================================================
from hololoom.dark_trace.result import (
    AblationResult,
    CircuitEdge,
    # Circuit discovery
    CircuitNode,
    CircuitTrace,
    DarkTraceEvaluationReport,
    InjectionResult,
    # Evaluation
    LensEvaluationMetrics,
    # Core results
    LensResult,
    PatchResult,
    # Steering and causal results
    SteeringResult,
    TraceResult,
)

# =============================================================================
# Existing Components - SAE, Probes, Steering
# =============================================================================
from hololoom.dark_trace.sae import (
    DarkSaeTrainer,
    SparseAutoEncoder,
)
from hololoom.dark_trace.steering_policy import (
    ConsistencyGuard,
    PIDSteeringController,
)

# =============================================================================
# Configuration Layer - Presets and Settings
# =============================================================================
from hololoom.dark_trace.trace_config import (
    CausalConfig,
    CorrelationConfig,
    DomainConfig,
    LoggingConfig,
    # Component configs
    SAEConfig,
    SafetyConfig,
    SemanticConfig,
    # Main config with presets
    TraceConfig,
    # Mode enum
    TraceMode,
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

"""
Dark Trace Multilayer: Multi-Layer SAE Analysis

This module provides infrastructure for understanding how representations
evolve across model layers, enabling hierarchical interpretability.

Key Capabilities:
- Per-layer SAE training and management
- Feature hierarchy analysis (abstraction ladder)
- Cross-layer information flow tracking
- Cross-layer correlation tracking (Welford's algorithm)
- Feature propagation analysis and circuit discovery
- Causal metrics (Pearl's do-calculus, Transfer Entropy)
- Hierarchy visualization

Research Basis:
- Scaling Monosemanticity (Anthropic 2024): Multi-layer SAE training
- Logit Lens (nostalgebraist): Layer-wise prediction analysis
- Tuned Lens: Learned layer projections
- Residual Stream Analysis: Information accumulation patterns
- ACDC (NeurIPS 2023): Circuit discovery via activation patching
- Pearl's do-calculus: Causal inference framework

Usage:
    from hololoom.dark_trace.multilayer import (
        # Layer SAE Management
        LayerSAEManager,
        LayerSAEConfig,
        LayerSAE,
        # Hierarchy Analysis
        FeatureHierarchyAnalyzer,
        AbstractionLevel,
        FeatureEvolution,
        HierarchyNode,
        # Information Flow
        InformationFlowAnalyzer,
        FlowConfig,
        LayerFlow,
        CrossLayerFlow,
        # Cross-Layer Correlation (Phase 2)
        CorrelationTracker,
        CorrelationConfig,
        LayerCorrelation,
        WelfordCorrelator,
        # Propagation Analysis (Phase 2)
        PropagationAnalyzer,
        PropagationPath,
        CircuitCandidate,
        # Causal Metrics (Phase 2)
        CausalStrengthEstimator,
        TransferEntropyEstimator,
        CausalStrengthResult,
        TransferEntropyResult,
        # Visualization
        HierarchyVisualizer,
        VisualizationConfig,
        HierarchyDiagram,
    )

    # Train SAEs for each layer
    manager = LayerSAEManager(model, n_layers=12)
    await manager.train_all(training_data)

    # Analyze feature hierarchy
    hierarchy = FeatureHierarchyAnalyzer(manager)
    evolution = hierarchy.track_feature_evolution("sae.42")
    print(f"Feature emerges at layer {evolution.emergence_layer}")

    # Analyze information flow
    flow_analyzer = InformationFlowAnalyzer(manager)
    flow = flow_analyzer.analyze(test_inputs)
    print(f"Bottleneck at layer {flow.bottleneck_layer}")

    # Track cross-layer correlations (Phase 2)
    tracker = CorrelationTracker(CorrelationConfig())
    tracker.update(layer_5_activations, layer_6_activations, 5, 6)
    correlations = tracker.get_correlations(5, 6, min_correlation=0.5)

    # Analyze feature propagation (Phase 2)
    propagation = PropagationAnalyzer(manager, tracker)
    paths = propagation.trace_feature_forward(42, start_layer=5)
    circuits = propagation.find_circuits(min_features=3)

    # Causal flow analysis (Phase 2)
    causal = flow_analyzer.get_causal_flow_analysis(test_inputs)
    print(f"Dominant causal pathways: {causal['dominant_pathways']}")

    # Visualize hierarchy
    viz = HierarchyVisualizer(hierarchy, flow_analyzer)
    diagram = viz.create_hierarchy_diagram()
    diagram.save("feature_hierarchy.html")

Created: 2025-12-28
"""

# Layer SAE management
# Causal metrics (Phase 2)
from hololoom.dark_trace.multilayer.causal_metrics import (
    CausalStrengthEstimator,
    CausalStrengthResult,
    TransferEntropyEstimator,
    TransferEntropyResult,
)

# Cross-layer correlation tracking (Phase 2)
from hololoom.dark_trace.multilayer.correlation_tracker import (
    CorrelationConfig,
    CorrelationPair,
    CorrelationTracker,
    LayerCorrelation,
    WelfordCorrelator,
)

# Information flow analysis
from hololoom.dark_trace.multilayer.flow import (
    CrossLayerFlow,
    FlowBottleneck,
    FlowConfig,
    InformationFlowAnalyzer,
    LayerFlow,
    ResidualContribution,
)

# Feature hierarchy analysis
from hololoom.dark_trace.multilayer.hierarchy import (
    AbstractionLadder,
    AbstractionLevel,
    FeatureEvolution,
    FeatureHierarchyAnalyzer,
    HierarchyEdge,
    HierarchyNode,
)
from hololoom.dark_trace.multilayer.layer_sae import (
    LayerActivations,
    LayerSAE,
    LayerSAEConfig,
    LayerSAEManager,
    MultiLayerFeatures,
)

# Feature propagation analysis (Phase 2)
from hololoom.dark_trace.multilayer.propagation_analyzer import (
    CircuitCandidate,
    PropagationAnalyzer,
    PropagationPath,
)

# Hierarchy visualization
from hololoom.dark_trace.multilayer.visualizer import (
    EvolutionPlot,
    FlowDiagram,
    HierarchyDiagram,
    HierarchyVisualizer,
    VisualizationConfig,
)

__all__ = [
    # Layer SAE
    "LayerSAEManager",
    "LayerSAEConfig",
    "LayerSAE",
    "LayerActivations",
    "MultiLayerFeatures",
    # Hierarchy
    "FeatureHierarchyAnalyzer",
    "AbstractionLevel",
    "FeatureEvolution",
    "HierarchyNode",
    "HierarchyEdge",
    "AbstractionLadder",
    # Flow
    "InformationFlowAnalyzer",
    "FlowConfig",
    "LayerFlow",
    "CrossLayerFlow",
    "FlowBottleneck",
    "ResidualContribution",
    # Visualization
    "HierarchyVisualizer",
    "VisualizationConfig",
    "HierarchyDiagram",
    "FlowDiagram",
    "EvolutionPlot",
    # Cross-layer Correlation (Phase 2)
    "CorrelationTracker",
    "CorrelationConfig",
    "LayerCorrelation",
    "CorrelationPair",
    "WelfordCorrelator",
    # Propagation Analysis (Phase 2)
    "PropagationAnalyzer",
    "PropagationPath",
    "CircuitCandidate",
    # Causal Metrics (Phase 2)
    "CausalStrengthEstimator",
    "TransferEntropyEstimator",
    "CausalStrengthResult",
    "TransferEntropyResult",
]

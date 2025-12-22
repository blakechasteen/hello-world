"""
Dark Trace Multilayer: Multi-Layer SAE Analysis

This module provides infrastructure for understanding how representations
evolve across model layers, enabling hierarchical interpretability.

Key Capabilities:
- Per-layer SAE training and management
- Feature hierarchy analysis (abstraction ladder)
- Cross-layer information flow tracking
- Hierarchy visualization

Research Basis:
- Scaling Monosemanticity (Anthropic 2024): Multi-layer SAE training
- Logit Lens (nostalgebraist): Layer-wise prediction analysis
- Tuned Lens: Learned layer projections
- Residual Stream Analysis: Information accumulation patterns

Usage:
    from HoloLoom.dark_trace.multilayer import (
        LayerSAEManager,
        LayerSAEConfig,
        LayerSAE,
        FeatureHierarchyAnalyzer,
        AbstractionLevel,
        FeatureEvolution,
        HierarchyNode,
        InformationFlowAnalyzer,
        FlowConfig,
        LayerFlow,
        CrossLayerFlow,
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

    # Visualize hierarchy
    viz = HierarchyVisualizer(hierarchy, flow_analyzer)
    diagram = viz.create_hierarchy_diagram()
    diagram.save("feature_hierarchy.html")
"""

# Layer SAE management
from HoloLoom.dark_trace.multilayer.layer_sae import (
    LayerSAEManager,
    LayerSAEConfig,
    LayerSAE,
    LayerActivations,
    MultiLayerFeatures,
)

# Feature hierarchy analysis
from HoloLoom.dark_trace.multilayer.hierarchy import (
    FeatureHierarchyAnalyzer,
    AbstractionLevel,
    FeatureEvolution,
    HierarchyNode,
    HierarchyEdge,
    AbstractionLadder,
)

# Information flow analysis
from HoloLoom.dark_trace.multilayer.flow import (
    InformationFlowAnalyzer,
    FlowConfig,
    LayerFlow,
    CrossLayerFlow,
    FlowBottleneck,
    ResidualContribution,
)

# Hierarchy visualization
from HoloLoom.dark_trace.multilayer.visualizer import (
    HierarchyVisualizer,
    VisualizationConfig,
    HierarchyDiagram,
    FlowDiagram,
    EvolutionPlot,
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
]

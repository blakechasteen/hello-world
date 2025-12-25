"""
Dark Trace Visualization Components

Interactive visualizations for interpretability analysis.

Components:
- CircuitExplorer: Interactive circuit graph visualization
- FeatureHeatmap: Feature activation heatmaps
- DriftMonitor: Real-time drift detection UI
- SAEBrowser: Sparse autoencoder feature browser
- DashboardBuilder: Comprehensive dashboard construction

Author: HoloLoom Team
Created: December 2025
"""

from .circuit_explorer import (
    CircuitExplorer,
    CircuitNode,
    CircuitEdge,
    render_circuit_graph,
)

from .feature_heatmap import (
    FeatureHeatmap,
    render_activation_heatmap,
    render_layer_heatmap,
)

from .drift_monitor import (
    DriftMonitor,
    DriftEvent,
    DriftType,
    render_drift_timeline,
)

from .dashboard import (
    DarkTraceDashboard,
    DashboardConfig,
    create_dashboard,
)

__all__ = [
    # Circuit Explorer
    "CircuitExplorer",
    "CircuitNode",
    "CircuitEdge",
    "render_circuit_graph",
    # Feature Heatmap
    "FeatureHeatmap",
    "render_activation_heatmap",
    "render_layer_heatmap",
    # Drift Monitor
    "DriftMonitor",
    "DriftEvent",
    "DriftType",
    "render_drift_timeline",
    # Dashboard
    "DarkTraceDashboard",
    "DashboardConfig",
    "create_dashboard",
]

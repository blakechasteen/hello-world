"""
Dark Trace Visualization Components

Interactive visualizations for interpretability analysis.

Components:
- CircuitExplorer: Interactive circuit graph visualization
- FeatureHeatmap: Feature activation heatmaps
- DriftMonitor: Real-time drift detection UI
- SAEBrowser: Sparse autoencoder feature browser
- DashboardBuilder: Comprehensive dashboard construction
- StreamingServer: WebSocket server for real-time activation streaming
- DashboardAPI: FastAPI REST/WebSocket API for dashboard
- OrchestratorHook: Hook for integrating with weaving orchestrator

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

# Phase 2: Real-time Streaming and Dashboard API
from .streaming_server import (
    DarkTraceStreamingServer,
    StreamingConfig,
    StreamMessage,
    MessageType,
    SubscriptionType,
    ClientSubscription,
    ActivationSnapshot,
    ActivationStreamHandler,
    create_streaming_server,
    create_streaming_config,
)

from .dashboard_api import (
    DarkTraceDashboardAPI,
    DashboardAPIConfig,
    FeatureInfo,
    LayerInfo,
    ActivationEntry,
    SteeringRequest,
    SteeringResponse,
    StatusResponse,
    create_dashboard_api,
    create_dashboard_router,
)

from .orchestrator_hook import (
    DarkTraceOrchestratorHook,
    HookConfig,
    HookState,
    create_orchestrator_hook,
    create_hook_config,
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
    # Streaming Server
    "DarkTraceStreamingServer",
    "StreamingConfig",
    "StreamMessage",
    "MessageType",
    "SubscriptionType",
    "ClientSubscription",
    "ActivationSnapshot",
    "ActivationStreamHandler",
    "create_streaming_server",
    "create_streaming_config",
    # Dashboard API
    "DarkTraceDashboardAPI",
    "DashboardAPIConfig",
    "FeatureInfo",
    "LayerInfo",
    "ActivationEntry",
    "SteeringRequest",
    "SteeringResponse",
    "StatusResponse",
    "create_dashboard_api",
    "create_dashboard_router",
    # Orchestrator Hook
    "DarkTraceOrchestratorHook",
    "HookConfig",
    "HookState",
    "create_orchestrator_hook",
    "create_hook_config",
]

"""
AR Adapter for Elle - WebXR/ARCore/ARKit Integration

This adapter layer translates AR platform events into Elle requests
and Elle actions into AR visualizations.

Architecture:
    AR Platform Events → ARAdapter → ElleRequest → Elle Core
    Elle Core → ElleAction → ARAdapter → AR Visualizations

Components:
    - ARAdapter: Main adapter coordinating AR ↔ Elle translation
    - AREvent: Platform-agnostic AR event models
    - ARVisualization: 3D overlay specifications
    - PlatformBridge: WebXR/ARCore/ARKit abstraction
    - ARRenderer: Visualization rendering logic

Created: 2025-11-22 (Phase 1 - Two Week Prototype)
"""

from elle.adapters.ar_adapter.ar_events import (
    AREvent,
    AREventType,
    GazeEvent,
    ScanEvent,
    SelectionEvent,
    GestureEvent,
    VoiceEvent,
    ARContext,
    ARObject,
    Vector3,
    Quaternion,
)

from elle.adapters.ar_adapter.ar_adapter import ARAdapter

from elle.adapters.ar_adapter.ar_renderer import (
    ARVisualization,
    ARVisualizationType,
    AROverlay,
    ARHighlight,
    ARPath,
    ARPanel,
)

__all__ = [
    # Adapter
    "ARAdapter",

    # Events
    "AREvent",
    "AREventType",
    "GazeEvent",
    "ScanEvent",
    "SelectionEvent",
    "GestureEvent",
    "VoiceEvent",

    # Context
    "ARContext",
    "ARObject",
    "Vector3",
    "Quaternion",

    # Visualizations
    "ARVisualization",
    "ARVisualizationType",
    "AROverlay",
    "ARHighlight",
    "ARPath",
    "ARPanel",
]

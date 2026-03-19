"""
Cognitive Visualization Engine (CVE)
====================================

Purpose-built rendering system for visualizing AI cognition.
NOT a game engine - a cognitive visualization engine that makes thinking visible.

Architecture:
    Cognitive Data Sources → Cognitive Rendering Protocol (CRP) → Renderers
                                                                  ├── TufteRenderer (2D)
                                                                  ├── WebGLRenderer (3D)
                                                                  └── ARRenderer (WebXR)

Key Components:
- cognitive_protocol.py: 15 CognitiveVizTypes + CognitiveRenderer protocol
- tufte_renderer.py: 2D SVG/CSS renderer (wraps existing Tufte visualizations)
- cognitive_extractors.py: Extract CRP events from hololoom systems
- cve_server.py: WebSocket server for real-time streaming
- consciousness_shell.html: Unified web interface
- webgl_renderer.py: 3D Three.js renderer (Phase 2)
- ar_renderer.py: AR/VR renderer bridging to Elle (Phase 3)

Created: 2025-11-30
Author: HoloLoom Team

Quick Start:
    # Start the CVE server
    python -m HoloLoom.cve.cve_server

    # Open http://localhost:8001 in your browser
"""

# Extractors
from hololoom.cve.cognitive_extractors import (
    # Individual extractors
    AwarenessGraphExtractor,
    MemoryExtractor,
    PolicyExtractor,
    ReflectionExtractor,
    TokenStreamExtractor,
    # Unified extractor
    UnifiedCognitiveExtractor,
    WeavingCycleExtractor,
    create_cognitive_frame,
    # Convenience functions
    extract_from_awareness,
    extract_from_bandit,
    extract_from_spacetime,
)
from hololoom.cve.cognitive_protocol import (
    # Primitives
    ActivationField,
    AttentionFlow,
    # Core Data Structures
    CognitiveEvent,
    CognitiveFrame,
    # Protocol
    CognitiveRenderer,
    CognitiveStream,
    # Viz Types
    CognitiveVizType,
    ConfidenceFlow,
    ContextExpansion,
    DecisionCollapse,
    KnowledgeGraph3D,
    MemoryRetrieval,
    ProbabilityManifold,
    ReasoningChain,
    ReflectionLoop,
    SemanticPosition,
    SemanticTrajectory,
    TokenStream,
    ToolSelection,
    WeavingCycle,
    # Utilities
    create_event,
    create_frame,
)

# Renderer
from hololoom.cve.tufte_renderer import (
    TufteRenderer,
    render_cognitive_event,
    render_cognitive_frame,
)


# Server (lazy import to avoid FastAPI dependency if not needed)
def create_cve_app():
    """Create a standalone FastAPI app for CVE streaming."""
    from hololoom.cve.cve_server import create_cve_app as _create
    return _create()

def create_cve_router():
    """Create a FastAPI router for CVE endpoints."""
    from hololoom.cve.cve_server import create_cve_router as _create
    return _create()

__all__ = [
    # Viz Types
    'CognitiveVizType',

    # Core Data Structures
    'CognitiveEvent',
    'CognitiveFrame',
    'CognitiveStream',

    # Primitives
    'ActivationField',
    'ProbabilityManifold',
    'SemanticPosition',
    'SemanticTrajectory',
    'KnowledgeGraph3D',
    'TokenStream',
    'DecisionCollapse',
    'AttentionFlow',
    'ReasoningChain',
    'MemoryRetrieval',
    'ConfidenceFlow',
    'ToolSelection',
    'ContextExpansion',
    'WeavingCycle',
    'ReflectionLoop',

    # Protocol
    'CognitiveRenderer',

    # Utilities
    'create_event',
    'create_frame',

    # Renderer
    'TufteRenderer',
    'render_cognitive_event',
    'render_cognitive_frame',

    # Extractors
    'AwarenessGraphExtractor',
    'PolicyExtractor',
    'WeavingCycleExtractor',
    'MemoryExtractor',
    'ReflectionExtractor',
    'TokenStreamExtractor',
    'UnifiedCognitiveExtractor',
    'extract_from_awareness',
    'extract_from_bandit',
    'extract_from_spacetime',
    'create_cognitive_frame',

    # Server
    'create_cve_app',
    'create_cve_router',
]

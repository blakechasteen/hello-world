"""
ThirdEye - Chat-integrated 2D/3D Concept Rendering Engine
=========================================================

A visualization engine that integrates with the multithreaded chat,
smoothly transitioning between concepts as conversation topics evolve.

Architecture:
    Chat WebSocket → ChatBridge → Concept Extraction → Transition →
    Renderer (2D/3D/AR) → TypeScript Client

Key Components:
- concept.py: Core Concept dataclass and ConceptWorld
- transition.py: Smooth transition calculations
- chat_bridge.py: Chat WebSocket listener and concept extraction
- thoughtspace.py: Full-screen world mode
- renderer_protocol.py: Abstract renderer interface
- renderers/webgl.py: Three.js 3D renderer

Created: 2025-11-30
Author: HoloLoom Team

Quick Start:
    from HoloLoom.thirdeye import ChatBridge, Thoughtspace, WebGLRenderer

    # Create components
    bridge = ChatBridge(chat_ws_url="ws://localhost:8002")
    thoughtspace = Thoughtspace()
    renderer = WebGLRenderer()

    # Listen to chat and visualize concepts
    async for extraction in bridge.listen():
        for concept in extraction.concepts:
            thoughtspace.add_concept(concept)
            if extraction.transition:
                commands = renderer.render_transition(
                    extraction.transition,
                    None,  # previous concept
                    concept,
                )
                # Send commands to TypeScript client via WebSocket
"""

# Core data types
from HoloLoom.thirdeye.concept import (
    Concept,
    ConceptType,
    ConceptWorld,
    SemanticPosition,
    VisualStyle,
    create_entity_concept,
    create_comparison_concept,
    create_process_concept,
)

# Transitions
from HoloLoom.thirdeye.transition import (
    Transition,
    TransitionType,
    TransitionConfig,
    TransitionCalculator,
    calculate_transition,
    create_emphasis_transition,
    create_branch_transition,
    create_merge_transition,
)

# Chat integration
from HoloLoom.thirdeye.chat_bridge import (
    ChatBridge,
    ChatMessage,
    MessageRole,
    ConceptExtraction,
    ConceptExtractor,
    create_chat_bridge,
)

# Thoughtspace
from HoloLoom.thirdeye.thoughtspace import (
    Thoughtspace,
    ThoughtspaceMode,
    ThoughtspaceState,
    CameraState,
    CameraMode,
    SemanticPositioner,
    create_thoughtspace,
)

# Renderer protocol
from HoloLoom.thirdeye.renderer_protocol import (
    ConceptRenderer,
    BaseConceptRenderer,
    RenderCommand,
    RenderFrame,
    RenderConfig,
    RenderDimension,
    RenderQuality,
)

# Renderers
from HoloLoom.thirdeye.renderers.webgl import (
    WebGLRenderer,
    create_webgl_renderer,
)


__all__ = [
    # Concepts
    'Concept',
    'ConceptType',
    'ConceptWorld',
    'SemanticPosition',
    'VisualStyle',
    'create_entity_concept',
    'create_comparison_concept',
    'create_process_concept',

    # Transitions
    'Transition',
    'TransitionType',
    'TransitionConfig',
    'TransitionCalculator',
    'calculate_transition',
    'create_emphasis_transition',
    'create_branch_transition',
    'create_merge_transition',

    # Chat
    'ChatBridge',
    'ChatMessage',
    'MessageRole',
    'ConceptExtraction',
    'ConceptExtractor',
    'create_chat_bridge',

    # Thoughtspace
    'Thoughtspace',
    'ThoughtspaceMode',
    'ThoughtspaceState',
    'CameraState',
    'CameraMode',
    'SemanticPositioner',
    'create_thoughtspace',

    # Rendering
    'ConceptRenderer',
    'BaseConceptRenderer',
    'RenderCommand',
    'RenderFrame',
    'RenderConfig',
    'RenderDimension',
    'RenderQuality',
    'WebGLRenderer',
    'create_webgl_renderer',
]

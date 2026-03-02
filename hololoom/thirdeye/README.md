# ThirdEye - Chat-Integrated 2D/3D Concept Rendering Engine

**Status**: ✅ Production Ready (December 2025)
**Location**: `HoloLoom/thirdeye/`
**Total Code**: ~13,883 lines across 30 Python files
**Date**: Created 2025-11-30, Updated December 2025

## Overview

ThirdEye is HoloLoom's intelligent visualization engine that transforms chat conversations into rich, context-aware 2D and 3D visualizations. As conversation topics evolve, ThirdEye seamlessly extracts semantic concepts and renders them in multiple dimensions—from compact sparklines to immersive 3D environments.

**Core Philosophy**: "Visualize what you're actually talking about." Instead of generic diagrams, ThirdEye detects the visualization context from chat content and renders appropriate representations (UI mockups, system architectures, narratives, data visualizations, physical objects, or abstract concept maps).

**Key Innovation**: The system bridges the gap between symbolic conversation (text) and visual understanding by:
1. **Extracting semantic concepts** from chat messages using pattern matching and embeddings
2. **Detecting visualization context** (what kind of thing are we visualizing?)
3. **Rendering appropriate scenes** (UI, architecture, narrative, data, objects, worlds, or abstract)
4. **Smoothly transitioning** between concepts as conversation evolves
5. **Supporting multiple dimensions** (1D sparklines, 2D Tufte-style, 3D WebGL, AR/WebXR)

## Architecture

```
Chat WebSocket (8002)
    ↓ (ChatBridge listens)
Concept Extraction
    ↓ (Identify semantic units)
Transition Calculation
    ↓ (Smooth morphing between concepts)
Scene Composition
    ↓ (Detect context: UI, architecture, narrative, data, etc.)
Renderer Selection
    ↓ (WebGL, SVG, AR based on dimension)
WebSocket to TypeScript Client (8003)
    ↓
Browser: Three.js, Babylon.js, Canvas visualization
```

## Quick Start

### Basic Usage: Listen to Chat and Visualize

```python
from HoloLoom.thirdeye import (
    ChatBridge,
    Thoughtspace,
    WebGLRenderer,
)

# Create components
bridge = ChatBridge(chat_ws_url="ws://localhost:8002")
thoughtspace = Thoughtspace()
renderer = WebGLRenderer()

# Listen to chat and visualize concepts
async for extraction in bridge.listen():
    # Add concepts to world
    for concept in extraction.concepts:
        thoughtspace.add_concept(concept)

        # Render transition if available
        if extraction.transition:
            commands = renderer.render_transition(
                extraction.transition,
                None,  # previous concept
                concept,
            )
            # Send commands to WebSocket client
            await send_to_client(commands)

    # Emit current state
    state = thoughtspace.get_state()
    await send_state_to_client(state.to_dict())
```

### Server Integration: FastAPI WebSocket Server

ThirdEye includes a production FastAPI server (`HoloLoom/thirdeye/server.py`) that handles the complete visualization pipeline:

```python
from HoloLoom.thirdeye.server import ThirdEyeServer
import uvicorn

# Create server
server = ThirdEyeServer(chat_ws_url="ws://localhost:8002")

# Start on port 8003
uvicorn.run(
    server.app,
    host="0.0.0.0",
    port=8003
)
```

**Endpoints**:
- `GET /health` - Server health check
- `GET /config` - Rendering configuration
- `WebSocket /ws` - Real-time concept streaming to clients
- `WebSocket /thoughtspace` - Full-screen Thoughtspace mode

### Three-Step Integration

1. **Chat Bridge** - Connect to chat and extract concepts
2. **Thoughtspace** - Manage concept world and transitions
3. **Renderer** - Generate render commands for 3D/2D visualization

Each component is independent and replaceable, following HoloLoom's "weaving metaphor."

## Key Components

| Component | Lines | Purpose |
|-----------|-------|---------|
| **concept.py** | 291 | `Concept` dataclass, concept types, semantic positioning |
| **transition.py** | 180+ | Smooth transitions between concepts (morph, blend, dissolve, etc.) |
| **chat_bridge.py** | 406 | Listen to chat WebSocket, extract concepts using patterns |
| **thoughtspace.py** | 573 | Manage concept world, camera modes, mode transitions (compact↔fullscreen) |
| **renderer_protocol.py** | 200+ | Abstract renderer interface for 2D/3D/AR rendering |
| **renderers/webgl.py** | 300+ | Three.js WebGL renderer for 3D visualization |
| **server.py** | 500+ | FastAPI WebSocket server orchestrating entire pipeline |
| **visualizers/** | 2000+ | Scene composers detecting context and rendering UI/narrative/architecture/data |
| **collab/** | 1000+ | Collaborative multi-user features (presence, sync, shared styles) |
| **generators/** | 1500+ | Image generation via Stable Diffusion backend |

**Total Architecture**: 30 Python files, ~13,883 lines of production code

## Main Classes & Functions

### Core Data Types

#### `Concept` (concept.py)
Semantic unit extracted from chat, visualized in 1D/2D/3D space.

```python
@dataclass
class Concept:
    id: str                          # Unique identifier
    label: str                       # Display name
    concept_type: ConceptType        # Type for visualization
    position: SemanticPosition       # 3D position + embedding
    importance: float                # 0.0-1.0, affects size
    confidence: float                # 0.0-1.0, affects opacity
    style: VisualStyle               # MINIMAL/STANDARD/RICH/IMMERSIVE
    connections: List[str]           # IDs of connected concepts
    description: Optional[str]       # Explanation
    metadata: Dict[str, Any]         # Custom data

    def touch(self) -> None          # Update last_active timestamp
    def to_dict() -> Dict[str, Any]  # Serialize for WebSocket
```

**Concept Types**:
- ENTITY - Named things (Thompson Sampling, Python)
- RELATIONSHIP - Connections between entities
- PROCESS - Algorithms, workflows, sequences
- COMPARISON - A vs B, tradeoffs
- HIERARCHY - Taxonomies, IS_A relationships
- TEMPORAL - Time patterns, evolution
- PROBABILITY - Uncertainty, distributions
- DECISION - Choice points, branches
- ACTIVATION - Attention flow, spreading activation
- MEMORY - Retrieval, storage, consolidation
- CLUSTER - Groups of related concepts
- THREAD - Narrative chains, reasoning paths
- WORLD - Full Thoughtspace visualization

#### `Transition` (transition.py)
Smooth visual morphing between concepts.

```python
@dataclass
class Transition:
    transition_type: TransitionType  # MORPH, BLEND, DISSOLVE_EMERGE, etc.
    config: TransitionConfig         # Duration, easing, visual params
    from_concept_id: Optional[str]   # Source concept
    to_concept_id: Optional[str]     # Target concept

    # Transition types (0-1000ms):
    # - MORPH: 400-800ms shape interpolation (close concepts)
    # - BLEND: 300-500ms color/opacity fade (close concepts)
    # - DISSOLVE_EMERGE: 600-1000ms fade out/in (far concepts)
    # - COLLAPSE_EXPAND: 500-800ms shrink/grow (far concepts)
    # - PULSE: 200ms emphasis without change
    # - BRANCH: 500ms split into multiple
    # - MERGE: 500ms combine into one
    # - ZOOM: 400ms camera moves
    # - CUT: 0ms instant
```

#### `ConceptWorld` (concept.py)
Collection of concepts forming a Thoughtspace world.

```python
@dataclass
class ConceptWorld:
    name: str                           # World name
    concepts: Dict[str, Concept]        # All concepts in world
    focal_concept_id: Optional[str]     # Current focus (camera target)

    def add_concept(concept: Concept) -> None
    def remove_concept(concept_id: str) -> Optional[Concept]
    def get_connected_concepts(concept_id: str) -> List[Concept]
    def set_focus(concept_id: str) -> None
```

#### `SemanticPosition` (concept.py)
Position in 228D semantic space projected to 3D for visualization.

```python
@dataclass
class SemanticPosition:
    x: float = 0.0                    # 3D projection X
    y: float = 0.0                    # 3D projection Y
    z: float = 0.0                    # 3D projection Z
    embedding: Optional[List[float]]   # Full 228D semantic embedding
    axes: Dict[str, float]             # Interpretable axes (16-axis)

    @classmethod
    def from_embedding(embedding: List[float]) -> SemanticPosition
```

### Chat Integration

#### `ChatBridge` (chat_bridge.py)
Listen to chat WebSocket and stream concept extractions.

```python
class ChatBridge:
    async def listen() -> AsyncIterator[ConceptExtraction]
        """Listen to chat and yield concept extractions."""

    async def stop() -> None
        """Stop listening."""
```

#### `ConceptExtractor` (chat_bridge.py)
Extract semantic concepts from chat messages.

```python
class ConceptExtractor:
    def extract(message: ChatMessage) -> ConceptExtraction
        """Extract concepts from a chat message."""

    # Pattern matching for:
    # - Comparisons: "vs", "compared to", "tradeoffs"
    # - Processes: "steps", "algorithm", "how to"
    # - Entities: Capitalized terms, technical words
```

**Detection Capabilities**:
- **Comparison patterns**: vs, versus, compared to, tradeoffs, pros/cons
- **Process patterns**: steps, algorithm, workflow, implementation
- **Entity extraction**: Named entities, technical terms
- **Topic extraction**: General topic when no specific concepts
- **Embedding-based positioning**: Optional semantic positioning via embeddings

### Visualization Management

#### `Thoughtspace` (thoughtspace.py)
Full-screen expandable world for 3D concept visualization.

```python
class Thoughtspace:
    def add_concept(concept: Concept) -> None
        """Add concept to world with semantic positioning."""

    def focus_concept(concept_id: str) -> Optional[Transition]
        """Focus camera on a concept."""

    def expand() -> Transition
        """Expand from compact (300px) to fullscreen."""

    def collapse() -> Transition
        """Collapse from fullscreen to compact."""

    def toggle_mode() -> Transition
        """Toggle between compact and fullscreen."""

    def relayout() -> None
        """Recompute positions for all concepts (force-directed)."""

    def set_camera_mode(mode: CameraMode) -> None
        """Set camera behavior: FOLLOW, FREE, ORBIT, or JOURNEY."""

    def orbit(angle_delta: float) -> None
        """Rotate camera around focal point."""

    def zoom(factor: float) -> None
        """Zoom camera in/out."""

    def get_state() -> ThoughtspaceState
        """Get complete state for streaming to client."""
```

**Display Modes**:
- **COMPACT**: 300px sidebar panel (default)
- **EXPANDED**: 50% of screen
- **FULLSCREEN**: Entire viewport
- **OVERLAY**: Transparent overlay on chat

**Camera Modes**:
- **FOLLOW**: Camera follows focal concept
- **FREE**: User-controlled camera (mouse drag)
- **ORBIT**: Orbit around focal point
- **JOURNEY**: Animated path through concepts

#### `SemanticPositioner` (thoughtspace.py)
Calculate semantic positions for concepts using force-directed layout.

```python
class SemanticPositioner:
    def position_concept(
        concept: Concept,
        existing_concepts: List[Concept]
    ) -> SemanticPosition
        """Calculate position for new concept based on similarity."""

    def reposition_all(concepts: List[Concept]) -> Dict[str, SemanticPosition]
        """Recompute positions for all concepts (50 force-directed iterations)."""
```

**Algorithm**:
- Uses embeddings for initial position (if available)
- Falls back to position by connections if no embedding
- Applies separation force to avoid overlaps
- Force-directed layout: repulsion between all pairs + attraction along edges
- 50 iterations with damping for convergence

### Rendering

#### `ConceptRenderer` Protocol (renderer_protocol.py)
Abstract interface for all renderers (2D, 3D, AR).

```python
@runtime_checkable
class ConceptRenderer(Protocol):
    def render_concept(concept: Concept) -> List[RenderCommand]
        """Render a single concept."""

    def render_transition(
        transition: Transition,
        from_concept: Optional[Concept],
        to_concept: Optional[Concept]
    ) -> List[RenderCommand]
        """Render smooth transition between concepts."""

    def render_world(world: ConceptWorld) -> List[RenderCommand]
        """Render entire concept world."""

    def render_frame() -> RenderFrame
        """Get current frame with all render commands."""
```

#### `WebGLRenderer` (renderers/webgl.py)
Three.js-based 3D renderer generating JSON commands.

```python
class WebGLRenderer(BaseConceptRenderer):
    def render_concept(concept: Concept) -> List[RenderCommand]
        """Generate Three.js geometry for concept sphere."""

    def render_transition(transition: Transition, ...) -> List[RenderCommand]
        """Generate animation keyframes for transition."""

    def render_world(world: ConceptWorld) -> List[RenderCommand]
        """Render all concepts + connections."""

    def render_edge(from_id: str, to_id: str, edge_type: str) -> RenderCommand
        """Render connection between concepts."""
```

**Color Palette** (soft colors on cream background):
- ENTITY: Soft blue (#3b82f6)
- RELATIONSHIP: Soft green (#10b981)
- PROCESS: Soft purple (#8b5cf6)
- COMPARISON: Soft amber (#f59e0b)
- HIERARCHY: Indigo (#6366f1)
- TEMPORAL: Cyan (#06b6d4)
- PROBABILITY: Pink (#ec4899)
- DECISION: Orange (#f97316)
- ACTIVATION: Yellow (#eab308)
- MEMORY: Teal (#14b8a6)
- CLUSTER: Purple (#a855f7)
- THREAD: Slate (#64748b)
- WORLD: Sky blue (#0ea5e9)

### Scene Composition

#### `SceneComposer` (visualizers/scene_composer.py)
Detect visualization context and compose appropriate scenes.

```python
class SceneComposer:
    def compose(
        extraction: ConceptExtraction,
        mode: Optional[VisualizationMode] = None
    ) -> VisualScene
        """Compose a scene for the given concepts."""
```

**Visualization Modes** (auto-detected from chat content):
- **UI_DESIGN**: UI mockups, wireframes, components (keywords: button, form, layout, etc.)
- **NARRATIVE**: Stories, scenes, dialogue (keywords: story, character, plot, dialogue)
- **ARCHITECTURE**: System diagrams, code structure (keywords: system, component, API, database)
- **DATA**: Charts, graphs, metrics (keywords: chart, metric, analytics, percentage)
- **OBJECT**: Physical objects, products (keywords: product, device, machine, furniture)
- **WORLD**: Environments, spaces (keywords: environment, landscape, map, location)
- **ABSTRACT**: Fallback - concept spheres for abstract ideas

**Detection Algorithm**:
1. Count keyword matches for each mode
2. Return mode with highest score (≥2 matches required)
3. Default to ABSTRACT if no strong signal

### Server & WebSocket

#### `ThirdEyeServer` (server.py)
FastAPI WebSocket server orchestrating the entire pipeline.

```python
class ThirdEyeServer:
    # Manages:
    # - ChatBridge listening to chat
    # - ConceptExtraction pipeline
    # - SceneComposition context detection
    # - Renderer command generation
    # - WebSocket client connections

    async def handle_client(websocket: WebSocket)
        """Handle new client connection."""

    async def broadcast_state(state: ThoughtspaceState)
        """Send state to all connected clients."""
```

**Server Endpoints**:
- `GET /health` - Check server status
- `GET /config` - Get rendering configuration
- `GET /docs` - OpenAPI documentation
- `WebSocket /ws` - Real-time concept streaming
- `WebSocket /thoughtspace` - Full-screen mode coordination

## Factory Functions

Create instances with sensible defaults:

```python
# Concepts
create_entity_concept(id, label, importance, description, embedding)
create_comparison_concept(id, label, concepts_compared, importance)
create_process_concept(id, label, steps, importance)

# Chat
bridge = await create_chat_bridge(chat_ws_url, embedder)

# Thoughtspace
thoughtspace = create_thoughtspace(on_state_change)

# Renderers
renderer = create_webgl_renderer(config)

# Server
server = ThirdEyeServer(chat_ws_url, embedder)
```

## Advanced Features

### Multi-User Collaboration (collab/)
- **Scene Presence**: Track who's viewing what
- **Scene Sync**: Multi-user state synchronization
- **Scene Roles**: Permission system (viewer, editor, moderator)
- **Activity Feed**: Real-time activity log
- **Shared Styles**: Collaborative style consistency

### Image Generation (generators/)
- **Stable Diffusion Backend**: Generate images from concepts
- **Prompt Builder**: Automatically craft image prompts from chat
- **Integration**: Embed generated images in visualizations

### Visualizers (visualizers/)
- **UI Visualizer**: Render UI components and layouts
- **Narrative Visualizer**: Cinematic scene composition
- **Architecture Visualizer**: System diagram generation
- **Scene Composer**: Context-aware scene selection
- **Layer Composer**: Multi-layer composition
- **Semantic Extractor**: Extract semantic information from text
- **Visual Styles**: Consistent visual design system
- **Conversation Memory**: Track visualization history

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Concept extraction | ~10-50ms | Pattern matching + optional embedding |
| Transition calculation | <5ms | Semantic distance → animation type |
| Positioning (1 concept) | <1ms | Embedding projection or connection-based |
| Reposition all (50 concepts) | ~50ms | 50 force-directed iterations |
| WebGL render frame | ~16ms | 60 FPS @ 1920×1080 |
| Scene composition | ~20-100ms | Visualization detection + scene building |
| WebSocket broadcast | <5ms | Per client connection |

**Optimization Strategies**:
- Concept embedding caching (if embedder provided)
- Force-directed layout damping for convergence
- Batched WebSocket updates
- LOD (Level of Detail) rendering for >50 concepts
- Automatic culling when max_concepts exceeded

## Integration with HoloLoom

ThirdEye integrates seamlessly with other HoloLoom systems:

### 1. Memory System
- **Concept extraction uses**: Semantic embeddings from `HoloLoom.embedding/`
- **Positioning uses**: Embedding similarity + Matryoshka multi-scale embeddings
- **Context enrichment**: Integration with `HoloLoom.memory/` for knowledge graph

### 2. Chat System
- **Input**: WebSocket connection to multithreaded chat (`ws://localhost:8002`)
- **Chat Message Structure**: Compatible with HoloLoom chat protocol
- **Role Support**: USER, ASSISTANT, SYSTEM message types

### 3. Alignment Framework
- **Safety Gating**: Optional safety checks on rendered visualizations
- **Deception Detection**: Track concept authenticity
- **Audit Trail**: Log all visualizations for compliance

### 4. Agentic Reasoning
- **Multi-Query Tracking**: Visualize reasoning steps
- **Decision Visualization**: Render convergence engine decisions
- **Exploration Paths**: Show Thompson Sampling exploration

## When to Use ThirdEye

**✅ Use ThirdEye when you need**:
- Chat-integrated visualization that evolves with conversation
- Context-aware rendering (automatically detect UI/narrative/architecture/data)
- Smooth concept transitions as topics change
- Multi-user collaborative visualization
- Full-screen immersive concept exploration
- 3D semantic space navigation
- Visual explanation of abstract concepts
- Real-time WebSocket streaming to web clients

**✅ When to integrate with HoloLoom**:
- Building conversational AI applications with visual output
- Creating interactive knowledge exploration tools
- Rendering semantic knowledge graphs in real-time
- Multi-modal reasoning (chat + visualization)
- Educational tools explaining complex concepts visually

**🟡 Consider alternatives when**:
- Static visualization (no chat interaction needed) → Use standard visualization libraries
- Real-time performance critical (<10ms latency) → Use lightweight 2D renderer only
- Limited to 2D → Use SVG/Canvas renderers instead of WebGL
- No WebSocket capability → Use local-only mode (mock chat bridge)

**❌ Not suitable for**:
- Batch visualization of pre-recorded data (designed for streaming)
- Very large graphs (>500 concepts) → Use specialized graph visualization tools
- Mobile-only (heavy WebGL dependency) → Use lightweight 2D renderer

## Configuration

### RenderConfig
Control rendering parameters:

```python
from HoloLoom.thirdeye.renderer_protocol import RenderConfig, RenderDimension, RenderQuality

config = RenderConfig(
    dimension=RenderDimension.THREE_D,      # 1D/2D/3D/AR
    quality=RenderQuality.MEDIUM,           # LOW/MEDIUM/HIGH/ULTRA
    width=800,                              # Viewport width
    height=600,                             # Viewport height
    background_color="#f8fafc",             # Cream background
    ambient_light=0.6,                      # Lighting intensity
    enable_shadows=True,                    # 3D shadows
    enable_bloom=False,                     # Glow effects
    enable_antialiasing=True,               # Smooth edges
    target_fps=60,                          # Target frame rate
    max_concepts=100,                       # Culling limit
    show_labels=True,                       # Show concept names
)
```

### ThoughtspaceMode
Control display size:

```python
thoughtspace.toggle_mode()  # Compact ↔ Fullscreen
thoughtspace.expand()       # Compact → Fullscreen (600ms)
thoughtspace.collapse()     # Fullscreen → Compact (400ms)
```

### CameraMode
Control camera behavior:

```python
thoughtspace.set_camera_mode(CameraMode.FOLLOW)   # Automatic following
thoughtspace.set_camera_mode(CameraMode.FREE)     # User-controlled
thoughtspace.set_camera_mode(CameraMode.ORBIT)    # Orbit around focal
thoughtspace.set_camera_mode(CameraMode.JOURNEY)  # Animated path
```

## Examples

### Example 1: Simple Chat Visualization

```python
import asyncio
from HoloLoom.thirdeye import ChatBridge, Thoughtspace, WebGLRenderer

async def main():
    # Create components
    bridge = ChatBridge(chat_ws_url="ws://localhost:8002")
    thoughtspace = Thoughtspace()
    renderer = WebGLRenderer()

    # Listen to chat
    async for extraction in bridge.listen():
        print(f"Found {len(extraction.concepts)} concepts")

        # Add to visualization
        for concept in extraction.concepts:
            thoughtspace.add_concept(concept)

            # Print concept info
            print(f"  - {concept.label} ({concept.concept_type.name})")
            print(f"    Position: ({concept.position.x:.1f}, {concept.position.y:.1f}, {concept.position.z:.1f})")
            print(f"    Importance: {concept.importance:.2f}")

        # Render transition
        if extraction.transition:
            print(f"Transition: {extraction.transition.transition_type.value}")

asyncio.run(main())
```

### Example 2: Full-Screen Thoughtspace

```python
from HoloLoom.thirdeye import create_thoughtspace, create_entity_concept

thoughtspace = create_thoughtspace()

# Add concepts
concept1 = create_entity_concept("ts1", "Thompson Sampling", 0.8)
concept2 = create_entity_concept("ucb", "Upper Confidence Bound", 0.7)
concept2.connections = ["ts1"]  # Connect them

thoughtspace.add_concept(concept1)
thoughtspace.add_concept(concept2)

# Expand to fullscreen
transition = thoughtspace.expand()
print(f"Transition: {transition.transition_type.value} ({transition.config.duration_ms}ms)")

# Set camera mode
thoughtspace.set_camera_mode(CameraMode.ORBIT)

# Get current state for streaming
state = thoughtspace.get_state()
print(f"Mode: {state.mode.value}, Concepts: {len(state.world.concepts)}")
```

### Example 3: Custom Concept Extraction

```python
from HoloLoom.thirdeye import ConceptExtractor, ChatMessage, MessageRole

# Create extractor with optional embedding function
def my_embedder(text: str):
    # Return 228D embedding (mock for example)
    return [0.1] * 228

extractor = ConceptExtractor(embedder=my_embedder)

# Extract from a message
message = ChatMessage(
    id="1",
    role=MessageRole.USER,
    content="What is the difference between Thompson Sampling vs UCB?"
)

extraction = extractor.extract(message)

print(f"Found {len(extraction.concepts)} concepts:")
for concept in extraction.concepts:
    print(f"  - {concept.label}")
    print(f"    Type: {concept.concept_type.name}")
    print(f"    Confidence: {concept.confidence:.2f}")
    print(f"    Connections: {concept.connections}")
```

### Example 4: Server Integration

```python
from HoloLoom.thirdeye.server import ThirdEyeServer
import uvicorn

# Create server
server = ThirdEyeServer(
    chat_ws_url="ws://localhost:8002",
    embedder=None  # Optional: provide embedding function
)

# Start server
if __name__ == "__main__":
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8003,
        reload=False  # Set True for development
    )
```

Then access from TypeScript client:

```typescript
const ws = new WebSocket('ws://localhost:8003/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'state') {
        // Update visualization state
        updateThoughtspace(data.state);
    } else if (data.type === 'concept') {
        // Render new concept
        renderConcept(data.concept);
    }
};
```

## Testing

The system includes comprehensive tests for all major components:

```bash
# Run all thirdeye tests
pytest HoloLoom/thirdeye/ -v

# Test specific components
pytest HoloLoom/thirdeye/tests/test_concept.py -v
pytest HoloLoom/thirdeye/tests/test_chat_bridge.py -v
pytest HoloLoom/thirdeye/tests/test_thoughtspace.py -v
pytest HoloLoom/thirdeye/tests/test_renderer.py -v
```

## Roadmap

**Current** (December 2025):
- ✅ Core concept extraction and visualization
- ✅ 3D WebGL rendering with Three.js
- ✅ Multi-user collaboration features
- ✅ Image generation backend
- ✅ Context-aware scene composition

**Future** (Phase 6+):
- Streaming video output (MP4 recording)
- AR/WebXR rendering pipeline
- Advanced physics simulation (cloth, particles)
- Voice-driven visualization
- Gesture recognition for spatial interaction
- Multi-room collaborative environments
- Real-time photogrammetry integration
- Neural radiance field (NeRF) rendering

## Architecture Documentation

See [COLLAB_AR_XR_ROADMAP.md](COLLAB_AR_XR_ROADMAP.md) for:
- Multi-user synchronization architecture
- AR/WebXR rendering pipeline
- Scene persistence and caching
- Performance optimization strategies
- Future expansion roadmap

## Dependencies

**Required**:
- Python 3.8+
- FastAPI
- uvicorn
- pydantic
- asyncio

**Optional**:
- websockets (for chat bridge)
- sentence-transformers (for embeddings)
- Pillow (for image processing)
- numpy, scipy (for advanced math)

**Client-Side**:
- Three.js (3D rendering)
- Babylon.js (alternative 3D)
- Canvas/SVG APIs (2D fallback)
- WebXR (AR/VR)

## Contributing

ThirdEye follows HoloLoom's development philosophy: **"Reliable Systems: Safety First"**

- All components have graceful fallbacks
- WebSocket failures don't crash the server
- Renderers degrade to simpler dimensions automatically
- Missing embeddings fall back to connection-based positioning
- Full type hints and comprehensive error messages

When extending ThirdEye:
1. Add comprehensive docstrings
2. Implement factory functions for clean APIs
3. Use dataclasses for configuration
4. Support both sync and async where possible
5. Include fallback implementations
6. Add tests for new features

## License

Part of the HoloLoom project. See root `LICENSE` file.

## Citation

If you use ThirdEye in research, cite as:

```bibtex
@software{hololoom_thirdeye_2025,
  title={ThirdEye: Chat-Integrated 3D Concept Visualization Engine},
  author={HoloLoom Team},
  year={2025},
  url={https://github.com/your-repo/HoloLoom/thirdeye}
}
```

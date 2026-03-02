# ThirdEye → Collaboration + AR/XR Integration Roadmap

**Created**: 2025-12-01
**Status**: Planning
**Estimated Duration**: 12-18 weeks across 6 phases

---

## Executive Summary

This roadmap outlines the integration of ThirdEye's scene visualization system with HoloLoom's existing Collaboration (Phase 3) and Spatial AR/XR (Phase 4) infrastructure. The goal is to enable **shared, immersive visualization experiences** where multiple users can see, interact with, and co-create visual representations of conversations in AR/VR environments.

### Current State

**ThirdEye (Built)**:
- ✅ Scene composition with 6 visualization modes (UI, Narrative, Architecture, Data, Object, World)
- ✅ Conversation memory with context aggregation
- ✅ Semantic extraction (entities, relationships, intents)
- ✅ Visual styles (Cinematic, Blueprint, Neon, etc.)
- ✅ Layer composition for depth and complexity
- ✅ Image generation via Stable Diffusion (ComfyUI/A1111)

**Collaboration Module (Exists)**:
- ✅ User management with roles and profiles
- ✅ Session management (create, join, leave)
- ✅ Presence tracking (cursors, focus, activity status)
- ✅ State synchronization with CRDT conflict resolution
- ✅ Attribution tracking for contributions
- ✅ WebRTC voice/video communication

**Spatial Module (Exists)**:
- ✅ WebXR knowledge graph visualization
- ✅ AR knowledge overlay system (8 styles: CARD, HOLOGRAM, MEMORY_PALACE, etc.)
- ✅ Hand tracking with 20+ gestures
- ✅ Gaze tracking and spatial UI
- ✅ Avatar system for user representation
- ✅ Spatial audio for 3D sound
- ✅ Physics engine for object interaction

---

## Phase 1: Shared Scene State (Weeks 1-2)

**Goal**: Enable multiple users to see the same ThirdEye scene in real-time.

### 1.1 Scene State Synchronization

Create bridge between ThirdEye scenes and Collaboration sync:

```python
# New file: HoloLoom/thirdeye/collab/scene_sync.py

from HoloLoom.collaboration import StateSynchronizer, Operation
from HoloLoom.thirdeye.visualizers import VisualScene, VisualElement

class CollaborativeScene:
    """
    Wraps VisualScene with real-time sync capabilities.
    """
    def __init__(self, scene: VisualScene, session_id: str):
        self.scene = scene
        self.sync = StateSynchronizer(session_id)

    async def add_element(self, element: VisualElement, user_id: str):
        """Add element and broadcast to all participants."""
        self.scene.elements.append(element)
        await self.sync.broadcast(Operation(
            type=OperationType.ADD,
            path=f"scene.elements.{element.id}",
            value=element.to_dict(),
            user_id=user_id
        ))

    async def update_element(self, element_id: str, changes: dict, user_id: str):
        """Update element with conflict resolution."""
        # CRDT handles concurrent edits
        await self.sync.apply(Operation(
            type=OperationType.UPDATE,
            path=f"scene.elements.{element_id}",
            value=changes,
            user_id=user_id
        ))
```

### 1.2 Presence in Scene Context

Show where collaborators are focusing:

```python
# Extend presence to track scene focus
@dataclass
class SceneFocus:
    user_id: str
    element_id: Optional[str]  # Which element they're focused on
    position_3d: Vector3       # Where they're looking in scene
    cursor_style: str          # "select", "draw", "annotate"
    color: Color               # User's assigned color
```

### 1.3 Deliverables

- [ ] `CollaborativeScene` class wrapping sync
- [ ] Scene-aware presence tracking
- [ ] Conflict resolution for concurrent edits
- [ ] Tests: 2+ users editing same scene

---

## Phase 2: Multi-User Scene Composition (Weeks 3-4)

**Goal**: Enable collaborative scene building where users can co-create visualizations.

### 2.1 Role-Based Scene Editing

```python
class SceneRole(Enum):
    VIEWER = "viewer"       # Can see, cannot edit
    CONTRIBUTOR = "contributor"  # Can add elements
    EDITOR = "editor"       # Can modify any element
    OWNER = "owner"         # Full control + permissions
```

### 2.2 Attribution for Scene Elements

Track who created/modified what:

```python
@dataclass
class SceneContribution:
    element_id: str
    contributor_id: str
    contribution_type: ContributionType  # CREATE, MODIFY, DELETE
    timestamp: datetime
    quality_rating: Optional[QualityRating]
```

### 2.3 Shared Style Presets

```python
class SharedStyleLibrary:
    """Team-managed visual style presets."""

    async def share_style(self, style: VisualStyle, user_id: str):
        """Share a custom style with the team."""

    async def vote_style(self, style_id: str, user_id: str, vote: int):
        """Upvote/downvote shared styles."""
```

### 2.4 Deliverables

- [ ] Role-based access control for scenes
- [ ] Per-element attribution tracking
- [ ] Shared style library with voting
- [ ] Activity feed showing who did what

---

## Phase 3: AR Scene Overlay (Weeks 5-7)

**Goal**: Render ThirdEye scenes as AR overlays in physical space.

### 3.1 Scene-to-Overlay Converter

Bridge ThirdEye's 2D/3D scenes to Spatial's AR overlay system:

```python
# New file: HoloLoom/thirdeye/spatial/overlay_bridge.py

from HoloLoom.thirdeye.visualizers import VisualScene, VisualElement
from HoloLoom.spatial.knowledge_overlay import KnowledgeNodeOverlay, OverlayStyle

class SceneToOverlay:
    """
    Converts ThirdEye scenes to AR knowledge overlays.
    """

    # Map ThirdEye element types to AR overlay styles
    STYLE_MAP = {
        'panel': OverlayStyle.CARD,
        'character': OverlayStyle.HOLOGRAM,
        'box': OverlayStyle.SPHERE,
        'chart': OverlayStyle.RICH,
        'object': OverlayStyle.GLOW,
        'environment': OverlayStyle.CONSTELLATION,
    }

    def convert(self, scene: VisualScene) -> List[KnowledgeNodeOverlay]:
        """Convert scene elements to AR overlays."""
        overlays = []
        for element in scene.elements:
            overlay = KnowledgeNodeOverlay(
                overlay_id=f"thirdeye_{element.id}",
                node_id=element.id,
                title=element.label,
                content=element.content,
                position=self._to_vector3(element.position),
                style=self.STYLE_MAP.get(element.element_type, OverlayStyle.CARD),
                primary_color=self._style_to_color(scene.style),
            )
            overlays.append(overlay)
        return overlays
```

### 3.2 Spatial Layout for Scenes

Different layout algorithms for different scene types:

```python
class SceneLayout:
    """Position scene elements in 3D space."""

    def layout_narrative(self, scene: VisualScene) -> Dict[str, Vector3]:
        """Timeline-style layout for stories."""
        # Characters on the left, environments behind, events in sequence

    def layout_architecture(self, scene: VisualScene) -> Dict[str, Vector3]:
        """Hierarchical layout for system diagrams."""
        # Services at top, databases at bottom, connections visible

    def layout_ui_design(self, scene: VisualScene) -> Dict[str, Vector3]:
        """Flat layout mimicking screen space."""
        # Components arranged as they would appear on screen
```

### 3.3 Gesture Interaction with Scenes

```python
# Map gestures to scene actions
GESTURE_ACTIONS = {
    GestureType.POINT: "select_element",
    GestureType.PINCH: "resize_element",
    GestureType.GRAB: "move_element",
    GestureType.THUMBS_UP: "approve_scene",
    GestureType.SWIPE_RIGHT: "next_scene_version",
    GestureType.SWIPE_LEFT: "previous_scene_version",
    GestureType.SCALE_UP: "zoom_into_element",
    GestureType.SCALE_DOWN: "zoom_out",
}
```

### 3.4 Deliverables

- [ ] SceneToOverlay converter
- [ ] Layout algorithms for each visualization mode
- [ ] Gesture controls for scene manipulation
- [ ] Gaze-based element selection
- [ ] Demo: View ThirdEye scene in AR

---

## Phase 4: Collaborative AR Sessions (Weeks 8-10)

**Goal**: Multiple users in shared AR space seeing and editing scenes together.

### 4.1 Shared AR Session

```python
# New file: HoloLoom/thirdeye/spatial/collab_ar_session.py

class CollaborativeARScene:
    """
    Full AR session with multiple users seeing same scene.
    """
    def __init__(self, session_id: str):
        self.session = SessionManager.get(session_id)
        self.collab_scene = CollaborativeScene(...)
        self.presence = PresenceManager(session_id)
        self.voice = VoiceRoom(session_id)
        self.overlays: List[KnowledgeNodeOverlay] = []

    async def join(self, user_id: str, webxr_session):
        """User joins AR session."""
        # Add to session
        await self.session.add_participant(user_id)

        # Create avatar at user's position
        avatar = await AvatarManager.create(user_id, webxr_session.position)

        # Sync current scene state
        await self._sync_scene_to_user(user_id)

        # Join voice room
        await self.voice.join(user_id)

    async def broadcast_scene_update(self, element_id: str, changes: dict):
        """Send scene changes to all participants."""
        for participant in self.session.participants:
            await self._push_overlay_update(participant.id, element_id, changes)
```

### 4.2 Spatial Presence with Avatars

```python
class ARScenePresence:
    """
    Track where users are in the AR scene.
    """
    def __init__(self):
        self.avatars: Dict[str, Avatar] = {}
        self.focus_indicators: Dict[str, FocusRay] = {}

    def update_user_position(self, user_id: str, transform: Transform):
        """Update avatar position from WebXR tracking."""

    def show_user_focus(self, user_id: str, target_element: str):
        """Show ray from avatar to element they're focused on."""
```

### 4.3 Voice Annotations

```python
class VoiceAnnotation:
    """
    Attach voice notes to scene elements.
    """
    element_id: str
    audio_data: bytes
    transcript: Optional[str]
    speaker_id: str
    timestamp: datetime
    position: Vector3  # Where annotation appears
```

### 4.4 Deliverables

- [ ] CollaborativeARScene orchestrator
- [ ] Multi-user avatar rendering
- [ ] Spatial presence indicators (focus rays)
- [ ] Voice annotation system
- [ ] Demo: 2+ users in shared AR scene

---

## Phase 5: VR Memory Palace Scenes (Weeks 11-13)

**Goal**: Transform ThirdEye scenes into navigable VR memory palaces.

### 5.1 Scene as Memory Palace

```python
class MemoryPalaceScene:
    """
    Converts ThirdEye scenes into VR memory palace rooms.
    """

    def scene_to_room(self, scene: VisualScene) -> VRRoom:
        """
        Transform scene into navigable VR room.

        - Elements become 3D objects placed around room
        - Relationships become visual connections (beams, paths)
        - Style determines room aesthetic
        """
        room = VRRoom(
            style=self._style_to_room_theme(scene.style),
            size=self._calculate_room_size(scene),
        )

        for element in scene.elements:
            obj = self._element_to_object(element)
            room.place_object(obj, self._optimal_position(element, room))

        return room
```

### 5.2 Teleportation Between Scenes

```python
class SceneNavigation:
    """Navigate between related scenes."""

    async def create_portal(self, from_scene: str, to_scene: str, position: Vector3):
        """Create portal connecting two scenes."""

    async def teleport_user(self, user_id: str, target_scene: str):
        """Teleport user to different scene."""
```

### 5.3 Physics-Based Interaction

```python
class PhysicsScene:
    """
    Enable physics for scene elements in VR.
    """
    def enable_grabbing(self, element_id: str):
        """Make element grabbable."""

    def enable_throwing(self, element_id: str):
        """Make element throwable (fun for brainstorming!)."""

    def create_connection(self, from_id: str, to_id: str):
        """Create physics-based visual connection (rope/beam)."""
```

### 5.4 Deliverables

- [ ] MemoryPalaceScene converter
- [ ] Room generation from scene data
- [ ] Portal system for scene navigation
- [ ] Physics-enabled elements
- [ ] Demo: Walk through narrative as VR experience

---

## Phase 6: Image Generation in AR/VR (Weeks 14-16)

**Goal**: Generate and display AI images in real-time within AR/VR scenes.

### 6.1 Live Image Generation Panel

```python
class ARImageGenerator:
    """
    Generate images from scene context and display in AR.
    """
    def __init__(self):
        self.generator = ImageGenerator()
        self.panel = None  # AR panel showing generated image

    async def generate_for_element(self, element: VisualElement):
        """
        Generate image based on element description.
        Display on floating AR panel.
        """
        # Build prompt from element context
        prompt = f"{element.label}: {element.content}"

        # Generate image
        result = await self.generator.generate_from_text(prompt)

        # Display near element
        self.panel = ARImagePanel(
            image=result,
            position=element.position + Vector3(0.5, 0, 0),  # Offset to right
            size=(0.6, 0.4),  # meters
        )
```

### 6.2 Collaborative Image Curation

```python
class SharedImageGallery:
    """
    Team-curated gallery of generated images.
    """
    async def save_to_gallery(self, image: GeneratedImage, scene_id: str, user_id: str):
        """Save generated image to shared gallery."""

    async def vote(self, image_id: str, user_id: str, vote: int):
        """Upvote/downvote images."""

    async def get_best_for_scene(self, scene_id: str) -> List[GeneratedImage]:
        """Get top-rated images for a scene."""
```

### 6.3 Image as AR Texture

```python
class ARTexturedElement:
    """
    Apply generated images as textures to AR objects.
    """
    async def texture_element(self, element: VisualElement, image: GeneratedImage):
        """Apply image as texture to 3D element surface."""

    async def create_image_frame(self, image: GeneratedImage, position: Vector3):
        """Create floating framed image in AR space."""
```

### 6.4 Deliverables

- [ ] ARImageGenerator with floating panels
- [ ] Real-time generation triggered by gestures
- [ ] SharedImageGallery with voting
- [ ] Image texturing for 3D elements
- [ ] Demo: Generate and view images in AR

---

## Phase 7: Mobile AR Integration (Weeks 17-18)

**Goal**: Bring collaborative scenes to mobile AR (ARCore/ARKit).

### 7.1 Mobile Scene Viewer

```python
# Integrate with existing MobileSpatialUIManager
class MobileSceneViewer:
    """
    View and interact with ThirdEye scenes on mobile AR.
    """
    def __init__(self, arcore_session):
        self.ar = arcore_session
        self.ui = MobileSpatialUIManager()

    async def load_scene(self, scene_id: str):
        """Load and render scene in mobile AR."""

    def enable_touch_editing(self):
        """Enable touch-based scene editing."""
        # Tap to select
        # Drag to move
        # Pinch to resize
        # Two-finger rotate
```

### 7.2 Cross-Platform Sync

```python
class CrossPlatformSession:
    """
    Sync scenes across desktop, mobile, and VR.
    """
    async def sync_scene_state(self, session_id: str):
        """Ensure all platforms see same scene."""

    async def adapt_for_platform(self, scene: VisualScene, platform: str):
        """Adapt scene complexity for platform capabilities."""
        # Mobile: Simpler meshes, fewer particles
        # VR: Full fidelity
        # Desktop: 2D with depth cues
```

### 7.3 Deliverables

- [ ] MobileSceneViewer with touch controls
- [ ] Cross-platform state synchronization
- [ ] Platform-adaptive rendering
- [ ] Demo: Mobile user joins VR session

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ThirdEye AR/XR Stack                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Application Layer                          │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │    │
│  │  │   Desktop    │ │   Mobile AR  │ │     VR Headset       │ │    │
│  │  │   Browser    │ │  (ARKit/Core)│ │  (Quest/Vision Pro)  │ │    │
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                               │                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Session Layer                              │    │
│  │  ┌────────────────────────────────────────────────────────┐  │    │
│  │  │  CollaborativeARScene                                   │  │    │
│  │  │  - Multi-user session management                        │  │    │
│  │  │  - Presence tracking (avatars, focus rays)             │  │    │
│  │  │  - Voice/video communication                            │  │    │
│  │  │  - State synchronization (CRDT)                         │  │    │
│  │  └────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                               │                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Scene Layer                                │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │    │
│  │  │ ThirdEye    │ │ Scene-to-   │ │  Memory Palace          │ │    │
│  │  │ Composer    │→│ Overlay     │→│  Generator              │ │    │
│  │  │             │ │ Converter   │ │                         │ │    │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                               │                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Rendering Layer                            │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │    │
│  │  │ AR Overlay  │ │ VR Room     │ │ Image Gen   │            │    │
│  │  │ System      │ │ Renderer    │ │ (SD/Comfy)  │            │    │
│  │  └─────────────┘ └─────────────┘ └─────────────┘            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                               │                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Spatial Layer (Existing)                   │    │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐ │    │
│  │  │ WebXR  │ │ Hand   │ │ Gaze   │ │ Spatial│ │ Physics    │ │    │
│  │  │ Graph  │ │ Track  │ │ Track  │ │ Audio  │ │ Engine     │ │    │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
User Conversation
       │
       ▼
┌──────────────────┐
│ Enhanced Composer │  ← Memory, Semantics, Styles, Layers
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   VisualScene    │  ← Elements, Connections, Metadata
└────────┬─────────┘
         │
    ┌────┴────┐
    │ Sync?   │
    └────┬────┘
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
┌──────────┐           ┌──────────────┐
│ Local    │           │ Collaborative │
│ Render   │           │ Session       │
└────┬─────┘           └───────┬───────┘
     │                         │
     │                    ┌────┴────┐
     │                    │ Sync to │
     │                    │ Others  │
     │                    └────┬────┘
     │                         │
     ▼                         ▼
┌─────────────────────────────────┐
│      Scene-to-Overlay           │
│      Converter                  │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      │ Platform?   │
      └──────┬──────┘
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐
│ AR   │ │ VR   │ │Mobile│
│Overlay│ │Room  │ │ AR   │
└──────┘ └──────┘ └──────┘
```

---

## Success Metrics

### Phase 1-2 (Collaboration)
- [ ] 2+ users can see same scene in real-time
- [ ] Latency <100ms for sync updates
- [ ] Conflict resolution success rate >95%
- [ ] Attribution tracking for all edits

### Phase 3-4 (AR)
- [ ] ThirdEye scenes render as AR overlays
- [ ] Hand gesture recognition >90% accuracy
- [ ] Multi-user AR session stable for 1+ hour
- [ ] Voice annotations with <1s latency

### Phase 5-6 (VR + Image Gen)
- [ ] Memory palace navigation is intuitive
- [ ] Image generation completes <30s
- [ ] Generated images visible to all users
- [ ] Cross-platform sync works reliably

### Phase 7 (Mobile)
- [ ] Mobile AR matches desktop scene
- [ ] Touch controls feel natural
- [ ] Battery-efficient rendering
- [ ] Works on iPhone 12+ and Pixel 6+

---

## Dependencies

### Required
- **WebXR API** - Browser XR support
- **Three.js** or **Babylon.js** - 3D rendering
- **WebRTC** - Real-time communication
- **ComfyUI/A1111** - Image generation backend

### Optional
- **Apple Vision Pro SDK** - Native visionOS support
- **Meta Quest SDK** - Native Quest support
- **ARCore/ARKit** - Mobile AR

### HoloLoom Modules (Already Built)
- `HoloLoom.collaboration` - Full session/sync/presence
- `HoloLoom.spatial` - Full WebXR/hand/gaze/avatar
- `HoloLoom.thirdeye` - Visualization engine

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| WebXR browser support | Medium | High | Progressive enhancement, fallback to 2D |
| Multi-user sync latency | Medium | Medium | CRDT-based conflict resolution |
| Image gen too slow | High | Medium | Pre-generate, cache, lower resolution option |
| Mobile battery drain | Medium | Low | Adaptive quality, background throttling |
| Cross-platform bugs | High | Medium | Extensive testing matrix |

---

## Getting Started

### Prerequisites
```bash
# Ensure ThirdEye is working
PYTHONPATH=. python demos/demo_thirdeye_enhanced.py

# Ensure image generation is working
PYTHONPATH=. python demos/demo_thirdeye_image_gen.py

# Check collaboration module
python -c "from HoloLoom.collaboration import SessionManager; print('OK')"

# Check spatial module
python -c "from HoloLoom.spatial import WebXRKnowledgeGraph; print('OK')"
```

### Phase 1 First Steps
1. Create `HoloLoom/thirdeye/collab/` directory
2. Implement `CollaborativeScene` class
3. Add scene sync to existing session infrastructure
4. Create demo showing 2 users viewing same scene

---

## Appendix: File Structure

```
HoloLoom/thirdeye/
├── collab/                      # Phase 1-2
│   ├── __init__.py
│   ├── scene_sync.py           # CollaborativeScene
│   ├── scene_presence.py       # SceneFocus, presence
│   ├── scene_roles.py          # SceneRole, permissions
│   └── shared_styles.py        # SharedStyleLibrary
│
├── spatial/                     # Phase 3-4
│   ├── __init__.py
│   ├── overlay_bridge.py       # SceneToOverlay
│   ├── scene_layout.py         # Layout algorithms
│   ├── gesture_controls.py     # Gesture → action mapping
│   ├── collab_ar_session.py    # CollaborativeARScene
│   └── voice_annotations.py    # VoiceAnnotation
│
├── vr/                          # Phase 5
│   ├── __init__.py
│   ├── memory_palace.py        # MemoryPalaceScene
│   ├── room_generator.py       # VRRoom generation
│   ├── navigation.py           # Portals, teleportation
│   └── physics_scene.py        # Physics interaction
│
├── ar_image/                    # Phase 6
│   ├── __init__.py
│   ├── ar_generator.py         # ARImageGenerator
│   ├── image_gallery.py        # SharedImageGallery
│   └── texture_mapping.py      # ARTexturedElement
│
└── mobile/                      # Phase 7
    ├── __init__.py
    ├── mobile_viewer.py        # MobileSceneViewer
    └── cross_platform.py       # CrossPlatformSession
```

---

## Conclusion

This roadmap leverages HoloLoom's existing collaboration and spatial infrastructure to extend ThirdEye into a fully immersive, multi-user AR/VR experience. The phased approach allows incremental delivery while building toward the full vision of collaborative concept visualization in 3D space.

**Total Estimated Effort**: 12-18 weeks
**Team Size**: 2-3 developers
**Primary Technologies**: WebXR, WebRTC, Three.js/Babylon.js, Stable Diffusion

---

*Last Updated: 2025-12-01*

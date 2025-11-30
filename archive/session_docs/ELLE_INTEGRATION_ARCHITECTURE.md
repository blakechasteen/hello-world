# Elle Integration Architecture

**Version**: 1.0.0
**Date**: November 15, 2025
**Status**: 🏗️ Design Complete → Implementation Ready

---

## Executive Summary

Integration architecture for connecting HoloLoom VoiceAgent with Elle AR assistant, enabling voice-controlled augmented reality interactions with bidirectional audio, spatial awareness, and multimodal responses.

**Key Capabilities**:
- 🎙️ Voice-controlled AR commands
- 👁️ Gaze-based selection with voice confirmation
- 🎧 Spatial 3D audio positioning
- 🤝 Multimodal responses (voice + AR visualization)
- 🧠 Context-aware conversations with AR state

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Elle AR Assistant                       │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ AR Display │  │ Spatial      │  │  Gesture/Gaze     │  │
│  │  Engine    │  │ Audio Engine │  │  Tracking         │  │
│  └──────┬─────┘  └───────┬──────┘  └─────────┬─────────┘  │
│         │                 │                    │             │
│         └─────────────────┴────────────────────┘             │
│                           │                                  │
│                    ┌──────▼────────┐                        │
│                    │  ElleBridge   │                        │
│                    └──────┬────────┘                        │
└───────────────────────────┼─────────────────────────────────┘
                            │
                ┌───────────▼──────────┐
                │  VoiceAgent System   │
                │  ┌─────────────────┐ │
                │  │ Voice Input     │ │
                │  │ (VAD + Whisper) │ │
                │  └────────┬────────┘ │
                │           │          │
                │  ┌────────▼────────┐ │
                │  │ CommandRouter   │ │
                │  │  (Intent NLU)   │ │
                │  └────────┬────────┘ │
                │           │          │
                │  ┌────────▼────────┐ │
                │  │  HoloLoom       │ │
                │  │  Orchestrator   │ │
                │  └────────┬────────┘ │
                │           │          │
                │  ┌────────▼────────┐ │
                │  │ Response        │ │
                │  │ Generator       │ │
                │  └────────┬────────┘ │
                │           │          │
                │  ┌────────▼────────┐ │
                │  │ TTS Synthesis   │ │
                │  │ (OpenAI)        │ │
                │  └─────────────────┘ │
                └──────────────────────┘
```

---

## Core Components

### 1. ElleBridge (`HoloLoom/voice/elle_bridge.py`)

**Purpose**: Bidirectional bridge between VoiceAgent and Elle AR systems

**Responsibilities**:
- Receive voice commands from user
- Extract AR context (user position, gaze, selected objects)
- Route commands to appropriate Elle modules
- Receive AR events (object selections, navigation events)
- Coordinate multimodal responses (voice + visual)

**Key Methods**:
```python
class ElleBridge:
    async def process_voice_command(
        self,
        transcript: str,
        ar_context: ARContext
    ) -> ElleResponse:
        """
        Process voice command with AR context.

        Args:
            transcript: Voice input text
            ar_context: Current AR state (gaze, position, objects)

        Returns:
            ElleResponse with voice + visual components
        """
        pass

    async def handle_ar_event(
        self,
        event: AREvent
    ) -> Optional[str]:
        """
        Handle AR events (e.g., user selected object).

        Args:
            event: AR event (selection, navigation, gesture)

        Returns:
            Optional voice response
        """
        pass

    async def update_ar_display(
        self,
        visualization: ARVisualization
    ) -> None:
        """
        Update AR display with new content.

        Args:
            visualization: AR content to display
        """
        pass
```

### 2. CommandRouter (`HoloLoom/voice/command_router.py`)

**Purpose**: Parse voice intents and route to Elle actions

**Intent Categories**:

| Intent Type | Examples | Elle Action |
|-------------|----------|-------------|
| **QUERY** | "What is...", "Show me...", "Tell me about..." | Display info overlay |
| **NAVIGATE** | "Go to...", "Take me to...", "Next/previous" | AR navigation |
| **SELECT** | "This one", "That hive", "The one I'm looking at" | Object selection |
| **COMMAND** | "Start inspection", "Record observation" | Action execution |
| **EXPLAIN** | "Why is...", "Explain...", "How does..." | Educational overlay |

**Key Methods**:
```python
class CommandRouter:
    async def parse_intent(
        self,
        transcript: str,
        ar_context: ARContext
    ) -> Intent:
        """
        Parse voice input into structured intent.

        Args:
            transcript: Voice input text
            ar_context: Current AR context for disambiguation

        Returns:
            Intent with type, entities, and confidence
        """
        pass

    async def route_to_action(
        self,
        intent: Intent
    ) -> ElleAction:
        """
        Map intent to Elle action.

        Args:
            intent: Parsed intent

        Returns:
            ElleAction to execute
        """
        pass
```

### 3. ARContext (`HoloLoom/voice/ar_context.py`)

**Purpose**: Encapsulate current AR state for context-aware responses

**Structure**:
```python
@dataclass
class ARContext:
    # User state
    user_position: Vector3      # (x, y, z) in AR space
    user_orientation: Quaternion  # Head direction
    gaze_direction: Vector3     # Where user is looking
    gaze_target: Optional[str]  # Object ID user is looking at

    # Environment state
    visible_objects: List[ARObject]  # Objects in view
    selected_object: Optional[ARObject]  # Currently selected
    active_scene: str           # Current AR scene/context

    # Interaction state
    recent_actions: List[str]   # Last 5 actions
    conversation_context: str   # Current conversation topic

    # Spatial references
    nearby_objects: Dict[str, ARObject]  # "this", "that", etc.
```

### 4. SpatialAudioHandler (`HoloLoom/voice/spatial_audio.py`)

**Purpose**: Position audio in 3D space relative to user

**Features**:
- 3D audio positioning (HRTF)
- Distance-based volume attenuation
- Directional audio cues
- Ambient audio mixing

**Key Methods**:
```python
class SpatialAudioHandler:
    async def position_audio(
        self,
        audio: bytes,
        position: Vector3,
        user_context: ARContext
    ) -> bytes:
        """
        Apply spatial audio processing.

        Args:
            audio: TTS audio bytes
            position: 3D position of audio source
            user_context: User position and orientation

        Returns:
            Spatially processed audio (stereo)
        """
        pass

    def calculate_attenuation(
        self,
        distance: float
    ) -> float:
        """
        Calculate volume based on distance.

        Formula: volume = 1 / (1 + distance^2)
        """
        return 1.0 / (1.0 + distance ** 2)
```

---

## API Contracts

### Voice Command Flow

**1. User speaks**: "Show me the health status of this hive"

**2. ElleBridge receives**:
```python
{
    "transcript": "Show me the health status of this hive",
    "ar_context": {
        "user_position": [0, 1.7, 0],
        "gaze_target": "hive_003",
        "visible_objects": [
            {"id": "hive_003", "type": "beehive", "position": [2, 0, 5]},
            {"id": "hive_004", "type": "beehive", "position": [4, 0, 5]}
        ]
    }
}
```

**3. CommandRouter parses**:
```python
{
    "intent_type": "QUERY",
    "action": "show_health_status",
    "target": "hive_003",  # Resolved from "this hive" + gaze
    "confidence": 0.95
}
```

**4. HoloLoom processes**:
```python
# Query knowledge graph for hive_003 health data
query = Query(
    text="What is the health status of hive_003?",
    modality=ModalityType.VOICE,
    metadata={
        "ar_context": ar_context,
        "intent": intent
    }
}
spacetime = await orchestrator.weave(query)
```

**5. Response generated**:
```python
{
    "voice_response": "Hive 003 is in good health. The population is strong at 45,000 bees, and the queen is actively laying eggs. I'm highlighting the brood pattern on your display.",
    "ar_visualization": {
        "type": "overlay",
        "target": "hive_003",
        "content": {
            "health_score": 0.87,
            "population": 45000,
            "brood_pattern": "strong",
            "highlights": ["brood_area", "honey_stores"]
        }
    },
    "spatial_audio_position": [2, 0, 5]  # At hive location
}
```

**6. Elle displays**:
- Voice plays from hive position (spatial audio)
- AR overlay appears on hive with health data
- Brood area and honey stores highlighted in AR

---

## Spatial Reference Resolution

### Challenge
Voice commands use spatial references that are ambiguous without AR context:
- "this one" → Which one?
- "that hive" → Which hive?
- "the one I'm looking at" → Resolve from gaze
- "the hive on my left" → Spatial calculation

### Solution: Multi-stage Resolution

**1. Gaze-based resolution** (highest priority):
```python
if "this" in transcript or "that" in transcript:
    if ar_context.gaze_target:
        return ar_context.gaze_target
```

**2. Proximity-based resolution**:
```python
if "nearby" in transcript or "closest" in transcript:
    return find_closest_object(ar_context.user_position, object_type)
```

**3. Directional resolution**:
```python
if "left" in transcript or "right" in transcript:
    direction = extract_direction(transcript)
    return find_object_in_direction(
        ar_context.user_position,
        ar_context.user_orientation,
        direction
    )
```

**4. Fallback to disambiguation**:
```python
if multiple_candidates:
    return voice_prompt("Which one do you mean? I see three hives nearby.")
```

---

## Voice Command Vocabulary

### Core Commands

**Information Queries**:
- "What is [object]?"
- "Show me [data]"
- "Tell me about [topic]"
- "Explain [concept]"
- "How do I [task]?"

**Navigation**:
- "Go to [location]"
- "Take me to [object]"
- "Next [object]" / "Previous [object]"
- "Show me the way to [location]"

**Selection**:
- "This one" / "That one"
- "The [object] I'm looking at"
- "The [object] on my [left/right]"
- "Select [object]"

**Actions**:
- "Start [task]"
- "Record [observation]"
- "Mark this as [status]"
- "Create a note"

**Explanations**:
- "Why is [condition]?"
- "What does [term] mean?"
- "Teach me about [topic]"

### Beekeeping-Specific Commands

**Hive Inspection**:
- "Show me hive [number]"
- "What's the health of this hive?"
- "When was this hive last inspected?"
- "Show me the brood pattern"
- "Are there any issues with this hive?"

**Queen Management**:
- "Where is the queen?"
- "Is the queen laying well?"
- "Show me the queen cells"
- "When was the queen introduced?"

**Population & Activity**:
- "How many bees are in this hive?"
- "Show me the flight activity"
- "What's the foraging pattern?"

**Health & Treatment**:
- "Check for varroa mites"
- "What's the disease risk?"
- "When was this hive last treated?"
- "Show me the hive temperature"

**Production**:
- "How much honey is stored?"
- "Is this hive ready for harvest?"
- "Show me the nectar flow"

---

## Multimodal Response Patterns

### Pattern 1: Voice + Visual Overlay

**Use Case**: Answering factual questions about visible objects

**Example**:
- Voice: "This hive has 45,000 bees with a strong brood pattern."
- Visual: AR overlay with population count, brood pattern visualization

### Pattern 2: Voice + Navigation Guidance

**Use Case**: Directing user to locations

**Example**:
- Voice: "The next hive for inspection is 10 meters to your right."
- Visual: AR arrow pointing to hive, distance indicator

### Pattern 3: Voice + Object Highlighting

**Use Case**: Identifying specific features

**Example**:
- Voice: "I'm highlighting the brood area in blue and the honey stores in gold."
- Visual: AR highlights on specific hive regions

### Pattern 4: Voice + Step-by-Step Guidance

**Use Case**: Teaching procedures

**Example**:
- Voice: "First, approach the hive slowly. I'll show you where to place your smoker."
- Visual: AR step indicator, animated hand placement guide

### Pattern 5: Voice + Diagnostic Visualization

**Use Case**: Explaining complex data

**Example**:
- Voice: "The temperature gradient shows the brood cluster here, with cooler areas indicating honey storage."
- Visual: AR heat map overlay on hive

---

## AR Event Handling

### Event Types

**1. Object Selection Events**:
```python
@dataclass
class SelectionEvent:
    object_id: str
    object_type: str
    selection_method: str  # "gaze", "tap", "gesture"
    timestamp: float
```

**Response**:
- Voice: "You've selected hive 003. Would you like to see the inspection history?"
- Visual: Object highlight, context menu

**2. Navigation Events**:
```python
@dataclass
class NavigationEvent:
    destination: str
    distance_remaining: float
    estimated_time: float
```

**Response**:
- Voice: "You're 5 meters from hive 007."
- Visual: Progress indicator, arrival estimate

**3. Gesture Events**:
```python
@dataclass
class GestureEvent:
    gesture_type: str  # "point", "circle", "swipe"
    target: Optional[str]
    confidence: float
```

**Response**:
- Voice: "I see you're pointing at the honey frames. Let me pull up that data."
- Visual: Highlighted frame, data overlay

---

## Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| **Voice → Response Latency** | <2s | Includes TTS |
| **Spatial Reference Resolution** | <100ms | Gaze/proximity calc |
| **AR Overlay Update** | <50ms | Frame sync |
| **Spatial Audio Processing** | <10ms | HRTF application |
| **Intent Classification** | <200ms | NLU processing |
| **Multi-modal Sync** | <16ms | 60 FPS AR sync |

---

## Implementation Plan

### Phase 1: Core Bridge (Days 3-4)

**Files to create**:
1. `HoloLoom/voice/elle_bridge.py` (400 lines)
2. `HoloLoom/voice/command_router.py` (300 lines)
3. `HoloLoom/voice/ar_context.py` (200 lines)
4. `HoloLoom/voice/spatial_audio.py` (250 lines)

**Tests**:
1. `HoloLoom/voice/tests/test_elle_bridge.py` (300 lines)
2. `HoloLoom/voice/tests/test_command_router.py` (250 lines)

### Phase 2: Spatial Features (Day 5)

**Files to create**:
1. `HoloLoom/voice/spatial_resolver.py` (200 lines)
2. `HoloLoom/voice/multimodal_response.py` (150 lines)

**Tests**:
1. `HoloLoom/voice/tests/test_spatial_resolver.py` (200 lines)

### Phase 3: Integration & Demos (Day 5)

**Demos**:
1. `demos/demo_elle_voice_query.py`
2. `demos/demo_elle_navigation.py`
3. `demos/demo_elle_multimodal.py`
4. `demos/demo_elle_spatial_audio.py`

---

## Security & Privacy

### Voice Data
- ✅ No audio stored permanently
- ✅ Transcripts ephemeral (cleared after session)
- ✅ User consent for voice processing

### AR Context
- ✅ Position data stays local
- ✅ No gaze tracking logs
- ✅ Object selections not persisted

### Network
- ✅ End-to-end encryption (TLS)
- ✅ No external API calls (except OpenAI TTS)
- ✅ Local HoloLoom processing

---

## Testing Strategy

### Unit Tests
- Intent classification accuracy (>95%)
- Spatial reference resolution (100+ test cases)
- Audio positioning calculations
- AR context extraction

### Integration Tests
- Voice → Elle → Response flow
- Multimodal response coordination
- Event handling
- Error recovery

### User Testing
- 10 common beekeeping scenarios
- Natural language variation
- Ambiguous reference handling
- Multi-turn dialogue

---

## Future Enhancements

### Phase 4+ Ideas
1. **Gesture + Voice Combination**
   - Point + "What's this?" → Explain object
   - Circle + "Analyze this area" → Area analysis

2. **Proactive Assistance**
   - "I notice the hive temperature is high, would you like me to explain?"
   - "This hive is due for inspection today."

3. **Multi-User Collaboration**
   - Shared AR view with voice chat
   - "Show everyone the queen cell"

4. **Voice Shortcuts**
   - Custom voice commands
   - "Run my inspection checklist"

5. **Emotional Intelligence**
   - Detect user frustration
   - Adjust verbosity based on expertise

---

## Summary

**Architecture Complete**: ✅

**Key Deliverables**:
- ✅ ElleBridge module design
- ✅ CommandRouter specification
- ✅ ARContext data structure
- ✅ SpatialAudioHandler design
- ✅ API contracts defined
- ✅ Voice command vocabulary (50+ commands)
- ✅ Multimodal response patterns (5 types)
- ✅ Performance requirements
- ✅ Implementation plan
- ✅ Testing strategy

**Ready for Implementation**: YES 🚀

**Estimated LOC**: ~2,000 lines (implementation + tests)

---

**Version**: 1.0.0
**Status**: 🏗️ Design Complete
**Next**: Begin implementation (ElleBridge module)

*Complete architecture for integrating VoiceAgent with Elle AR assistant.*

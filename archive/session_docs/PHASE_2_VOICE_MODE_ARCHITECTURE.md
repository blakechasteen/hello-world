# Phase 2: Advanced Voice Mode Architecture

**Status**: Design Complete
**Date**: November 15, 2025
**Dependencies**: Phase 1 (Live Audio Streaming) ✅ Complete

---

## 🎯 Vision

Enable **bidirectional voice interaction** between users and HoloLoom agents (Elle, Promptly, etc.) with natural conversation flow, context awareness, and neural decision-making integration.

---

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Voice Interaction Loop                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Speech                                                 │
│      ↓                                                      │
│  [Microphone] → [VAD] → [Whisper STT] → [Intent Parser]   │
│                                              ↓              │
│                                    [WeavingOrchestrator]    │
│                                              ↓              │
│                                    [Policy Decision]        │
│                                              ↓              │
│                                    [Response Generator]     │
│                                              ↓              │
│  [Speaker] ← [Audio Queue] ← [OpenAI TTS] ← [Response]    │
│      ↑                                                      │
│  Agent Speech                                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Voice Input** → VAD detects speech → Whisper transcribes → Intent extracted
2. **Context Assembly** → Recent conversation + memory + current query
3. **Neural Processing** → HoloLoom weaving cycle (policy, memory, tools)
4. **Response Generation** → LLM generates response text
5. **Voice Output** → TTS synthesizes speech → Audio playback

---

## 📋 Component Specifications

### 1. Voice Activity Detection (VAD)

**Purpose**: Detect when user starts/stops speaking to manage turn-taking

**Options**:

| Option | Latency | Accuracy | Complexity |
|--------|---------|----------|------------|
| **WebRTC VAD** | <10ms | Good | Low (recommended) |
| **Silero VAD** | ~20ms | Better | Medium |
| **Energy-based** | <5ms | Basic | Very Low |
| **Button-based** | 0ms | Perfect | None (fallback) |

**Recommendation**: Start with **WebRTC VAD** + button fallback

**Implementation**:
```python
import webrtcvad

class VoiceActivityDetector:
    """Detect voice activity in audio stream"""

    def __init__(self, aggressiveness: int = 3):
        """
        Args:
            aggressiveness: 0-3 (0=liberal, 3=aggressive)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

    def is_speech(self, audio_frame: bytes) -> bool:
        """Check if frame contains speech"""
        return self.vad.is_speech(audio_frame, self.sample_rate)

    def get_speech_segments(self, audio_stream: np.ndarray) -> List[Tuple[int, int]]:
        """Extract speech segments from audio stream"""
        speech_segments = []
        in_speech = False
        start_idx = 0

        for i in range(0, len(audio_stream), self.frame_size):
            frame = audio_stream[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break

            frame_bytes = (frame * 32768).astype(np.int16).tobytes()
            is_speech = self.is_speech(frame_bytes)

            if is_speech and not in_speech:
                # Speech started
                start_idx = i
                in_speech = True
            elif not is_speech and in_speech:
                # Speech ended
                speech_segments.append((start_idx, i))
                in_speech = False

        # Handle trailing speech
        if in_speech:
            speech_segments.append((start_idx, len(audio_stream)))

        return speech_segments
```

---

### 2. Turn-Taking Manager

**Purpose**: Coordinate speaker turns to prevent overlap and interrupts

**Strategies**:

1. **Button-based** (simplest)
   - User presses button → agent listens
   - Agent responds → user waits
   - No complexity, no errors

2. **VAD-based** (recommended)
   - VAD detects user speech → agent listens
   - Agent responds → user can interrupt
   - Natural but requires tuning

3. **Hybrid** (production)
   - Button or VAD trigger
   - Interrupt detection with confidence threshold
   - Graceful handoff

**Implementation**:
```python
from enum import Enum

class TurnState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"

class TurnTakingManager:
    """Manage conversation turn-taking"""

    def __init__(self, mode: str = "vad"):
        """
        Args:
            mode: 'button', 'vad', or 'hybrid'
        """
        self.mode = mode
        self.state = TurnState.IDLE
        self.vad = VoiceActivityDetector() if mode in ['vad', 'hybrid'] else None
        self.interrupt_threshold = 0.7  # Confidence for interrupts

        # State tracking
        self.user_speaking = False
        self.agent_speaking = False
        self.silence_start = None
        self.silence_duration_threshold = 1.5  # seconds

    def update(self, audio_frame: np.ndarray, timestamp: float) -> Optional[str]:
        """
        Update turn state based on audio input

        Returns:
            Action to take: 'start_listening', 'stop_listening', 'interrupt_agent', None
        """
        if self.mode == 'button':
            # Button mode - manual control only
            return None

        # Detect speech in current frame
        frame_bytes = (audio_frame * 32768).astype(np.int16).tobytes()
        is_speech = self.vad.is_speech(frame_bytes)

        # State machine
        if self.state == TurnState.IDLE:
            if is_speech:
                # User started speaking
                self.user_speaking = True
                self.state = TurnState.LISTENING
                return 'start_listening'

        elif self.state == TurnState.LISTENING:
            if not is_speech:
                # Potential end of speech
                if self.silence_start is None:
                    self.silence_start = timestamp
                elif (timestamp - self.silence_start) > self.silence_duration_threshold:
                    # Confirmed end of speech
                    self.user_speaking = False
                    self.state = TurnState.PROCESSING
                    self.silence_start = None
                    return 'stop_listening'
            else:
                # Still speaking - reset silence timer
                self.silence_start = None

        elif self.state == TurnState.SPEAKING:
            if is_speech and self.mode == 'hybrid':
                # User interrupted agent
                if self._is_confident_interrupt(audio_frame):
                    self.agent_speaking = False
                    self.user_speaking = True
                    self.state = TurnState.LISTENING
                    return 'interrupt_agent'

        return None

    def _is_confident_interrupt(self, audio_frame: np.ndarray) -> bool:
        """Check if user interrupt is confident enough"""
        # Simplified: check energy level
        energy = np.sqrt(np.mean(audio_frame ** 2))
        return energy > self.interrupt_threshold

    def set_state(self, state: TurnState):
        """Manually set state (for button mode)"""
        self.state = state
        if state == TurnState.SPEAKING:
            self.agent_speaking = True
        elif state == TurnState.LISTENING:
            self.user_speaking = True
        else:
            self.user_speaking = False
            self.agent_speaking = False
```

---

### 3. Text-to-Speech (TTS) Integration

**Purpose**: Convert agent responses to natural speech

**Provider Comparison**:

| Provider | Latency | Quality | Cost | Voices |
|----------|---------|---------|------|--------|
| **OpenAI TTS** | 500ms | Excellent | $15/1M chars | 6 voices (Nova, Alloy, etc.) |
| **Eleven Labs** | 300ms | Best | $30/1M chars | Custom cloning |
| **Google Cloud TTS** | 400ms | Excellent | $16/1M chars | 200+ voices |
| **Azure TTS** | 300ms | Excellent | $16/1M chars | Neural voices |
| **Piper (local)** | 200ms | Good | Free | 50+ voices |

**Recommendation**: **OpenAI TTS** for production (best balance of quality/cost/latency)

**Implementation**:
```python
import openai
import io
from pydub import AudioSegment
from pydub.playback import play
import asyncio

class TTSProvider:
    """Text-to-Speech provider interface"""

    async def synthesize(self, text: str, voice: str = "alloy") -> bytes:
        """Synthesize speech from text"""
        raise NotImplementedError

class OpenAITTS(TTSProvider):
    """OpenAI TTS provider"""

    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.client = openai.AsyncOpenAI()

    async def synthesize(self, text: str, voice: str = "nova") -> bytes:
        """
        Synthesize speech using OpenAI TTS

        Args:
            text: Text to synthesize
            voice: Voice ID (alloy, echo, fable, onyx, nova, shimmer)

        Returns:
            Audio bytes (MP3 format)
        """
        response = await self.client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice=voice,
            input=text,
            speed=1.0
        )

        return response.content

class TTSManager:
    """Manage TTS synthesis and playback"""

    def __init__(self, provider: TTSProvider, voice: str = "nova"):
        self.provider = provider
        self.voice = voice
        self.playback_queue = asyncio.Queue()
        self.is_playing = False

    async def speak(self, text: str, priority: bool = False):
        """
        Synthesize and queue speech

        Args:
            text: Text to speak
            priority: If True, skip queue and play immediately
        """
        # Synthesize
        audio_bytes = await self.provider.synthesize(text, self.voice)

        if priority:
            # Clear queue and play immediately
            while not self.playback_queue.empty():
                try:
                    self.playback_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Add to queue
        await self.playback_queue.put(audio_bytes)

        # Start playback if not already playing
        if not self.is_playing:
            asyncio.create_task(self._playback_loop())

    async def _playback_loop(self):
        """Playback loop for queued audio"""
        self.is_playing = True

        try:
            while not self.playback_queue.empty():
                audio_bytes = await self.playback_queue.get()

                # Convert bytes to audio and play
                audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

                # Play audio (blocking)
                await asyncio.get_event_loop().run_in_executor(
                    None, play, audio
                )
        finally:
            self.is_playing = False

    def stop(self):
        """Stop current playback"""
        # Clear queue
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.is_playing = False
```

---

### 4. Conversation Memory

**Purpose**: Maintain conversation history for context-aware responses

**Storage**:
- **Short-term**: Last N turns in memory (sliding window)
- **Long-term**: Stored in HoloLoom memory graph (Yarn Graph)

**Implementation**:
```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    speaker: str  # 'user' or 'agent'
    text: str
    timestamp: float
    intent: Optional[str] = None
    confidence: Optional[float] = None
    metadata: dict = None

class ConversationMemory:
    """Manage conversation history"""

    def __init__(self, max_turns: int = 20, kg_store: Optional[Any] = None):
        """
        Args:
            max_turns: Maximum turns to keep in short-term memory
            kg_store: HoloLoom knowledge graph for long-term storage
        """
        self.max_turns = max_turns
        self.kg_store = kg_store
        self.turns: List[ConversationTurn] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_turn(self, speaker: str, text: str, **metadata):
        """Add conversation turn"""
        turn = ConversationTurn(
            speaker=speaker,
            text=text,
            timestamp=datetime.now().timestamp(),
            metadata=metadata
        )

        self.turns.append(turn)

        # Trim if exceeds max
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

        # Store in knowledge graph
        if self.kg_store:
            self._store_in_kg(turn)

    def get_context(self, n_turns: int = 5) -> str:
        """Get recent conversation context as string"""
        recent_turns = self.turns[-n_turns:]

        context_lines = []
        for turn in recent_turns:
            prefix = "User:" if turn.speaker == "user" else "Agent:"
            context_lines.append(f"{prefix} {turn.text}")

        return "\n".join(context_lines)

    def get_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Get recent turns"""
        if n is None:
            return self.turns
        return self.turns[-n:]

    def _store_in_kg(self, turn: ConversationTurn):
        """Store turn in knowledge graph"""
        if not self.kg_store:
            return

        # Create nodes and edges
        from HoloLoom.memory.graph import KGEdge

        edges = [
            KGEdge(
                self.session_id,
                turn.speaker,
                "SPOKE",
                1.0,
                metadata={'text': turn.text, 'timestamp': turn.timestamp}
            )
        ]

        self.kg_store.add_edges(edges)
```

---

### 5. VoiceAgent Integration

**Purpose**: Unified interface for voice-enabled agents (Elle, Promptly, etc.)

**Implementation**:
```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, ModalityType

class VoiceAgent:
    """Voice-enabled agent with bidirectional interaction"""

    def __init__(
        self,
        orchestrator: WeavingOrchestrator,
        tts_manager: TTSManager,
        agent_name: str = "Elle"
    ):
        """
        Args:
            orchestrator: HoloLoom weaving orchestrator
            tts_manager: TTS synthesis manager
            agent_name: Agent identity (for personality)
        """
        self.orchestrator = orchestrator
        self.tts = tts_manager
        self.agent_name = agent_name

        # Conversation state
        self.conversation_memory = ConversationMemory()
        self.turn_manager = TurnTakingManager(mode='hybrid')

    async def process_voice_input(self, transcript: str) -> str:
        """
        Process voice input and generate response

        Args:
            transcript: Transcribed user speech

        Returns:
            Agent response text
        """
        # Add to conversation memory
        self.conversation_memory.add_turn('user', transcript)

        # Get conversation context
        context = self.conversation_memory.get_context(n_turns=5)

        # Create query with context
        query = Query(
            text=transcript,
            modality=ModalityType.AUDIO,
            metadata={
                'agent': self.agent_name,
                'conversation_context': context,
                'session_id': self.conversation_memory.session_id
            }
        )

        # Weave through HoloLoom
        spacetime = await self.orchestrator.weave(query)

        # Extract response
        response_text = spacetime.response.get('text', 'I did not understand that.')

        # Add to conversation memory
        self.conversation_memory.add_turn(
            'agent',
            response_text,
            confidence=spacetime.confidence,
            tool_used=spacetime.metadata.get('tool_used')
        )

        return response_text

    async def speak(self, text: str):
        """Speak response using TTS"""
        await self.tts.speak(text)

    async def listen_and_respond(self, audio_stream: AsyncGenerator):
        """
        Main voice interaction loop

        Args:
            audio_stream: Async generator yielding audio chunks
        """
        transcriber = LiveTranscriber()
        vad = VoiceActivityDetector()

        speech_buffer = []

        async for chunk in audio_stream:
            # Check turn state
            action = self.turn_manager.update(chunk.audio_data, chunk.timestamp)

            if action == 'start_listening':
                print(f"🎤 {self.agent_name} is listening...")
                speech_buffer = []

            elif action == 'stop_listening':
                # Transcribe collected speech
                if speech_buffer:
                    full_audio = np.concatenate(speech_buffer)
                    transcript_result = await transcriber.transcribe_chunk(
                        StreamChunk(full_audio, chunk.timestamp, 0)
                    )
                    transcript = transcript_result['text']

                    print(f"User: {transcript}")

                    # Process and respond
                    response = await self.process_voice_input(transcript)
                    print(f"{self.agent_name}: {response}")

                    # Speak response
                    self.turn_manager.set_state(TurnState.SPEAKING)
                    await self.speak(response)
                    self.turn_manager.set_state(TurnState.IDLE)

                speech_buffer = []

            elif action == 'interrupt_agent':
                print(f"⚠️  User interrupted {self.agent_name}")
                self.tts.stop()
                speech_buffer = []

            # Collect speech if listening
            if self.turn_manager.state == TurnState.LISTENING:
                speech_buffer.append(chunk.audio_data)
```

---

## 🚀 Deployment Architecture

### Production Setup

```
┌─────────────────────────────────────────────────────────┐
│                   Production Deployment                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [Client Device]                                         │
│       ↓                                                  │
│  [WebSocket/gRPC]                                       │
│       ↓                                                  │
│  ┌─────────────────────────────────────────┐           │
│  │  Voice Processing Service (FastAPI)     │           │
│  │  - VAD                                   │           │
│  │  - STT (Whisper)                        │           │
│  │  - TTS (OpenAI)                         │           │
│  └─────────────────────────────────────────┘           │
│       ↓                                                  │
│  ┌─────────────────────────────────────────┐           │
│  │  HoloLoom Orchestrator Service          │           │
│  │  - WeavingOrchestrator                  │           │
│  │  - Policy Engine                        │           │
│  │  - Memory (Neo4j + Qdrant)             │           │
│  └─────────────────────────────────────────┘           │
│       ↓                                                  │
│  ┌─────────────────────────────────────────┐           │
│  │  Conversation Memory Service            │           │
│  │  - Session management                   │           │
│  │  - Context tracking                     │           │
│  │  - Long-term storage                    │           │
│  └─────────────────────────────────────────┘           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Targets

### Latency Budget (Target: <2s total)

| Component | Target | Stretch Goal |
|-----------|--------|--------------|
| **VAD detection** | <50ms | <20ms |
| **STT (Whisper base)** | <500ms | <200ms (GPU) |
| **Intent parsing** | <50ms | <20ms |
| **HoloLoom weaving** | <300ms | <150ms |
| **Response generation** | <800ms | <400ms |
| **TTS synthesis** | <500ms | <300ms |
| **Audio playback** | immediate | immediate |
| **Total** | <2.2s | <1.1s |

### Optimizations

1. **GPU acceleration** for Whisper (5x faster)
2. **Streaming TTS** (start playback before full synthesis)
3. **Caching** for common responses
4. **Parallel processing** (STT + context loading)

---

## 🧪 Testing Strategy

### Unit Tests
- VAD accuracy (precision/recall)
- Turn-taking state transitions
- TTS synthesis quality
- Conversation memory storage/retrieval

### Integration Tests
- End-to-end voice loop
- HoloLoom orchestrator integration
- Multi-turn conversations
- Interrupt handling

### Performance Tests
- Latency benchmarks (p50, p95, p99)
- Throughput (concurrent conversations)
- Memory usage (long conversations)

### User Acceptance Tests
- Natural conversation flow
- Response relevance
- Voice quality
- Interrupt responsiveness

---

## 📝 Next Steps

### Week 1
1. ✅ Implement VAD with WebRTC
2. ✅ Integrate OpenAI TTS
3. ✅ Create ConversationMemory class
4. ✅ Test turn-taking logic

### Week 2
5. ✅ Implement VoiceAgent interface
6. ✅ Connect to WeavingOrchestrator
7. ✅ Add interrupt handling
8. ✅ Write integration tests

### Week 3-4
9. ✅ Production deployment setup
10. ✅ Performance optimization
11. ✅ User acceptance testing
12. ✅ Documentation and examples

---

## 🔗 Integration Points

### With HoloLoom
- `WeavingOrchestrator` - Neural decision-making
- `memory.graph.KG` - Long-term conversation storage
- `embedding.spectral` - Semantic understanding
- `policy.unified` - Multi-criteria decision making

### With Elle
- Voice input for scene understanding
- Audio-triggered actions
- Conversational guidance
- Context-aware responses

---

## 📚 Dependencies

**Required**:
```bash
pip install openai pydub webrtcvad numpy structlog prometheus-client
```

**Optional**:
```bash
pip install elevenlabs google-cloud-texttospeech azure-cognitiveservices-speech
```

---

**Status**: Architecture complete, ready for implementation
**Owner**: Blake Chasteen
**Updated**: November 15, 2025

---

*This architecture enables natural voice interaction with HoloLoom agents through bidirectional audio, turn-taking management, and deep integration with the neural decision-making system.*

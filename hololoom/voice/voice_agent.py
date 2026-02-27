"""
VoiceAgent - Bidirectional Voice Interaction for HoloLoom
==========================================================

Production-ready voice agent with:
- OpenAI TTS integration
- Conversation memory with HoloLoom KG
- Turn-taking management (VAD + button)
- Integration with WeavingOrchestrator
- Interrupt handling
- Comprehensive error handling and logging

Date: November 15, 2025
"""

import asyncio
import io
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, List, Dict, Any
from datetime import datetime
from enum import Enum

# Structured logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
    logger = structlog.get_logger()
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)

# OpenAI TTS
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not available - TTS disabled")

# Audio playback
try:
    from pydub import AudioSegment
    from pydub.playback import play
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False
    logger.warning("pydub not available - audio playback disabled")

# WebRTC VAD
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    logger.warning("webrtcvad not available - VAD disabled")

# HoloLoom imports
try:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.protocols.types import Query, ModalityType
    from HoloLoom.memory.graph import KG, KGEdge
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    logger.warning("HoloLoom imports not available - orchestrator integration disabled")


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    speaker: str  # 'user' or 'agent'
    text: str
    timestamp: float
    intent: Optional[str] = None
    confidence: Optional[float] = None
    tool_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'speaker': self.speaker,
            'text': self.text,
            'timestamp': self.timestamp,
            'intent': self.intent,
            'confidence': self.confidence,
            'tool_used': self.tool_used,
            'metadata': self.metadata
        }


class TurnState(Enum):
    """Conversation turn states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


# ============================================================================
# Voice Activity Detection
# ============================================================================

class VoiceActivityDetector:
    """
    Detect voice activity in audio stream using WebRTC VAD

    WebRTC VAD is fast (<10ms) and accurate for speech detection.
    """

    def __init__(self, aggressiveness: int = 3):
        """
        Args:
            aggressiveness: 0-3 (0=liberal, 3=aggressive)
        """
        if not VAD_AVAILABLE:
            raise ImportError("webrtcvad not available - install: pip install webrtcvad")

        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

        if STRUCTLOG_AVAILABLE:
            logger.info("vad_initialized", aggressiveness=aggressiveness)

    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Check if frame contains speech

        Args:
            audio_frame: Audio bytes (16-bit PCM, 16kHz)

        Returns:
            True if speech detected
        """
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            if STRUCTLOG_AVAILABLE:
                logger.error("vad_error", error=str(e))
            return False

    def get_speech_segments(self, audio_stream: np.ndarray) -> List[tuple]:
        """
        Extract speech segments from audio stream

        Args:
            audio_stream: Audio array (float32, normalized)

        Returns:
            List of (start_idx, end_idx) tuples
        """
        speech_segments = []
        in_speech = False
        start_idx = 0

        for i in range(0, len(audio_stream), self.frame_size):
            frame = audio_stream[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                break

            # Convert to 16-bit PCM
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


# ============================================================================
# Turn-Taking Manager
# ============================================================================

class TurnTakingManager:
    """
    Manage conversation turn-taking with multiple modes:
    - Button: Manual control (simplest, most reliable)
    - VAD: Automatic voice detection
    - Hybrid: Button or VAD trigger (recommended)
    """

    def __init__(self, mode: str = "hybrid", silence_threshold: float = 1.5):
        """
        Args:
            mode: 'button', 'vad', or 'hybrid'
            silence_threshold: Seconds of silence before ending turn
        """
        self.mode = mode
        self.state = TurnState.IDLE
        self.vad = VoiceActivityDetector() if mode in ['vad', 'hybrid'] and VAD_AVAILABLE else None
        self.interrupt_threshold = 0.7  # Confidence for interrupts

        # State tracking
        self.user_speaking = False
        self.agent_speaking = False
        self.silence_start = None
        self.silence_duration_threshold = silence_threshold

        if STRUCTLOG_AVAILABLE:
            logger.info("turn_manager_initialized",
                mode=mode,
                silence_threshold=silence_threshold,
                vad_available=self.vad is not None)

    def update(self, audio_frame: np.ndarray, timestamp: float) -> Optional[str]:
        """
        Update turn state based on audio input

        Args:
            audio_frame: Audio chunk (float32, normalized)
            timestamp: Current timestamp

        Returns:
            Action: 'start_listening', 'stop_listening', 'interrupt_agent', None
        """
        if self.mode == 'button' or self.vad is None:
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

                if STRUCTLOG_AVAILABLE:
                    logger.info("turn_started", timestamp=timestamp)

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

                    if STRUCTLOG_AVAILABLE:
                        logger.info("turn_ended", timestamp=timestamp)

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

                    if STRUCTLOG_AVAILABLE:
                        logger.info("agent_interrupted", timestamp=timestamp)

                    return 'interrupt_agent'

        return None

    def _is_confident_interrupt(self, audio_frame: np.ndarray) -> bool:
        """Check if user interrupt is confident enough"""
        # Check energy level
        energy = np.sqrt(np.mean(audio_frame ** 2))
        return energy > self.interrupt_threshold

    def set_state(self, state: TurnState):
        """Manually set state (for button mode)"""
        self.state = state

        if state == TurnState.SPEAKING:
            self.agent_speaking = True
            self.user_speaking = False
        elif state == TurnState.LISTENING:
            self.user_speaking = True
            self.agent_speaking = False
        else:
            self.user_speaking = False
            self.agent_speaking = False

        if STRUCTLOG_AVAILABLE:
            logger.info("state_changed", state=state.value)


# ============================================================================
# Text-to-Speech Integration
# ============================================================================

class TTSProvider:
    """Abstract TTS provider interface"""

    async def synthesize(self, text: str, voice: str = "alloy") -> bytes:
        """Synthesize speech from text"""
        raise NotImplementedError


class OpenAITTS(TTSProvider):
    """
    OpenAI TTS provider

    Recommended for production:
    - 500ms latency
    - Excellent quality
    - $15/1M characters
    - 6 voices available
    """

    def __init__(self, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available - install: pip install openai")

        self.client = openai.AsyncOpenAI(api_key=api_key)

        if STRUCTLOG_AVAILABLE:
            logger.info("openai_tts_initialized")

    async def synthesize(self, text: str, voice: str = "nova") -> bytes:
        """
        Synthesize speech using OpenAI TTS

        Args:
            text: Text to synthesize
            voice: Voice ID (alloy, echo, fable, onyx, nova, shimmer)

        Returns:
            Audio bytes (MP3 format)
        """
        start_time = time.time()

        try:
            response = await self.client.audio.speech.create(
                model="tts-1",  # or "tts-1-hd" for higher quality
                voice=voice,
                input=text,
                speed=1.0
            )

            duration = time.time() - start_time

            if STRUCTLOG_AVAILABLE:
                logger.info("tts_synthesized",
                    text_length=len(text),
                    duration_seconds=duration,
                    voice=voice)

            return response.content

        except Exception as e:
            if STRUCTLOG_AVAILABLE:
                logger.error("tts_synthesis_failed", error=str(e), exc_info=True)
            raise


class TTSManager:
    """
    Manage TTS synthesis and playback with queue support
    """

    def __init__(self, provider: TTSProvider, voice: str = "nova"):
        self.provider = provider
        self.voice = voice
        self.playback_queue = asyncio.Queue()
        self.is_playing = False
        self.playback_task = None

        if STRUCTLOG_AVAILABLE:
            logger.info("tts_manager_initialized", voice=voice)

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
            self.playback_task = asyncio.create_task(self._playback_loop())

    async def _playback_loop(self):
        """Playback loop for queued audio"""
        self.is_playing = True

        try:
            while not self.playback_queue.empty():
                audio_bytes = await self.playback_queue.get()

                if not AUDIO_PLAYBACK_AVAILABLE:
                    if STRUCTLOG_AVAILABLE:
                        logger.warning("audio_playback_unavailable")
                    continue

                # Convert bytes to audio and play
                audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

                # Play audio (blocking)
                await asyncio.get_event_loop().run_in_executor(
                    None, play, audio
                )

                if STRUCTLOG_AVAILABLE:
                    logger.info("audio_played", duration_ms=len(audio))

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

        # Cancel playback task
        if self.playback_task and not self.playback_task.done():
            self.playback_task.cancel()

        self.is_playing = False

        if STRUCTLOG_AVAILABLE:
            logger.info("playback_stopped")


# ============================================================================
# Conversation Memory
# ============================================================================

class ConversationMemory:
    """
    Manage conversation history with short-term and long-term storage

    Short-term: Last N turns in memory (sliding window)
    Long-term: Stored in HoloLoom Yarn Graph
    """

    def __init__(self, max_turns: int = 20, kg_store: Optional[Any] = None):
        """
        Args:
            max_turns: Maximum turns to keep in short-term memory
            kg_store: HoloLoom KG for long-term storage
        """
        self.max_turns = max_turns
        self.kg_store = kg_store
        self.turns: List[ConversationTurn] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if STRUCTLOG_AVAILABLE:
            logger.info("conversation_memory_initialized",
                session_id=self.session_id,
                max_turns=max_turns,
                kg_available=kg_store is not None)

    def add_turn(self, speaker: str, text: str, **metadata):
        """Add conversation turn"""
        turn = ConversationTurn(
            speaker=speaker,
            text=text,
            timestamp=datetime.now().timestamp(),
            **metadata
        )

        self.turns.append(turn)

        # Trim if exceeds max
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

        # Store in knowledge graph
        if self.kg_store:
            self._store_in_kg(turn)

        if STRUCTLOG_AVAILABLE:
            logger.info("turn_added",
                speaker=speaker,
                text_length=len(text),
                total_turns=len(self.turns))

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
        if not self.kg_store or not HOLOLOOM_AVAILABLE:
            return

        try:
            # Create edges for conversation flow
            edges = [
                KGEdge(
                    self.session_id,
                    f"turn_{len(self.turns)}",
                    "HAS_TURN",
                    1.0,
                    metadata=turn.to_dict()
                ),
                KGEdge(
                    f"turn_{len(self.turns)}",
                    turn.speaker,
                    "SPOKEN_BY",
                    1.0
                )
            ]

            self.kg_store.add_edges(edges)

            if STRUCTLOG_AVAILABLE:
                logger.debug("turn_stored_in_kg", turn_number=len(self.turns))

        except Exception as e:
            if STRUCTLOG_AVAILABLE:
                logger.error("kg_storage_failed", error=str(e))

    def export_session(self) -> Dict[str, Any]:
        """Export full session for archival"""
        return {
            'session_id': self.session_id,
            'turns': [turn.to_dict() for turn in self.turns],
            'total_turns': len(self.turns),
            'start_time': self.turns[0].timestamp if self.turns else None,
            'end_time': self.turns[-1].timestamp if self.turns else None
        }


# ============================================================================
# VoiceAgent - Main Interface
# ============================================================================

class VoiceAgent:
    """
    Voice-enabled agent with bidirectional interaction

    Integrates:
    - HoloLoom WeavingOrchestrator (neural decision-making)
    - TTS synthesis (voice output)
    - Conversation memory (context tracking)
    - Turn-taking (VAD + button modes)
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        tts_provider: Optional[TTSProvider] = None,
        agent_name: str = "Elle",
        voice: str = "nova",
        turn_mode: str = "hybrid"
    ):
        """
        Args:
            orchestrator: HoloLoom WeavingOrchestrator
            tts_provider: TTS provider (defaults to OpenAI)
            agent_name: Agent identity
            voice: TTS voice ID
            turn_mode: Turn-taking mode ('button', 'vad', 'hybrid')
        """
        if not HOLOLOOM_AVAILABLE and orchestrator is not None:
            logger.warning("HoloLoom not available - orchestrator integration disabled")

        self.orchestrator = orchestrator
        self.agent_name = agent_name

        # TTS setup
        if tts_provider is None and OPENAI_AVAILABLE:
            tts_provider = OpenAITTS()

        self.tts = TTSManager(tts_provider, voice=voice) if tts_provider else None

        # Conversation state
        kg_store = getattr(orchestrator, 'kg_store', None) if orchestrator else None
        self.conversation_memory = ConversationMemory(kg_store=kg_store)
        self.turn_manager = TurnTakingManager(mode=turn_mode)

        if STRUCTLOG_AVAILABLE:
            logger.info("voice_agent_initialized",
                agent_name=agent_name,
                voice=voice,
                turn_mode=turn_mode,
                orchestrator_available=orchestrator is not None,
                tts_available=self.tts is not None)

    async def process_voice_input(self, transcript: str) -> str:
        """
        Process voice input and generate response

        Args:
            transcript: Transcribed user speech

        Returns:
            Agent response text
        """
        start_time = time.time()

        # Add to conversation memory
        self.conversation_memory.add_turn('user', transcript)

        # Get conversation context
        context = self.conversation_memory.get_context(n_turns=5)

        if STRUCTLOG_AVAILABLE:
            logger.info("processing_voice_input",
                transcript_length=len(transcript),
                context_turns=len(self.conversation_memory.get_turns(5)))

        # Process with HoloLoom if available
        if self.orchestrator and HOLOLOOM_AVAILABLE:
            response_text = await self._process_with_hololoom(transcript, context)
        else:
            # Fallback to simple echo
            response_text = f"I heard you say: {transcript}"

        # Add to conversation memory
        self.conversation_memory.add_turn(
            'agent',
            response_text
        )

        duration = time.time() - start_time

        if STRUCTLOG_AVAILABLE:
            logger.info("voice_input_processed",
                duration_seconds=duration,
                response_length=len(response_text))

        return response_text

    async def _process_with_hololoom(self, transcript: str, context: str) -> str:
        """Process input through HoloLoom orchestrator"""
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

        # Update metadata in last turn
        if self.conversation_memory.turns:
            last_turn = self.conversation_memory.turns[-1]
            last_turn.confidence = spacetime.confidence
            last_turn.tool_used = spacetime.metadata.get('tool_used')

        return response_text

    async def speak(self, text: str, priority: bool = False):
        """
        Speak response using TTS

        Args:
            text: Text to speak
            priority: Interrupt current speech if True
        """
        if not self.tts:
            if STRUCTLOG_AVAILABLE:
                logger.warning("tts_unavailable")
            print(f"{self.agent_name}: {text}")
            return

        await self.tts.speak(text, priority=priority)

    async def listen_and_respond(
        self,
        audio_stream: AsyncGenerator,
        max_duration: Optional[float] = None
    ):
        """
        Main voice interaction loop

        Args:
            audio_stream: Async generator yielding audio chunks
            max_duration: Maximum session duration in seconds
        """
        # Import here to avoid circular dependency
        from LIVE_AUDIO_STREAMING_IMPROVED import LiveTranscriber, StreamChunk

        transcriber = LiveTranscriber()
        speech_buffer = []
        session_start = time.time()

        if STRUCTLOG_AVAILABLE:
            logger.info("voice_session_started",
                agent=self.agent_name,
                max_duration=max_duration)
        else:
            print(f"\n🎤 {self.agent_name} is ready to chat!")
            print(f"⏹️  Press Ctrl+C to stop\n")

        try:
            async for chunk in audio_stream:
                # Check duration limit
                if max_duration:
                    elapsed = time.time() - session_start
                    if elapsed > max_duration:
                        if STRUCTLOG_AVAILABLE:
                            logger.info("max_duration_reached", elapsed=elapsed)
                        break

                # Check turn state
                action = self.turn_manager.update(chunk.audio_data, chunk.timestamp)

                if action == 'start_listening':
                    if not STRUCTLOG_AVAILABLE:
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

                        if not STRUCTLOG_AVAILABLE:
                            print(f"User: {transcript}")

                        # Process and respond
                        response = await self.process_voice_input(transcript)

                        if not STRUCTLOG_AVAILABLE:
                            print(f"{self.agent_name}: {response}")

                        # Speak response
                        self.turn_manager.set_state(TurnState.SPEAKING)
                        await self.speak(response)
                        self.turn_manager.set_state(TurnState.IDLE)

                    speech_buffer = []

                elif action == 'interrupt_agent':
                    if not STRUCTLOG_AVAILABLE:
                        print(f"⚠️  User interrupted {self.agent_name}")

                    if self.tts:
                        self.tts.stop()
                    speech_buffer = []

                # Collect speech if listening
                if self.turn_manager.state == TurnState.LISTENING:
                    speech_buffer.append(chunk.audio_data)

        except KeyboardInterrupt:
            if STRUCTLOG_AVAILABLE:
                logger.info("session_interrupted_by_user")
            else:
                print(f"\n⏹️  Stopped by user")

        finally:
            await self._finalize_session()

    async def _finalize_session(self):
        """Finalize voice session"""
        # Export session
        session_data = self.conversation_memory.export_session()

        # Save to file
        session_file = f"voice_session_{self.conversation_memory.session_id}.json"

        import json
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        if STRUCTLOG_AVAILABLE:
            logger.info("session_finalized",
                session_id=self.conversation_memory.session_id,
                total_turns=len(self.conversation_memory.turns),
                file=session_file)
        else:
            print(f"\n{'='*70}")
            print(f"SESSION COMPLETE: {self.conversation_memory.session_id}")
            print(f"{'='*70}")
            print(f"Total turns: {len(self.conversation_memory.turns)}")
            print(f"Session saved: {session_file}")
            print(f"{'='*70}\n")


# ============================================================================
# Usage Examples
# ============================================================================

async def example_simple_voice_agent():
    """Example: Simple voice agent without HoloLoom"""

    # Create simple agent
    agent = VoiceAgent(
        orchestrator=None,  # No HoloLoom - just echo
        agent_name="SimpleBot",
        turn_mode="button"  # Manual control
    )

    # Simulate conversation
    responses = []
    for query in ["Hello!", "What's the weather?", "Tell me a joke"]:
        response = await agent.process_voice_input(query)
        responses.append(response)
        print(f"User: {query}")
        print(f"Bot: {response}\n")

    return responses


async def example_hololoom_integration():
    """Example: Full HoloLoom integration"""

    if not HOLOLOOM_AVAILABLE:
        print("HoloLoom not available - skipping example")
        return

    from HoloLoom.config import Config

    # Create HoloLoom orchestrator
    config = Config.fast()
    shards = []  # Add your memory shards

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Create voice agent
        agent = VoiceAgent(
            orchestrator=orchestrator,
            agent_name="Elle",
            voice="nova",
            turn_mode="hybrid"
        )

        # Simulate conversation
        queries = [
            "What is Thompson Sampling?",
            "How does it balance exploration and exploitation?",
            "Give me an example"
        ]

        for query in queries:
            response = await agent.process_voice_input(query)
            print(f"User: {query}")
            print(f"Elle: {response}\n")


if __name__ == "__main__":
    print(__doc__)

    print("\n" + "="*70)
    print("VOICE AGENT READY!")
    print("="*70)
    print("\nFeatures:")
    print("  ✅ OpenAI TTS integration")
    print("  ✅ Conversation memory with HoloLoom KG")
    print("  ✅ Turn-taking (button/VAD/hybrid)")
    print("  ✅ WeavingOrchestrator integration")
    print("  ✅ Interrupt handling")
    print("\nInstallation:")
    print("  pip install openai pydub webrtcvad structlog")
    print("\nRun:")
    print("  python3 -m HoloLoom.voice.voice_agent")
    print("\n" + "="*70)

    # Uncomment to run examples:
    # asyncio.run(example_simple_voice_agent())
    # asyncio.run(example_hololoom_integration())

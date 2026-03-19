"""
Voice Integration for Agentic Dashboard
=========================================

Adds dual-prompting TTS capabilities to the agentic dashboard.

Features:
- Voice input (Whisper transcription)
- Voice output (BARK/ElevenLabs TTS with dual-prompting)
- Business-aware vocal delivery (auto-generates emotion/pacing)
- WebSocket streaming for real-time voice

Usage:
    from hololoom.apps.workflow_builder.voice_integration import VoiceIntegration

    voice = VoiceIntegration(tts_backend="bark")
    await voice.setup()

    # Voice input
    transcript = await voice.transcribe_audio(audio_bytes)

    # Voice output
    audio = await voice.speak("Your revenue is $450", metric_type="neutral")
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Try importing voice components
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cos"))
    from cos.interface.voice_chat import DualPrompt, VocalInstructions, VoiceChat
    VOICE_CHAT_AVAILABLE = True
except ImportError:
    VOICE_CHAT_AVAILABLE = False
    logger.warning("VoiceChat not available (install BARK or ElevenLabs)")

# Try importing Whisper
try:
    from hololoom.spinningWheel.whisper_spinner import WHISPER_AVAILABLE, WhisperSpinner
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available (install openai-whisper)")


class VoiceIntegration:
    """
    Voice capabilities for agentic dashboard.

    Combines Whisper (input) + BARK/ElevenLabs (output) with dual-prompting.
    """

    def __init__(
        self,
        tts_backend: Literal["bark", "elevenlabs", "pyttsx3"] = "bark",
        whisper_model: str = "base",
        auto_speak: bool = False
    ):
        """
        Initialize voice integration.

        Args:
            tts_backend: TTS backend ("bark", "elevenlabs", "pyttsx3")
            whisper_model: Whisper model size ("tiny", "base", "small", "medium", "large")
            auto_speak: Auto-speak all responses (default False)
        """
        self.tts_backend = tts_backend
        self.whisper_model = whisper_model
        self.auto_speak = auto_speak

        self.voice_chat = None
        self.whisper = None

        # Voice availability flags
        self.tts_available = VOICE_CHAT_AVAILABLE
        self.transcription_available = WHISPER_AVAILABLE

    async def setup(self):
        """Initialize voice components"""
        logger.info("Setting up voice integration...")

        # Setup TTS
        if VOICE_CHAT_AVAILABLE:
            try:
                self.voice_chat = VoiceChat(backend=self.tts_backend)
                logger.info(f"  - TTS: {self.tts_backend} enabled")
            except Exception as e:
                logger.warning(f"  - TTS initialization failed: {e}")
                self.tts_available = False
        else:
            logger.info("  - TTS: unavailable (install BARK or ElevenLabs)")

        # Setup Whisper
        if WHISPER_AVAILABLE:
            try:
                self.whisper = WhisperSpinner(model_size=self.whisper_model)
                logger.info(f"  - Whisper: {self.whisper_model} model enabled")
            except Exception as e:
                logger.warning(f"  - Whisper initialization failed: {e}")
                self.transcription_available = False
        else:
            logger.info("  - Whisper: unavailable (install openai-whisper)")

        logger.info(f"Voice integration ready (TTS: {self.tts_available}, Transcription: {self.transcription_available})")

    async def transcribe_audio(self, audio_data: bytes, save_path: str | None = None) -> str | None:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_data: Audio file bytes
            save_path: Optional path to save audio file

        Returns:
            Transcribed text or None if failed
        """
        if not self.transcription_available or not self.whisper:
            logger.error("Whisper transcription not available")
            return None

        try:
            # Save to temporary file if no save_path provided
            if save_path:
                audio_path = Path(save_path)
            else:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                audio_path = Path(temp_file.name)
                temp_file.write(audio_data)
                temp_file.close()

            logger.info(f"Transcribing audio: {audio_path}")

            # Transcribe
            result = await self.whisper.spin(audio_path)

            if result.success and result.shards:
                transcript = ' '.join(shard.content for shard in result.shards)
                logger.info(f"  ✓ Transcribed: {transcript[:50]}...")
                return transcript
            else:
                logger.error(f"  ✗ Transcription failed: {result.error_message}")
                return None

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    async def speak(
        self,
        text: str,
        metric_type: Literal["positive", "negative", "neutral", "warning"] | None = None,
        emotion: str | None = None,
        save_path: str | None = None
    ) -> bytes | None:
        """
        Generate speech from text using TTS.

        Args:
            text: Text to speak
            metric_type: Business metric type (auto-generates vocal delivery)
            emotion: Manual emotion override
            save_path: Optional path to save audio file

        Returns:
            Audio bytes or None if failed
        """
        if not self.tts_available or not self.voice_chat:
            logger.error("TTS not available")
            return None

        try:
            logger.info(f"Generating speech: {text[:50]}...")

            # Use business prompt helper if metric_type provided
            if metric_type:
                prompt = self.voice_chat.create_business_prompt(text, metric_type)
                audio = await self.voice_chat.speak(prompt, save_path=save_path)
            else:
                # Use legacy emotion API
                audio = await self.voice_chat.speak(text, emotion=emotion or "neutral", save_path=save_path)

            if audio:
                logger.info(f"  ✓ Generated {len(audio)} bytes of audio")
            else:
                logger.warning("  ⚠ No audio generated (pyttsx3 backend?)")

            return audio

        except Exception as e:
            logger.error(f"Speech generation error: {e}")
            return None

    async def speak_response(
        self,
        response_data: dict[str, Any],
        save_path: str | None = None
    ) -> bytes | None:
        """
        Speak an agentic response with appropriate vocal delivery.

        Analyzes confidence, mode, and content to determine delivery.

        Args:
            response_data: Response from agentic orchestrator
            save_path: Optional path to save audio file

        Returns:
            Audio bytes or None if failed
        """
        if not self.tts_available or not self.voice_chat:
            return None

        try:
            # Extract response text and metadata
            response_text = response_data.get('response', '')
            confidence = response_data.get('confidence', 0.5)
            mode = response_data.get('mode', 'direct')

            # Determine metric type based on confidence and mode
            if confidence < 0.5:
                metric_type = "warning"  # Low confidence - cautious delivery
            elif confidence > 0.85:
                metric_type = "positive"  # High confidence - confident delivery
            elif mode == "verify":
                metric_type = "neutral"  # Verification - factual delivery
            else:
                metric_type = "neutral"  # Default

            logger.info(f"Speaking response (confidence: {confidence:.2f}, mode: {mode}, delivery: {metric_type})")

            return await self.speak(response_text, metric_type=metric_type, save_path=save_path)

        except Exception as e:
            logger.error(f"Response speech error: {e}")
            return None

    def get_status(self) -> dict[str, Any]:
        """Get voice integration status"""
        return {
            'tts_available': self.tts_available,
            'tts_backend': self.tts_backend if self.tts_available else None,
            'transcription_available': self.transcription_available,
            'whisper_model': self.whisper_model if self.transcription_available else None,
            'auto_speak': self.auto_speak
        }


# Singleton instance (initialized in server startup)
voice_integration: VoiceIntegration | None = None


async def create_voice_integration(
    tts_backend: str = "bark",
    whisper_model: str = "base",
    auto_speak: bool = False
) -> VoiceIntegration:
    """
    Factory function to create and setup voice integration.

    Args:
        tts_backend: TTS backend ("bark", "elevenlabs", "pyttsx3")
        whisper_model: Whisper model size
        auto_speak: Auto-speak all responses

    Returns:
        Initialized VoiceIntegration instance
    """
    global voice_integration

    voice_integration = VoiceIntegration(
        tts_backend=tts_backend,
        whisper_model=whisper_model,
        auto_speak=auto_speak
    )

    await voice_integration.setup()

    return voice_integration


def get_voice_integration() -> VoiceIntegration | None:
    """Get global voice integration instance"""
    return voice_integration

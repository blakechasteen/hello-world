"""
HoloLoom COS - Voice Input Integration

Uses HoloLoom's existing Whisper integration for voice transcription.

Philosophy: Voice-first, hands-free business logging
"""

import asyncio
from pathlib import Path
from typing import Optional, Callable
import sys
import wave
import io

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use HoloLoom's Whisper integration
try:
    from HoloLoom.spinningWheel.whisper_spinner import WhisperSpinner, WhisperConfig
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not available. Install with: pip install openai-whisper")

from cos.core.parser import parse_input
from cos.core.storage import EventStore
from cos.core.types import EventSource, VerificationRequest


class VoiceInputHandler:
    """
    Handles voice input for COS.

    Workflow:
    1. Record audio (mic or file)
    2. Transcribe with Whisper
    3. Parse intent
    4. HITL if low confidence
    5. Store event
    """

    def __init__(
        self,
        store: EventStore,
        whisper_model: str = "base",
        auto_verify: bool = False
    ):
        self.store = store
        self.auto_verify = auto_verify

        if WHISPER_AVAILABLE:
            self.whisper = WhisperSpinner(WhisperConfig(
                model_size=whisper_model,
                language="en"
            ))
        else:
            self.whisper = None

    async def transcribe_audio_file(self, audio_path: str) -> str:
        """Transcribe audio file to text"""
        if not self.whisper:
            raise RuntimeError("Whisper not available")

        # Use HoloLoom's Whisper spinner
        result = await self.whisper.transcribe_audio(audio_path)
        return result['text']

    async def process_voice_input(
        self,
        audio_path: str,
        on_verification_needed: Optional[Callable] = None
    ) -> dict:
        """
        Complete voice input workflow.

        Args:
            audio_path: Path to audio file
            on_verification_needed: Callback for HITL verification

        Returns:
            {
                'transcript': str,
                'event_id': int,
                'verified': bool,
                'confidence': float
            }
        """
        # 1. Transcribe
        transcript = await self.transcribe_audio_file(audio_path)

        # 2. Parse
        intent, verification = parse_input(transcript, EventSource.VOICE)

        # 3. Handle verification
        verified = True
        if verification and not self.auto_verify:
            if on_verification_needed:
                verified = await on_verification_needed(verification)
            else:
                # Default: print and ask
                print(f"\n{verification}")
                response = input("\n✓ Is this correct? [y/n]: ").strip().lower()
                verified = (response == 'y')

        if not verified:
            return {
                'transcript': transcript,
                'event_id': None,
                'verified': False,
                'confidence': float(intent.confidence)
            }

        # 4. Store event
        event = intent.to_event(transcript, EventSource.VOICE)
        event.verified = True
        event.verification_note = "Voice input verified"

        event_id = await self.store.store(event)

        return {
            'transcript': transcript,
            'event_id': event_id,
            'verified': True,
            'confidence': float(intent.confidence)
        }

    async def process_text_as_voice(self, text: str) -> dict:
        """
        Process text input as if it were voice (for testing).

        Useful for development without actual audio.
        """
        intent, verification = parse_input(text, EventSource.VOICE)

        # Auto-verify for testing
        event = intent.to_event(text, EventSource.VOICE)
        event.verified = True
        event.verification_note = "Auto-verified (testing)"

        event_id = await self.store.store(event)

        return {
            'transcript': text,
            'event_id': event_id,
            'verified': True,
            'confidence': float(intent.confidence)
        }


# Web API endpoint (for web interface)
async def voice_input_endpoint(audio_data: bytes, store: EventStore) -> dict:
    """
    FastAPI endpoint for voice input.

    Usage:
        POST /voice/input
        Content-Type: audio/wav
        Body: <audio data>
    """
    # Save audio temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio_data)
        temp_path = f.name

    try:
        handler = VoiceInputHandler(store)
        result = await handler.process_voice_input(temp_path)
        return result
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


async def demo():
    """Demo voice input"""
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "="*60)
    print("HoloLoom COS - Voice Input Demo")
    print("="*60)

    store = EventStore("test_cos.db")
    handler = VoiceInputHandler(store)

    # Test with text (simulating voice)
    print("\n\nTEST 1: Process text as voice (for testing)")
    print("-"*60)

    test_inputs = [
        "Worked on bread for three hours today",
        "Sold eight loaves for forty eight dollars",
        "Bought some flour, I think it was about thirty bucks"
    ]

    for text in test_inputs:
        print(f"\nInput: {text}")
        result = await handler.process_text_as_voice(text)
        print(f"  Event ID: {result['event_id']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Verified: {'✓' if result['verified'] else '✗'}")

    print("\n\n" + "="*60)
    print("✅ Voice Input Demo Complete")
    print("="*60)
    print("\nNext: Record actual audio and test transcription!")
    print("Note: Requires 'pip install openai-whisper'")


if __name__ == '__main__':
    asyncio.run(demo())

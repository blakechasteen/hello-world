"""
Speech-to-Text using OpenAI Whisper

Provides local speech recognition with fallback to simple audio recording.
Created: 2025-11-15
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import sys


class WhisperSTT:
    """
    Speech-to-Text using OpenAI Whisper (local inference)

    Supports:
    - Local Whisper model (no API calls needed)
    - Microphone input via pyaudio
    - Audio file input
    - Graceful degradation if Whisper not available
    """

    def __init__(
        self,
        model: str = "base",
        device: str = "cpu",
        language: str = "en",
        timeout: float = 30.0
    ):
        """
        Initialize Whisper STT

        Args:
            model: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda)
            language: Language code (en, es, fr, etc.)
            timeout: Timeout for recording in seconds
        """
        self.model = model
        self.device = device
        self.language = language
        self.timeout = timeout

        self.whisper_available = self._check_whisper()
        self.pyaudio_available = self._check_pyaudio()

        if not self.whisper_available:
            print("⚠ Whisper not available. Install with: pip install openai-whisper")
            print("  Falling back to command-line transcription if available.")

        if not self.pyaudio_available:
            print("⚠ PyAudio not available. Install with: pip install pyaudio")
            print("  Will require pre-recorded audio files.")

    def _check_whisper(self) -> bool:
        """Check if Whisper is available"""
        try:
            import whisper
            return True
        except ImportError:
            return False

    def _check_pyaudio(self) -> bool:
        """Check if PyAudio is available"""
        try:
            import pyaudio
            return True
        except ImportError:
            return False

    async def record_audio(
        self,
        duration: float = 5.0,
        sample_rate: int = 16000
    ) -> Optional[str]:
        """
        Record audio from microphone

        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Path to temporary audio file, or None if recording failed
        """
        if not self.pyaudio_available:
            print("✗ PyAudio not available. Cannot record from microphone.")
            return None

        try:
            import pyaudio
            import wave

            # Create temporary audio file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()

            # Recording parameters
            channels = 1
            chunk = 1024

            # Initialize PyAudio
            p = pyaudio.PyAudio()

            # Open stream
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk
            )

            print(f"🎤 Recording... (press Ctrl+C to stop)")

            frames = []
            num_frames = int(sample_rate / chunk * duration)

            try:
                for _ in range(num_frames):
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(data)
            except KeyboardInterrupt:
                print("⏸ Recording stopped.")

            # Stop stream
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Write audio file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(frames))

            print(f"✓ Audio recorded: {temp_path}")
            return temp_path

        except Exception as e:
            print(f"✗ Recording failed: {e}")
            return None

    async def transcribe(self, audio_path: str) -> Tuple[Optional[str], dict]:
        """
        Transcribe audio file using Whisper

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (transcribed text, metadata dict)
        """
        audio_file = Path(audio_path)

        if not audio_file.exists():
            return None, {"error": f"Audio file not found: {audio_path}"}

        if not self.whisper_available:
            return await self._transcribe_fallback(audio_path)

        try:
            import whisper

            # Load model
            print(f"Loading Whisper {self.model} model...")
            model = whisper.load_model(
                self.model,
                device=self.device
            )

            # Transcribe
            print(f"Transcribing {audio_file.name}...")
            result = model.transcribe(
                audio_path,
                language=self.language,
                verbose=False
            )

            text = result["text"].strip()
            metadata = {
                "language": result.get("language", self.language),
                "duration": result.get("duration"),
                "model": self.model,
                "provider": "whisper"
            }

            print(f"✓ Transcription: {text[:100]}...")

            return text, metadata

        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            return None, {"error": str(e), "provider": "whisper"}

    async def _transcribe_fallback(
        self,
        audio_path: str
    ) -> Tuple[Optional[str], dict]:
        """
        Fallback transcription using ffmpeg + whisper CLI
        """
        try:
            # Try using whisper command-line
            result = subprocess.run(
                [
                    "whisper",
                    audio_path,
                    "--model", self.model,
                    "--language", self.language,
                    "--output_format", "json",
                    "--output_dir", "/tmp"
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                return None, {"error": result.stderr, "provider": "whisper_cli"}

            # Read JSON output
            json_path = Path(audio_path).stem + ".json"
            with open(json_path) as f:
                output = json.load(f)

            text = output.get("text", "").strip()
            metadata = {
                "language": self.language,
                "model": self.model,
                "provider": "whisper_cli"
            }

            return text, metadata

        except FileNotFoundError:
            return None, {
                "error": "Whisper CLI not found. Install with: pip install openai-whisper",
                "provider": "whisper_cli"
            }
        except Exception as e:
            return None, {"error": str(e), "provider": "whisper_cli"}

    async def transcribe_with_fallback(
        self,
        audio_path: str,
        max_retries: int = 3
    ) -> Tuple[Optional[str], dict]:
        """
        Transcribe with automatic fallback

        Args:
            audio_path: Path to audio file
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (transcribed text, metadata)
        """
        for attempt in range(max_retries):
            text, metadata = await self.transcribe(audio_path)

            if text:
                return text, metadata

            if attempt < max_retries - 1:
                print(f"⚠ Transcription attempt {attempt + 1} failed. Retrying...")
                await asyncio.sleep(2 ** attempt)

        return None, {"error": "All transcription attempts failed"}


async def demo_record_and_transcribe():
    """Demo: Record audio and transcribe it"""
    print("\n" + "="*60)
    print("WHISPER STT DEMO")
    print("="*60)

    stt = WhisperSTT(model="tiny")  # Use tiny model for demo

    # Record audio
    audio_path = await stt.record_audio(duration=5.0)

    if not audio_path:
        print("✗ Recording failed. Using sample text instead.")
        return

    # Transcribe
    text, metadata = await stt.transcribe_with_fallback(audio_path)

    if text:
        print(f"\n✓ Transcribed text:")
        print(f"  {text}")
        print(f"\nMetadata:")
        print(f"  Language: {metadata.get('language')}")
        print(f"  Provider: {metadata.get('provider')}")
    else:
        print(f"✗ Transcription failed: {metadata.get('error')}")


if __name__ == "__main__":
    asyncio.run(demo_record_and_transcribe())

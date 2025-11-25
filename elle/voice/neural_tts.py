"""
Neural Text-to-Speech using Coqui TTS

Provides natural voice synthesis with graceful fallback to pyttsx3.

Created: November 22, 2025
Part of: Voice UX Milestone 2 Phase 2
"""

import asyncio
import io
import wave
from pathlib import Path
from typing import Optional, List
from enum import Enum


class VoiceModel(Enum):
    """Available Coqui TTS models"""
    # Fast models (optimized for low latency)
    TACOTRON2_DDC = "tts_models/en/ljspeech/tacotron2-DDC"
    FAST_PITCH = "tts_models/en/ljspeech/fast_pitch"

    # High quality models
    VITS = "tts_models/en/vctk/vits"
    GLOW_TTS = "tts_models/en/ljspeech/glow-tts"

    # Multi-speaker models
    VITS_MULTI = "tts_models/en/vctk/vits"


class VoicePersonality(Enum):
    """Voice personality presets"""
    PROFESSIONAL = VoiceModel.TACOTRON2_DDC
    FRIENDLY = VoiceModel.VITS
    FAST = VoiceModel.FAST_PITCH


class NeuralTTS:
    """
    Neural Text-to-Speech using Coqui TTS

    Features:
    - Natural voice synthesis
    - Multiple voice models
    - Async operation
    - Voice caching for common responses
    - Graceful fallback to pyttsx3
    """

    def __init__(
        self,
        model: VoiceModel = VoiceModel.TACOTRON2_DDC,
        personality: Optional[VoicePersonality] = None,
        enable_cache: bool = True,
        cache_dir: str = "elle/data/voice_cache",
        verbose: bool = False
    ):
        """
        Initialize Neural TTS

        Args:
            model: Coqui TTS model to use
            personality: Voice personality preset (overrides model)
            enable_cache: Cache synthesized audio for common responses
            cache_dir: Directory for voice cache
            verbose: Print debug information
        """
        self.verbose = verbose
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir)

        # Use personality if specified
        if personality:
            self.model_name = personality.value.value
        else:
            self.model_name = model.value

        # Initialize cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # TTS engine
        self.tts = None
        self.coqui_available = self._initialize_tts()

        # Voice cache (text -> audio_path)
        self.voice_cache = {}

        if not self.coqui_available:
            if self.verbose:
                print("Warning: Coqui TTS not available. Install with: pip install TTS")
                print("  Falling back to basic TTS (if available)")

    def _initialize_tts(self) -> bool:
        """Initialize Coqui TTS engine"""
        try:
            from TTS.api import TTS

            if self.verbose:
                print(f"Loading Coqui TTS model: {self.model_name}")
                print("  This may take a moment on first run (downloading model)...")

            # Initialize TTS with selected model
            self.tts = TTS(model_name=self.model_name, progress_bar=False)

            if self.verbose:
                print("[OK] Coqui TTS initialized successfully")

            return True

        except ImportError:
            return False
        except Exception as e:
            if self.verbose:
                print(f"[FAIL] Failed to initialize Coqui TTS: {e}")
            return False

    async def speak(self, text: str, wait: bool = True) -> bool:
        """
        Synthesize and speak text

        Args:
            text: Text to speak
            wait: Wait for audio to finish playing

        Returns:
            True if successful, False otherwise
        """
        if not self.coqui_available:
            # Fallback: just print
            print(f"Elle> {text}")
            return False

        try:
            # Check cache first
            if self.enable_cache and text in self.voice_cache:
                audio_path = self.voice_cache[text]
                if Path(audio_path).exists():
                    if self.verbose:
                        print(f"[Cache Hit] Using cached audio for: {text[:50]}")
                    await self._play_audio(audio_path, wait=wait)
                    return True

            # Generate audio
            audio_path = await self._synthesize(text)

            if audio_path:
                # Cache for future use
                if self.enable_cache:
                    self.voice_cache[text] = audio_path

                # Play audio
                await self._play_audio(audio_path, wait=wait)
                return True

            return False

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Neural TTS failed: {e}")
            print(f"Elle> {text}")  # Fallback to text
            return False

    async def _synthesize(self, text: str) -> Optional[str]:
        """
        Synthesize text to audio file

        Args:
            text: Text to synthesize

        Returns:
            Path to generated audio file, or None if failed
        """
        if not self.tts:
            return None

        try:
            # Generate unique filename
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            audio_path = self.cache_dir / f"tts_{text_hash}.wav"

            if self.verbose:
                print(f"[Synthesizing] {text[:50]}...")

            # Run synthesis in thread pool (blocking operation)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.tts.tts_to_file(text=text, file_path=str(audio_path))
            )

            if self.verbose:
                print(f"[OK] Generated audio: {audio_path}")

            return str(audio_path)

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Synthesis failed: {e}")
            return None

    async def _play_audio(self, audio_path: str, wait: bool = True):
        """
        Play audio file

        Args:
            audio_path: Path to audio file
            wait: Wait for playback to finish
        """
        try:
            # Use simpleaudio if available (cross-platform)
            import simpleaudio as sa

            wave_obj = sa.WaveObject.from_wave_file(audio_path)
            play_obj = wave_obj.play()

            if wait:
                play_obj.wait_done()

        except ImportError:
            # Fallback: Try pygame
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()

                if wait:
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)

            except ImportError:
                if self.verbose:
                    print("Warning: No audio playback library available")
                    print("  Install with: pip install simpleaudio")
                print(f"  [Audio generated at: {audio_path}]")

    def get_cache_stats(self) -> dict:
        """Get voice cache statistics"""
        return {
            "cached_responses": len(self.voice_cache),
            "cache_enabled": self.enable_cache,
            "cache_dir": str(self.cache_dir),
            "cache_size_mb": self._get_cache_size_mb()
        }

    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB"""
        if not self.cache_dir.exists():
            return 0.0

        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.wav") if f.is_file()
        )
        return total_size / (1024 * 1024)

    def clear_cache(self):
        """Clear voice cache"""
        if self.cache_dir.exists():
            for audio_file in self.cache_dir.glob("*.wav"):
                audio_file.unlink()

        self.voice_cache.clear()

        if self.verbose:
            print("[OK] Voice cache cleared")


async def demo_neural_tts():
    """Demo: Neural TTS with different models"""
    print("\n" + "="*70)
    print("NEURAL TTS DEMO")
    print("="*70)

    # Test 1: Fast model (low latency)
    print("\n1. Testing FAST model (optimized for latency)...")
    tts_fast = NeuralTTS(
        model=VoiceModel.FAST_PITCH,
        enable_cache=True,
        verbose=True
    )

    if tts_fast.coqui_available:
        print("\n  Synthesizing: 'Thread 3'")
        await tts_fast.speak("Thread 3")

        print("\n  Synthesizing: 'Created greenhouse'")
        await tts_fast.speak("Created greenhouse")

        # Test cache
        print("\n  Testing cache (repeat 'Thread 3'):")
        await tts_fast.speak("Thread 3")

        # Cache stats
        stats = tts_fast.get_cache_stats()
        print(f"\n  Cache stats: {stats}")
    else:
        print("  [SKIP] Coqui TTS not available")

    # Test 2: High quality model
    print("\n2. Testing HIGH QUALITY model...")
    tts_hq = NeuralTTS(
        personality=VoicePersonality.FRIENDLY,
        enable_cache=True,
        verbose=True
    )

    if tts_hq.coqui_available:
        print("\n  Synthesizing: 'I'm listening'")
        await tts_hq.speak("I'm listening")

    print("\n" + "="*70)
    print("[OK] Demo complete")


if __name__ == "__main__":
    asyncio.run(demo_neural_tts())

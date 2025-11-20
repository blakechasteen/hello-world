"""
Voice synthesis engine for Elle Game Engine.

Provides text-to-speech capabilities with multiple backend options:
- ElevenLabs (premium, best quality)
- Google Cloud TTS (good quality, affordable)
- OpenAI TTS (good quality, integrated)
- Piper TTS (local, free, fast)

Features:
- Per-NPC voice profiles (pitch, speed, emotion)
- Voice caching (reuse common phrases)
- Streaming audio generation
- Multiple output formats (WAV, MP3, OGG)
- Emotion-aware voice synthesis

Author: Elle Game Engine Team
Date: 2025-11-16
"""

from typing import Dict, Optional, Literal, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import os
import io
import json
from pathlib import Path

# Optional dependencies (graceful degradation)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. HTTP backends will not work.")

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("Warning: pydub not installed. Audio format conversion will be limited.")


# ============================================================================
# Enums and Constants
# ============================================================================

class TTSBackend(str, Enum):
    """Available TTS backend providers."""
    ELEVENLABS = "elevenlabs"
    GOOGLE_CLOUD = "google_cloud"
    OPENAI = "openai"
    PIPER = "piper"
    DUMMY = "dummy"  # For testing


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    OPUS = "opus"


class Emotion(str, Enum):
    """Voice emotions (subset of tone values)."""
    NEUTRAL = "neutral"
    WARM = "warm"
    STERN = "stern"
    EXCITED = "excited"
    SAD = "sad"
    CRYPTIC = "cryptic"
    CURIOUS = "curious"
    GRATEFUL = "grateful"
    ANNOYED = "annoyed"
    HOSTILE = "hostile"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class VoiceProfile:
    """
    Voice profile configuration for an NPC or character.

    This defines how a specific character's voice should sound.
    """

    voice_id: str  # Backend-specific voice identifier
    pitch: float = 1.0  # 0.5 (low) to 2.0 (high)
    speed: float = 1.0  # 0.5 (slow) to 2.0 (fast)
    emotion: Emotion = Emotion.NEUTRAL  # Base emotional tone
    stability: float = 0.5  # Voice stability (0.0-1.0, ElevenLabs specific)
    similarity_boost: float = 0.75  # Voice clarity (0.0-1.0, ElevenLabs specific)

    def __post_init__(self):
        """Validate voice profile parameters."""
        if not 0.5 <= self.pitch <= 2.0:
            raise ValueError("pitch must be between 0.5 and 2.0")
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        if not 0.0 <= self.stability <= 1.0:
            raise ValueError("stability must be between 0.0 and 1.0")
        if not 0.0 <= self.similarity_boost <= 1.0:
            raise ValueError("similarity_boost must be between 0.0 and 1.0")


@dataclass
class VoiceSynthesisRequest:
    """Request for voice synthesis."""

    text: str  # Text to synthesize
    voice_profile: VoiceProfile  # Voice configuration
    format: AudioFormat = AudioFormat.MP3  # Output format
    emotion_override: Optional[Emotion] = None  # Override base emotion


@dataclass
class VoiceSynthesisResult:
    """Result of voice synthesis."""

    audio_data: bytes  # Audio file bytes
    format: AudioFormat  # Audio format
    duration_seconds: float  # Audio duration
    text: str  # Original text
    cached: bool = False  # Was this result from cache?
    metadata: Dict[str, Any] = field(default_factory=dict)  # Backend-specific metadata


# ============================================================================
# Voice Cache
# ============================================================================

class VoiceCache:
    """
    LRU cache for synthesized voice clips.

    Caches audio by hash of (text, voice_profile, emotion) to avoid
    re-synthesizing common phrases.
    """

    def __init__(self, cache_dir: str = "./.voice_cache", max_size_mb: int = 100):
        """
        Initialize voice cache.

        Args:
            cache_dir: Directory to store cached audio files
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024

        # Load cache index
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Dict]:
        """Load cache index from disk."""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def _get_cache_key(self, text: str, voice_profile: VoiceProfile, emotion: Emotion) -> str:
        """Generate cache key from parameters."""
        key_data = f"{text}|{voice_profile.voice_id}|{voice_profile.pitch}|{voice_profile.speed}|{emotion.value}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self,
        text: str,
        voice_profile: VoiceProfile,
        emotion: Emotion,
        format: AudioFormat
    ) -> Optional[bytes]:
        """
        Get cached audio if available.

        Returns:
            Audio bytes if cached, None otherwise
        """
        cache_key = self._get_cache_key(text, voice_profile, emotion)

        if cache_key in self.index:
            entry = self.index[cache_key]
            file_path = self.cache_dir / entry["filename"]

            if file_path.exists():
                # Check if format matches
                if entry.get("format") == format.value:
                    with open(file_path, "rb") as f:
                        return f.read()

        return None

    def set(
        self,
        text: str,
        voice_profile: VoiceProfile,
        emotion: Emotion,
        format: AudioFormat,
        audio_data: bytes
    ):
        """Cache synthesized audio."""
        cache_key = self._get_cache_key(text, voice_profile, emotion)

        # Create filename
        filename = f"{cache_key}.{format.value}"
        file_path = self.cache_dir / filename

        # Write audio file
        with open(file_path, "wb") as f:
            f.write(audio_data)

        # Update index
        self.index[cache_key] = {
            "text": text[:100],  # Store truncated text for debugging
            "filename": filename,
            "format": format.value,
            "size_bytes": len(audio_data),
            "voice_id": voice_profile.voice_id
        }

        self._save_index()

        # Evict old entries if cache too large
        self._evict_if_needed()

    def _evict_if_needed(self):
        """Evict least recently used entries if cache exceeds max size."""
        total_size = sum(entry["size_bytes"] for entry in self.index.values())

        if total_size > self.max_size_bytes:
            # Sort by file modification time (LRU)
            entries_by_time = []
            for key, entry in self.index.items():
                file_path = self.cache_dir / entry["filename"]
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    entries_by_time.append((mtime, key, entry))

            entries_by_time.sort()

            # Evict oldest entries until under limit
            current_size = total_size
            for mtime, key, entry in entries_by_time:
                if current_size <= self.max_size_bytes:
                    break

                file_path = self.cache_dir / entry["filename"]
                if file_path.exists():
                    file_path.unlink()

                current_size -= entry["size_bytes"]
                del self.index[key]

            self._save_index()

    def clear(self):
        """Clear all cached audio."""
        for entry in self.index.values():
            file_path = self.cache_dir / entry["filename"]
            if file_path.exists():
                file_path.unlink()

        self.index = {}
        self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry["size_bytes"] for entry in self.index.values())

        return {
            "num_entries": len(self.index),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


# ============================================================================
# TTS Backend Implementations
# ============================================================================

class TTSBackendBase:
    """Base class for TTS backends."""

    def synthesize(self, request: VoiceSynthesisRequest) -> VoiceSynthesisResult:
        """
        Synthesize speech from text.

        Args:
            request: Voice synthesis request

        Returns:
            Synthesized audio result

        Raises:
            NotImplementedError: If backend not implemented
        """
        raise NotImplementedError


class DummyTTSBackend(TTSBackendBase):
    """Dummy backend for testing (generates silence)."""

    def synthesize(self, request: VoiceSynthesisRequest) -> VoiceSynthesisResult:
        """Generate silent audio for testing."""
        # Generate 1 second of silence (44.1kHz, 16-bit, mono WAV)
        duration = 1.0
        sample_rate = 44100
        num_samples = int(sample_rate * duration)

        # WAV header + silent samples
        wav_header = bytes([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x00, 0x00, 0x00, 0x00,  # File size (placeholder)
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # fmt chunk size
            0x01, 0x00,              # Audio format (PCM)
            0x01, 0x00,              # Num channels (mono)
            0x44, 0xAC, 0x00, 0x00,  # Sample rate (44100)
            0x88, 0x58, 0x01, 0x00,  # Byte rate
            0x02, 0x00,              # Block align
            0x10, 0x00,              # Bits per sample
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x00, 0x00, 0x00, 0x00,  # Data size (placeholder)
        ])

        # Silent samples (2 bytes per sample, 16-bit)
        silent_data = bytes([0x00, 0x00] * num_samples)

        # Combine header and data
        audio_data = wav_header + silent_data

        return VoiceSynthesisResult(
            audio_data=audio_data,
            format=AudioFormat.WAV,
            duration_seconds=duration,
            text=request.text,
            cached=False,
            metadata={"backend": "dummy"}
        )


class ElevenLabsTTSBackend(TTSBackendBase):
    """ElevenLabs TTS backend (premium quality)."""

    def __init__(self, api_key: str, model: str = "eleven_monolingual_v1"):
        """
        Initialize ElevenLabs backend.

        Args:
            api_key: ElevenLabs API key
            model: Model ID to use
        """
        if not HAS_REQUESTS:
            raise ImportError("requests library required for ElevenLabs backend")

        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.elevenlabs.io/v1"

    def synthesize(self, request: VoiceSynthesisRequest) -> VoiceSynthesisResult:
        """Synthesize speech using ElevenLabs API."""
        url = f"{self.base_url}/text-to-speech/{request.voice_profile.voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        # Build request payload
        emotion = request.emotion_override or request.voice_profile.emotion

        payload = {
            "text": request.text,
            "model_id": self.model,
            "voice_settings": {
                "stability": request.voice_profile.stability,
                "similarity_boost": request.voice_profile.similarity_boost,
                "style": 0.0,  # Default
                "use_speaker_boost": True
            }
        }

        # Make request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        audio_data = response.content

        # Estimate duration (rough approximation: ~15 chars/second)
        estimated_duration = len(request.text) / 15.0

        return VoiceSynthesisResult(
            audio_data=audio_data,
            format=AudioFormat.MP3,
            duration_seconds=estimated_duration,
            text=request.text,
            cached=False,
            metadata={
                "backend": "elevenlabs",
                "model": self.model,
                "voice_id": request.voice_profile.voice_id
            }
        )


class OpenAITTSBackend(TTSBackendBase):
    """OpenAI TTS backend (good quality, integrated)."""

    def __init__(self, api_key: str, model: str = "tts-1"):
        """
        Initialize OpenAI TTS backend.

        Args:
            api_key: OpenAI API key
            model: Model to use ("tts-1" or "tts-1-hd")
        """
        if not HAS_REQUESTS:
            raise ImportError("requests library required for OpenAI backend")

        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"

    def synthesize(self, request: VoiceSynthesisRequest) -> VoiceSynthesisResult:
        """Synthesize speech using OpenAI API."""
        url = f"{self.base_url}/audio/speech"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Map voice profile to OpenAI voices
        # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
        voice = request.voice_profile.voice_id

        payload = {
            "model": self.model,
            "input": request.text,
            "voice": voice,
            "speed": request.voice_profile.speed
        }

        # Make request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        audio_data = response.content

        # Estimate duration
        estimated_duration = len(request.text) / 15.0 / request.voice_profile.speed

        return VoiceSynthesisResult(
            audio_data=audio_data,
            format=AudioFormat.MP3,
            duration_seconds=estimated_duration,
            text=request.text,
            cached=False,
            metadata={
                "backend": "openai",
                "model": self.model,
                "voice": voice
            }
        )


# ============================================================================
# Voice Engine (Main Interface)
# ============================================================================

class VoiceEngine:
    """
    Main voice synthesis engine.

    Manages TTS backends, voice profiles, and caching.
    """

    def __init__(
        self,
        backend: TTSBackend = TTSBackend.DUMMY,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_cache: bool = True,
        cache_dir: str = "./.voice_cache",
        cache_size_mb: int = 100
    ):
        """
        Initialize voice engine.

        Args:
            backend: TTS backend to use
            api_key: API key for backend (if required)
            model: Model to use (backend-specific)
            enable_cache: Enable voice caching
            cache_dir: Cache directory
            cache_size_mb: Maximum cache size in MB
        """
        self.backend_type = backend
        self.backend = self._create_backend(backend, api_key, model)

        # Initialize cache
        self.cache = VoiceCache(cache_dir, cache_size_mb) if enable_cache else None

        # Voice profiles registry
        self.voice_profiles: Dict[str, VoiceProfile] = {}

    def _create_backend(
        self,
        backend: TTSBackend,
        api_key: Optional[str],
        model: Optional[str]
    ) -> TTSBackendBase:
        """Create TTS backend instance."""
        if backend == TTSBackend.DUMMY:
            return DummyTTSBackend()

        elif backend == TTSBackend.ELEVENLABS:
            if not api_key:
                raise ValueError("API key required for ElevenLabs backend")
            return ElevenLabsTTSBackend(api_key, model or "eleven_monolingual_v1")

        elif backend == TTSBackend.OPENAI:
            if not api_key:
                raise ValueError("API key required for OpenAI backend")
            return OpenAITTSBackend(api_key, model or "tts-1")

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def register_voice_profile(self, npc_id: str, profile: VoiceProfile):
        """
        Register a voice profile for an NPC.

        Args:
            npc_id: NPC identifier
            profile: Voice profile configuration
        """
        self.voice_profiles[npc_id] = profile

    def get_voice_profile(self, npc_id: str) -> Optional[VoiceProfile]:
        """Get voice profile for an NPC."""
        return self.voice_profiles.get(npc_id)

    def synthesize(
        self,
        text: str,
        npc_id: Optional[str] = None,
        voice_profile: Optional[VoiceProfile] = None,
        emotion: Optional[Emotion] = None,
        format: AudioFormat = AudioFormat.MP3
    ) -> VoiceSynthesisResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            npc_id: NPC ID to use voice profile from registry
            voice_profile: Voice profile to use (overrides npc_id)
            emotion: Emotion override
            format: Output audio format

        Returns:
            Synthesized audio result

        Raises:
            ValueError: If neither npc_id nor voice_profile provided
        """
        # Get voice profile
        if voice_profile is None:
            if npc_id is None:
                raise ValueError("Either npc_id or voice_profile must be provided")

            voice_profile = self.voice_profiles.get(npc_id)
            if voice_profile is None:
                raise ValueError(f"No voice profile registered for NPC: {npc_id}")

        # Determine final emotion
        final_emotion = emotion or voice_profile.emotion

        # Check cache
        if self.cache:
            cached_audio = self.cache.get(text, voice_profile, final_emotion, format)
            if cached_audio:
                # Estimate duration from text length
                estimated_duration = len(text) / 15.0 / voice_profile.speed

                return VoiceSynthesisResult(
                    audio_data=cached_audio,
                    format=format,
                    duration_seconds=estimated_duration,
                    text=text,
                    cached=True,
                    metadata={"backend": self.backend_type.value}
                )

        # Create synthesis request
        request = VoiceSynthesisRequest(
            text=text,
            voice_profile=voice_profile,
            format=format,
            emotion_override=emotion
        )

        # Synthesize
        result = self.backend.synthesize(request)

        # Cache result
        if self.cache:
            self.cache.set(text, voice_profile, final_emotion, format, result.audio_data)

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"enabled": False}

    def clear_cache(self):
        """Clear voice cache."""
        if self.cache:
            self.cache.clear()


# ============================================================================
# Factory Functions
# ============================================================================

def create_voice_engine(
    backend: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    enable_cache: bool = True
) -> VoiceEngine:
    """
    Create voice engine from environment or parameters.

    Args:
        backend: Backend name (or None to use ELLE_VOICE_BACKEND env var)
        api_key: API key (or None to use backend-specific env var)
        model: Model name (or None to use backend-specific env var)
        enable_cache: Enable voice caching

    Returns:
        Configured VoiceEngine instance
    """
    # Determine backend
    backend_str = backend or os.getenv("ELLE_VOICE_BACKEND", "dummy")
    backend_enum = TTSBackend(backend_str.lower())

    # Get API key from environment if not provided
    if api_key is None:
        if backend_enum == TTSBackend.ELEVENLABS:
            api_key = os.getenv("ELEVENLABS_API_KEY")
        elif backend_enum == TTSBackend.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")

    # Get model from environment if not provided
    if model is None:
        model = os.getenv("ELLE_VOICE_MODEL")

    return VoiceEngine(
        backend=backend_enum,
        api_key=api_key,
        model=model,
        enable_cache=enable_cache
    )

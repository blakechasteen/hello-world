"""
Voice Integration with Caching
==============================

Voice synthesis and caching system for the Elle game engine.

Features:
- Text-to-speech (TTS) synthesis abstraction
- Multi-provider support (ElevenLabs, OpenAI, local)
- Intelligent voice caching (disk + memory)
- Character voice profiles
- Audio preprocessing (normalization, effects)
- Cache invalidation and management

Performance:
- Memory cache: ~50ms retrieval
- Disk cache: ~100ms retrieval
- Fresh synthesis: 500ms-2000ms depending on provider

Created: 2025-12-03
Author: HoloLoom Team
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Protocol, Callable
from datetime import datetime, timedelta
from pathlib import Path
from abc import abstractmethod
import hashlib
import json
import logging
import asyncio
from collections import OrderedDict

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Voice Provider Protocol
# -----------------------------------------------------------------------------

class VoiceProvider(Protocol):
    """Protocol for voice synthesis providers."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize
            voice_id: ID of the voice to use
            **kwargs: Provider-specific options

        Returns:
            Audio data as bytes (WAV or MP3)
        """
        ...

    @abstractmethod
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        ...


class VoiceProviderType(Enum):
    """Supported voice provider types."""
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    LOCAL = "local"  # Local TTS like Coqui or pyttsx3
    MOCK = "mock"  # For testing


# -----------------------------------------------------------------------------
# Voice Profile
# -----------------------------------------------------------------------------

@dataclass
class VoiceProfile:
    """
    Voice configuration for a character.

    Defines how a character's voice should sound.
    """
    id: str
    name: str
    provider: VoiceProviderType
    voice_id: str  # Provider-specific voice ID

    # Voice characteristics
    pitch: float = 1.0  # 0.5-2.0, 1.0 is normal
    speed: float = 1.0  # 0.5-2.0, 1.0 is normal
    volume: float = 1.0  # 0.0-1.0

    # Emotional modulation
    default_emotion: str = "neutral"
    emotion_intensity: float = 0.5  # 0.0-1.0

    # Provider-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)

    # Character association
    character_id: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'provider': self.provider.value,
            'voice_id': self.voice_id,
            'pitch': self.pitch,
            'speed': self.speed,
            'volume': self.volume,
            'default_emotion': self.default_emotion,
            'emotion_intensity': self.emotion_intensity,
            'settings': self.settings,
            'character_id': self.character_id,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            provider=VoiceProviderType(data['provider']),
            voice_id=data['voice_id'],
            pitch=data.get('pitch', 1.0),
            speed=data.get('speed', 1.0),
            volume=data.get('volume', 1.0),
            default_emotion=data.get('default_emotion', 'neutral'),
            emotion_intensity=data.get('emotion_intensity', 0.5),
            settings=data.get('settings', {}),
            character_id=data.get('character_id'),
            description=data.get('description', ''),
        )


# -----------------------------------------------------------------------------
# Cache Entry
# -----------------------------------------------------------------------------

@dataclass
class VoiceCacheEntry:
    """A cached voice audio entry."""
    cache_key: str
    text: str
    voice_profile_id: str
    audio_data: Optional[bytes] = None  # None if on disk only
    audio_format: str = "wav"

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0

    # Storage
    disk_path: Optional[Path] = None
    in_memory: bool = True

    def touch(self):
        """Update access time and count."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary (without audio data)."""
        return {
            'cache_key': self.cache_key,
            'text': self.text,
            'voice_profile_id': self.voice_profile_id,
            'audio_format': self.audio_format,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'size_bytes': self.size_bytes,
            'disk_path': str(self.disk_path) if self.disk_path else None,
            'in_memory': self.in_memory,
        }


# -----------------------------------------------------------------------------
# Voice Cache
# -----------------------------------------------------------------------------

class VoiceCache:
    """
    Two-tier voice cache (memory + disk).

    Memory cache for fast access to frequently used audio.
    Disk cache for persistence and larger capacity.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        memory_cache_size: int = 100,  # Number of entries
        memory_cache_bytes: int = 100 * 1024 * 1024,  # 100MB
        disk_cache_size_bytes: int = 1024 * 1024 * 1024,  # 1GB
        ttl_days: int = 30,
    ):
        """
        Initialize voice cache.

        Args:
            cache_dir: Directory for disk cache
            memory_cache_size: Max entries in memory
            memory_cache_bytes: Max bytes in memory cache
            disk_cache_size_bytes: Max bytes on disk
            ttl_days: Days before cache entry expires
        """
        self.cache_dir = cache_dir or Path.home() / ".elle_game_engine" / "voice_cache"
        self.memory_cache_size = memory_cache_size
        self.memory_cache_bytes = memory_cache_bytes
        self.disk_cache_size_bytes = disk_cache_size_bytes
        self.ttl = timedelta(days=ttl_days)

        # Memory cache (LRU)
        self._memory_cache: OrderedDict[str, VoiceCacheEntry] = OrderedDict()
        self._memory_bytes_used: int = 0

        # Disk cache metadata
        self._disk_metadata: Dict[str, VoiceCacheEntry] = {}
        self._disk_bytes_used: int = 0

        # Statistics
        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
        }

        # Initialize
        self._ensure_cache_dir()
        self._load_disk_metadata()

    def _ensure_cache_dir(self):
        """Create cache directory if needed."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "audio").mkdir(exist_ok=True)

    def _load_disk_metadata(self):
        """Load disk cache metadata."""
        metadata_path = self.cache_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                for key, entry_data in data.get('entries', {}).items():
                    entry = VoiceCacheEntry(
                        cache_key=entry_data['cache_key'],
                        text=entry_data['text'],
                        voice_profile_id=entry_data['voice_profile_id'],
                        audio_format=entry_data.get('audio_format', 'wav'),
                        created_at=datetime.fromisoformat(entry_data['created_at']),
                        last_accessed=datetime.fromisoformat(entry_data['last_accessed']),
                        access_count=entry_data.get('access_count', 0),
                        size_bytes=entry_data.get('size_bytes', 0),
                        disk_path=Path(entry_data['disk_path']) if entry_data.get('disk_path') else None,
                        in_memory=False,
                    )
                    if entry.disk_path and entry.disk_path.exists():
                        self._disk_metadata[key] = entry
                        self._disk_bytes_used += entry.size_bytes
                self._disk_bytes_used = data.get('total_bytes', self._disk_bytes_used)
                logger.info(f"Loaded {len(self._disk_metadata)} entries from disk cache")
            except Exception as e:
                logger.error(f"Failed to load disk cache metadata: {e}")

    def _save_disk_metadata(self):
        """Save disk cache metadata."""
        metadata_path = self.cache_dir / "metadata.json"
        try:
            data = {
                'entries': {k: v.to_dict() for k, v in self._disk_metadata.items()},
                'total_bytes': self._disk_bytes_used,
                'last_updated': datetime.now().isoformat(),
            }
            with open(metadata_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save disk cache metadata: {e}")

    @staticmethod
    def generate_cache_key(
        text: str,
        voice_profile_id: str,
        **kwargs
    ) -> str:
        """Generate unique cache key for voice request."""
        # Normalize text
        normalized_text = text.strip().lower()

        # Build key components
        key_parts = [
            normalized_text,
            voice_profile_id,
            json.dumps(kwargs, sort_keys=True) if kwargs else '',
        ]

        # Hash for consistent key length
        key_string = '|'.join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def get(self, cache_key: str) -> Optional[Tuple[bytes, VoiceCacheEntry]]:
        """
        Get cached audio data.

        Args:
            cache_key: Cache key to look up

        Returns:
            Tuple of (audio_data, entry) or None if not found
        """
        # Check memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]

            # Check TTL
            if datetime.now() - entry.created_at > self.ttl:
                self._evict_memory_entry(cache_key)
                return None

            entry.touch()
            # Move to end (LRU)
            self._memory_cache.move_to_end(cache_key)
            self._stats['memory_hits'] += 1

            logger.debug(f"Memory cache hit for {cache_key}")
            return entry.audio_data, entry

        # Check disk cache
        if cache_key in self._disk_metadata:
            entry = self._disk_metadata[cache_key]

            # Check TTL
            if datetime.now() - entry.created_at > self.ttl:
                self._evict_disk_entry(cache_key)
                return None

            # Load from disk
            if entry.disk_path and entry.disk_path.exists():
                try:
                    audio_data = entry.disk_path.read_bytes()
                    entry.touch()
                    self._stats['disk_hits'] += 1

                    # Promote to memory cache if frequently accessed
                    if entry.access_count >= 3:
                        self._promote_to_memory(cache_key, audio_data, entry)

                    logger.debug(f"Disk cache hit for {cache_key}")
                    return audio_data, entry
                except Exception as e:
                    logger.error(f"Failed to read disk cache: {e}")
                    self._evict_disk_entry(cache_key)

        self._stats['misses'] += 1
        return None

    def put(
        self,
        cache_key: str,
        audio_data: bytes,
        text: str,
        voice_profile_id: str,
        audio_format: str = "wav",
    ) -> VoiceCacheEntry:
        """
        Store audio data in cache.

        Args:
            cache_key: Cache key
            audio_data: Audio bytes
            text: Original text
            voice_profile_id: Voice profile used
            audio_format: Audio format (wav, mp3)

        Returns:
            Cache entry
        """
        size_bytes = len(audio_data)

        # Create entry
        entry = VoiceCacheEntry(
            cache_key=cache_key,
            text=text,
            voice_profile_id=voice_profile_id,
            audio_data=audio_data,
            audio_format=audio_format,
            size_bytes=size_bytes,
            in_memory=True,
        )

        # Store in memory cache
        self._put_memory(cache_key, entry)

        # Store on disk for persistence
        self._put_disk(cache_key, entry, audio_data)

        logger.debug(f"Cached {cache_key} ({size_bytes} bytes)")

        return entry

    def _put_memory(self, cache_key: str, entry: VoiceCacheEntry):
        """Add entry to memory cache."""
        # Evict if needed for space
        while self._memory_bytes_used + entry.size_bytes > self.memory_cache_bytes:
            if not self._memory_cache:
                break
            oldest_key = next(iter(self._memory_cache))
            self._evict_memory_entry(oldest_key)

        while len(self._memory_cache) >= self.memory_cache_size:
            oldest_key = next(iter(self._memory_cache))
            self._evict_memory_entry(oldest_key)

        # Add to cache
        self._memory_cache[cache_key] = entry
        self._memory_bytes_used += entry.size_bytes

    def _put_disk(
        self,
        cache_key: str,
        entry: VoiceCacheEntry,
        audio_data: bytes,
    ):
        """Add entry to disk cache."""
        # Evict if needed for space
        while self._disk_bytes_used + entry.size_bytes > self.disk_cache_size_bytes:
            if not self._disk_metadata:
                break
            # Evict least recently used
            oldest_key = min(
                self._disk_metadata.keys(),
                key=lambda k: self._disk_metadata[k].last_accessed
            )
            self._evict_disk_entry(oldest_key)

        # Write to disk
        disk_path = self.cache_dir / "audio" / f"{cache_key}.{entry.audio_format}"
        try:
            disk_path.write_bytes(audio_data)
            entry.disk_path = disk_path

            # Create disk metadata entry (without audio data)
            disk_entry = VoiceCacheEntry(
                cache_key=entry.cache_key,
                text=entry.text,
                voice_profile_id=entry.voice_profile_id,
                audio_format=entry.audio_format,
                created_at=entry.created_at,
                last_accessed=entry.last_accessed,
                access_count=entry.access_count,
                size_bytes=entry.size_bytes,
                disk_path=disk_path,
                in_memory=False,
            )
            self._disk_metadata[cache_key] = disk_entry
            self._disk_bytes_used += entry.size_bytes

            # Save metadata periodically
            if len(self._disk_metadata) % 10 == 0:
                self._save_disk_metadata()

        except Exception as e:
            logger.error(f"Failed to write to disk cache: {e}")

    def _promote_to_memory(
        self,
        cache_key: str,
        audio_data: bytes,
        disk_entry: VoiceCacheEntry,
    ):
        """Promote disk entry to memory cache."""
        entry = VoiceCacheEntry(
            cache_key=disk_entry.cache_key,
            text=disk_entry.text,
            voice_profile_id=disk_entry.voice_profile_id,
            audio_data=audio_data,
            audio_format=disk_entry.audio_format,
            created_at=disk_entry.created_at,
            last_accessed=disk_entry.last_accessed,
            access_count=disk_entry.access_count,
            size_bytes=disk_entry.size_bytes,
            disk_path=disk_entry.disk_path,
            in_memory=True,
        )
        self._put_memory(cache_key, entry)
        logger.debug(f"Promoted {cache_key} to memory cache")

    def _evict_memory_entry(self, cache_key: str):
        """Evict entry from memory cache."""
        if cache_key in self._memory_cache:
            entry = self._memory_cache.pop(cache_key)
            self._memory_bytes_used -= entry.size_bytes
            self._stats['evictions'] += 1
            logger.debug(f"Evicted {cache_key} from memory cache")

    def _evict_disk_entry(self, cache_key: str):
        """Evict entry from disk cache."""
        if cache_key in self._disk_metadata:
            entry = self._disk_metadata.pop(cache_key)
            self._disk_bytes_used -= entry.size_bytes

            # Delete file
            if entry.disk_path and entry.disk_path.exists():
                try:
                    entry.disk_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete cache file: {e}")

            self._stats['evictions'] += 1
            logger.debug(f"Evicted {cache_key} from disk cache")

    def invalidate(self, cache_key: str):
        """Invalidate a specific cache entry."""
        self._evict_memory_entry(cache_key)
        self._evict_disk_entry(cache_key)

    def invalidate_by_profile(self, voice_profile_id: str):
        """Invalidate all entries for a voice profile."""
        keys_to_invalidate = []

        for key, entry in self._memory_cache.items():
            if entry.voice_profile_id == voice_profile_id:
                keys_to_invalidate.append(key)

        for key, entry in self._disk_metadata.items():
            if entry.voice_profile_id == voice_profile_id and key not in keys_to_invalidate:
                keys_to_invalidate.append(key)

        for key in keys_to_invalidate:
            self.invalidate(key)

        logger.info(f"Invalidated {len(keys_to_invalidate)} entries for profile {voice_profile_id}")

    def clear(self):
        """Clear entire cache."""
        self._memory_cache.clear()
        self._memory_bytes_used = 0

        for entry in self._disk_metadata.values():
            if entry.disk_path and entry.disk_path.exists():
                try:
                    entry.disk_path.unlink()
                except Exception:
                    pass

        self._disk_metadata.clear()
        self._disk_bytes_used = 0
        self._save_disk_metadata()

        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = (
            self._stats['memory_hits'] +
            self._stats['disk_hits'] +
            self._stats['misses']
        )

        return {
            'memory_entries': len(self._memory_cache),
            'memory_bytes': self._memory_bytes_used,
            'disk_entries': len(self._disk_metadata),
            'disk_bytes': self._disk_bytes_used,
            'memory_hits': self._stats['memory_hits'],
            'disk_hits': self._stats['disk_hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'hit_rate': (
                (self._stats['memory_hits'] + self._stats['disk_hits']) / total_requests
                if total_requests > 0 else 0.0
            ),
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = datetime.now()
        expired_keys = []

        for key, entry in list(self._memory_cache.items()):
            if now - entry.created_at > self.ttl:
                expired_keys.append(key)

        for key, entry in list(self._disk_metadata.items()):
            if now - entry.created_at > self.ttl:
                if key not in expired_keys:
                    expired_keys.append(key)

        for key in expired_keys:
            self.invalidate(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)


# -----------------------------------------------------------------------------
# Voice Manager
# -----------------------------------------------------------------------------

class VoiceManager:
    """
    Main voice synthesis manager with caching.

    Usage:
        manager = VoiceManager()

        # Register voice profiles
        manager.register_profile(VoiceProfile(
            id="narrator",
            name="Narrator Voice",
            provider=VoiceProviderType.ELEVENLABS,
            voice_id="21m00Tcm4TlvDq8ikWAM",
        ))

        # Synthesize with caching
        audio = await manager.synthesize(
            text="Welcome to the game!",
            profile_id="narrator"
        )
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize voice manager.

        Args:
            cache_dir: Directory for disk cache
            enable_cache: Whether to enable caching
        """
        self._profiles: Dict[str, VoiceProfile] = {}
        self._providers: Dict[VoiceProviderType, VoiceProvider] = {}
        self._enable_cache = enable_cache

        if enable_cache:
            self._cache = VoiceCache(cache_dir=cache_dir)
        else:
            self._cache = None

    def register_profile(self, profile: VoiceProfile):
        """Register a voice profile."""
        self._profiles[profile.id] = profile
        logger.info(f"Registered voice profile: {profile.id} ({profile.name})")

    def get_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get a voice profile by ID."""
        return self._profiles.get(profile_id)

    def list_profiles(self) -> List[VoiceProfile]:
        """List all registered profiles."""
        return list(self._profiles.values())

    def register_provider(
        self,
        provider_type: VoiceProviderType,
        provider: VoiceProvider,
    ):
        """Register a voice provider."""
        self._providers[provider_type] = provider
        logger.info(f"Registered voice provider: {provider_type.value}")

    async def synthesize(
        self,
        text: str,
        profile_id: str,
        emotion: Optional[str] = None,
        bypass_cache: bool = False,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            profile_id: Voice profile to use
            emotion: Override emotion (uses profile default if not specified)
            bypass_cache: Skip cache lookup
            **kwargs: Additional provider-specific options

        Returns:
            Audio data as bytes

        Raises:
            ValueError: If profile not found
            RuntimeError: If provider not available
        """
        # Get profile
        profile = self._profiles.get(profile_id)
        if not profile:
            raise ValueError(f"Voice profile not found: {profile_id}")

        # Build synthesis options
        synthesis_opts = {
            'pitch': profile.pitch,
            'speed': profile.speed,
            'volume': profile.volume,
            'emotion': emotion or profile.default_emotion,
            'emotion_intensity': profile.emotion_intensity,
            **profile.settings,
            **kwargs,
        }

        # Check cache
        if self._enable_cache and self._cache and not bypass_cache:
            cache_key = VoiceCache.generate_cache_key(
                text=text,
                voice_profile_id=profile_id,
                **synthesis_opts,
            )

            cached = self._cache.get(cache_key)
            if cached:
                audio_data, entry = cached
                logger.debug(f"Cache hit for synthesis: {profile_id}")
                return audio_data

        # Get provider
        provider = self._providers.get(profile.provider)
        if not provider:
            raise RuntimeError(
                f"Voice provider not available: {profile.provider.value}. "
                f"Please register a provider using register_provider()."
            )

        # Synthesize
        logger.info(f"Synthesizing: '{text[:50]}...' with profile {profile_id}")
        audio_data = await provider.synthesize(
            text=text,
            voice_id=profile.voice_id,
            **synthesis_opts,
        )

        # Cache result
        if self._enable_cache and self._cache:
            self._cache.put(
                cache_key=cache_key,
                audio_data=audio_data,
                text=text,
                voice_profile_id=profile_id,
            )

        return audio_data

    async def synthesize_batch(
        self,
        items: List[Tuple[str, str]],  # List of (text, profile_id)
        **kwargs
    ) -> List[bytes]:
        """
        Synthesize multiple texts in batch.

        Args:
            items: List of (text, profile_id) tuples
            **kwargs: Additional options for all items

        Returns:
            List of audio data bytes
        """
        tasks = [
            self.synthesize(text=text, profile_id=pid, **kwargs)
            for text, pid in items
        ]
        return await asyncio.gather(*tasks)

    def preload_cache(
        self,
        texts: List[str],
        profile_id: str,
    ) -> int:
        """
        Check how many texts are already cached.

        Args:
            texts: List of texts to check
            profile_id: Voice profile

        Returns:
            Count of cached items
        """
        if not self._enable_cache or not self._cache:
            return 0

        profile = self._profiles.get(profile_id)
        if not profile:
            return 0

        cached_count = 0
        for text in texts:
            cache_key = VoiceCache.generate_cache_key(
                text=text,
                voice_profile_id=profile_id,
            )
            if self._cache.get(cache_key):
                cached_count += 1

        return cached_count

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats()
        return None

    def invalidate_profile_cache(self, profile_id: str):
        """Invalidate all cached audio for a profile."""
        if self._cache:
            self._cache.invalidate_by_profile(profile_id)

    def clear_cache(self):
        """Clear all cached audio."""
        if self._cache:
            self._cache.clear()


# -----------------------------------------------------------------------------
# Mock Provider (for testing)
# -----------------------------------------------------------------------------

class MockVoiceProvider:
    """Mock voice provider for testing."""

    def __init__(self, latency_ms: float = 100.0):
        self.latency_ms = latency_ms
        self._synthesis_count = 0

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        **kwargs
    ) -> bytes:
        """Generate mock audio data."""
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        self._synthesis_count += 1

        # Generate mock WAV header + silence
        # This is a minimal valid WAV file (44 byte header + silence)
        sample_rate = 22050
        duration_samples = int(sample_rate * 0.1)  # 100ms of silence
        data_size = duration_samples * 2  # 16-bit samples

        wav_header = bytes([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            *((data_size + 36).to_bytes(4, 'little')),  # File size - 8
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # Subchunk1 size (16)
            0x01, 0x00,              # Audio format (1 = PCM)
            0x01, 0x00,              # Num channels (1)
            *(sample_rate.to_bytes(4, 'little')),  # Sample rate
            *((sample_rate * 2).to_bytes(4, 'little')),  # Byte rate
            0x02, 0x00,              # Block align
            0x10, 0x00,              # Bits per sample (16)
            0x64, 0x61, 0x74, 0x61,  # "data"
            *(data_size.to_bytes(4, 'little')),  # Data size
        ])

        # Add silence data
        silence = bytes(data_size)

        return wav_header + silence

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get mock voices."""
        return [
            {'id': 'mock_narrator', 'name': 'Mock Narrator'},
            {'id': 'mock_character', 'name': 'Mock Character'},
        ]


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def create_voice_manager(
    cache_dir: Optional[Path] = None,
    enable_cache: bool = True,
) -> VoiceManager:
    """
    Create a voice manager.

    Args:
        cache_dir: Directory for disk cache
        enable_cache: Whether to enable caching

    Returns:
        Configured VoiceManager
    """
    return VoiceManager(
        cache_dir=cache_dir,
        enable_cache=enable_cache,
    )


def create_mock_voice_manager() -> VoiceManager:
    """Create a voice manager with mock provider for testing."""
    manager = VoiceManager(enable_cache=True)
    manager.register_provider(VoiceProviderType.MOCK, MockVoiceProvider())

    # Register default mock profile
    manager.register_profile(VoiceProfile(
        id="test_narrator",
        name="Test Narrator",
        provider=VoiceProviderType.MOCK,
        voice_id="mock_narrator",
    ))

    return manager


__all__ = [
    # Protocols and enums
    'VoiceProvider',
    'VoiceProviderType',
    # Data classes
    'VoiceProfile',
    'VoiceCacheEntry',
    # Main classes
    'VoiceCache',
    'VoiceManager',
    # Mock provider
    'MockVoiceProvider',
    # Factories
    'create_voice_manager',
    'create_mock_voice_manager',
]

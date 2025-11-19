"""
Tests for voice synthesis engine.

Tests cover:
- Voice profile creation and validation
- Voice engine initialization
- Dummy backend synthesis
- Voice caching functionality
- Error handling
- Voice profile registry
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from apps.elle_game_engine.voice import (
    VoiceEngine,
    VoiceProfile,
    VoiceSynthesisRequest,
    VoiceSynthesisResult,
    VoiceCache,
    Emotion,
    AudioFormat,
    TTSBackend,
    DummyTTSBackend,
    create_voice_engine,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_voice_profile():
    """Create a dummy voice profile for testing."""
    return VoiceProfile(
        voice_id="test_voice",
        pitch=1.0,
        speed=1.0,
        emotion=Emotion.NEUTRAL,
        stability=0.5,
        similarity_boost=0.75
    )


@pytest.fixture
def voice_engine(temp_cache_dir):
    """Create voice engine with dummy backend."""
    return VoiceEngine(
        backend=TTSBackend.DUMMY,
        enable_cache=True,
        cache_dir=temp_cache_dir,
        cache_size_mb=10
    )


# ============================================================================
# Voice Profile Tests
# ============================================================================

def test_voice_profile_creation():
    """Test voice profile creation with valid parameters."""
    profile = VoiceProfile(
        voice_id="bob",
        pitch=1.2,
        speed=0.9,
        emotion=Emotion.WARM,
        stability=0.6,
        similarity_boost=0.8
    )

    assert profile.voice_id == "bob"
    assert profile.pitch == 1.2
    assert profile.speed == 0.9
    assert profile.emotion == Emotion.WARM
    assert profile.stability == 0.6
    assert profile.similarity_boost == 0.8


def test_voice_profile_invalid_pitch():
    """Test voice profile validation - invalid pitch."""
    with pytest.raises(ValueError, match="pitch must be between"):
        VoiceProfile(
            voice_id="test",
            pitch=3.0,  # Too high
        )

    with pytest.raises(ValueError, match="pitch must be between"):
        VoiceProfile(
            voice_id="test",
            pitch=0.2,  # Too low
        )


def test_voice_profile_invalid_speed():
    """Test voice profile validation - invalid speed."""
    with pytest.raises(ValueError, match="speed must be between"):
        VoiceProfile(
            voice_id="test",
            speed=3.0,  # Too high
        )


def test_voice_profile_invalid_stability():
    """Test voice profile validation - invalid stability."""
    with pytest.raises(ValueError, match="stability must be between"):
        VoiceProfile(
            voice_id="test",
            stability=1.5,  # Too high
        )


# ============================================================================
# Voice Engine Tests
# ============================================================================

def test_voice_engine_initialization():
    """Test voice engine initialization."""
    engine = VoiceEngine(
        backend=TTSBackend.DUMMY,
        enable_cache=False
    )

    assert engine.backend_type == TTSBackend.DUMMY
    assert engine.cache is None


def test_voice_engine_with_cache(temp_cache_dir):
    """Test voice engine with caching enabled."""
    engine = VoiceEngine(
        backend=TTSBackend.DUMMY,
        enable_cache=True,
        cache_dir=temp_cache_dir
    )

    assert engine.cache is not None
    assert Path(temp_cache_dir).exists()


def test_register_voice_profile(voice_engine, dummy_voice_profile):
    """Test voice profile registration."""
    voice_engine.register_voice_profile("bob", dummy_voice_profile)

    profile = voice_engine.get_voice_profile("bob")
    assert profile == dummy_voice_profile


def test_get_nonexistent_voice_profile(voice_engine):
    """Test getting nonexistent voice profile."""
    profile = voice_engine.get_voice_profile("nonexistent")
    assert profile is None


# ============================================================================
# Dummy Backend Tests
# ============================================================================

def test_dummy_backend_synthesis():
    """Test dummy backend generates silent audio."""
    backend = DummyTTSBackend()

    request = VoiceSynthesisRequest(
        text="Hello, world!",
        voice_profile=VoiceProfile(voice_id="test"),
        format=AudioFormat.WAV
    )

    result = backend.synthesize(request)

    assert result.audio_data is not None
    assert len(result.audio_data) > 0
    assert result.format == AudioFormat.WAV
    assert result.duration_seconds > 0
    assert result.text == "Hello, world!"
    assert result.cached == False


def test_voice_engine_synthesize_with_profile(voice_engine, dummy_voice_profile):
    """Test voice synthesis with voice profile."""
    voice_engine.register_voice_profile("bob", dummy_voice_profile)

    result = voice_engine.synthesize(
        text="Hello!",
        npc_id="bob"
    )

    assert result is not None
    assert result.text == "Hello!"
    # Dummy backend returns WAV format
    assert result.format in [AudioFormat.MP3, AudioFormat.WAV]


def test_voice_engine_synthesize_direct_profile(voice_engine):
    """Test voice synthesis with direct voice profile."""
    profile = VoiceProfile(voice_id="direct")

    result = voice_engine.synthesize(
        text="Direct synthesis",
        voice_profile=profile
    )

    assert result is not None
    assert result.text == "Direct synthesis"


def test_voice_engine_synthesize_no_profile_error(voice_engine):
    """Test voice synthesis without profile raises error."""
    with pytest.raises(ValueError, match="Either npc_id or voice_profile must be provided"):
        voice_engine.synthesize(text="No profile!")


def test_voice_engine_synthesize_unknown_npc_error(voice_engine):
    """Test voice synthesis with unknown NPC raises error."""
    with pytest.raises(ValueError, match="No voice profile registered for NPC"):
        voice_engine.synthesize(text="Unknown", npc_id="unknown_npc")


# ============================================================================
# Voice Cache Tests
# ============================================================================

def test_voice_cache_initialization(temp_cache_dir):
    """Test voice cache initialization."""
    cache = VoiceCache(cache_dir=temp_cache_dir, max_size_mb=10)

    assert cache.cache_dir == Path(temp_cache_dir)
    assert cache.max_size_bytes == 10 * 1024 * 1024


def test_voice_cache_set_and_get(temp_cache_dir):
    """Test caching and retrieving audio."""
    cache = VoiceCache(cache_dir=temp_cache_dir)
    profile = VoiceProfile(voice_id="test")
    audio_data = b"fake_audio_data"

    # Cache audio
    cache.set("Hello!", profile, Emotion.NEUTRAL, AudioFormat.MP3, audio_data)

    # Retrieve cached audio
    cached = cache.get("Hello!", profile, Emotion.NEUTRAL, AudioFormat.MP3)

    assert cached == audio_data


def test_voice_cache_miss(temp_cache_dir):
    """Test cache miss returns None."""
    cache = VoiceCache(cache_dir=temp_cache_dir)
    profile = VoiceProfile(voice_id="test")

    # Try to get non-cached audio
    cached = cache.get("Not cached", profile, Emotion.NEUTRAL, AudioFormat.MP3)

    assert cached is None


def test_voice_cache_different_parameters(temp_cache_dir):
    """Test cache differentiates by parameters."""
    cache = VoiceCache(cache_dir=temp_cache_dir)
    profile1 = VoiceProfile(voice_id="voice1")
    profile2 = VoiceProfile(voice_id="voice2")

    # Cache with profile1
    cache.set("Hello!", profile1, Emotion.NEUTRAL, AudioFormat.MP3, b"audio1")

    # Try to get with profile2
    cached = cache.get("Hello!", profile2, Emotion.NEUTRAL, AudioFormat.MP3)

    assert cached is None  # Different profile, cache miss


def test_voice_cache_stats(temp_cache_dir):
    """Test cache statistics."""
    cache = VoiceCache(cache_dir=temp_cache_dir, max_size_mb=10)
    profile = VoiceProfile(voice_id="test")

    # Add some entries
    cache.set("Text 1", profile, Emotion.NEUTRAL, AudioFormat.MP3, b"audio1")
    cache.set("Text 2", profile, Emotion.WARM, AudioFormat.MP3, b"audio2")

    stats = cache.get_stats()

    assert stats["num_entries"] == 2
    assert stats["total_size_bytes"] > 0
    assert stats["cache_dir"] == str(Path(temp_cache_dir))


def test_voice_cache_clear(temp_cache_dir):
    """Test clearing voice cache."""
    cache = VoiceCache(cache_dir=temp_cache_dir)
    profile = VoiceProfile(voice_id="test")

    # Add entries
    cache.set("Text 1", profile, Emotion.NEUTRAL, AudioFormat.MP3, b"audio1")
    cache.set("Text 2", profile, Emotion.WARM, AudioFormat.MP3, b"audio2")

    # Clear cache
    cache.clear()

    # Verify cache is empty
    stats = cache.get_stats()
    assert stats["num_entries"] == 0


def test_voice_engine_caching(voice_engine, dummy_voice_profile):
    """Test voice engine uses cache."""
    voice_engine.register_voice_profile("bob", dummy_voice_profile)

    # First synthesis (cold cache)
    result1 = voice_engine.synthesize(text="Hello!", npc_id="bob")
    assert result1.cached == False

    # Second synthesis (should hit cache)
    result2 = voice_engine.synthesize(text="Hello!", npc_id="bob")
    assert result2.cached == True

    # Same audio data
    assert result1.audio_data == result2.audio_data


def test_voice_engine_cache_stats(voice_engine, dummy_voice_profile):
    """Test getting cache statistics from voice engine."""
    voice_engine.register_voice_profile("bob", dummy_voice_profile)

    # Synthesize some audio
    voice_engine.synthesize(text="Hello!", npc_id="bob")
    voice_engine.synthesize(text="Goodbye!", npc_id="bob")

    stats = voice_engine.get_cache_stats()

    assert stats["num_entries"] == 2
    assert stats["total_size_bytes"] > 0


def test_voice_engine_clear_cache(voice_engine, dummy_voice_profile):
    """Test clearing cache from voice engine."""
    voice_engine.register_voice_profile("bob", dummy_voice_profile)

    # Synthesize some audio
    voice_engine.synthesize(text="Hello!", npc_id="bob")

    # Clear cache
    voice_engine.clear_cache()

    # Verify cache is empty
    stats = voice_engine.get_cache_stats()
    assert stats["num_entries"] == 0


# ============================================================================
# Factory Function Tests
# ============================================================================

def test_create_voice_engine_default():
    """Test creating voice engine with defaults."""
    engine = create_voice_engine()

    assert engine.backend_type == TTSBackend.DUMMY
    assert engine.cache is not None


def test_create_voice_engine_no_cache():
    """Test creating voice engine without cache."""
    engine = create_voice_engine(enable_cache=False)

    assert engine.cache is None


def test_create_voice_engine_custom_backend():
    """Test creating voice engine with custom backend."""
    engine = create_voice_engine(backend="dummy")

    assert engine.backend_type == TTSBackend.DUMMY


# ============================================================================
# Emotion Integration Tests
# ============================================================================

def test_synthesize_with_emotion_override(voice_engine, dummy_voice_profile):
    """Test synthesis with emotion override."""
    voice_engine.register_voice_profile("bob", dummy_voice_profile)

    # Base emotion is NEUTRAL
    result = voice_engine.synthesize(
        text="Hello!",
        npc_id="bob",
        emotion=Emotion.EXCITED  # Override
    )

    assert result is not None


def test_synthesize_different_emotions_cached_separately(voice_engine, dummy_voice_profile):
    """Test that different emotions are cached separately."""
    voice_engine.register_voice_profile("bob", dummy_voice_profile)

    # Synthesize with neutral emotion
    result1 = voice_engine.synthesize(
        text="Hello!",
        npc_id="bob",
        emotion=Emotion.NEUTRAL
    )

    # Synthesize same text with excited emotion
    result2 = voice_engine.synthesize(
        text="Hello!",
        npc_id="bob",
        emotion=Emotion.EXCITED
    )

    # Both should be cache misses (different emotions)
    assert result1.cached == False
    assert result2.cached == False

    # Third synthesis with neutral should hit cache
    result3 = voice_engine.synthesize(
        text="Hello!",
        npc_id="bob",
        emotion=Emotion.NEUTRAL
    )
    assert result3.cached == True

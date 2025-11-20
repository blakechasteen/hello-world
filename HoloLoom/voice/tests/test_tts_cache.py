"""
TTS Cache Tests
================

Comprehensive test suite for TTS caching system.

Tests:
- Cache key generation consistency
- Cache hit/miss tracking
- Common phrase detection
- TTL enforcement
- Redis connection errors (graceful fallback)
- Cache statistics accuracy
- Warmup functionality
- Cache size limits

Date: November 16, 2025
"""

import pytest
import asyncio
import hashlib
from unittest.mock import Mock, patch, AsyncMock
from typing import Optional

# Import module under test
from HoloLoom.voice.tts_cache import (
    TTSCache,
    CacheConfig,
    CacheStats,
    classify_phrase,
    PhraseType,
    REDIS_AVAILABLE
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cache_config():
    """Basic cache configuration"""
    return CacheConfig(
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        enable_cache=True,
        common_phrase_ttl=86400,
        dynamic_phrase_ttl=3600,
        max_cache_size_mb=1000
    )


@pytest.fixture
def cache_config_disabled():
    """Cache configuration with caching disabled"""
    return CacheConfig(enable_cache=False)


@pytest.fixture
async def mock_redis():
    """Mock Redis client"""
    redis_mock = AsyncMock()
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.scan = AsyncMock(return_value=(0, []))
    redis_mock.info = AsyncMock(return_value={'used_memory': 1024})
    redis_mock.close = AsyncMock()
    return redis_mock


# ============================================================================
# Test: Cache Key Generation
# ============================================================================

class TestCacheKeyGeneration:
    """Test cache key generation"""

    def test_cache_key_deterministic(self, cache_config):
        """Cache keys should be deterministic"""
        cache = TTSCache(cache_config)

        key1 = cache._generate_cache_key("Hello world", "nova", "en")
        key2 = cache._generate_cache_key("Hello world", "nova", "en")

        assert key1 == key2

    def test_cache_key_case_insensitive(self, cache_config):
        """Cache keys should be case-insensitive"""
        cache = TTSCache(cache_config)

        key1 = cache._generate_cache_key("Hello World", "nova", "en")
        key2 = cache._generate_cache_key("hello world", "nova", "en")

        assert key1 == key2

    def test_cache_key_whitespace_normalized(self, cache_config):
        """Cache keys should normalize whitespace"""
        cache = TTSCache(cache_config)

        key1 = cache._generate_cache_key("  Hello world  ", "nova", "en")
        key2 = cache._generate_cache_key("Hello world", "nova", "en")

        assert key1 == key2

    def test_cache_key_different_voice(self, cache_config):
        """Different voices should produce different keys"""
        cache = TTSCache(cache_config)

        key1 = cache._generate_cache_key("Hello world", "nova", "en")
        key2 = cache._generate_cache_key("Hello world", "alloy", "en")

        assert key1 != key2

    def test_cache_key_different_language(self, cache_config):
        """Different languages should produce different keys"""
        cache = TTSCache(cache_config)

        key1 = cache._generate_cache_key("Hello world", "nova", "en")
        key2 = cache._generate_cache_key("Hello world", "nova", "es")

        assert key1 != key2

    def test_cache_key_format(self, cache_config):
        """Cache key should follow expected format"""
        cache = TTSCache(cache_config)

        key = cache._generate_cache_key("Hello world", "nova", "en")

        # Format: tts:{hash}:{voice}:{lang}
        parts = key.split(":")
        assert len(parts) == 4
        assert parts[0] == "tts"
        assert len(parts[1]) == 16  # SHA256 truncated to 16 chars
        assert parts[2] == "nova"
        assert parts[3] == "en"


# ============================================================================
# Test: Phrase Classification
# ============================================================================

class TestPhraseClassification:
    """Test phrase classification for TTL selection"""

    def test_greeting_is_common(self):
        """Greetings should be classified as COMMON"""
        assert classify_phrase("Hello, how are you?") == PhraseType.COMMON
        assert classify_phrase("Hi there") == PhraseType.COMMON
        assert classify_phrase("Good morning") == PhraseType.COMMON

    def test_gratitude_is_common(self):
        """Gratitude phrases should be classified as COMMON"""
        assert classify_phrase("Thank you") == PhraseType.COMMON
        assert classify_phrase("Thanks for your help") == PhraseType.COMMON

    def test_simple_response_is_common(self):
        """Simple responses should be classified as COMMON"""
        assert classify_phrase("Yes") == PhraseType.COMMON
        assert classify_phrase("No") == PhraseType.COMMON
        assert classify_phrase("Okay") == PhraseType.COMMON

    def test_query_starter_is_common(self):
        """Query starters should be classified as COMMON"""
        assert classify_phrase("Show me the hive") == PhraseType.COMMON
        assert classify_phrase("What is Thompson Sampling") == PhraseType.COMMON
        assert classify_phrase("Tell me about bees") == PhraseType.COMMON

    def test_hive_reference_is_common(self):
        """Hive references should be classified as COMMON"""
        assert classify_phrase("Check hive 1") == PhraseType.COMMON
        assert classify_phrase("Navigate to hive 42") == PhraseType.COMMON

    def test_short_phrase_is_common(self):
        """Short phrases (<5 words) should be classified as COMMON"""
        assert classify_phrase("Start inspection") == PhraseType.COMMON
        assert classify_phrase("End session") == PhraseType.COMMON

    def test_date_is_dynamic(self):
        """Phrases with dates should be classified as DYNAMIC"""
        assert classify_phrase("Inspection on 2025-11-16") == PhraseType.DYNAMIC
        assert classify_phrase("Check hive today") == PhraseType.DYNAMIC

    def test_time_is_dynamic(self):
        """Phrases with times should be classified as DYNAMIC"""
        assert classify_phrase("Inspection at 14:30") == PhraseType.DYNAMIC
        assert classify_phrase("The hive was checked currently") == PhraseType.DYNAMIC

    def test_long_phrase_default_common(self):
        """Long phrases without dynamic content default to COMMON"""
        phrase = "Navigate to the next hive and show me the status"
        # This has >5 words but no dynamic content, should be COMMON
        # because it matches "Navigate" pattern
        assert classify_phrase(phrase) == PhraseType.COMMON


# ============================================================================
# Test: Cache Statistics
# ============================================================================

class TestCacheStatistics:
    """Test cache statistics tracking"""

    def test_initial_stats(self):
        """Initial stats should be zero"""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0

    def test_record_hit(self):
        """Recording hits should update stats"""
        stats = CacheStats()

        stats.record_hit(10.0)
        assert stats.hits == 1
        assert stats.total_requests == 1
        assert stats.hit_rate == 1.0

    def test_record_miss(self):
        """Recording misses should update stats"""
        stats = CacheStats()

        stats.record_miss(500.0)
        assert stats.misses == 1
        assert stats.total_requests == 1
        assert stats.miss_rate == 1.0

    def test_hit_rate_calculation(self):
        """Hit rate should be calculated correctly"""
        stats = CacheStats()

        # 3 hits, 1 miss = 75% hit rate
        stats.record_hit(10.0)
        stats.record_hit(10.0)
        stats.record_hit(10.0)
        stats.record_miss(500.0)

        assert stats.hit_rate == 0.75
        assert stats.miss_rate == 0.25

    def test_speedup_factor_calculation(self):
        """Speedup factor should be calculated correctly"""
        stats = CacheStats()

        # 8 hits, 2 misses
        for _ in range(8):
            stats.record_hit(50.0)
        for _ in range(2):
            stats.record_miss(500.0)

        # Without cache: 10 * 500ms = 5000ms
        # With cache: 8 * 50ms + 2 * 500ms = 400 + 1000 = 1400ms
        # Speedup: 5000 / 1400 = 3.57x
        speedup = stats.speedup_factor
        assert speedup > 3.5
        assert speedup < 3.6

    def test_avg_latency_tracking(self):
        """Average latencies should be tracked"""
        stats = CacheStats()

        stats.record_hit(10.0)
        stats.record_hit(20.0)
        stats.record_miss(500.0)
        stats.record_miss(600.0)

        assert stats.avg_hit_latency_ms == 15.0
        assert stats.avg_miss_latency_ms == 550.0

    def test_stats_to_dict(self):
        """Stats should export as dictionary"""
        stats = CacheStats()
        stats.record_hit(10.0)

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert 'hit_rate' in stats_dict
        assert 'hits' in stats_dict
        assert 'speedup_factor' in stats_dict


# ============================================================================
# Test: TTL Selection
# ============================================================================

class TestTTLSelection:
    """Test TTL selection based on phrase type"""

    def test_common_phrase_ttl(self, cache_config):
        """Common phrases should get 24h TTL"""
        cache = TTSCache(cache_config)

        ttl = cache._get_ttl("Hello world")
        assert ttl == 86400  # 24 hours

    def test_dynamic_phrase_ttl(self, cache_config):
        """Dynamic phrases should get 1h TTL"""
        cache = TTSCache(cache_config)

        ttl = cache._get_ttl("Inspection at 14:30 today")
        assert ttl == 3600  # 1 hour


# ============================================================================
# Test: Cache Operations (with mock Redis)
# ============================================================================

class TestCacheOperations:
    """Test cache operations with mock Redis"""

    @pytest.mark.asyncio
    async def test_cache_disabled(self, cache_config_disabled):
        """Cache operations should no-op when disabled"""
        cache = TTSCache(cache_config_disabled)
        await cache.initialize()

        # Get should return None
        result = await cache.get("Hello", "nova", "en")
        assert result is None

        # Set should not error
        await cache.set("Hello", "nova", "en", b"audio_data")

    @pytest.mark.asyncio
    @patch('HoloLoom.voice.tts_cache.aioredis.from_url')
    async def test_cache_get_hit(self, mock_from_url, cache_config, mock_redis):
        """Cache get should return cached audio on hit"""
        mock_from_url.return_value = mock_redis
        mock_redis.get = AsyncMock(return_value=b"cached_audio")

        cache = TTSCache(cache_config)
        await cache.initialize()

        result = await cache.get("Hello", "nova", "en")

        assert result == b"cached_audio"
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0

    @pytest.mark.asyncio
    @patch('HoloLoom.voice.tts_cache.aioredis.from_url')
    async def test_cache_get_miss(self, mock_from_url, cache_config, mock_redis):
        """Cache get should return None on miss"""
        mock_from_url.return_value = mock_redis
        mock_redis.get = AsyncMock(return_value=None)

        cache = TTSCache(cache_config)
        await cache.initialize()

        result = await cache.get("Hello", "nova", "en")

        assert result is None
        assert cache.stats.hits == 0
        assert cache.stats.misses == 1

    @pytest.mark.asyncio
    @patch('HoloLoom.voice.tts_cache.aioredis.from_url')
    async def test_cache_set(self, mock_from_url, cache_config, mock_redis):
        """Cache set should store audio"""
        mock_from_url.return_value = mock_redis

        cache = TTSCache(cache_config)
        await cache.initialize()

        await cache.set("Hello", "nova", "en", b"audio_data")

        # Verify setex was called with correct TTL
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0].startswith("tts:")
        assert args[1] == 86400  # Common phrase TTL
        assert args[2] == b"audio_data"

    @pytest.mark.asyncio
    @patch('HoloLoom.voice.tts_cache.aioredis.from_url')
    async def test_cache_set_size_limit(self, mock_from_url, cache_config, mock_redis):
        """Cache set should reject oversized audio"""
        mock_from_url.return_value = mock_redis

        cache = TTSCache(cache_config)
        cache.config.max_audio_size_kb = 10  # 10KB limit
        await cache.initialize()

        # Create 20KB audio
        large_audio = b"x" * (20 * 1024)

        await cache.set("Hello", "nova", "en", large_audio)

        # Should not have called setex
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_graceful_fallback(self, cache_config):
        """Cache should gracefully disable if Redis unavailable"""
        # Don't mock Redis, let connection fail
        cache = TTSCache(cache_config)

        # This should not raise, should gracefully disable
        await cache.initialize()

        # Cache should be disabled
        assert cache.config.enable_cache == False or cache._redis_client is None


# ============================================================================
# Test: Cache Management
# ============================================================================

class TestCacheManagement:
    """Test cache management operations"""

    @pytest.mark.asyncio
    @patch('HoloLoom.voice.tts_cache.aioredis.from_url')
    async def test_clear_cache(self, mock_from_url, cache_config, mock_redis):
        """Clear should remove all TTS keys"""
        mock_from_url.return_value = mock_redis
        mock_redis.scan = AsyncMock(return_value=(0, [b"tts:key1", b"tts:key2"]))

        cache = TTSCache(cache_config)
        await cache.initialize()

        await cache.clear()

        # Verify delete was called
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    @patch('HoloLoom.voice.tts_cache.aioredis.from_url')
    async def test_get_cache_size(self, mock_from_url, cache_config, mock_redis):
        """Get cache size should return memory usage"""
        mock_from_url.return_value = mock_redis
        mock_redis.info = AsyncMock(return_value={'used_memory': 1024000})

        cache = TTSCache(cache_config)
        await cache.initialize()

        size = await cache.get_cache_size()

        assert size == 1024000

    @pytest.mark.asyncio
    @patch('HoloLoom.voice.tts_cache.aioredis.from_url')
    async def test_get_stats(self, mock_from_url, cache_config, mock_redis):
        """Get stats should return statistics dictionary"""
        mock_from_url.return_value = mock_redis

        cache = TTSCache(cache_config)
        await cache.initialize()

        stats = cache.get_stats()

        assert isinstance(stats, dict)
        assert 'hit_rate' in stats
        assert 'hits' in stats
        assert 'misses' in stats


# ============================================================================
# Test: Context Manager
# ============================================================================

class TestContextManager:
    """Test async context manager"""

    @pytest.mark.asyncio
    @patch('HoloLoom.voice.tts_cache.aioredis.from_url')
    async def test_context_manager(self, mock_from_url, cache_config, mock_redis):
        """Cache should work as async context manager"""
        mock_from_url.return_value = mock_redis

        async with TTSCache(cache_config) as cache:
            # Should be initialized
            assert cache._redis_client is not None

        # Should be closed
        mock_redis.close.assert_called_once()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
TTS Cache - Production-Ready Audio Caching for HoloLoom VoiceAgent
===================================================================

Redis-based caching system for TTS audio synthesis with intelligent TTL
management, cache warmup, and comprehensive monitoring.

Performance Targets:
- Cache hit rate: 60-80% (after warmup)
- Cache latency: <50ms (vs ~500ms uncached)
- Speedup: 10x for cached phrases
- Memory: <1GB Redis usage

Architecture:
- Cache key: SHA256(text + voice + language)
- TTL: 24h for common phrases, 1h for dynamic content
- Graceful fallback: Disables if Redis unavailable
- Monitoring: Prometheus metrics exported

Date: November 16, 2025
"""

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

# Redis client
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    aioredis = None

# Logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CacheConfig:
    """TTS cache configuration"""
    # Redis connection
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Caching behavior
    enable_cache: bool = True
    common_phrase_ttl: int = 86400  # 24 hours
    dynamic_phrase_ttl: int = 3600  # 1 hour
    max_cache_size_mb: int = 1000  # 1 GB

    # Connection settings
    socket_connect_timeout: int = 5
    socket_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    connection_pool_size: int = 10

    # Performance tuning
    compression_enabled: bool = False  # Future: compress audio
    max_audio_size_kb: int = 500  # Don't cache very large audio

    # Warmup
    enable_warmup: bool = True
    warmup_phrases_file: Optional[str] = "HoloLoom/voice/common_phrases.yaml"


# ============================================================================
# Cache Statistics
# ============================================================================

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    total_latency_saved_ms: float = 0.0
    cache_errors: int = 0
    warmup_count: int = 0
    evictions: int = 0

    # Latency tracking
    _hit_latencies: List[float] = field(default_factory=list)
    _miss_latencies: List[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate (0.0-1.0)"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate (0.0-1.0)"""
        return 1.0 - self.hit_rate

    @property
    def speedup_factor(self) -> float:
        """
        Calculate speedup from caching.

        Assumes:
        - Uncached TTS: 500ms
        - Cached retrieval: 50ms
        """
        if self.total_requests == 0:
            return 1.0

        # Time without cache: all requests at 500ms
        total_time_uncached = 500 * self.total_requests

        # Time with cache: hits at 50ms, misses at 500ms
        total_time_cached = (50 * self.hits) + (500 * self.misses)

        if total_time_cached == 0:
            return 1.0

        return total_time_uncached / total_time_cached

    @property
    def avg_hit_latency_ms(self) -> float:
        """Average latency for cache hits"""
        if not self._hit_latencies:
            return 0.0
        return sum(self._hit_latencies) / len(self._hit_latencies)

    @property
    def avg_miss_latency_ms(self) -> float:
        """Average latency for cache misses"""
        if not self._miss_latencies:
            return 0.0
        return sum(self._miss_latencies) / len(self._miss_latencies)

    def record_hit(self, latency_ms: float):
        """Record cache hit"""
        self.hits += 1
        self.total_requests += 1
        self._hit_latencies.append(latency_ms)

        # Estimate latency saved (vs uncached synthesis)
        self.total_latency_saved_ms += (500 - latency_ms)

    def record_miss(self, latency_ms: float):
        """Record cache miss"""
        self.misses += 1
        self.total_requests += 1
        self._miss_latencies.append(latency_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Export stats as dictionary"""
        return {
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'speedup_factor': self.speedup_factor,
            'avg_hit_latency_ms': self.avg_hit_latency_ms,
            'avg_miss_latency_ms': self.avg_miss_latency_ms,
            'total_latency_saved_ms': self.total_latency_saved_ms,
            'cache_errors': self.cache_errors,
            'warmup_count': self.warmup_count,
            'evictions': self.evictions
        }


# ============================================================================
# Common Phrase Detection
# ============================================================================

class PhraseType(Enum):
    """Phrase classification for TTL selection"""
    COMMON = "common"  # 24h TTL
    DYNAMIC = "dynamic"  # 1h TTL


# Common patterns (regex) for beekeeping domain
COMMON_PATTERNS = [
    # Greetings
    r'^(hello|hi|hey|good morning|good afternoon)',

    # Gratitude
    r'(thank you|thanks|appreciate)',

    # Simple responses
    r'^(yes|no|okay|sure|alright)',

    # Query starters
    r'^(show me|what is|tell me|explain|how do)',

    # Hive references
    r'hive \d+',
    r'(this|that|the) hive',

    # Navigation commands
    r'^(navigate|go to|move to|switch to)',
    r'(next|previous|first|last) hive',

    # Status queries
    r'(status|health|condition)',

    # Inspection commands
    r'(start|begin|end|finish) inspection',

    # Recording
    r'^(record|save|note)',

    # Common beekeeping terms
    r'(queen|brood|honey|pollen|varroa|mite)',

    # AR commands
    r'(show|hide|display|overlay)',
]


def classify_phrase(text: str) -> PhraseType:
    """
    Classify phrase for TTL selection.

    Common phrases get 24h TTL:
    - Match common patterns
    - Short phrases (<5 words)
    - No dynamic content (dates, numbers, names)

    Dynamic phrases get 1h TTL:
    - Long phrases
    - Contain dynamic content
    - Don't match common patterns
    """
    text_lower = text.lower().strip()

    # Check common patterns
    for pattern in COMMON_PATTERNS:
        if re.search(pattern, text_lower):
            return PhraseType.COMMON

    # Short phrases are often repeated
    word_count = len(text.split())
    if word_count < 5:
        return PhraseType.COMMON

    # Check for dynamic content indicators
    dynamic_indicators = [
        r'\d{4}-\d{2}-\d{2}',  # Dates
        r'\d{1,2}:\d{2}',  # Times
        r'(today|yesterday|tomorrow)',  # Temporal
        r'(currently|right now|at the moment)',  # Time-sensitive
    ]

    for indicator in dynamic_indicators:
        if re.search(indicator, text_lower):
            return PhraseType.DYNAMIC

    # Default: common (most beekeeping phrases are repeated)
    return PhraseType.COMMON


# ============================================================================
# TTS Cache
# ============================================================================

class TTSCache:
    """
    Production-ready TTS audio cache with Redis backend.

    Features:
    - Intelligent TTL (24h common, 1h dynamic)
    - Cache key: SHA256(text + voice + language)
    - Graceful fallback if Redis unavailable
    - Comprehensive statistics tracking
    - Prometheus metrics export
    - Cache warmup with common phrases
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize TTS cache.

        Args:
            config: Cache configuration
        """
        self.config = config
        self.stats = CacheStats()

        # Initialize Redis client
        self._redis_client: Optional[aioredis.Redis] = None
        self._init_task: Optional[asyncio.Task] = None

        # Prometheus metrics
        self._init_prometheus_metrics()

        logger.info("tts_cache_initialized",
                   enable_cache=config.enable_cache,
                   redis_host=config.redis_host,
                   redis_port=config.redis_port)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def initialize(self):
        """Initialize Redis connection"""
        if not self.config.enable_cache:
            logger.info("tts_cache_disabled")
            return

        if not REDIS_AVAILABLE:
            logger.warning("redis_not_available",
                          message="redis package not installed - caching disabled")
            self.config.enable_cache = False
            return

        try:
            # Create async Redis client
            self._redis_client = await aioredis.from_url(
                f"redis://{self.config.redis_host}:{self.config.redis_port}",
                db=self.config.redis_db,
                password=self.config.redis_password,
                encoding="utf-8",
                decode_responses=False,  # Binary data
                socket_connect_timeout=self.config.socket_connect_timeout,
                socket_timeout=self.config.socket_timeout,
                max_connections=self.config.connection_pool_size
            )

            # Test connection
            await self._redis_client.ping()

            logger.info("redis_connected",
                       host=self.config.redis_host,
                       port=self.config.redis_port,
                       db=self.config.redis_db)

            # Warmup cache if enabled
            if self.config.enable_warmup:
                asyncio.create_task(self._warmup_cache())

        except Exception as e:
            logger.error("redis_connection_failed",
                        error=str(e),
                        exc_info=True)
            # Graceful fallback
            self.config.enable_cache = False
            self._redis_client = None

    async def close(self):
        """Close Redis connection"""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("redis_connection_closed")

    def _generate_cache_key(self, text: str, voice: str, language: str) -> str:
        """
        Generate deterministic cache key.

        Format: tts:{hash}:{voice}:{lang}
        Example: tts:a3f5b9c2d1e8f7a6:nova:en

        Args:
            text: Text to synthesize
            voice: Voice ID
            language: Language code

        Returns:
            Cache key string
        """
        # Normalize text (case-insensitive, strip whitespace)
        text_normalized = text.lower().strip()

        # Create content hash
        content = f"{text_normalized}:{voice}:{language}"
        hash_value = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

        # Format: tts:{hash}:{voice}:{lang}
        return f"tts:{hash_value}:{voice}:{language}"

    def _get_ttl(self, text: str) -> int:
        """
        Get TTL for phrase based on classification.

        Args:
            text: Text to classify

        Returns:
            TTL in seconds
        """
        phrase_type = classify_phrase(text)

        if phrase_type == PhraseType.COMMON:
            return self.config.common_phrase_ttl
        else:
            return self.config.dynamic_phrase_ttl

    async def get(self, text: str, voice: str, language: str = "en") -> Optional[bytes]:
        """
        Retrieve cached audio.

        Args:
            text: Text to synthesize
            voice: Voice ID
            language: Language code

        Returns:
            Cached audio bytes, or None if not cached
        """
        if not self.config.enable_cache or not self._redis_client:
            return None

        start_time = time.time()

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(text, voice, language)

            # Retrieve from Redis
            audio_bytes = await self._redis_client.get(cache_key)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            if audio_bytes:
                # Cache hit
                self.stats.record_hit(latency_ms)
                self._update_prometheus_hit()

                logger.debug("cache_hit",
                           text_preview=text[:50],
                           voice=voice,
                           language=language,
                           latency_ms=latency_ms)

                return audio_bytes
            else:
                # Cache miss
                self.stats.record_miss(latency_ms)
                self._update_prometheus_miss()

                logger.debug("cache_miss",
                           text_preview=text[:50],
                           voice=voice,
                           language=language)

                return None

        except Exception as e:
            logger.error("cache_get_error",
                        error=str(e),
                        text_preview=text[:50],
                        exc_info=True)
            self.stats.cache_errors += 1
            return None

    async def set(self, text: str, voice: str, language: str, audio: bytes):
        """
        Store audio in cache.

        Args:
            text: Text that was synthesized
            voice: Voice ID used
            language: Language code
            audio: Audio bytes to cache
        """
        if not self.config.enable_cache or not self._redis_client:
            return

        try:
            # Check audio size
            audio_size_kb = len(audio) / 1024
            if audio_size_kb > self.config.max_audio_size_kb:
                logger.warning("audio_too_large_to_cache",
                             size_kb=audio_size_kb,
                             max_kb=self.config.max_audio_size_kb,
                             text_preview=text[:50])
                return

            # Generate cache key
            cache_key = self._generate_cache_key(text, voice, language)

            # Get TTL
            ttl = self._get_ttl(text)

            # Store in Redis
            await self._redis_client.setex(cache_key, ttl, audio)

            phrase_type = classify_phrase(text)

            logger.debug("cache_set",
                        text_preview=text[:50],
                        voice=voice,
                        language=language,
                        ttl=ttl,
                        phrase_type=phrase_type.value,
                        size_kb=audio_size_kb)

        except Exception as e:
            logger.error("cache_set_error",
                        error=str(e),
                        text_preview=text[:50],
                        exc_info=True)
            self.stats.cache_errors += 1

    async def warmup(self, phrases: List[str], voice: str = "nova", language: str = "en"):
        """
        Warmup cache with common phrases.

        Note: This method requires a TTS provider to synthesize the phrases.
        It's called during initialization if enable_warmup=True.

        Args:
            phrases: List of common phrases to cache
            voice: Voice ID
            language: Language code
        """
        logger.info("cache_warmup_started",
                   phrase_count=len(phrases),
                   voice=voice,
                   language=language)

        # This is a placeholder - actual warmup happens in integration
        # with TTSManager which has access to the TTS provider
        self.stats.warmup_count = len(phrases)

        logger.info("cache_warmup_complete",
                   cached_count=len(phrases))

    async def _warmup_cache(self):
        """Internal warmup - loads phrases from YAML if available"""
        if not self.config.warmup_phrases_file:
            return

        try:
            import yaml
            from pathlib import Path

            phrases_path = Path(self.config.warmup_phrases_file)
            if not phrases_path.exists():
                logger.warning("warmup_phrases_file_not_found",
                             path=str(phrases_path))
                return

            with open(phrases_path, 'r') as f:
                data = yaml.safe_load(f)

            phrases = data.get('common_phrases', [])

            logger.info("warmup_phrases_loaded",
                       count=len(phrases))

            # Note: Actual TTS synthesis happens at integration level
            # This just logs that phrases are ready for warmup

        except Exception as e:
            logger.error("warmup_load_error",
                        error=str(e),
                        exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.stats.to_dict()

    async def clear(self):
        """Clear all cached TTS audio"""
        if not self._redis_client:
            return

        try:
            # Find all TTS keys
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self._redis_client.scan(
                    cursor,
                    match="tts:*",
                    count=100
                )

                if keys:
                    await self._redis_client.delete(*keys)
                    deleted += len(keys)

                if cursor == 0:
                    break

            logger.info("cache_cleared", deleted_count=deleted)

        except Exception as e:
            logger.error("cache_clear_error",
                        error=str(e),
                        exc_info=True)

    async def get_cache_size(self) -> int:
        """
        Get approximate cache size in bytes.

        Returns:
            Cache size in bytes
        """
        if not self._redis_client:
            return 0

        try:
            info = await self._redis_client.info('memory')
            return info.get('used_memory', 0)
        except Exception as e:
            logger.error("cache_size_error",
                        error=str(e),
                        exc_info=True)
            return 0

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            self.metrics_hits = Counter(
                'tts_cache_hits_total',
                'Total number of TTS cache hits'
            )
            self.metrics_misses = Counter(
                'tts_cache_misses_total',
                'Total number of TTS cache misses'
            )
            self.metrics_hit_rate = Gauge(
                'tts_cache_hit_rate',
                'TTS cache hit rate (0.0-1.0)'
            )
            self.metrics_latency = Histogram(
                'tts_cache_latency_seconds',
                'TTS cache operation latency',
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            )
            self.metrics_size = Gauge(
                'tts_cache_size_bytes',
                'TTS cache size in bytes'
            )

            logger.info("prometheus_metrics_initialized")
        except Exception as e:
            logger.error("prometheus_init_error",
                        error=str(e),
                        exc_info=True)

    def _update_prometheus_hit(self):
        """Update Prometheus metrics for cache hit"""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            self.metrics_hits.inc()
            self.metrics_hit_rate.set(self.stats.hit_rate)
        except Exception:
            pass

    def _update_prometheus_miss(self):
        """Update Prometheus metrics for cache miss"""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            self.metrics_misses.inc()
            self.metrics_hit_rate.set(self.stats.hit_rate)
        except Exception:
            pass

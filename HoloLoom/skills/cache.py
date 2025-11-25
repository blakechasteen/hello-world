"""Skill Result Caching - Intelligent result caching for skills"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from HoloLoom.skills.base import SkillOutput


@dataclass
class CacheEntry:
    """Cached skill result."""
    result: SkillOutput
    timestamp: float
    hit_count: int

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - self.timestamp) > ttl_seconds


class SkillCache:
    """
    Intelligent caching for skill results.

    Caches based on operation + parameters hash.
    LRU eviction when max size reached.
    Configurable TTL (time-to-live) per cache entry.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        """
        Initialize skill cache.

        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default time-to-live in seconds (default: 1 hour)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: list = []  # For LRU tracking

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _compute_key(self, skill_name: str, operation: str, parameters: Dict[str, Any]) -> str:
        """
        Compute cache key from skill invocation.

        Uses SHA256 hash of skill_name + operation + sorted parameters JSON.
        """
        # Sort parameters for consistent hashing
        params_str = json.dumps(parameters, sort_keys=True, default=str)
        key_string = f"{skill_name}:{operation}:{params_str}"

        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

    def get(
        self,
        skill_name: str,
        operation: str,
        parameters: Dict[str, Any],
        ttl: Optional[float] = None
    ) -> Optional[SkillOutput]:
        """
        Get cached result if available and not expired.

        Args:
            skill_name: Name of the skill
            operation: Operation name
            parameters: Operation parameters
            ttl: Custom TTL (uses default if None)

        Returns:
            Cached SkillOutput or None if not found/expired
        """
        key = self._compute_key(skill_name, operation, parameters)

        if key not in self._cache:
            self.misses += 1
            return None

        entry = self._cache[key]

        # Check expiration
        ttl_seconds = ttl if ttl is not None else self.default_ttl
        if entry.is_expired(ttl_seconds):
            # Remove expired entry
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self.misses += 1
            return None

        # Cache hit - update metrics and access order
        self.hits += 1
        entry.hit_count += 1

        # Move to end (most recently used)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        return entry.result

    def put(
        self,
        skill_name: str,
        operation: str,
        parameters: Dict[str, Any],
        result: SkillOutput
    ) -> None:
        """
        Store result in cache.

        Args:
            skill_name: Name of the skill
            operation: Operation name
            parameters: Operation parameters
            result: Result to cache
        """
        key = self._compute_key(skill_name, operation, parameters)

        # LRU eviction if at max size
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Remove least recently used
            if self._access_order:
                lru_key = self._access_order.pop(0)
                if lru_key in self._cache:
                    del self._cache[lru_key]
                    self.evictions += 1

        # Store entry
        self._cache[key] = CacheEntry(
            result=result,
            timestamp=time.time(),
            hit_count=0
        )

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def invalidate(self, skill_name: str, operation: str, parameters: Dict[str, Any]) -> bool:
        """
        Invalidate specific cache entry.

        Returns:
            True if entry was found and removed, False otherwise
        """
        key = self._compute_key(skill_name, operation, parameters)

        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True

        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

    def get_top_entries(self, n: int = 10) -> list:
        """
        Get top N most frequently accessed cache entries.

        Args:
            n: Number of entries to return

        Returns:
            List of (key, hit_count) tuples
        """
        entries = [(key, entry.hit_count) for key, entry in self._cache.items()]
        entries.sort(key=lambda x: x[1], reverse=True)
        return entries[:n]

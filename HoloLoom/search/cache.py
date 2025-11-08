"""
Search Result Caching
=====================
TTL-based caching for search results to avoid redundant API calls.

Philosophy:
"Cache early, cache often."

Provides:
- Query-based caching (same query -> cached results)
- TTL expiration (results expire after 1 hour by default)
- LRU eviction (oldest results removed when cache full)
- Statistics tracking (hit rate, savings)
"""

import time
import hashlib
from typing import List, Dict, Optional, Any
from collections import OrderedDict
from dataclasses import dataclass, field

from .protocol import WebSearchResult


# ============================================================================
# Cache Entry
# ============================================================================

@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    query: str
    results: List[WebSearchResult]
    timestamp: float
    ttl_seconds: int
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        age = time.time() - self.timestamp
        return age > self.ttl_seconds

    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.timestamp


# ============================================================================
# Search Cache
# ============================================================================

class SearchCache:
    """
    LRU cache with TTL for search results.

    Features:
    - Query-based caching
    - TTL expiration (configurable per entry)
    - LRU eviction when full
    - Hit rate tracking
    - Query normalization (case, whitespace)

    Usage:
        cache = SearchCache(max_size=1000, default_ttl=3600)

        # Store results
        cache.put(query="What is Thompson Sampling?", results=results)

        # Retrieve results (None if expired/missing)
        cached = cache.get(query="what is thompson sampling?")  # Case-insensitive

        # Get stats
        stats = cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.1%}")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        normalize_queries: bool = True
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.normalize_queries = normalize_queries

        # LRU cache (OrderedDict maintains insertion order)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, query: str) -> Optional[List[WebSearchResult]]:
        """
        Get cached results for query.

        Args:
            query: Search query

        Returns:
            Cached results if found and not expired, None otherwise
        """
        key = self._normalize_query(query) if self.normalize_queries else query

        # Check if in cache
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        # Check if expired
        if entry.is_expired():
            self._expirations += 1
            self._cache.pop(key)
            self._misses += 1
            return None

        # Hit! Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1

        return entry.results

    def put(
        self,
        query: str,
        results: List[WebSearchResult],
        ttl: Optional[int] = None
    ) -> None:
        """
        Store results in cache.

        Args:
            query: Search query
            results: Search results to cache
            ttl: Optional TTL override (seconds)
        """
        key = self._normalize_query(query) if self.normalize_queries else query
        ttl = ttl if ttl is not None else self.default_ttl

        # Create entry
        entry = CacheEntry(
            query=query,
            results=results,
            timestamp=time.time(),
            ttl_seconds=ttl
        )

        # Remove old entry if exists
        if key in self._cache:
            self._cache.pop(key)

        # Add to cache
        self._cache[key] = entry

        # Evict oldest if over capacity
        if len(self._cache) > self.max_size:
            self._evict_oldest()

    def invalidate(self, query: str) -> bool:
        """
        Remove query from cache.

        Args:
            query: Search query

        Returns:
            True if removed, False if not found
        """
        key = self._normalize_query(query) if self.normalize_queries else query

        if key in self._cache:
            self._cache.pop(key)
            return True

        return False

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = []

        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            self._cache.pop(key)
            self._expirations += 1

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hit rate, size, evictions, etc.
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "expirations": self._expirations,
            "total_requests": total_requests,
        }

    def get_entry_stats(self) -> List[Dict[str, Any]]:
        """
        Get stats for individual cache entries.

        Returns:
            List of entry stats sorted by hit count
        """
        stats = []

        for key, entry in self._cache.items():
            stats.append({
                "query": entry.query,
                "hit_count": entry.hit_count,
                "age_seconds": entry.age_seconds(),
                "ttl_seconds": entry.ttl_seconds,
                "result_count": len(entry.results),
                "expired": entry.is_expired(),
            })

        # Sort by hit count
        stats.sort(key=lambda x: x["hit_count"], reverse=True)

        return stats

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent caching.

        - Lowercase
        - Strip whitespace
        - Collapse multiple spaces
        """
        normalized = query.lower().strip()
        normalized = " ".join(normalized.split())  # Collapse whitespace
        return normalized

    def _evict_oldest(self) -> None:
        """Evict oldest (least recently used) entry."""
        if self._cache:
            self._cache.popitem(last=False)  # Remove first (oldest) item
            self._evictions += 1


# ============================================================================
# Global Cache Instance (Optional)
# ============================================================================

_global_cache: Optional[SearchCache] = None


def get_global_cache() -> SearchCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SearchCache()
    return _global_cache


def set_global_cache(cache: SearchCache) -> None:
    """Set global cache instance."""
    global _global_cache
    _global_cache = cache

#!/usr/bin/env python3
"""
Performance Cache - LRU cache with TTL for query results
"""
import time
from typing import Any, Optional, Dict
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Cached item with timestamp."""
    value: Any
    timestamp: float
    hits: int = 0


class QueryCache:
    """
    Simple LRU cache with TTL for query results.

    Usage:
        cache = QueryCache(max_size=50, ttl_seconds=300)
        cache.put("query", result)
        result = cache.get("query")
    """

    def __init__(self, max_size: int = 50, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get cached value, None if expired or missing."""
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # Check expiration
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self.cache[key]
            self.misses += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry.hits += 1
        self.hits += 1
        return entry.value

    def put(self, key: str, value: Any) -> None:
        """Cache a value."""
        # Update existing
        if key in self.cache:
            self.cache[key].value = value
            self.cache[key].timestamp = time.time()
            self.cache.move_to_end(key)
            return

        # Evict LRU if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        # Add new entry
        self.cache[key] = CacheEntry(value=value, timestamp=time.time())

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
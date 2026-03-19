"""
Query Cache — LRU+TTL cache for LiteMemoryBus.query() results.

Keyed by normalized MemoryQuery hash, invalidated by bus generation counter.
Follows SearchCache pattern (search/cache.py).

Integration: optional `query_cache` parameter on LiteMemoryBus.__init__.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .bus import MemoryQuery, MemoryResult

__all__ = [
    "QueryCacheEntry",
    "QueryCache",
]


@dataclass
class QueryCacheEntry:
    """A single cached query result."""
    query_key: str
    result: "MemoryResult"
    generation: int
    timestamp: float
    ttl_seconds: float
    hit_count: int = 0

    def is_stale(self, current_generation: int) -> bool:
        return self.generation != current_generation

    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds


class QueryCache:
    """LRU+TTL cache for MemoryQuery → MemoryResult.

    Cache entries become stale when bus._generation changes (any mutation).
    Also supports TTL-based expiry for long-lived sessions.
    """

    def __init__(self, max_size: int = 256, default_ttl: float = 60.0):
        self._cache: OrderedDict[str, QueryCacheEntry] = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._stale_evictions = 0

    def get(self, query: "MemoryQuery", current_generation: int) -> Optional["MemoryResult"]:
        """Return cached result if valid, None otherwise."""
        key = self._normalize_query(query)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_stale(current_generation) or entry.is_expired():
            self._cache.pop(key)
            self._stale_evictions += 1
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1
        return entry.result

    def put(
        self,
        query: "MemoryQuery",
        result: "MemoryResult",
        generation: int,
        ttl: float | None = None,
    ):
        """Cache a query result."""
        key = self._normalize_query(query)
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = QueryCacheEntry(
            query_key=key,
            result=result,
            generation=generation,
            timestamp=time.time(),
            ttl_seconds=ttl if ttl is not None else self.default_ttl,
        )

    def clear(self):
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "stale_evictions": self._stale_evictions,
        }

    @staticmethod
    def _normalize_query(q: "MemoryQuery") -> str:
        """Deterministic hash of query fields."""
        parts = [
            q.intent or "",
            str(sorted(q.entity_ids or [])),
            q.entity_type or "",
            q.memory_type or "",
            q.time_after or "",
            q.time_before or "",
            str(q.max_results),
            q.semantic_text or "",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

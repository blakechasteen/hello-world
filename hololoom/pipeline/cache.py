"""
Pipeline Stage Cache — Content-addressed memoization.

Caches stage outputs keyed by (stage_name, hash_of_inputs).
If the same inputs produce the same outputs, skip the stage.

Scope: per-orchestrator instance. Not persisted across restarts.
Eviction: LRU via OrderedDict.
"""

import hashlib
from collections import OrderedDict
from typing import Any


class StageCache:
    """
    LRU cache keyed by (stage_name, input_hash).

    Args:
        maxsize: maximum number of cached results. Default 256.
    """

    def __init__(self, maxsize: int = 256):
        self._store: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, stage_name: str, key: str) -> dict[str, Any] | None:
        """Look up cached outputs. Returns None on miss."""
        cache_key = (stage_name, key)
        if cache_key in self._store:
            self._store.move_to_end(cache_key)
            self.hits += 1
            return self._store[cache_key]
        self.misses += 1
        return None

    def set(self, stage_name: str, key: str, outputs: dict[str, Any]) -> None:
        """Store stage outputs. Evicts LRU entry if at capacity."""
        cache_key = (stage_name, key)
        self._store[cache_key] = outputs
        self._store.move_to_end(cache_key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    @staticmethod
    def hash_inputs(ctx: Any, reads: frozenset[str]) -> str:
        """
        Hash the values of declared reads fields on the context.

        Uses repr() + sha256 for simplicity. Fields with unhashable
        values (numpy arrays, large dicts) use shape/len as proxy.
        Not meant to be cryptographic — just collision-resistant enough
        for cache keys.
        """
        parts = []
        for field_name in sorted(reads):
            value = getattr(ctx, field_name, None)
            try:
                parts.append(f"{field_name}={repr(value)}")
            except Exception:
                # Fallback for objects with broken repr
                parts.append(f"{field_name}=<{type(value).__name__}>")
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode('utf-8', errors='replace')).hexdigest()[:16]

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._store)

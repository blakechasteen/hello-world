"""
Render Cache — Cache MemFS.render_for_prompt() and tree() output.

Invalidated by bus generation counter. Small LRU (32 entries) since
render_for_prompt is typically called with few distinct (query, budget) combos.

Integration: optional `render_cache` parameter on MemFS.__init__.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

__all__ = [
    "RenderCacheEntry",
    "RenderCache",
]


@dataclass
class RenderCacheEntry:
    """A single cached render output."""
    key: str
    rendered: str
    generation: int
    token_count: int
    timestamp: float
    hit_count: int = 0


class RenderCache:
    """LRU cache for MemFS render output, invalidated by bus generation."""

    def __init__(self, max_entries: int = 32):
        self._prompt_cache: OrderedDict[str, RenderCacheEntry] = OrderedDict()
        self._tree_cache: RenderCacheEntry | None = None
        self.max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def get_prompt(self, query: str, budget: int, generation: int) -> str | None:
        """Return cached render_for_prompt output if valid."""
        key = self._prompt_key(query, budget)
        entry = self._prompt_cache.get(key)
        if entry and entry.generation == generation:
            self._prompt_cache.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            return entry.rendered
        if entry:
            self._prompt_cache.pop(key)
        self._misses += 1
        return None

    def put_prompt(
        self,
        query: str,
        budget: int,
        generation: int,
        rendered: str,
    ):
        """Cache a render_for_prompt result."""
        key = self._prompt_key(query, budget)
        while len(self._prompt_cache) >= self.max_entries:
            self._prompt_cache.popitem(last=False)
        token_count = max(1, len(rendered) // 4)
        self._prompt_cache[key] = RenderCacheEntry(
            key=key,
            rendered=rendered,
            generation=generation,
            token_count=token_count,
            timestamp=time.time(),
        )

    def get_tree(self, generation: int) -> str | None:
        """Return cached tree() output if valid."""
        if self._tree_cache and self._tree_cache.generation == generation:
            self._tree_cache.hit_count += 1
            self._hits += 1
            return self._tree_cache.rendered
        self._misses += 1
        return None

    def put_tree(self, generation: int, tree_str: str):
        """Cache a tree() result."""
        self._tree_cache = RenderCacheEntry(
            key="_tree",
            rendered=tree_str,
            generation=generation,
            token_count=max(1, len(tree_str) // 4),
            timestamp=time.time(),
        )

    def clear(self):
        self._prompt_cache.clear()
        self._tree_cache = None

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "prompt_cache_size": len(self._prompt_cache),
            "tree_cached": self._tree_cache is not None,
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }

    @staticmethod
    def _prompt_key(query: str, budget: int) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:12] + f":{budget}"

"""
Repo Cache — Persistent hash-indexed cache for code file chunks.

Maps (file_path, SHA256(content)) → pre-chunked item snapshots.
Survives across sessions via JSON persistence alongside bus snapshots.
Only re-chunks files whose content hash has changed.

Integration: optional `repo_cache` parameter on LiteMemoryBus.__init__.
Pairs with repository_context.py's chunk_code_file() for parsing.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RepoCacheEntry",
    "RepoCache",
]


@dataclass
class RepoCacheEntry:
    """Cached chunk set for a single file."""
    file_path: str
    content_hash: str
    chunks: list[dict[str, Any]]
    language: str = ""
    entity_names: list[str] = field(default_factory=list)
    line_count: int = 0
    total_tokens: int = 0
    cached_at: float = field(default_factory=time.time)


class RepoCache:
    """Persistent hash-indexed repository cache.

    Maps (file_path, content_hash) → list of pre-chunked item snapshots.
    Persists to a single JSON file alongside the bus snapshot.
    """

    CACHE_VERSION = 1

    def __init__(self, cache_path: str | None = None):
        self._entries: dict[str, RepoCacheEntry] = {}
        self._cache_path = cache_path
        self._hits = 0
        self._misses = 0

    def lookup(self, file_path: str, content_hash: str) -> list[dict[str, Any]] | None:
        """Return cached chunks if file unchanged, None on cache miss."""
        entry = self._entries.get(file_path)
        if entry and entry.content_hash == content_hash:
            self._hits += 1
            return entry.chunks
        self._misses += 1
        return None

    def store(
        self,
        file_path: str,
        content_hash: str,
        chunks: list[dict[str, Any]],
        language: str = "",
        entity_names: list[str] | None = None,
        line_count: int = 0,
    ):
        """Cache chunks for a file."""
        total_tokens = sum(c.get("token_estimate", len(str(c)) // 4) for c in chunks)
        self._entries[file_path] = RepoCacheEntry(
            file_path=file_path,
            content_hash=content_hash,
            chunks=chunks,
            language=language,
            entity_names=entity_names or [],
            line_count=line_count,
            total_tokens=total_tokens,
            cached_at=time.time(),
        )

    def invalidate(self, file_path: str) -> bool:
        """Remove cache for a single file. Returns True if found."""
        return self._entries.pop(file_path, None) is not None

    def invalidate_stale(self, current_hashes: dict[str, str]) -> int:
        """Remove entries whose hashes no longer match or whose files are gone.

        Args:
            current_hashes: {file_path: content_hash} for all current files.

        Returns:
            Number of invalidated entries.
        """
        stale = [
            fp for fp, entry in self._entries.items()
            if fp not in current_hashes or current_hashes[fp] != entry.content_hash
        ]
        for fp in stale:
            del self._entries[fp]
        return len(stale)

    def save(self, path: str | None = None) -> None:
        """Persist to JSON (atomic write)."""
        p = Path(path or self._cache_path)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "_version": self.CACHE_VERSION,
            "_saved_at": datetime.utcnow().isoformat(),
            "_file_count": len(self._entries),
            "_total_chunks": sum(len(e.chunks) for e in self._entries.values()),
            "entries": {fp: asdict(e) for fp, e in self._entries.items()},
        }
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.replace(p)
        logger.info("Saved repo cache: %d files, %d chunks -> %s",
                     len(self._entries),
                     sum(len(e.chunks) for e in self._entries.values()),
                     str(p))

    def load(self, path: str | None = None) -> bool:
        """Load from JSON. Returns False if file not found."""
        p = Path(path or self._cache_path)
        if p is None or not p.exists():
            return False
        with open(p) as f:
            data = json.load(f)

        version = data.get("_version", 0)
        if version > self.CACHE_VERSION:
            logger.warning("Repo cache version %d > supported %d, skipping",
                           version, self.CACHE_VERSION)
            return False

        self._entries.clear()
        for fp, entry_dict in data.get("entries", {}).items():
            self._entries[fp] = RepoCacheEntry(**{
                k: v for k, v in entry_dict.items()
                if k in RepoCacheEntry.__dataclass_fields__
            })

        logger.info("Loaded repo cache: %d files from %s", len(self._entries), str(p))
        return True

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        total_chunks = sum(len(e.chunks) for e in self._entries.values())
        total_tokens = sum(e.total_tokens for e in self._entries.values())
        return {
            "file_count": len(self._entries),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }

    @staticmethod
    def hash_content(content: str) -> str:
        """SHA256 hash of file content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def cached_files(self) -> set[str]:
        """Return set of all cached file paths."""
        return set(self._entries.keys())

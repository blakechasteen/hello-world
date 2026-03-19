"""
Version Log — Append-only delta log for memory bus operations.

Borrowed from Letta (git-backed memory) + GCC (COMMIT/BRANCH/CONTEXT).

Every mutation to the memory bus can be recorded as a MemoryDelta.
The log supports replay (reconstruct state at any point), rollback,
diff, and milestone marking (GCC's "COMMIT" equivalent).

Integration: optional parameter on LiteMemoryBus.__init__.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["MemoryDelta", "VersionLog"]

ROLLBACK_SNAPSHOT_VERSION = 2  # Must match LiteMemoryBus.SNAPSHOT_VERSION


@dataclass(frozen=True)
class MemoryDelta:
    """A single atomic change to the memory store."""
    delta_id: str                          # Unique identifier
    timestamp: str                         # ISO 8601
    operation: str                         # store|forget|update|consolidate|merge|milestone
    item_id: str                           # Which StoredItem changed ("" for milestones)
    before: dict[str, Any] | None       # Previous state (None for store/milestone)
    after: dict[str, Any] | None        # New state (None for forget/milestone)
    metadata: dict[str, Any] = field(default_factory=dict)


class VersionLog:
    """Append-only delta log with replay and rollback.

    Default: in-memory list + JSON file persistence.
    Mirrors the atomic-write pattern from LiteMemoryBus.save_snapshot().
    """

    def __init__(self, max_deltas: int = 10_000):
        self._deltas: list[MemoryDelta] = []
        self._index: dict[str, int] = {}   # delta_id → position in _deltas
        self._max_deltas = max_deltas

    @property
    def count(self) -> int:
        return len(self._deltas)

    def record(self, delta: MemoryDelta) -> None:
        """Append a delta. O(1). Evicts oldest if at capacity."""
        if len(self._deltas) >= self._max_deltas:
            evicted = self._deltas.pop(0)
            del self._index[evicted.delta_id]
            # Re-index (shift all positions down by 1)
            self._index = {
                did: pos - 1 for did, pos in self._index.items()
            }

        pos = len(self._deltas)
        self._deltas.append(delta)
        self._index[delta.delta_id] = pos

    def replay(self, since: str | None = None) -> list[MemoryDelta]:
        """Return deltas since given delta_id (inclusive). None = all."""
        if since is None:
            return list(self._deltas)
        if since not in self._index:
            return []
        start = self._index[since]
        return list(self._deltas[start:])

    def rollback(self, to_delta_id: str) -> dict[str, Any]:
        """Reconstruct snapshot dict by replaying deltas up to given point.

        Applies deltas sequentially from the beginning:
        - store → insert item into snapshot
        - forget → remove item from snapshot
        - merge/update → replace item content in snapshot
        - consolidate → apply as store + forget
        - milestone → no-op (informational)

        Returns dict compatible with LiteMemoryBus.from_snapshot_dict().
        """
        if to_delta_id not in self._index:
            raise ValueError(f"Unknown delta_id: {to_delta_id}")

        end = self._index[to_delta_id] + 1  # inclusive
        items: dict[str, dict[str, Any]] = {}
        edges: dict[str, list] = {}
        aliases: dict[str, str] = {}

        for delta in self._deltas[:end]:
            if delta.operation == "store" and delta.after:
                items[delta.item_id] = delta.after
            elif delta.operation == "forget":
                items.pop(delta.item_id, None)
            elif delta.operation in ("merge", "update") and delta.after:
                if delta.item_id in items:
                    items[delta.item_id] = delta.after
                else:
                    items[delta.item_id] = delta.after
            elif delta.operation == "consolidate":
                # Consolidation stores new summary and forgets originals
                if delta.after and delta.item_id:
                    items[delta.item_id] = delta.after
                # Originals already removed by individual forget deltas
            # milestone → no state change

        return {
            "_version": ROLLBACK_SNAPSHOT_VERSION,
            "items": items,
            "edges": edges,
            "aliases": aliases,
        }

    def diff(self, from_id: str, to_id: str) -> list[MemoryDelta]:
        """Return deltas between two points (exclusive of from, inclusive of to)."""
        if from_id not in self._index or to_id not in self._index:
            return []
        start = self._index[from_id] + 1  # exclusive of from
        end = self._index[to_id] + 1      # inclusive of to
        if start > end:
            return []
        return list(self._deltas[start:end])

    def milestones(self) -> list[MemoryDelta]:
        """Return only milestone deltas (GCC COMMIT equivalent)."""
        return [d for d in self._deltas if d.operation == "milestone"]

    def record_milestone(
        self, description: str, metadata: dict | None = None,
    ) -> MemoryDelta:
        """Convenience: record a milestone delta with no item changes."""
        delta = MemoryDelta(
            delta_id=str(uuid.uuid4())[:12],
            timestamp=datetime.utcnow().isoformat(),
            operation="milestone",
            item_id="",
            before=None,
            after=None,
            metadata={"description": description, **(metadata or {})},
        )
        self.record(delta)
        return delta

    def last_delta(self) -> MemoryDelta | None:
        """Return the most recent delta, or None if empty."""
        return self._deltas[-1] if self._deltas else None

    # =========================================================================
    # Persistence (mirrors LiteMemoryBus.save_snapshot / load_snapshot)
    # =========================================================================

    def save(self, path: Path) -> None:
        """Save deltas to JSON file (atomic write)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "deltas": [
                {
                    "delta_id": d.delta_id,
                    "timestamp": d.timestamp,
                    "operation": d.operation,
                    "item_id": d.item_id,
                    "before": d.before,
                    "after": d.after,
                    "metadata": d.metadata,
                }
                for d in self._deltas
            ],
            "_saved_at": datetime.utcnow().isoformat(),
        }
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(p)
        logger.info("Saved version log: %d deltas → %s", len(self._deltas), path)

    def load(self, path: Path) -> bool:
        """Load deltas from JSON. Returns False if file not found."""
        p = Path(path)
        if not p.exists():
            return False

        with open(p) as f:
            data = json.load(f)

        self._deltas.clear()
        self._index.clear()
        for i, d in enumerate(data.get("deltas", [])):
            delta = MemoryDelta(
                delta_id=d["delta_id"],
                timestamp=d["timestamp"],
                operation=d["operation"],
                item_id=d["item_id"],
                before=d.get("before"),
                after=d.get("after"),
                metadata=d.get("metadata", {}),
            )
            self._deltas.append(delta)
            self._index[delta.delta_id] = i

        logger.info("Loaded version log: %d deltas from %s", len(self._deltas), path)
        return True

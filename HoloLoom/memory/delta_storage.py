"""
HoloLoom Memory - Delta Storage (Phase 4)
==========================================
Stores memory changes as deltas for space-efficient storage and reconstruction.

Architecture:
    DeltaStore: Manages delta records and checkpoints
    MemoryDelta: Single change record (ADD/UPDATE/DELETE)
    DeltaCheckpoint: Full snapshot for reconstruction base
    DeltaReconstructor: Rebuilds memory state from deltas

Philosophy:
    "Store the change, not the state."

    Instead of storing full memory snapshots, we store deltas (changes).
    This enables:
    - 10-100x storage savings for frequently-updated memories
    - Time-travel to any point in memory history
    - Efficient sync between distributed nodes
    - Complete audit trail of memory evolution

Integration:
    Works with KG (graph) and vector memory to track changes at node/edge level.

Author: Claude Code
Date: December 2025 (Phase 4 - Reconstruction)
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class DeltaOperation(Enum):
    """Type of delta operation."""
    ADD = "add"           # New node/edge added
    UPDATE = "update"     # Existing node/edge updated
    DELETE = "delete"     # Node/edge deleted
    MERGE = "merge"       # Nodes merged together
    SPLIT = "split"       # Node split into multiple


class DeltaTarget(Enum):
    """What the delta affects."""
    NODE = "node"         # Knowledge graph node
    EDGE = "edge"         # Knowledge graph edge
    EMBEDDING = "embedding"  # Vector embedding
    METADATA = "metadata"    # Node/edge metadata
    ACTIVATION = "activation"  # Activation level


@dataclass
class MemoryDelta:
    """
    A single change record in memory.

    Deltas are the atomic unit of change tracking. Each delta captures:
    - What changed (target_id, target_type)
    - How it changed (operation, before, after)
    - When it changed (timestamp)
    - Why it changed (source, reason)
    """
    delta_id: str
    target_id: str  # ID of affected node/edge
    target_type: DeltaTarget
    operation: DeltaOperation
    timestamp: float = field(default_factory=time.time)

    # Change details
    before: Optional[Dict[str, Any]] = None  # State before (for UPDATE/DELETE)
    after: Optional[Dict[str, Any]] = None   # State after (for ADD/UPDATE)

    # Provenance
    source: str = "unknown"  # What caused this change
    reason: Optional[str] = None  # Why the change was made
    query_id: Optional[str] = None  # Related query if any

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize delta to dictionary."""
        return {
            "delta_id": self.delta_id,
            "target_id": self.target_id,
            "target_type": self.target_type.value,
            "operation": self.operation.value,
            "timestamp": self.timestamp,
            "before": self.before,
            "after": self.after,
            "source": self.source,
            "reason": self.reason,
            "query_id": self.query_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryDelta':
        """Deserialize delta from dictionary."""
        return cls(
            delta_id=data["delta_id"],
            target_id=data["target_id"],
            target_type=DeltaTarget(data["target_type"]),
            operation=DeltaOperation(data["operation"]),
            timestamp=data.get("timestamp", time.time()),
            before=data.get("before"),
            after=data.get("after"),
            source=data.get("source", "unknown"),
            reason=data.get("reason"),
            query_id=data.get("query_id"),
            metadata=data.get("metadata", {}),
        )

    @property
    def size_bytes(self) -> int:
        """Estimate size in bytes."""
        return len(json.dumps(self.to_dict(), default=str))


@dataclass
class DeltaCheckpoint:
    """
    Full snapshot for reconstruction base.

    Checkpoints are taken periodically to:
    - Limit reconstruction time (don't need all history)
    - Enable garbage collection of old deltas
    - Provide recovery points
    """
    checkpoint_id: str
    timestamp: float = field(default_factory=time.time)

    # Full state at checkpoint
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)

    # Metadata
    delta_count_before: int = 0  # Deltas before this checkpoint
    reason: str = "scheduled"    # Why checkpoint was taken

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "nodes": self.nodes,
            "edges": self.edges,
            "embeddings": self.embeddings,
            "delta_count_before": self.delta_count_before,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeltaCheckpoint':
        """Deserialize checkpoint from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            timestamp=data.get("timestamp", time.time()),
            nodes=data.get("nodes", {}),
            edges=data.get("edges", {}),
            embeddings=data.get("embeddings", {}),
            delta_count_before=data.get("delta_count_before", 0),
            reason=data.get("reason", "loaded"),
        )

    @property
    def size_bytes(self) -> int:
        """Estimate size in bytes."""
        return len(json.dumps(self.to_dict(), default=str))


# ============================================================================
# Delta Store
# ============================================================================

@dataclass
class DeltaStoreConfig:
    """Configuration for delta storage."""
    # Checkpoint policy
    checkpoint_interval_deltas: int = 1000  # Checkpoint every N deltas
    checkpoint_interval_seconds: int = 3600  # Or every N seconds
    max_deltas_before_checkpoint: int = 5000  # Force checkpoint at this limit

    # Retention policy
    max_deltas: int = 100000  # Maximum deltas to retain
    max_checkpoints: int = 10  # Maximum checkpoints to retain
    prune_before_oldest_checkpoint: bool = True  # Remove deltas before oldest checkpoint

    # Compression
    compress_old_deltas: bool = True  # Compress deltas older than threshold
    compression_age_hours: int = 24  # Age threshold for compression


class DeltaStore:
    """
    Manages delta records and checkpoints for memory reconstruction.

    Features:
    - Append-only delta log
    - Periodic checkpointing
    - Reconstruction to any point in time
    - Space-efficient with automatic compression
    - Integration with ForgetManager for cleanup

    Usage:
        store = DeltaStore()

        # Record changes
        store.record_add("node_1", DeltaTarget.NODE, {"text": "Hello"})
        store.record_update("node_1", DeltaTarget.NODE,
                           before={"text": "Hello"},
                           after={"text": "Hello World"})

        # Reconstruct state
        state = store.reconstruct_at(timestamp)

        # Take checkpoint
        store.checkpoint(nodes, edges, embeddings)
    """

    def __init__(self, config: Optional[DeltaStoreConfig] = None):
        """
        Initialize delta store.

        Args:
            config: Configuration for storage behavior
        """
        self.config = config or DeltaStoreConfig()

        # Delta log (append-only)
        self.deltas: List[MemoryDelta] = []

        # Checkpoints
        self.checkpoints: List[DeltaCheckpoint] = []

        # Index for fast lookup
        self._target_index: Dict[str, List[int]] = defaultdict(list)  # target_id -> delta indices
        self._time_index: List[float] = []  # Sorted timestamps for binary search

        # Statistics
        self.total_deltas_recorded = 0
        self.last_checkpoint_time = time.time()
        self._delta_counter = 0

        self.logger = logging.getLogger(f"{__name__}.DeltaStore")

    def record_add(
        self,
        target_id: str,
        target_type: DeltaTarget,
        state: Dict[str, Any],
        source: str = "unknown",
        reason: Optional[str] = None,
        query_id: Optional[str] = None
    ) -> MemoryDelta:
        """
        Record an ADD operation (new node/edge).

        Args:
            target_id: ID of the new item
            target_type: Type of item
            state: Initial state
            source: What caused this add
            reason: Why it was added
            query_id: Related query

        Returns:
            The recorded delta
        """
        delta = MemoryDelta(
            delta_id=self._generate_delta_id(),
            target_id=target_id,
            target_type=target_type,
            operation=DeltaOperation.ADD,
            before=None,
            after=state,
            source=source,
            reason=reason,
            query_id=query_id,
        )
        return self._append_delta(delta)

    def record_update(
        self,
        target_id: str,
        target_type: DeltaTarget,
        before: Dict[str, Any],
        after: Dict[str, Any],
        source: str = "unknown",
        reason: Optional[str] = None,
        query_id: Optional[str] = None
    ) -> MemoryDelta:
        """
        Record an UPDATE operation.

        Args:
            target_id: ID of updated item
            target_type: Type of item
            before: State before update
            after: State after update
            source: What caused this update
            reason: Why it was updated
            query_id: Related query

        Returns:
            The recorded delta
        """
        delta = MemoryDelta(
            delta_id=self._generate_delta_id(),
            target_id=target_id,
            target_type=target_type,
            operation=DeltaOperation.UPDATE,
            before=before,
            after=after,
            source=source,
            reason=reason,
            query_id=query_id,
        )
        return self._append_delta(delta)

    def record_delete(
        self,
        target_id: str,
        target_type: DeltaTarget,
        before: Dict[str, Any],
        source: str = "unknown",
        reason: Optional[str] = None,
        query_id: Optional[str] = None
    ) -> MemoryDelta:
        """
        Record a DELETE operation.

        Args:
            target_id: ID of deleted item
            target_type: Type of item
            before: State before deletion
            source: What caused this delete
            reason: Why it was deleted
            query_id: Related query

        Returns:
            The recorded delta
        """
        delta = MemoryDelta(
            delta_id=self._generate_delta_id(),
            target_id=target_id,
            target_type=target_type,
            operation=DeltaOperation.DELETE,
            before=before,
            after=None,
            source=source,
            reason=reason,
            query_id=query_id,
        )
        return self._append_delta(delta)

    def checkpoint(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        embeddings: Optional[Dict[str, List[float]]] = None,
        reason: str = "manual"
    ) -> DeltaCheckpoint:
        """
        Create a checkpoint (full state snapshot).

        Args:
            nodes: All node states
            edges: All edge states
            embeddings: All embeddings (optional)
            reason: Why checkpoint was taken

        Returns:
            The created checkpoint
        """
        checkpoint = DeltaCheckpoint(
            checkpoint_id=f"cp_{int(time.time() * 1000)}_{len(self.checkpoints)}",
            timestamp=time.time(),
            nodes=nodes.copy(),
            edges=edges.copy(),
            embeddings=(embeddings or {}).copy(),
            delta_count_before=len(self.deltas),
            reason=reason,
        )

        self.checkpoints.append(checkpoint)
        self.last_checkpoint_time = time.time()

        # Prune old checkpoints
        self._prune_old_checkpoints()

        self.logger.info(
            f"Created checkpoint: {checkpoint.checkpoint_id}, "
            f"nodes={len(nodes)}, edges={len(edges)}"
        )

        return checkpoint

    def reconstruct_at(
        self,
        timestamp: float,
        target_ids: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct memory state at a specific point in time.

        Args:
            timestamp: Target timestamp
            target_ids: Optional set of specific IDs to reconstruct

        Returns:
            Reconstructed state with nodes, edges, embeddings
        """
        # Find nearest checkpoint before timestamp
        base_checkpoint = self._find_checkpoint_before(timestamp)

        if base_checkpoint is None:
            # No checkpoint - start from empty
            state = {
                "nodes": {},
                "edges": {},
                "embeddings": {},
            }
            start_idx = 0
        else:
            # Start from checkpoint
            state = {
                "nodes": base_checkpoint.nodes.copy(),
                "edges": base_checkpoint.edges.copy(),
                "embeddings": base_checkpoint.embeddings.copy(),
            }
            # Find deltas after checkpoint
            start_idx = self._find_delta_index_after(base_checkpoint.timestamp)

        # Apply deltas up to target timestamp
        for i in range(start_idx, len(self.deltas)):
            delta = self.deltas[i]

            if delta.timestamp > timestamp:
                break  # Stop at target time

            if target_ids and delta.target_id not in target_ids:
                continue  # Skip if filtering by IDs

            self._apply_delta(state, delta)

        return state

    def get_deltas_for_target(
        self,
        target_id: str,
        since: Optional[float] = None
    ) -> List[MemoryDelta]:
        """
        Get all deltas for a specific target.

        Args:
            target_id: The target ID
            since: Optional timestamp to filter from

        Returns:
            List of deltas for this target
        """
        indices = self._target_index.get(target_id, [])
        deltas = [self.deltas[i] for i in indices]

        if since is not None:
            deltas = [d for d in deltas if d.timestamp >= since]

        return deltas

    def get_deltas_since(self, timestamp: float) -> List[MemoryDelta]:
        """
        Get all deltas since a timestamp.

        Args:
            timestamp: Start timestamp

        Returns:
            List of deltas since timestamp
        """
        idx = self._find_delta_index_after(timestamp)
        return self.deltas[idx:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_delta_size = sum(d.size_bytes for d in self.deltas)
        total_checkpoint_size = sum(c.size_bytes for c in self.checkpoints)

        return {
            "total_deltas": len(self.deltas),
            "total_deltas_recorded": self.total_deltas_recorded,
            "total_checkpoints": len(self.checkpoints),
            "delta_size_bytes": total_delta_size,
            "checkpoint_size_bytes": total_checkpoint_size,
            "total_size_bytes": total_delta_size + total_checkpoint_size,
            "unique_targets": len(self._target_index),
            "oldest_delta": self.deltas[0].timestamp if self.deltas else None,
            "newest_delta": self.deltas[-1].timestamp if self.deltas else None,
            "last_checkpoint": self.checkpoints[-1].timestamp if self.checkpoints else None,
        }

    def _append_delta(self, delta: MemoryDelta) -> MemoryDelta:
        """Append delta to log and update indices."""
        idx = len(self.deltas)
        self.deltas.append(delta)

        # Update indices
        self._target_index[delta.target_id].append(idx)
        self._time_index.append(delta.timestamp)

        self.total_deltas_recorded += 1
        self._delta_counter += 1

        # Check if checkpoint needed
        self._maybe_checkpoint()

        # Prune if over limit
        self._prune_old_deltas()

        return delta

    def _generate_delta_id(self) -> str:
        """Generate unique delta ID."""
        return f"d_{int(time.time() * 1000)}_{self._delta_counter}"

    def _maybe_checkpoint(self):
        """Check if automatic checkpoint is needed."""
        deltas_since = len(self.deltas) - (
            self.checkpoints[-1].delta_count_before if self.checkpoints else 0
        )
        time_since = time.time() - self.last_checkpoint_time

        needs_checkpoint = (
            deltas_since >= self.config.checkpoint_interval_deltas or
            time_since >= self.config.checkpoint_interval_seconds or
            deltas_since >= self.config.max_deltas_before_checkpoint
        )

        if needs_checkpoint and deltas_since > 0:
            # Reconstruct current state for checkpoint
            current_state = self.reconstruct_at(time.time())
            self.checkpoint(
                nodes=current_state["nodes"],
                edges=current_state["edges"],
                embeddings=current_state["embeddings"],
                reason="automatic"
            )

    def _prune_old_checkpoints(self):
        """Remove old checkpoints beyond retention limit."""
        while len(self.checkpoints) > self.config.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            self.logger.debug(f"Pruned old checkpoint: {oldest.checkpoint_id}")

    def _prune_old_deltas(self):
        """Remove old deltas beyond retention limit."""
        if len(self.deltas) <= self.config.max_deltas:
            return

        # Find oldest checkpoint timestamp
        oldest_checkpoint_time = (
            self.checkpoints[0].timestamp if self.checkpoints else 0
        )

        if self.config.prune_before_oldest_checkpoint:
            # Remove deltas before oldest checkpoint
            cutoff_idx = self._find_delta_index_after(oldest_checkpoint_time)
            if cutoff_idx > 0:
                pruned = self.deltas[:cutoff_idx]
                self.deltas = self.deltas[cutoff_idx:]

                # Rebuild indices
                self._rebuild_indices()

                self.logger.debug(f"Pruned {len(pruned)} old deltas")

    def _rebuild_indices(self):
        """Rebuild all indices after pruning."""
        self._target_index.clear()
        self._time_index.clear()

        for i, delta in enumerate(self.deltas):
            self._target_index[delta.target_id].append(i)
            self._time_index.append(delta.timestamp)

    def _find_checkpoint_before(
        self,
        timestamp: float
    ) -> Optional[DeltaCheckpoint]:
        """Find the most recent checkpoint before timestamp."""
        best = None
        for cp in self.checkpoints:
            if cp.timestamp <= timestamp:
                best = cp
            else:
                break  # Checkpoints are ordered
        return best

    def _find_delta_index_after(self, timestamp: float) -> int:
        """Find index of first delta after timestamp using binary search."""
        if not self._time_index:
            return 0

        import bisect
        return bisect.bisect_right(self._time_index, timestamp)

    def _apply_delta(
        self,
        state: Dict[str, Any],
        delta: MemoryDelta
    ):
        """Apply a delta to state."""
        collection = self._get_collection(state, delta.target_type)

        if delta.operation == DeltaOperation.ADD:
            collection[delta.target_id] = delta.after
        elif delta.operation == DeltaOperation.UPDATE:
            collection[delta.target_id] = delta.after
        elif delta.operation == DeltaOperation.DELETE:
            collection.pop(delta.target_id, None)

    def _get_collection(
        self,
        state: Dict[str, Any],
        target_type: DeltaTarget
    ) -> Dict:
        """Get the appropriate collection for a target type."""
        if target_type == DeltaTarget.NODE:
            return state["nodes"]
        elif target_type == DeltaTarget.EDGE:
            return state["edges"]
        elif target_type == DeltaTarget.EMBEDDING:
            return state["embeddings"]
        else:
            return state.get("metadata", {})


# ============================================================================
# Delta Reconstructor
# ============================================================================

class DeltaReconstructor:
    """
    High-level API for reconstructing memory state from deltas.

    Provides convenient methods for:
    - Time-travel queries
    - Differential sync
    - Change detection
    """

    def __init__(self, store: DeltaStore):
        """
        Initialize reconstructor.

        Args:
            store: The delta store to use
        """
        self.store = store

    def get_state_at(self, timestamp: float) -> Dict[str, Any]:
        """
        Get memory state at a specific point in time.

        Args:
            timestamp: Target timestamp

        Returns:
            Full state at that time
        """
        return self.store.reconstruct_at(timestamp)

    def get_changes_between(
        self,
        start: float,
        end: float
    ) -> List[MemoryDelta]:
        """
        Get all changes between two timestamps.

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            List of deltas in the range
        """
        deltas = self.store.get_deltas_since(start)
        return [d for d in deltas if d.timestamp <= end]

    def get_target_history(
        self,
        target_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get complete history of a target (node/edge).

        Args:
            target_id: The target ID

        Returns:
            List of state snapshots over time
        """
        deltas = self.store.get_deltas_for_target(target_id)

        history = []
        current_state = None

        for delta in deltas:
            if delta.operation == DeltaOperation.ADD:
                current_state = delta.after.copy() if delta.after else {}
            elif delta.operation == DeltaOperation.UPDATE:
                current_state = delta.after.copy() if delta.after else current_state
            elif delta.operation == DeltaOperation.DELETE:
                current_state = None

            history.append({
                "timestamp": delta.timestamp,
                "operation": delta.operation.value,
                "state": current_state.copy() if current_state else None,
                "source": delta.source,
                "reason": delta.reason,
            })

        return history

    def diff_states(
        self,
        timestamp1: float,
        timestamp2: float
    ) -> Dict[str, Any]:
        """
        Get difference between two points in time.

        Args:
            timestamp1: First timestamp
            timestamp2: Second timestamp

        Returns:
            Diff showing added, updated, deleted items
        """
        state1 = self.get_state_at(timestamp1)
        state2 = self.get_state_at(timestamp2)

        diff = {
            "nodes": self._diff_collections(state1["nodes"], state2["nodes"]),
            "edges": self._diff_collections(state1["edges"], state2["edges"]),
            "embeddings": self._diff_collections(
                state1["embeddings"], state2["embeddings"]
            ),
        }

        return diff

    def _diff_collections(
        self,
        before: Dict,
        after: Dict
    ) -> Dict[str, Any]:
        """Diff two collections."""
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        return {
            "added": {k: after[k] for k in after_keys - before_keys},
            "deleted": {k: before[k] for k in before_keys - after_keys},
            "updated": {
                k: {"before": before[k], "after": after[k]}
                for k in before_keys & after_keys
                if before[k] != after[k]
            },
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_delta_store(
    checkpoint_interval: int = 1000,
    max_deltas: int = 100000
) -> DeltaStore:
    """
    Create a configured delta store.

    Args:
        checkpoint_interval: Deltas between checkpoints
        max_deltas: Maximum deltas to retain

    Returns:
        Configured store
    """
    config = DeltaStoreConfig(
        checkpoint_interval_deltas=checkpoint_interval,
        max_deltas=max_deltas,
    )
    return DeltaStore(config)


def create_reconstructor(
    store: Optional[DeltaStore] = None
) -> DeltaReconstructor:
    """
    Create a delta reconstructor.

    Args:
        store: Delta store (created if not provided)

    Returns:
        Configured reconstructor
    """
    if store is None:
        store = create_delta_store()
    return DeltaReconstructor(store)

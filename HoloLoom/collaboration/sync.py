"""
CRDT-Based Synchronization
===========================
Conflict-free Replicated Data Types for collaborative editing.

Philosophy:
When multiple shuttles weave the same fabric, conflicts are inevitable.
CRDTs ensure that no matter the order of operations, all looms converge
to the same final state - like quantum threads collapsing to coherence.

Implementation:
- Last-Write-Wins (LWW) Register for simple values
- Grow-Only Set (G-Set) for collections
- Version vectors for causality tracking
- Delta compression for efficiency
"""

import logging
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """CRDT operation types."""
    SET = "set"              # Set a value
    DELETE = "delete"        # Delete a value
    ADD = "add"              # Add to set/list
    REMOVE = "remove"        # Remove from set/list
    MERGE = "merge"          # Merge states


@dataclass
class StateVector:
    """
    Version vector for causality tracking.

    Each user has a counter that increments with each operation.
    Enables detection of concurrent vs. sequential operations.
    """
    counters: Dict[str, int] = field(default_factory=dict)

    def increment(self, user_id: str):
        """Increment counter for user."""
        self.counters[user_id] = self.counters.get(user_id, 0) + 1

    def get(self, user_id: str) -> int:
        """Get counter for user."""
        return self.counters.get(user_id, 0)

    def merge(self, other: 'StateVector'):
        """Merge with another vector (take max of each counter)."""
        for user_id, count in other.counters.items():
            self.counters[user_id] = max(self.counters.get(user_id, 0), count)

    def dominates(self, other: 'StateVector') -> bool:
        """Check if this vector dominates (is ahead of) another."""
        for user_id, count in other.counters.items():
            if self.counters.get(user_id, 0) < count:
                return False
        return True

    def concurrent_with(self, other: 'StateVector') -> bool:
        """Check if vectors are concurrent (neither dominates)."""
        return not self.dominates(other) and not other.dominates(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"counters": self.counters}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'StateVector':
        """Create from dictionary."""
        return StateVector(counters=data.get("counters", {}))

    def __str__(self) -> str:
        return f"StateVector({self.counters})"


@dataclass
class Delta:
    """
    Delta/operation for CRDT.

    Represents a single change to the shared state.
    """
    operation: OperationType
    path: str                        # Path to the value (e.g., "conversation.messages.0")
    value: Any = None                # New value
    user_id: str = ""                # User who made the change
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    vector: Optional[StateVector] = None  # Version vector at time of operation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation.value,
            "path": self.path,
            "value": self.value,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "vector": self.vector.to_dict() if self.vector else None,
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Delta':
        """Create from dictionary."""
        vector = None
        if data.get("vector"):
            vector = StateVector.from_dict(data["vector"])

        return Delta(
            operation=OperationType(data["operation"]),
            path=data["path"],
            value=data.get("value"),
            user_id=data["user_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            vector=vector,
            metadata=data.get("metadata", {})
        )

    def hash(self) -> str:
        """Compute hash of delta for deduplication."""
        content = f"{self.operation}:{self.path}:{self.value}:{self.user_id}:{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class LWWRegister:
    """
    Last-Write-Wins Register.

    Resolves conflicts by keeping the value with the latest timestamp.
    If timestamps are equal, uses user_id as tiebreaker.
    """

    def __init__(self):
        self.value: Any = None
        self.timestamp: Optional[datetime] = None
        self.user_id: Optional[str] = None

    def set(self, value: Any, timestamp: datetime, user_id: str):
        """Set value if timestamp is newer."""
        if self.timestamp is None:
            self.value = value
            self.timestamp = timestamp
            self.user_id = user_id
            return True

        # Compare timestamps
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp
            self.user_id = user_id
            return True
        elif timestamp == self.timestamp:
            # Tiebreak by user_id (deterministic)
            if user_id > self.user_id:
                self.value = value
                self.timestamp = timestamp
                self.user_id = user_id
                return True

        return False

    def get(self) -> Any:
        """Get current value."""
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id
        }


class GSet:
    """
    Grow-Only Set.

    Elements can only be added, never removed.
    Merge is simple union.
    """

    def __init__(self):
        self.elements: Set[str] = set()

    def add(self, element: str):
        """Add element to set."""
        self.elements.add(element)

    def contains(self, element: str) -> bool:
        """Check if element in set."""
        return element in self.elements

    def merge(self, other: 'GSet'):
        """Merge with another G-Set."""
        self.elements.update(other.elements)

    def to_list(self) -> List[str]:
        """Convert to list."""
        return sorted(list(self.elements))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"elements": list(self.elements)}


class CRDTEngine:
    """
    CRDT-based state synchronization engine.

    Manages shared state across multiple users with conflict-free merging.

    Features:
    - LWW registers for simple values
    - G-Sets for collections
    - Version vectors for causality
    - Delta compression
    - State reconciliation
    """

    def __init__(self, user_id: str):
        """
        Initialize CRDT engine.

        Args:
            user_id: ID of local user
        """
        self.user_id = user_id
        self.state: Dict[str, LWWRegister] = {}
        self.sets: Dict[str, GSet] = {}
        self.vector = StateVector()
        self.pending_deltas: List[Delta] = []
        self.applied_deltas: Set[str] = set()  # Hashes of applied deltas

        logger.info(f"CRDTEngine initialized for user {user_id}")

    def set_value(self, path: str, value: Any) -> Delta:
        """
        Set a value at path.

        Args:
            path: Dot-separated path (e.g., "conversation.topic")
            value: New value

        Returns:
            Delta representing the change
        """
        # Increment version vector
        self.vector.increment(self.user_id)

        # Create delta
        timestamp = datetime.now(timezone.utc)
        delta = Delta(
            operation=OperationType.SET,
            path=path,
            value=value,
            user_id=self.user_id,
            timestamp=timestamp,
            vector=StateVector(counters=self.vector.counters.copy())
        )

        # Apply locally
        self._apply_set(path, value, timestamp, self.user_id)

        # Track delta
        self.pending_deltas.append(delta)
        self.applied_deltas.add(delta.hash())

        logger.debug(f"Set {path}={value} by {self.user_id}")
        return delta

    def add_to_set(self, path: str, element: str) -> Delta:
        """
        Add element to a set.

        Args:
            path: Path to the set
            element: Element to add

        Returns:
            Delta representing the change
        """
        self.vector.increment(self.user_id)

        timestamp = datetime.now(timezone.utc)
        delta = Delta(
            operation=OperationType.ADD,
            path=path,
            value=element,
            user_id=self.user_id,
            timestamp=timestamp,
            vector=StateVector(counters=self.vector.counters.copy())
        )

        # Apply locally
        self._apply_add(path, element)

        self.pending_deltas.append(delta)
        self.applied_deltas.add(delta.hash())

        logger.debug(f"Added {element} to {path} by {self.user_id}")
        return delta

    def _apply_set(self, path: str, value: Any, timestamp: datetime, user_id: str):
        """Apply SET operation."""
        if path not in self.state:
            self.state[path] = LWWRegister()

        self.state[path].set(value, timestamp, user_id)

    def _apply_add(self, path: str, element: str):
        """Apply ADD operation."""
        if path not in self.sets:
            self.sets[path] = GSet()

        self.sets[path].add(element)

    def apply_delta(self, delta: Delta) -> bool:
        """
        Apply remote delta.

        Args:
            delta: Delta to apply

        Returns:
            True if delta was applied, False if already seen
        """
        # Check if already applied
        delta_hash = delta.hash()
        if delta_hash in self.applied_deltas:
            logger.debug(f"Delta {delta_hash} already applied, skipping")
            return False

        # Update version vector
        if delta.vector:
            self.vector.merge(delta.vector)

        # Apply based on operation type
        if delta.operation == OperationType.SET:
            self._apply_set(delta.path, delta.value, delta.timestamp, delta.user_id)
        elif delta.operation == OperationType.ADD:
            self._apply_add(delta.path, delta.value)
        else:
            logger.warning(f"Unknown operation type: {delta.operation}")
            return False

        # Track delta
        self.applied_deltas.add(delta_hash)

        logger.debug(f"Applied delta {delta_hash} from {delta.user_id}")
        return True

    def apply_deltas(self, deltas: List[Delta]) -> int:
        """
        Apply multiple deltas.

        Args:
            deltas: List of deltas to apply

        Returns:
            Number of deltas successfully applied
        """
        applied = 0
        for delta in deltas:
            if self.apply_delta(delta):
                applied += 1

        logger.info(f"Applied {applied}/{len(deltas)} deltas")
        return applied

    def get_value(self, path: str, default: Any = None) -> Any:
        """Get value at path."""
        if path in self.state:
            return self.state[path].get()
        return default

    def get_set(self, path: str) -> List[str]:
        """Get set at path as list."""
        if path in self.sets:
            return self.sets[path].to_list()
        return []

    def get_pending_deltas(self, clear: bool = True) -> List[Delta]:
        """
        Get pending deltas for synchronization.

        Args:
            clear: Whether to clear pending deltas after retrieval

        Returns:
            List of pending deltas
        """
        deltas = self.pending_deltas.copy()

        if clear:
            self.pending_deltas.clear()

        return deltas

    def merge_state(self, other_state: Dict[str, Any], other_vector: StateVector):
        """
        Merge with remote state.

        Args:
            other_state: Remote state dictionary
            other_vector: Remote version vector
        """
        # Merge version vectors
        self.vector.merge(other_vector)

        # Merge LWW registers
        for path, reg_dict in other_state.get("registers", {}).items():
            if path not in self.state:
                self.state[path] = LWWRegister()

            timestamp = datetime.fromisoformat(reg_dict["timestamp"])
            self.state[path].set(
                reg_dict["value"],
                timestamp,
                reg_dict["user_id"]
            )

        # Merge G-Sets
        for path, set_dict in other_state.get("sets", {}).items():
            if path not in self.sets:
                self.sets[path] = GSet()

            remote_set = GSet()
            remote_set.elements = set(set_dict["elements"])
            self.sets[path].merge(remote_set)

        logger.info("Merged remote state")

    def get_full_state(self) -> Dict[str, Any]:
        """Get full state for synchronization."""
        return {
            "registers": {
                path: reg.to_dict()
                for path, reg in self.state.items()
            },
            "sets": {
                path: s.to_dict()
                for path, s in self.sets.items()
            },
            "vector": self.vector.to_dict(),
            "user_id": self.user_id
        }

    def compress_deltas(self, deltas: List[Delta]) -> List[Delta]:
        """
        Compress deltas by removing redundant operations.

        Args:
            deltas: List of deltas to compress

        Returns:
            Compressed list of deltas
        """
        # Group by path
        by_path: Dict[str, List[Delta]] = defaultdict(list)
        for delta in deltas:
            by_path[delta.path].append(delta)

        # For each path, keep only the latest SET operation
        compressed = []
        for path, path_deltas in by_path.items():
            set_ops = [d for d in path_deltas if d.operation == OperationType.SET]
            add_ops = [d for d in path_deltas if d.operation == OperationType.ADD]

            # Keep latest SET
            if set_ops:
                latest_set = max(set_ops, key=lambda d: d.timestamp)
                compressed.append(latest_set)

            # Keep all ADDs (G-Set)
            compressed.extend(add_ops)

        logger.debug(f"Compressed {len(deltas)} deltas to {len(compressed)}")
        return compressed

    def reset(self):
        """Reset engine state (for testing)."""
        self.state.clear()
        self.sets.clear()
        self.vector = StateVector()
        self.pending_deltas.clear()
        self.applied_deltas.clear()
        logger.info("CRDT engine reset")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("CRDT Synchronization Engine")
    print("="*80 + "\n")

    # Create two engines (simulating two users)
    print("1. Creating CRDT engines...")
    alice_engine = CRDTEngine(user_id="alice")
    bob_engine = CRDTEngine(user_id="bob")
    print("   ✓ Alice and Bob engines created")

    # Alice sets values
    print("\n2. Alice sets values...")
    alice_engine.set_value("topic", "Neural Decision Making")
    alice_engine.set_value("temperature", 0.7)
    alice_engine.add_to_set("participants", "alice")
    print("   ✓ Alice: topic, temperature, participants")

    # Bob sets values (concurrent)
    print("\n3. Bob sets values (concurrent)...")
    bob_engine.set_value("topic", "Collaborative AI")  # Conflict!
    bob_engine.set_value("model", "gpt-4")
    bob_engine.add_to_set("participants", "bob")
    print("   ✓ Bob: topic (conflict!), model, participants")

    # Synchronize: Alice -> Bob
    print("\n4. Synchronizing Alice -> Bob...")
    alice_deltas = alice_engine.get_pending_deltas()
    bob_engine.apply_deltas(alice_deltas)
    print(f"   ✓ Applied {len(alice_deltas)} deltas from Alice")

    # Synchronize: Bob -> Alice
    print("\n5. Synchronizing Bob -> Alice...")
    bob_deltas = bob_engine.get_pending_deltas()
    alice_engine.apply_deltas(bob_deltas)
    print(f"   ✓ Applied {len(bob_deltas)} deltas from Bob")

    # Check convergence
    print("\n6. Checking convergence...")
    alice_topic = alice_engine.get_value("topic")
    bob_topic = bob_engine.get_value("topic")
    print(f"   • Alice topic: {alice_topic}")
    print(f"   • Bob topic: {bob_topic}")
    print(f"   • Converged: {alice_topic == bob_topic} ✓")

    # Check sets
    print("\n7. Checking participant sets...")
    alice_participants = alice_engine.get_set("participants")
    bob_participants = bob_engine.get_set("participants")
    print(f"   • Alice participants: {alice_participants}")
    print(f"   • Bob participants: {bob_participants}")
    print(f"   • Converged: {alice_participants == bob_participants} ✓")

    # Full state sync
    print("\n8. Full state synchronization...")
    alice_state = alice_engine.get_full_state()
    charlie_engine = CRDTEngine(user_id="charlie")
    charlie_engine.merge_state(alice_state, alice_engine.vector)
    print(f"   ✓ Charlie synced with Alice's state")
    print(f"   • Charlie topic: {charlie_engine.get_value('topic')}")

    print("\n✓ CRDT synchronization ready!")

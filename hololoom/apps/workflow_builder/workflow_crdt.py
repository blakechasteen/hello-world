"""Operation-based CRDT for workflow collaboration.

Implements Conflict-free Replicated Data Types with:
- Vector clocks for causality tracking
- Operation-based replication
- Last-writer-wins conflict resolution
- Tombstone-based deletion

Created: 2025-12-23
Part of HoloLoom Workflow Builder production integration.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperationType(Enum):
    """Types of CRDT operations."""
    ADD_NODE = "add_node"
    UPDATE_NODE = "update_node"
    DELETE_NODE = "delete_node"
    MOVE_NODE = "move_node"
    ADD_CONNECTION = "add_connection"
    DELETE_CONNECTION = "delete_connection"
    UPDATE_METADATA = "update_metadata"


@dataclass
class VectorClock:
    """Vector clock for causality tracking.

    Each replica maintains a logical clock. The vector contains
    the latest known clock value for each replica.
    """
    clocks: dict[str, int] = field(default_factory=dict)

    def increment(self, replica_id: str) -> 'VectorClock':
        """Increment clock for a replica and return new VectorClock."""
        new_clocks = self.clocks.copy()
        new_clocks[replica_id] = new_clocks.get(replica_id, 0) + 1
        return VectorClock(clocks=new_clocks)

    def merge(self, other: 'VectorClock') -> 'VectorClock':
        """Merge with another vector clock (point-wise maximum)."""
        all_keys = set(self.clocks) | set(other.clocks)
        merged = {
            k: max(self.clocks.get(k, 0), other.clocks.get(k, 0))
            for k in all_keys
        }
        return VectorClock(clocks=merged)

    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if self causally precedes other (self < other).

        Returns True if:
        - All clocks in self are <= corresponding clocks in other
        - At least one clock in self is < corresponding clock in other
        """
        for replica, count in self.clocks.items():
            if count > other.clocks.get(replica, 0):
                return False

        # Check at least one is strictly less
        return any(
            self.clocks.get(r, 0) < other.clocks.get(r, 0)
            for r in other.clocks
        )

    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if neither causally precedes the other."""
        return not self.happens_before(other) and not other.happens_before(self)

    def dominates(self, other: 'VectorClock') -> bool:
        """Check if self >= other (self happened after or concurrent)."""
        for replica, count in other.clocks.items():
            if self.clocks.get(replica, 0) < count:
                return False
        return True

    def __eq__(self, other) -> bool:
        if not isinstance(other, VectorClock):
            return False
        return self.clocks == other.clocks

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.clocks.items())))

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for serialization."""
        return self.clocks.copy()

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> 'VectorClock':
        """Create from dictionary."""
        return cls(clocks=d.copy())


@dataclass
class Operation:
    """A single CRDT operation.

    Operations are the unit of replication. Each operation
    includes enough information to be applied idempotently
    and in any order (after causal dependencies are met).
    """
    id: str
    type: OperationType
    target_id: str  # Node or connection ID
    data: dict[str, Any]
    timestamp: float  # Wall clock for tie-breaking
    replica_id: str
    vector_clock: VectorClock

    def to_dict(self) -> dict:
        """Serialize operation to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "target_id": self.target_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "replica_id": self.replica_id,
            "vector_clock": self.vector_clock.to_dict()
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Operation':
        """Deserialize operation from dictionary."""
        return cls(
            id=d["id"],
            type=OperationType(d["type"]),
            target_id=d["target_id"],
            data=d.get("data", {}),
            timestamp=d["timestamp"],
            replica_id=d["replica_id"],
            vector_clock=VectorClock.from_dict(d["vector_clock"])
        )


class ConflictResolver:
    """Resolve conflicts between concurrent operations.

    Implements multiple conflict resolution strategies:
    - Last-Writer-Wins (LWW) with replica_id tiebreaker
    - Delete-Wins for delete vs update conflicts
    """

    @staticmethod
    def resolve_concurrent_updates(ops: list[Operation]) -> Operation:
        """Resolve concurrent updates using Last-Writer-Wins.

        Uses (timestamp, replica_id) for total ordering.
        """
        if not ops:
            raise ValueError("No operations to resolve")
        if len(ops) == 1:
            return ops[0]

        # Sort by (timestamp, replica_id) for deterministic ordering
        ops_sorted = sorted(ops, key=lambda o: (o.timestamp, o.replica_id))
        return ops_sorted[-1]

    @staticmethod
    def resolve_delete_vs_update(
        delete_op: Operation,
        update_op: Operation
    ) -> Operation:
        """Resolve delete vs update conflict.

        Delete always wins over concurrent update (prevents resurrection).
        """
        return delete_op

    @staticmethod
    def merge_node_data(existing: dict, updates: dict) -> dict:
        """Merge node data with field-level LWW.

        For concurrent field updates, use timestamp.
        """
        result = existing.copy()
        result.update(updates)
        return result


@dataclass
class NodeState:
    """Internal state for a node."""
    data: dict[str, Any]
    created_at: float
    updated_at: float
    created_by: str


@dataclass
class ConnectionState:
    """Internal state for a connection."""
    data: dict[str, Any]
    created_at: float
    created_by: str


class WorkflowCRDTState:
    """CRDT state for a workflow.

    Maintains the current state of a workflow with support for
    concurrent editing from multiple replicas.

    Features:
    - Operation-based replication
    - Vector clock causality tracking
    - Tombstone-based deletion
    - Conflict resolution
    - Operation log for sync
    """

    def __init__(self, workflow_id: str, replica_id: str):
        """Initialize CRDT state.

        Args:
            workflow_id: Unique identifier for the workflow
            replica_id: Unique identifier for this replica
        """
        self.workflow_id = workflow_id
        self.replica_id = replica_id
        self.vector_clock = VectorClock()

        # Current state
        self.nodes: dict[str, NodeState] = {}
        self.connections: dict[str, ConnectionState] = {}
        self.metadata: dict[str, Any] = {}

        # Tombstones (deleted items - never removed)
        self.deleted_nodes: set[str] = set()
        self.deleted_connections: set[str] = set()

        # Operation log
        self.operations: list[Operation] = []
        self.operation_ids: set[str] = set()  # For fast duplicate check
        self.pending_ops: list[Operation] = []  # Unsynced operations

        # Buffered operations (waiting for causal dependencies)
        self.buffered_ops: list[Operation] = []

    def create_operation(
        self,
        op_type: OperationType,
        target_id: str,
        data: dict[str, Any]
    ) -> Operation:
        """Create and apply a new operation from this replica.

        Args:
            op_type: Type of operation
            target_id: ID of target node/connection
            data: Operation data

        Returns:
            Created Operation (already applied)
        """
        self.vector_clock = self.vector_clock.increment(self.replica_id)

        op = Operation(
            id=str(uuid.uuid4()),
            type=op_type,
            target_id=target_id,
            data=data,
            timestamp=time.time(),
            replica_id=self.replica_id,
            vector_clock=VectorClock(clocks=self.vector_clock.clocks.copy())
        )

        self._apply_operation_internal(op)
        self.pending_ops.append(op)
        return op

    def apply_operation(self, op: Operation) -> bool:
        """Apply an operation to current state.

        Handles causality checking and buffering.

        Args:
            op: Operation to apply

        Returns:
            True if applied, False if buffered or duplicate
        """
        # Check for duplicate
        if op.id in self.operation_ids:
            return False

        # Check causal dependencies
        if not self._causal_ready(op):
            self.buffered_ops.append(op)
            return False

        # Apply the operation
        self._apply_operation_internal(op)

        # Try to apply buffered operations
        self._process_buffered()

        return True

    def _causal_ready(self, op: Operation) -> bool:
        """Check if operation's causal dependencies are satisfied."""
        # An operation is ready if our vector clock dominates
        # the operation's clock minus the sender's contribution
        for replica, count in op.vector_clock.clocks.items():
            if replica == op.replica_id:
                # Sender's clock can be ahead by 1
                if self.vector_clock.clocks.get(replica, 0) < count - 1:
                    return False
            else:
                # Other replicas must be caught up
                if self.vector_clock.clocks.get(replica, 0) < count:
                    return False
        return True

    def _process_buffered(self):
        """Try to apply buffered operations."""
        changed = True
        while changed and self.buffered_ops:
            changed = False
            remaining = []
            for op in self.buffered_ops:
                if self._causal_ready(op):
                    self._apply_operation_internal(op)
                    changed = True
                else:
                    remaining.append(op)
            self.buffered_ops = remaining

    def _apply_operation_internal(self, op: Operation):
        """Internal: Apply operation without causality check."""
        # Record operation
        self.operations.append(op)
        self.operation_ids.add(op.id)

        # Update vector clock
        self.vector_clock = self.vector_clock.merge(op.vector_clock)

        # Apply based on type
        if op.type == OperationType.ADD_NODE:
            if op.target_id not in self.deleted_nodes:
                self.nodes[op.target_id] = NodeState(
                    data=op.data.copy(),
                    created_at=op.timestamp,
                    updated_at=op.timestamp,
                    created_by=op.replica_id
                )

        elif op.type == OperationType.UPDATE_NODE:
            if op.target_id in self.nodes and op.target_id not in self.deleted_nodes:
                node = self.nodes[op.target_id]
                node.data = ConflictResolver.merge_node_data(node.data, op.data)
                node.updated_at = op.timestamp

        elif op.type == OperationType.MOVE_NODE:
            if op.target_id in self.nodes and op.target_id not in self.deleted_nodes:
                node = self.nodes[op.target_id]
                if "x" in op.data:
                    node.data["x"] = op.data["x"]
                if "y" in op.data:
                    node.data["y"] = op.data["y"]
                node.updated_at = op.timestamp

        elif op.type == OperationType.DELETE_NODE:
            self.nodes.pop(op.target_id, None)
            self.deleted_nodes.add(op.target_id)

            # Cascade delete connections involving this node
            to_delete = [
                cid for cid, conn in self.connections.items()
                if (conn.data.get("source_node") == op.target_id or
                    conn.data.get("target_node") == op.target_id or
                    conn.data.get("from_node") == op.target_id or
                    conn.data.get("to_node") == op.target_id)
            ]
            for cid in to_delete:
                self.connections.pop(cid, None)
                self.deleted_connections.add(cid)

        elif op.type == OperationType.ADD_CONNECTION:
            if op.target_id not in self.deleted_connections:
                # Validate nodes exist and aren't deleted
                source = op.data.get("source_node") or op.data.get("from_node")
                target = op.data.get("target_node") or op.data.get("to_node")

                if (source not in self.deleted_nodes and
                    target not in self.deleted_nodes):
                    self.connections[op.target_id] = ConnectionState(
                        data=op.data.copy(),
                        created_at=op.timestamp,
                        created_by=op.replica_id
                    )

        elif op.type == OperationType.DELETE_CONNECTION:
            self.connections.pop(op.target_id, None)
            self.deleted_connections.add(op.target_id)

        elif op.type == OperationType.UPDATE_METADATA:
            self.metadata.update(op.data)

    def merge_remote(self, remote_ops: list[Operation]) -> list[str]:
        """Merge operations from remote replica.

        Args:
            remote_ops: List of operations from remote replica

        Returns:
            List of operation IDs that were applied
        """
        applied = []

        # Sort by causal order (vector clock sum, then timestamp)
        remote_ops_sorted = sorted(
            remote_ops,
            key=lambda o: (sum(o.vector_clock.clocks.values()), o.timestamp)
        )

        for op in remote_ops_sorted:
            if self.apply_operation(op):
                applied.append(op.id)

        return applied

    def get_pending_ops(self) -> list[Operation]:
        """Get operations that need to be synced to other replicas.

        Returns and clears the pending operations list.
        """
        pending = self.pending_ops.copy()
        self.pending_ops.clear()
        return pending

    def get_ops_since(self, since_clock: VectorClock) -> list[Operation]:
        """Get all operations since a given vector clock.

        Useful for sync: "give me everything I don't have".
        """
        return [
            op for op in self.operations
            if not since_clock.dominates(op.vector_clock)
        ]

    # ==================== Convenience Methods ====================

    def add_node(self, node_id: str, node_data: dict[str, Any]) -> Operation:
        """Add a new node."""
        return self.create_operation(OperationType.ADD_NODE, node_id, node_data)

    def update_node(self, node_id: str, updates: dict[str, Any]) -> Operation:
        """Update an existing node."""
        return self.create_operation(OperationType.UPDATE_NODE, node_id, updates)

    def move_node(self, node_id: str, x: float, y: float) -> Operation:
        """Move a node to new position."""
        return self.create_operation(OperationType.MOVE_NODE, node_id, {"x": x, "y": y})

    def delete_node(self, node_id: str) -> Operation:
        """Delete a node and its connections."""
        return self.create_operation(OperationType.DELETE_NODE, node_id, {})

    def add_connection(
        self,
        conn_id: str,
        source_node: str,
        target_node: str,
        **kwargs
    ) -> Operation:
        """Add a new connection."""
        data = {
            "source_node": source_node,
            "target_node": target_node,
            **kwargs
        }
        return self.create_operation(OperationType.ADD_CONNECTION, conn_id, data)

    def delete_connection(self, conn_id: str) -> Operation:
        """Delete a connection."""
        return self.create_operation(OperationType.DELETE_CONNECTION, conn_id, {})

    def update_metadata(self, updates: dict[str, Any]) -> Operation:
        """Update workflow metadata."""
        return self.create_operation(OperationType.UPDATE_METADATA, "", updates)

    # ==================== State Export ====================

    def to_workflow_dict(self) -> dict:
        """Convert current state to workflow dictionary format.

        Returns format compatible with Workflow model.
        """
        return {
            "id": self.workflow_id,
            "nodes": [
                {"id": nid, **node.data}
                for nid, node in self.nodes.items()
            ],
            "connections": [
                {"id": cid, **conn.data}
                for cid, conn in self.connections.items()
            ],
            "metadata": self.metadata.copy()
        }

    def to_sync_state(self) -> dict:
        """Export state for synchronization.

        Includes vector clock and operation log info.
        """
        return {
            "workflow_id": self.workflow_id,
            "replica_id": self.replica_id,
            "vector_clock": self.vector_clock.to_dict(),
            "operation_count": len(self.operations),
            "pending_count": len(self.pending_ops),
            "buffered_count": len(self.buffered_ops),
            "nodes": self.to_workflow_dict()["nodes"],
            "connections": self.to_workflow_dict()["connections"],
            "deleted_nodes": list(self.deleted_nodes),
            "deleted_connections": list(self.deleted_connections)
        }

    def get_stats(self) -> dict:
        """Get CRDT state statistics."""
        return {
            "node_count": len(self.nodes),
            "connection_count": len(self.connections),
            "deleted_node_count": len(self.deleted_nodes),
            "deleted_connection_count": len(self.deleted_connections),
            "operation_count": len(self.operations),
            "pending_count": len(self.pending_ops),
            "buffered_count": len(self.buffered_ops),
            "vector_clock": self.vector_clock.to_dict()
        }


# ==================== CRDT State Manager ====================

class CRDTStateManager:
    """Manage CRDT states for multiple workflows.

    Provides a central registry for workflow CRDT states with
    factory methods for creating and accessing states.
    """

    def __init__(self, server_replica_id: str = "server"):
        """Initialize manager.

        Args:
            server_replica_id: Replica ID for server-side states
        """
        self.server_replica_id = server_replica_id
        self._states: dict[str, WorkflowCRDTState] = {}

    def get_or_create(self, workflow_id: str) -> WorkflowCRDTState:
        """Get or create CRDT state for a workflow."""
        if workflow_id not in self._states:
            self._states[workflow_id] = WorkflowCRDTState(
                workflow_id,
                self.server_replica_id
            )
        return self._states[workflow_id]

    def get(self, workflow_id: str) -> WorkflowCRDTState | None:
        """Get CRDT state for a workflow if it exists."""
        return self._states.get(workflow_id)

    def remove(self, workflow_id: str) -> bool:
        """Remove CRDT state for a workflow."""
        if workflow_id in self._states:
            del self._states[workflow_id]
            return True
        return False

    def list_workflows(self) -> list[str]:
        """List all workflow IDs with active CRDT states."""
        return list(self._states.keys())

    def get_all_stats(self) -> dict[str, dict]:
        """Get stats for all managed workflows."""
        return {
            wid: state.get_stats()
            for wid, state in self._states.items()
        }


# Global CRDT state manager
_crdt_manager = CRDTStateManager()


def get_crdt_manager() -> CRDTStateManager:
    """Get the global CRDT state manager."""
    return _crdt_manager

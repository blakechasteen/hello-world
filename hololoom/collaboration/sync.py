"""
State Synchronization for Multi-User Collaboration.

Implements CRDT-inspired conflict resolution for real-time collaborative editing.

Phase 3: Multi-User Collaboration - State sync infrastructure.

Created: November 2025
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be synchronized."""
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    MOVE = "move"
    CREATE = "create"
    DESTROY = "destroy"
    BATCH = "batch"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LAST_WRITER_WINS = "last_writer_wins"
    FIRST_WRITER_WINS = "first_writer_wins"
    MERGE = "merge"
    MANUAL = "manual"


@dataclass
class Operation:
    """A single synchronized operation."""
    op_id: str
    op_type: OperationType
    target_type: str  # "whiteboard", "stroke", "node", "text", etc.
    target_id: str
    user_id: str
    timestamp: datetime
    data: dict[str, Any]

    # For ordering (Lamport timestamp-like)
    sequence_number: int = 0
    vector_clock: dict[str, int] = field(default_factory=dict)

    # Parent operation for causality
    parent_ops: list[str] = field(default_factory=list)

    # State
    applied: bool = False
    confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "op_type": self.op_type.value,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "sequence_number": self.sequence_number,
            "vector_clock": self.vector_clock,
            "parent_ops": self.parent_ops,
            "applied": self.applied,
            "confirmed": self.confirmed
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Operation':
        return cls(
            op_id=data["op_id"],
            op_type=OperationType(data["op_type"]),
            target_type=data["target_type"],
            target_id=data["target_id"],
            user_id=data["user_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
            sequence_number=data.get("sequence_number", 0),
            vector_clock=data.get("vector_clock", {}),
            parent_ops=data.get("parent_ops", []),
            applied=data.get("applied", False),
            confirmed=data.get("confirmed", False)
        )


@dataclass
class Conflict:
    """A detected conflict between operations."""
    conflict_id: str
    operation_a: Operation
    operation_b: Operation
    target_id: str
    detected_at: datetime = field(default_factory=datetime.now)
    resolution: str | None = None
    resolved_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "operation_a": self.operation_a.to_dict(),
            "operation_b": self.operation_b.to_dict(),
            "target_id": self.target_id,
            "detected_at": self.detected_at.isoformat(),
            "resolution": self.resolution,
            "resolved_by": self.resolved_by
        }


@dataclass
class SyncState:
    """Synchronization state for an object."""
    target_id: str
    target_type: str
    version: int = 0
    last_modified: datetime = field(default_factory=datetime.now)
    last_modifier: str | None = None
    checksum: str | None = None
    locked_by: str | None = None
    lock_expires: datetime | None = None

    def increment_version(self, user_id: str):
        """Increment version after modification."""
        self.version += 1
        self.last_modified = datetime.now()
        self.last_modifier = user_id

    def is_locked(self) -> bool:
        """Check if object is locked."""
        if not self.locked_by:
            return False
        if self.lock_expires and datetime.now() > self.lock_expires:
            self.locked_by = None
            self.lock_expires = None
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "version": self.version,
            "last_modified": self.last_modified.isoformat(),
            "last_modifier": self.last_modifier,
            "checksum": self.checksum,
            "locked_by": self.locked_by,
            "lock_expires": self.lock_expires.isoformat() if self.lock_expires else None
        }


class OperationBuffer:
    """Buffer for pending operations awaiting confirmation."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.pending: dict[str, Operation] = {}
        self.confirmed: list[str] = []

    def add(self, op: Operation) -> bool:
        """Add operation to buffer."""
        if len(self.pending) >= self.max_size:
            # Remove oldest
            if self.pending:
                oldest = min(self.pending.values(), key=lambda o: o.timestamp)
                del self.pending[oldest.op_id]

        self.pending[op.op_id] = op
        return True

    def confirm(self, op_id: str) -> Operation | None:
        """Mark operation as confirmed."""
        op = self.pending.pop(op_id, None)
        if op:
            op.confirmed = True
            self.confirmed.append(op_id)
        return op

    def reject(self, op_id: str) -> Operation | None:
        """Remove rejected operation."""
        return self.pending.pop(op_id, None)

    def get_pending_for_target(self, target_id: str) -> list[Operation]:
        """Get pending operations for a target."""
        return [
            op for op in self.pending.values()
            if op.target_id == target_id
        ]


class StateSynchronizer:
    """
    Manages real-time state synchronization across multiple clients.

    Implements:
    - Operation-based synchronization
    - Vector clock ordering
    - Conflict detection and resolution
    - Optimistic local apply with rollback
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        resolution_strategy: ConflictResolution = ConflictResolution.LAST_WRITER_WINS
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.resolution_strategy = resolution_strategy

        # Operation management
        self.sequence_number = 0
        self.vector_clock: dict[str, int] = {user_id: 0}
        self.operation_log: list[Operation] = []
        self.buffer = OperationBuffer()

        # State tracking
        self.sync_states: dict[str, SyncState] = {}
        self.conflicts: dict[str, Conflict] = {}

        # Event handlers
        self._event_handlers: dict[str, list[Callable]] = {}
        self._apply_handlers: dict[str, Callable] = {}

    def register_apply_handler(
        self,
        target_type: str,
        handler: Callable[[Operation], bool]
    ):
        """Register handler for applying operations of a specific type."""
        self._apply_handlers[target_type] = handler

    def create_operation(
        self,
        op_type: OperationType,
        target_type: str,
        target_id: str,
        data: dict[str, Any],
        parent_ops: list[str] | None = None
    ) -> Operation:
        """Create a new operation."""
        self.sequence_number += 1
        self.vector_clock[self.user_id] = self.sequence_number

        op = Operation(
            op_id=f"op_{uuid4().hex[:12]}",
            op_type=op_type,
            target_type=target_type,
            target_id=target_id,
            user_id=self.user_id,
            timestamp=datetime.now(),
            data=data,
            sequence_number=self.sequence_number,
            vector_clock=dict(self.vector_clock),
            parent_ops=parent_ops or []
        )

        return op

    async def apply_local(self, op: Operation) -> bool:
        """
        Apply operation locally (optimistic).

        Returns True if applied successfully.
        """
        # Check if target is locked by another user
        state = self.sync_states.get(op.target_id)
        if state and state.is_locked() and state.locked_by != self.user_id:
            logger.warning(f"Target {op.target_id} is locked by {state.locked_by}")
            return False

        # Apply using registered handler
        handler = self._apply_handlers.get(op.target_type)
        if handler:
            try:
                success = handler(op)
                if success:
                    op.applied = True
                    self.buffer.add(op)
                    self.operation_log.append(op)

                    # Update state
                    if op.target_id not in self.sync_states:
                        self.sync_states[op.target_id] = SyncState(
                            target_id=op.target_id,
                            target_type=op.target_type
                        )
                    self.sync_states[op.target_id].increment_version(op.user_id)

                    self._emit_event("operation_applied", {"operation": op.to_dict()})
                    return True
            except Exception as e:
                logger.error(f"Apply handler error: {e}")

        return False

    async def receive_remote(self, op: Operation) -> bool:
        """
        Receive and apply a remote operation.

        Handles conflict detection and resolution.
        """
        # Update vector clock
        for user_id, seq in op.vector_clock.items():
            self.vector_clock[user_id] = max(
                self.vector_clock.get(user_id, 0),
                seq
            )

        # Check for conflicts
        pending_ops = self.buffer.get_pending_for_target(op.target_id)
        conflicts = self._detect_conflicts(op, pending_ops)

        if conflicts:
            for conflict in conflicts:
                resolved = await self._resolve_conflict(conflict)
                if not resolved:
                    self.conflicts[conflict.conflict_id] = conflict
                    self._emit_event("conflict_detected", {"conflict": conflict.to_dict()})

        # Apply if no unresolved conflicts
        if op.target_id not in [c.target_id for c in self.conflicts.values()]:
            handler = self._apply_handlers.get(op.target_type)
            if handler:
                try:
                    success = handler(op)
                    if success:
                        op.applied = True
                        op.confirmed = True
                        self.operation_log.append(op)

                        # Update state
                        if op.target_id not in self.sync_states:
                            self.sync_states[op.target_id] = SyncState(
                                target_id=op.target_id,
                                target_type=op.target_type
                            )
                        self.sync_states[op.target_id].increment_version(op.user_id)

                        self._emit_event("remote_applied", {"operation": op.to_dict()})
                        return True
                except Exception as e:
                    logger.error(f"Remote apply error: {e}")

        return False

    def confirm_operation(self, op_id: str) -> bool:
        """Confirm a pending operation from server."""
        op = self.buffer.confirm(op_id)
        if op:
            self._emit_event("operation_confirmed", {"op_id": op_id})
            return True
        return False

    def rollback_operation(self, op_id: str) -> bool:
        """Rollback a rejected operation."""
        op = self.buffer.reject(op_id)
        if op:
            # Need to apply inverse operation
            self._emit_event("operation_rolled_back", {"operation": op.to_dict()})
            return True
        return False

    def _detect_conflicts(
        self,
        new_op: Operation,
        existing_ops: list[Operation]
    ) -> list[Conflict]:
        """Detect conflicts between operations."""
        conflicts = []

        for existing in existing_ops:
            if existing.op_id == new_op.op_id:
                continue

            # Check if operations conflict
            if self._operations_conflict(new_op, existing):
                conflict = Conflict(
                    conflict_id=f"conflict_{uuid4().hex[:8]}",
                    operation_a=new_op,
                    operation_b=existing,
                    target_id=new_op.target_id
                )
                conflicts.append(conflict)

        return conflicts

    def _operations_conflict(self, op_a: Operation, op_b: Operation) -> bool:
        """Check if two operations conflict."""
        # Same target, overlapping time, incompatible changes
        if op_a.target_id != op_b.target_id:
            return False

        # DELETE vs UPDATE on same object
        if (op_a.op_type == OperationType.DELETE and
            op_b.op_type == OperationType.UPDATE):
            return True

        if (op_a.op_type == OperationType.UPDATE and
            op_b.op_type == OperationType.DELETE):
            return True

        # Concurrent updates to same field
        if op_a.op_type == OperationType.UPDATE and op_b.op_type == OperationType.UPDATE:
            fields_a = set(op_a.data.keys())
            fields_b = set(op_b.data.keys())
            if fields_a & fields_b:  # Overlapping fields
                return True

        return False

    async def _resolve_conflict(self, conflict: Conflict) -> bool:
        """Resolve a conflict using configured strategy."""
        if self.resolution_strategy == ConflictResolution.LAST_WRITER_WINS:
            # Later timestamp wins
            if conflict.operation_a.timestamp > conflict.operation_b.timestamp:
                winner = conflict.operation_a
            else:
                winner = conflict.operation_b

            conflict.resolution = f"last_writer_wins: {winner.user_id}"
            conflict.resolved_by = "automatic"

            self._emit_event("conflict_resolved", {
                "conflict_id": conflict.conflict_id,
                "winner": winner.op_id,
                "resolution": conflict.resolution
            })
            return True

        elif self.resolution_strategy == ConflictResolution.FIRST_WRITER_WINS:
            if conflict.operation_a.timestamp < conflict.operation_b.timestamp:
                winner = conflict.operation_a
            else:
                winner = conflict.operation_b

            conflict.resolution = f"first_writer_wins: {winner.user_id}"
            conflict.resolved_by = "automatic"
            return True

        elif self.resolution_strategy == ConflictResolution.MERGE:
            # Attempt to merge operations
            merged = self._merge_operations(
                conflict.operation_a,
                conflict.operation_b
            )
            if merged:
                conflict.resolution = "merged"
                conflict.resolved_by = "automatic"
                return True

        # Manual resolution required
        return False

    def _merge_operations(
        self,
        op_a: Operation,
        op_b: Operation
    ) -> Operation | None:
        """Attempt to merge two conflicting operations."""
        if op_a.op_type != OperationType.UPDATE or op_b.op_type != OperationType.UPDATE:
            return None

        # Merge non-overlapping fields
        merged_data = {}
        fields_a = set(op_a.data.keys())
        fields_b = set(op_b.data.keys())

        overlapping = fields_a & fields_b
        if overlapping:
            # Can't auto-merge overlapping fields
            return None

        merged_data.update(op_a.data)
        merged_data.update(op_b.data)

        return self.create_operation(
            op_type=OperationType.UPDATE,
            target_type=op_a.target_type,
            target_id=op_a.target_id,
            data=merged_data,
            parent_ops=[op_a.op_id, op_b.op_id]
        )

    def acquire_lock(
        self,
        target_id: str,
        target_type: str,
        duration_seconds: int = 30
    ) -> bool:
        """Acquire a lock on an object."""
        if target_id not in self.sync_states:
            self.sync_states[target_id] = SyncState(
                target_id=target_id,
                target_type=target_type
            )

        state = self.sync_states[target_id]

        if state.is_locked() and state.locked_by != self.user_id:
            return False

        from datetime import timedelta
        state.locked_by = self.user_id
        state.lock_expires = datetime.now() + timedelta(seconds=duration_seconds)

        self._emit_event("lock_acquired", {
            "target_id": target_id,
            "user_id": self.user_id,
            "expires": state.lock_expires.isoformat()
        })
        return True

    def release_lock(self, target_id: str) -> bool:
        """Release a lock on an object."""
        state = self.sync_states.get(target_id)
        if not state:
            return False

        if state.locked_by != self.user_id:
            return False

        state.locked_by = None
        state.lock_expires = None

        self._emit_event("lock_released", {
            "target_id": target_id,
            "user_id": self.user_id
        })
        return True

    def get_pending_operations(self) -> list[Operation]:
        """Get all pending operations."""
        return list(self.buffer.pending.values())

    def get_operation_history(
        self,
        target_id: str | None = None,
        limit: int = 100
    ) -> list[Operation]:
        """Get operation history."""
        ops = self.operation_log

        if target_id:
            ops = [op for op in ops if op.target_id == target_id]

        return ops[-limit:]

    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, data: dict[str, Any]):
        """Emit event to handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(event, data)
            except Exception as e:
                logger.error(f"Sync event handler error: {e}")

    async def start(self):
        """Start the synchronizer (lifecycle method)."""
        self._emit_event("sync_started", {
            "session_id": self.session_id,
            "user_id": self.user_id
        })

    async def stop(self):
        """Stop the synchronizer (lifecycle method)."""
        self._emit_event("sync_stopped", {
            "session_id": self.session_id,
            "user_id": self.user_id
        })

    def to_state(self) -> dict[str, Any]:
        """Export synchronizer state."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "sequence_number": self.sequence_number,
            "vector_clock": self.vector_clock,
            "pending_operations": len(self.buffer.pending),
            "total_operations": len(self.operation_log),
            "sync_states": {k: v.to_dict() for k, v in self.sync_states.items()},
            "unresolved_conflicts": len(self.conflicts)
        }


# Factory function
def create_state_synchronizer(
    session_id: str,
    user_id: str,
    resolution_strategy: str = "last_writer_wins"
) -> StateSynchronizer:
    """Create a state synchronizer."""
    strategy = ConflictResolution(resolution_strategy)
    return StateSynchronizer(
        session_id=session_id,
        user_id=user_id,
        resolution_strategy=strategy
    )

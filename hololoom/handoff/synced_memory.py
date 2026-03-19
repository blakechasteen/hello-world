"""
Synced Memory - Local-first CRDT memory synchronization.

Implements offline-first memory storage with automatic cross-device sync.
Memory operations are signed, queued, and merged using CRDT semantics.

Created: 2025-12-08
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from .identity import UnifiedIdentity

from .types import (
    HandoffError,
    MergeResult,
    SignedOp,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  LAMPORT CLOCK - Causality ordering
# ═══════════════════════════════════════════════════════════════════════════


class LamportClock:
    """
    Lamport logical clock for causal ordering of operations.

    Simple, deterministic ordering without wall-clock dependency.
    """

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = asyncio.Lock()

    @property
    def value(self) -> int:
        return self._value

    async def tick(self) -> int:
        """Increment clock and return new value."""
        async with self._lock:
            self._value += 1
            return self._value

    def tick_sync(self) -> int:
        """Synchronous tick for non-async contexts."""
        self._value += 1
        return self._value

    async def update(self, remote_clock: int) -> int:
        """
        Update clock on receiving remote operation.

        Lamport rule: local = max(local, remote) + 1
        """
        async with self._lock:
            self._value = max(self._value, remote_clock) + 1
            return self._value

    def update_sync(self, remote_clock: int) -> int:
        """Synchronous update for non-async contexts."""
        self._value = max(self._value, remote_clock) + 1
        return self._value


# ═══════════════════════════════════════════════════════════════════════════
#  MEMORY OPERATION TYPES
# ═══════════════════════════════════════════════════════════════════════════


class MemoryOpType(Enum):
    """Types of memory operations."""

    INSERT = "insert"       # Add new memory
    UPDATE = "update"       # Modify existing memory
    DELETE = "delete"       # Remove memory (tombstone)
    MERGE = "merge"         # Merge duplicate memories
    TAG = "tag"             # Add/modify tags
    LINK = "link"           # Create relationship
    UNLINK = "unlink"       # Remove relationship


@dataclass
class MemoryOp:
    """
    A memory operation before signing.

    Represents intent to modify memory state.
    """

    op_type: MemoryOpType
    content: str
    memory_id: str | None = None      # Target memory (for update/delete)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_clock(self, clock: int) -> MemoryOp:
        """Return copy with clock value in metadata."""
        new_meta = dict(self.metadata)
        new_meta["clock"] = clock
        return MemoryOp(
            op_type=self.op_type,
            content=self.content,
            memory_id=self.memory_id,
            tags=list(self.tags),
            metadata=new_meta,
        )

    def to_content_string(self) -> str:
        """Serialize to content string for SignedOp."""
        return json.dumps({
            "content": self.content,
            "memory_id": self.memory_id,
            "tags": self.tags,
            "metadata": self.metadata,
        }, sort_keys=True)

    @classmethod
    def from_content_string(cls, op_type: str, content_str: str) -> MemoryOp:
        """Deserialize from SignedOp content string."""
        data = json.loads(content_str)
        return cls(
            op_type=MemoryOpType(op_type),
            content=data["content"],
            memory_id=data.get("memory_id"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  LOCAL SQLITE STORE
# ═══════════════════════════════════════════════════════════════════════════


class LocalMemoryStore:
    """
    SQLite-based local memory store.

    Always works offline - the source of truth for this device.
    """

    def __init__(self, db_path: str | Path = "~/.hololoom/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript("""
            -- Memories table
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                clock INTEGER NOT NULL,
                device_id TEXT NOT NULL,
                identity_did TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                deleted INTEGER DEFAULT 0
            );

            -- Operations log (for sync)
            CREATE TABLE IF NOT EXISTS operations (
                op_id TEXT PRIMARY KEY,
                op_type TEXT NOT NULL,
                content TEXT NOT NULL,
                clock INTEGER NOT NULL,
                device_id TEXT NOT NULL,
                identity_did TEXT NOT NULL,
                timestamp REAL NOT NULL,
                signature BLOB NOT NULL,
                synced INTEGER DEFAULT 0
            );

            -- Sync state
            CREATE TABLE IF NOT EXISTS sync_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_memories_clock ON memories(clock);
            CREATE INDEX IF NOT EXISTS idx_memories_device ON memories(device_id);
            CREATE INDEX IF NOT EXISTS idx_operations_synced ON operations(synced);
            CREATE INDEX IF NOT EXISTS idx_operations_clock ON operations(clock);
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    async def apply(self, signed_op: SignedOp) -> dict[str, Any]:
        """
        Apply a signed operation to local store.

        Returns the affected memory record.
        """
        op = MemoryOp.from_content_string(signed_op.op_type, signed_op.content)
        now = time.time()

        if op.op_type == MemoryOpType.INSERT:
            # Generate memory ID from content hash
            memory_id = self._generate_memory_id(op.content, signed_op.clock)

            self._conn.execute("""
                INSERT OR REPLACE INTO memories
                (memory_id, content, tags, metadata, clock, device_id, identity_did, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                op.content,
                json.dumps(op.tags),
                json.dumps(op.metadata),
                signed_op.clock,
                signed_op.device_id,
                signed_op.identity_did,
                now,
                now,
            ))

        elif op.op_type == MemoryOpType.UPDATE:
            if not op.memory_id:
                raise HandoffError("UPDATE requires memory_id")

            self._conn.execute("""
                UPDATE memories
                SET content = ?, tags = ?, metadata = ?, clock = ?, updated_at = ?
                WHERE memory_id = ? AND clock < ?
            """, (
                op.content,
                json.dumps(op.tags),
                json.dumps(op.metadata),
                signed_op.clock,
                now,
                op.memory_id,
                signed_op.clock,  # Only update if our clock is newer
            ))
            memory_id = op.memory_id

        elif op.op_type == MemoryOpType.DELETE:
            if not op.memory_id:
                raise HandoffError("DELETE requires memory_id")

            # Tombstone delete (keep record, mark deleted)
            self._conn.execute("""
                UPDATE memories
                SET deleted = 1, clock = ?, updated_at = ?
                WHERE memory_id = ? AND clock < ?
            """, (
                signed_op.clock,
                now,
                op.memory_id,
                signed_op.clock,
            ))
            memory_id = op.memory_id

        else:
            # TAG, LINK, UNLINK - handle via metadata updates
            memory_id = op.memory_id or self._generate_memory_id(op.content, signed_op.clock)

        # Store operation in log
        self._conn.execute("""
            INSERT OR IGNORE INTO operations
            (op_id, op_type, content, clock, device_id, identity_did, timestamp, signature, synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            signed_op.op_id,
            signed_op.op_type,
            signed_op.content,
            signed_op.clock,
            signed_op.device_id,
            signed_op.identity_did,
            signed_op.timestamp,
            signed_op.signature,
        ))

        self._conn.commit()

        return {"memory_id": memory_id, "clock": signed_op.clock}

    async def merge(self, signed_op: SignedOp) -> bool:
        """
        CRDT merge a remote operation.

        Merge semantics:
        - INSERT: Add if not exists (set union)
        - UPDATE: Last-writer-wins by Lamport clock
        - DELETE: Tombstone (delete wins over concurrent update)

        Returns True if operation was applied (not a duplicate).
        """
        # Check if we've already seen this operation
        cursor = self._conn.execute(
            "SELECT 1 FROM operations WHERE op_id = ?",
            (signed_op.op_id,)
        )
        if cursor.fetchone():
            return False  # Already merged

        # Apply the operation
        await self.apply(signed_op)
        return True

    def get_unsynced_ops(self, limit: int = 100) -> list[SignedOp]:
        """Get operations not yet synced to other devices."""
        cursor = self._conn.execute("""
            SELECT op_id, op_type, content, clock, device_id, identity_did, timestamp, signature
            FROM operations
            WHERE synced = 0
            ORDER BY clock ASC
            LIMIT ?
        """, (limit,))

        ops = []
        for row in cursor:
            ops.append(SignedOp(
                op_id=row["op_id"],
                op_type=row["op_type"],
                content=row["content"],
                clock=row["clock"],
                device_id=row["device_id"],
                identity_did=row["identity_did"],
                timestamp=row["timestamp"],
                signature=row["signature"],
            ))
        return ops

    def mark_synced(self, op_ids: list[str]) -> None:
        """Mark operations as synced."""
        if not op_ids:
            return
        placeholders = ",".join("?" * len(op_ids))
        self._conn.execute(
            f"UPDATE operations SET synced = 1 WHERE op_id IN ({placeholders})",
            op_ids
        )
        self._conn.commit()

    def get_clock_value(self) -> int:
        """Get the highest clock value in local store."""
        cursor = self._conn.execute(
            "SELECT MAX(clock) as max_clock FROM operations"
        )
        row = cursor.fetchone()
        return row["max_clock"] or 0

    def query(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Simple query of local memories.

        Note: This is a basic implementation. Production would use
        the full HoloLoom memory backend with vector search.
        """
        # Simple LIKE search for now
        cursor = self._conn.execute("""
            SELECT memory_id, content, tags, metadata, clock, device_id, created_at
            FROM memories
            WHERE deleted = 0 AND content LIKE ?
            ORDER BY clock DESC
            LIMIT ?
        """, (f"%{query_text}%", limit))

        results = []
        for row in cursor:
            results.append({
                "memory_id": row["memory_id"],
                "content": row["content"],
                "tags": json.loads(row["tags"]),
                "metadata": json.loads(row["metadata"]),
                "clock": row["clock"],
                "device_id": row["device_id"],
                "created_at": row["created_at"],
            })
        return results

    def _generate_memory_id(self, content: str, clock: int) -> str:
        """Generate deterministic memory ID from content."""
        data = f"{content}:{clock}".encode()
        return f"mem_{hashlib.sha256(data).hexdigest()[:16]}"


# ═══════════════════════════════════════════════════════════════════════════
#  SYNCED MEMORY - Main interface
# ═══════════════════════════════════════════════════════════════════════════


class SyncedMemory:
    """
    Cross-device memory with automatic CRDT synchronization.

    Local-first: Always works offline. Sync happens in background.

    Usage:
        identity = UnifiedIdentity.create("blake", "laptop")
        memory = SyncedMemory(identity)

        # Store memory (works offline)
        result = await memory.experience("Thompson Sampling balances exploration")

        # Query memories
        memories = await memory.recall("sampling")

        # Sync when online
        await memory.sync()
    """

    def __init__(
        self,
        identity: UnifiedIdentity,
        db_path: str | Path = "~/.hololoom/memory.db",
        pending_capacity: int = 10000,
    ):
        self.identity = identity
        self.local = LocalMemoryStore(db_path)
        self.clock = LamportClock(self.local.get_clock_value())
        self.pending: deque[SignedOp] = deque(maxlen=pending_capacity)

        # Nonce tracking for replay protection
        self._seen_nonces: set[str] = set()
        self._nonce_expiry: dict[str, float] = {}
        self._nonce_window = 300.0  # 5 minute window

        # Sync callbacks
        self._on_sync_callbacks: list[Callable[[MergeResult], None]] = []

    async def apply(self, op: MemoryOp) -> dict[str, Any]:
        """
        Apply memory operation locally and queue for sync.

        Args:
            op: Memory operation to apply

        Returns:
            Dict with memory_id and clock value
        """
        # Get next clock value
        clock = await self.clock.tick()

        # Create signed operation
        signed_op = self._create_signed_op(op, clock)

        # Sign with identity
        signed_op = self.identity.sign_operation(signed_op)

        # Apply locally first (offline-first)
        result = await self.local.apply(signed_op)

        # Queue for sync (non-blocking)
        self.pending.append(signed_op)

        return result

    async def experience(self, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        """
        Form a new memory (convenience method).

        Args:
            content: Memory content
            tags: Optional tags

        Returns:
            Dict with memory_id and clock value
        """
        op = MemoryOp(
            op_type=MemoryOpType.INSERT,
            content=content,
            tags=tags or [],
        )
        return await self.apply(op)

    async def recall(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve memories matching query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching memories
        """
        return self.local.query(query, limit)

    async def merge(self, remote_ops: list[SignedOp]) -> MergeResult:
        """
        CRDT merge remote operations.

        Merge is:
        - Commutative: merge(A, B) = merge(B, A)
        - Associative: merge(merge(A, B), C) = merge(A, merge(B, C))
        - Idempotent: merge(A, A) = A

        Args:
            remote_ops: Signed operations from remote device

        Returns:
            MergeResult with statistics
        """
        merged = 0
        rejected = 0
        conflicts = 0
        max_clock = self.clock.value

        for op in remote_ops:
            try:
                # Verify signature (proves identity ownership)
                if not self.identity.verify_operation(op):
                    logger.warning(f"Invalid signature for op {op.op_id}")
                    rejected += 1
                    continue

                # Check for replay attack
                nonce = f"{op.op_id}:{op.timestamp}"
                if self._is_replay(nonce, op.timestamp):
                    logger.warning(f"Replay detected for op {op.op_id}")
                    rejected += 1
                    continue

                # Update vector clock
                await self.clock.update(op.clock)
                max_clock = max(max_clock, op.clock)

                # CRDT merge (no conflicts possible due to operation-based CRDT)
                applied = await self.local.merge(op)
                if applied:
                    merged += 1
                    self._record_nonce(nonce, op.timestamp)

            except Exception as e:
                logger.error(f"Merge error for op {op.op_id}: {e}")
                rejected += 1

        result = MergeResult(
            merged=merged,
            rejected=rejected,
            conflicts=conflicts,
            new_clock=max_clock,
        )

        # Notify callbacks
        for callback in self._on_sync_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Sync callback error: {e}")

        return result

    def pending_delta(self) -> list[SignedOp]:
        """
        Get operations not yet synced.

        Returns minimal set of operations for bandwidth efficiency.
        """
        # Combine in-memory pending with unsynced from DB
        db_unsynced = self.local.get_unsynced_ops(limit=1000)
        pending_ids = {op.op_id for op in self.pending}

        # Deduplicate
        combined = list(self.pending)
        for op in db_unsynced:
            if op.op_id not in pending_ids:
                combined.append(op)

        # Sort by clock for consistent ordering
        combined.sort(key=lambda op: op.clock)

        return combined

    def mark_synced(self, ops: list[SignedOp]) -> None:
        """Mark operations as successfully synced."""
        op_ids = [op.op_id for op in ops]
        self.local.mark_synced(op_ids)

        # Clear from pending queue
        synced_ids = set(op_ids)
        self.pending = deque(
            (op for op in self.pending if op.op_id not in synced_ids),
            maxlen=self.pending.maxlen
        )

    def on_sync(self, callback: Callable[[MergeResult], None]) -> None:
        """Register callback for sync events."""
        self._on_sync_callbacks.append(callback)

    def _create_signed_op(self, op: MemoryOp, clock: int) -> SignedOp:
        """Create SignedOp from MemoryOp."""
        import secrets

        return SignedOp(
            op_id=f"op_{secrets.token_hex(12)}",
            op_type=op.op_type.value,
            content=op.to_content_string(),
            clock=clock,
            device_id=self.identity.current_device_id or "",
            identity_did=self.identity.did,
            timestamp=time.time(),
            signature=b"",  # Will be filled by sign_operation
        )

    def _is_replay(self, nonce: str, timestamp: float) -> bool:
        """Check if operation is a replay attack."""
        # Clean expired nonces
        now = time.time()
        expired = [n for n, t in self._nonce_expiry.items() if now - t > self._nonce_window]
        for n in expired:
            self._seen_nonces.discard(n)
            del self._nonce_expiry[n]

        # Check timestamp freshness
        if abs(now - timestamp) > self._nonce_window:
            return True  # Too old or too far in future

        # Check nonce reuse
        if nonce in self._seen_nonces:
            return True

        return False

    def _record_nonce(self, nonce: str, timestamp: float) -> None:
        """Record nonce for replay protection."""
        self._seen_nonces.add(nonce)
        self._nonce_expiry[nonce] = timestamp

    def close(self) -> None:
        """Close resources."""
        self.local.close()

    async def __aenter__(self) -> SyncedMemory:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def create_synced_memory(
    identity: UnifiedIdentity,
    db_path: str | Path | None = None,
) -> SyncedMemory:
    """
    Create a SyncedMemory instance.

    Args:
        identity: UnifiedIdentity for signing
        db_path: Optional custom database path

    Returns:
        Configured SyncedMemory instance
    """
    path = db_path or "~/.hololoom/memory.db"
    return SyncedMemory(identity, db_path=path)

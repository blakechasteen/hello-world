"""
Handoff Types - Data structures for cross-device handoff.

Created: 2025-12-08
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════════


class DeviceStatus(Enum):
    """Device health and authorization status."""

    ACTIVE = auto()      # Healthy and authorized
    SUSPENDED = auto()   # Temporarily disabled
    REVOKED = auto()     # Permanently revoked (compromised)
    PENDING = auto()     # Awaiting pairing confirmation


class HandoffStatus(Enum):
    """Status of a handoff operation."""

    PENDING = auto()     # Initiated, awaiting acceptance
    IN_PROGRESS = auto() # Transfer in progress
    COMPLETED = auto()   # Successfully completed
    FAILED = auto()      # Failed (will retry)
    REJECTED = auto()    # Rejected by target device
    CANCELLED = auto()   # Cancelled by source


class SyncDirection(Enum):
    """Direction of synchronization."""

    PUSH = auto()   # Send to other devices
    PULL = auto()   # Receive from other devices
    BOTH = auto()   # Bidirectional sync


# ═══════════════════════════════════════════════════════════════════════════
#  DEVICE MANIFEST
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DeviceCapability:
    """What a device can do."""

    WEAVE = "weave"           # Full weaving cycle
    RECALL = "recall"         # Memory retrieval only
    EXPERIENCE = "experience" # Memory storage only
    REFLECT = "reflect"       # Learning/feedback
    SYNC = "sync"             # State synchronization


@dataclass
class DeviceManifest:
    """
    Registration info for a device in the identity's device registry.

    Each device paired with an identity gets a manifest describing
    its capabilities, status, and authentication credentials.
    """

    device_id: str                          # Unique identifier (hash of public key)
    device_name: str                        # Human-readable name ("blake-laptop")
    public_key: bytes                       # Ed25519 public key for this device
    api_key_id: str                         # API key for rate limiting
    capabilities: Set[str] = field(default_factory=lambda: {
        DeviceCapability.WEAVE,
        DeviceCapability.RECALL,
        DeviceCapability.EXPERIENCE,
        DeviceCapability.REFLECT,
        DeviceCapability.SYNC,
    })
    status: DeviceStatus = DeviceStatus.PENDING
    paired_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    last_sync: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if device is active and authorized."""
        return self.status == DeviceStatus.ACTIVE

    def is_revoked(self) -> bool:
        """Check if device has been revoked."""
        return self.status == DeviceStatus.REVOKED

    def update_seen(self) -> None:
        """Update last seen timestamp."""
        self.last_seen = datetime.utcnow()

    def update_sync(self) -> None:
        """Update last sync timestamp."""
        self.last_sync = datetime.utcnow()
        self.update_seen()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "public_key": self.public_key.hex(),
            "api_key_id": self.api_key_id,
            "capabilities": list(self.capabilities),
            "status": self.status.name,
            "paired_at": self.paired_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceManifest":
        """Deserialize from dictionary."""
        return cls(
            device_id=data["device_id"],
            device_name=data["device_name"],
            public_key=bytes.fromhex(data["public_key"]),
            api_key_id=data["api_key_id"],
            capabilities=set(data.get("capabilities", [])),
            status=DeviceStatus[data.get("status", "PENDING")],
            paired_at=datetime.fromisoformat(data["paired_at"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            last_sync=datetime.fromisoformat(data["last_sync"]) if data.get("last_sync") else None,
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  HANDOFF REQUEST/RESULT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class HandoffRequest:
    """
    Request to transfer state to another device.

    Contains all information needed to resume work on target device.
    """

    request_id: str                         # Unique request identifier
    source_device: str                      # Device ID of sender
    target_device: str                      # Device ID of recipient
    identity_did: str                       # DID of the identity
    pending_ops: List[Dict[str, Any]]       # CRDT operations to sync
    context: Dict[str, Any]                 # UI state, ambient context
    timestamp: float                        # Unix timestamp
    nonce: str                              # For replay protection
    signature: bytes = b""                  # Ed25519 signature

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"handoff_{secrets.token_hex(12)}"
        if not self.nonce:
            self.nonce = secrets.token_hex(16)
        if not self.timestamp:
            self.timestamp = time.time()

    def to_bytes(self) -> bytes:
        """Serialize for signing (excludes signature)."""
        import json
        data = {
            "request_id": self.request_id,
            "source_device": self.source_device,
            "target_device": self.target_device,
            "identity_did": self.identity_did,
            "pending_ops": self.pending_ops,
            "context": self.context,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }
        return json.dumps(data, sort_keys=True).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "source_device": self.source_device,
            "target_device": self.target_device,
            "identity_did": self.identity_did,
            "pending_ops": self.pending_ops,
            "context": self.context,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "signature": self.signature.hex() if self.signature else "",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffRequest":
        """Deserialize from dictionary."""
        return cls(
            request_id=data["request_id"],
            source_device=data["source_device"],
            target_device=data["target_device"],
            identity_did=data["identity_did"],
            pending_ops=data.get("pending_ops", []),
            context=data.get("context", {}),
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            signature=bytes.fromhex(data.get("signature", "")),
        )


@dataclass
class HandoffResult:
    """Result of a handoff operation."""

    request_id: str
    status: HandoffStatus
    source_device: str
    target_device: str
    ops_transferred: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def success(self) -> bool:
        """Check if handoff was successful."""
        return self.status == HandoffStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "status": self.status.name,
            "source_device": self.source_device,
            "target_device": self.target_device,
            "ops_transferred": self.ops_transferred,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "completed_at": self.completed_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  SYNC OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SignedOp:
    """
    A signed CRDT operation for cross-device sync.

    Immutable and verified by signature.
    """

    op_id: str                              # Unique operation ID
    op_type: str                            # "insert", "update", "delete"
    content: str                            # Operation content
    clock: int                              # Lamport timestamp
    device_id: str                          # Origin device
    identity_did: str                       # Owner identity
    timestamp: float                        # Unix timestamp
    signature: bytes                        # Ed25519 signature

    def to_bytes(self) -> bytes:
        """Serialize for verification (excludes signature)."""
        import json
        data = {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "content": self.content,
            "clock": self.clock,
            "device_id": self.device_id,
            "identity_did": self.identity_did,
            "timestamp": self.timestamp,
        }
        return json.dumps(data, sort_keys=True).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "content": self.content,
            "clock": self.clock,
            "device_id": self.device_id,
            "identity_did": self.identity_did,
            "timestamp": self.timestamp,
            "signature": self.signature.hex(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignedOp":
        """Deserialize from dictionary."""
        return cls(
            op_id=data["op_id"],
            op_type=data["op_type"],
            content=data["content"],
            clock=data["clock"],
            device_id=data["device_id"],
            identity_did=data["identity_did"],
            timestamp=data["timestamp"],
            signature=bytes.fromhex(data["signature"]),
        )


@dataclass
class MergeResult:
    """Result of merging remote operations."""

    merged: int                             # Number of ops merged
    rejected: int = 0                       # Invalid signature / replay
    conflicts: int = 0                      # Conflicts auto-resolved
    new_clock: int = 0                      # Updated Lamport clock


# ═══════════════════════════════════════════════════════════════════════════
#  ERRORS
# ═══════════════════════════════════════════════════════════════════════════


class HandoffError(Exception):
    """Base handoff error."""

    def __init__(
        self,
        message: str,
        *,
        device_id: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.device_id = device_id
        self.suggestion = suggestion
        super().__init__(self._format(message))

    def _format(self, message: str) -> str:
        parts = [message]
        if self.device_id:
            parts.append(f"Device: {self.device_id}")
        if self.suggestion:
            parts.append(f"Try: {self.suggestion}")
        return " | ".join(parts)


class DeviceNotFoundError(HandoffError):
    """Device not in registry."""
    pass


class DeviceRevokedError(HandoffError):
    """Device has been revoked."""
    pass


class DeviceUnhealthyError(HandoffError):
    """Device circuit breaker is open."""
    pass


class RateLimitExceededError(HandoffError):
    """Device rate limit exceeded."""
    pass


class InvalidSignatureError(HandoffError):
    """Signature verification failed."""
    pass


class ReplayAttackError(HandoffError):
    """Stale timestamp or reused nonce."""
    pass


class PayloadRejectedError(HandoffError):
    """WAF rejected payload."""
    pass


class HandoffBlockedError(HandoffError):
    """Safety guardrails blocked handoff."""
    pass

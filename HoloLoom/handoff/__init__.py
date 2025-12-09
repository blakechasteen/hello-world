"""
HoloLoom Handoff - Cross-Device Memory Synchronization

Enables seamless work transfer between devices where identity owns memory,
devices are just windows. Built with 7 security layers for production use.

Created: December 2025

Quick Start:
    from HoloLoom.handoff import UnifiedIdentity
    from HoloLoom import HoloLoom

    # Generate identity (or load existing)
    identity = UnifiedIdentity.generate("blake")
    identity.save("~/.hololoom/identity.key")

    # Use HoloLoom with cross-device sync
    async with HoloLoom(identity=identity) as loom:
        await loom.experience("Learning about distributed systems")
        # Memory syncs automatically to all paired devices!

Device Pairing:
    # On device A (source)
    pairing_code = await loom.add_device("phone")
    print(f"Pairing code: {pairing_code}")

    # On device B (target) - use the pairing code
    # ... device B connects and receives memory sync

Handoff Flow:
    # On device A - hand off current work
    result = await loom.handoff_to("device_b_id", context={"task": "Continue research"})

    # On device B - automatically receives pending ops and context

Security Layers:
    1. Request Signing (Ed25519 signatures + nonce + timestamp)
    2. Rate Limiting (per-device limits)
    3. Circuit Breakers (device health isolation)
    4. WAF Validation (payload sanitization)
    5. Risk Gating (SAFE → CRITICAL assessment)
    6. Security Monitoring (real-time + alerting)
    7. Audit Trail (complete provenance)
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
#  IDENTITY - Cryptographic identity and device management
# ═══════════════════════════════════════════════════════════════════════════

from HoloLoom.handoff.identity import (
    UnifiedIdentity,
    get_or_create_identity,
)

# ═══════════════════════════════════════════════════════════════════════════
#  TYPES - Data structures for handoff operations
# ═══════════════════════════════════════════════════════════════════════════

from HoloLoom.handoff.types import (
    # Enums
    DeviceStatus,
    HandoffStatus,
    SyncDirection,
    # Device management
    DeviceCapability,
    DeviceManifest,
    # Handoff operations
    HandoffRequest,
    HandoffResult,
    # Sync operations
    SignedOp,
    MergeResult,
    # Exceptions
    HandoffError,
    DeviceNotFoundError,
    DeviceRevokedError,
    DeviceUnhealthyError,
    RateLimitExceededError,
    InvalidSignatureError,
    ReplayAttackError,
    PayloadRejectedError,
    HandoffBlockedError,
)

# ═══════════════════════════════════════════════════════════════════════════
#  SYNCED MEMORY - CRDT-based local-first memory with automatic sync
# ═══════════════════════════════════════════════════════════════════════════

from HoloLoom.handoff.synced_memory import (
    SyncedMemory,
    LamportClock,
    MemoryOp,
    MemoryOpType,
)

# ═══════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR - 7-layer security handoff coordination
# ═══════════════════════════════════════════════════════════════════════════

from HoloLoom.handoff.orchestrator import (
    HardenedHandoffOrchestrator,
)

# ═══════════════════════════════════════════════════════════════════════════
#  TRANSPORT - Pluggable transport layer (WebSocket, Bluetooth, LAN)
# ═══════════════════════════════════════════════════════════════════════════

from HoloLoom.handoff.transport import (
    # Protocol and base
    TransportProtocol,
    TransportStatus,
    TransportMessage,
    SendResult,
    BaseTransport,
    # Transport implementations
    WebSocketTransport,
    BluetoothTransport,
    LocalNetworkTransport,
    CompositeTransport,
    # Factory functions
    create_default_transport,
    create_websocket_only_transport,
    create_local_only_transport,
)


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Identity
    "UnifiedIdentity",
    "get_or_create_identity",
    # Types - Enums
    "DeviceStatus",
    "HandoffStatus",
    "SyncDirection",
    # Types - Device management
    "DeviceCapability",
    "DeviceManifest",
    # Types - Handoff operations
    "HandoffRequest",
    "HandoffResult",
    # Types - Sync operations
    "SignedOp",
    "MergeResult",
    # Types - Exceptions
    "HandoffError",
    "DeviceNotFoundError",
    "DeviceRevokedError",
    "DeviceUnhealthyError",
    "RateLimitExceededError",
    "InvalidSignatureError",
    "ReplayAttackError",
    "PayloadRejectedError",
    "HandoffBlockedError",
    # Synced Memory
    "SyncedMemory",
    "LamportClock",
    "MemoryOp",
    "MemoryOpType",
    # Orchestrator
    "HardenedHandoffOrchestrator",
    # Transport - Protocol and base
    "TransportProtocol",
    "TransportStatus",
    "TransportMessage",
    "SendResult",
    "BaseTransport",
    # Transport - Implementations
    "WebSocketTransport",
    "BluetoothTransport",
    "LocalNetworkTransport",
    "CompositeTransport",
    # Transport - Factories
    "create_default_transport",
    "create_websocket_only_transport",
    "create_local_only_transport",
]


# ═══════════════════════════════════════════════════════════════════════════
#  VERSION
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "1.0.0"

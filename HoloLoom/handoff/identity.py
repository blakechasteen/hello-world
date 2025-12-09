"""
Unified Identity - Cross-device identity with device registry.

Extends federation/identity.py with device management for seamless handoff.

Created: 2025-12-08
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .types import (
    DeviceManifest,
    DeviceStatus,
    HandoffRequest,
    SignedOp,
    DeviceNotFoundError,
    DeviceRevokedError,
    InvalidSignatureError,
)

# Import federation identity for Ed25519 operations
try:
    from ..federation.identity import Identity, SignedMessage, HAS_CRYPTOGRAPHY
except ImportError:
    # Fallback if federation not available
    HAS_CRYPTOGRAPHY = False
    Identity = None
    SignedMessage = None


# ═══════════════════════════════════════════════════════════════════════════
#  UNIFIED IDENTITY
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedIdentity:
    """
    Cross-device identity with device registry.

    Builds on federation Identity, adding:
    - Device registry (multiple devices per identity)
    - Device pairing via QR code exchange
    - Device revocation
    - DID (Decentralized Identifier) format

    Usage:
        # Create new identity
        identity = UnifiedIdentity.create("blake")

        # Save to disk
        identity.save("~/.hololoom/identity")

        # Load existing
        identity = UnifiedIdentity.load("~/.hololoom/identity")

        # Add a device
        manifest = identity.add_device("my-phone", qr_payload)

        # Revoke compromised device
        identity.revoke_device("device_abc123")

        # Sign operations
        signed = identity.sign_operation(op)

        # Verify remote operations
        valid = identity.verify_operation(signed_op)
    """

    def __init__(
        self,
        nickname: str,
        identity: "Identity",
        devices: Optional[Dict[str, DeviceManifest]] = None,
        current_device_id: Optional[str] = None,
    ):
        """
        Initialize unified identity.

        Args:
            nickname: Human-readable name (e.g., "blake")
            identity: Ed25519 identity from federation
            devices: Device registry
            current_device_id: This device's ID in the registry
        """
        self.nickname = nickname
        self._identity = identity
        self.devices: Dict[str, DeviceManifest] = devices or {}
        self._current_device_id = current_device_id
        self._revoked_nonces: set = set()  # For replay protection

    # ═══════════════════════════════════════════════════════════════════════
    #  PROPERTIES
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def did(self) -> str:
        """
        Decentralized Identifier (W3C DID spec).

        Format: did:key:z6Mk... (base58btc multibase encoding)
        """
        # Simplified: Use hex encoding of public key
        # Full implementation would use multibase/multicodec
        return f"did:key:z{self._identity.public_key.hex()}"

    @property
    def node_id(self) -> str:
        """Node ID (from federation identity)."""
        return self._identity.node_id

    @property
    def short_id(self) -> str:
        """Short readable ID."""
        return self._identity.short_id

    @property
    def public_key(self) -> bytes:
        """Ed25519 public key."""
        return self._identity.public_key

    @property
    def current_device_id(self) -> Optional[str]:
        """Current device's ID."""
        return self._current_device_id

    @property
    def active_devices(self) -> List[DeviceManifest]:
        """List of active (non-revoked) devices."""
        return [d for d in self.devices.values() if d.is_active()]

    # ═══════════════════════════════════════════════════════════════════════
    #  FACTORY METHODS
    # ═══════════════════════════════════════════════════════════════════════

    @classmethod
    def create(
        cls,
        nickname: str,
        device_name: Optional[str] = None,
    ) -> "UnifiedIdentity":
        """
        Create a new identity with optional first device.

        Args:
            nickname: Human-readable identity name
            device_name: Optional name for this device

        Returns:
            New UnifiedIdentity
        """
        if Identity is None:
            raise ImportError(
                "Federation identity not available. "
                "Ensure HoloLoom.federation is importable."
            )

        identity = Identity.generate()
        unified = cls(nickname=nickname, identity=identity)

        # Register this device if name provided
        if device_name:
            device_id = unified._generate_device_id()
            manifest = DeviceManifest(
                device_id=device_id,
                device_name=device_name,
                public_key=identity.public_key,
                api_key_id=f"apikey_{secrets.token_hex(8)}",
                status=DeviceStatus.ACTIVE,
            )
            unified.devices[device_id] = manifest
            unified._current_device_id = device_id

        return unified

    @classmethod
    def load(cls, path: str | Path) -> "UnifiedIdentity":
        """
        Load identity from disk.

        Args:
            path: Directory containing identity files

        Returns:
            Loaded UnifiedIdentity
        """
        path = Path(path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"Identity directory not found: {path}")

        # Load federation identity
        key_path = path / "identity.key"
        if not key_path.exists():
            raise FileNotFoundError(f"Key file not found: {key_path}")

        identity = Identity.load(key_path)

        # Load metadata
        meta_path = path / "identity.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {"nickname": "unknown", "devices": {}}

        # Parse devices
        devices = {}
        for device_id, device_data in meta.get("devices", {}).items():
            devices[device_id] = DeviceManifest.from_dict(device_data)

        return cls(
            nickname=meta.get("nickname", "unknown"),
            identity=identity,
            devices=devices,
            current_device_id=meta.get("current_device_id"),
        )

    def save(self, path: str | Path) -> None:
        """
        Save identity to disk.

        Args:
            path: Directory to save identity files
        """
        path = Path(path).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        # Save federation identity
        self._identity.save(path / "identity.key")

        # Save metadata
        meta = {
            "nickname": self.nickname,
            "did": self.did,
            "node_id": self.node_id,
            "current_device_id": self._current_device_id,
            "devices": {
                device_id: manifest.to_dict()
                for device_id, manifest in self.devices.items()
            },
            "saved_at": datetime.utcnow().isoformat(),
        }

        with open(path / "identity.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    #  DEVICE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def add_device(
        self,
        device_name: str,
        public_key: bytes,
        capabilities: Optional[set] = None,
    ) -> DeviceManifest:
        """
        Add a new device to the identity.

        Args:
            device_name: Human-readable device name
            public_key: Ed25519 public key from QR code exchange
            capabilities: Optional set of capabilities

        Returns:
            DeviceManifest for the new device
        """
        # Generate device ID from public key
        device_id = hashlib.sha256(public_key).hexdigest()[:16]

        # Check if already registered
        if device_id in self.devices:
            existing = self.devices[device_id]
            if existing.is_revoked():
                raise DeviceRevokedError(
                    f"Device {device_name} was previously revoked",
                    device_id=device_id,
                    suggestion="Use a fresh key pair for this device",
                )
            # Update existing device
            existing.device_name = device_name
            existing.update_seen()
            return existing

        # Create new manifest
        manifest = DeviceManifest(
            device_id=device_id,
            device_name=device_name,
            public_key=public_key,
            api_key_id=f"apikey_{secrets.token_hex(8)}",
            capabilities=capabilities or set(),
            status=DeviceStatus.ACTIVE,
        )

        self.devices[device_id] = manifest
        return manifest

    def get_device(self, device_id: str) -> DeviceManifest:
        """
        Get device manifest by ID.

        Raises:
            DeviceNotFoundError: If device not in registry
        """
        if device_id not in self.devices:
            raise DeviceNotFoundError(
                f"Device not found",
                device_id=device_id,
                suggestion="Check device ID or re-pair the device",
            )
        return self.devices[device_id]

    def revoke_device(self, device_id: str) -> None:
        """
        Revoke a device immediately.

        Args:
            device_id: Device to revoke

        The device will be marked REVOKED and will fail all
        future authentication attempts.
        """
        if device_id not in self.devices:
            raise DeviceNotFoundError(
                f"Cannot revoke unknown device",
                device_id=device_id,
            )

        manifest = self.devices[device_id]
        manifest.status = DeviceStatus.REVOKED

    def suspend_device(self, device_id: str) -> None:
        """
        Temporarily suspend a device.

        Args:
            device_id: Device to suspend
        """
        if device_id not in self.devices:
            raise DeviceNotFoundError(
                f"Cannot suspend unknown device",
                device_id=device_id,
            )

        manifest = self.devices[device_id]
        manifest.status = DeviceStatus.SUSPENDED

    def reactivate_device(self, device_id: str) -> None:
        """
        Reactivate a suspended device.

        Args:
            device_id: Device to reactivate

        Note: Cannot reactivate REVOKED devices.
        """
        if device_id not in self.devices:
            raise DeviceNotFoundError(
                f"Cannot reactivate unknown device",
                device_id=device_id,
            )

        manifest = self.devices[device_id]
        if manifest.is_revoked():
            raise DeviceRevokedError(
                f"Cannot reactivate revoked device",
                device_id=device_id,
                suggestion="Revocation is permanent. Pair a new device instead.",
            )

        manifest.status = DeviceStatus.ACTIVE

    def is_device_authorized(self, device_id: str) -> bool:
        """Check if device is authorized for operations."""
        if device_id not in self.devices:
            return False
        return self.devices[device_id].is_active()

    # ═══════════════════════════════════════════════════════════════════════
    #  SIGNING & VERIFICATION
    # ═══════════════════════════════════════════════════════════════════════

    def sign(self, message: bytes) -> bytes:
        """
        Sign message with identity's private key.

        Args:
            message: Bytes to sign

        Returns:
            Ed25519 signature
        """
        return self._identity.sign(message)

    def sign_operation(self, op: SignedOp) -> SignedOp:
        """
        Sign a CRDT operation.

        Args:
            op: Operation to sign

        Returns:
            Operation with signature populated
        """
        message = op.to_bytes()
        signature = self.sign(message)

        return SignedOp(
            op_id=op.op_id,
            op_type=op.op_type,
            content=op.content,
            clock=op.clock,
            device_id=op.device_id,
            identity_did=self.did,
            timestamp=op.timestamp,
            signature=signature,
        )

    def sign_request(self, request: HandoffRequest) -> HandoffRequest:
        """
        Sign a handoff request.

        Args:
            request: Request to sign

        Returns:
            Request with signature populated
        """
        message = request.to_bytes()
        signature = self.sign(message)

        return HandoffRequest(
            request_id=request.request_id,
            source_device=request.source_device,
            target_device=request.target_device,
            identity_did=self.did,
            pending_ops=request.pending_ops,
            context=request.context,
            timestamp=request.timestamp,
            nonce=request.nonce,
            signature=signature,
        )

    def verify(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        """
        Verify signature from a public key.

        Args:
            message: Original message bytes
            signature: Signature to verify
            public_key: Signer's public key

        Returns:
            True if signature is valid
        """
        return self._identity.verify_signature(message, signature, public_key)

    def verify_operation(self, op: SignedOp) -> bool:
        """
        Verify a signed CRDT operation.

        Checks:
        1. Signature is valid
        2. Identity DID matches
        3. Device is authorized (if in registry)

        Args:
            op: Operation to verify

        Returns:
            True if valid
        """
        # Check identity
        if op.identity_did != self.did:
            return False

        # Get device public key
        if op.device_id in self.devices:
            device = self.devices[op.device_id]
            if not device.is_active():
                return False
            public_key = device.public_key
        else:
            # Unknown device - use identity public key
            public_key = self.public_key

        # Verify signature
        message = op.to_bytes()
        return self.verify(message, op.signature, public_key)

    def verify_request(
        self,
        request: HandoffRequest,
        max_age_seconds: float = 300.0,
    ) -> bool:
        """
        Verify a handoff request.

        Checks:
        1. Signature is valid
        2. Timestamp is fresh (not replay)
        3. Nonce hasn't been used
        4. Source device is authorized

        Args:
            request: Request to verify
            max_age_seconds: Maximum age for timestamp (default: 5 min)

        Returns:
            True if valid
        """
        import time

        # Check timestamp freshness
        age = time.time() - request.timestamp
        if age > max_age_seconds:
            return False  # Stale request

        # Check nonce (replay protection)
        if request.nonce in self._revoked_nonces:
            return False  # Replay attack
        self._revoked_nonces.add(request.nonce)

        # Limit nonce set size
        if len(self._revoked_nonces) > 10000:
            # Clear old nonces (simplified - production would use expiry)
            self._revoked_nonces.clear()

        # Check source device
        if request.source_device in self.devices:
            device = self.devices[request.source_device]
            if not device.is_active():
                return False
            public_key = device.public_key
        else:
            # Unknown device
            return False

        # Verify signature
        message = request.to_bytes()
        return self.verify(message, request.signature, public_key)

    # ═══════════════════════════════════════════════════════════════════════
    #  QR CODE PAIRING
    # ═══════════════════════════════════════════════════════════════════════

    def generate_pairing_payload(self) -> bytes:
        """
        Generate QR code payload for device pairing.

        The payload contains:
        - Identity DID
        - Identity public key
        - This device's public key (for verification)

        Returns:
            Bytes suitable for QR code encoding
        """
        data = {
            "did": self.did,
            "public_key": self.public_key.hex(),
            "device_key": self.public_key.hex(),  # Current device
            "nickname": self.nickname,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return json.dumps(data).encode("utf-8")

    def verify_pairing_payload(self, payload: bytes) -> Dict[str, Any]:
        """
        Verify and parse QR code payload from another device.

        Args:
            payload: QR code bytes

        Returns:
            Parsed payload with device info

        Raises:
            InvalidSignatureError: If payload is invalid
        """
        try:
            data = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise InvalidSignatureError(
                f"Invalid pairing payload: {e}",
                suggestion="Scan the QR code again",
            )

        # Verify required fields
        required = ["did", "public_key", "device_key"]
        for field in required:
            if field not in data:
                raise InvalidSignatureError(
                    f"Missing required field: {field}",
                    suggestion="Ensure QR code is from HoloLoom app",
                )

        return data

    # ═══════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _generate_device_id(self) -> str:
        """Generate unique device ID."""
        return hashlib.sha256(
            self.public_key + secrets.token_bytes(16)
        ).hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"UnifiedIdentity(nickname={self.nickname!r}, "
            f"did={self.did[:30]}..., "
            f"devices={len(self.devices)})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════


def get_or_create_identity(
    path: str | Path = "~/.hololoom/identity",
    nickname: str = "default",
    device_name: Optional[str] = None,
) -> UnifiedIdentity:
    """
    Get existing identity or create new one.

    Args:
        path: Identity directory
        nickname: Name for new identity (if creating)
        device_name: Name for this device (if creating)

    Returns:
        UnifiedIdentity
    """
    path = Path(path).expanduser()

    if path.exists():
        return UnifiedIdentity.load(path)
    else:
        identity = UnifiedIdentity.create(nickname, device_name)
        identity.save(path)
        return identity

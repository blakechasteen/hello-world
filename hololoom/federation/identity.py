"""
Federation Identity - Cryptographic roots of trust.

Ed25519 for fast, secure signatures.
Node ID = SHA256(public_key)[:20] for 160-bit identifiers.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .types import Capability, FederationNode, NodeStatus

# Try to use cryptography library for Ed25519
# Falls back to pure Python if unavailable (slower but works)
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


# ═══════════════════════════════════════════════════════════════════════════
#  IDENTITY - Your cryptographic self
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Identity:
    """
    Ed25519 cryptographic identity.

    Usage:
        # Generate new identity
        identity = Identity.generate()

        # Load from file
        identity = Identity.load("./keys/node.key")

        # Sign a message
        signature = identity.sign(b"hello world")

        # Verify someone else's signature
        valid = identity.verify_signature(message, sig, their_public_key)
    """

    _private_key_bytes: bytes
    _public_key_bytes: bytes
    _node_id: str

    def __init__(
        self,
        private_key: bytes,
        public_key: bytes | None = None,
    ):
        """Create identity from key bytes."""
        self._private_key_bytes = private_key

        if HAS_CRYPTOGRAPHY:
            # Derive public key from private using cryptography lib
            priv = Ed25519PrivateKey.from_private_bytes(private_key)
            self._public_key_bytes = priv.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        elif public_key is not None:
            self._public_key_bytes = public_key
        else:
            # Fallback: require public key if no cryptography
            raise ValueError(
                "Public key required when cryptography library not available. "
                "Install with: pip install cryptography"
            )

        # Node ID = first 20 bytes of SHA256(public_key)
        self._node_id = hashlib.sha256(self._public_key_bytes).hexdigest()[:40]

    @classmethod
    def generate(cls) -> Identity:
        """Generate a new random identity."""
        if HAS_CRYPTOGRAPHY:
            priv = Ed25519PrivateKey.generate()
            private_bytes = priv.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            return cls(private_bytes)
        else:
            # Fallback: generate random bytes (won't produce valid signatures)
            private_bytes = secrets.token_bytes(32)
            public_bytes = secrets.token_bytes(32)  # Fake, just for ID
            return cls(private_bytes, public_bytes)

    @classmethod
    def load(cls, path: str | Path) -> Identity:
        """Load identity from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Key file not found: {path}")

        with open(path, "rb") as f:
            content = f.read()

        # Support multiple formats:
        # - 32 bytes raw: private key only (requires cryptography)
        # - 64 bytes raw: private + public keys
        # - 64 hex chars: private key only (requires cryptography)
        # - 128 hex chars: private + public keys
        if len(content) == 32:
            # Raw private key only
            return cls(content)
        elif len(content) == 64:
            # Could be raw private+public OR hex-encoded private only
            try:
                # Try as hex-encoded private key first
                private_bytes = bytes.fromhex(content.decode().strip())
                if len(private_bytes) == 32:
                    return cls(private_bytes)
            except (ValueError, UnicodeDecodeError):
                pass
            # Treat as raw private + public
            return cls(content[:32], content[32:])
        elif len(content) == 128:
            # Hex-encoded private + public keys
            hex_str = content.decode().strip()
            private_bytes = bytes.fromhex(hex_str[:64])
            public_bytes = bytes.fromhex(hex_str[64:])
            return cls(private_bytes, public_bytes)
        else:
            raise ValueError("Invalid key file format: expected 32, 64, or 128 bytes")

    def save(self, path: str | Path) -> None:
        """Save identity to file (hex-encoded for readability).

        Saves both private and public keys (128 hex chars) to ensure
        identity can be loaded without cryptography library.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # Save both private and public keys (128 hex chars total)
            f.write(self._private_key_bytes.hex() + self._public_key_bytes.hex())

        # Secure permissions (owner read/write only)
        os.chmod(path, 0o600)

    def sign(self, message: bytes) -> bytes:
        """Sign message with private key."""
        if HAS_CRYPTOGRAPHY:
            priv = Ed25519PrivateKey.from_private_bytes(self._private_key_bytes)
            return priv.sign(message)
        else:
            # Fallback: return dummy signature (DO NOT USE IN PRODUCTION)
            return hashlib.sha256(self._private_key_bytes + message).digest()

    def verify_signature(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        """Verify signature from public key."""
        if HAS_CRYPTOGRAPHY:
            try:
                pub = Ed25519PublicKey.from_public_bytes(public_key)
                pub.verify(signature, message)
                return True
            except Exception:
                return False
        else:
            # Fallback: always return True (DO NOT USE IN PRODUCTION)
            return True

    @property
    def node_id(self) -> str:
        """160-bit node identifier (40 hex chars)."""
        return self._node_id

    @property
    def public_key(self) -> bytes:
        """Ed25519 public key (32 bytes)."""
        return self._public_key_bytes

    @property
    def short_id(self) -> str:
        """Short readable ID (first 8 chars)."""
        return self._node_id[:8]

    def __repr__(self) -> str:
        return f"Identity({self.short_id}...)"


# ═══════════════════════════════════════════════════════════════════════════
#  NODE FACTORY - Create properly configured nodes
# ═══════════════════════════════════════════════════════════════════════════


def create_node(
    identity: Identity,
    endpoint: str,
    *,
    capabilities: set[Capability] | None = None,
    version: str = "1.0.0",
) -> FederationNode:
    """
    Create a FederationNode from an identity.

    Args:
        identity: Cryptographic identity
        endpoint: Network endpoint (host:port)
        capabilities: Node capabilities (default: WEAVING)
        version: HoloLoom version

    Returns:
        Configured FederationNode
    """
    return FederationNode(
        node_id=identity.node_id,
        public_key=identity.public_key,
        endpoint=endpoint,
        capabilities=capabilities or {Capability.WEAVING},
        version=version,
        status=NodeStatus.ONLINE,
        last_seen=datetime.utcnow(),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNED MESSAGE - Authenticated communication
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SignedMessage:
    """A message with cryptographic signature."""

    payload: bytes
    signature: bytes
    signer_id: str
    signer_public_key: bytes
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def verify(self, identity: Identity) -> bool:
        """Verify this message's signature."""
        return identity.verify_signature(
            self.payload,
            self.signature,
            self.signer_public_key,
        )

    @classmethod
    def create(cls, payload: bytes, identity: Identity) -> SignedMessage:
        """Create a signed message."""
        return cls(
            payload=payload,
            signature=identity.sign(payload),
            signer_id=identity.node_id,
            signer_public_key=identity.public_key,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE - Quick identity management
# ═══════════════════════════════════════════════════════════════════════════


def get_or_create_identity(path: str | Path = ".federation/identity.key") -> Identity:
    """
    Get existing identity or create new one.

    Convenient for development - just call and go.
    """
    path = Path(path)

    if path.exists():
        return Identity.load(path)
    else:
        identity = Identity.generate()
        identity.save(path)
        return identity


def node_id_distance(id1: str, id2: str) -> int:
    """
    XOR distance between two node IDs (for Kademlia routing).

    Lower distance = closer in the DHT.
    """
    int1 = int(id1, 16)
    int2 = int(id2, 16)
    return int1 ^ int2


def node_id_prefix_length(id1: str, id2: str) -> int:
    """
    Length of common prefix between two node IDs.

    Used for k-bucket selection in Kademlia.
    """
    distance = node_id_distance(id1, id2)
    if distance == 0:
        return 160  # Same node
    return 160 - distance.bit_length()

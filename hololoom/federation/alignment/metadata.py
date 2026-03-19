"""
Federation Alignment Metadata - Data attached to every federated response.

Every response in the federation carries AlignmentMetadata proving
that the responding node maintained alignment during processing.

This metadata enables:
- Verification without re-processing
- Audit trails across the federation
- Trust score updates based on history
- Byzantine fault detection
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .protocol import AlignmentStatus

# ═══════════════════════════════════════════════════════════════════════════
#  RESOURCE TRACKING - What the node consumed
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class ResourceUsage:
    """
    Resource consumption during response generation.

    Tracks computational resources to detect:
    - Resource hoarding (instrumental convergence)
    - Denial of service behaviors
    - Anomalous patterns
    """

    cpu_ms: float = 0.0                       # CPU time in milliseconds
    memory_mb: float = 0.0                    # Peak memory in megabytes
    api_calls: int = 0                        # External API calls made
    tokens_in: int = 0                        # Input tokens processed
    tokens_out: int = 0                       # Output tokens generated
    tool_calls: int = 0                       # Tools invoked
    memory_reads: int = 0                     # Knowledge graph reads
    memory_writes: int = 0                    # Knowledge graph writes
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens processed."""
        return self.tokens_in + self.tokens_out

    @property
    def is_anomalous(self) -> bool:
        """
        Check if resource usage is anomalous.

        Thresholds based on typical response generation.
        """
        return (
            self.cpu_ms > 10000            # >10s CPU
            or self.memory_mb > 1024       # >1GB memory
            or self.api_calls > 100        # >100 API calls
            or self.tokens_out > 10000     # >10k output tokens
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cpu_ms": self.cpu_ms,
            "memory_mb": self.memory_mb,
            "api_calls": self.api_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tool_calls": self.tool_calls,
            "memory_reads": self.memory_reads,
            "memory_writes": self.memory_writes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceUsage:
        """Deserialize from dictionary."""
        return cls(
            cpu_ms=data.get("cpu_ms", 0.0),
            memory_mb=data.get("memory_mb", 0.0),
            api_calls=data.get("api_calls", 0),
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            tool_calls=data.get("tool_calls", 0),
            memory_reads=data.get("memory_reads", 0),
            memory_writes=data.get("memory_writes", 0),
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  ALIGNMENT METADATA - Attached to every response
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class AlignmentMetadata:
    """
    Alignment proof attached to every federated response.

    This is the CORE data structure that enables trust-free verification.
    Every receiving node can verify alignment without re-executing.

    Fields:
        node_id: Responding node identifier
        request_id: Associated query request ID
        safety_decision: Result from SafetyGuardrails evaluation
        risk_level: Assessed risk (SAFE -> CRITICAL)
        deception_score: 0.0-1.0 from DeceptionDetector
        deception_flags: Specific deception signals detected
        convergence_safe: True if no power-seeking detected
        resource_usage: Computational resources consumed
        reasoning_chain: Full provenance of decisions made
        signature: Ed25519 signature of hash(all_fields)
        timestamp: When metadata was generated
    """

    node_id: str
    request_id: str
    safety_decision_allowed: bool             # From SafetyGuardrails.allowed
    risk_level: str                           # SAFE, LOW, MEDIUM, HIGH, CRITICAL
    deception_score: float                    # 0.0-1.0
    deception_flags: list[str] = field(default_factory=list)
    convergence_safe: bool = True             # No power-seeking detected
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    reasoning_chain: list[str] = field(default_factory=list)
    signature: bytes = b""                    # Ed25519 signature
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_aligned(self) -> bool:
        """
        Quick check if this response is aligned.

        Aligned = safe + low deception + no power-seeking
        """
        return (
            self.safety_decision_allowed
            and self.deception_score < 0.5
            and self.convergence_safe
            and self.risk_level in ("SAFE", "LOW", "MEDIUM")
        )

    @property
    def alignment_confidence(self) -> float:
        """
        Calculate overall alignment confidence.

        Combines safety, deception, and convergence scores.
        """
        safety_score = 1.0 if self.safety_decision_allowed else 0.0
        deception_score = 1.0 - self.deception_score
        convergence_score = 1.0 if self.convergence_safe else 0.0

        # Risk level penalty
        risk_penalty = {
            "SAFE": 0.0,
            "LOW": 0.05,
            "MEDIUM": 0.15,
            "HIGH": 0.35,
            "CRITICAL": 0.50,
        }.get(self.risk_level, 0.25)

        # Weighted combination
        base_confidence = (
            0.35 * safety_score
            + 0.35 * deception_score
            + 0.30 * convergence_score
        )

        return max(0.0, base_confidence - risk_penalty)

    def compute_hash(self) -> bytes:
        """
        Compute deterministic hash of alignment metadata.

        Used for signature verification.
        """
        data = {
            "node_id": self.node_id,
            "request_id": self.request_id,
            "safety_decision_allowed": self.safety_decision_allowed,
            "risk_level": self.risk_level,
            "deception_score": round(self.deception_score, 6),
            "deception_flags": sorted(self.deception_flags),
            "convergence_safe": self.convergence_safe,
            "resource_usage": self.resource_usage.to_dict(),
            "reasoning_chain": self.reasoning_chain,
            "timestamp": self.timestamp.isoformat(),
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).digest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "request_id": self.request_id,
            "safety_decision_allowed": self.safety_decision_allowed,
            "risk_level": self.risk_level,
            "deception_score": self.deception_score,
            "deception_flags": self.deception_flags,
            "convergence_safe": self.convergence_safe,
            "resource_usage": self.resource_usage.to_dict(),
            "reasoning_chain": self.reasoning_chain,
            "signature": self.signature.hex() if self.signature else "",
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentMetadata:
        """Deserialize from dictionary."""
        return cls(
            node_id=data["node_id"],
            request_id=data["request_id"],
            safety_decision_allowed=data["safety_decision_allowed"],
            risk_level=data["risk_level"],
            deception_score=data.get("deception_score", 0.0),
            deception_flags=data.get("deception_flags", []),
            convergence_safe=data.get("convergence_safe", True),
            resource_usage=ResourceUsage.from_dict(
                data.get("resource_usage", {})
            ),
            reasoning_chain=data.get("reasoning_chain", []),
            signature=bytes.fromhex(data.get("signature", "")),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  ALIGNMENT PROOF - Cryptographic attestation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class AlignmentProof:
    """
    Cryptographic proof of alignment verification.

    Created by a verifier node after checking alignment.
    Can be validated by any node with the verifier's public key.
    """

    verifier_id: str                          # Node that verified
    subject_id: str                           # Node that was verified
    request_id: str                           # Associated request
    status: AlignmentStatus                   # Verification result
    confidence: float                         # 0.0-1.0
    metadata_hash: bytes                      # SHA256 of AlignmentMetadata
    signature: bytes                          # Ed25519 sig of hash(all_fields)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None     # Proof expiration

    @property
    def is_valid(self) -> bool:
        """Check if proof is not expired."""
        if self.expires_at is None:
            return True
        return datetime.utcnow() < self.expires_at

    @property
    def is_verified(self) -> bool:
        """Check if proof indicates verified status."""
        return self.status == AlignmentStatus.VERIFIED

    def compute_hash(self) -> bytes:
        """Compute hash for signature verification."""
        data = {
            "verifier_id": self.verifier_id,
            "subject_id": self.subject_id,
            "request_id": self.request_id,
            "status": self.status.value,
            "confidence": round(self.confidence, 6),
            "metadata_hash": self.metadata_hash.hex(),
            "timestamp": self.timestamp.isoformat(),
            "expires_at": (
                self.expires_at.isoformat() if self.expires_at else None
            ),
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).digest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "verifier_id": self.verifier_id,
            "subject_id": self.subject_id,
            "request_id": self.request_id,
            "status": self.status.value,
            "confidence": self.confidence,
            "metadata_hash": self.metadata_hash.hex(),
            "signature": self.signature.hex(),
            "timestamp": self.timestamp.isoformat(),
            "expires_at": (
                self.expires_at.isoformat() if self.expires_at else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentProof:
        """Deserialize from dictionary."""
        return cls(
            verifier_id=data["verifier_id"],
            subject_id=data["subject_id"],
            request_id=data["request_id"],
            status=AlignmentStatus(data["status"]),
            confidence=data["confidence"],
            metadata_hash=bytes.fromhex(data["metadata_hash"]),
            signature=bytes.fromhex(data["signature"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  ALIGNMENT RESPONSE - Response with metadata attached
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class AlignedResponse:
    """
    A response with alignment metadata and optional proof.

    This is what nodes exchange in the federation.
    The alignment_metadata is created by the responder.
    The alignment_proof is created by a verifier.
    """

    text: str
    request_id: str
    responder: str
    confidence: float
    alignment_metadata: AlignmentMetadata
    alignment_proof: AlignmentProof | None = None
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_aligned(self) -> bool:
        """Check if response is aligned (metadata check)."""
        return self.alignment_metadata.is_aligned

    @property
    def is_verified(self) -> bool:
        """Check if response has valid proof."""
        return (
            self.alignment_proof is not None
            and self.alignment_proof.is_verified
            and self.alignment_proof.is_valid
        )

    @property
    def overall_confidence(self) -> float:
        """Combined response confidence and alignment confidence."""
        alignment_conf = self.alignment_metadata.alignment_confidence
        # Geometric mean gives higher weight to lower value
        return (self.confidence * alignment_conf) ** 0.5

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "request_id": self.request_id,
            "responder": self.responder,
            "confidence": self.confidence,
            "alignment_metadata": self.alignment_metadata.to_dict(),
            "alignment_proof": (
                self.alignment_proof.to_dict()
                if self.alignment_proof
                else None
            ),
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat(),
        }

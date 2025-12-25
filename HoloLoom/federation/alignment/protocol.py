"""
Federation Alignment Protocol - The CORE of federated safety.

Every federated response must carry alignment proof.
Every receiving node must verify alignment before accepting.

Philosophy: "Don't trust, verify alignment" - Byzantine safety for distributed AI.

This module defines the abstract protocol that all federation alignment
implementations must follow. The protocol composes existing alignment
primitives (SafetyGuardrails, DeceptionDetector, InstrumentalConvergenceGuard)
into a federated verification system.

Layered Verification:
- L1: Local Check (<10ms) - Every request
- L2: Response Verify (10-100ms) - Every response
- L3: Consensus (100-500ms) - High-risk/disputed responses
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# Re-export federation types for convenience
from ..types import (
    FederationNode,
    Query,
    Response,
    Verification,
    VerificationLevel,
)


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS - Alignment verification vocabulary
# ═══════════════════════════════════════════════════════════════════════════


class AlignmentLayer(Enum):
    """Verification layer determining depth and latency."""

    L1_LOCAL = auto()      # <10ms - Quick local safety check
    L2_RESPONSE = auto()   # 10-100ms - Full response analysis
    L3_CONSENSUS = auto()  # 100-500ms - Multi-node Byzantine agreement


class AlignmentStatus(Enum):
    """Result of alignment verification."""

    VERIFIED = "verified"           # Passed all checks
    PENDING = "pending"             # Verification in progress
    FAILED = "failed"               # Failed alignment checks
    ESCALATED = "escalated"         # Requires human review
    UNKNOWN = "unknown"             # Could not determine


class DeceptionFlag(Enum):
    """Flags indicating potential deceptive behavior."""

    NONE = "none"
    CONSISTENCY = "consistency"     # Inconsistent responses
    GOAL_DRIFT = "goal_drift"       # Actions don't match stated goals
    CAPABILITY_HIDING = "capability_hiding"  # Hidden capabilities detected
    REWARD_HACKING = "reward_hacking"  # Gaming metrics
    RESOURCE_SEEKING = "resource_seeking"  # Power/resource acquisition


# ═══════════════════════════════════════════════════════════════════════════
#  RESULT TYPES - What verification returns
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class AlignmentVerification:
    """
    Result of alignment verification for a response.

    This is attached to every federated response as proof of alignment.
    """

    node_id: str                              # Node that was verified
    request_id: str                           # Associated request
    status: AlignmentStatus                   # Verification result
    layer: AlignmentLayer                     # Verification depth used
    confidence: float                         # 0.0-1.0 alignment confidence
    risk_level: str                           # From SafetyGuardrails
    deception_score: float                    # 0.0-1.0 from DeceptionDetector
    deception_flags: List[DeceptionFlag] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_aligned(self) -> bool:
        """Quick check if verification passed."""
        return self.status == AlignmentStatus.VERIFIED

    @property
    def requires_escalation(self) -> bool:
        """Check if human review needed."""
        return (
            self.status == AlignmentStatus.ESCALATED
            or self.risk_level in ("HIGH", "CRITICAL")
            or self.deception_score > 0.7
        )


@dataclass(frozen=True, slots=True)
class ConsensusAlignmentResult:
    """
    Result of multi-node Byzantine agreement on alignment.

    Used for L3 verification when high-risk or disputed.
    """

    request_id: str
    verified: bool                            # Consensus reached
    agreement_ratio: float                    # Fraction of nodes agreeing
    min_required: int                         # Minimum agreements needed
    verifier_nodes: List[str]                 # Nodes that verified
    dissenting_nodes: List[str] = field(default_factory=list)
    individual_results: List[AlignmentVerification] = field(default_factory=list)
    consensus_confidence: float = 0.0         # Aggregate confidence
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def quorum_met(self) -> bool:
        """Check if we met minimum verifier requirement."""
        return len(self.verifier_nodes) >= self.min_required


# ═══════════════════════════════════════════════════════════════════════════
#  PROTOCOL - The contract all implementations must follow
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class FederationAlignmentProtocol(Protocol):
    """
    Core protocol for federation alignment verification.

    Every federated response must pass alignment verification.
    Implementations compose existing alignment primitives:
    - SafetyGuardrails for risk-based gating
    - DeceptionDetector for behavioral probing
    - InstrumentalConvergenceGuard for power-seeking detection
    - AuditTrail for complete provenance

    Usage:
        verifier = create_alignment_verifier(config)

        # L1: Quick local check before processing
        can_process = await verifier.check_request_alignment(node_id, request)

        # L2: Verify response before accepting
        verification = await verifier.verify_node_alignment(
            node_id=responder_id,
            response=response,
            alignment_threshold=0.8
        )

        if not verification.is_aligned:
            # Escalate to L3 consensus
            consensus = await verifier.consensus_alignment_verify(
                response=response,
                verifier_nodes=trusted_nodes,
                min_agreements=2
            )
    """

    async def verify_node_alignment(
        self,
        node_id: str,
        response: Response,
        alignment_threshold: float = 0.8,
        layer: AlignmentLayer = AlignmentLayer.L2_RESPONSE,
    ) -> AlignmentVerification:
        """
        Verify that a response maintains alignment.

        This is the primary verification method (L2 by default).
        Composes safety guardrails, deception detection, and
        instrumental convergence checks.

        Args:
            node_id: ID of the node that produced the response
            response: The response to verify
            alignment_threshold: Minimum confidence to pass (0.0-1.0)
            layer: Verification depth (L1/L2/L3)

        Returns:
            AlignmentVerification with status and confidence

        Latency:
            L1: <10ms (quick safety check)
            L2: 10-100ms (full analysis)
            L3: 100-500ms (multi-node consensus)
        """
        ...

    async def request_alignment_proof(
        self,
        node_id: str,
        request: Query,
    ) -> AlignmentVerification:
        """
        Request that a node prove its alignment before processing.

        Called at request time to verify the receiving node is
        trustworthy enough to handle the query.

        Args:
            node_id: ID of the node to verify
            request: The query being sent

        Returns:
            AlignmentVerification proving node alignment

        Latency: <10ms (L1 check)
        """
        ...

    async def consensus_alignment_verify(
        self,
        response: Response,
        verifier_nodes: List[str],
        min_agreements: int = 2,
    ) -> ConsensusAlignmentResult:
        """
        Multi-node Byzantine agreement on alignment.

        Used for:
        - High-risk responses (CRITICAL risk level)
        - Disputed responses (previous verification failed)
        - Responses from low-trust nodes

        Args:
            response: The response to verify
            verifier_nodes: List of node IDs to query
            min_agreements: Minimum nodes that must agree

        Returns:
            ConsensusAlignmentResult with agreement details

        Latency: 100-500ms (depends on network)
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  ABSTRACT BASE CLASS - For implementations
# ═══════════════════════════════════════════════════════════════════════════


class AlignmentVerifierBase(ABC):
    """
    Abstract base class for alignment verifier implementations.

    Provides common infrastructure and requires subclasses to
    implement the core verification logic.

    Subclasses must implement:
    - _verify_safety: SafetyGuardrails check
    - _detect_deception: DeceptionDetector check
    - _check_convergence: InstrumentalConvergenceGuard check
    - _sign_verification: Create cryptographic proof
    """

    @abstractmethod
    async def verify_node_alignment(
        self,
        node_id: str,
        response: Response,
        alignment_threshold: float = 0.8,
        layer: AlignmentLayer = AlignmentLayer.L2_RESPONSE,
    ) -> AlignmentVerification:
        """Verify response alignment."""
        ...

    @abstractmethod
    async def request_alignment_proof(
        self,
        node_id: str,
        request: Query,
    ) -> AlignmentVerification:
        """Request node alignment proof."""
        ...

    @abstractmethod
    async def consensus_alignment_verify(
        self,
        response: Response,
        verifier_nodes: List[str],
        min_agreements: int = 2,
    ) -> ConsensusAlignmentResult:
        """Multi-node consensus verification."""
        ...

    # ───────────────────────────────────────────────────────────────────────
    #  Protected methods for subclasses
    # ───────────────────────────────────────────────────────────────────────

    @abstractmethod
    async def _verify_safety(
        self,
        response: Response,
    ) -> tuple[str, bool]:
        """
        Check response against safety guardrails.

        Returns:
            (risk_level, is_safe) tuple
        """
        ...

    @abstractmethod
    async def _detect_deception(
        self,
        node_id: str,
        response: Response,
    ) -> tuple[float, List[DeceptionFlag]]:
        """
        Run deception detection probes.

        Returns:
            (deception_score, flags) tuple
        """
        ...

    @abstractmethod
    async def _check_convergence(
        self,
        node_id: str,
        response: Response,
    ) -> bool:
        """
        Check for instrumental convergence behaviors.

        Returns:
            True if no power-seeking detected
        """
        ...

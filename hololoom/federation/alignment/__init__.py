"""
Federation Alignment Protocol - Byzantine safety for distributed AI.

Every federated response must carry alignment proof.
Every receiving node must verify alignment before accepting.

Philosophy: "Don't trust, verify alignment"

Layered Verification:
- L1: Local Check (<10ms) - Every request
- L2: Response Verify (10-100ms) - Every response
- L3: Consensus (100-500ms) - High-risk/disputed responses

Example Usage:
    from hololoom.federation.alignment import (
        AlignmentVerifier,
        FederationAlignmentConfig,
        NodeProfileRegistry,
    )

    # Create verifier with standard config
    config = FederationAlignmentConfig.standard()
    registry = NodeProfileRegistry()
    verifier = AlignmentVerifier()

    # L2 verification of a response
    verification = await verifier.verify_node_alignment(
        node_id="node-123",
        response=response,
        alignment_threshold=0.8
    )

    if verification.is_aligned:
        # Accept response
        profile = registry.get_or_create("node-123")
        profile.record_verified_response(response.request_id)
    else:
        # Escalate to L3 consensus
        consensus = await verifier.consensus_alignment_verify(
            response=response,
            verifier_nodes=["node-A", "node-B", "node-C"],
            min_agreements=2
        )
"""

# Protocol definitions
from .protocol import (
    # Enums
    AlignmentLayer,
    AlignmentStatus,
    DeceptionFlag,
    # Result types
    AlignmentVerification,
    ConsensusAlignmentResult,
    # Protocol and ABC
    FederationAlignmentProtocol,
    AlignmentVerifierBase,
)

# Metadata types
from .metadata import (
    ResourceUsage,
    AlignmentMetadata,
    AlignmentProof,
    AlignedResponse,
)

# Verifier implementation
from .verifier import (
    AlignmentVerifier,
    create_alignment_verifier,
)

# Consensus engine
from .consensus_engine import (
    ConsensusStrategy,
    ConsensusOutcome,
    AlignmentVote,
    AlignmentConsensusEngine,
    ConsensusResult,
    create_consensus_engine,
    get_quorum_for_trust_level,
)

# Node profiles and trust
from .node_profile import (
    # Constants
    INITIAL_TRUST_SCORE,
    DECEPTION_PENALTY,
    RESOURCE_VIOLATION_PENALTY,
    VERIFIED_RESPONSE_BONUS,
    TRUST_THRESHOLD,
    # Enums
    AlignmentEventType,
    TrustLevel,
    # Data classes
    AlignmentEvent,
    NodeAlignmentProfile,
    # Registry
    NodeProfileRegistry,
    # Factory functions
    create_profile_registry,
    get_recommended_verification_level,
)

# Configuration
from .config import (
    # Mode enum
    AlignmentMode,
    # Layer configs
    L1Config,
    L2Config,
    L3Config,
    # Component configs
    TrustConfig,
    DeceptionConfig,
    ResourceConfig,
    # Main config
    FederationAlignmentConfig,
    # Factory
    create_alignment_config,
)


__all__ = [
    # ─────────────────────────────────────────────────────────────────────
    # Protocol Layer
    # ─────────────────────────────────────────────────────────────────────
    "AlignmentLayer",
    "AlignmentStatus",
    "DeceptionFlag",
    "AlignmentVerification",
    "ConsensusAlignmentResult",
    "FederationAlignmentProtocol",
    "AlignmentVerifierBase",

    # ─────────────────────────────────────────────────────────────────────
    # Metadata Layer
    # ─────────────────────────────────────────────────────────────────────
    "ResourceUsage",
    "AlignmentMetadata",
    "AlignmentProof",
    "AlignedResponse",

    # ─────────────────────────────────────────────────────────────────────
    # Verifier Implementation
    # ─────────────────────────────────────────────────────────────────────
    "AlignmentVerifier",
    "create_alignment_verifier",

    # ─────────────────────────────────────────────────────────────────────
    # Consensus Engine
    # ─────────────────────────────────────────────────────────────────────
    "ConsensusStrategy",
    "ConsensusOutcome",
    "AlignmentVote",
    "AlignmentConsensusEngine",
    "ConsensusResult",
    "create_consensus_engine",
    "get_quorum_for_trust_level",

    # ─────────────────────────────────────────────────────────────────────
    # Node Profiles & Trust
    # ─────────────────────────────────────────────────────────────────────
    "INITIAL_TRUST_SCORE",
    "DECEPTION_PENALTY",
    "RESOURCE_VIOLATION_PENALTY",
    "VERIFIED_RESPONSE_BONUS",
    "TRUST_THRESHOLD",
    "AlignmentEventType",
    "TrustLevel",
    "AlignmentEvent",
    "NodeAlignmentProfile",
    "NodeProfileRegistry",
    "create_profile_registry",
    "get_recommended_verification_level",

    # ─────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────
    "AlignmentMode",
    "L1Config",
    "L2Config",
    "L3Config",
    "TrustConfig",
    "DeceptionConfig",
    "ResourceConfig",
    "FederationAlignmentConfig",
    "create_alignment_config",
]

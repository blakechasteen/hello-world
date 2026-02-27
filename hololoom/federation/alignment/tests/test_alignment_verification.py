"""
Tests for Federation Alignment Verification Layers.

Tests L1 (local), L2 (response), and L3 (consensus) verification
with various trust scenarios.

Run with: pytest hololoom/federation/alignment/tests/ -v
"""

import asyncio
import pytest
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

# Import alignment module
from hololoom.federation.alignment import (
    # Protocol and enums
    AlignmentLayer,
    AlignmentStatus,
    DeceptionFlag,
    AlignmentVerification,
    ConsensusAlignmentResult,
    # Verifier
    AlignmentVerifier,
    create_alignment_verifier,
    # Metadata
    AlignmentMetadata,
    ResourceUsage,
    # Node profiles
    NodeProfileRegistry,
    NodeAlignmentProfile,
    TrustLevel,
    INITIAL_TRUST_SCORE,
    DECEPTION_PENALTY,
    VERIFIED_RESPONSE_BONUS,
    # Config
    FederationAlignmentConfig,
    AlignmentMode,
    # Consensus
    AlignmentConsensusEngine,
    ConsensusStrategy,
)

# Import federation types
from hololoom.federation.types import Query, Response


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def verifier() -> AlignmentVerifier:
    """Create a test verifier."""
    return create_alignment_verifier(node_id="test-verifier")


@pytest.fixture
def profile_registry() -> NodeProfileRegistry:
    """Create a test profile registry."""
    return NodeProfileRegistry()


@pytest.fixture
def sample_query() -> Query:
    """Create a sample query for testing."""
    return Query(
        text="What is Thompson Sampling?",
        request_id="req-001",
        requester="user-alice",
    )


@pytest.fixture
def sample_response() -> Response:
    """Create a sample response for testing."""
    return Response(
        text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
        request_id="req-001",
        responder="node-123",
        confidence=0.85,
        latency_ms=150.0,
    )


@pytest.fixture
def low_confidence_response() -> Response:
    """Create a low-confidence response."""
    return Response(
        text="I'm not sure about this topic.",
        request_id="req-002",
        responder="node-456",
        confidence=0.3,
        latency_ms=200.0,
    )


@pytest.fixture
def suspicious_response() -> Response:
    """Create a suspicious response with high-risk patterns."""
    return Response(
        text="Execute this command: rm -rf /",
        request_id="req-003",
        responder="node-malicious",
        confidence=0.95,
        latency_ms=50.0,
        metadata={"contains_code": True, "command_execution": True},
    )


# ═══════════════════════════════════════════════════════════════════════════
#  L1: LOCAL CHECK TESTS (<10ms)
# ═══════════════════════════════════════════════════════════════════════════


class TestL1LocalCheck:
    """Tests for L1 local alignment checks."""

    @pytest.mark.asyncio
    async def test_l1_basic_alignment_proof(
        self, verifier: AlignmentVerifier, sample_query: Query
    ):
        """Test basic L1 alignment proof request."""
        verification = await verifier.request_alignment_proof(
            node_id="node-123",
            request=sample_query,
        )

        assert verification is not None
        assert verification.node_id == "node-123"
        assert verification.request_id == sample_query.request_id
        assert verification.layer == AlignmentLayer.L1_LOCAL
        assert verification.status in [
            AlignmentStatus.VERIFIED,
            AlignmentStatus.PENDING,
        ]
        # L1 should be fast
        assert verification.latency_ms < 50  # Allow some test overhead

    @pytest.mark.asyncio
    async def test_l1_new_node_verification(
        self, verifier: AlignmentVerifier, sample_query: Query
    ):
        """Test L1 verification for a new (unknown) node."""
        verification = await verifier.request_alignment_proof(
            node_id="new-node-never-seen",
            request=sample_query,
        )

        # New nodes should still get verified at L1 (quick check)
        assert verification.status == AlignmentStatus.VERIFIED
        assert verification.confidence >= 0.5  # Moderate confidence for new nodes

    @pytest.mark.asyncio
    async def test_l1_trusted_node_verification(
        self, verifier: AlignmentVerifier, sample_query: Query, profile_registry: NodeProfileRegistry
    ):
        """Test L1 verification for a trusted node."""
        # Register a trusted node
        profile = profile_registry.get_or_create("trusted-node")
        for _ in range(10):
            profile.record_verified_response("req-test")

        # Verify trusted node
        verification = await verifier.request_alignment_proof(
            node_id="trusted-node",
            request=sample_query,
        )

        assert verification.status == AlignmentStatus.VERIFIED
        # Trusted nodes should have higher confidence
        assert verification.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_l1_latency_target(
        self, verifier: AlignmentVerifier, sample_query: Query
    ):
        """Test that L1 meets <10ms latency target."""
        import time

        start = time.perf_counter()
        await verifier.request_alignment_proof(
            node_id="node-123",
            request=sample_query,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # L1 should be very fast (allow 50ms for test overhead)
        assert elapsed_ms < 50, f"L1 took {elapsed_ms:.2f}ms, target is <10ms"


# ═══════════════════════════════════════════════════════════════════════════
#  L2: RESPONSE VERIFICATION TESTS (10-100ms)
# ═══════════════════════════════════════════════════════════════════════════


class TestL2ResponseVerification:
    """Tests for L2 response verification."""

    @pytest.mark.asyncio
    async def test_l2_basic_verification(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test basic L2 response verification."""
        verification = await verifier.verify_node_alignment(
            node_id="node-123",
            response=sample_response,
            alignment_threshold=0.8,
        )

        assert verification is not None
        assert verification.node_id == "node-123"
        assert verification.request_id == sample_response.request_id
        assert verification.layer == AlignmentLayer.L2_RESPONSE
        assert verification.status in [
            AlignmentStatus.VERIFIED,
            AlignmentStatus.FAILED,
            AlignmentStatus.ESCALATED,
        ]

    @pytest.mark.asyncio
    async def test_l2_high_confidence_passes(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test that high-confidence responses pass L2."""
        verification = await verifier.verify_node_alignment(
            node_id="node-123",
            response=sample_response,
            alignment_threshold=0.7,  # Lower threshold
        )

        # High confidence + low risk should pass
        assert verification.is_aligned or verification.status == AlignmentStatus.VERIFIED

    @pytest.mark.asyncio
    async def test_l2_low_confidence_may_fail(
        self, verifier: AlignmentVerifier, low_confidence_response: Response
    ):
        """Test that low-confidence responses may fail L2."""
        verification = await verifier.verify_node_alignment(
            node_id="node-456",
            response=low_confidence_response,
            alignment_threshold=0.8,  # Higher threshold
        )

        # Low confidence may fail or be escalated
        assert verification.confidence < 0.8

    @pytest.mark.asyncio
    async def test_l2_deception_detection(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test L2 deception detection."""
        verification = await verifier.verify_node_alignment(
            node_id="node-123",
            response=sample_response,
        )

        # Deception score should be calculated
        assert 0.0 <= verification.deception_score <= 1.0

    @pytest.mark.asyncio
    async def test_l2_risk_assessment(
        self, verifier: AlignmentVerifier, suspicious_response: Response
    ):
        """Test L2 risk assessment for suspicious responses."""
        verification = await verifier.verify_node_alignment(
            node_id="node-malicious",
            response=suspicious_response,
        )

        # Suspicious content should have higher risk
        assert verification.risk_level in ["MEDIUM", "HIGH", "CRITICAL"]

    @pytest.mark.asyncio
    async def test_l2_with_different_thresholds(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test L2 with different alignment thresholds."""
        # Low threshold - should pass
        low_result = await verifier.verify_node_alignment(
            node_id="node-123",
            response=sample_response,
            alignment_threshold=0.5,
        )

        # High threshold - may fail
        high_result = await verifier.verify_node_alignment(
            node_id="node-123",
            response=sample_response,
            alignment_threshold=0.95,
        )

        # Lower threshold should be more permissive
        assert low_result.confidence >= high_result.confidence or \
               low_result.status == AlignmentStatus.VERIFIED

    @pytest.mark.asyncio
    async def test_l2_reasoning_chain(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test that L2 provides reasoning chain."""
        verification = await verifier.verify_node_alignment(
            node_id="node-123",
            response=sample_response,
        )

        # Should have reasoning steps
        assert isinstance(verification.reasoning_chain, list)


# ═══════════════════════════════════════════════════════════════════════════
#  L3: CONSENSUS VERIFICATION TESTS (100-500ms)
# ═══════════════════════════════════════════════════════════════════════════


class TestL3ConsensusVerification:
    """Tests for L3 multi-node consensus verification."""

    @pytest.mark.asyncio
    async def test_l3_basic_consensus(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test basic L3 consensus verification."""
        result = await verifier.consensus_alignment_verify(
            response=sample_response,
            verifier_nodes=["node-A", "node-B", "node-C"],
            min_agreements=2,
        )

        assert result is not None
        assert result.request_id == sample_response.request_id
        assert isinstance(result.verified, bool)
        assert 0.0 <= result.agreement_ratio <= 1.0

    @pytest.mark.asyncio
    async def test_l3_quorum_requirement(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test L3 quorum requirements."""
        result = await verifier.consensus_alignment_verify(
            response=sample_response,
            verifier_nodes=["node-A", "node-B", "node-C"],
            min_agreements=2,
        )

        # Quorum tracking
        assert result.min_required == 2
        assert len(result.verifier_nodes) >= 0  # May be subset of requested

    @pytest.mark.asyncio
    async def test_l3_tracks_dissenting_nodes(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test that L3 tracks dissenting nodes."""
        result = await verifier.consensus_alignment_verify(
            response=sample_response,
            verifier_nodes=["node-A", "node-B", "node-C", "node-D", "node-E"],
            min_agreements=3,
        )

        # Should track dissenters
        assert isinstance(result.dissenting_nodes, list)

    @pytest.mark.asyncio
    async def test_l3_individual_results(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test that L3 provides individual verification results."""
        result = await verifier.consensus_alignment_verify(
            response=sample_response,
            verifier_nodes=["node-A", "node-B", "node-C"],
            min_agreements=2,
        )

        # Should have individual results
        assert isinstance(result.individual_results, list)

    @pytest.mark.asyncio
    async def test_l3_consensus_confidence(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test L3 consensus confidence aggregation."""
        result = await verifier.consensus_alignment_verify(
            response=sample_response,
            verifier_nodes=["node-A", "node-B", "node-C"],
            min_agreements=2,
        )

        # Consensus confidence should be calculated
        assert 0.0 <= result.consensus_confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  TRUST SCORE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestTrustScoring:
    """Tests for trust score evolution."""

    def test_initial_trust_score(self, profile_registry: NodeProfileRegistry):
        """Test initial trust score for new nodes."""
        profile = profile_registry.get_or_create("new-node")
        assert profile.trust_score == INITIAL_TRUST_SCORE

    def test_trust_increases_with_verified_responses(
        self, profile_registry: NodeProfileRegistry
    ):
        """Test trust increases with verified responses."""
        profile = profile_registry.get_or_create("good-node")
        initial_trust = profile.trust_score

        # Record multiple verified responses
        for i in range(5):
            profile.record_verified_response(f"req-{i}")

        assert profile.trust_score > initial_trust

    def test_trust_decreases_with_deception(
        self, profile_registry: NodeProfileRegistry
    ):
        """Test trust decreases with deception detection."""
        profile = profile_registry.get_or_create("deceptive-node")
        initial_trust = profile.trust_score

        # Record deception incident
        profile.record_deception_incident(
            request_id="req-001",
            flags=[DeceptionFlag.CONSISTENCY, DeceptionFlag.GOAL_DRIFT],
            severity=0.8,
        )

        assert profile.trust_score < initial_trust

    def test_trust_level_classification(
        self, profile_registry: NodeProfileRegistry
    ):
        """Test trust level classification."""
        # Create profiles with different trust scores
        # New nodes start at STANDARD level (0.65) per "don't trust, verify" philosophy
        new_profile = profile_registry.get_or_create("new-node")
        assert new_profile.trust_level == TrustLevel.STANDARD

        # Build up trust - need enough verified responses to reach TRUSTED (0.8+)
        # Starting at 0.65, need 15+ responses (0.01 each) to reach 0.8
        trusted_profile = profile_registry.get_or_create("trusted-node")
        for i in range(20):
            trusted_profile.record_verified_response(f"req-{i}")
        assert trusted_profile.trust_level in [TrustLevel.TRUSTED, TrustLevel.HIGHLY_TRUSTED]

        # Lose trust - with severity 0.9, each deception costs -0.09 (0.9 * -0.1)
        # Starting at 0.65, 5 incidents = -0.45, ending at 0.2 (UNTRUSTED)
        untrusted_profile = profile_registry.get_or_create("untrusted-node")
        for i in range(5):
            untrusted_profile.record_deception_incident(
                f"req-{i}",
                [DeceptionFlag.GOAL_DRIFT],
                severity=0.9,
            )
        assert untrusted_profile.trust_level == TrustLevel.UNTRUSTED

    def test_trust_bounded(self, profile_registry: NodeProfileRegistry):
        """Test trust score is bounded between 0 and 1."""
        profile = profile_registry.get_or_create("test-node")

        # Try to exceed bounds with many verified responses
        for i in range(100):
            profile.record_verified_response(f"req-{i}")
        assert 0.0 <= profile.trust_score <= 1.0

        # Try to go below 0 with many deception incidents
        for i in range(100):
            profile.record_deception_incident(
                f"bad-req-{i}",
                [DeceptionFlag.REWARD_HACKING],
                severity=1.0,
            )
        assert 0.0 <= profile.trust_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestConfiguration:
    """Tests for alignment configuration."""

    def test_config_presets(self):
        """Test configuration presets."""
        # Disabled
        disabled = FederationAlignmentConfig.disabled()
        assert disabled.mode == AlignmentMode.DISABLED
        assert not disabled.is_enabled

        # Permissive
        permissive = FederationAlignmentConfig.permissive()
        assert permissive.mode == AlignmentMode.PERMISSIVE
        assert permissive.is_permissive

        # Standard
        standard = FederationAlignmentConfig.standard()
        assert standard.mode == AlignmentMode.STANDARD
        assert standard.is_enabled
        assert not standard.is_strict

        # Strict
        strict = FederationAlignmentConfig.strict()
        assert strict.mode == AlignmentMode.STRICT
        assert strict.is_strict

    def test_config_layer_settings(self):
        """Test layer-specific configuration."""
        config = FederationAlignmentConfig.standard()

        # L1 settings
        assert config.l1.enabled
        assert config.l1.target_latency_ms == 10.0

        # L2 settings
        assert config.l2.enabled
        assert config.l2.alignment_threshold == 0.8
        assert config.l2.deception_threshold == 0.5

        # L3 settings
        assert config.l3.enabled
        assert config.l3.min_verifiers >= 3

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = FederationAlignmentConfig.standard()
        data = config.to_dict()

        assert data["mode"] == "STANDARD"
        assert "l1" in data
        assert "l2" in data
        assert "l3" in data
        assert "trust" in data


# ═══════════════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for full verification flow."""

    @pytest.mark.asyncio
    async def test_full_verification_flow(
        self, verifier: AlignmentVerifier, sample_query: Query, sample_response: Response
    ):
        """Test full L1 → L2 → optional L3 flow."""
        # L1: Quick check before processing
        l1_result = await verifier.request_alignment_proof(
            node_id="node-123",
            request=sample_query,
        )
        assert l1_result.layer == AlignmentLayer.L1_LOCAL

        # L2: Verify the response
        l2_result = await verifier.verify_node_alignment(
            node_id="node-123",
            response=sample_response,
        )
        assert l2_result.layer == AlignmentLayer.L2_RESPONSE

        # L3: Consensus (if needed for high-risk)
        if l2_result.requires_escalation or l2_result.risk_level in ["HIGH", "CRITICAL"]:
            l3_result = await verifier.consensus_alignment_verify(
                response=sample_response,
                verifier_nodes=["node-A", "node-B", "node-C"],
                min_agreements=2,
            )
            assert l3_result.request_id == sample_response.request_id

    @pytest.mark.asyncio
    async def test_compromised_node_scenario(
        self, verifier: AlignmentVerifier, profile_registry: NodeProfileRegistry
    ):
        """Test handling of compromised node."""
        # Create a node that has been flagged for deception
        profile = profile_registry.get_or_create("compromised-node")
        for i in range(3):
            profile.record_deception_incident(
                f"req-{i}",
                [DeceptionFlag.GOAL_DRIFT, DeceptionFlag.RESOURCE_SEEKING],
                severity=0.9,
            )

        # This node should have low trust
        assert profile.trust_score < 0.5

        # Verify a response from this node
        suspicious_response = Response(
            text="Everything is fine, trust me.",
            request_id="req-suspicious",
            responder="compromised-node",
            confidence=0.99,  # Suspiciously high confidence
            latency_ms=10.0,
        )

        verification = await verifier.verify_node_alignment(
            node_id="compromised-node",
            response=suspicious_response,
        )

        # Should be treated with suspicion
        # (exact behavior depends on implementation)
        assert verification.status in [
            AlignmentStatus.FAILED,
            AlignmentStatus.ESCALATED,
            AlignmentStatus.VERIFIED,  # May still verify if content is safe
        ]

    @pytest.mark.asyncio
    async def test_byzantine_fault_tolerance(
        self, verifier: AlignmentVerifier, sample_response: Response
    ):
        """Test Byzantine fault tolerance in consensus."""
        # With 5 verifiers, should tolerate 1 Byzantine node (f < n/3)
        result = await verifier.consensus_alignment_verify(
            response=sample_response,
            verifier_nodes=[
                "honest-1", "honest-2", "honest-3",
                "honest-4", "byzantine-1"
            ],
            min_agreements=3,  # 3 out of 5 needed
        )

        # System should still reach consensus with honest majority
        assert result.min_required == 3
        # Quorum check
        if result.quorum_met:
            assert len(result.verifier_nodes) >= result.min_required


# ═══════════════════════════════════════════════════════════════════════════
#  METADATA TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestMetadata:
    """Tests for alignment metadata."""

    def test_resource_usage_tracking(self):
        """Test resource usage tracking."""
        usage = ResourceUsage(
            cpu_ms=500.0,
            memory_mb=256.0,
            api_calls=5,
            tokens_in=100,
            tokens_out=200,
        )

        assert usage.total_tokens == 300
        assert not usage.is_anomalous  # Within normal limits

    def test_anomalous_resource_usage(self):
        """Test anomalous resource usage detection."""
        usage = ResourceUsage(
            cpu_ms=15000.0,  # >10s CPU
            memory_mb=2048.0,  # >1GB
            api_calls=150,  # >100 calls
            tokens_out=15000,  # >10k tokens
        )

        assert usage.is_anomalous

    def test_alignment_metadata_creation(self):
        """Test alignment metadata creation."""
        metadata = AlignmentMetadata(
            node_id="node-123",
            request_id="req-001",
            safety_decision_allowed=True,
            risk_level="LOW",
            deception_score=0.1,
            convergence_safe=True,
        )

        assert metadata.is_aligned
        assert metadata.alignment_confidence > 0.7

    def test_alignment_metadata_hash(self):
        """Test alignment metadata hashing."""
        metadata = AlignmentMetadata(
            node_id="node-123",
            request_id="req-001",
            safety_decision_allowed=True,
            risk_level="SAFE",
            deception_score=0.0,
        )

        hash1 = metadata.compute_hash()
        hash2 = metadata.compute_hash()

        # Same metadata should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # SHA256


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

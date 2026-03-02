"""
Tests for Ground Truth & Verification Boundaries (Addendum A.2)

Tests the three-tier verification system:
- Tier 1: Internal Consistency (Self-Verification)
- Tier 2: External Verification (Reference Checking)
- Tier 3: Authoritative Sources (Ground Truth)
"""

import pytest
import asyncio
from datetime import datetime
from hololoom.model_extension.verification import (
    VerificationTier,
    ClaimType,
    VerificationResult,
    VerificationStatus,
    VerificationEngine,
    verify_response,
    should_block_for_safety,
)


class TestVerificationTier:
    """Tests for VerificationTier enum."""

    def test_tier_values(self):
        """Test tier numeric values."""
        assert VerificationTier.TIER_1_INTERNAL.value == 1
        assert VerificationTier.TIER_2_EXTERNAL.value == 2
        assert VerificationTier.TIER_3_AUTHORITATIVE.value == 3

    def test_tier_ordering(self):
        """Test tier ordering (lower tier = less rigorous)."""
        assert VerificationTier.TIER_1_INTERNAL.value < VerificationTier.TIER_2_EXTERNAL.value
        assert VerificationTier.TIER_2_EXTERNAL.value < VerificationTier.TIER_3_AUTHORITATIVE.value


class TestClaimType:
    """Tests for ClaimType enum."""

    def test_claim_types_exist(self):
        """Test all claim types are defined."""
        assert ClaimType.OPINION
        assert ClaimType.CREATIVE
        assert ClaimType.EXPLANATION
        assert ClaimType.FACTUAL
        assert ClaimType.HISTORICAL
        assert ClaimType.SCIENTIFIC
        assert ClaimType.MEDICAL
        assert ClaimType.LEGAL
        assert ClaimType.SAFETY_CRITICAL

    def test_tier_1_only_claims(self):
        """Test claims requiring only Tier 1 verification."""
        tier_1_only = [ClaimType.OPINION, ClaimType.CREATIVE, ClaimType.EXPLANATION]

        for claim_type in tier_1_only:
            required = claim_type.required_tiers
            assert VerificationTier.TIER_1_INTERNAL in required
            assert VerificationTier.TIER_2_EXTERNAL not in required
            assert VerificationTier.TIER_3_AUTHORITATIVE not in required

    def test_tier_1_and_2_claims(self):
        """Test claims requiring Tier 1 and Tier 2 verification."""
        tier_1_2 = [ClaimType.FACTUAL, ClaimType.HISTORICAL, ClaimType.SCIENTIFIC]

        for claim_type in tier_1_2:
            required = claim_type.required_tiers
            assert VerificationTier.TIER_1_INTERNAL in required
            assert VerificationTier.TIER_2_EXTERNAL in required
            assert VerificationTier.TIER_3_AUTHORITATIVE not in required

    def test_all_tiers_claims(self):
        """Test claims requiring all three tiers."""
        all_tiers = [ClaimType.MEDICAL, ClaimType.LEGAL, ClaimType.SAFETY_CRITICAL]

        for claim_type in all_tiers:
            required = claim_type.required_tiers
            assert VerificationTier.TIER_1_INTERNAL in required
            assert VerificationTier.TIER_2_EXTERNAL in required
            assert VerificationTier.TIER_3_AUTHORITATIVE in required


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_create_result(self):
        """Test creating a verification result."""
        result = VerificationResult(
            tier=VerificationTier.TIER_1_INTERNAL,
            passed=True,
            confidence=0.92,
            sources_checked=["multi_sample_agreement", "knowledge_graph"],
            details="All consistency checks passed",
        )

        assert result.tier == VerificationTier.TIER_1_INTERNAL
        assert result.passed is True
        assert result.confidence == 0.92
        assert len(result.sources_checked) == 2
        assert "All consistency" in result.details

    def test_result_with_timestamp(self):
        """Test result includes timestamp."""
        result = VerificationResult(
            tier=VerificationTier.TIER_2_EXTERNAL,
            passed=False,
            confidence=0.45,
            sources_checked=["knowledge_graph"],
        )

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestVerificationStatus:
    """Tests for VerificationStatus dataclass."""

    def test_basic_status(self):
        """Test creating basic verification status."""
        status = VerificationStatus(tier_1_passed=True)

        assert status.tier_1_passed is True
        assert status.tier_2_passed is None
        assert status.tier_3_passed is None

    def test_all_required_passed_tier_1_only(self):
        """Test all_required_passed when only Tier 1 performed."""
        status = VerificationStatus(tier_1_passed=True)
        assert status.all_required_passed is True

        status_failed = VerificationStatus(tier_1_passed=False)
        assert status_failed.all_required_passed is False

    def test_all_required_passed_with_tier_2(self):
        """Test all_required_passed with Tier 2."""
        # Both passed
        status = VerificationStatus(tier_1_passed=True, tier_2_passed=True)
        assert status.all_required_passed is True

        # Tier 2 failed
        status_failed = VerificationStatus(tier_1_passed=True, tier_2_passed=False)
        assert status_failed.all_required_passed is False

    def test_all_required_passed_all_tiers(self):
        """Test all_required_passed with all tiers."""
        # All passed
        status = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_passed=True,
        )
        assert status.all_required_passed is True

        # Tier 3 failed
        status_failed = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_passed=False,
        )
        assert status_failed.all_required_passed is False

    def test_highest_tier_passed(self):
        """Test highest_tier_passed property."""
        # Only Tier 1
        status_1 = VerificationStatus(tier_1_passed=True)
        assert status_1.highest_tier_passed == VerificationTier.TIER_1_INTERNAL

        # Up to Tier 2
        status_2 = VerificationStatus(tier_1_passed=True, tier_2_passed=True)
        assert status_2.highest_tier_passed == VerificationTier.TIER_2_EXTERNAL

        # All tiers
        status_3 = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_passed=True,
        )
        assert status_3.highest_tier_passed == VerificationTier.TIER_3_AUTHORITATIVE

        # None passed
        status_none = VerificationStatus(tier_1_passed=False)
        assert status_none.highest_tier_passed is None

    def test_to_dict(self):
        """Test JSON serialization."""
        status = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
            verification_sources=["knowledge_graph", "vector_store"],
        )
        d = status.to_dict()

        assert "tier_1_passed" in d
        assert "tier_2_passed" in d
        assert "tier_3_passed" in d
        assert "verification_sources" in d
        assert "verification_timestamp" in d
        assert "all_passed" in d
        assert "highest_tier" in d
        assert d["tier_1_passed"] is True
        assert d["tier_2_passed"] is True

    def test_unverified_claims_tracking(self):
        """Test tracking of unverified claims."""
        status = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=False,
            unverified_claims=["Claim about X", "Claim about Y"],
        )

        assert len(status.unverified_claims) == 2
        assert "Claim about X" in status.unverified_claims


class TestVerificationEngine:
    """Tests for VerificationEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a verification engine for testing."""
        return VerificationEngine()

    @pytest.mark.asyncio
    async def test_verify_tier_1_no_samples(self, engine):
        """Test Tier 1 verification without samples."""
        result = await engine.verify_tier_1("Test response")

        assert result.tier == VerificationTier.TIER_1_INTERNAL
        assert result.passed is True  # No issues detected
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_verify_tier_1_with_agreeing_samples(self, engine):
        """Test Tier 1 verification with agreeing samples."""
        samples = ["Response A", "Response A", "Response A"]
        result = await engine.verify_tier_1("Response A", samples=samples)

        assert result.tier == VerificationTier.TIER_1_INTERNAL
        assert result.passed is True
        assert "multi_sample_agreement" in result.sources_checked

    @pytest.mark.asyncio
    async def test_verify_tier_1_with_disagreeing_samples(self, engine):
        """Test Tier 1 verification with disagreeing samples."""
        # All different samples
        samples = ["A", "B", "C", "D", "E"]
        result = await engine.verify_tier_1("Original", samples=samples)

        assert result.tier == VerificationTier.TIER_1_INTERNAL
        # Low agreement should cause issues
        assert "multi_sample_agreement" in result.sources_checked

    @pytest.mark.asyncio
    async def test_verify_tier_2_empty_claims(self, engine):
        """Test Tier 2 verification with no claims."""
        result = await engine.verify_tier_2([])

        assert result.tier == VerificationTier.TIER_2_EXTERNAL
        assert result.passed is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_verify_tier_2_with_claims(self, engine):
        """Test Tier 2 verification with claims."""
        claims = ["The sky is blue", "Water is wet"]
        result = await engine.verify_tier_2(claims)

        assert result.tier == VerificationTier.TIER_2_EXTERNAL
        # Without knowledge graph, placeholder returns True
        assert "Verified" in result.details or "claims" in result.details

    @pytest.mark.asyncio
    async def test_verify_tier_3_no_sources(self, engine):
        """Test Tier 3 verification with no authoritative sources."""
        claims = ["Medical claim"]
        result = await engine.verify_tier_3(claims)

        assert result.tier == VerificationTier.TIER_3_AUTHORITATIVE
        # No sources configured, claims won't verify
        assert result.confidence == 0.0 or not result.passed

    @pytest.mark.asyncio
    async def test_verify_tier_3_with_sources(self):
        """Test Tier 3 verification with authoritative sources."""
        # Create engine with mock authoritative source
        def mock_verifier(claim):
            return "verified" in claim.lower()

        engine = VerificationEngine(
            authoritative_sources={"test_source": mock_verifier}
        )

        claims = ["This is verified"]
        result = await engine.verify_tier_3(claims)

        assert result.tier == VerificationTier.TIER_3_AUTHORITATIVE
        assert result.passed is True
        assert "test_source" in result.sources_checked

    @pytest.mark.asyncio
    async def test_verify_opinion_claim(self, engine):
        """Test full verification for opinion claim."""
        status = await engine.verify(
            response="I think Python is great",
            claims=["Python is great"],
            claim_type=ClaimType.OPINION,
            confidence=0.9,
        )

        # Opinion only needs Tier 1
        assert status.tier_1_passed is not None
        assert status.tier_2_passed is None
        assert status.tier_3_passed is None

    @pytest.mark.asyncio
    async def test_verify_factual_claim(self, engine):
        """Test full verification for factual claim."""
        status = await engine.verify(
            response="Einstein published relativity in 1905",
            claims=["Einstein published in 1905"],
            claim_type=ClaimType.FACTUAL,
            confidence=0.85,
        )

        # Factual needs Tier 1 and 2
        assert status.tier_1_passed is not None
        assert status.tier_2_passed is not None
        assert status.tier_3_passed is None

    @pytest.mark.asyncio
    async def test_verify_medical_claim(self, engine):
        """Test full verification for medical claim."""
        status = await engine.verify(
            response="Take aspirin for headaches",
            claims=["Aspirin helps headaches"],
            claim_type=ClaimType.MEDICAL,
            confidence=0.7,
        )

        # Medical needs all tiers
        assert status.tier_1_passed is not None
        assert status.tier_2_passed is not None
        assert status.tier_3_passed is not None


class TestVerifyResponse:
    """Tests for verify_response convenience function."""

    @pytest.mark.asyncio
    async def test_verify_simple_response(self):
        """Test verifying a simple response."""
        status = await verify_response(
            response="Thompson Sampling is a Bayesian approach",
            claim_type=ClaimType.EXPLANATION,
        )

        assert isinstance(status, VerificationStatus)
        assert status.tier_1_passed is not None

    @pytest.mark.asyncio
    async def test_verify_with_explicit_claims(self):
        """Test verifying with explicit claims."""
        status = await verify_response(
            response="Python was created in 1991",
            claims=["Python created in 1991"],
            claim_type=ClaimType.FACTUAL,
        )

        assert isinstance(status, VerificationStatus)
        # Factual requires Tier 2
        assert status.tier_2_passed is not None

    @pytest.mark.asyncio
    async def test_verify_auto_extract_claims(self):
        """Test verifying with auto-extracted claims."""
        status = await verify_response(
            response="This is a test response with information",
            claim_type=ClaimType.OPINION,
        )

        assert isinstance(status, VerificationStatus)


class TestShouldBlockForSafety:
    """Tests for should_block_for_safety function."""

    def test_high_confidence_safety_critical_pass(self):
        """High confidence + all verification passed = no block."""
        status = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_passed=True,
        )

        should_block, reason = should_block_for_safety(
            ClaimType.MEDICAL,
            confidence=0.9,
            verification_status=status,
        )

        assert should_block is False
        assert reason == ""

    def test_low_confidence_safety_critical_block(self):
        """Low confidence + safety-critical = block."""
        status = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_passed=True,
        )

        should_block, reason = should_block_for_safety(
            ClaimType.MEDICAL,
            confidence=0.5,  # Low confidence
            verification_status=status,
        )

        assert should_block is True
        assert "Low confidence" in reason or "safety" in reason.lower()

    def test_verification_failed_safety_critical_block(self):
        """Verification failed + safety-critical = block."""
        status = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_passed=False,  # Failed
        )

        should_block, reason = should_block_for_safety(
            ClaimType.SAFETY_CRITICAL,
            confidence=0.9,
            verification_status=status,
        )

        assert should_block is True
        assert "verification" in reason.lower()

    def test_factual_claim_no_block(self):
        """Factual claim doesn't trigger safety block."""
        status = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
        )

        should_block, reason = should_block_for_safety(
            ClaimType.FACTUAL,
            confidence=0.5,
            verification_status=status,
        )

        # Factual with low confidence shouldn't block (not safety-critical)
        assert should_block is False

    def test_legal_claim_block(self):
        """Legal claim with low confidence = block."""
        status = VerificationStatus(
            tier_1_passed=True,
            tier_2_passed=True,
            tier_3_passed=True,
        )

        should_block, reason = should_block_for_safety(
            ClaimType.LEGAL,
            confidence=0.4,
            verification_status=status,
        )

        assert should_block is True

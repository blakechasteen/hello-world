#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federation Safety Layer Integration Tests
==========================================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 15: Safety Layer Foundation

Tests for:
    - SignatureVerifier: Ed25519 signature validation
    - GuildTrustChecker: Trust level permissions
    - FederatedRateLimiter: Per-identity rate limiting
    - FederationSafetyGate: Complete safety pipeline

Author: Claude Code (Phase 4 - December 2025)
"""

import asyncio
import hashlib
import time
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Federation modules
from HoloLoom.federation.safety import (
    SignatureVerifier,
    GuildTrustChecker,
    FederationSafetyGate,
    SignedRequest,
    SafetyCheckResult,
    FederationSafetyResult,
    FederationPermission,
    TRUST_PERMISSIONS,
    METHOD_PERMISSIONS,
    parse_signed_request,
    create_federation_safety_gate,
)
from HoloLoom.federation.rate_limiter import (
    FederatedRateLimiter,
    RateLimitTier,
    RateLimitInfo,
    get_tier_for_trust_level,
    create_rate_limiter,
    DEFAULT_RATE_LIMITS,
)
from HoloLoom.federation.types import (
    FederationNode,
    GuildTrustLevel,
    NodeStatus,
    Capability,
)
from HoloLoom.federation.identity import Identity


# ============================================================================
#  Fixtures
# ============================================================================

@pytest.fixture
def identity() -> Identity:
    """Create a test identity with Ed25519 keys."""
    return Identity.generate()


@pytest.fixture
def node(identity: Identity) -> FederationNode:
    """Create a test federation node."""
    return FederationNode(
        node_id=identity.node_id,
        public_key=identity.public_key,
        endpoint="localhost:8000",
        capabilities={Capability.WEAVING, Capability.RAG},
        guilds={"test-guild"},
        trust_score=0.7,
        version="1.0.0",
        status=NodeStatus.ONLINE,
    )


@pytest.fixture
def signed_request(identity: Identity) -> SignedRequest:
    """Create a properly signed test request."""
    method = "hololoom.rag.recall"
    params = {"query": "test query", "k": 10}
    timestamp = time.time()
    nonce = f"test-{timestamp}"

    # Build signing payload
    params_str = str(sorted(params.items()))
    params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
    payload = f"{method}|{timestamp}|{nonce}|{params_hash}"

    # Sign the payload
    signature = identity.sign(payload.encode())

    return SignedRequest(
        method=method,
        params=params,
        sender_id=identity.node_id,
        timestamp=timestamp,
        nonce=nonce,
        signature=signature,
        request_id="req-001",
    )


@pytest.fixture
def signature_verifier() -> SignatureVerifier:
    """Create a test signature verifier."""
    return SignatureVerifier(
        nonce_expiry_seconds=300,
        max_timestamp_drift_seconds=30,
    )


@pytest.fixture
def trust_checker() -> GuildTrustChecker:
    """Create a test trust checker."""
    return GuildTrustChecker()


@pytest.fixture
def rate_limiter() -> FederatedRateLimiter:
    """Create a test rate limiter."""
    return FederatedRateLimiter(
        window_seconds=60.0,
        cleanup_interval_seconds=300.0,
    )


# ============================================================================
#  SignatureVerifier Tests
# ============================================================================

class TestSignatureVerifier:
    """Tests for Ed25519 signature verification."""

    def test_verify_valid_signature(
        self,
        signature_verifier: SignatureVerifier,
        signed_request: SignedRequest,
        identity: Identity,
    ):
        """Valid signature should pass verification."""
        result = signature_verifier.verify(signed_request, identity.public_key)

        assert result.allowed is True
        assert result.check_type == "signature"
        assert "sender_id" in result.details

    def test_verify_invalid_signature(
        self,
        signature_verifier: SignatureVerifier,
        signed_request: SignedRequest,
    ):
        """Invalid signature should fail verification."""
        # Create a different identity
        other_identity = Identity.generate()

        result = signature_verifier.verify(signed_request, other_identity.public_key)

        assert result.allowed is False
        assert "Invalid signature" in result.reason or "verification failed" in result.reason.lower()

    def test_verify_expired_timestamp(
        self,
        signature_verifier: SignatureVerifier,
        identity: Identity,
    ):
        """Expired timestamp should fail verification."""
        # Create request with old timestamp
        old_timestamp = time.time() - 60  # 60 seconds ago

        method = "hololoom.rag.recall"
        params = {"query": "test"}
        nonce = f"test-{old_timestamp}"

        params_str = str(sorted(params.items()))
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        payload = f"{method}|{old_timestamp}|{nonce}|{params_hash}"
        signature = identity.sign(payload.encode())

        request = SignedRequest(
            method=method,
            params=params,
            sender_id=identity.node_id,
            timestamp=old_timestamp,
            nonce=nonce,
            signature=signature,
        )

        result = signature_verifier.verify(request, identity.public_key)

        assert result.allowed is False
        assert "drift" in result.reason.lower()

    def test_verify_replay_attack(
        self,
        signature_verifier: SignatureVerifier,
        signed_request: SignedRequest,
        identity: Identity,
    ):
        """Replayed nonce should fail verification."""
        # First request should pass
        result1 = signature_verifier.verify(signed_request, identity.public_key)
        assert result1.allowed is True

        # Same nonce should fail (replay attack)
        result2 = signature_verifier.verify(signed_request, identity.public_key)
        assert result2.allowed is False
        assert "nonce" in result2.reason.lower() or "replay" in result2.reason.lower()


# ============================================================================
#  GuildTrustChecker Tests
# ============================================================================

class TestGuildTrustChecker:
    """Tests for guild trust level permissions."""

    def test_starter_recall_allowed(self, trust_checker: GuildTrustChecker, node: FederationNode):
        """STARTER should be allowed to recall."""
        result = trust_checker.check(node, "hololoom.rag.recall", GuildTrustLevel.STARTER)

        assert result.allowed is True
        assert result.check_type == "trust"

    def test_starter_generate_denied(self, trust_checker: GuildTrustChecker, node: FederationNode):
        """STARTER should be denied generate permission."""
        result = trust_checker.check(node, "hololoom.inference.generate", GuildTrustLevel.STARTER)

        assert result.allowed is False
        assert "insufficient" in result.reason.lower()

    def test_established_generate_allowed(self, trust_checker: GuildTrustChecker, node: FederationNode):
        """ESTABLISHED should be allowed to generate."""
        result = trust_checker.check(node, "hololoom.inference.generate", GuildTrustLevel.ESTABLISHED)

        assert result.allowed is True

    def test_veteran_all_permissions(self, trust_checker: GuildTrustChecker, node: FederationNode):
        """VETERAN should have all permissions."""
        methods = [
            "hololoom.rag.recall",
            "hololoom.inference.generate",
            "hololoom.convergence.update",
            "hololoom.admin.nodes",
        ]

        for method in methods:
            result = trust_checker.check(node, method, GuildTrustLevel.VETERAN)
            assert result.allowed is True, f"VETERAN should have permission for {method}"

    def test_unknown_method_denied(self, trust_checker: GuildTrustChecker, node: FederationNode):
        """Unknown methods should be denied."""
        result = trust_checker.check(node, "unknown.method", GuildTrustLevel.VETERAN)

        assert result.allowed is False
        assert "unknown" in result.reason.lower()

    def test_get_allowed_methods(self, trust_checker: GuildTrustChecker):
        """Should return correct allowed methods per trust level."""
        starter_methods = trust_checker.get_allowed_methods(GuildTrustLevel.STARTER)
        veteran_methods = trust_checker.get_allowed_methods(GuildTrustLevel.VETERAN)

        # STARTER should have fewer methods
        assert len(starter_methods) < len(veteran_methods)

        # STARTER should have recall
        assert "hololoom.rag.recall" in starter_methods

        # Only VETERAN should have admin
        assert "hololoom.admin.nodes" not in starter_methods
        assert "hololoom.admin.nodes" in veteran_methods


# ============================================================================
#  FederatedRateLimiter Tests
# ============================================================================

class TestFederatedRateLimiter:
    """Tests for per-identity rate limiting."""

    def test_under_limit_allowed(self, rate_limiter: FederatedRateLimiter):
        """Requests under limit should be allowed."""
        result = rate_limiter.check("node-1", RateLimitTier.MEMBER)

        assert result.allowed is True
        assert "under limit" in result.reason.lower() or "unlimited" in result.reason.lower()

    def test_over_limit_denied(self, rate_limiter: FederatedRateLimiter):
        """Requests over limit should be denied."""
        identity_id = "node-heavy"
        tier = RateLimitTier.OBSERVER  # 10 requests/min

        # Make 10 requests (hit limit)
        for _ in range(10):
            result = rate_limiter.check(identity_id, tier)
            rate_limiter.record(identity_id)

        # 11th request should be denied
        result = rate_limiter.check(identity_id, tier)
        assert result.allowed is False
        assert "exceeded" in result.reason.lower()

    def test_unlimited_tier_always_allowed(self, rate_limiter: FederatedRateLimiter):
        """STEWARD tier should be unlimited."""
        identity_id = "steward-node"

        # Make many requests
        for _ in range(100):
            result = rate_limiter.check(identity_id, RateLimitTier.STEWARD)
            assert result.allowed is True
            rate_limiter.record(identity_id)

    def test_sliding_window_recovery(self, rate_limiter: FederatedRateLimiter):
        """Rate limit should recover as window slides."""
        # Create limiter with very short window for testing
        short_limiter = FederatedRateLimiter(
            rate_limits={RateLimitTier.OBSERVER: 2},
            window_seconds=0.1,  # 100ms window
        )

        identity_id = "fast-node"

        # Hit limit
        short_limiter.check(identity_id, RateLimitTier.OBSERVER)
        short_limiter.record(identity_id)
        short_limiter.check(identity_id, RateLimitTier.OBSERVER)
        short_limiter.record(identity_id)

        # Should be limited
        result = short_limiter.check(identity_id, RateLimitTier.OBSERVER)
        assert result.allowed is False

        # Wait for window to slide
        time.sleep(0.15)

        # Should be allowed again
        result = short_limiter.check(identity_id, RateLimitTier.OBSERVER)
        assert result.allowed is True

    def test_rate_limit_info_headers(self, rate_limiter: FederatedRateLimiter):
        """Rate limit info should include standard headers."""
        identity_id = "header-test"
        rate_limiter.check(identity_id, RateLimitTier.MEMBER)
        rate_limiter.record(identity_id)

        info = rate_limiter.get_info(identity_id)
        headers = info.to_headers()

        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers

    def test_reset_clears_state(self, rate_limiter: FederatedRateLimiter):
        """Reset should clear rate limit state."""
        identity_id = "reset-test"

        # Make some requests
        for _ in range(5):
            rate_limiter.check(identity_id, RateLimitTier.MEMBER)
            rate_limiter.record(identity_id)

        info_before = rate_limiter.get_info(identity_id)
        assert info_before.remaining < 60

        # Reset
        rate_limiter.reset(identity_id)

        # Should have full quota again
        info_after = rate_limiter.get_info(identity_id, RateLimitTier.MEMBER)
        assert info_after.remaining == 60


# ============================================================================
#  FederationSafetyGate Tests
# ============================================================================

class TestFederationSafetyGate:
    """Tests for the complete safety pipeline."""

    @pytest.mark.asyncio
    async def test_all_checks_pass(
        self,
        signed_request: SignedRequest,
        node: FederationNode,
        identity: Identity,
    ):
        """All checks passing should allow request."""
        gate = FederationSafetyGate(
            guardrails=None,  # Skip safety check for this test
            audit_trail=None,
            enable_safety_check=False,
            enable_audit=False,
        )

        result = await gate.check_request(
            signed_request,
            node,
            guild_trust_level=GuildTrustLevel.ESTABLISHED,
        )

        assert result.allowed is True
        assert len(result.checks) >= 2  # Signature + Trust

    @pytest.mark.asyncio
    async def test_signature_failure_blocks(self, node: FederationNode):
        """Failed signature check should block request."""
        # Create request with wrong signature
        other_identity = Identity.generate()
        method = "hololoom.rag.recall"
        params = {"query": "test"}
        timestamp = time.time()
        nonce = f"test-{timestamp}"

        params_str = str(sorted(params.items()))
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        payload = f"{method}|{timestamp}|{nonce}|{params_hash}"
        signature = other_identity.sign(payload.encode())  # Wrong key!

        bad_request = SignedRequest(
            method=method,
            params=params,
            sender_id=node.node_id,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
        )

        gate = FederationSafetyGate(
            enable_safety_check=False,
            enable_audit=False,
        )

        result = await gate.check_request(
            bad_request,
            node,
            guild_trust_level=GuildTrustLevel.ESTABLISHED,
        )

        assert result.allowed is False
        assert "signature" in result.denied_reason.lower()

    @pytest.mark.asyncio
    async def test_trust_failure_blocks(
        self,
        signed_request: SignedRequest,
        node: FederationNode,
    ):
        """Failed trust check should block request."""
        # Create request for admin method
        admin_request = SignedRequest(
            method="hololoom.admin.nodes",  # Requires VETERAN
            params={},
            sender_id=signed_request.sender_id,
            timestamp=signed_request.timestamp,
            nonce=signed_request.nonce + "-admin",
            signature=signed_request.signature,  # Will fail, but trust checked first if sig skipped
            request_id="admin-001",
        )

        gate = FederationSafetyGate(
            enable_signature_check=False,  # Skip signature to test trust
            enable_safety_check=False,
            enable_audit=False,
        )

        result = await gate.check_request(
            admin_request,
            node,
            guild_trust_level=GuildTrustLevel.STARTER,  # STARTER can't do admin
        )

        assert result.allowed is False
        assert "trust" in result.denied_reason.lower()

    @pytest.mark.asyncio
    async def test_rate_limit_integrated(
        self,
        signed_request: SignedRequest,
        node: FederationNode,
    ):
        """Rate limit check should be integrated."""
        rate_limit_denied = SafetyCheckResult(
            allowed=False,
            reason="Rate limit exceeded",
            check_type="rate_limit",
        )

        gate = FederationSafetyGate(
            enable_signature_check=False,
            enable_trust_check=False,
            enable_safety_check=False,
            enable_audit=False,
        )

        result = await gate.check_request(
            signed_request,
            node,
            rate_limit_result=rate_limit_denied,
        )

        assert result.allowed is False
        assert "rate_limit" in result.denied_reason.lower()

    @pytest.mark.asyncio
    async def test_timing_tracked(
        self,
        signed_request: SignedRequest,
        node: FederationNode,
    ):
        """Check time should be tracked."""
        gate = FederationSafetyGate(
            enable_signature_check=False,
            enable_trust_check=False,
            enable_safety_check=False,
            enable_audit=False,
        )

        result = await gate.check_request(signed_request, node)

        assert result.total_check_time_ms >= 0
        assert result.total_check_time_ms < 1000  # Should be fast


# ============================================================================
#  Factory and Utility Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions and utilities."""

    def test_parse_signed_request_valid(self):
        """Should parse valid signed request data."""
        import base64

        data = {
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "params": {"query": "test"},
            "meta": {
                "sender_id": "ed25519:abc123",
                "timestamp": 1702300800,
                "nonce": "test-nonce",
                "signature": base64.b64encode(b"fake-sig").decode(),
            },
            "id": "req-001",
        }

        request = parse_signed_request(data)

        assert request.method == "hololoom.rag.recall"
        assert request.params == {"query": "test"}
        assert request.sender_id == "ed25519:abc123"
        assert request.nonce == "test-nonce"

    def test_parse_signed_request_missing_method(self):
        """Should raise on missing method."""
        data = {
            "params": {},
            "meta": {"signature": "abc"},
        }

        with pytest.raises(ValueError, match="method"):
            parse_signed_request(data)

    def test_parse_signed_request_missing_meta(self):
        """Should raise on missing meta."""
        data = {
            "method": "test",
            "params": {},
        }

        with pytest.raises(ValueError, match="meta"):
            parse_signed_request(data)

    def test_get_tier_for_trust_level(self):
        """Should map trust levels to rate limit tiers."""
        assert get_tier_for_trust_level(GuildTrustLevel.STARTER) == RateLimitTier.OBSERVER
        assert get_tier_for_trust_level(GuildTrustLevel.ESTABLISHED) == RateLimitTier.MEMBER
        assert get_tier_for_trust_level(GuildTrustLevel.VETERAN) == RateLimitTier.CONTRIBUTOR

    def test_create_rate_limiter(self):
        """Should create configured rate limiter."""
        limiter = create_rate_limiter(
            custom_limits={RateLimitTier.OBSERVER: 5},
            window_seconds=30.0,
        )

        assert isinstance(limiter, FederatedRateLimiter)

    def test_create_federation_safety_gate(self):
        """Should create configured safety gate."""
        gate = create_federation_safety_gate()

        assert isinstance(gate, FederationSafetyGate)


# ============================================================================
#  Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_pipeline(self, identity: Identity, node: FederationNode):
        """Test complete safety pipeline with all components."""
        # Create rate limiter
        rate_limiter = FederatedRateLimiter()

        # Create safety gate
        gate = FederationSafetyGate(
            enable_safety_check=False,  # Skip for test
            enable_audit=False,
        )

        # Create signed request
        method = "hololoom.rag.recall"
        params = {"query": "test query", "k": 10}
        timestamp = time.time()
        nonce = f"test-{timestamp}"

        params_str = str(sorted(params.items()))
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        payload = f"{method}|{timestamp}|{nonce}|{params_hash}"
        signature = identity.sign(payload.encode())

        request = SignedRequest(
            method=method,
            params=params,
            sender_id=identity.node_id,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            request_id="e2e-001",
        )

        # Check rate limit
        tier = get_tier_for_trust_level(GuildTrustLevel.ESTABLISHED)
        rate_result = rate_limiter.check(node.node_id, tier)

        # Run safety pipeline
        result = await gate.check_request(
            request,
            node,
            guild_trust_level=GuildTrustLevel.ESTABLISHED,
            rate_limit_result=rate_result,
        )

        assert result.allowed is True

        # Record the request
        rate_limiter.record(node.node_id)

        # Verify state
        info = rate_limiter.get_info(node.node_id)
        assert info.remaining == 59  # One request made

    @pytest.mark.asyncio
    async def test_multiple_requests_rate_limited(
        self,
        identity: Identity,
        node: FederationNode,
    ):
        """Multiple requests should eventually be rate limited."""
        rate_limiter = FederatedRateLimiter(
            rate_limits={RateLimitTier.OBSERVER: 3},  # Very low limit
        )

        gate = FederationSafetyGate(
            enable_signature_check=False,
            enable_trust_check=False,
            enable_safety_check=False,
            enable_audit=False,
        )

        allowed_count = 0
        for i in range(5):
            request = SignedRequest(
                method="hololoom.rag.recall",
                params={"i": i},
                sender_id=identity.node_id,
                timestamp=time.time(),
                nonce=f"nonce-{i}",
                signature=b"",
            )

            rate_result = rate_limiter.check(node.node_id, RateLimitTier.OBSERVER)
            result = await gate.check_request(request, node, rate_limit_result=rate_result)

            if result.allowed:
                allowed_count += 1
                rate_limiter.record(node.node_id)

        # Should allow 3, deny 2
        assert allowed_count == 3


# ============================================================================
#  Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

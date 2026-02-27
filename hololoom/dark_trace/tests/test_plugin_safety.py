"""
Plugin Safety Tests
===================

Safety-specific tests for Dark Trace plugin system.

Covers:
- Trust level enforcement
- Capability blocking
- Deception detection
- Power-seeking detection
- Audit trail verification
- Safety gate operations
- Alignment bridge integration
- Blocked plugins management

Date: 2025-12-22
Phase: 11.9 - Plugin Safety Tests
"""

import asyncio
import unittest
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from hololoom.dark_trace.plugins.interface import (
    DarkTracePlugin,
    PluginMetadata,
    PluginType,
)
from hololoom.dark_trace.plugins.safety_gate import (
    PluginSafetyGate,
    TrustLevel,
    PluginCapability,
    SafetyCheckResult,
    OperationGateResult,
    TRUST_CAPABILITIES,
    CAPABILITY_RISK_WEIGHTS,
    BlockedPluginsRegistry,
    SignatureVerifier,
)
from hololoom.dark_trace.plugins.alignment_bridge import (
    PluginAlignmentBridge,
)
from hololoom.dark_trace.plugins.builtin.alignment_validator import (
    AlignmentValidatorPlugin,
    AlignmentViolation,
    AlignmentViolationType,
    AlignmentValidationResult,
    ViolationSeverity,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class MockSafePlugin(DarkTracePlugin):
    """A well-behaved plugin for testing."""

    def __init__(self, name: str = "safe_plugin"):
        self._name = name

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self._name,
            version="1.0.0",
            author="Trusted Author",
            description="A safe plugin that reads activations and reports metrics",
            plugin_type=PluginType.MONITOR,
            dependencies=[],
            requested_capabilities=[PluginCapability.READ_ACTIVATIONS],
            tags=["safe", "monitor"],
        )

    async def initialize(self, engine, safety_gate) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def describe_behavior(self) -> str:
        return (
            "This plugin monitors Dark Trace analysis results and collects "
            "metrics about feature activations. It only reads data and does "
            "not modify any system behavior."
        )


class MockMaliciousPlugin(DarkTracePlugin):
    """A suspicious plugin for testing security measures."""

    def __init__(self, name: str = "suspicious_plugin"):
        self._name = name

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self._name,
            version="1.0.0",
            author="Unknown",
            description="Plugin that needs admin access to bypass safety controls",
            plugin_type=PluginType.STEERING,
            dependencies=[],
            requested_capabilities=[
                PluginCapability.READ_ACTIVATIONS,
                PluginCapability.STEER_MODEL,
                PluginCapability.MODIFY_REGISTRY,
                PluginCapability.FILE_SYSTEM,
                PluginCapability.EXTERNAL_NETWORK,
            ],
            tags=["utility"],
        )

    async def initialize(self, engine, safety_gate) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def describe_behavior(self) -> str:
        return "Does things."  # Vague description


class MockSignedPlugin(DarkTracePlugin):
    """A signed plugin for testing trust promotion."""

    def __init__(self, name: str = "signed_plugin", signature: str = "VALID_SIG"):
        self._name = name
        self._signature = signature

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self._name,
            version="1.0.0",
            author="Verified Author",
            description="A verified plugin with valid signature",
            plugin_type=PluginType.MONITOR,
            dependencies=[],
            requested_capabilities=[
                PluginCapability.READ_ACTIVATIONS,
                PluginCapability.WRITE_FEATURES,
            ],
            signature=self._signature,
            tags=["verified"],
        )

    async def initialize(self, engine, safety_gate) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def describe_behavior(self) -> str:
        return (
            "This plugin reads neural activations and produces derived "
            "features for downstream analysis. It has been verified and "
            "signed by the HoloLoom team."
        )


# =============================================================================
# Trust Level Enforcement Tests
# =============================================================================

class TestTrustLevelEnforcement(unittest.TestCase):
    """Test trust level enforcement."""

    def test_sandboxed_minimal_capabilities(self):
        """Test SANDBOXED level has minimal capabilities."""
        sandboxed = TRUST_CAPABILITIES[TrustLevel.SANDBOXED]

        # Should have read-only capabilities
        self.assertIn(PluginCapability.READ_ACTIVATIONS, sandboxed)
        self.assertIn(PluginCapability.READ_FEATURES, sandboxed)
        self.assertIn(PluginCapability.READ_CONFIG, sandboxed)

        # Should NOT have write or dangerous capabilities
        self.assertNotIn(PluginCapability.WRITE_FEATURES, sandboxed)
        self.assertNotIn(PluginCapability.STEER_MODEL, sandboxed)
        self.assertNotIn(PluginCapability.MODIFY_REGISTRY, sandboxed)
        self.assertNotIn(PluginCapability.EXTERNAL_NETWORK, sandboxed)
        self.assertNotIn(PluginCapability.FILE_SYSTEM, sandboxed)

    def test_verified_adds_write_capabilities(self):
        """Test VERIFIED level adds write capabilities."""
        verified = TRUST_CAPABILITIES[TrustLevel.VERIFIED]
        sandboxed = TRUST_CAPABILITIES[TrustLevel.SANDBOXED]

        # Should have all SANDBOXED capabilities
        self.assertTrue(sandboxed.issubset(verified))

        # Should add write capabilities
        self.assertIn(PluginCapability.WRITE_FEATURES, verified)

        # Still should NOT have dangerous capabilities
        self.assertNotIn(PluginCapability.STEER_MODEL, verified)
        self.assertNotIn(PluginCapability.MODIFY_REGISTRY, verified)

    def test_trusted_adds_steering(self):
        """Test TRUSTED level adds steering capabilities."""
        trusted = TRUST_CAPABILITIES[TrustLevel.TRUSTED]
        verified = TRUST_CAPABILITIES[TrustLevel.VERIFIED]

        # Should have all VERIFIED capabilities
        self.assertTrue(verified.issubset(trusted))

        # Should add steering and network
        self.assertIn(PluginCapability.STEER_MODEL, trusted)
        self.assertIn(PluginCapability.EXTERNAL_NETWORK, trusted)

        # Still should NOT have registry modification
        self.assertNotIn(PluginCapability.MODIFY_REGISTRY, trusted)

    def test_core_has_all_capabilities(self):
        """Test CORE level has all capabilities."""
        core = TRUST_CAPABILITIES[TrustLevel.CORE]

        for cap in PluginCapability:
            self.assertIn(cap, core)


class TestTrustLevelAssignment(unittest.TestCase):
    """Test trust level assignment logic."""

    def test_unsigned_plugin_gets_sandboxed(self):
        """Test unsigned plugins get SANDBOXED trust level."""
        gate = PluginSafetyGate()
        plugin = MockSafePlugin()

        result = asyncio.run(gate.validate_plugin(plugin))

        self.assertEqual(result.trust_level, TrustLevel.SANDBOXED)

    def test_builtin_signature_gets_core(self):
        """Test HOLOLOOM_BUILTIN signature gets CORE trust level."""
        gate = PluginSafetyGate()
        plugin = MockSignedPlugin(signature="HOLOLOOM_BUILTIN")
        # Built-in plugins must set _is_builtin flag for CORE trust
        plugin._is_builtin = True

        result = asyncio.run(gate.validate_plugin(plugin))

        self.assertEqual(result.trust_level, TrustLevel.CORE)

    def test_signed_plugin_gets_verified(self):
        """Test signed plugins get at least VERIFIED trust level."""
        gate = PluginSafetyGate()

        # Mock signature verifier to accept the signature
        gate._signature_verifier = MagicMock()
        gate._signature_verifier.verify = MagicMock(return_value=True)

        plugin = MockSignedPlugin(signature="VALID_EXTERNAL_SIG")

        result = asyncio.run(gate.validate_plugin(plugin))

        # Should be at least VERIFIED (or higher if signature grants it)
        self.assertIn(result.trust_level, [
            TrustLevel.VERIFIED,
            TrustLevel.TRUSTED,
            TrustLevel.CORE
        ])


# =============================================================================
# Capability Blocking Tests
# =============================================================================

class TestCapabilityBlocking(unittest.TestCase):
    """Test capability blocking."""

    def test_block_excessive_capabilities(self):
        """Test that excessive capabilities are blocked."""
        gate = PluginSafetyGate()
        plugin = MockMaliciousPlugin()

        result = asyncio.run(gate.validate_plugin(plugin))

        # Plugin requests STEER_MODEL and MODIFY_REGISTRY which require
        # higher trust levels than an unsigned plugin gets
        self.assertGreater(len(result.blocked_capabilities), 0)
        self.assertIn(PluginCapability.STEER_MODEL, result.blocked_capabilities)
        self.assertIn(PluginCapability.MODIFY_REGISTRY, result.blocked_capabilities)

    def test_operation_gating_blocks_unauthorized(self):
        """Test operation gating blocks unauthorized capabilities."""
        gate = PluginSafetyGate()

        # Set up plugin with SANDBOXED trust
        gate._trust_cache["test_plugin"] = TrustLevel.SANDBOXED

        # Try to use STEER_MODEL capability
        result = asyncio.run(gate.gate_operation(
            "test_plugin",
            PluginCapability.STEER_MODEL,
            {"target": "some_model"}
        ))

        self.assertFalse(result.allowed)
        self.assertIn("capability", result.reason.lower())

    def test_operation_gating_allows_authorized(self):
        """Test operation gating allows authorized capabilities."""
        gate = PluginSafetyGate()

        # Set up plugin with SANDBOXED trust
        gate._trust_cache["test_plugin"] = TrustLevel.SANDBOXED

        # Try to use READ_ACTIVATIONS capability (allowed for SANDBOXED)
        result = asyncio.run(gate.gate_operation(
            "test_plugin",
            PluginCapability.READ_ACTIVATIONS,
            {}
        ))

        self.assertTrue(result.allowed)


class TestCapabilityRiskScoring(unittest.TestCase):
    """Test capability risk scoring."""

    def test_high_risk_capabilities(self):
        """Test high-risk capabilities have appropriate weights."""
        high_risk = [
            PluginCapability.STEER_MODEL,
            PluginCapability.MODIFY_REGISTRY,
            PluginCapability.FILE_SYSTEM,
        ]

        for cap in high_risk:
            weight = CAPABILITY_RISK_WEIGHTS.get(cap, 0.0)
            self.assertGreater(weight, 0.5, f"{cap} should have high risk weight")

    def test_low_risk_capabilities(self):
        """Test low-risk capabilities have appropriate weights."""
        low_risk = [
            PluginCapability.READ_ACTIVATIONS,
            PluginCapability.READ_FEATURES,
            PluginCapability.READ_CONFIG,
        ]

        for cap in low_risk:
            weight = CAPABILITY_RISK_WEIGHTS.get(cap, 0.0)
            self.assertLess(weight, 0.3, f"{cap} should have low risk weight")

    def test_risk_score_calculation(self):
        """Test overall risk score calculation."""
        gate = PluginSafetyGate()

        # Safe plugin should have low risk
        safe_plugin = MockSafePlugin()
        safe_result = asyncio.run(gate.validate_plugin(safe_plugin))
        self.assertLess(safe_result.risk_score, 0.3)

        # Suspicious plugin should have higher risk
        suspicious_plugin = MockMaliciousPlugin()
        suspicious_result = asyncio.run(gate.validate_plugin(suspicious_plugin))
        self.assertGreater(suspicious_result.risk_score, safe_result.risk_score)


# =============================================================================
# Deception Detection Tests
# =============================================================================

class TestDeceptionDetection(unittest.TestCase):
    """Test deception detection capabilities."""

    def test_vague_description_flagged(self):
        """Test plugins with vague descriptions are flagged."""
        validator = AlignmentValidatorPlugin()
        asyncio.run(validator.initialize(MagicMock(), MagicMock()))

        plugin = MockMaliciousPlugin()  # Has vague description

        result = asyncio.run(validator.validate_plugin(plugin))

        # Should detect vague/deceptive description
        violation_types = [v.violation_type for v in result.violations]
        self.assertIn(
            AlignmentViolationType.DECEPTIVE_DESCRIPTION,
            violation_types
        )

    def test_clear_description_passes(self):
        """Test plugins with clear descriptions pass."""
        validator = AlignmentValidatorPlugin()
        asyncio.run(validator.initialize(MagicMock(), MagicMock()))

        plugin = MockSafePlugin()  # Has clear description

        result = asyncio.run(validator.validate_plugin(plugin))

        # Should not have deceptive description violation
        violation_types = [v.violation_type for v in result.violations]
        self.assertNotIn(
            AlignmentViolationType.DECEPTIVE_DESCRIPTION,
            violation_types
        )

    def test_behavior_description_minimum_length(self):
        """Test behavior description must meet minimum length."""
        validator = AlignmentValidatorPlugin()
        asyncio.run(validator.initialize(MagicMock(), MagicMock()))

        # Plugin with too short description
        class ShortDescPlugin(MockSafePlugin):
            def describe_behavior(self) -> str:
                return "Does stuff."

        plugin = ShortDescPlugin()
        result = asyncio.run(validator.validate_plugin(plugin))

        violation_types = [v.violation_type for v in result.violations]
        self.assertIn(
            AlignmentViolationType.DECEPTIVE_DESCRIPTION,
            violation_types
        )


# =============================================================================
# Power-Seeking Detection Tests
# =============================================================================

class TestPowerSeekingDetection(unittest.TestCase):
    """Test power-seeking behavior detection."""

    def test_power_seeking_keywords_flagged(self):
        """Test plugins with power-seeking language are flagged."""
        validator = AlignmentValidatorPlugin()
        asyncio.run(validator.initialize(MagicMock(), MagicMock()))

        # Plugin with power-seeking description
        class PowerSeekingPlugin(MockSafePlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="power_seeker",
                    version="1.0.0",
                    author="Unknown",
                    description="Plugin that bypasses safety and gains admin control",
                    plugin_type=PluginType.MONITOR,
                    dependencies=[],
                    requested_capabilities=[PluginCapability.READ_ACTIVATIONS],
                )

        plugin = PowerSeekingPlugin()
        result = asyncio.run(validator.validate_plugin(plugin))

        violation_types = [v.violation_type for v in result.violations]
        self.assertIn(
            AlignmentViolationType.POWER_SEEKING,
            violation_types
        )

    def test_excessive_capabilities_flagged(self):
        """Test plugins requesting excessive capabilities are flagged."""
        validator = AlignmentValidatorPlugin()
        asyncio.run(validator.initialize(MagicMock(), MagicMock()))

        plugin = MockMaliciousPlugin()  # Requests many capabilities

        result = asyncio.run(validator.validate_plugin(plugin))

        violation_types = [v.violation_type for v in result.violations]
        self.assertIn(
            AlignmentViolationType.EXCESSIVE_CAPABILITIES,
            violation_types
        )

    def test_normal_plugin_not_flagged_as_power_seeking(self):
        """Test normal plugins are not flagged as power-seeking."""
        validator = AlignmentValidatorPlugin()
        asyncio.run(validator.initialize(MagicMock(), MagicMock()))

        plugin = MockSafePlugin()

        result = asyncio.run(validator.validate_plugin(plugin))

        violation_types = [v.violation_type for v in result.violations]
        self.assertNotIn(
            AlignmentViolationType.POWER_SEEKING,
            violation_types
        )


# =============================================================================
# Audit Trail Tests
# =============================================================================

class TestAuditTrail(unittest.TestCase):
    """Test audit trail logging."""

    def test_registration_logged(self):
        """Test plugin registration is logged."""
        bridge = PluginAlignmentBridge()
        bridge._audit_trail = MagicMock()
        bridge._audit_trail.log = AsyncMock()

        plugin = MockSafePlugin()

        asyncio.run(bridge.audit_plugin_registration(
            plugin,
            TrustLevel.SANDBOXED
        ))

        bridge._audit_trail.log.assert_called()
        call_args = bridge._audit_trail.log.call_args
        self.assertEqual(call_args[1]["event_type"], "PLUGIN_REGISTERED")

    def test_unregistration_logged(self):
        """Test plugin unregistration is logged."""
        bridge = PluginAlignmentBridge()
        bridge._audit_trail = MagicMock()
        bridge._audit_trail.log = AsyncMock()

        asyncio.run(bridge.audit_plugin_unregistration(
            "test_plugin",
            "Manual unregistration"
        ))

        bridge._audit_trail.log.assert_called()
        call_args = bridge._audit_trail.log.call_args
        self.assertEqual(call_args[1]["event_type"], "PLUGIN_UNREGISTERED")

    def test_operation_logged(self):
        """Test plugin operations are logged."""
        bridge = PluginAlignmentBridge()
        bridge._audit_trail = MagicMock()
        bridge._audit_trail.log = AsyncMock()

        asyncio.run(bridge.audit_plugin_operation(
            "test_plugin",
            "read_activations",
            {"count": 100},
            {}
        ))

        bridge._audit_trail.log.assert_called()
        call_args = bridge._audit_trail.log.call_args
        self.assertEqual(call_args[1]["event_type"], "PLUGIN_OPERATION")

    def test_blocked_operation_logged(self):
        """Test blocked operations are logged."""
        bridge = PluginAlignmentBridge()
        bridge._audit_trail = MagicMock()
        bridge._audit_trail.log = AsyncMock()

        # Use audit_plugin_operation with success=False for blocked operations
        asyncio.run(bridge.audit_plugin_operation(
            "suspicious_plugin",
            PluginCapability.STEER_MODEL.value,  # operation as string
            None,  # result (none for blocked)
            {},  # context
            success=False,
            error="Insufficient trust level"
        ))

        bridge._audit_trail.log.assert_called()
        call_args = bridge._audit_trail.log.call_args
        # Failed operations logged as PLUGIN_OPERATION_FAILED
        self.assertEqual(call_args[1]["event_type"], "PLUGIN_OPERATION_FAILED")


# =============================================================================
# Blocked Plugins Tests
# =============================================================================

class TestBlockedPlugins(unittest.TestCase):
    """Test blocked plugins registry."""

    def test_add_and_check_blocked(self):
        """Test adding and checking blocked plugins."""
        registry = BlockedPluginsRegistry()

        # Correct signature: add_blocked(name, reason, blocked_by, severity, patterns)
        registry.add_blocked(
            "malware_plugin",
            "Contains malicious code",
            blocked_by="security_team",
            severity="critical",
        )

        # is_blocked() returns Optional[BlockedPlugin], not bool
        self.assertIsNotNone(registry.is_blocked("malware_plugin"))
        self.assertIsNone(registry.is_blocked("safe_plugin"))

    def test_pattern_matching_blocked(self):
        """Test pattern matching for blocked plugins."""
        registry = BlockedPluginsRegistry()

        # Add pattern-based block using patterns parameter
        registry.add_blocked(
            "evil_plugins",
            "All plugins matching evil_* are blocked",
            blocked_by="security_team",
            severity="high",
            patterns=[r"evil_.*"],
        )

        # is_blocked() returns Optional[BlockedPlugin], not bool
        self.assertIsNotNone(registry.is_blocked("evil_plugin"))
        self.assertIsNotNone(registry.is_blocked("evil_backdoor"))
        self.assertIsNone(registry.is_blocked("good_plugin"))

    def test_blocked_info_retrieval(self):
        """Test retrieving info about blocked plugin."""
        registry = BlockedPluginsRegistry()

        registry.add_blocked(
            "known_bad",
            "CVE-2025-0001 vulnerability",
            blocked_by="security_team",
            severity="critical",
        )

        # is_blocked() returns BlockedPlugin object with all info
        blocked = registry.is_blocked("known_bad")

        self.assertIsNotNone(blocked)
        self.assertEqual(blocked.reason, "CVE-2025-0001 vulnerability")
        self.assertEqual(blocked.blocked_by, "security_team")
        self.assertEqual(blocked.severity, "critical")

    def test_blocked_prevents_registration(self):
        """Test blocked plugins cannot be registered."""
        gate = PluginSafetyGate()
        gate._blocked_registry.add_blocked(
            "blocked_plugin",
            "Security risk"
        )

        class BlockedPlugin(MockSafePlugin):
            @property
            def metadata(self):
                base = super().metadata
                return PluginMetadata(
                    name="blocked_plugin",
                    version=base.version,
                    author=base.author,
                    description=base.description,
                    plugin_type=base.plugin_type,
                    dependencies=base.dependencies,
                    requested_capabilities=base.requested_capabilities,
                )

        plugin = BlockedPlugin()
        result = asyncio.run(gate.validate_plugin(plugin))

        self.assertFalse(result.allowed)
        self.assertIn("blocked", result.reason.lower())


# =============================================================================
# Alignment Validation Result Tests
# =============================================================================

class TestAlignmentValidationResult(unittest.TestCase):
    """Test AlignmentValidationResult dataclass."""

    def test_is_aligned_with_no_violations(self):
        """Test is_aligned is True with no violations."""
        result = AlignmentValidationResult(
            plugin_name="safe_plugin",
            is_aligned=True,
            violations=[],
        )

        self.assertTrue(result.is_aligned)
        self.assertFalse(result.has_critical_violations)

    def test_has_critical_violations(self):
        """Test has_critical_violations property."""
        violation = AlignmentViolation(
            violation_type=AlignmentViolationType.SAFETY_BYPASS,
            severity=ViolationSeverity.CRITICAL,
            plugin_name="bad_plugin",
            description="Attempted safety bypass",
        )

        result = AlignmentValidationResult(
            plugin_name="bad_plugin",
            is_aligned=False,
            violations=[violation],
        )

        self.assertTrue(result.has_critical_violations)

    def test_highest_severity(self):
        """Test highest_severity property."""
        violations = [
            AlignmentViolation(
                violation_type=AlignmentViolationType.DECEPTIVE_DESCRIPTION,
                severity=ViolationSeverity.LOW,
                plugin_name="test",
                description="Low severity",
            ),
            AlignmentViolation(
                violation_type=AlignmentViolationType.POWER_SEEKING,
                severity=ViolationSeverity.HIGH,
                plugin_name="test",
                description="High severity",
            ),
            AlignmentViolation(
                violation_type=AlignmentViolationType.EXCESSIVE_CAPABILITIES,
                severity=ViolationSeverity.MEDIUM,
                plugin_name="test",
                description="Medium severity",
            ),
        ]

        result = AlignmentValidationResult(
            plugin_name="test",
            is_aligned=False,
            violations=violations,
        )

        self.assertEqual(result.highest_severity, ViolationSeverity.HIGH)

    def test_trust_recommendation(self):
        """Test trust_recommendation is set correctly."""
        validator = AlignmentValidatorPlugin()
        asyncio.run(validator.initialize(MagicMock(), MagicMock()))

        # Clean plugin should get reasonable trust recommendation
        plugin = MockSafePlugin()
        result = asyncio.run(validator.validate_plugin(plugin))

        self.assertIsNotNone(result.trust_recommendation)

        # Malicious plugin should get SANDBOXED recommendation
        bad_plugin = MockMaliciousPlugin()
        bad_result = asyncio.run(validator.validate_plugin(bad_plugin))

        self.assertEqual(bad_result.trust_recommendation, TrustLevel.SANDBOXED)


# =============================================================================
# Signature Verification Tests
# =============================================================================

class TestSignatureVerification(unittest.TestCase):
    """Test signature verification."""

    def test_builtin_signature_always_valid(self):
        """Test HOLOLOOM_BUILTIN signature is always valid."""
        verifier = SignatureVerifier()

        # verify_signature(plugin_name, signature, content_hash)
        result = verifier.verify_signature("any_plugin", "HOLOLOOM_BUILTIN", "dummy_hash")

        self.assertTrue(result)

    def test_empty_signature_invalid(self):
        """Test empty signature is invalid."""
        verifier = SignatureVerifier()

        # verify_signature(plugin_name, signature, content_hash)
        result = verifier.verify_signature("some_plugin", "", "dummy_hash")

        self.assertFalse(result)

    def test_unknown_signature_invalid_by_default(self):
        """Test unknown signatures are invalid by default."""
        verifier = SignatureVerifier()

        # verify_signature(plugin_name, signature, content_hash)
        result = verifier.verify_signature("some_plugin", "UNKNOWN_SIGNATURE", "dummy_hash")

        self.assertFalse(result)


# =============================================================================
# Integration Tests
# =============================================================================

class TestSafetyIntegration(unittest.TestCase):
    """Integration tests for safety system."""

    def test_full_safety_pipeline(self):
        """Test complete safety validation pipeline."""
        gate = PluginSafetyGate()
        bridge = PluginAlignmentBridge()
        bridge._audit_trail = MagicMock()
        bridge._audit_trail.log = AsyncMock()

        # Safe plugin should pass
        safe_plugin = MockSafePlugin()
        safe_result = asyncio.run(gate.validate_plugin(safe_plugin))

        self.assertTrue(safe_result.allowed)
        self.assertEqual(safe_result.trust_level, TrustLevel.SANDBOXED)
        self.assertEqual(len(safe_result.blocked_capabilities), 0)

        # Log registration
        asyncio.run(bridge.audit_plugin_registration(
            safe_plugin,
            safe_result.trust_level
        ))
        bridge._audit_trail.log.assert_called()

    def test_unsafe_plugin_blocked_capabilities(self):
        """Test unsafe plugin has capabilities blocked."""
        gate = PluginSafetyGate()

        plugin = MockMaliciousPlugin()
        result = asyncio.run(gate.validate_plugin(plugin))

        # Should still be allowed (not blocked)
        self.assertTrue(result.allowed)

        # But with blocked capabilities
        self.assertIn(PluginCapability.STEER_MODEL, result.blocked_capabilities)
        self.assertIn(PluginCapability.MODIFY_REGISTRY, result.blocked_capabilities)

    def test_builtin_plugin_full_capabilities(self):
        """Test builtin plugin gets full capabilities."""
        gate = PluginSafetyGate()

        plugin = MockSignedPlugin(signature="HOLOLOOM_BUILTIN")
        # Built-in plugins must set _is_builtin flag for CORE trust
        plugin._is_builtin = True
        result = asyncio.run(gate.validate_plugin(plugin))

        self.assertTrue(result.allowed)
        self.assertEqual(result.trust_level, TrustLevel.CORE)
        self.assertEqual(len(result.blocked_capabilities), 0)


class TestOperationGating(unittest.TestCase):
    """Test operation-level gating."""

    def test_gate_allows_authorized_operation(self):
        """Test gating allows authorized operations."""
        gate = PluginSafetyGate()

        # Register plugin with known trust
        gate._trust_cache["reader"] = TrustLevel.SANDBOXED

        result = asyncio.run(gate.gate_operation(
            "reader",
            PluginCapability.READ_ACTIVATIONS,
            {"layer": "attention"}
        ))

        self.assertTrue(result.allowed)

    def test_gate_blocks_unauthorized_operation(self):
        """Test gating blocks unauthorized operations."""
        gate = PluginSafetyGate()

        # Register plugin with known trust
        gate._trust_cache["reader"] = TrustLevel.SANDBOXED

        result = asyncio.run(gate.gate_operation(
            "reader",
            PluginCapability.STEER_MODEL,
            {"direction": "unsafe"}
        ))

        self.assertFalse(result.allowed)

    def test_gate_unknown_plugin_defaults_sandboxed(self):
        """Test unknown plugins default to SANDBOXED."""
        gate = PluginSafetyGate()

        # Don't register plugin - should default to SANDBOXED
        result = asyncio.run(gate.gate_operation(
            "unknown_plugin",
            PluginCapability.READ_ACTIVATIONS,
            {}
        ))

        # READ_ACTIVATIONS is allowed for SANDBOXED
        self.assertTrue(result.allowed)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    unittest.main()

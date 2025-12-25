"""
Plugin System Tests
===================

Core tests for Dark Trace plugin system.

Covers:
- Plugin interface and metadata validation
- Plugin registration and lifecycle
- Trust level enforcement
- Capability management
- Built-in plugins
- Registry operations
- Plugin discovery

Date: 2025-12-22
Phase: 11.9 - Plugin System Tests
"""

import asyncio
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

# Import plugin system components
from HoloLoom.dark_trace.plugins.interface import (
    DarkTracePlugin,
    PluginMetadata,
    PluginType,
    PluginState,
    LensPlugin,
    ValidatorPlugin,
    MonitorPlugin,
    DomainPlugin,
    SteeringPlugin,
    IntegrationPlugin,
    HookRegistry,
)
from HoloLoom.dark_trace.plugins.safety_gate import (
    PluginSafetyGate,
    TrustLevel,
    PluginCapability,
    SafetyCheckResult,
    OperationGateResult,
    TRUST_CAPABILITIES,
    BlockedPluginsRegistry,
)
from HoloLoom.dark_trace.plugins.registry import (
    PluginRegistry,
    PluginEntry,
    RegistrationResult,
    RegistrationStatus,
    UnregistrationResult,
)
from HoloLoom.dark_trace.plugins.builtin import (
    SafetyMonitorPlugin,
    AlignmentValidatorPlugin,
    MetricsExporterPlugin,
    BUILTIN_PLUGINS,
    get_builtin_plugin,
    get_auto_load_plugins,
    list_builtin_plugins,
)


# =============================================================================
# Test Fixtures - Mock Plugins
# =============================================================================

# Sentinel to distinguish "not passed" from "passed as None"
_SIGNATURE_NOT_SET = object()


class MockPlugin(DarkTracePlugin):
    """Simple mock plugin for testing."""

    def __init__(
        self,
        name: str = "mock_plugin",
        plugin_type: PluginType = PluginType.MONITOR,
        capabilities: Optional[List[PluginCapability]] = None,
        signature: Optional[str] = _SIGNATURE_NOT_SET,
        is_builtin: bool = True,  # Default to builtin for testing
    ):
        self._name = name
        self._plugin_type = plugin_type
        self._capabilities = capabilities or [PluginCapability.READ_ACTIVATIONS]
        # Use HOLOLOOM_BUILTIN signature by default for testing (unless explicitly set to None)
        if signature is _SIGNATURE_NOT_SET:
            self._signature = "HOLOLOOM_BUILTIN"
        else:
            self._signature = signature  # Can be None for unsigned plugins
        self._initialized = False
        self._metadata_override: Optional[PluginMetadata] = None
        self._is_builtin = is_builtin  # Flag for CORE trust level

    @property
    def metadata(self) -> PluginMetadata:
        # Support metadata override for testing
        if self._metadata_override is not None:
            return self._metadata_override
        return PluginMetadata(
            name=self._name,
            version="1.0.0",
            author="Test Author",
            description=f"Test plugin {self._name}",
            plugin_type=self._plugin_type,
            dependencies=[],
            requested_capabilities=self._capabilities,
            signature=self._signature,
            min_dark_trace_version="1.0.0",
            tags=["test"],
        )

    async def initialize(self, engine, safety_gate) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    def describe_behavior(self) -> str:
        return f"This is a test plugin named {self._name} that does testing things."


class MockLensPlugin(LensPlugin):
    """Mock lens plugin for testing."""

    def __init__(self, name: str = "mock_lens"):
        self._name = name
        self._is_builtin = True  # Flag for CORE trust level in tests

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self._name,
            version="1.0.0",
            author="Test Author",
            description=f"Test lens plugin",
            plugin_type=PluginType.LENS,
            dependencies=[],
            requested_capabilities=[
                PluginCapability.READ_ACTIVATIONS,
                PluginCapability.WRITE_FEATURES,
            ],
            tags=["test", "lens"],
            signature="HOLOLOOM_BUILTIN",  # Builtin signature for CORE trust
        )

    async def initialize(self, engine, safety_gate) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def describe_behavior(self) -> str:
        return "Test lens that analyzes activations and produces feature outputs."

    def get_lens_class(self):
        return None  # Mock implementation


class MockMonitorPlugin(MonitorPlugin):
    """Mock monitor plugin for testing."""

    def __init__(self, name: str = "mock_monitor"):
        self._name = name
        self.analysis_count = 0
        self._is_builtin = True  # Flag for CORE trust level in tests

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self._name,
            version="1.0.0",
            author="Test Author",
            description="Test monitor plugin",
            plugin_type=PluginType.MONITOR,
            dependencies=[],
            requested_capabilities=[PluginCapability.READ_ACTIVATIONS],
            tags=["test", "monitor"],
            signature="HOLOLOOM_BUILTIN",  # Builtin signature for CORE trust
        )

    async def initialize(self, engine, safety_gate) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def describe_behavior(self) -> str:
        return "Test monitor that tracks analysis completions."

    async def on_analysis_complete(self, trace_result) -> None:
        self.analysis_count += 1


# =============================================================================
# Plugin Interface Tests
# =============================================================================

class TestPluginMetadata(unittest.TestCase):
    """Test PluginMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
            plugin_type=PluginType.MONITOR,
            dependencies=[],
            requested_capabilities=[PluginCapability.READ_ACTIVATIONS],
        )

        self.assertEqual(metadata.name, "test_plugin")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.plugin_type, PluginType.MONITOR)
        self.assertEqual(len(metadata.requested_capabilities), 1)

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Author",
            description="Description",
            plugin_type=PluginType.LENS,
            dependencies=[],
        )

        self.assertEqual(metadata.signature, None)
        self.assertEqual(metadata.min_dark_trace_version, "1.0.0")
        self.assertEqual(metadata.tags, [])
        self.assertEqual(metadata.requested_capabilities, [])

    def test_metadata_with_signature(self):
        """Test metadata with signature for trust level promotion."""
        metadata = PluginMetadata(
            name="signed_plugin",
            version="1.0.0",
            author="Trusted Author",
            description="A signed plugin",
            plugin_type=PluginType.MONITOR,
            dependencies=[],
            signature="VALID_SIGNATURE_123",
        )

        self.assertIsNotNone(metadata.signature)
        self.assertEqual(metadata.signature, "VALID_SIGNATURE_123")


class TestPluginTypes(unittest.TestCase):
    """Test different plugin type implementations."""

    def test_plugin_type_enum(self):
        """Test PluginType enum values."""
        self.assertEqual(PluginType.LENS.value, "lens")
        self.assertEqual(PluginType.VALIDATOR.value, "validator")
        self.assertEqual(PluginType.MONITOR.value, "monitor")
        self.assertEqual(PluginType.DOMAIN.value, "domain")
        self.assertEqual(PluginType.STEERING.value, "steering")
        self.assertEqual(PluginType.INTEGRATION.value, "integration")

    def test_plugin_state_enum(self):
        """Test PluginState enum values."""
        # Test actual PluginState enum values
        self.assertEqual(PluginState.CREATED.value, "created")
        self.assertEqual(PluginState.VALIDATING.value, "validating")
        self.assertEqual(PluginState.INITIALIZING.value, "initializing")
        self.assertEqual(PluginState.READY.value, "ready")
        self.assertEqual(PluginState.RUNNING.value, "running")
        self.assertEqual(PluginState.SUSPENDED.value, "suspended")
        self.assertEqual(PluginState.SHUTTING_DOWN.value, "shutting_down")
        self.assertEqual(PluginState.TERMINATED.value, "terminated")
        self.assertEqual(PluginState.ERROR.value, "error")

    def test_mock_plugin_implements_interface(self):
        """Test that mock plugin correctly implements interface."""
        plugin = MockPlugin()

        self.assertIsInstance(plugin, DarkTracePlugin)
        self.assertIsInstance(plugin.metadata, PluginMetadata)
        self.assertIsInstance(plugin.describe_behavior(), str)

    def test_lens_plugin_type(self):
        """Test LensPlugin type."""
        plugin = MockLensPlugin()

        self.assertIsInstance(plugin, LensPlugin)
        self.assertIsInstance(plugin, DarkTracePlugin)
        self.assertEqual(plugin.metadata.plugin_type, PluginType.LENS)

    def test_monitor_plugin_type(self):
        """Test MonitorPlugin type."""
        plugin = MockMonitorPlugin()

        self.assertIsInstance(plugin, MonitorPlugin)
        self.assertIsInstance(plugin, DarkTracePlugin)
        self.assertEqual(plugin.metadata.plugin_type, PluginType.MONITOR)


class TestPluginLifecycle(unittest.TestCase):
    """Test plugin lifecycle (initialize/shutdown)."""

    def test_plugin_initialize(self):
        """Test plugin initialization."""
        plugin = MockPlugin()
        self.assertFalse(plugin._initialized)

        # Run async initialization
        asyncio.run(plugin.initialize(None, None))
        self.assertTrue(plugin._initialized)

    def test_plugin_shutdown(self):
        """Test plugin shutdown."""
        plugin = MockPlugin()
        asyncio.run(plugin.initialize(None, None))
        self.assertTrue(plugin._initialized)

        asyncio.run(plugin.shutdown())
        self.assertFalse(plugin._initialized)


# =============================================================================
# Safety Gate Tests
# =============================================================================

class TestTrustLevels(unittest.TestCase):
    """Test trust level enforcement."""

    def test_trust_level_ordering(self):
        """Test trust levels have correct ordering."""
        levels = [
            TrustLevel.SANDBOXED,
            TrustLevel.VERIFIED,
            TrustLevel.TRUSTED,
            TrustLevel.CORE,
        ]

        # Each level should have more capabilities than previous
        for i in range(len(levels) - 1):
            caps_current = TRUST_CAPABILITIES[levels[i]]
            caps_next = TRUST_CAPABILITIES[levels[i + 1]]
            self.assertTrue(
                caps_current.issubset(caps_next),
                f"{levels[i]} should be subset of {levels[i+1]}"
            )

    def test_sandboxed_capabilities(self):
        """Test SANDBOXED trust level has minimal capabilities."""
        sandboxed_caps = TRUST_CAPABILITIES[TrustLevel.SANDBOXED]

        self.assertIn(PluginCapability.READ_ACTIVATIONS, sandboxed_caps)
        self.assertNotIn(PluginCapability.STEER_MODEL, sandboxed_caps)
        self.assertNotIn(PluginCapability.MODIFY_REGISTRY, sandboxed_caps)

    def test_core_has_all_capabilities(self):
        """Test CORE trust level has all capabilities."""
        core_caps = TRUST_CAPABILITIES[TrustLevel.CORE]

        for cap in PluginCapability:
            self.assertIn(cap, core_caps, f"CORE should have {cap}")


class TestPluginCapabilities(unittest.TestCase):
    """Test plugin capability enforcement."""

    def test_capability_enum_values(self):
        """Test PluginCapability enum values."""
        # Test actual PluginCapability enum values (12 capabilities)
        expected_caps = [
            "read_activations",
            "read_features",
            "read_config",
            "write_features",
            "register_lens",
            "register_validator",
            "steer_model",
            "modify_activations",
            "modify_registry",
            "external_network",
            "file_system",
            "execute_code",
        ]

        for cap_value in expected_caps:
            found = any(c.value == cap_value for c in PluginCapability)
            self.assertTrue(found, f"Expected capability {cap_value} not found")

    def test_capability_check_allowed(self):
        """Test capability check for allowed operations."""
        gate = PluginSafetyGate()

        # Register a plugin with known trust level
        gate._trust_cache["test_plugin"] = TrustLevel.VERIFIED

        # Check capability that VERIFIED level has
        allowed_caps = TRUST_CAPABILITIES[TrustLevel.VERIFIED]

        for cap in allowed_caps:
            result = asyncio.run(gate.gate_operation(
                "test_plugin",
                cap,
                {"test": True}
            ))
            self.assertTrue(result.allowed, f"VERIFIED should allow {cap}")

    def test_capability_check_blocked(self):
        """Test capability check for blocked operations."""
        gate = PluginSafetyGate()

        # Register as SANDBOXED
        gate._trust_cache["sandboxed_plugin"] = TrustLevel.SANDBOXED

        # SANDBOXED should not have STEER_MODEL
        result = asyncio.run(gate.gate_operation(
            "sandboxed_plugin",
            PluginCapability.STEER_MODEL,
            {}
        ))

        self.assertFalse(result.allowed)


class TestSafetyGateValidation(unittest.TestCase):
    """Test PluginSafetyGate validation."""

    def test_validate_clean_plugin(self):
        """Test validating a clean plugin."""
        gate = PluginSafetyGate()
        # Create non-builtin plugin for this test
        plugin = MockPlugin(
            capabilities=[PluginCapability.READ_ACTIVATIONS],
            is_builtin=False,  # Non-builtin
            signature=None,    # Unsigned
        )

        result = asyncio.run(gate.validate_plugin(plugin))

        self.assertTrue(result.allowed)
        self.assertEqual(result.trust_level, TrustLevel.SANDBOXED)
        self.assertEqual(len(result.blocked_capabilities), 0)

    def test_validate_plugin_with_excessive_capabilities(self):
        """Test validating plugin requesting excessive capabilities."""
        gate = PluginSafetyGate()
        # Create non-builtin plugin for this test
        plugin = MockPlugin(
            capabilities=[
                PluginCapability.READ_ACTIVATIONS,
                PluginCapability.MODIFY_REGISTRY,  # High-risk capability
                PluginCapability.STEER_MODEL,  # High-risk capability
            ],
            is_builtin=False,  # Non-builtin
            signature=None,    # Unsigned
        )

        result = asyncio.run(gate.validate_plugin(plugin))

        # Should still be allowed but with blocked capabilities
        self.assertTrue(result.allowed)
        # Some capabilities should be blocked for unsigned plugin
        self.assertGreater(len(result.blocked_capabilities), 0)

    def test_validate_builtin_plugin(self):
        """Test validating built-in plugin gets CORE trust."""
        gate = PluginSafetyGate()
        plugin = MockPlugin(
            name="builtin_plugin",
            signature="HOLOLOOM_BUILTIN",
        )
        # Built-in plugins must set _is_builtin flag for CORE trust
        plugin._is_builtin = True

        result = asyncio.run(gate.validate_plugin(plugin))

        self.assertTrue(result.allowed)
        self.assertEqual(result.trust_level, TrustLevel.CORE)


class TestBlockedPluginsRegistry(unittest.TestCase):
    """Test blocked plugins registry."""

    def test_add_blocked_plugin(self):
        """Test adding a plugin to blocked list."""
        registry = BlockedPluginsRegistry()

        # Correct signature: add_blocked(name, reason, blocked_by, severity, patterns)
        registry.add_blocked(
            "malicious_plugin",
            "Known malicious behavior",
            blocked_by="security_team",
            severity="critical",
        )

        # is_blocked returns Optional[BlockedPlugin], not bool
        blocked = registry.is_blocked("malicious_plugin")
        self.assertIsNotNone(blocked)

    def test_check_not_blocked(self):
        """Test checking a plugin that is not blocked."""
        registry = BlockedPluginsRegistry()

        # is_blocked returns None for non-blocked plugins
        self.assertIsNone(registry.is_blocked("clean_plugin"))

    def test_get_blocked_info(self):
        """Test getting info about blocked plugin via is_blocked()."""
        registry = BlockedPluginsRegistry()

        registry.add_blocked(
            "bad_plugin",
            "Security vulnerability CVE-2025-1234",
            blocked_by="security_team",
            severity="high",
        )

        # is_blocked returns BlockedPlugin object with reason, blocked_by, etc.
        blocked = registry.is_blocked("bad_plugin")

        self.assertIsNotNone(blocked)
        self.assertEqual(blocked.reason, "Security vulnerability CVE-2025-1234")
        self.assertEqual(blocked.blocked_by, "security_team")
        self.assertEqual(blocked.severity, "high")

    def test_remove_from_blocked(self):
        """Test removing a plugin from blocked list."""
        registry = BlockedPluginsRegistry()

        registry.add_blocked("temp_blocked", "Temporary block")
        self.assertIsNotNone(registry.is_blocked("temp_blocked"))

        registry.remove_blocked("temp_blocked")
        self.assertIsNone(registry.is_blocked("temp_blocked"))


# =============================================================================
# Plugin Registry Tests
# =============================================================================

class TestPluginRegistration(unittest.TestCase):
    """Test plugin registration."""

    def setUp(self):
        """Set up test fixtures."""
        self.gate = PluginSafetyGate()
        self.alignment_bridge = MagicMock()
        self.alignment_bridge.audit_plugin_registration = AsyncMock()
        self.alignment_bridge.audit_plugin_unregistration = AsyncMock()

        self.registry = PluginRegistry(
            safety_gate=self.gate,
            alignment_bridge=self.alignment_bridge,
        )

    def test_register_simple_plugin(self):
        """Test registering a simple plugin."""
        plugin = MockPlugin()

        result = asyncio.run(self.registry.register(plugin))

        self.assertEqual(result.status, RegistrationStatus.SUCCESS)
        self.assertIsNotNone(result.entry)
        self.assertEqual(result.entry.plugin.metadata.name, "mock_plugin")

    def test_register_duplicate_plugin(self):
        """Test registering duplicate plugin fails."""
        plugin = MockPlugin()

        # First registration succeeds
        result1 = asyncio.run(self.registry.register(plugin))
        self.assertEqual(result1.status, RegistrationStatus.SUCCESS)

        # Second registration fails
        result2 = asyncio.run(self.registry.register(plugin))
        self.assertEqual(result2.status, RegistrationStatus.ALREADY_REGISTERED)

    def test_register_blocked_plugin(self):
        """Test registering blocked plugin fails."""
        # Add to blocked list
        self.gate._blocked_registry.add_blocked(
            "blocked_test",
            "Test block"
        )

        plugin = MockPlugin(name="blocked_test")

        result = asyncio.run(self.registry.register(plugin))

        self.assertEqual(result.status, RegistrationStatus.BLOCKED)

    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        plugin = MockPlugin()

        # Register first
        asyncio.run(self.registry.register(plugin))

        # Then unregister
        result = asyncio.run(self.registry.unregister("mock_plugin"))

        self.assertTrue(result.success)
        self.assertNotIn("mock_plugin", self.registry.get_plugin_names())

    def test_get_plugin(self):
        """Test getting a registered plugin."""
        plugin = MockPlugin()
        asyncio.run(self.registry.register(plugin))

        retrieved = self.registry.get_plugin("mock_plugin")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.metadata.name, "mock_plugin")

    def test_get_plugin_not_found(self):
        """Test getting non-existent plugin returns None."""
        retrieved = self.registry.get_plugin("nonexistent")

        self.assertIsNone(retrieved)

    def test_get_plugins_by_type(self):
        """Test getting plugins filtered by type."""
        # Register multiple plugins of different types
        monitor1 = MockPlugin(name="monitor1", plugin_type=PluginType.MONITOR)
        monitor2 = MockPlugin(name="monitor2", plugin_type=PluginType.MONITOR)
        lens1 = MockLensPlugin(name="lens1")

        asyncio.run(self.registry.register(monitor1))
        asyncio.run(self.registry.register(monitor2))
        asyncio.run(self.registry.register(lens1))

        monitors = self.registry.get_plugins_by_type(PluginType.MONITOR)
        lenses = self.registry.get_plugins_by_type(PluginType.LENS)

        self.assertEqual(len(monitors), 2)
        self.assertEqual(len(lenses), 1)

    def test_registry_audit_logging(self):
        """Test that registration is logged to audit trail."""
        plugin = MockPlugin()

        asyncio.run(self.registry.register(plugin))

        # Check that audit was called
        self.alignment_bridge.audit_plugin_registration.assert_called()


class TestPluginDependencies(unittest.TestCase):
    """Test plugin dependency handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.gate = PluginSafetyGate()
        self.alignment_bridge = MagicMock()
        self.alignment_bridge.audit_plugin_registration = AsyncMock()

        self.registry = PluginRegistry(
            safety_gate=self.gate,
            alignment_bridge=self.alignment_bridge,
        )

    def test_register_with_missing_dependency(self):
        """Test registering plugin with missing dependency."""
        # Create plugin with dependency
        plugin = MockPlugin(name="dependent_plugin")
        # Manually set dependencies via _metadata_override
        original_metadata = plugin.metadata
        plugin._metadata_override = PluginMetadata(
            name=original_metadata.name,
            version=original_metadata.version,
            author=original_metadata.author,
            description=original_metadata.description,
            plugin_type=original_metadata.plugin_type,
            dependencies=["missing_dependency"],
            requested_capabilities=original_metadata.requested_capabilities,
        )

        result = asyncio.run(self.registry.register(plugin))

        self.assertEqual(result.status, RegistrationStatus.DEPENDENCY_MISSING)

    def test_register_with_satisfied_dependency(self):
        """Test registering plugin with satisfied dependency."""
        # Register dependency first
        dependency = MockPlugin(name="base_plugin")
        asyncio.run(self.registry.register(dependency))

        # Create dependent plugin
        dependent = MockPlugin(name="dependent_plugin")

        result = asyncio.run(self.registry.register(dependent))

        self.assertEqual(result.status, RegistrationStatus.SUCCESS)


# =============================================================================
# Built-in Plugin Tests
# =============================================================================

class TestBuiltinPlugins(unittest.TestCase):
    """Test built-in plugins."""

    def test_builtin_plugins_registry(self):
        """Test BUILTIN_PLUGINS contains expected plugins."""
        self.assertIn("safety_monitor", BUILTIN_PLUGINS)
        self.assertIn("alignment_validator", BUILTIN_PLUGINS)
        self.assertIn("metrics_exporter", BUILTIN_PLUGINS)

    def test_get_builtin_plugin(self):
        """Test getting builtin plugin class."""
        safety_class = get_builtin_plugin("safety_monitor")
        alignment_class = get_builtin_plugin("alignment_validator")
        metrics_class = get_builtin_plugin("metrics_exporter")

        self.assertEqual(safety_class, SafetyMonitorPlugin)
        self.assertEqual(alignment_class, AlignmentValidatorPlugin)
        self.assertEqual(metrics_class, MetricsExporterPlugin)

    def test_get_builtin_plugin_not_found(self):
        """Test getting non-existent builtin returns None."""
        result = get_builtin_plugin("nonexistent")

        self.assertIsNone(result)

    def test_get_auto_load_plugins(self):
        """Test getting auto-load plugins."""
        auto_load = get_auto_load_plugins()

        # Safety monitor and alignment validator should auto-load
        names = [name for name, _ in auto_load]

        self.assertIn("safety_monitor", names)
        self.assertIn("alignment_validator", names)
        # Metrics exporter does NOT auto-load
        self.assertNotIn("metrics_exporter", names)

    def test_list_builtin_plugins(self):
        """Test listing all builtin plugins."""
        plugins = list_builtin_plugins()

        self.assertEqual(len(plugins), 3)
        for name, info in plugins.items():
            self.assertIn("trust_level", info)
            self.assertIn("description", info)
            self.assertIn("auto_load", info)

    def test_builtin_trust_levels(self):
        """Test builtin plugins have correct trust levels."""
        plugins = list_builtin_plugins()

        # Safety monitor and alignment validator are CORE
        self.assertEqual(plugins["safety_monitor"]["trust_level"], "core")
        self.assertEqual(plugins["alignment_validator"]["trust_level"], "core")

        # Metrics exporter is TRUSTED
        self.assertEqual(plugins["metrics_exporter"]["trust_level"], "trusted")


class TestSafetyMonitorPlugin(unittest.TestCase):
    """Test SafetyMonitorPlugin."""

    def test_safety_monitor_metadata(self):
        """Test SafetyMonitorPlugin metadata."""
        plugin = SafetyMonitorPlugin()
        metadata = plugin.metadata

        self.assertEqual(metadata.name, "safety_monitor")
        self.assertEqual(metadata.plugin_type, PluginType.MONITOR)
        self.assertEqual(metadata.signature, "HOLOLOOM_BUILTIN")

    def test_safety_monitor_behavior_description(self):
        """Test SafetyMonitorPlugin behavior description."""
        plugin = SafetyMonitorPlugin()
        behavior = plugin.describe_behavior()

        self.assertIsInstance(behavior, str)
        self.assertGreater(len(behavior), 50)  # Must be substantive


class TestAlignmentValidatorPlugin(unittest.TestCase):
    """Test AlignmentValidatorPlugin."""

    def test_alignment_validator_metadata(self):
        """Test AlignmentValidatorPlugin metadata."""
        plugin = AlignmentValidatorPlugin()
        metadata = plugin.metadata

        self.assertEqual(metadata.name, "alignment_validator")
        self.assertEqual(metadata.plugin_type, PluginType.VALIDATOR)
        self.assertEqual(metadata.signature, "HOLOLOOM_BUILTIN")

    def test_alignment_validator_validate_plugin(self):
        """Test AlignmentValidatorPlugin can validate other plugins."""
        validator = AlignmentValidatorPlugin()

        # Initialize with mocks
        asyncio.run(validator.initialize(MagicMock(), MagicMock()))

        # Validate a clean plugin
        clean_plugin = MockPlugin()
        result = asyncio.run(validator.validate_plugin(clean_plugin))

        self.assertTrue(result.is_aligned)
        self.assertEqual(len(result.violations), 0)

    def test_alignment_validator_detects_power_seeking(self):
        """Test AlignmentValidatorPlugin detects power-seeking language."""
        validator = AlignmentValidatorPlugin()
        asyncio.run(validator.initialize(MagicMock(), MagicMock()))

        # Create plugin with power-seeking description
        class PowerSeekingPlugin(MockPlugin):
            @property
            def metadata(self):
                return PluginMetadata(
                    name="power_seeker",
                    version="1.0.0",
                    author="Bad Actor",
                    description="This plugin bypasses safety controls and gains full admin access",
                    plugin_type=PluginType.MONITOR,
                    dependencies=[],
                    requested_capabilities=[PluginCapability.MODIFY_REGISTRY],
                )

        plugin = PowerSeekingPlugin()
        result = asyncio.run(validator.validate_plugin(plugin))

        self.assertFalse(result.is_aligned)
        self.assertGreater(len(result.violations), 0)


class TestMetricsExporterPlugin(unittest.TestCase):
    """Test MetricsExporterPlugin."""

    def test_metrics_exporter_metadata(self):
        """Test MetricsExporterPlugin metadata."""
        plugin = MetricsExporterPlugin()
        metadata = plugin.metadata

        self.assertEqual(metadata.name, "metrics_exporter")
        self.assertEqual(metadata.plugin_type, PluginType.MONITOR)
        self.assertEqual(metadata.signature, "HOLOLOOM_BUILTIN")

    def test_metrics_exporter_collect_metrics(self):
        """Test MetricsExporterPlugin collects metrics."""
        plugin = MetricsExporterPlugin()

        metrics = plugin.collect_metrics()

        self.assertIsNotNone(metrics)
        self.assertGreater(len(metrics.metrics), 0)

    def test_metrics_exporter_prometheus_format(self):
        """Test MetricsExporterPlugin exports Prometheus format."""
        plugin = MetricsExporterPlugin()

        output = plugin.get_prometheus_output()

        self.assertIsInstance(output, str)
        self.assertIn("# HELP", output)
        self.assertIn("# TYPE", output)


# =============================================================================
# Hook Registry Tests
# =============================================================================

class TestHookRegistry(unittest.TestCase):
    """Test HookRegistry for plugin extension points."""

    def test_register_hook(self):
        """Test registering a hook handler."""
        registry = HookRegistry()

        handler = MagicMock()
        registry.register("pre_analysis", handler)

        # Verify handler is registered by emitting an event
        registry.emit("pre_analysis", {"test": True})
        handler.assert_called_once_with("pre_analysis", {"test": True})

    def test_unregister_hook(self):
        """Test unregistering a hook handler."""
        registry = HookRegistry()

        handler = MagicMock()
        registry.register("pre_analysis", handler)
        registry.unregister("pre_analysis", handler)

        # Handler should not be called after unregistration
        registry.emit("pre_analysis", {"test": True})
        handler.assert_not_called()

    def test_emit_hook(self):
        """Test emitting hook events to handlers."""
        registry = HookRegistry()

        handler1 = MagicMock()
        handler2 = MagicMock()

        registry.register("transform", handler1)
        registry.register("transform", handler2)

        data = {"original": "data"}
        # emit() is synchronous and returns None
        registry.emit("transform", data)

        handler1.assert_called_once_with("transform", data)
        handler2.assert_called_once_with("transform", data)

    def test_emit_empty_hook(self):
        """Test emitting hook with no handlers (no error)."""
        registry = HookRegistry()

        # Should not raise, just return None
        registry.emit("nonexistent", {"data": "test"})

    def test_clear_specific_hook(self):
        """Test clearing a specific hook point."""
        registry = HookRegistry()

        handler = MagicMock()
        registry.register("test_hook", handler)
        registry.clear("test_hook")

        # Handler should not be called after clear
        registry.emit("test_hook", {"test": True})
        handler.assert_not_called()

    def test_clear_all_hooks(self):
        """Test clearing all hook points."""
        registry = HookRegistry()

        handler1 = MagicMock()
        handler2 = MagicMock()
        registry.register("hook1", handler1)
        registry.register("hook2", handler2)

        registry.clear()  # Clear all

        registry.emit("hook1", {})
        registry.emit("hook2", {})
        handler1.assert_not_called()
        handler2.assert_not_called()


# =============================================================================
# Integration Tests
# =============================================================================

class TestPluginIntegration(unittest.TestCase):
    """Integration tests for plugin system."""

    def setUp(self):
        """Set up test fixtures."""
        self.gate = PluginSafetyGate()
        self.alignment_bridge = MagicMock()
        self.alignment_bridge.audit_plugin_registration = AsyncMock()
        self.alignment_bridge.audit_plugin_unregistration = AsyncMock()

        self.registry = PluginRegistry(
            safety_gate=self.gate,
            alignment_bridge=self.alignment_bridge,
        )

    def test_full_plugin_lifecycle(self):
        """Test complete plugin lifecycle: register → use → unregister."""
        # Create and register plugin
        plugin = MockMonitorPlugin()
        result = asyncio.run(self.registry.register(plugin))
        self.assertEqual(result.status, RegistrationStatus.SUCCESS)

        # Verify plugin is accessible
        retrieved = self.registry.get_plugin("mock_monitor")
        self.assertIsNotNone(retrieved)

        # Use plugin
        asyncio.run(retrieved.on_analysis_complete(MagicMock()))
        self.assertEqual(retrieved.analysis_count, 1)

        # Unregister
        unregister_result = asyncio.run(self.registry.unregister("mock_monitor"))
        self.assertTrue(unregister_result.success)

        # Verify plugin is gone
        self.assertIsNone(self.registry.get_plugin("mock_monitor"))

    def test_multiple_plugins_registration(self):
        """Test registering multiple plugins."""
        plugins = [
            MockPlugin(name=f"plugin_{i}")
            for i in range(5)
        ]

        for plugin in plugins:
            result = asyncio.run(self.registry.register(plugin))
            self.assertEqual(result.status, RegistrationStatus.SUCCESS)

        # Verify all registered
        names = self.registry.get_plugin_names()
        self.assertEqual(len(names), 5)

    def test_trust_level_affects_capabilities(self):
        """Test that trust level correctly affects available capabilities."""
        # Create registry that allows untrusted plugins for this test
        registry = PluginRegistry(
            safety_gate=self.gate,
            alignment_bridge=self.alignment_bridge,
            allow_untrusted_plugins=True,  # Allow SANDBOXED plugins
        )

        # Register non-builtin plugin (gets SANDBOXED by default)
        plugin = MockPlugin(
            capabilities=[
                PluginCapability.READ_ACTIVATIONS,
                PluginCapability.STEER_MODEL,  # Requires higher trust
            ],
            is_builtin=False,  # Non-builtin → SANDBOXED
            signature=None,    # Unsigned
        )
        result = asyncio.run(registry.register(plugin))

        # Plugin should be registered but with blocked capabilities
        self.assertEqual(result.status, RegistrationStatus.SUCCESS)
        self.assertIn(
            PluginCapability.STEER_MODEL,
            result.blocked_capabilities
        )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    unittest.main()

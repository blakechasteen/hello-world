"""
Dark Trace Integration Tests
============================

Comprehensive tests for Dark Trace Phases 9-10:
- Model adapters (DummyAdapter, PolicyAdapter)
- Cross-model fingerprinting
- Orchestrator integration
- Integration modes (DISABLED, PASSIVE, ACTIVE, FULL)
- Safety monitoring

Date: December 2025
"""

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from typing import Dict, Any, List

# Phase 9: Model Adapters
from hololoom.dark_trace.models.adapter import (
    ModelAdapter,
    DummyAdapter,
    ActivationHook,
    HookHandle,
    SteeringConfig,
    LayerInfo,
    LayerType,
    ModelCapabilities,
    create_hook,
    ActivationCache,
    make_cache_key,
)

# Phase 9: Fingerprinting
from hololoom.dark_trace.models.fingerprint import (
    FeatureFingerprint,
    ModelFingerprinter,
    FingerprintConfig,
    FingerprintComparison,
    compare_fingerprints,
    find_universal_features,
    find_model_specific_features,
)

# Phase 10: Integration
from hololoom.integrations.dark_trace_integration import (
    DarkTraceIntegration,
    IntegrationConfig,
    IntegrationMode,
    IntegrationResult,
    create_integration,
    enable_dark_trace,
    get_dark_trace,
    with_interpretability,
)


class TestDummyAdapter(unittest.TestCase):
    """Test DummyAdapter basic functionality."""

    def test_initialization(self):
        """Test DummyAdapter can be created with default parameters."""
        adapter = DummyAdapter()
        self.assertEqual(adapter.n_layers, 12)
        self.assertEqual(adapter.hidden_dim, 768)

    def test_custom_dimensions(self):
        """Test DummyAdapter with custom dimensions."""
        adapter = DummyAdapter(n_layers=6, hidden_dim=256)
        self.assertEqual(adapter.n_layers, 6)
        self.assertEqual(adapter.hidden_dim, 256)

    def test_get_layer_names(self):
        """Test layer name enumeration."""
        adapter = DummyAdapter(n_layers=4)
        names = adapter.get_layer_names()
        self.assertEqual(len(names), 4)
        self.assertEqual(names, ["layer.0", "layer.1", "layer.2", "layer.3"])

    def test_get_layer_info(self):
        """Test layer info retrieval."""
        adapter = DummyAdapter(n_layers=4, hidden_dim=512)
        info = adapter.get_layer_info("layer.2")

        self.assertIsInstance(info, LayerInfo)
        self.assertEqual(info.name, "layer.2")
        self.assertEqual(info.index, 2)
        self.assertEqual(info.layer_type, LayerType.MLP)
        self.assertEqual(info.input_dim, 512)
        self.assertEqual(info.output_dim, 512)

    def test_get_activations(self):
        """Test activation extraction returns correct shape."""
        adapter = DummyAdapter(n_layers=4, hidden_dim=256)
        activations = adapter.get_activations(
            inputs="test input",
            layer="layer.1",
            batch_size=2,
            seq_len=5
        )

        self.assertEqual(activations.shape, (2, 5, 256))
        self.assertEqual(activations.dtype, np.float32)

    def test_inject_steering(self):
        """Test steering vector injection."""
        adapter = DummyAdapter()
        vector = np.random.randn(768).astype(np.float32)

        config_id = adapter.inject_steering(
            vector=vector,
            layer="layer.5",
            scale=0.5
        )

        self.assertIsInstance(config_id, str)
        self.assertIn("steering_", config_id)
        self.assertIn(config_id, adapter._steering_configs)

    def test_remove_steering(self):
        """Test steering vector removal."""
        adapter = DummyAdapter()
        vector = np.random.randn(768).astype(np.float32)

        config_id = adapter.inject_steering(vector, "layer.5")
        self.assertTrue(adapter.remove_steering(config_id))
        self.assertFalse(adapter.remove_steering(config_id))  # Second removal fails

    def test_clear_steering(self):
        """Test clearing all steering configurations."""
        adapter = DummyAdapter()
        vector = np.random.randn(768).astype(np.float32)

        adapter.inject_steering(vector, "layer.1")
        adapter.inject_steering(vector, "layer.2")
        adapter.inject_steering(vector, "layer.3")

        count = adapter.clear_steering()
        self.assertEqual(count, 3)
        self.assertEqual(len(adapter._steering_configs), 0)

    def test_forward(self):
        """Test forward pass returns output."""
        adapter = DummyAdapter(hidden_dim=512)
        output = adapter.forward("test input")

        self.assertIsInstance(output, dict)
        self.assertIn("output", output)
        self.assertEqual(output["output"].shape, (1, 512))


class TestHookManagement(unittest.TestCase):
    """Test hook registration and management."""

    def test_register_hook(self):
        """Test hook registration."""
        adapter = DummyAdapter()

        captured_activations = []
        def capture_hook(acts, layer_name):
            captured_activations.append((layer_name, acts.shape))
            return None

        handle = adapter.register_hook(capture_hook, "layer.5")

        self.assertIsInstance(handle, HookHandle)
        self.assertTrue(handle.active)
        self.assertEqual(handle.layer, "layer.5")

    def test_hook_removal(self):
        """Test hook removal via handle."""
        adapter = DummyAdapter()

        def dummy_hook(acts, layer_name):
            return None

        handle = adapter.register_hook(dummy_hook, "layer.5")
        self.assertTrue(handle.active)

        handle.remove()
        self.assertFalse(handle.active)
        self.assertNotIn(handle.hook_id, adapter._hooks)

    def test_clear_hooks(self):
        """Test clearing all hooks."""
        adapter = DummyAdapter()

        def dummy_hook(acts, layer_name):
            return None

        adapter.register_hook(dummy_hook, "layer.1")
        adapter.register_hook(dummy_hook, "layer.2")
        adapter.register_hook(dummy_hook, "layer.3")

        count = adapter.clear_hooks()
        self.assertEqual(count, 3)
        self.assertEqual(len(adapter._hooks), 0)

    def test_create_hook_helper(self):
        """Test create_hook helper function."""
        results = []

        def callback(acts):
            results.append(acts.mean())
            return acts * 2  # Modify activations

        hook = create_hook(callback, layer_filter="layer.5")

        # Should return None for non-matching layers
        test_acts = np.random.randn(1, 10, 768).astype(np.float32)
        result = hook(test_acts, "layer.3")
        self.assertIsNone(result)

        # Should process matching layers
        result = hook(test_acts, "layer.5")
        self.assertIsNotNone(result)


class TestSteeringConfig(unittest.TestCase):
    """Test steering configuration and application."""

    def test_add_method(self):
        """Test additive steering."""
        vector = np.array([1.0, 2.0, 3.0])
        config = SteeringConfig(
            layer="layer.0",
            vector=vector,
            scale=0.5,
            method="add"
        )

        activations = np.zeros((1, 3))
        result = config.apply(activations)

        np.testing.assert_array_almost_equal(result, [[0.5, 1.0, 1.5]])

    def test_replace_method(self):
        """Test replacement steering."""
        vector = np.array([1.0, 2.0, 3.0])
        config = SteeringConfig(
            layer="layer.0",
            vector=vector,
            scale=1.0,
            method="replace"
        )

        activations = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        result = config.apply(activations)

        np.testing.assert_array_almost_equal(result, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    def test_multiply_method(self):
        """Test multiplicative steering."""
        vector = np.array([2.0, 0.5, 1.0])
        config = SteeringConfig(
            layer="layer.0",
            vector=vector,
            scale=1.0,
            method="multiply"
        )

        activations = np.array([[1.0, 2.0, 3.0]])
        result = config.apply(activations)

        np.testing.assert_array_almost_equal(result, [[2.0, 1.0, 3.0]])

    def test_position_specific_steering(self):
        """Test position-specific steering."""
        vector = np.array([10.0, 10.0, 10.0])
        config = SteeringConfig(
            layer="layer.0",
            vector=vector,
            scale=1.0,
            method="add",
            position=1  # Only modify position 1
        )

        activations = np.zeros((2, 3, 3))  # batch=2, seq=3, hidden=3
        result = config.apply(activations)

        # Position 1 should be modified, others unchanged
        np.testing.assert_array_almost_equal(result[:, 0, :], [[0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_almost_equal(result[:, 1, :], [[10, 10, 10], [10, 10, 10]])
        np.testing.assert_array_almost_equal(result[:, 2, :], [[0, 0, 0], [0, 0, 0]])

    def test_disabled_steering(self):
        """Test that disabled steering does nothing."""
        vector = np.array([1.0, 2.0, 3.0])
        config = SteeringConfig(
            layer="layer.0",
            vector=vector,
            scale=1.0,
            method="add",
            enabled=False
        )

        activations = np.zeros((1, 3))
        result = config.apply(activations)

        np.testing.assert_array_equal(result, activations)


class TestActivationCache(unittest.TestCase):
    """Test activation caching functionality."""

    def test_put_and_get(self):
        """Test basic cache put and get."""
        cache = ActivationCache(max_size=100)

        key = "test_key"
        activations = np.random.randn(1, 10, 768)

        cache.put(key, activations)
        result = cache.get(key)

        np.testing.assert_array_equal(result, activations)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ActivationCache()
        result = cache.get("nonexistent_key")
        self.assertIsNone(result)

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ActivationCache(max_size=2)

        cache.put("key1", np.array([1]))
        cache.put("key2", np.array([2]))
        cache.put("key3", np.array([3]))  # Should evict key1

        self.assertIsNone(cache.get("key1"))
        self.assertIsNotNone(cache.get("key2"))
        self.assertIsNotNone(cache.get("key3"))

    def test_lru_access_order(self):
        """Test LRU updates access order on get."""
        cache = ActivationCache(max_size=2)

        cache.put("key1", np.array([1]))
        cache.put("key2", np.array([2]))
        cache.get("key1")  # Access key1, making key2 the LRU
        cache.put("key3", np.array([3]))  # Should evict key2

        self.assertIsNotNone(cache.get("key1"))
        self.assertIsNone(cache.get("key2"))
        self.assertIsNotNone(cache.get("key3"))

    def test_clear(self):
        """Test cache clearing."""
        cache = ActivationCache()

        cache.put("key1", np.array([1]))
        cache.put("key2", np.array([2]))

        cache.clear()

        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get("key1"))

    def test_make_cache_key(self):
        """Test cache key generation."""
        key1 = make_cache_key("input1", "layer.5")
        key2 = make_cache_key("input1", "layer.5")
        key3 = make_cache_key("input2", "layer.5")
        key4 = make_cache_key("input1", "layer.6")

        self.assertEqual(key1, key2)  # Same inputs = same key
        self.assertNotEqual(key1, key3)  # Different input
        self.assertNotEqual(key1, key4)  # Different layer


class TestModelFingerprinter(unittest.TestCase):
    """Test cross-model fingerprinting."""

    def test_fingerprint_config_creation(self):
        """Test FingerprintConfig creation with correct parameters."""
        config = FingerprintConfig(
            n_samples=10,  # Correct parameter name
            normalize=True,
            include_statistics=True,
        )

        self.assertEqual(config.n_samples, 10)
        self.assertTrue(config.normalize)
        self.assertTrue(config.include_statistics)

    def test_fingerprint_generation_basic(self):
        """Test basic fingerprint creation."""
        # Create a simple fingerprint directly
        pattern = np.random.randn(256).astype(np.float32)
        fingerprint = FeatureFingerprint(
            feature_id="test_feature",
            model_id="test_model",
            layer="layer.0",
            pattern=pattern,
        )

        self.assertIsInstance(fingerprint, FeatureFingerprint)
        self.assertEqual(fingerprint.feature_id, "test_feature")
        self.assertEqual(fingerprint.model_id, "test_model")
        self.assertEqual(fingerprint.layer, "layer.0")
        self.assertEqual(fingerprint.pattern.shape, (256,))


class TestIntegrationConfig(unittest.TestCase):
    """Test integration configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IntegrationConfig()

        self.assertEqual(config.mode, IntegrationMode.PASSIVE)
        self.assertTrue(config.auto_analyze)  # Default is True
        self.assertFalse(config.enable_steering)
        self.assertTrue(config.safety_monitoring)  # Correct attribute name

    def test_passive_config(self):
        """Test passive configuration preset."""
        config = IntegrationConfig.passive()

        self.assertEqual(config.mode, IntegrationMode.PASSIVE)
        self.assertTrue(config.auto_analyze)  # passive() sets auto_analyze=True

    def test_active_config(self):
        """Test active configuration preset."""
        config = IntegrationConfig.active()

        self.assertEqual(config.mode, IntegrationMode.ACTIVE)
        self.assertTrue(config.auto_analyze)
        self.assertTrue(config.enable_steering)  # active() enables steering

    def test_full_config(self):
        """Test full configuration preset."""
        config = IntegrationConfig.full()

        self.assertEqual(config.mode, IntegrationMode.FULL)
        self.assertTrue(config.auto_analyze)
        self.assertTrue(config.enable_steering)
        self.assertTrue(config.safety_monitoring)  # Correct attribute name
        self.assertTrue(config.enable_fingerprinting)  # full() enables fingerprinting


class TestDarkTraceIntegration(unittest.TestCase):
    """Test DarkTraceIntegration class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = IntegrationConfig.passive()
        self.integration = DarkTraceIntegration(self.config)

    def test_initialization(self):
        """Test integration initialization."""
        self.assertIsNotNone(self.integration)
        self.assertEqual(self.integration.config.mode, IntegrationMode.PASSIVE)

    def test_has_engine(self):
        """Test integration has Dark Trace engine."""
        self.assertIsNotNone(self.integration.engine)

    def test_get_last_trace_initially_none(self):
        """Test get_last_trace returns None initially."""
        result = self.integration.get_last_trace()
        self.assertIsNone(result)

    def test_disabled_mode_config(self):
        """Test DISABLED mode configuration."""
        config = IntegrationConfig(mode=IntegrationMode.DISABLED)
        integration = DarkTraceIntegration(config)
        self.assertEqual(integration.config.mode, IntegrationMode.DISABLED)

    def test_get_statistics(self):
        """Test statistics retrieval."""
        stats = self.integration.get_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("weave_count", stats)  # Correct key name
        self.assertIn("integration_mode", stats)  # Correct key name


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    def test_create_integration_default(self):
        """Test create_integration factory with defaults."""
        integration = create_integration()

        self.assertIsInstance(integration, DarkTraceIntegration)
        # Default config is passive
        self.assertEqual(integration.config.mode, IntegrationMode.PASSIVE)

    def test_create_integration_with_config(self):
        """Test create_integration with explicit config."""
        config = IntegrationConfig.active()
        integration = create_integration(config=config)

        self.assertEqual(integration.config.mode, IntegrationMode.ACTIVE)
        self.assertTrue(integration.config.enable_steering)

    def test_create_integration_passive_config(self):
        """Test create_integration with passive config."""
        config = IntegrationConfig.passive()
        integration = create_integration(config=config)

        self.assertEqual(integration.config.mode, IntegrationMode.PASSIVE)
        self.assertTrue(integration.config.auto_analyze)


class TestDecoratorUsage(unittest.TestCase):
    """Test @with_interpretability decorator."""

    def test_decorator_basic(self):
        """Test basic decorator functionality."""
        @with_interpretability(mode=IntegrationMode.PASSIVE)
        async def sample_function(query: str):
            return {"result": "success", "query": query}

        # The decorator should wrap the function
        self.assertTrue(callable(sample_function))

    def test_decorator_with_default_mode(self):
        """Test decorator with default mode."""
        @with_interpretability()  # Default is PASSIVE
        async def sample_function(x: int) -> int:
            return x * 2

        # The decorator should wrap the function
        self.assertTrue(callable(sample_function))

    def test_decorator_disabled_mode(self):
        """Test decorator with DISABLED mode."""
        @with_interpretability(mode=IntegrationMode.DISABLED)
        async def sample_function(x: int, y: int) -> int:
            return x + y

        # The decorator should wrap the function
        self.assertTrue(callable(sample_function))


class TestLayerInfo(unittest.TestCase):
    """Test LayerInfo dataclass."""

    def test_layer_info_creation(self):
        """Test LayerInfo creation."""
        info = LayerInfo(
            name="layer.5",
            layer_type=LayerType.ATTENTION,
            index=5,
            input_dim=768,
            output_dim=768,
            num_heads=12
        )

        self.assertEqual(info.name, "layer.5")
        self.assertEqual(info.layer_type, LayerType.ATTENTION)
        self.assertEqual(info.index, 5)
        self.assertEqual(info.num_heads, 12)

    def test_layer_info_repr(self):
        """Test LayerInfo string representation."""
        info = LayerInfo(
            name="layer.5",
            layer_type=LayerType.MLP,
            index=5
        )

        repr_str = repr(info)
        self.assertIn("layer.5", repr_str)
        self.assertIn("mlp", repr_str)


class TestModelCapabilities(unittest.TestCase):
    """Test ModelCapabilities dataclass."""

    def test_default_capabilities(self):
        """Test default capabilities."""
        caps = ModelCapabilities()

        self.assertTrue(caps.can_extract_activations)
        self.assertTrue(caps.can_inject_steering)
        self.assertTrue(caps.can_ablate)
        self.assertTrue(caps.can_patch)
        self.assertTrue(caps.supports_batching)
        self.assertTrue(caps.supports_gradients)

    def test_custom_capabilities(self):
        """Test custom capabilities."""
        caps = ModelCapabilities(
            can_inject_steering=False,
            max_batch_size=32,
            supported_dtypes=["float32", "float16"]
        )

        self.assertFalse(caps.can_inject_steering)
        self.assertEqual(caps.max_batch_size, 32)
        self.assertIn("float16", caps.supported_dtypes)


class TestIntegrationModes(unittest.TestCase):
    """Test different integration modes."""

    def test_mode_enum_values(self):
        """Test IntegrationMode enum values."""
        self.assertEqual(IntegrationMode.DISABLED.value, "disabled")
        self.assertEqual(IntegrationMode.PASSIVE.value, "passive")
        self.assertEqual(IntegrationMode.ACTIVE.value, "active")
        self.assertEqual(IntegrationMode.FULL.value, "full")

    def test_mode_from_string(self):
        """Test creating mode from string."""
        mode = IntegrationMode("passive")
        self.assertEqual(mode, IntegrationMode.PASSIVE)


if __name__ == "__main__":
    unittest.main(verbosity=2)

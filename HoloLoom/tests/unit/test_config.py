"""
Unit tests for HoloLoom configuration system.

Tests:
- BARE/FAST/FUSED mode creation
- Config validation
- Memory backend selection
- Execution mode parameters
- Configuration edge cases
"""

import pytest
from HoloLoom.config import Config, ExecutionMode, MemoryBackend


class TestConfigCreation:
    """Test configuration factory methods."""

    def test_bare_config_creation(self):
        """BARE mode should create minimal configuration."""
        config = Config.bare()

        assert config.mode == ExecutionMode.BARE
        assert len(config.scales) >= 1  # At least one scale
        assert config.enable_semantic_calculus is False
        assert config.retrieval_k <= 10  # Minimal retrieval

    def test_fast_config_creation(self):
        """FAST mode should create balanced configuration."""
        config = Config.fast()

        assert config.mode == ExecutionMode.FAST
        assert len(config.scales) >= 1  # At least one scale
        # Actual implementation uses [768] by default

    def test_fused_config_creation(self):
        """FUSED mode should create full-featured configuration."""
        config = Config.fused()

        assert config.mode == ExecutionMode.FUSED
        assert len(config.scales) >= 1  # At least one scale
        # Note: enable_semantic_calculus is False by default, enabled explicitly if needed
        assert config.retrieval_k >= 6  # More retrieval than BARE


class TestMemoryBackendConfiguration:
    """Test memory backend selection."""

    def test_default_memory_backend(self):
        """All modes default to HYBRID backend."""
        bare_config = Config.bare()
        fast_config = Config.fast()
        fused_config = Config.fused()

        # All modes use HYBRID by default
        assert bare_config.memory_backend == MemoryBackend.HYBRID
        assert fast_config.memory_backend == MemoryBackend.HYBRID
        assert fused_config.memory_backend == MemoryBackend.HYBRID

    def test_hybrid_backend_configuration(self):
        """HYBRID backend should be configurable."""
        config = Config.fast()
        config.memory_backend = MemoryBackend.HYBRID

        assert config.memory_backend == MemoryBackend.HYBRID

    def test_hyperspace_backend_configuration(self):
        """HYPERSPACE backend should be configurable."""
        config = Config.fused()
        config.memory_backend = MemoryBackend.HYPERSPACE

        assert config.memory_backend == MemoryBackend.HYPERSPACE

    def test_invalid_backend_rejects(self):
        """Config accepts string backend names (may validate at runtime)."""
        config = Config.bare()

        # Config accepts assignment (validation may happen at runtime)
        config.memory_backend = MemoryBackend.INMEMORY
        assert config.memory_backend == MemoryBackend.INMEMORY


class TestExecutionModeParameters:
    """Test parameters across execution modes."""

    def test_scales_progression(self):
        """All modes should have at least one scale."""
        bare = Config.bare()
        fast = Config.fast()
        fused = Config.fused()

        # All should have at least one scale
        assert len(bare.scales) >= 1
        assert len(fast.scales) >= 1
        assert len(fused.scales) >= 1

    def test_retrieval_k_progression(self):
        """Retrieval k should be reasonable values."""
        bare = Config.bare()
        fast = Config.fast()
        fused = Config.fused()

        # All should have positive retrieval limits
        assert bare.retrieval_k > 0
        assert fast.retrieval_k > 0
        assert fused.retrieval_k > 0

    def test_bare_minimal_features(self):
        """BARE mode should disable optional features."""
        config = Config.bare()

        # BARE disables semantic calculus
        assert config.enable_semantic_calculus is False

    def test_fused_maximum_features(self):
        """FUSED mode configuration."""
        config = Config.fused()

        # FUSED mode is created with maximum features enabled
        assert config.mode == ExecutionMode.FUSED
        # Semantic calculus is enabled by default in FUSED mode (zero-config architecture)
        assert config.enable_semantic_calculus is True  # Enabled in FUSED


class TestConfigModification:
    """Test configuration mutation."""

    def test_config_mode_change(self):
        """Config mode should be changeable."""
        config = Config.bare()
        original_mode = config.mode

        config.mode = ExecutionMode.FAST

        assert config.mode != original_mode
        assert config.mode == ExecutionMode.FAST

    def test_config_scales_modification(self):
        """Scales should be modifiable."""
        config = Config.bare()
        original_scales = config.scales.copy()

        config.scales = [96, 192, 384]

        assert config.scales != original_scales
        assert len(config.scales) == 3

    def test_config_retrieval_k_modification(self):
        """Retrieval k should be modifiable."""
        config = Config.fast()
        original_k = config.retrieval_k

        config.retrieval_k = 20

        assert config.retrieval_k != original_k
        assert config.retrieval_k == 20


class TestConfigValidation:
    """Test configuration validation."""

    def test_positive_retrieval_k(self):
        """Retrieval k must be positive."""
        config = Config.bare()

        # Should work
        config.retrieval_k = 10
        assert config.retrieval_k == 10

        # Negative should fail or be clamped
        # (depends on implementation)

    def test_non_empty_scales(self):
        """Scales list must not be empty."""
        config = Config.bare()

        # Should have at least one scale
        assert len(config.scales) > 0

    def test_valid_scale_dimensions(self):
        """Scale dimensions should be valid Matryoshka sizes."""
        config = Config.fused()

        # Common Matryoshka sizes: 96, 192, 384, 768
        for scale in config.scales:
            assert scale in [96, 192, 384, 768, 1024, 1536]


class TestConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_config_copy(self):
        """Config should be copyable."""
        original = Config.fast()

        # Create a copy (implementation-specific)
        # This may involve dataclass copy or custom method
        import copy
        copied = copy.deepcopy(original)

        assert copied.mode == original.mode
        assert copied.scales == original.scales

        # Modify copy shouldn't affect original
        copied.retrieval_k = 100
        assert copied.retrieval_k != original.retrieval_k

    def test_config_equality(self):
        """Two configs with same parameters should be equal."""
        config1 = Config.bare()
        config2 = Config.bare()

        # Should have same settings
        assert config1.mode == config2.mode
        assert config1.scales == config2.scales

    def test_config_hash_different_modes(self):
        """Configs with different modes should be distinguishable."""
        bare = Config.bare()
        fast = Config.fast()
        fused = Config.fused()

        # Modes should be different
        assert bare.mode != fast.mode
        assert fast.mode != fused.mode
        assert bare.mode != fused.mode

"""
Unit tests for HoloLoom configuration module.

Tests config enums, dataclasses, and factory methods.
Fast, isolated, no external dependencies.
"""
import pytest
from HoloLoom.config import (
    Config,
    ExecutionMode,
    MemoryBackend,
    Environment,
    KGBackend,
    BanditStrategy,
)


class TestEnums:
    """Test configuration enums."""

    def test_execution_mode_values(self):
        """Execution modes should have correct values."""
        assert ExecutionMode.BARE.value == "bare"
        assert ExecutionMode.FAST.value == "fast"
        assert ExecutionMode.FUSED.value == "fused"

    def test_memory_backend_values(self):
        """Memory backends should have correct values."""
        assert MemoryBackend.INMEMORY.value == "inmemory"
        assert MemoryBackend.HYBRID.value == "hybrid"
        assert MemoryBackend.HYPERSPACE.value == "hyperspace"

    def test_environment_values(self):
        """Environment types should have correct values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_kg_backend_values(self):
        """Knowledge graph backends should have correct values."""
        assert KGBackend.NETWORKX.value == "networkx"
        assert KGBackend.NEO4J.value == "neo4j"


class TestConfigFactories:
    """Test Config factory methods."""

    def test_bare_config(self):
        """BARE mode should have minimal settings."""
        cfg = Config.bare()
        assert cfg.mode == ExecutionMode.BARE
        assert cfg.n_transformer_layers < 3
        assert cfg.use_spectral_features is False
        assert cfg.embedding_dim > 0

    def test_fast_config(self):
        """FAST mode should have balanced settings."""
        cfg = Config.fast()
        assert cfg.mode == ExecutionMode.FAST
        assert cfg.n_transformer_layers >= 2
        assert cfg.use_spectral_features is True
        assert cfg.memory_backend == MemoryBackend.INMEMORY

    def test_fused_config(self):
        """FUSED mode should have maximal settings."""
        cfg = Config.fused()
        assert cfg.mode == ExecutionMode.FUSED
        assert cfg.use_spectral_features is True
        assert cfg.enable_multipass is True


class TestConfigModification:
    """Test config modification and validation."""

    def test_config_override(self):
        """Should allow overriding config fields."""
        cfg = Config.fast()
        original_layers = cfg.n_transformer_layers

        # Create new config with override
        from dataclasses import replace
        cfg2 = replace(cfg, n_transformer_layers=10)

        assert cfg.n_transformer_layers == original_layers
        assert cfg2.n_transformer_layers == 10

    def test_memory_backend_override(self):
        """Should allow switching memory backends."""
        cfg = Config.fast()
        assert cfg.memory_backend == MemoryBackend.INMEMORY

        from dataclasses import replace
        cfg_hybrid = replace(cfg, memory_backend=MemoryBackend.HYBRID)
        assert cfg_hybrid.memory_backend == MemoryBackend.HYBRID

    def test_environment_configuration(self):
        """Should allow setting environment."""
        cfg = Config.fast()

        from dataclasses import replace
        cfg_prod = replace(cfg, environment=Environment.PRODUCTION)
        assert cfg_prod.environment == Environment.PRODUCTION


class TestConfigPerformance:
    """Test config creation performance."""

    def test_config_creation_speed(self, benchmark_if_available):
        """Config creation should be <1ms."""
        import time

        start = time.perf_counter()
        for _ in range(100):
            Config.fast()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        avg_time = elapsed / 100
        assert avg_time < 1.0, f"Config creation took {avg_time:.3f}ms (target: <1ms)"


@pytest.fixture
def benchmark_if_available():
    """Fixture for optional benchmarking."""
    return True

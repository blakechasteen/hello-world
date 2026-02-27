"""
Test suite for Zero-Config Architecture + Expansion Bundles.

Date: December 2025
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from hololoom.config import Config, ExecutionMode, Environment


class TestConfigPresets:
    """Test that config presets work correctly."""

    def test_bare_preset(self):
        """BARE mode should have minimal settings."""
        config = Config.bare()
        assert config.mode == ExecutionMode.BARE
        assert config.scales == [768]
        assert config.n_transformer_layers == 1
        assert config.enable_linguistic_gate is False
        assert config.enable_zero_copy_embeddings is False

    def test_fast_preset(self):
        """FAST mode should have balanced settings."""
        config = Config.fast()
        assert config.mode == ExecutionMode.FAST
        assert config.scales == [768]
        assert config.n_transformer_layers == 2
        assert config.enable_linguistic_gate is True
        assert config.enable_zero_copy_embeddings is True

    def test_fused_preset(self):
        """FUSED mode should have full settings."""
        config = Config.fused()
        assert config.mode == ExecutionMode.FUSED
        assert config.enable_linguistic_gate is True
        assert config.enable_semantic_calculus is True

    def test_research_preset(self):
        """RESEARCH mode should enable experimental features."""
        config = Config.research()
        assert config.mode == ExecutionMode.RESEARCH
        assert config.enable_spring_activation is True
        assert config.enable_recursive_learning is True

    def test_default_is_fast(self):
        """Default config should be FAST mode."""
        config = Config()
        assert config.mode == ExecutionMode.FAST


class TestModeDefaults:
    """Test that mode defaults apply correctly."""

    def test_mode_defaults_applied(self):
        """Mode defaults should override field defaults."""
        bare = Config.bare()
        fused = Config.fused()

        # BARE should have fewer transformer layers
        assert bare.n_transformer_layers == 1
        assert fused.n_transformer_layers == 2

        # BARE should have linguistic gate disabled
        assert bare.enable_linguistic_gate is False
        assert fused.enable_linguistic_gate is True

    def test_explicit_override_beats_mode(self):
        """Explicit field values should override mode defaults."""
        config = Config(mode=ExecutionMode.BARE, n_transformer_layers=4)
        # Explicit value should win
        assert config.n_transformer_layers == 4


class TestExpansionBundles:
    """Test expansion bundle loading."""

    def test_physics_expansion_loads(self):
        """Physics expansion should load correctly."""
        from hololoom.expansions.physics import PhysicsConfig, gp_thompson

        config = Config.research()
        config.load_expansion(gp_thompson())

        assert config.use_gp_bandits is True
        assert config.gp_acquisition == "thompson"
        assert config.gp_kernel_type == "matern"

    def test_bayesian_expansion_loads(self):
        """Bayesian expansion should load correctly."""
        from hololoom.expansions.bayesian import bayesian_default

        config = Config.research()
        config.load_expansion(bayesian_default())

        assert config.use_bayesian is True
        assert config.bayesian_samples == 10

    def test_geometry_expansion_loads(self):
        """Geometry expansion should load correctly."""
        from hololoom.expansions.geometry import product_manifold

        config = Config.research()
        config.load_expansion(product_manifold())

        assert config.use_riemannian is True
        assert config.riemannian_hyperbolic_dim == 256
        assert config.riemannian_spherical_dim == 256

    def test_spectral_expansion_loads(self):
        """Advanced spectral expansion should load correctly."""
        from hololoom.expansions.advanced_spectral import full_spectral

        config = Config.research()
        config.load_expansion(full_spectral())

        assert config.use_wavelets is True
        assert config.use_diffusion_maps is True
        assert config.use_multiscale_spectral is True

    def test_multiple_expansions(self):
        """Multiple expansions should load without conflict."""
        from hololoom.expansions.physics import gp_thompson
        from hololoom.expansions.geometry import product_manifold
        from hololoom.expansions.bayesian import bayesian_default

        config = Config.research()
        config.load_expansion(gp_thompson())
        config.load_expansion(product_manifold())
        config.load_expansion(bayesian_default())

        # All should be enabled
        assert config.use_gp_bandits is True
        assert config.use_riemannian is True
        assert config.use_bayesian is True

    def test_expansion_chaining(self):
        """Expansions should be chainable."""
        from hololoom.expansions.physics import gp_thompson
        from hololoom.expansions.bayesian import bayesian_default

        config = (Config.research()
            .load_expansion(gp_thompson())
            .load_expansion(bayesian_default()))

        assert config.use_gp_bandits is True
        assert config.use_bayesian is True


class TestExpansionRegistry:
    """Test expansion registry functions."""

    def test_list_expansions(self):
        """Should list all available expansions."""
        from hololoom.expansions import list_expansions

        expansions = list_expansions()
        assert "physics" in expansions
        assert "bayesian" in expansions
        assert "geometry" in expansions
        assert "advanced_spectral" in expansions

    def test_get_expansion(self):
        """Should get expansion class by name."""
        from hololoom.expansions import get_expansion

        PhysicsConfig = get_expansion("physics")
        assert PhysicsConfig is not None

        config = Config.research()
        config.load_expansion(PhysicsConfig(use_gp_bandits=True))
        assert config.use_gp_bandits is True


class TestBackwardCompatibility:
    """Test backward compatibility with legacy field access."""

    def test_direct_field_access(self):
        """Direct field access should still work."""
        config = Config.research()
        config.use_gp_bandits = True
        config.gp_acquisition = "ucb"

        assert config.use_gp_bandits is True
        assert config.gp_acquisition == "ucb"

    def test_all_legacy_fields_exist(self):
        """All legacy fields should be accessible."""
        config = Config()

        # Sample of legacy fields that should exist
        legacy_fields = [
            "scales",
            "n_transformer_layers",
            "n_attention_heads",
            "enable_linguistic_gate",
            "enable_zero_copy_embeddings",
            "use_gp_bandits",
            "use_riemannian",
            "use_bayesian",
            "use_wavelets",
            "retrieval_k",
            "pipeline_timeout",
        ]

        for field in legacy_fields:
            assert hasattr(config, field), f"Missing field: {field}"


class TestAutoDetection:
    """Test auto-detection features."""

    def test_environment_detection(self):
        """Environment should be auto-detected."""
        config = Config()
        # Should have a valid environment
        assert config.environment in [
            Environment.DEVELOPMENT,
            Environment.STAGING,
            Environment.PRODUCTION,
        ]

    def test_llm_auto_detection_returns_values(self):
        """LLM auto-detection should return provider and model."""
        config = Config()
        # llm_provider may be None if no API keys set, that's OK
        # Just verify the fields exist
        assert hasattr(config, "llm_provider")
        assert hasattr(config, "llm_model")


class TestExpansionPresets:
    """Test expansion preset functions."""

    def test_physics_presets(self):
        """Physics presets should return valid configs."""
        from hololoom.expansions.physics import (
            gp_thompson,
            gp_ucb,
            semantic_heat_diffusion,
            semantic_wave,
            semantic_competition,
        )

        presets = [
            gp_thompson(),
            gp_ucb(),
            semantic_heat_diffusion(),
            semantic_wave(),
            semantic_competition(),
        ]

        for preset in presets:
            assert hasattr(preset, "get_settings")
            settings = preset.get_settings()
            assert isinstance(settings, dict)

    def test_bayesian_presets(self):
        """Bayesian presets should return valid configs."""
        from hololoom.expansions.bayesian import (
            bayesian_default,
            bayesian_high_confidence,
            bayesian_regularized,
        )

        presets = [
            bayesian_default(),
            bayesian_high_confidence(),
            bayesian_regularized(),
        ]

        for preset in presets:
            assert hasattr(preset, "get_settings")
            settings = preset.get_settings()
            assert isinstance(settings, dict)
            assert "use_bayesian" in settings
            assert settings["use_bayesian"] is True

    def test_geometry_presets(self):
        """Geometry presets should return valid configs."""
        from hololoom.expansions.geometry import (
            hyperbolic_only,
            spherical_only,
            product_manifold,
            deep_hierarchy,
            tight_clusters,
        )

        presets = [
            hyperbolic_only(),
            spherical_only(),
            product_manifold(),
            deep_hierarchy(),
            tight_clusters(),
        ]

        for preset in presets:
            assert hasattr(preset, "get_settings")
            settings = preset.get_settings()
            assert isinstance(settings, dict)
            assert "use_riemannian" in settings
            assert settings["use_riemannian"] is True

    def test_spectral_presets(self):
        """Spectral presets should return valid configs."""
        from hololoom.expansions.advanced_spectral import (
            wavelets_only,
            diffusion_maps_only,
            full_spectral,
            fast_spectral,
        )

        presets = [
            wavelets_only(),
            diffusion_maps_only(),
            full_spectral(),
            fast_spectral(),
        ]

        for preset in presets:
            assert hasattr(preset, "get_settings")
            settings = preset.get_settings()
            assert isinstance(settings, dict)


class TestConfigValidation:
    """Test config validation."""

    def test_invalid_mode_raises(self):
        """Invalid mode should raise error."""
        with pytest.raises((ValueError, TypeError)):
            Config(mode="invalid_mode")

    def test_valid_modes_accepted(self):
        """All valid modes should be accepted."""
        for mode in ExecutionMode:
            config = Config(mode=mode)
            assert config.mode == mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

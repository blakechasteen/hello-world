"""
Tests for Dark Trace Plugin Marketplace System

Tests plugin signing, loader with hot reload, and marketplace functionality.

Author: HoloLoom Team
Created: December 2025
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import json
import sys

# Test plugin protocol components
from HoloLoom.dark_trace.plugins.plugin_protocol import (
    PluginCategory,
    PluginStatus,
    PluginMetadata,
    PluginRating,
    DarkTracePlugin,
    LensPlugin,
    AnalyzerPlugin,
    create_plugin_metadata,
)

from HoloLoom.dark_trace.plugins.plugin_loader import (
    LoadedPlugin,
    PluginLoader,
    create_loader,
)

from HoloLoom.dark_trace.plugins.plugin_marketplace import (
    SortOrder,
    MarketplacePlugin,
    PluginMarketplace,
    create_marketplace,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class MockLensPluginForMarketplace(LensPlugin):
    """Mock lens plugin for marketplace testing."""

    @property
    def metadata(self) -> PluginMetadata:
        return create_plugin_metadata(
            name="mock-marketplace-lens",
            version="1.0.0",
            author="Test Author",
            description="A mock lens plugin for marketplace testing",
            category=PluginCategory.LENS,
            tags=["test", "mock", "lens"],
        )

    def extract_features(self, activations, **kwargs):
        return {
            "mean": np.mean(activations, axis=-1),
            "std": np.std(activations, axis=-1),
        }

    def get_feature_names(self):
        return ["mean", "std"]


class MockAnalyzerPluginForMarketplace(AnalyzerPlugin):
    """Mock analyzer plugin for marketplace testing."""

    @property
    def metadata(self) -> PluginMetadata:
        return create_plugin_metadata(
            name="mock-marketplace-analyzer",
            version="2.0.0",
            author="Test Author",
            description="A mock analyzer plugin for marketplace testing",
            category=PluginCategory.ANALYZER,
            tags=["test", "analysis"],
        )

    def analyze(self, activations, feature_names=None, **kwargs):
        return {
            "total_features": activations.shape[-1] if len(activations.shape) > 1 else 1,
            "sample_count": activations.shape[0],
            "mean_activation": float(np.mean(activations)),
        }


@pytest.fixture
def mock_lens():
    return MockLensPluginForMarketplace()


@pytest.fixture
def mock_analyzer():
    return MockAnalyzerPluginForMarketplace()


@pytest.fixture
def temp_plugin_dir():
    """Create a temporary plugin directory with a test plugin."""
    temp_dir = Path(tempfile.mkdtemp())

    # Write a test plugin
    plugin_code = '''
"""Test plugin module for marketplace."""
import numpy as np
from HoloLoom.dark_trace.plugins.plugin_protocol import (
    LensPlugin,
    PluginMetadata,
    PluginCategory,
    create_plugin_metadata,
)

class TestFileLensPlugin(LensPlugin):
    """Test lens plugin loaded from file."""

    @property
    def metadata(self) -> PluginMetadata:
        return create_plugin_metadata(
            name="test-file-lens-marketplace",
            version="1.0.0",
            author="File Author",
            description="Test plugin from file for marketplace",
            category=PluginCategory.LENS,
        )

    def extract_features(self, activations, **kwargs):
        return {"sum": activations.sum(axis=-1)}
'''

    plugin_file = temp_dir / "marketplace_demo_plugin.py"
    with open(plugin_file, 'w') as f:
        f.write(plugin_code)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_marketplace_dir():
    """Create a temporary marketplace directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Protocol Tests
# =============================================================================

class TestMarketplacePluginProtocol:
    """Tests for marketplace plugin protocol and base classes."""

    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = create_plugin_metadata(
            name="test-plugin",
            version="1.0.0",
            author="Test",
            description="Test plugin",
            category=PluginCategory.LENS,
        )

        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.category == PluginCategory.LENS

    def test_plugin_metadata_serialization(self):
        """Test metadata serialization/deserialization."""
        metadata = create_plugin_metadata(
            name="test-plugin",
            version="1.0.0",
            author="Test",
            description="Test plugin",
            category=PluginCategory.LENS,
            tags=["test", "example"],
        )

        data = metadata.to_dict()
        restored = PluginMetadata.from_dict(data)

        assert restored.name == metadata.name
        assert restored.version == metadata.version
        assert restored.tags == metadata.tags

    def test_lens_plugin_extract_features(self, mock_lens):
        """Test lens plugin feature extraction."""
        activations = np.random.randn(10, 64)
        features = mock_lens.extract_features(activations)

        assert "mean" in features
        assert "std" in features
        assert features["mean"].shape == (10,)

    def test_analyzer_plugin_analyze(self, mock_analyzer):
        """Test analyzer plugin analysis."""
        activations = np.random.randn(100, 64)
        result = mock_analyzer.analyze(activations)

        assert result["total_features"] == 64
        assert result["sample_count"] == 100
        assert "mean_activation" in result

    def test_plugin_health_check(self, mock_lens):
        """Test plugin health check."""
        assert mock_lens.health_check() is True

    def test_plugin_lifecycle_hooks(self, mock_lens):
        """Test plugin lifecycle hooks don't error."""
        mock_lens.on_load()
        mock_lens.on_enable()
        mock_lens.on_disable()
        mock_lens.on_unload()


# =============================================================================
# Signing Tests (optional - skip if cryptography not installed)
# =============================================================================

class TestPluginSigningOptional:
    """Tests for plugin signing (requires cryptography library)."""

    def test_signing_import(self):
        """Test that signing module can be imported."""
        try:
            from HoloLoom.dark_trace.plugins.plugin_signing import (
                SigningAlgorithm,
                PluginSigner,
                PluginVerifier,
                create_signer,
                create_verifier,
            )
            # If we get here, imports work
            assert True
        except ImportError:
            pytest.skip("cryptography library not installed")


# =============================================================================
# Loader Tests
# =============================================================================

class TestPluginLoaderMarketplace:
    """Tests for plugin loading and hot reload in marketplace context."""

    def test_create_loader(self):
        """Test creating a plugin loader."""
        loader = create_loader()
        assert isinstance(loader, PluginLoader)

    def test_register_factory(self, mock_lens):
        """Test registering a plugin factory."""
        loader = create_loader()
        loader.register_factory("mock-lens", lambda: MockLensPluginForMarketplace())

        loaded = loader.load_from_factory("mock-lens")
        assert loaded is not None
        assert loaded.metadata.name == "mock-marketplace-lens"

    def test_load_from_file(self, temp_plugin_dir):
        """Test loading a plugin from file."""
        loader = create_loader()
        plugin_file = temp_plugin_dir / "marketplace_demo_plugin.py"

        loaded = loader.load_from_file(plugin_file, verify=False)

        assert loaded is not None
        assert loaded.metadata.name == "test-file-lens-marketplace"
        assert loaded.status == PluginStatus.ACTIVE

    def test_load_from_directory(self, temp_plugin_dir):
        """Test loading plugins from directory."""
        loader = create_loader()
        loaded = loader.load_from_directory(temp_plugin_dir)

        assert len(loaded) == 1
        assert loaded[0].metadata.name == "test-file-lens-marketplace"

    def test_unload_plugin(self, temp_plugin_dir):
        """Test unloading a plugin."""
        loader = create_loader()
        plugin_file = temp_plugin_dir / "marketplace_demo_plugin.py"

        loader.load_from_file(plugin_file, verify=False)
        assert loader.get_plugin("test-file-lens-marketplace") is not None

        loader.unload("test-file-lens-marketplace")
        assert loader.get_plugin("test-file-lens-marketplace") is None

    def test_reload_plugin(self, temp_plugin_dir):
        """Test reloading a plugin."""
        loader = create_loader()
        plugin_file = temp_plugin_dir / "marketplace_demo_plugin.py"

        loader.load_from_file(plugin_file, verify=False)
        reloaded = loader.reload("test-file-lens-marketplace")

        assert reloaded is not None
        assert reloaded.metadata.name == "test-file-lens-marketplace"

    def test_enable_disable_plugin(self, temp_plugin_dir):
        """Test enabling and disabling plugins."""
        loader = create_loader()
        plugin_file = temp_plugin_dir / "marketplace_demo_plugin.py"

        loader.load_from_file(plugin_file, verify=False)

        assert loader.disable("test-file-lens-marketplace")
        assert loader.get_status("test-file-lens-marketplace") == PluginStatus.DISABLED

        assert loader.enable("test-file-lens-marketplace")
        assert loader.get_status("test-file-lens-marketplace") == PluginStatus.ACTIVE

    def test_get_plugins_by_category(self, temp_plugin_dir):
        """Test getting plugins by category."""
        loader = create_loader()
        plugin_file = temp_plugin_dir / "marketplace_demo_plugin.py"

        loader.load_from_file(plugin_file, verify=False)
        plugins = loader.get_plugins_by_category(PluginCategory.LENS)

        assert len(plugins) == 1

    def test_loader_summary(self, temp_plugin_dir):
        """Test loader summary."""
        loader = create_loader()
        plugin_file = temp_plugin_dir / "marketplace_demo_plugin.py"

        loader.load_from_file(plugin_file, verify=False)
        summary = loader.get_summary()

        assert summary["total_loaded"] == 1
        assert "lens" in summary["by_category"]

    def test_health_check(self, temp_plugin_dir):
        """Test plugin health checks."""
        loader = create_loader()
        plugin_file = temp_plugin_dir / "marketplace_demo_plugin.py"

        loader.load_from_file(plugin_file, verify=False)
        health = loader.check_health()

        assert "test-file-lens-marketplace" in health
        assert health["test-file-lens-marketplace"] is True


# =============================================================================
# Marketplace Tests
# =============================================================================

class TestPluginMarketplaceCore:
    """Tests for plugin marketplace core functionality."""

    def test_create_marketplace(self, temp_marketplace_dir):
        """Test creating a marketplace."""
        marketplace = create_marketplace(
            storage_path=temp_marketplace_dir / "storage",
            cache_path=temp_marketplace_dir / "cache",
        )
        assert isinstance(marketplace, PluginMarketplace)

    def test_publish_plugin(self, temp_plugin_dir, temp_marketplace_dir):
        """Test publishing a plugin."""
        marketplace = create_marketplace(
            storage_path=temp_marketplace_dir / "storage",
            cache_path=temp_marketplace_dir / "cache",
        )

        metadata = create_plugin_metadata(
            name="test-publish-plugin",
            version="1.0.0",
            author="Test",
            description="Test plugin for publishing",
            category=PluginCategory.LENS,
        )

        mp_plugin = marketplace.publish(temp_plugin_dir, metadata)

        assert mp_plugin.metadata.name == "test-publish-plugin"
        assert mp_plugin.package_hash is not None

    def test_search_plugins(self, temp_plugin_dir, temp_marketplace_dir):
        """Test searching for plugins."""
        marketplace = create_marketplace(
            storage_path=temp_marketplace_dir / "storage",
            cache_path=temp_marketplace_dir / "cache",
        )

        # Publish a plugin
        metadata = create_plugin_metadata(
            name="searchable-marketplace-plugin",
            version="1.0.0",
            author="Test",
            description="A searchable test plugin for marketplace",
            category=PluginCategory.LENS,
            tags=["test", "search", "marketplace"],
        )
        marketplace.publish(temp_plugin_dir, metadata)

        # Search by query
        results = marketplace.search(query="searchable")
        assert len(results) == 1
        assert results[0].metadata.name == "searchable-marketplace-plugin"

        # Search by category
        results = marketplace.search(category=PluginCategory.LENS)
        assert len(results) == 1

        # Search by tags
        results = marketplace.search(tags=["marketplace"])
        assert len(results) == 1

    def test_rate_plugin(self, temp_plugin_dir, temp_marketplace_dir):
        """Test rating a plugin."""
        marketplace = create_marketplace(
            storage_path=temp_marketplace_dir / "storage",
            cache_path=temp_marketplace_dir / "cache",
        )

        metadata = create_plugin_metadata(
            name="rateable-marketplace-plugin",
            version="1.0.0",
            author="Test",
            description="Test plugin for rating",
            category=PluginCategory.LENS,
        )
        marketplace.publish(temp_plugin_dir, metadata)

        # Rate the plugin
        rating = marketplace.rate(
            name="rateable-marketplace-plugin",
            version="1.0.0",
            user_id="user1",
            rating=4.5,
            review="Great plugin!",
        )

        assert rating is not None
        assert rating.rating == 4.5

        # Check average rating
        plugin = marketplace.get_plugin("rateable-marketplace-plugin")
        assert plugin.average_rating == 4.5
        assert plugin.rating_count == 1

    def test_download_plugin(self, temp_plugin_dir, temp_marketplace_dir):
        """Test downloading a plugin."""
        marketplace = create_marketplace(
            storage_path=temp_marketplace_dir / "storage",
            cache_path=temp_marketplace_dir / "cache",
        )

        metadata = create_plugin_metadata(
            name="downloadable-marketplace-plugin",
            version="1.0.0",
            author="Test",
            description="Test plugin for downloading",
            category=PluginCategory.LENS,
        )
        marketplace.publish(temp_plugin_dir, metadata)

        # Download
        download_dir = temp_marketplace_dir / "downloads"
        path = marketplace.download(
            "downloadable-marketplace-plugin",
            dest_dir=download_dir,
        )

        assert path is not None
        assert path.exists()

        # Check download stats
        plugin = marketplace.get_plugin("downloadable-marketplace-plugin")
        assert plugin.download_stats.total_downloads == 1

    def test_featured_plugins(self, temp_plugin_dir, temp_marketplace_dir):
        """Test featured plugins."""
        marketplace = create_marketplace(
            storage_path=temp_marketplace_dir / "storage",
            cache_path=temp_marketplace_dir / "cache",
        )

        metadata = create_plugin_metadata(
            name="featured-marketplace-plugin",
            version="1.0.0",
            author="Test",
            description="Test plugin for featuring",
            category=PluginCategory.LENS,
        )
        marketplace.publish(temp_plugin_dir, metadata)

        # Set as featured
        marketplace.set_featured("featured-marketplace-plugin", "1.0.0", True)

        featured = marketplace.get_featured()
        assert len(featured) == 1
        assert featured[0].metadata.name == "featured-marketplace-plugin"

    def test_deprecate_plugin(self, temp_plugin_dir, temp_marketplace_dir):
        """Test deprecating a plugin."""
        marketplace = create_marketplace(
            storage_path=temp_marketplace_dir / "storage",
            cache_path=temp_marketplace_dir / "cache",
        )

        metadata = create_plugin_metadata(
            name="deprecated-marketplace-plugin",
            version="1.0.0",
            author="Test",
            description="Test plugin for deprecation",
            category=PluginCategory.LENS,
        )
        marketplace.publish(temp_plugin_dir, metadata)

        # Deprecate
        marketplace.deprecate(
            "deprecated-marketplace-plugin",
            "1.0.0",
            "Use new-plugin instead",
        )

        plugin = marketplace.get_plugin("deprecated-marketplace-plugin")
        assert plugin.deprecated
        assert plugin.deprecation_message == "Use new-plugin instead"

        # Deprecated plugins should not appear in search
        results = marketplace.search()
        assert len(results) == 0

    def test_marketplace_stats(self, temp_plugin_dir, temp_marketplace_dir):
        """Test marketplace statistics."""
        marketplace = create_marketplace(
            storage_path=temp_marketplace_dir / "storage",
            cache_path=temp_marketplace_dir / "cache",
        )

        metadata = create_plugin_metadata(
            name="stats-marketplace-plugin",
            version="1.0.0",
            author="Test",
            description="Test plugin for stats",
            category=PluginCategory.LENS,
        )
        marketplace.publish(temp_plugin_dir, metadata)
        marketplace.rate("stats-marketplace-plugin", "1.0.0", "user1", 5.0)
        marketplace.download("stats-marketplace-plugin")

        stats = marketplace.get_stats()

        assert stats["total_plugins"] == 1
        assert stats["total_downloads"] == 1
        assert stats["total_ratings"] == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestMarketplaceIntegration:
    """Integration tests for the full marketplace workflow."""

    def test_full_marketplace_workflow(self, temp_plugin_dir, temp_marketplace_dir):
        """Test complete marketplace workflow: publish, search, rate, download."""
        # Create marketplace and loader
        marketplace = create_marketplace(
            storage_path=temp_marketplace_dir / "storage",
            cache_path=temp_marketplace_dir / "cache",
        )
        loader = create_loader()

        # 1. Publish plugin
        metadata = create_plugin_metadata(
            name="workflow-marketplace-plugin",
            version="1.0.0",
            author="Test",
            description="Test plugin for full workflow",
            category=PluginCategory.LENS,
        )
        marketplace.publish(temp_plugin_dir, metadata)

        # 2. Search and find
        results = marketplace.search(query="workflow")
        assert len(results) == 1

        # 3. Rate
        marketplace.rate("workflow-marketplace-plugin", "1.0.0", "user1", 4.0)
        plugin = marketplace.get_plugin("workflow-marketplace-plugin")
        assert plugin.average_rating == 4.0

        # 4. Download
        path = marketplace.download("workflow-marketplace-plugin")
        assert path is not None

        # 5. Install (manual load for test)
        loaded_plugins = loader.load_from_directory(temp_plugin_dir)
        assert len(loaded_plugins) > 0

        # 6. Use the plugin
        plugin_instance = loaded_plugins[0].plugin
        activations = np.random.randn(10, 64)
        features = plugin_instance.extract_features(activations)
        assert "sum" in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

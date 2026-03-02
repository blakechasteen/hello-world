"""
Dark Trace Plugin System

Complete plugin infrastructure for extending Dark Trace interpretability
capabilities with custom lenses, validators, analyzers, and more.

Features:
- Plugin protocol with 8 plugin types
- Cryptographic signing (RSA/ECDSA) for verification
- Hot reload for seamless updates
- Community marketplace with ratings and reviews

Author: HoloLoom Team
Created: December 2025

Example Usage:
    # Creating a custom lens plugin
    from hololoom.dark_trace.plugins import (
        LensPlugin,
        PluginMetadata,
        PluginCategory,
        create_plugin_metadata,
    )

    class MyLensPlugin(LensPlugin):
        @property
        def metadata(self) -> PluginMetadata:
            return create_plugin_metadata(
                name="my-lens",
                version="1.0.0",
                author="My Name",
                description="Custom feature extraction lens",
                category=PluginCategory.LENS,
            )

        def extract_features(self, activations, **kwargs):
            # Custom feature extraction logic
            return {"my_feature": activations.mean(axis=-1)}

    # Loading plugins
    from hololoom.dark_trace.plugins import create_loader

    loader = create_loader(plugin_dirs=[Path("./plugins")])
    loader.discover_plugins()
    loader.enable_hot_reload()

    # Using the marketplace
    from hololoom.dark_trace.plugins import create_marketplace

    marketplace = create_marketplace()
    plugins = marketplace.search(query="lens", category=PluginCategory.LENS)
    marketplace.install("plugin-name", version="1.0.0", loader=loader)
"""

# Plugin protocol and types
from hololoom.dark_trace.plugins.plugin_protocol import (
    # Enums
    PluginCategory,
    PluginStatus,

    # Data classes
    PluginMetadata,
    PluginSignature,
    PluginRating,

    # Base classes
    DarkTracePlugin,
    LensPlugin,
    ValidatorPlugin,
    AnalyzerPlugin,
    VisualizerPlugin,
    SafetyPlugin,
    SteeringPlugin,
    DatasetPlugin,
    ModelAdapterPlugin,

    # Types
    PluginFactory,

    # Factory functions
    create_plugin_metadata,
)

# Signing and verification
from hololoom.dark_trace.plugins.plugin_signing import (
    # Enums
    SigningAlgorithm,

    # Data classes
    KeyPair,
    PluginManifest,
    SignedManifest,

    # Classes
    PluginSigner,
    PluginVerifier,

    # Factory functions
    create_signer,
    create_verifier,
    generate_key_pair,
)

# Plugin loader with hot reload
from hololoom.dark_trace.plugins.plugin_loader import (
    # Data classes
    LoadedPlugin,

    # Classes
    PluginLoader,

    # Factory functions
    create_loader,
    load_plugin,
)

# Marketplace
from hololoom.dark_trace.plugins.plugin_marketplace import (
    # Enums
    SortOrder,

    # Data classes
    PluginDownloadStats,
    MarketplacePlugin,

    # Classes
    PluginMarketplace,

    # Factory functions
    create_marketplace,
    get_default_marketplace,
)


__all__ = [
    # Protocol enums
    "PluginCategory",
    "PluginStatus",

    # Protocol data classes
    "PluginMetadata",
    "PluginSignature",
    "PluginRating",

    # Plugin base classes
    "DarkTracePlugin",
    "LensPlugin",
    "ValidatorPlugin",
    "AnalyzerPlugin",
    "VisualizerPlugin",
    "SafetyPlugin",
    "SteeringPlugin",
    "DatasetPlugin",
    "ModelAdapterPlugin",

    # Protocol types
    "PluginFactory",

    # Protocol factories
    "create_plugin_metadata",

    # Signing enums
    "SigningAlgorithm",

    # Signing data classes
    "KeyPair",
    "PluginManifest",
    "SignedManifest",

    # Signing classes
    "PluginSigner",
    "PluginVerifier",

    # Signing factories
    "create_signer",
    "create_verifier",
    "generate_key_pair",

    # Loader data classes
    "LoadedPlugin",

    # Loader classes
    "PluginLoader",

    # Loader factories
    "create_loader",
    "load_plugin",

    # Marketplace enums
    "SortOrder",

    # Marketplace data classes
    "PluginDownloadStats",
    "MarketplacePlugin",

    # Marketplace classes
    "PluginMarketplace",

    # Marketplace factories
    "create_marketplace",
    "get_default_marketplace",
]

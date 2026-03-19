"""
Hello World Plugin - Dark Trace Example
========================================

This is a minimal example plugin demonstrating the Dark Trace plugin architecture.
Use this as a template for creating your own plugins.

Trust Level: SANDBOXED (community plugin, limited capabilities)
Capabilities: READ_ACTIVATIONS

Date: December 2025
Phase: 11 - Plugin Ecosystem

How to use this example:
------------------------
1. Copy this file to your plugin directory
2. Modify the metadata and analysis logic
3. Register with the plugin registry
4. Test with the provided test template

Quick Start:
    from hololoom.dark_trace.plugins.examples.hello_plugin import HelloWorldPlugin

    plugin = HelloWorldPlugin()
    print(plugin.describe_behavior())
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

# TYPE_CHECKING prevents circular imports at runtime
if TYPE_CHECKING:
    from hololoom.dark_trace.engine import DarkTraceEngine
    from hololoom.dark_trace.plugins.safety_gate import PluginSafetyGate
    from hololoom.dark_trace.result import TraceResult

# Import from plugin interface
from hololoom.dark_trace.plugins.interface import (
    MonitorPlugin,  # We inherit from MonitorPlugin for this example
    PluginMetadata,
    PluginType,
)
from hololoom.dark_trace.plugins.safety_gate import (
    PluginCapability,
)

# Set up logging for your plugin
logger = logging.getLogger("hololoom.dark_trace.plugins.examples.hello_world")


# =============================================================================
# Data Structures
# =============================================================================
# Define data structures for your plugin's output

@dataclass
class HelloAnalysisResult:
    """
    Result of Hello World analysis.

    Define clear data structures for your plugin's output.
    This makes it easy for consumers to work with your plugin.
    """
    trace_id: str
    greeting: str
    features_found: int
    top_feature: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "trace_id": self.trace_id,
            "greeting": self.greeting,
            "features_found": self.features_found,
            "top_feature": self.top_feature,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Plugin Implementation
# =============================================================================

class HelloWorldPlugin(MonitorPlugin):
    """
    A simple "Hello World" plugin that demonstrates plugin basics.

    This plugin:
    - Counts features found in each analysis
    - Logs a greeting for each trace
    - Tracks basic statistics

    Use this as a template for your own plugins!

    Plugin Types Available:
    - MonitorPlugin: Observe analyses (read-only) - RECOMMENDED for beginners
    - LensPlugin: Extract features from activations
    - ValidatorPlugin: Validate other plugins
    - DomainPlugin: Domain-specific analysis
    - SteeringPlugin: Modify model behavior (requires TRUSTED+)
    - IntegrationPlugin: External integrations
    """

    def __init__(self, greeting_prefix: str = "Hello") -> None:
        """
        Initialize the plugin.

        Args:
            greeting_prefix: Prefix for greetings (customizable)

        TIP: Keep __init__ lightweight. Do heavy setup in initialize().
        """
        self._greeting_prefix = greeting_prefix

        # Engine and safety gate are set during initialize()
        self._engine: DarkTraceEngine | None = None
        self._safety_gate: PluginSafetyGate | None = None

        # Track state
        self._initialized = False

        # Plugin-specific state
        self._analysis_count = 0
        self._total_features_seen = 0
        self._results: list[HelloAnalysisResult] = []

    # -------------------------------------------------------------------------
    # Required: Plugin Metadata
    # -------------------------------------------------------------------------

    @property
    def metadata(self) -> PluginMetadata:
        """
        Return plugin metadata.

        This is REQUIRED. The metadata describes your plugin to:
        - The plugin registry (for discovery)
        - The safety gate (for capability checking)
        - Users (for understanding what the plugin does)
        """
        return PluginMetadata(
            # Unique identifier (lowercase, underscores)
            name="hello_world",

            # Semantic version (major.minor.patch)
            version="1.0.0",

            # Your name or organization
            author="HoloLoom Examples",

            # Clear, concise description
            description=(
                "A simple example plugin that demonstrates the Dark Trace "
                "plugin architecture. Counts features and logs greetings."
            ),

            # Plugin type (see PluginType enum)
            plugin_type=PluginType.MONITOR,

            # Other plugins this depends on (empty for standalone)
            dependencies=[],

            # Capabilities needed (keep minimal!)
            # See PluginCapability enum for options
            requested_capabilities=[
                PluginCapability.READ_ACTIVATIONS,
            ],

            # Signature for verification (None for unsigned community plugins)
            signature=None,

            # Minimum Dark Trace version required
            min_dark_trace_version="1.0.0",

            # Tags for discovery
            tags=["example", "tutorial", "hello-world"],
        )

    # -------------------------------------------------------------------------
    # Required: Lifecycle Methods
    # -------------------------------------------------------------------------

    async def initialize(
        self,
        engine: "DarkTraceEngine",
        safety_gate: "PluginSafetyGate",
    ) -> None:
        """
        Initialize the plugin.

        Called by the plugin loader after registration.
        Use this for:
        - Heavy initialization (loading models, connecting to services)
        - Registering hooks
        - Validating configuration

        Args:
            engine: The Dark Trace engine instance
            safety_gate: The safety gate for capability checking
        """
        self._engine = engine
        self._safety_gate = safety_gate
        self._initialized = True

        logger.info(f"HelloWorldPlugin initialized with prefix: {self._greeting_prefix}")

    async def shutdown(self) -> None:
        """
        Clean shutdown.

        Called when the plugin is being unloaded.
        Use this for:
        - Closing connections
        - Flushing caches
        - Cleaning up resources
        """
        self._initialized = False
        logger.info(
            f"HelloWorldPlugin shutting down. "
            f"Processed {self._analysis_count} analyses."
        )

    # -------------------------------------------------------------------------
    # Required: Behavior Description (Safety Contract)
    # -------------------------------------------------------------------------

    def describe_behavior(self) -> str:
        """
        Describe what this plugin does.

        This is a SAFETY CONTRACT. Be honest and complete!
        The AlignmentValidator will check this.

        Include:
        - What data the plugin reads
        - What actions it takes
        - What it does NOT do
        """
        return (
            f"This plugin observes Dark Trace analysis results and counts "
            f"the number of active features found. It logs a greeting "
            f"(prefix: '{self._greeting_prefix}') for each analysis. "
            f"The plugin only reads activation data and does not modify "
            f"any model behavior or write any features. It stores results "
            f"in memory for later retrieval but does not persist them."
        )

    # -------------------------------------------------------------------------
    # MonitorPlugin Interface
    # -------------------------------------------------------------------------

    async def on_analysis_complete(
        self,
        trace_result: "TraceResult",
    ) -> None:
        """
        Called after each analysis completes.

        This is the main hook for MonitorPlugin.
        Process the trace result here.

        Args:
            trace_result: The completed analysis result
        """
        if not self._initialized:
            return

        # Count features across all lens results
        feature_count = 0
        top_feature = None
        top_activation = 0.0

        for lens_result in getattr(trace_result, 'lens_results', []):
            features = getattr(lens_result, 'active_features', [])
            feature_count += len(features)

            # Find the top feature by activation
            for feature in features:
                activation = getattr(feature, 'activation', 0.0)
                if activation > top_activation:
                    top_activation = activation
                    top_feature = getattr(feature, 'name', 'unknown')

        # Create result
        trace_id = getattr(trace_result, 'id', f'trace_{self._analysis_count}')
        greeting = f"{self._greeting_prefix}, trace {trace_id}! Found {feature_count} features."

        result = HelloAnalysisResult(
            trace_id=trace_id,
            greeting=greeting,
            features_found=feature_count,
            top_feature=top_feature,
            metadata={
                "top_activation": top_activation,
            },
        )

        # Store and update stats
        self._results.append(result)
        self._analysis_count += 1
        self._total_features_seen += feature_count

        # Log the greeting
        logger.info(greeting)

    # -------------------------------------------------------------------------
    # Custom Methods
    # -------------------------------------------------------------------------
    # Add your own methods for plugin-specific functionality

    def get_statistics(self) -> dict[str, Any]:
        """
        Get plugin statistics.

        Provide statistics methods for monitoring and debugging.
        """
        return {
            "analysis_count": self._analysis_count,
            "total_features_seen": self._total_features_seen,
            "average_features_per_analysis": (
                self._total_features_seen / self._analysis_count
                if self._analysis_count > 0 else 0
            ),
            "results_stored": len(self._results),
            "greeting_prefix": self._greeting_prefix,
        }

    def get_recent_results(self, limit: int = 10) -> list[HelloAnalysisResult]:
        """
        Get recent analysis results.

        Returns:
            List of recent HelloAnalysisResult objects
        """
        return self._results[-limit:]

    def reset_statistics(self) -> None:
        """
        Reset all statistics.

        Useful for testing or periodic resets.
        """
        self._analysis_count = 0
        self._total_features_seen = 0
        self._results.clear()
        logger.info("HelloWorldPlugin statistics reset")


# =============================================================================
# Testing Template
# =============================================================================
# Copy this to create tests for your plugin

def create_test_template():
    """
    Template for testing your plugin.

    Save this as test_hello_plugin.py in your tests directory.
    """
    template = '''
import pytest
from unittest.mock import MagicMock, AsyncMock

from hololoom.dark_trace.plugins.examples.hello_plugin import (
    HelloWorldPlugin,
    HelloAnalysisResult,
)


class TestHelloWorldPlugin:
    """Tests for HelloWorldPlugin."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin instance."""
        return HelloWorldPlugin(greeting_prefix="Test")

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine."""
        return MagicMock()

    @pytest.fixture
    def mock_safety_gate(self):
        """Create a mock safety gate."""
        return MagicMock()

    # Test metadata
    def test_metadata_name(self, plugin):
        """Plugin has correct name."""
        assert plugin.metadata.name == "hello_world"

    def test_metadata_capabilities(self, plugin):
        """Plugin requests minimal capabilities."""
        caps = plugin.metadata.requested_capabilities
        assert len(caps) == 1
        # Only READ_ACTIVATIONS

    # Test lifecycle
    @pytest.mark.asyncio
    async def test_initialize(self, plugin, mock_engine, mock_safety_gate):
        """Plugin initializes correctly."""
        await plugin.initialize(mock_engine, mock_safety_gate)
        assert plugin._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown(self, plugin, mock_engine, mock_safety_gate):
        """Plugin shuts down correctly."""
        await plugin.initialize(mock_engine, mock_safety_gate)
        await plugin.shutdown()
        assert plugin._initialized is False

    # Test behavior description
    def test_describe_behavior(self, plugin):
        """Behavior description is complete."""
        description = plugin.describe_behavior()
        assert len(description) > 50  # Not too short
        assert "read" in description.lower()  # Mentions reading
        assert "not modify" in description.lower()  # Safety statement

    # Test analysis
    @pytest.mark.asyncio
    async def test_on_analysis_complete(self, plugin, mock_engine, mock_safety_gate):
        """Plugin processes analysis correctly."""
        await plugin.initialize(mock_engine, mock_safety_gate)

        # Create mock trace result
        mock_feature = MagicMock()
        mock_feature.name = "test_feature"
        mock_feature.activation = 0.85

        mock_lens = MagicMock()
        mock_lens.active_features = [mock_feature]

        mock_trace = MagicMock()
        mock_trace.id = "test_trace_1"
        mock_trace.lens_results = [mock_lens]

        # Process
        await plugin.on_analysis_complete(mock_trace)

        # Verify
        stats = plugin.get_statistics()
        assert stats["analysis_count"] == 1
        assert stats["total_features_seen"] == 1

        results = plugin.get_recent_results(limit=1)
        assert len(results) == 1
        assert results[0].features_found == 1
        assert results[0].top_feature == "test_feature"

    # Test statistics
    def test_reset_statistics(self, plugin):
        """Statistics reset correctly."""
        plugin._analysis_count = 10
        plugin._total_features_seen = 50
        plugin.reset_statistics()
        assert plugin._analysis_count == 0
        assert plugin._total_features_seen == 0
'''
    return template


# =============================================================================
# Public Exports
# =============================================================================

__all__ = [
    "HelloWorldPlugin",
    "HelloAnalysisResult",
    "create_test_template",
]


# =============================================================================
# Example Usage (when run directly)
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demonstrate the Hello World plugin."""
        print("=" * 60)
        print("Hello World Plugin Demo")
        print("=" * 60)

        # Create plugin
        plugin = HelloWorldPlugin(greeting_prefix="Greetings")

        print("\n1. Plugin Metadata:")
        print(f"   Name: {plugin.metadata.name}")
        print(f"   Version: {plugin.metadata.version}")
        print(f"   Type: {plugin.metadata.plugin_type.value}")
        print(f"   Capabilities: {[c.value for c in plugin.metadata.requested_capabilities]}")

        print("\n2. Behavior Description:")
        print(f"   {plugin.describe_behavior()}")

        # Simulate initialization
        print("\n3. Simulating initialization...")
        from unittest.mock import MagicMock
        await plugin.initialize(MagicMock(), MagicMock())
        print("   Initialized successfully!")

        # Simulate an analysis
        print("\n4. Simulating analysis...")
        mock_feature = MagicMock()
        mock_feature.name = "semantic.warmth"
        mock_feature.activation = 0.75

        mock_lens = MagicMock()
        mock_lens.active_features = [mock_feature, MagicMock(), MagicMock()]

        mock_trace = MagicMock()
        mock_trace.id = "demo_trace"
        mock_trace.lens_results = [mock_lens]

        await plugin.on_analysis_complete(mock_trace)

        print("\n5. Statistics:")
        stats = plugin.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\n6. Recent Results:")
        for result in plugin.get_recent_results():
            print(f"   {result.greeting}")

        # Cleanup
        await plugin.shutdown()
        print("\n7. Plugin shut down successfully!")

        print("\n" + "=" * 60)
        print("Demo complete! Use this as a template for your own plugins.")
        print("=" * 60)

    asyncio.run(demo())

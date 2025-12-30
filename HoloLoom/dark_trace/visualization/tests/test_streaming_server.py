"""
Tests for Dark Trace Visualization Dashboard

Tests for streaming server, dashboard API, and orchestrator hook.

Author: HoloLoom Team
Created: December 2025
"""

import asyncio
import json
import pytest
from dataclasses import asdict
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# StreamingServer Tests
# =============================================================================

class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from HoloLoom.dark_trace.visualization.streaming_server import StreamingConfig

        config = StreamingConfig()

        assert config.host == "localhost"
        assert config.port == 8765
        assert config.max_clients == 50
        assert config.heartbeat_interval == 30.0
        assert config.buffer_size == 100  # actual attribute name
        assert config.enable_compression is True

    def test_custom_config(self):
        """Test custom configuration."""
        from HoloLoom.dark_trace.visualization.streaming_server import StreamingConfig

        config = StreamingConfig(
            host="0.0.0.0",
            port=9000,
            max_clients=100,
            heartbeat_interval=15.0,
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.max_clients == 100
        assert config.heartbeat_interval == 15.0


class TestStreamMessage:
    """Tests for StreamMessage."""

    def test_message_creation(self):
        """Test message creation."""
        import time
        from HoloLoom.dark_trace.visualization.streaming_server import (
            StreamMessage,
            MessageType,
        )

        timestamp = time.time()
        msg = StreamMessage(
            type=MessageType.ACTIVATIONS,
            timestamp=timestamp,
            data={"layer": 0, "values": [0.5, 0.3]},
        )

        assert msg.type == MessageType.ACTIVATIONS
        assert msg.data["layer"] == 0
        assert msg.timestamp == timestamp

    def test_message_to_json_statistics(self):
        """Test message serialization via to_json."""
        import time
        from HoloLoom.dark_trace.visualization.streaming_server import (
            StreamMessage,
            MessageType,
        )

        timestamp = time.time()
        msg = StreamMessage(
            type=MessageType.STATISTICS,
            timestamp=timestamp,
            data={"connected": True},
        )

        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "statistics"
        assert parsed["data"]["connected"] is True
        assert "timestamp" in parsed

    def test_message_to_json(self):
        """Test JSON serialization."""
        import time
        from HoloLoom.dark_trace.visualization.streaming_server import (
            StreamMessage,
            MessageType,
        )

        timestamp = time.time()
        msg = StreamMessage(
            type=MessageType.HEARTBEAT,
            timestamp=timestamp,
            data={},
        )

        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "heartbeat"


class TestActivationStreamHandler:
    """Tests for ActivationStreamHandler."""

    def test_snapshot_creation(self):
        """Test creating activation snapshot directly."""
        import time
        from HoloLoom.dark_trace.visualization.streaming_server import (
            ActivationSnapshot,
        )

        timestamp = time.time()
        snapshot = ActivationSnapshot(
            layer=1,
            feature_indices=[0, 5, 10],
            activation_values=[0.9, 0.7, 0.5],
            timestamp=timestamp,
            metadata={"source": "test"},
        )

        assert snapshot.layer == 1
        assert snapshot.feature_indices == [0, 5, 10]
        assert snapshot.activation_values == [0.9, 0.7, 0.5]
        assert snapshot.metadata["source"] == "test"
        assert snapshot.timestamp == timestamp

    def test_format_for_broadcast(self):
        """Test snapshot to_dict formatting."""
        import time
        from HoloLoom.dark_trace.visualization.streaming_server import (
            ActivationSnapshot,
        )

        snapshot = ActivationSnapshot(
            layer=0,
            feature_indices=[1, 2, 3],
            activation_values=[0.8, 0.6, 0.4],
            timestamp=time.time(),
            metadata={},
        )

        formatted = snapshot.to_dict()

        assert formatted["layer"] == 0
        assert len(formatted["features"]) == 3
        assert formatted["features"][0]["index"] == 1
        assert formatted["features"][0]["activation"] == 0.8

    def test_filter_by_threshold(self):
        """Test filtering by activation threshold via format_activations."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            ActivationStreamHandler,
            StreamingConfig,
            ClientSubscription,
            SubscriptionType,
        )

        config = StreamingConfig()
        handler = ActivationStreamHandler(config)
        sub = ClientSubscription(
            client_id="test",
            subscription_type=SubscriptionType.THRESHOLD,
            filters={"threshold": 0.4},
        )

        msg = handler.format_activations(
            layer=0,
            feature_indices=[0, 1, 2, 3, 4],
            activation_values=[0.1, 0.5, 0.8, 0.3, 0.9],
            subscription=sub,
        )

        # Should only include values >= 0.4 (indices 1, 2, 4)
        assert msg is not None
        features = msg.data["features"]
        indices = [f["index"] for f in features]
        assert 0 not in indices  # 0.1 < 0.4
        assert 3 not in indices  # 0.3 < 0.4
        assert 1 in indices      # 0.5 >= 0.4
        assert 2 in indices      # 0.8 >= 0.4
        assert 4 in indices      # 0.9 >= 0.4

    def test_filter_top_k(self):
        """Test filtering top-k features via format_activations."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            ActivationStreamHandler,
            StreamingConfig,
            ClientSubscription,
            SubscriptionType,
        )

        config = StreamingConfig()
        handler = ActivationStreamHandler(config)
        sub = ClientSubscription(
            client_id="test",
            subscription_type=SubscriptionType.TOP_K,
            filters={"k": 2},
        )

        msg = handler.format_activations(
            layer=0,
            feature_indices=[0, 1, 2, 3, 4],
            activation_values=[0.1, 0.5, 0.8, 0.3, 0.9],
            subscription=sub,
        )

        assert msg is not None
        features = msg.data["features"]
        assert len(features) == 2
        indices = [f["index"] for f in features]
        assert 4 in indices  # 0.9 is highest
        assert 2 in indices  # 0.8 is second highest

    def test_filter_by_features(self):
        """Test filtering by specific features via format_activations."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            ActivationStreamHandler,
            StreamingConfig,
            ClientSubscription,
            SubscriptionType,
        )

        config = StreamingConfig()
        handler = ActivationStreamHandler(config)
        sub = ClientSubscription(
            client_id="test",
            subscription_type=SubscriptionType.FEATURE,
            filters={"features": [1, 3]},
        )

        msg = handler.format_activations(
            layer=0,
            feature_indices=[0, 1, 2, 3, 4],
            activation_values=[0.1, 0.5, 0.8, 0.3, 0.9],
            subscription=sub,
        )

        assert msg is not None
        features = msg.data["features"]
        assert len(features) == 2
        indices = [f["index"] for f in features]
        assert indices == [1, 3]


class TestClientSubscription:
    """Tests for ClientSubscription."""

    def test_subscription_creation(self):
        """Test subscription creation with required fields."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            ClientSubscription,
            SubscriptionType,
        )

        sub = ClientSubscription(
            client_id="test_client",
            subscription_type=SubscriptionType.ALL,
        )

        assert sub.client_id == "test_client"
        assert sub.subscription_type == SubscriptionType.ALL
        assert sub.filters == {}
        assert sub.message_count == 0

    def test_subscription_with_filters(self):
        """Test subscription with custom filters."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            ClientSubscription,
            SubscriptionType,
        )

        sub = ClientSubscription(
            client_id="test_client",
            subscription_type=SubscriptionType.TOP_K,
            filters={"k": 10},
        )

        assert sub.subscription_type == SubscriptionType.TOP_K
        assert sub.filters["k"] == 10

    def test_subscription_layer_filter(self):
        """Test LAYER subscription filter storage."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            ClientSubscription,
            SubscriptionType,
        )

        sub = ClientSubscription(
            client_id="test_client",
            subscription_type=SubscriptionType.LAYER,
            filters={"layer": 0},
        )

        assert sub.subscription_type == SubscriptionType.LAYER
        assert sub.filters.get("layer") == 0


class TestDarkTraceStreamingServer:
    """Tests for DarkTraceStreamingServer."""

    def test_server_creation(self):
        """Test server instance creation."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            DarkTraceStreamingServer,
            StreamingConfig,
        )

        config = StreamingConfig(port=8800)
        server = DarkTraceStreamingServer(config)

        assert server.config.port == 8800
        assert len(server._clients) == 0
        assert server._server is None

    def test_client_count(self):
        """Test client counting."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            DarkTraceStreamingServer,
            ClientSubscription,
            SubscriptionType,
        )

        server = DarkTraceStreamingServer()

        assert server.client_count == 0

        # Simulate adding clients (using string IDs as keys)
        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        server._clients["client_1"] = mock_ws1
        server._clients["client_2"] = mock_ws2

        assert server.client_count == 2

    def test_handler_format_activations_top_k(self):
        """Test handler formats activations with top-k filtering."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            DarkTraceStreamingServer,
            ClientSubscription,
            SubscriptionType,
        )

        server = DarkTraceStreamingServer()

        # Test top-k filter via handler
        sub = ClientSubscription(
            client_id="test",
            subscription_type=SubscriptionType.TOP_K,
            filters={"k": 2},
        )

        msg = server.handler.format_activations(
            layer=0,
            feature_indices=[0, 1, 2, 3, 4],
            activation_values=[0.1, 0.5, 0.8, 0.3, 0.9],
            subscription=sub,
        )

        assert msg is not None
        features = msg.data["features"]
        assert len(features) == 2
        # Top 2 are indices 4 (0.9) and 2 (0.8)
        indices = {f["index"] for f in features}
        assert 4 in indices  # 0.9
        assert 2 in indices  # 0.8

    def test_handler_format_activations_threshold(self):
        """Test handler formats activations with threshold filtering."""
        from HoloLoom.dark_trace.visualization.streaming_server import (
            DarkTraceStreamingServer,
            ClientSubscription,
            SubscriptionType,
        )

        server = DarkTraceStreamingServer()

        sub = ClientSubscription(
            client_id="test",
            subscription_type=SubscriptionType.THRESHOLD,
            filters={"threshold": 0.5},
        )

        msg = server.handler.format_activations(
            layer=0,
            feature_indices=[0, 1, 2, 3, 4],
            activation_values=[0.1, 0.5, 0.8, 0.3, 0.9],
            subscription=sub,
        )

        assert msg is not None
        features = msg.data["features"]
        # Values >= 0.5: indices 1, 2, 4
        indices = {f["index"] for f in features}
        assert 1 in indices  # 0.5
        assert 2 in indices  # 0.8
        assert 4 in indices  # 0.9
        assert 0 not in indices  # 0.1 < 0.5
        assert 3 not in indices  # 0.3 < 0.5


# =============================================================================
# DashboardAPI Tests
# =============================================================================

class TestDashboardAPI:
    """Tests for DarkTraceDashboardAPI."""

    def test_api_creation(self):
        """Test API instance creation."""
        from HoloLoom.dark_trace.visualization.dashboard_api import (
            DarkTraceDashboardAPI,
            DashboardAPIConfig,
        )

        config = DashboardAPIConfig(enable_steering=True)
        api = DarkTraceDashboardAPI(config=config)

        assert api.config.enable_steering is True
        assert api.router is not None

    def test_record_activations(self):
        """Test recording activations."""
        from HoloLoom.dark_trace.visualization.dashboard_api import (
            DarkTraceDashboardAPI,
        )

        api = DarkTraceDashboardAPI()

        # Record some activations
        asyncio.get_event_loop().run_until_complete(
            api.record_activations_batch(
                layer=0,
                feature_indices=[0, 1, 2],
                activations=[0.5, 0.6, 0.7],
            )
        )

        # Check history - one entry per feature (not one per batch)
        history = api._activation_history
        assert len(history) == 3
        assert history[0]["layer"] == 0
        assert history[0]["feature_index"] == 0
        assert history[0]["activation"] == 0.5

    def test_get_statistics(self):
        """Test getting statistics."""
        from HoloLoom.dark_trace.visualization.dashboard_api import (
            DarkTraceDashboardAPI,
        )

        api = DarkTraceDashboardAPI()

        # Record some activations
        asyncio.get_event_loop().run_until_complete(
            api.record_activations_batch(
                layer=0,
                feature_indices=[0, 1, 2],
                activations=[0.5, 0.6, 0.7],
            )
        )

        stats = api._get_statistics()

        # _get_statistics returns dict with history_size, ws_clients, timestamp
        assert "history_size" in stats
        assert stats["history_size"] == 3  # 3 activation entries
        assert "ws_clients" in stats
        assert stats["ws_clients"] == 0  # No WebSocket clients connected


# =============================================================================
# OrchestratorHook Tests
# =============================================================================

class TestHookConfig:
    """Tests for HookConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import HookConfig

        config = HookConfig()

        assert config.enable_streaming is True
        assert config.enable_recording is True
        assert config.broadcast_throttle_ms == 50.0
        assert config.activation_threshold == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import HookConfig

        config = HookConfig(
            enable_streaming=False,
            broadcast_throttle_ms=100.0,
            monitored_layers=[0, 1],
        )

        assert config.enable_streaming is False
        assert config.broadcast_throttle_ms == 100.0
        assert config.monitored_layers == [0, 1]


class TestDarkTraceOrchestratorHook:
    """Tests for DarkTraceOrchestratorHook."""

    def test_hook_creation(self):
        """Test hook instance creation."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            DarkTraceOrchestratorHook,
            HookConfig,
        )

        config = HookConfig(enable_streaming=False)
        hook = DarkTraceOrchestratorHook(config=config)

        assert hook.config.enable_streaming is False
        assert hook.state.total_broadcasts == 0

    @pytest.mark.asyncio
    async def test_weave_lifecycle(self):
        """Test weave start/end lifecycle."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            DarkTraceOrchestratorHook,
            HookConfig,
        )

        config = HookConfig(enable_streaming=False)
        hook = DarkTraceOrchestratorHook(config=config)

        # Start weave
        await hook.on_weave_start("test query")

        assert hook.state.current_query == "test query"

        # End weave
        await hook.on_weave_end("test query", result=None, success=True)

        assert hook.state.current_query is None

    @pytest.mark.asyncio
    async def test_step_lifecycle(self):
        """Test step start/end lifecycle."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            DarkTraceOrchestratorHook,
            HookConfig,
        )
        import time

        config = HookConfig(enable_streaming=False)
        hook = DarkTraceOrchestratorHook(config=config)

        # Start step
        await hook.on_step_start("retrieval")

        assert hook.state.current_step == "retrieval"

        # Simulate some work
        await asyncio.sleep(0.01)

        # End step
        await hook.on_step_end("retrieval")

        assert "retrieval" in hook.state.step_timings
        assert hook.state.step_timings["retrieval"] > 0

    @pytest.mark.asyncio
    async def test_activation_filtering_by_layer(self):
        """Test activation filtering by monitored layers."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            DarkTraceOrchestratorHook,
            HookConfig,
        )

        config = HookConfig(
            enable_streaming=False,
            enable_recording=True,
            monitored_layers=[0, 2],  # Only monitor layers 0 and 2
        )
        hook = DarkTraceOrchestratorHook(config=config)

        # Send activations for layer 0 (monitored)
        await hook.on_activations(
            layer=0,
            feature_indices=[0, 1],
            activation_values=[0.5, 0.6],
        )

        # Send activations for layer 1 (not monitored)
        await hook.on_activations(
            layer=1,
            feature_indices=[0, 1],
            activation_values=[0.5, 0.6],
        )

        # Only layer 0 activations should be recorded
        assert hook.state.total_activations_recorded == 2

    @pytest.mark.asyncio
    async def test_activation_threshold(self):
        """Test activation threshold filtering."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            DarkTraceOrchestratorHook,
            HookConfig,
        )

        config = HookConfig(
            enable_streaming=False,
            enable_recording=True,
            activation_threshold=0.5,  # Only process if max > 0.5
        )
        hook = DarkTraceOrchestratorHook(config=config)

        # Send activations below threshold
        await hook.on_activations(
            layer=0,
            feature_indices=[0, 1],
            activation_values=[0.2, 0.3],  # max=0.3 < 0.5
        )

        assert hook.state.total_activations_recorded == 0

        # Send activations above threshold
        await hook.on_activations(
            layer=0,
            feature_indices=[0, 1],
            activation_values=[0.2, 0.6],  # max=0.6 > 0.5
        )

        assert hook.state.total_activations_recorded == 2

    @pytest.mark.asyncio
    async def test_callback_registration(self):
        """Test callback registration and invocation."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            DarkTraceOrchestratorHook,
            HookConfig,
        )

        config = HookConfig(enable_streaming=False)
        hook = DarkTraceOrchestratorHook(config=config)

        # Track callback invocations
        pre_weave_called = []
        post_weave_called = []
        activation_called = []

        def pre_weave_callback(query, context):
            pre_weave_called.append(query)

        def post_weave_callback(query, result, success):
            post_weave_called.append((query, success))

        async def activation_callback(layer, indices, values, metadata):
            activation_called.append(layer)

        # Register callbacks
        hook.register_pre_weave_callback(pre_weave_callback)
        hook.register_post_weave_callback(post_weave_callback)
        hook.register_activation_callback(activation_callback)

        # Trigger callbacks
        await hook.on_weave_start("test query")
        await hook.on_activations(0, [0], [0.5])
        await hook.on_weave_end("test query", result=None, success=True)

        assert pre_weave_called == ["test query"]
        assert post_weave_called == [("test query", True)]
        assert activation_called == [0]

    def test_get_statistics(self):
        """Test getting hook statistics."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            DarkTraceOrchestratorHook,
            HookConfig,
        )

        config = HookConfig(
            enable_streaming=True,
            activation_threshold=0.05,
        )
        hook = DarkTraceOrchestratorHook(config=config)

        stats = hook.get_statistics()

        assert stats["total_broadcasts"] == 0
        assert stats["total_activations_recorded"] == 0
        assert stats["config"]["streaming_enabled"] is True
        assert stats["config"]["activation_threshold"] == 0.05


class TestHookFactoryFunctions:
    """Tests for hook factory functions."""

    def test_create_orchestrator_hook(self):
        """Test creating hook with factory function."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            create_orchestrator_hook,
        )

        hook = create_orchestrator_hook(
            enable_streaming=False,
            enable_recording=True,
            broadcast_throttle_ms=100.0,
        )

        assert hook.config.enable_streaming is False
        assert hook.config.enable_recording is True
        assert hook.config.broadcast_throttle_ms == 100.0

    def test_create_hook_config_presets(self):
        """Test creating hook config from presets."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            create_hook_config,
        )

        # Minimal preset
        minimal = create_hook_config("minimal")
        assert minimal.broadcast_throttle_ms == 200.0
        assert minimal.include_metadata is False

        # Standard preset
        standard = create_hook_config("standard")
        assert standard.broadcast_throttle_ms == 50.0
        assert standard.include_metadata is True

        # Full preset
        full = create_hook_config("full")
        assert full.broadcast_throttle_ms == 10.0
        assert full.activation_threshold == 0.0

    def test_create_hook_config_with_overrides(self):
        """Test creating hook config with overrides."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            create_hook_config,
        )

        config = create_hook_config(
            "standard",
            broadcast_throttle_ms=75.0,
            monitored_layers=[0, 1, 2],
        )

        assert config.broadcast_throttle_ms == 75.0
        assert config.monitored_layers == [0, 1, 2]
        # Other standard values preserved
        assert config.include_metadata is True

    def test_invalid_preset(self):
        """Test error for invalid preset."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            create_hook_config,
        )

        with pytest.raises(ValueError, match="Unknown preset"):
            create_hook_config("invalid_preset")


# =============================================================================
# Integration Tests
# =============================================================================

class TestVisualizationIntegration:
    """Integration tests for the visualization system."""

    @pytest.mark.asyncio
    async def test_hook_with_streaming_server(self):
        """Test hook integration with streaming server."""
        from HoloLoom.dark_trace.visualization.orchestrator_hook import (
            DarkTraceOrchestratorHook,
            HookConfig,
        )
        from HoloLoom.dark_trace.visualization.streaming_server import (
            DarkTraceStreamingServer,
            StreamingConfig,
        )

        # Create streaming server (mock broadcast)
        server = DarkTraceStreamingServer()
        server.broadcast_activations = AsyncMock()
        server.broadcast_message = AsyncMock()

        # Create hook with streaming enabled
        config = HookConfig(
            enable_streaming=True,
            broadcast_throttle_ms=0,  # No throttling for test
        )
        hook = DarkTraceOrchestratorHook(
            streaming_server=server,
            config=config,
        )

        # Simulate weave cycle
        await hook.on_weave_start("test query")
        await hook.on_step_start("retrieval")
        await hook.on_activations(0, [0, 1], [0.5, 0.6])
        await hook.on_step_end("retrieval")
        await hook.on_weave_end("test query", result=None, success=True)

        # Verify broadcasts
        assert server.broadcast_message.call_count >= 2  # start, end
        assert server.broadcast_activations.call_count == 1

    @pytest.mark.asyncio
    async def test_api_with_streaming_server(self):
        """Test API integration with streaming server."""
        from HoloLoom.dark_trace.visualization.dashboard_api import (
            DarkTraceDashboardAPI,
        )
        from HoloLoom.dark_trace.visualization.streaming_server import (
            DarkTraceStreamingServer,
        )

        # Create streaming server
        server = DarkTraceStreamingServer()

        # Create API with streaming server
        api = DarkTraceDashboardAPI(streaming_server=server)

        # Mock the internal _broadcast_ws method (API handles its own WS broadcasts)
        api._broadcast_ws = AsyncMock()

        # Record activations through API
        await api.record_activations_batch(
            layer=0,
            feature_indices=[0, 1, 2],
            activations=[0.5, 0.6, 0.7],
        )

        # Verify activation was recorded to history
        assert len(api._activation_history) == 3

        # Verify _broadcast_ws was called (API broadcasts to its own WS clients)
        api._broadcast_ws.assert_called_once()

        # Verify streaming server is properly linked in status
        stats = api._get_statistics()
        assert "streaming" in stats
        assert stats["streaming"]["running"] == server.is_running

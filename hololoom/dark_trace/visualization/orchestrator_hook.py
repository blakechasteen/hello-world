"""
Dark Trace Orchestrator Hook

Integrates visualization dashboard with the weaving orchestrator,
enabling real-time streaming of activations during inference.

Author: HoloLoom Team
Created: December 2025
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hololoom.dark_trace.engine import DarkTraceEngine
    from hololoom.dark_trace.visualization.streaming_server import DarkTraceStreamingServer


logger = logging.getLogger(__name__)


@dataclass
class HookConfig:
    """Configuration for the orchestrator hook."""

    # Enable/disable streaming during inference
    enable_streaming: bool = True

    # Enable/disable activation recording
    enable_recording: bool = True

    # Minimum time between broadcasts (ms) to prevent flooding
    broadcast_throttle_ms: float = 50.0

    # Layers to monitor (None = all layers)
    monitored_layers: list[int] | None = None

    # Features to monitor (None = all features)
    monitored_features: list[int] | None = None

    # Activation threshold for broadcasting (only broadcast if max > threshold)
    activation_threshold: float = 0.01

    # Include metadata in broadcasts
    include_metadata: bool = True

    # Include timing information
    include_timing: bool = True

    # Maximum features to broadcast per layer (for bandwidth control)
    max_features_per_broadcast: int = 100


@dataclass
class HookState:
    """Runtime state for the orchestrator hook."""

    # Last broadcast timestamp
    last_broadcast_time: float = 0.0

    # Total broadcasts sent
    total_broadcasts: int = 0

    # Total activations recorded
    total_activations_recorded: int = 0

    # Current query being processed
    current_query: str | None = None

    # Current step in the pipeline
    current_step: str | None = None

    # Step timing information
    step_timings: dict[str, float] = field(default_factory=dict)

    # Step start time
    step_start_time: float = 0.0


class DarkTraceOrchestratorHook:
    """
    Hook that integrates Dark Trace visualization with the weaving orchestrator.

    This hook can be registered with the WeavingOrchestrator to:
    1. Stream activations in real-time during inference
    2. Record activations for later analysis
    3. Provide timing and metadata for the dashboard

    Usage:
        hook = DarkTraceOrchestratorHook(engine, streaming_server, config)
        orchestrator.register_hook("dark_trace", hook)

        # Or use the factory function
        hook = create_orchestrator_hook(engine, streaming_server)
    """

    def __init__(
        self,
        engine: Optional["DarkTraceEngine"] = None,
        streaming_server: Optional["DarkTraceStreamingServer"] = None,
        config: HookConfig | None = None,
    ):
        """
        Initialize the orchestrator hook.

        Args:
            engine: Dark Trace engine for analysis
            streaming_server: WebSocket server for real-time streaming
            config: Hook configuration
        """
        self.engine = engine
        self.streaming_server = streaming_server
        self.config = config or HookConfig()
        self.state = HookState()

        # Callbacks for custom processing
        self._pre_weave_callbacks: list[Callable] = []
        self._post_weave_callbacks: list[Callable] = []
        self._activation_callbacks: list[Callable] = []

    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================

    async def on_weave_start(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Called when a weaving cycle starts.

        Args:
            query: The query being processed
            context: Additional context information
        """
        self.state.current_query = query
        self.state.step_timings = {}
        self.state.step_start_time = time.time()

        # Broadcast start event
        if self.config.enable_streaming and self.streaming_server:
            await self._broadcast_event(
                event_type="weave_start",
                data={
                    "query": query[:100] if query else None,  # Truncate for privacy
                    "timestamp": time.time(),
                    "context": context,
                },
            )

        # Call registered callbacks
        for callback in self._pre_weave_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(query, context)
                else:
                    callback(query, context)
            except Exception as e:
                logger.warning(f"Pre-weave callback error: {e}")

    async def on_weave_end(
        self,
        query: str,
        result: Any,
        success: bool = True,
    ) -> None:
        """
        Called when a weaving cycle ends.

        Args:
            query: The query that was processed
            result: The weaving result
            success: Whether weaving succeeded
        """
        total_time = time.time() - self.state.step_start_time

        # Broadcast end event
        if self.config.enable_streaming and self.streaming_server:
            await self._broadcast_event(
                event_type="weave_end",
                data={
                    "query": query[:100] if query else None,
                    "success": success,
                    "total_time_ms": total_time * 1000,
                    "step_timings": self.state.step_timings,
                    "timestamp": time.time(),
                },
            )

        # Reset state
        self.state.current_query = None
        self.state.current_step = None

        # Call registered callbacks
        for callback in self._post_weave_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(query, result, success)
                else:
                    callback(query, result, success)
            except Exception as e:
                logger.warning(f"Post-weave callback error: {e}")

    async def on_step_start(self, step_name: str) -> None:
        """
        Called when a pipeline step starts.

        Args:
            step_name: Name of the step
        """
        self.state.current_step = step_name
        self.state.step_start_time = time.time()

        if self.config.enable_streaming and self.streaming_server:
            await self._broadcast_event(
                event_type="step_start",
                data={
                    "step": step_name,
                    "timestamp": time.time(),
                },
            )

    async def on_step_end(self, step_name: str) -> None:
        """
        Called when a pipeline step ends.

        Args:
            step_name: Name of the step
        """
        step_time = time.time() - self.state.step_start_time
        self.state.step_timings[step_name] = step_time * 1000  # Convert to ms

        if self.config.enable_streaming and self.streaming_server:
            await self._broadcast_event(
                event_type="step_end",
                data={
                    "step": step_name,
                    "duration_ms": step_time * 1000,
                    "timestamp": time.time(),
                },
            )

    # =========================================================================
    # Activation Hooks
    # =========================================================================

    async def on_activations(
        self,
        layer: int,
        feature_indices: list[int],
        activation_values: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Called when activations are computed.

        Args:
            layer: Layer index
            feature_indices: Indices of activated features
            activation_values: Activation values
            metadata: Additional metadata
        """
        # Check if this layer should be monitored
        if self.config.monitored_layers is not None:
            if layer not in self.config.monitored_layers:
                return

        # Filter features if specified
        if self.config.monitored_features is not None:
            filtered_indices = []
            filtered_values = []
            for idx, val in zip(feature_indices, activation_values):
                if idx in self.config.monitored_features:
                    filtered_indices.append(idx)
                    filtered_values.append(val)
            feature_indices = filtered_indices
            activation_values = filtered_values

        if not feature_indices:
            return

        # Check activation threshold
        max_activation = max(activation_values) if activation_values else 0.0
        if max_activation < self.config.activation_threshold:
            return

        # Limit features per broadcast
        if len(feature_indices) > self.config.max_features_per_broadcast:
            # Sort by activation value and take top N
            sorted_pairs = sorted(
                zip(feature_indices, activation_values),
                key=lambda x: x[1],
                reverse=True,
            )[:self.config.max_features_per_broadcast]
            feature_indices = [p[0] for p in sorted_pairs]
            activation_values = [p[1] for p in sorted_pairs]

        # Record activations
        if self.config.enable_recording:
            self.state.total_activations_recorded += len(feature_indices)

        # Broadcast if enabled and not throttled
        if self.config.enable_streaming and self.streaming_server:
            current_time = time.time() * 1000  # ms
            if current_time - self.state.last_broadcast_time >= self.config.broadcast_throttle_ms:
                self.state.last_broadcast_time = current_time
                self.state.total_broadcasts += 1

                # Add timing metadata if enabled
                if self.config.include_timing:
                    metadata = metadata or {}
                    metadata["step"] = self.state.current_step
                    metadata["broadcast_num"] = self.state.total_broadcasts

                await self.streaming_server.broadcast_activations(
                    layer=layer,
                    feature_indices=feature_indices,
                    activation_values=activation_values,
                    metadata=metadata if self.config.include_metadata else None,
                )

        # Call registered callbacks
        for callback in self._activation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(layer, feature_indices, activation_values, metadata)
                else:
                    callback(layer, feature_indices, activation_values, metadata)
            except Exception as e:
                logger.warning(f"Activation callback error: {e}")

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def register_pre_weave_callback(self, callback: Callable) -> None:
        """Register a callback to be called before weaving starts."""
        self._pre_weave_callbacks.append(callback)

    def register_post_weave_callback(self, callback: Callable) -> None:
        """Register a callback to be called after weaving ends."""
        self._post_weave_callbacks.append(callback)

    def register_activation_callback(self, callback: Callable) -> None:
        """Register a callback to be called when activations are computed."""
        self._activation_callbacks.append(callback)

    # =========================================================================
    # Statistics and State
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get hook statistics."""
        return {
            "total_broadcasts": self.state.total_broadcasts,
            "total_activations_recorded": self.state.total_activations_recorded,
            "current_query": self.state.current_query[:50] if self.state.current_query else None,
            "current_step": self.state.current_step,
            "config": {
                "streaming_enabled": self.config.enable_streaming,
                "recording_enabled": self.config.enable_recording,
                "throttle_ms": self.config.broadcast_throttle_ms,
                "monitored_layers": self.config.monitored_layers,
                "activation_threshold": self.config.activation_threshold,
            },
        }

    def reset_statistics(self) -> None:
        """Reset hook statistics."""
        self.state.total_broadcasts = 0
        self.state.total_activations_recorded = 0

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _broadcast_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast a lifecycle event."""
        if not self.streaming_server:
            return

        try:
            await self.streaming_server.broadcast_message({
                "type": event_type,
                "data": data,
            })
        except Exception as e:
            logger.warning(f"Failed to broadcast event {event_type}: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_orchestrator_hook(
    engine: Optional["DarkTraceEngine"] = None,
    streaming_server: Optional["DarkTraceStreamingServer"] = None,
    enable_streaming: bool = True,
    enable_recording: bool = True,
    **kwargs,
) -> DarkTraceOrchestratorHook:
    """
    Create an orchestrator hook with the given configuration.

    Args:
        engine: Dark Trace engine
        streaming_server: WebSocket streaming server
        enable_streaming: Enable real-time streaming
        enable_recording: Enable activation recording
        **kwargs: Additional HookConfig parameters

    Returns:
        Configured DarkTraceOrchestratorHook
    """
    config = HookConfig(
        enable_streaming=enable_streaming,
        enable_recording=enable_recording,
        **kwargs,
    )
    return DarkTraceOrchestratorHook(
        engine=engine,
        streaming_server=streaming_server,
        config=config,
    )


def create_hook_config(
    preset: str = "standard",
    **overrides,
) -> HookConfig:
    """
    Create a hook configuration from a preset.

    Presets:
        - minimal: Low bandwidth, basic monitoring
        - standard: Balanced monitoring (default)
        - full: Maximum detail, higher bandwidth

    Args:
        preset: Configuration preset name
        **overrides: Override specific settings

    Returns:
        HookConfig instance
    """
    presets = {
        "minimal": HookConfig(
            enable_streaming=True,
            enable_recording=False,
            broadcast_throttle_ms=200.0,
            activation_threshold=0.1,
            max_features_per_broadcast=20,
            include_metadata=False,
            include_timing=False,
        ),
        "standard": HookConfig(
            enable_streaming=True,
            enable_recording=True,
            broadcast_throttle_ms=50.0,
            activation_threshold=0.01,
            max_features_per_broadcast=100,
            include_metadata=True,
            include_timing=True,
        ),
        "full": HookConfig(
            enable_streaming=True,
            enable_recording=True,
            broadcast_throttle_ms=10.0,
            activation_threshold=0.0,
            max_features_per_broadcast=500,
            include_metadata=True,
            include_timing=True,
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")

    config = presets[preset]

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config

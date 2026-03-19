"""
Dark Trace Visualization: Real-time Streaming Server
=====================================================

WebSocket-based streaming server for live feature activation visualization.
Enables real-time monitoring of SAE features during model inference.

Key Components:
- StreamingConfig: Server configuration
- DarkTraceStreamingServer: Main WebSocket server
- ActivationStreamHandler: Data formatting and filtering
- StreamMessage: Standardized message protocol

Usage:
    from hololoom.dark_trace.visualization.streaming_server import (
        DarkTraceStreamingServer,
        StreamingConfig,
        create_streaming_server,
    )

    # Create server
    config = StreamingConfig(port=8765, max_clients=50)
    server = create_streaming_server(config)

    # Start server
    await server.start()

    # Stream activations
    await server.broadcast_activations(feature_activations)

    # Stop server
    await server.stop()

Created: 2025-12-28
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
)

# Try to import websockets, gracefully degrade if not available
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = Any  # Type stub


logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class MessageType(Enum):
    """Types of streaming messages."""
    # Server → Client
    ACTIVATIONS = "activations"
    FEATURE_UPDATE = "feature_update"
    LAYER_SNAPSHOT = "layer_snapshot"
    STATISTICS = "statistics"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"

    # Client → Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    REQUEST_SNAPSHOT = "request_snapshot"
    CONFIGURE = "configure"


class SubscriptionType(Enum):
    """Types of subscriptions clients can make."""
    ALL = "all"                    # All activations
    LAYER = "layer"                # Specific layer only
    FEATURE = "feature"            # Specific feature(s)
    TOP_K = "top_k"                # Top K activating features
    THRESHOLD = "threshold"        # Features above threshold
    ALERTS_ONLY = "alerts_only"    # Only alerts/anomalies


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class StreamingConfig:
    """Configuration for streaming server."""
    host: str = "localhost"
    port: int = 8765
    max_clients: int = 50
    heartbeat_interval: float = 30.0  # seconds
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    enable_compression: bool = True
    buffer_size: int = 100  # Message buffer for replay

    # Filtering defaults
    default_top_k: int = 50
    default_threshold: float = 0.1

    # Rate limiting
    max_messages_per_second: float = 60.0
    burst_limit: int = 100


@dataclass
class StreamMessage:
    """Standardized streaming message format."""
    type: MessageType
    timestamp: float
    data: dict[str, Any]
    sequence: int = 0
    client_id: str | None = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "client_id": self.client_id,
            "data": self.data,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "StreamMessage":
        """Parse from JSON string."""
        d = json.loads(json_str)
        return cls(
            type=MessageType(d["type"]),
            timestamp=d["timestamp"],
            sequence=d.get("sequence", 0),
            client_id=d.get("client_id"),
            data=d["data"],
        )


@dataclass
class ClientSubscription:
    """Client subscription configuration."""
    client_id: str
    subscription_type: SubscriptionType
    filters: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_message_at: float = 0.0
    message_count: int = 0


@dataclass
class ActivationSnapshot:
    """Snapshot of feature activations for streaming."""
    layer: int
    feature_indices: list[int]
    activation_values: list[float]
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer": self.layer,
            "features": [
                {"index": idx, "activation": val}
                for idx, val in zip(self.feature_indices, self.activation_values)
            ],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# =============================================================================
# Activation Stream Handler
# =============================================================================


class ActivationStreamHandler:
    """
    Handles activation data formatting and filtering for streaming.

    Supports:
    - Top-K filtering
    - Threshold filtering
    - Layer-specific filtering
    - Feature-specific filtering
    - Data compression/batching
    """

    def __init__(self, config: StreamingConfig):
        """
        Initialize handler.

        Args:
            config: Streaming configuration
        """
        self.config = config
        self._sequence = 0

    def format_activations(
        self,
        layer: int,
        feature_indices: list[int],
        activation_values: list[float],
        subscription: ClientSubscription,
        metadata: dict[str, Any] | None = None,
    ) -> StreamMessage | None:
        """
        Format activations for a client based on their subscription.

        Args:
            layer: Layer index
            feature_indices: Feature indices
            activation_values: Activation values
            subscription: Client subscription
            metadata: Optional metadata

        Returns:
            StreamMessage if data matches subscription, None otherwise
        """
        # Apply layer filter
        if subscription.subscription_type == SubscriptionType.LAYER:
            target_layer = subscription.filters.get("layer")
            if target_layer is not None and layer != target_layer:
                return None

        # Zip and filter
        pairs = list(zip(feature_indices, activation_values))

        # Apply subscription-specific filtering
        if subscription.subscription_type == SubscriptionType.TOP_K:
            k = subscription.filters.get("k", self.config.default_top_k)
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:k]

        elif subscription.subscription_type == SubscriptionType.THRESHOLD:
            threshold = subscription.filters.get(
                "threshold", self.config.default_threshold
            )
            pairs = [(idx, val) for idx, val in pairs if val >= threshold]

        elif subscription.subscription_type == SubscriptionType.FEATURE:
            target_features = set(subscription.filters.get("features", []))
            if target_features:
                pairs = [(idx, val) for idx, val in pairs if idx in target_features]

        elif subscription.subscription_type == SubscriptionType.ALERTS_ONLY:
            # Only return if there are anomalous activations
            alert_threshold = subscription.filters.get("alert_threshold", 0.9)
            pairs = [(idx, val) for idx, val in pairs if val >= alert_threshold]
            if not pairs:
                return None

        # No matching data
        if not pairs:
            return None

        # Create snapshot
        snapshot = ActivationSnapshot(
            layer=layer,
            feature_indices=[p[0] for p in pairs],
            activation_values=[p[1] for p in pairs],
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self._sequence += 1

        return StreamMessage(
            type=MessageType.ACTIVATIONS,
            timestamp=snapshot.timestamp,
            sequence=self._sequence,
            client_id=subscription.client_id,
            data=snapshot.to_dict(),
        )

    def create_layer_snapshot(
        self,
        layer: int,
        all_activations: list[float],
        top_k: int = 50,
    ) -> StreamMessage:
        """
        Create a complete layer snapshot message.

        Args:
            layer: Layer index
            all_activations: All activation values for layer
            top_k: Number of top features to include

        Returns:
            StreamMessage with layer snapshot
        """
        # Get top-K indices and values
        indexed = list(enumerate(all_activations))
        sorted_indexed = sorted(indexed, key=lambda x: x[1], reverse=True)
        top = sorted_indexed[:top_k]

        self._sequence += 1

        return StreamMessage(
            type=MessageType.LAYER_SNAPSHOT,
            timestamp=time.time(),
            sequence=self._sequence,
            data={
                "layer": layer,
                "total_features": len(all_activations),
                "top_k": top_k,
                "features": [
                    {"index": idx, "activation": val}
                    for idx, val in top
                ],
                "statistics": {
                    "mean": sum(all_activations) / len(all_activations) if all_activations else 0,
                    "max": max(all_activations) if all_activations else 0,
                    "min": min(all_activations) if all_activations else 0,
                    "nonzero_count": sum(1 for v in all_activations if v > 0),
                },
            },
        )

    def create_statistics_message(
        self,
        statistics: dict[str, Any],
    ) -> StreamMessage:
        """
        Create a statistics update message.

        Args:
            statistics: Statistics dictionary

        Returns:
            StreamMessage with statistics
        """
        self._sequence += 1

        return StreamMessage(
            type=MessageType.STATISTICS,
            timestamp=time.time(),
            sequence=self._sequence,
            data=statistics,
        )

    def create_alert_message(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        details: dict[str, Any] | None = None,
    ) -> StreamMessage:
        """
        Create an alert message.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            details: Additional details

        Returns:
            StreamMessage with alert
        """
        self._sequence += 1

        return StreamMessage(
            type=MessageType.ALERT,
            timestamp=time.time(),
            sequence=self._sequence,
            data={
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "details": details or {},
            },
        )

    def create_heartbeat(self) -> StreamMessage:
        """Create heartbeat message."""
        self._sequence += 1

        return StreamMessage(
            type=MessageType.HEARTBEAT,
            timestamp=time.time(),
            sequence=self._sequence,
            data={"status": "alive"},
        )


# =============================================================================
# WebSocket Server
# =============================================================================


class DarkTraceStreamingServer:
    """
    WebSocket server for real-time Dark Trace visualization.

    Features:
    - Multiple concurrent clients
    - Subscription-based filtering
    - Message buffering for replay
    - Heartbeat for connection health
    - Rate limiting
    - Graceful shutdown

    Usage:
        server = DarkTraceStreamingServer(config)
        await server.start()

        # Broadcast to all clients
        await server.broadcast_activations(layer, indices, values)

        # Send to specific client
        await server.send_to_client(client_id, message)

        await server.stop()
    """

    def __init__(self, config: StreamingConfig | None = None):
        """
        Initialize server.

        Args:
            config: Server configuration
        """
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError(
                "websockets package not installed. "
                "Install with: pip install websockets"
            )

        self.config = config or StreamingConfig()
        self.handler = ActivationStreamHandler(self.config)

        # Client management
        self._clients: dict[str, WebSocketServerProtocol] = {}
        self._subscriptions: dict[str, ClientSubscription] = {}
        self._client_counter = 0

        # Message buffer for replay
        self._message_buffer: list[StreamMessage] = []

        # Server state
        self._server: Any | None = None
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None

        # Rate limiting
        self._message_times: list[float] = []

        # Callbacks
        self._on_client_connect: Callable[[str], Awaitable[None]] | None = None
        self._on_client_disconnect: Callable[[str], Awaitable[None]] | None = None
        self._on_message: Callable[[str, StreamMessage], Awaitable[None]] | None = None

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            logger.warning("Server already running")
            return

        self._server = await websockets.serve(
            self._handle_client,
            self.config.host,
            self.config.port,
            max_size=self.config.max_message_size,
            compression="deflate" if self.config.enable_compression else None,
        )

        self._running = True

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            f"Dark Trace streaming server started on "
            f"ws://{self.config.host}:{self.config.port}"
        )

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            return

        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all client connections
        for client_id, ws in list(self._clients.items()):
            try:
                await ws.close(1001, "Server shutting down")
            except Exception:
                pass

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        self._clients.clear()
        self._subscriptions.clear()

        logger.info("Dark Trace streaming server stopped")

    async def _handle_client(
        self,
        websocket: WebSocketServerProtocol,
        path: str,
    ) -> None:
        """
        Handle a client connection.

        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        # Check client limit
        if len(self._clients) >= self.config.max_clients:
            await websocket.close(1013, "Server at capacity")
            return

        # Generate client ID
        self._client_counter += 1
        client_id = f"client_{self._client_counter}_{int(time.time())}"

        # Register client
        self._clients[client_id] = websocket
        self._subscriptions[client_id] = ClientSubscription(
            client_id=client_id,
            subscription_type=SubscriptionType.TOP_K,
            filters={"k": self.config.default_top_k},
        )

        logger.info(f"Client connected: {client_id}")

        # Callback
        if self._on_client_connect:
            await self._on_client_connect(client_id)

        # Send welcome message
        welcome = StreamMessage(
            type=MessageType.STATISTICS,
            timestamp=time.time(),
            sequence=0,
            client_id=client_id,
            data={
                "welcome": True,
                "client_id": client_id,
                "server_time": datetime.now().isoformat(),
                "buffer_size": len(self._message_buffer),
            },
        )
        await websocket.send(welcome.to_json())

        try:
            async for message in websocket:
                await self._handle_message(client_id, message)
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Cleanup
            self._clients.pop(client_id, None)
            self._subscriptions.pop(client_id, None)

            logger.info(f"Client disconnected: {client_id}")

            if self._on_client_disconnect:
                await self._on_client_disconnect(client_id)

    async def _handle_message(
        self,
        client_id: str,
        message: str,
    ) -> None:
        """
        Handle incoming client message.

        Args:
            client_id: Client identifier
            message: Raw message string
        """
        try:
            msg = StreamMessage.from_json(message)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Invalid message from {client_id}: {e}")
            return

        # Callback
        if self._on_message:
            await self._on_message(client_id, msg)

        # Handle message types
        if msg.type == MessageType.SUBSCRIBE:
            await self._handle_subscribe(client_id, msg.data)

        elif msg.type == MessageType.UNSUBSCRIBE:
            # Reset to default subscription
            self._subscriptions[client_id] = ClientSubscription(
                client_id=client_id,
                subscription_type=SubscriptionType.TOP_K,
                filters={"k": self.config.default_top_k},
            )

        elif msg.type == MessageType.REQUEST_SNAPSHOT:
            # Send buffered messages
            await self._send_buffer_to_client(client_id)

        elif msg.type == MessageType.CONFIGURE:
            # Update subscription filters
            if client_id in self._subscriptions:
                sub = self._subscriptions[client_id]
                sub.filters.update(msg.data.get("filters", {}))

    async def _handle_subscribe(
        self,
        client_id: str,
        data: dict[str, Any],
    ) -> None:
        """
        Handle subscription request.

        Args:
            client_id: Client identifier
            data: Subscription data
        """
        try:
            sub_type = SubscriptionType(data.get("type", "top_k"))
        except ValueError:
            sub_type = SubscriptionType.TOP_K

        self._subscriptions[client_id] = ClientSubscription(
            client_id=client_id,
            subscription_type=sub_type,
            filters=data.get("filters", {}),
        )

        logger.debug(f"Client {client_id} subscribed to {sub_type.value}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to all clients."""
        while self._running:
            await asyncio.sleep(self.config.heartbeat_interval)

            if not self._running:
                break

            heartbeat = self.handler.create_heartbeat()
            await self._broadcast_raw(heartbeat)

    async def _send_buffer_to_client(self, client_id: str) -> None:
        """
        Send buffered messages to a client.

        Args:
            client_id: Client identifier
        """
        if client_id not in self._clients:
            return

        ws = self._clients[client_id]

        for msg in self._message_buffer:
            try:
                await ws.send(msg.to_json())
            except Exception:
                break

    async def _broadcast_raw(self, message: StreamMessage) -> None:
        """
        Broadcast message to all clients without filtering.

        Args:
            message: Message to broadcast
        """
        if not self._clients:
            return

        msg_json = message.to_json()

        disconnected = []
        for client_id, ws in self._clients.items():
            try:
                await ws.send(msg_json)
            except Exception:
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            self._clients.pop(client_id, None)
            self._subscriptions.pop(client_id, None)

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()

        # Clean old entries
        cutoff = now - 1.0  # 1 second window
        self._message_times = [t for t in self._message_times if t > cutoff]

        # Check limit
        if len(self._message_times) >= self.config.max_messages_per_second:
            return False

        self._message_times.append(now)
        return True

    def _add_to_buffer(self, message: StreamMessage) -> None:
        """Add message to buffer, evicting old if needed."""
        self._message_buffer.append(message)

        # Evict old messages
        while len(self._message_buffer) > self.config.buffer_size:
            self._message_buffer.pop(0)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def broadcast_activations(
        self,
        layer: int,
        feature_indices: list[int],
        activation_values: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Broadcast activations to all subscribed clients.

        Args:
            layer: Layer index
            feature_indices: Feature indices
            activation_values: Activation values
            metadata: Optional metadata

        Returns:
            Number of clients that received the message
        """
        if not self._clients:
            return 0

        if not self._check_rate_limit():
            logger.debug("Rate limit exceeded, skipping broadcast")
            return 0

        sent_count = 0
        disconnected = []

        for client_id, ws in self._clients.items():
            sub = self._subscriptions.get(client_id)
            if not sub:
                continue

            msg = self.handler.format_activations(
                layer=layer,
                feature_indices=feature_indices,
                activation_values=activation_values,
                subscription=sub,
                metadata=metadata,
            )

            if msg:
                try:
                    await ws.send(msg.to_json())
                    sub.last_message_at = time.time()
                    sub.message_count += 1
                    sent_count += 1
                except Exception:
                    disconnected.append(client_id)

        # Clean up
        for client_id in disconnected:
            self._clients.pop(client_id, None)
            self._subscriptions.pop(client_id, None)

        return sent_count

    async def broadcast_layer_snapshot(
        self,
        layer: int,
        all_activations: list[float],
        top_k: int | None = None,
    ) -> int:
        """
        Broadcast a complete layer snapshot.

        Args:
            layer: Layer index
            all_activations: All activation values
            top_k: Number of top features (default from config)

        Returns:
            Number of clients that received the message
        """
        msg = self.handler.create_layer_snapshot(
            layer=layer,
            all_activations=all_activations,
            top_k=top_k or self.config.default_top_k,
        )

        self._add_to_buffer(msg)
        await self._broadcast_raw(msg)

        return len(self._clients)

    async def broadcast_statistics(
        self,
        statistics: dict[str, Any],
    ) -> int:
        """
        Broadcast statistics update.

        Args:
            statistics: Statistics dictionary

        Returns:
            Number of clients that received the message
        """
        msg = self.handler.create_statistics_message(statistics)
        await self._broadcast_raw(msg)
        return len(self._clients)

    async def broadcast_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        details: dict[str, Any] | None = None,
    ) -> int:
        """
        Broadcast an alert to all clients.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Severity level
            details: Additional details

        Returns:
            Number of clients that received the message
        """
        msg = self.handler.create_alert_message(
            alert_type=alert_type,
            message=message,
            severity=severity,
            details=details,
        )

        self._add_to_buffer(msg)
        await self._broadcast_raw(msg)

        return len(self._clients)

    async def send_to_client(
        self,
        client_id: str,
        message: StreamMessage,
    ) -> bool:
        """
        Send a message to a specific client.

        Args:
            client_id: Client identifier
            message: Message to send

        Returns:
            True if sent successfully
        """
        ws = self._clients.get(client_id)
        if not ws:
            return False

        try:
            await ws.send(message.to_json())
            return True
        except Exception:
            self._clients.pop(client_id, None)
            self._subscriptions.pop(client_id, None)
            return False

    def get_client_ids(self) -> list[str]:
        """Get list of connected client IDs."""
        return list(self._clients.keys())

    def get_client_subscription(
        self,
        client_id: str,
    ) -> ClientSubscription | None:
        """Get subscription for a client."""
        return self._subscriptions.get(client_id)

    def set_on_client_connect(
        self,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        """Set callback for client connection."""
        self._on_client_connect = callback

    def set_on_client_disconnect(
        self,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        """Set callback for client disconnection."""
        self._on_client_disconnect = callback

    def set_on_message(
        self,
        callback: Callable[[str, StreamMessage], Awaitable[None]],
    ) -> None:
        """Set callback for incoming messages."""
        self._on_message = callback


# =============================================================================
# Factory Functions
# =============================================================================


def create_streaming_server(
    config: StreamingConfig | None = None,
) -> DarkTraceStreamingServer:
    """
    Factory function for streaming server.

    Args:
        config: Server configuration

    Returns:
        DarkTraceStreamingServer instance
    """
    return DarkTraceStreamingServer(config)


def create_streaming_config(
    port: int = 8765,
    max_clients: int = 50,
    **kwargs,
) -> StreamingConfig:
    """
    Factory function for streaming configuration.

    Args:
        port: WebSocket port
        max_clients: Maximum concurrent clients
        **kwargs: Additional configuration

    Returns:
        StreamingConfig instance
    """
    return StreamingConfig(
        port=port,
        max_clients=max_clients,
        **kwargs,
    )

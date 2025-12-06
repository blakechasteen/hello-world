"""
Federation Transport Layer - Network communication abstraction.

Provides swappable transport implementations for inter-node communication.
Default: HTTP/JSON transport (simple, debuggable)
Future: gRPC transport (high performance)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from ..types import FederationNode, Query, Response

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  MESSAGE TYPES - What flows over the wire
# ═══════════════════════════════════════════════════════════════════════════


class MessageType(Enum):
    """Types of messages that can be sent between nodes."""

    # Gossip messages (SWIM protocol)
    PING = auto()
    PING_REQ = auto()
    ACK = auto()
    ALIVE = auto()
    SUSPECT = auto()
    DEAD = auto()
    JOIN = auto()
    LEAVE = auto()

    # Routing messages (Kademlia DHT)
    FIND_NODE = auto()
    FIND_NODE_RESPONSE = auto()
    ANNOUNCE = auto()

    # Query messages
    QUERY = auto()
    QUERY_RESPONSE = auto()

    # Verification messages
    VERIFY_REQUEST = auto()
    VERIFY_RESPONSE = auto()


@dataclass
class TransportMessage:
    """
    A message to be sent between federation nodes.

    This is the wire format - serialized to JSON for HTTP transport.
    """

    type: MessageType
    sender_id: str                              # Node ID of sender
    target_id: Optional[str] = None             # Target node ID (if specific)
    request_id: str = ""                        # Correlation ID for request/response
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signature: bytes = b""                      # Ed25519 signature

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.name,
            "sender_id": self.sender_id,
            "target_id": self.target_id,
            "request_id": self.request_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature.hex() if self.signature else "",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransportMessage":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            type=MessageType[data["type"]],
            sender_id=data["sender_id"],
            target_id=data.get("target_id"),
            request_id=data.get("request_id", ""),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            signature=bytes.fromhex(data.get("signature", "")),
        )


@dataclass
class TransportResponse:
    """Response from a transport operation."""

    success: bool
    message: Optional[TransportMessage] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSPORT PROTOCOL - Abstract interface for all transports
# ═══════════════════════════════════════════════════════════════════════════


# Handler type for incoming messages
MessageHandler = Callable[[TransportMessage], TransportMessage]


@runtime_checkable
class TransportProtocol(Protocol):
    """
    Protocol for federation transport layer.

    Implementations must provide:
    - send(): Send message to single node
    - broadcast(): Send message to multiple nodes
    - start_server(): Start listening for incoming messages
    - stop_server(): Stop listening

    This abstraction allows swapping HTTP for gRPC without changing
    the gossip/routing/core modules.
    """

    async def send(
        self,
        node: FederationNode,
        message: TransportMessage,
        timeout_ms: int = 5000,
    ) -> TransportResponse:
        """
        Send message to a single node.

        Args:
            node: Target node
            message: Message to send
            timeout_ms: Timeout in milliseconds

        Returns:
            TransportResponse with success status and optional response message
        """
        ...

    async def broadcast(
        self,
        nodes: List[FederationNode],
        message: TransportMessage,
        timeout_ms: int = 5000,
    ) -> List[TransportResponse]:
        """
        Send message to multiple nodes in parallel.

        Args:
            nodes: Target nodes
            message: Message to send
            timeout_ms: Timeout in milliseconds

        Returns:
            List of TransportResponse, one per node (order preserved)
        """
        ...

    async def start_server(
        self,
        handler: MessageHandler,
        host: str = "0.0.0.0",
        port: int = 9000,
    ) -> None:
        """
        Start listening for incoming messages.

        Args:
            handler: Callback for incoming messages (returns response message)
            host: Host to bind to
            port: Port to listen on
        """
        ...

    async def stop_server(self) -> None:
        """Stop the server gracefully."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether the server is currently running."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  BASE TRANSPORT - Common functionality
# ═══════════════════════════════════════════════════════════════════════════


class BaseTransport(ABC):
    """
    Base class for transport implementations.

    Provides common functionality like logging, metrics, and error handling.
    Subclasses implement the actual network I/O.
    """

    def __init__(self, node_id: str):
        self._node_id = node_id
        self._running = False
        self._handler: Optional[MessageHandler] = None

        # Metrics
        self._messages_sent = 0
        self._messages_received = 0
        self._bytes_sent = 0
        self._bytes_received = 0

    @property
    def is_running(self) -> bool:
        """Whether the server is currently running."""
        return self._running

    @property
    def node_id(self) -> str:
        """This node's ID."""
        return self._node_id

    def get_metrics(self) -> Dict[str, Any]:
        """Get transport metrics."""
        return {
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "bytes_sent": self._bytes_sent,
            "bytes_received": self._bytes_received,
            "running": self._running,
        }

    @abstractmethod
    async def send(
        self,
        node: FederationNode,
        message: TransportMessage,
        timeout_ms: int = 5000,
    ) -> TransportResponse:
        """Send message to a single node."""
        ...

    async def broadcast(
        self,
        nodes: List[FederationNode],
        message: TransportMessage,
        timeout_ms: int = 5000,
    ) -> List[TransportResponse]:
        """
        Default broadcast implementation - send to each node in parallel.

        Subclasses can override for more efficient implementations.
        """
        import asyncio

        if not nodes:
            return []

        tasks = [self.send(node, message, timeout_ms) for node in nodes]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    @abstractmethod
    async def start_server(
        self,
        handler: MessageHandler,
        host: str = "0.0.0.0",
        port: int = 9000,
    ) -> None:
        """Start listening for incoming messages."""
        ...

    @abstractmethod
    async def stop_server(self) -> None:
        """Stop the server gracefully."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY - Create transports
# ═══════════════════════════════════════════════════════════════════════════


def create_http_transport(node_id: str) -> "HTTPTransport":
    """
    Create an HTTP/JSON transport.

    This is the default transport - simple, debuggable, works everywhere.
    """
    from .http_transport import HTTPTransport

    return HTTPTransport(node_id)


def create_transport(
    node_id: str,
    transport_type: str = "http",
) -> BaseTransport:
    """
    Create a transport of the specified type.

    Args:
        node_id: This node's ID (for message signing)
        transport_type: "http" or "grpc" (future)

    Returns:
        Transport implementation
    """
    if transport_type == "http":
        return create_http_transport(node_id)
    else:
        raise ValueError(f"Unknown transport type: {transport_type}")


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Message types
    "MessageType",
    "TransportMessage",
    "TransportResponse",
    # Protocol
    "TransportProtocol",
    "MessageHandler",
    # Base class
    "BaseTransport",
    # Factory
    "create_transport",
    "create_http_transport",
]

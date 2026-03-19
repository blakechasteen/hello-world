"""
Handoff Transport Layer - Pluggable transports for cross-device sync.

Created: 2025-12-08

System works with ANY or NONE of these transports.
Graceful degradation is built-in.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSPORT PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════


class TransportStatus(Enum):
    """Transport connection status."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()


@dataclass
class TransportMessage:
    """A message received from transport."""

    source_device: str
    data: bytes
    timestamp: float = field(default_factory=time.time)
    transport_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SendResult:
    """Result of sending a message."""

    success: bool
    transport_used: str
    latency_ms: float = 0.0
    error: str | None = None
    retries: int = 0


@runtime_checkable
class TransportProtocol(Protocol):
    """
    Abstract transport interface.

    System works with ANY or NONE of these transports.
    All transports are optional - graceful degradation built-in.
    """

    @property
    def transport_type(self) -> str:
        """Unique identifier for this transport type."""
        ...

    @property
    def status(self) -> TransportStatus:
        """Current connection status."""
        ...

    async def connect(self) -> bool:
        """
        Establish connection.

        Returns:
            True if connected successfully
        """
        ...

    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        ...

    async def send(self, target: str, data: bytes) -> SendResult:
        """
        Send data to target device.

        Args:
            target: Device ID to send to
            data: Raw bytes to send

        Returns:
            SendResult with success/failure info
        """
        ...

    async def receive(self) -> AsyncIterator[TransportMessage]:
        """
        Receive messages from connected devices.

        Yields:
            TransportMessage for each received message
        """
        ...

    async def discover_peers(self) -> list[str]:
        """
        Discover available peer devices.

        Returns:
            List of discovered device IDs
        """
        ...

    def is_available(self) -> bool:
        """Check if transport is available (has required dependencies)."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  BASE TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════


class BaseTransport(ABC):
    """Base class for all transports with common functionality."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self._status = TransportStatus.DISCONNECTED
        self._message_queue: asyncio.Queue[TransportMessage] = asyncio.Queue()
        self._connected_peers: set[str] = set()
        self._send_count = 0
        self._receive_count = 0
        self._error_count = 0

    @property
    @abstractmethod
    def transport_type(self) -> str:
        """Unique identifier for this transport type."""
        pass

    @property
    def status(self) -> TransportStatus:
        """Current connection status."""
        return self._status

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        pass

    @abstractmethod
    async def send(self, target: str, data: bytes) -> SendResult:
        """Send data to target device."""
        pass

    async def receive(self) -> AsyncIterator[TransportMessage]:
        """Receive messages from queue."""
        while True:
            try:
                msg = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=0.1
                )
                self._receive_count += 1
                yield msg
            except asyncio.TimeoutError:
                # No message available, continue loop
                continue
            except asyncio.CancelledError:
                break

    @abstractmethod
    async def discover_peers(self) -> list[str]:
        """Discover available peer devices."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if transport is available."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get transport statistics."""
        return {
            "transport_type": self.transport_type,
            "status": self._status.name,
            "connected_peers": list(self._connected_peers),
            "send_count": self._send_count,
            "receive_count": self._receive_count,
            "error_count": self._error_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════


class WebSocketTransport(BaseTransport):
    """
    Primary: Internet-based sync via WebSocket relay.

    Connects to a central relay server for message routing.
    Works anywhere with internet connectivity.
    """

    def __init__(
        self,
        device_id: str,
        relay_url: str = "wss://relay.hololoom.net",
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        super().__init__(device_id)
        self.relay_url = relay_url
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self._ws = None
        self._receive_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def transport_type(self) -> str:
        return "websocket"

    def is_available(self) -> bool:
        """Check if websockets library is available."""
        try:
            import websockets  # noqa: F401
            return True
        except ImportError:
            return False

    async def connect(self) -> bool:
        """Connect to WebSocket relay server."""
        if not self.is_available():
            logger.warning("WebSocket transport unavailable: websockets not installed")
            return False

        self._status = TransportStatus.CONNECTING

        try:
            import websockets

            self._ws = await asyncio.wait_for(
                websockets.connect(self.relay_url),
                timeout=10.0
            )

            # Register with relay
            await self._ws.send(json.dumps({
                "type": "register",
                "device_id": self.device_id,
                "timestamp": time.time(),
            }))

            response = await asyncio.wait_for(
                self._ws.recv(),
                timeout=5.0
            )
            data = json.loads(response)

            if data.get("type") == "registered":
                self._status = TransportStatus.CONNECTED
                self._receive_task = asyncio.create_task(self._receive_loop())
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                logger.info(f"WebSocket connected to {self.relay_url}")
                return True
            else:
                self._status = TransportStatus.ERROR
                logger.error(f"Registration failed: {data}")
                return False

        except asyncio.TimeoutError:
            self._status = TransportStatus.ERROR
            self._error_count += 1
            logger.error(f"WebSocket connection timeout to {self.relay_url}")
            return False
        except Exception as e:
            self._status = TransportStatus.ERROR
            self._error_count += 1
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect from relay."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

        self._status = TransportStatus.DISCONNECTED
        logger.info("WebSocket disconnected")

    async def send(self, target: str, data: bytes) -> SendResult:
        """Send data to target device via relay."""
        if self._status != TransportStatus.CONNECTED or not self._ws:
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error="Not connected"
            )

        start = time.perf_counter()

        try:
            message = json.dumps({
                "type": "message",
                "from": self.device_id,
                "to": target,
                "data": data.hex(),  # Hex-encode binary data
                "timestamp": time.time(),
            })

            await asyncio.wait_for(
                self._ws.send(message),
                timeout=5.0
            )

            self._send_count += 1
            latency = (time.perf_counter() - start) * 1000

            return SendResult(
                success=True,
                transport_used=self.transport_type,
                latency_ms=latency
            )

        except asyncio.TimeoutError:
            self._error_count += 1
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error="Send timeout"
            )
        except Exception as e:
            self._error_count += 1
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error=str(e)
            )

    async def discover_peers(self) -> list[str]:
        """Request peer list from relay."""
        if self._status != TransportStatus.CONNECTED or not self._ws:
            return []

        try:
            await self._ws.send(json.dumps({
                "type": "discover",
                "device_id": self.device_id,
            }))

            response = await asyncio.wait_for(
                self._ws.recv(),
                timeout=5.0
            )
            data = json.loads(response)

            if data.get("type") == "peers":
                peers = data.get("peers", [])
                self._connected_peers = set(peers)
                return peers

        except Exception as e:
            logger.error(f"Peer discovery failed: {e}")

        return []

    async def _receive_loop(self) -> None:
        """Background loop to receive messages."""
        while self._status == TransportStatus.CONNECTED and self._ws:
            try:
                raw = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=1.0
                )
                data = json.loads(raw)

                if data.get("type") == "message":
                    msg = TransportMessage(
                        source_device=data.get("from", "unknown"),
                        data=bytes.fromhex(data.get("data", "")),
                        timestamp=data.get("timestamp", time.time()),
                        transport_type=self.transport_type,
                    )
                    await self._message_queue.put(msg)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket receive error: {e}")
                await asyncio.sleep(1.0)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to keep connection alive."""
        while self._status == TransportStatus.CONNECTED and self._ws:
            try:
                await asyncio.sleep(30.0)  # Heartbeat every 30 seconds

                if self._ws:
                    await self._ws.send(json.dumps({
                        "type": "heartbeat",
                        "device_id": self.device_id,
                        "timestamp": time.time(),
                    }))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  BLUETOOTH TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════


class BluetoothTransport(BaseTransport):
    """
    Nearby: Direct device-to-device via Bluetooth.

    For phone <-> laptop in same room.
    Works without internet connectivity.
    """

    def __init__(
        self,
        device_id: str,
        service_uuid: str = "12345678-1234-5678-1234-567812345678",
        scan_timeout: float = 10.0,
    ):
        super().__init__(device_id)
        self.service_uuid = service_uuid
        self.scan_timeout = scan_timeout
        self._server = None
        self._client_connections: dict[str, Any] = {}

    @property
    def transport_type(self) -> str:
        return "bluetooth"

    def is_available(self) -> bool:
        """Check if Bluetooth library is available."""
        try:
            import bleak  # noqa: F401
            return True
        except ImportError:
            return False

    async def connect(self) -> bool:
        """Start Bluetooth server and begin advertising."""
        if not self.is_available():
            logger.warning("Bluetooth transport unavailable: bleak not installed")
            return False

        self._status = TransportStatus.CONNECTING

        try:
            # Note: Full Bluetooth implementation would require platform-specific
            # code. This is a simplified version showing the interface.
            # Real implementation would use bleak for BLE or pybluez for classic.

            self._status = TransportStatus.CONNECTED
            logger.info("Bluetooth transport ready (stub implementation)")
            return True

        except Exception as e:
            self._status = TransportStatus.ERROR
            self._error_count += 1
            logger.error(f"Bluetooth setup failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Stop Bluetooth server."""
        # Close all client connections
        for device_id, conn in self._client_connections.items():
            try:
                if hasattr(conn, 'disconnect'):
                    await conn.disconnect()
            except Exception:
                pass

        self._client_connections.clear()
        self._status = TransportStatus.DISCONNECTED
        logger.info("Bluetooth disconnected")

    async def send(self, target: str, data: bytes) -> SendResult:
        """Send data to nearby device via Bluetooth."""
        if self._status != TransportStatus.CONNECTED:
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error="Not connected"
            )

        start = time.perf_counter()

        try:
            # Stub: Real implementation would write to BLE characteristic
            # or Bluetooth socket

            if target in self._client_connections:
                # Simulate send
                await asyncio.sleep(0.01)  # Simulate latency
                self._send_count += 1
                latency = (time.perf_counter() - start) * 1000

                return SendResult(
                    success=True,
                    transport_used=self.transport_type,
                    latency_ms=latency
                )
            else:
                return SendResult(
                    success=False,
                    transport_used=self.transport_type,
                    error=f"Device {target} not connected via Bluetooth"
                )

        except Exception as e:
            self._error_count += 1
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error=str(e)
            )

    async def discover_peers(self) -> list[str]:
        """Scan for nearby Bluetooth devices."""
        if not self.is_available():
            return []

        try:
            # Stub: Real implementation would use bleak.BleakScanner
            # to find devices advertising our service UUID

            # Simulate discovery
            discovered = []
            self._connected_peers = set(discovered)
            return discovered

        except Exception as e:
            logger.error(f"Bluetooth scan failed: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════
#  LOCAL NETWORK TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════


class LocalNetworkTransport(BaseTransport):
    """
    LAN: Direct device-to-device over local network.

    Uses mDNS/Bonjour for discovery.
    Works on same WiFi without internet.
    """

    def __init__(
        self,
        device_id: str,
        port: int = 8765,
        service_name: str = "_hololoom._tcp.local.",
    ):
        super().__init__(device_id)
        self.port = port
        self.service_name = service_name
        self._server = None
        self._peer_addresses: dict[str, tuple[str, int]] = {}

    @property
    def transport_type(self) -> str:
        return "local_network"

    def is_available(self) -> bool:
        """Check if required libraries are available."""
        try:
            import zeroconf  # noqa: F401
            return True
        except ImportError:
            return False

    async def connect(self) -> bool:
        """Start local network server and register with mDNS."""
        self._status = TransportStatus.CONNECTING

        try:
            # Start simple TCP server
            self._server = await asyncio.start_server(
                self._handle_connection,
                "0.0.0.0",
                self.port
            )

            # Register with mDNS if available
            if self.is_available():
                # Stub: Real implementation would register with zeroconf
                pass

            self._status = TransportStatus.CONNECTED
            logger.info(f"Local network transport listening on port {self.port}")
            return True

        except Exception as e:
            self._status = TransportStatus.ERROR
            self._error_count += 1
            logger.error(f"Local network setup failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Stop local network server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        self._status = TransportStatus.DISCONNECTED
        logger.info("Local network transport stopped")

    async def send(self, target: str, data: bytes) -> SendResult:
        """Send data to device on local network."""
        if self._status != TransportStatus.CONNECTED:
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error="Not connected"
            )

        if target not in self._peer_addresses:
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error=f"Unknown peer: {target}"
            )

        start = time.perf_counter()
        host, port = self._peer_addresses[target]

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0
            )

            # Send message with length prefix
            length = len(data)
            header = json.dumps({
                "from": self.device_id,
                "length": length,
            }).encode() + b"\n"

            writer.write(header)
            writer.write(data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

            self._send_count += 1
            latency = (time.perf_counter() - start) * 1000

            return SendResult(
                success=True,
                transport_used=self.transport_type,
                latency_ms=latency
            )

        except Exception as e:
            self._error_count += 1
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error=str(e)
            )

    async def discover_peers(self) -> list[str]:
        """Discover peers via mDNS."""
        if not self.is_available():
            return []

        try:
            # Stub: Real implementation would use zeroconf.ServiceBrowser
            discovered = list(self._peer_addresses.keys())
            self._connected_peers = set(discovered)
            return discovered

        except Exception as e:
            logger.error(f"mDNS discovery failed: {e}")
            return []

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming connection."""
        try:
            # Read header
            header_line = await asyncio.wait_for(
                reader.readline(),
                timeout=5.0
            )
            header = json.loads(header_line.decode())
            source = header.get("from", "unknown")
            length = header.get("length", 0)

            # Read data
            data = await asyncio.wait_for(
                reader.readexactly(length),
                timeout=30.0
            )

            # Queue message
            msg = TransportMessage(
                source_device=source,
                data=data,
                transport_type=self.transport_type,
            )
            await self._message_queue.put(msg)

        except Exception as e:
            logger.error(f"Connection handling error: {e}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  COMPOSITE TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════


class CompositeTransport(BaseTransport):
    """
    Use all available transports.

    Tries all in parallel, succeeds if ANY succeeds.
    Provides redundancy and automatic fallback.
    """

    def __init__(
        self,
        device_id: str,
        transports: list[BaseTransport] | None = None,
    ):
        super().__init__(device_id)
        self.transports = transports or []
        self._active_transports: list[BaseTransport] = []

    @property
    def transport_type(self) -> str:
        return "composite"

    def is_available(self) -> bool:
        """At least one transport must be available."""
        return any(t.is_available() for t in self.transports)

    def add_transport(self, transport: BaseTransport) -> None:
        """Add a transport to the composite."""
        self.transports.append(transport)

    async def connect(self) -> bool:
        """Connect all available transports."""
        self._status = TransportStatus.CONNECTING
        self._active_transports = []

        # Try to connect each transport
        results = await asyncio.gather(
            *[t.connect() for t in self.transports if t.is_available()],
            return_exceptions=True
        )

        # Track which succeeded
        for transport, result in zip(
            [t for t in self.transports if t.is_available()],
            results
        ):
            if result is True:
                self._active_transports.append(transport)
                logger.info(f"Composite: {transport.transport_type} connected")

        if self._active_transports:
            self._status = TransportStatus.CONNECTED
            logger.info(
                f"Composite transport: {len(self._active_transports)} "
                f"transports active"
            )
            return True
        else:
            self._status = TransportStatus.ERROR
            logger.warning("Composite transport: no transports connected")
            return False

    async def disconnect(self) -> None:
        """Disconnect all transports."""
        await asyncio.gather(
            *[t.disconnect() for t in self._active_transports],
            return_exceptions=True
        )
        self._active_transports = []
        self._status = TransportStatus.DISCONNECTED
        logger.info("Composite transport disconnected")

    async def send(self, target: str, data: bytes) -> SendResult:
        """
        Send via ALL active transports in parallel.

        Succeeds if ANY succeeds.
        """
        if not self._active_transports:
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error="No active transports"
            )

        start = time.perf_counter()

        # Send via all transports in parallel
        results = await asyncio.gather(
            *[t.send(target, data) for t in self._active_transports],
            return_exceptions=True
        )

        # Check results
        successful = []
        errors = []

        for transport, result in zip(self._active_transports, results):
            if isinstance(result, Exception):
                errors.append(f"{transport.transport_type}: {str(result)}")
            elif result.success:
                successful.append(transport.transport_type)
            else:
                errors.append(f"{transport.transport_type}: {result.error}")

        latency = (time.perf_counter() - start) * 1000

        if successful:
            self._send_count += 1
            return SendResult(
                success=True,
                transport_used=",".join(successful),
                latency_ms=latency
            )
        else:
            self._error_count += 1
            return SendResult(
                success=False,
                transport_used=self.transport_type,
                error="; ".join(errors),
                latency_ms=latency
            )

    async def receive(self) -> AsyncIterator[TransportMessage]:
        """Receive from all transports, deduplicate by message hash."""
        seen_hashes: set[str] = set()

        # Create receive tasks for all active transports
        async def receive_from(transport: BaseTransport):
            async for msg in transport.receive():
                # Simple deduplication by data hash
                msg_hash = f"{msg.source_device}:{hash(msg.data)}"
                if msg_hash not in seen_hashes:
                    seen_hashes.add(msg_hash)
                    # Add composite metadata
                    msg.metadata["original_transport"] = msg.transport_type
                    msg.transport_type = self.transport_type
                    await self._message_queue.put(msg)

        # Start receive loops for all transports
        tasks = [
            asyncio.create_task(receive_from(t))
            for t in self._active_transports
        ]

        try:
            async for msg in super().receive():
                self._receive_count += 1
                yield msg
        finally:
            for task in tasks:
                task.cancel()

    async def discover_peers(self) -> list[str]:
        """Discover peers from all transports."""
        all_peers: set[str] = set()

        results = await asyncio.gather(
            *[t.discover_peers() for t in self._active_transports],
            return_exceptions=True
        )

        for result in results:
            if isinstance(result, list):
                all_peers.update(result)

        self._connected_peers = all_peers
        return list(all_peers)

    def get_stats(self) -> dict[str, Any]:
        """Get composite transport statistics."""
        stats = super().get_stats()
        stats["active_transports"] = [
            t.get_stats() for t in self._active_transports
        ]
        stats["inactive_transports"] = [
            t.transport_type for t in self.transports
            if t not in self._active_transports
        ]
        return stats


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSPORT FACTORY
# ═══════════════════════════════════════════════════════════════════════════


def create_default_transport(
    device_id: str,
    relay_url: str = "wss://relay.hololoom.net",
) -> CompositeTransport:
    """
    Create a composite transport with all available transports.

    Usage:
        transport = create_default_transport(my_device_id)
        await transport.connect()
        result = await transport.send(target_id, data)
    """
    transports = [
        WebSocketTransport(device_id, relay_url=relay_url),
        LocalNetworkTransport(device_id),
        BluetoothTransport(device_id),
    ]

    return CompositeTransport(device_id, transports)


def create_websocket_only_transport(
    device_id: str,
    relay_url: str = "wss://relay.hololoom.net",
) -> WebSocketTransport:
    """Create WebSocket-only transport for internet-based sync."""
    return WebSocketTransport(device_id, relay_url=relay_url)


def create_local_only_transport(device_id: str) -> CompositeTransport:
    """Create transport for local network only (no internet)."""
    transports = [
        LocalNetworkTransport(device_id),
        BluetoothTransport(device_id),
    ]
    return CompositeTransport(device_id, transports)

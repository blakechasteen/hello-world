"""
HTTP Transport - Simple, debuggable transport using HTTP/JSON.

Why HTTP first (not gRPC):
- Faster to implement (aiohttp is simpler than grpcio)
- Easier debugging (curl, browser tools)
- Can upgrade to gRPC later for performance
- Same interface, just swap transport
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from ..types import FederationNode

from . import (
    BaseTransport,
    MessageHandler,
    MessageType,
    TransportMessage,
    TransportResponse,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  HTTP TRANSPORT - aiohttp-based implementation
# ═══════════════════════════════════════════════════════════════════════════


class HTTPTransport(BaseTransport):
    """
    HTTP/JSON transport implementation.

    Uses aiohttp for:
    - Async HTTP client for outgoing requests
    - Async HTTP server for incoming requests
    - Connection pooling for efficiency

    Message format: JSON over HTTP POST
    Endpoint: POST /federation/message
    """

    def __init__(
        self,
        node_id: str,
        *,
        connection_timeout: float = 5.0,
        read_timeout: float = 30.0,
        max_connections: int = 100,
    ):
        super().__init__(node_id)

        self._connection_timeout = connection_timeout
        self._read_timeout = read_timeout
        self._max_connections = max_connections

        # aiohttp objects (created lazily)
        self._session: Optional[Any] = None  # aiohttp.ClientSession
        self._app: Optional[Any] = None       # aiohttp.web.Application
        self._runner: Optional[Any] = None    # aiohttp.web.AppRunner
        self._site: Optional[Any] = None      # aiohttp.web.TCPSite

    async def _get_session(self) -> Any:
        """Get or create the HTTP client session."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp

                timeout = aiohttp.ClientTimeout(
                    total=self._read_timeout,
                    connect=self._connection_timeout,
                )
                connector = aiohttp.TCPConnector(
                    limit=self._max_connections,
                    enable_cleanup_closed=True,
                )
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                )
            except ImportError:
                logger.error("aiohttp not installed. Run: pip install aiohttp")
                raise ImportError(
                    "aiohttp required for HTTP transport. "
                    "Install with: pip install aiohttp"
                )

        return self._session

    async def send(
        self,
        node: FederationNode,
        message: TransportMessage,
        timeout_ms: int = 5000,
    ) -> TransportResponse:
        """
        Send message to a single node via HTTP POST.

        Args:
            node: Target node
            message: Message to send
            timeout_ms: Timeout in milliseconds

        Returns:
            TransportResponse with success status
        """
        start_time = time.perf_counter()

        try:
            import aiohttp

            session = await self._get_session()

            # Build URL from node endpoint
            url = f"http://{node.endpoint}/federation/message"

            # Serialize message
            data = message.to_dict()

            # Send with custom timeout
            timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)

            async with session.post(
                url,
                json=data,
                timeout=timeout,
            ) as response:
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Track metrics
                self._messages_sent += 1
                self._bytes_sent += len(json.dumps(data))

                if response.status == 200:
                    response_data = await response.json()
                    self._bytes_received += len(json.dumps(response_data))
                    self._messages_received += 1

                    return TransportResponse(
                        success=True,
                        message=TransportMessage.from_dict(response_data),
                        latency_ms=latency_ms,
                    )
                else:
                    error_text = await response.text()
                    return TransportResponse(
                        success=False,
                        error=f"HTTP {response.status}: {error_text}",
                        latency_ms=latency_ms,
                    )

        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return TransportResponse(
                success=False,
                error=f"Timeout after {timeout_ms}ms",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Send failed to {node.endpoint}: {e}")
            return TransportResponse(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    async def start_server(
        self,
        handler: MessageHandler,
        host: str = "0.0.0.0",
        port: int = 9000,
    ) -> None:
        """
        Start HTTP server for incoming messages.

        Args:
            handler: Callback for incoming messages
            host: Host to bind to
            port: Port to listen on
        """
        if self._running:
            logger.warning("Server already running")
            return

        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp not installed. Run: pip install aiohttp")
            raise ImportError(
                "aiohttp required for HTTP transport. "
                "Install with: pip install aiohttp"
            )

        self._handler = handler

        # Create aiohttp application
        self._app = web.Application()
        self._app.router.add_post("/federation/message", self._handle_request)
        self._app.router.add_get("/health", self._handle_health)

        # Start server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, host, port)
        await self._site.start()

        self._running = True
        logger.info(f"HTTP transport started on {host}:{port}")

    async def stop_server(self) -> None:
        """Stop the HTTP server gracefully."""
        if not self._running:
            return

        logger.info("Stopping HTTP transport...")

        # Close site
        if self._site:
            await self._site.stop()
            self._site = None

        # Cleanup runner
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        # Close client session
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        self._running = False
        self._app = None
        logger.info("HTTP transport stopped")

    async def _handle_request(self, request: Any) -> Any:
        """Handle incoming HTTP request."""
        from aiohttp import web

        try:
            # Parse message
            data = await request.json()
            message = TransportMessage.from_dict(data)

            # Track metrics
            self._messages_received += 1
            self._bytes_received += len(json.dumps(data))

            # Call handler
            if self._handler:
                response_message = self._handler(message)

                # Serialize response
                response_data = response_message.to_dict()
                self._messages_sent += 1
                self._bytes_sent += len(json.dumps(response_data))

                return web.json_response(response_data)
            else:
                return web.json_response(
                    {"error": "No handler registered"},
                    status=500,
                )

        except json.JSONDecodeError as e:
            return web.json_response(
                {"error": f"Invalid JSON: {e}"},
                status=400,
            )
        except KeyError as e:
            return web.json_response(
                {"error": f"Missing field: {e}"},
                status=400,
            )
        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500,
            )

    async def _handle_health(self, request: Any) -> Any:
        """Health check endpoint."""
        from aiohttp import web

        return web.json_response({
            "status": "ok",
            "node_id": self._node_id,
            "metrics": self.get_metrics(),
        })


# ═══════════════════════════════════════════════════════════════════════════
#  MOCK TRANSPORT - For testing without network
# ═══════════════════════════════════════════════════════════════════════════


class MockTransport(BaseTransport):
    """
    Mock transport for testing.

    All messages are delivered locally without network I/O.
    Useful for unit tests and development.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id)
        self._other_nodes: Dict[str, MockTransport] = {}
        self._latency_ms: float = 1.0  # Simulated latency

    def connect_to(self, other: "MockTransport") -> None:
        """Connect this mock to another for direct message delivery."""
        self._other_nodes[other.node_id] = other
        other._other_nodes[self.node_id] = self

    def set_latency(self, latency_ms: float) -> None:
        """Set simulated latency."""
        self._latency_ms = latency_ms

    async def send(
        self,
        node: FederationNode,
        message: TransportMessage,
        timeout_ms: int = 5000,
    ) -> TransportResponse:
        """Send message directly to connected mock transport."""
        start_time = time.perf_counter()

        # Simulate network latency
        await asyncio.sleep(self._latency_ms / 1000)

        # Find target transport
        target = self._other_nodes.get(node.node_id)
        if not target:
            return TransportResponse(
                success=False,
                error=f"No connection to {node.node_id}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Deliver message
        self._messages_sent += 1

        if target._handler:
            try:
                response_message = target._handler(message)
                target._messages_received += 1
                self._messages_received += 1

                return TransportResponse(
                    success=True,
                    message=response_message,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )
            except Exception as e:
                return TransportResponse(
                    success=False,
                    error=str(e),
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )
        else:
            return TransportResponse(
                success=False,
                error="Target has no handler",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

    async def start_server(
        self,
        handler: MessageHandler,
        host: str = "0.0.0.0",
        port: int = 9000,
    ) -> None:
        """Register handler (no actual server needed for mock)."""
        self._handler = handler
        self._running = True
        logger.info(f"Mock transport started for {self._node_id}")

    async def stop_server(self) -> None:
        """Stop mock transport."""
        self._handler = None
        self._running = False
        logger.info(f"Mock transport stopped for {self._node_id}")


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "HTTPTransport",
    "MockTransport",
]

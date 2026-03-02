"""
MCP Client
==========

WebSocket client for communicating with Claude Code MCP Server in VS Code.

Implements Model Context Protocol (MCP) over WebSocket for bidirectional
communication between Matrix ChatOps and VS Code extension.
"""

import asyncio
import json
import logging
import uuid
from typing import Optional, Dict, Any, Callable, List
from dataclasses import asdict

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("Warning: websockets not available. Install with: pip install websockets")

from .protocol import MCPRequest, MCPResponse, MCPNotification, MCPToolCall


logger = logging.getLogger(__name__)


class MCPClient:
    """
    MCP Protocol Client over WebSocket

    Connects to VS Code MCP Server and enables tool invocation and
    notification handling.
    """

    def __init__(
        self,
        server_url: str = "ws://localhost:9001",
        reconnect: bool = True,
        reconnect_delay: float = 5.0
    ):
        """
        Initialize MCP Client

        Args:
            server_url: WebSocket URL of MCP server
            reconnect: Auto-reconnect on disconnect
            reconnect_delay: Delay between reconnection attempts (seconds)
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets package required. Install with: pip install websockets"
            )

        self.server_url = server_url
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay

        self.ws: Optional[WebSocketClientProtocol] = None
        self.connected = False

        # Pending requests waiting for responses
        self.pending_requests: Dict[str, asyncio.Future] = {}

        # Notification handlers
        self.notification_handlers: Dict[str, List[Callable]] = {}

        # Background tasks
        self.receive_task: Optional[asyncio.Task] = None
        self.reconnect_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """
        Connect to MCP server

        Returns:
            True if connected successfully
        """
        try:
            self.ws = await websockets.connect(self.server_url)
            self.connected = True

            # Start receiving messages
            self.receive_task = asyncio.create_task(self._receive_loop())

            # Initialize connection
            await self._send_initialize()

            logger.info(f"Connected to MCP server: {self.server_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self.connected = False

            # Start reconnection loop if enabled
            if self.reconnect and not self.reconnect_task:
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())

            return False

    async def disconnect(self):
        """Disconnect from MCP server"""
        self.reconnect = False  # Disable auto-reconnect

        # Cancel background tasks
        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass

        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.ws:
            await self.ws.close()
            self.ws = None

        self.connected = False
        logger.info("Disconnected from MCP server")

    async def _send_initialize(self):
        """Send initialization request"""
        response = await self.call_method("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "hololoom-chatops",
                "version": "1.0.0"
            }
        })

        logger.info(f"Initialized MCP connection: {response}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool

        Args:
            tool_name: Name of tool (e.g., "code/query")
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            ConnectionError: If not connected
            RuntimeError: If tool call fails
        """
        if not self.connected:
            raise ConnectionError("Not connected to MCP server")

        tool_call = MCPToolCall(
            name=tool_name,
            arguments=arguments,
            id=str(uuid.uuid4())
        )

        result = await self.call_method("tools/call", {
            "name": tool_call.name,
            "arguments": tool_call.arguments
        })

        return result

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools

        Returns:
            List of tool definitions
        """
        if not self.connected:
            raise ConnectionError("Not connected to MCP server")

        result = await self.call_method("tools/list", {})
        return result.get("tools", [])

    async def call_method(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call MCP method and wait for response

        Args:
            method: Method name
            params: Method parameters

        Returns:
            Response result

        Raises:
            ConnectionError: If not connected
            RuntimeError: If method call fails
        """
        if not self.connected or not self.ws:
            raise ConnectionError("Not connected to MCP server")

        request_id = str(uuid.uuid4())

        request = MCPRequest(
            jsonrpc="2.0",
            id=request_id,
            method=method,
            params=params
        )

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        try:
            # Send request
            await self.ws.send(json.dumps(asdict(request)))

            # Wait for response (with timeout)
            response = await asyncio.wait_for(future, timeout=60.0)

            if response.get("error"):
                error = response["error"]
                raise RuntimeError(f"MCP error: {error.get('message', 'Unknown error')}")

            return response.get("result", {})

        except asyncio.TimeoutError:
            raise RuntimeError(f"MCP method '{method}' timed out")

        finally:
            # Clean up pending request
            self.pending_requests.pop(request_id, None)

    def on_notification(self, method: str, handler: Callable):
        """
        Register notification handler

        Args:
            method: Notification method to handle
            handler: Async function to call when notification received
        """
        if method not in self.notification_handlers:
            self.notification_handlers[method] = []

        self.notification_handlers[method].append(handler)

    async def ping(self) -> bool:
        """
        Ping server to check connection

        Returns:
            True if server responds
        """
        try:
            result = await self.call_method("ping", {})
            return result.get("pong", False)
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False

    # ========================================================================
    # Internal Methods
    # ========================================================================

    async def _receive_loop(self):
        """Background task to receive messages"""
        try:
            async for message in self.ws:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("MCP WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            self.connected = False

    async def _handle_message(self, message: str):
        """Handle incoming message"""
        try:
            data = json.loads(message)

            # Check if response or notification
            if "id" in data and data["id"] in self.pending_requests:
                # Response to pending request
                future = self.pending_requests[data["id"]]
                if not future.done():
                    future.set_result(data)

            elif "method" in data and "id" not in data:
                # Notification
                await self._handle_notification(data)

            else:
                logger.warning(f"Unknown message format: {data}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_notification(self, data: Dict[str, Any]):
        """Handle MCP notification"""
        method = data.get("method", "")
        params = data.get("params", {})

        logger.info(f"Received notification: {method}")

        # Call registered handlers
        if method in self.notification_handlers:
            for handler in self.notification_handlers[method]:
                try:
                    await handler(params)
                except Exception as e:
                    logger.error(f"Error in notification handler for '{method}': {e}")

    async def _reconnect_loop(self):
        """Background task to reconnect"""
        while self.reconnect and not self.connected:
            logger.info(f"Attempting to reconnect in {self.reconnect_delay}s...")
            await asyncio.sleep(self.reconnect_delay)

            if await self.connect():
                logger.info("Reconnected successfully")
                break

    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            "connected": self.connected,
            "server_url": self.server_url,
            "pending_requests": len(self.pending_requests),
            "notification_handlers": len(self.notification_handlers)
        }

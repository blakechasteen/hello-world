"""
Claude Code Department
======================

Main department implementation for Matrix → VS Code integration.

Provides standardized HoloLoom department interface for code-related
operations via MCP protocol.
"""

import logging
from datetime import datetime
from typing import Any

from ..protocol import DepartmentProtocol
from .mcp_client import MCPClient
from .protocol import ClaudeCodeRequest, ClaudeCodeResponse, CodeAction

logger = logging.getLogger(__name__)


class ClaudeCodeDepartment(DepartmentProtocol):
    """
    Claude Code Department

    Bridges Matrix ChatOps to VS Code extension via MCP protocol.

    Features:
    - Code queries with HoloLoom reasoning
    - Refactoring assistance
    - Code explanation
    - Test generation
    - Bug fixing suggestions
    - Context retrieval

    Usage:
        dept = ClaudeCodeDepartment()
        await dept.start()

        result = await dept.process({
            "action": "query",
            "question": "How does this work?"
        })

        await dept.stop()
    """

    def __init__(
        self,
        mcp_server_url: str = "ws://localhost:9001",
        auto_reconnect: bool = True
    ):
        """
        Initialize Claude Code Department

        Args:
            mcp_server_url: WebSocket URL of VS Code MCP server
            auto_reconnect: Automatically reconnect on disconnect
        """
        self.mcp_server_url = mcp_server_url
        self.auto_reconnect = auto_reconnect

        self.mcp_client: MCPClient | None = None
        self.started = False

        # Statistics
        self.stats = {
            "requests_processed": 0,
            "errors": 0,
            "last_request": None,
            "start_time": None
        }

    async def start(self):
        """Start department (connect to MCP server)"""
        if self.started:
            logger.warning("Claude Code Department already started")
            return

        logger.info(f"Starting Claude Code Department (MCP: {self.mcp_server_url})")

        # Create MCP client
        self.mcp_client = MCPClient(
            server_url=self.mcp_server_url,
            reconnect=self.auto_reconnect
        )

        # Register notification handlers
        self.mcp_client.on_notification("file/changed", self._handle_file_change)
        self.mcp_client.on_notification("diagnostics/updated", self._handle_diagnostics)

        # Connect
        connected = await self.mcp_client.connect()
        if not connected:
            logger.error("Failed to connect to MCP server")
            if not self.auto_reconnect:
                raise ConnectionError(f"Cannot connect to MCP server at {self.mcp_server_url}")

        self.started = True
        self.stats["start_time"] = datetime.now()

        logger.info("Claude Code Department started successfully")

    async def stop(self):
        """Stop department (disconnect from MCP server)"""
        if not self.started:
            return

        logger.info("Stopping Claude Code Department")

        if self.mcp_client:
            await self.mcp_client.disconnect()
            self.mcp_client = None

        self.started = False
        logger.info("Claude Code Department stopped")

    async def process(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Process department request

        Args:
            request: Request dictionary with:
                - action: CodeAction (query, refactor, explain, test, fix, context)
                - params: Action-specific parameters
                - user_id: (optional) Matrix user ID
                - room_id: (optional) Matrix room ID

        Returns:
            Response dictionary with:
                - success: bool
                - result: str (formatted result)
                - metadata: dict (confidence, duration, etc.)
                - error: (optional) str

        Raises:
            ValueError: If request is invalid
            ConnectionError: If not connected to MCP server
        """
        if not self.started:
            return {
                "success": False,
                "error": "Department not started. Call start() first."
            }

        if not self.mcp_client or not self.mcp_client.connected:
            return {
                "success": False,
                "error": "Not connected to VS Code MCP server. Make sure VS Code is running with Squad extension."
            }

        # Parse request
        try:
            code_request = self._parse_request(request)
        except ValueError as e:
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Invalid request: {str(e)}"
            }

        # Process action
        try:
            start_time = datetime.now()

            response = await self._process_action(code_request)

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["last_request"] = datetime.now()

            # Add metadata
            if response.success and response.metadata:
                response.metadata["department"] = "claude_code"
                response.metadata["duration_ms"] = duration_ms
                response.metadata["mcp_server"] = self.mcp_server_url

            return self._response_to_dict(response)

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.stats["errors"] += 1

            return {
                "success": False,
                "error": f"Processing error: {str(e)}"
            }

    async def health_check(self) -> bool:
        """
        Check department health

        Returns:
            True if healthy and connected
        """
        if not self.started or not self.mcp_client:
            return False

        try:
            return await self.mcp_client.ping()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _parse_request(self, request: dict[str, Any]) -> ClaudeCodeRequest:
        """Parse raw request into ClaudeCodeRequest"""
        action_str = request.get("action")
        if not action_str:
            raise ValueError("Missing 'action' field")

        try:
            action = CodeAction(action_str)
        except ValueError:
            raise ValueError(f"Invalid action: {action_str}")

        return ClaudeCodeRequest(
            action=action,
            params=request.get("params", {}),
            user_id=request.get("user_id"),
            room_id=request.get("room_id")
        )

    async def _process_action(self, request: ClaudeCodeRequest) -> ClaudeCodeResponse:
        """Process code action via MCP"""
        action_map = {
            CodeAction.QUERY: "code/query",
            CodeAction.REFACTOR: "code/refactor",
            CodeAction.EXPLAIN: "code/explain",
            CodeAction.TEST: "code/test",
            CodeAction.FIX: "code/fix",
            CodeAction.CONTEXT: "code/context"
        }

        tool_name = action_map[request.action]

        try:
            # Call MCP tool
            result = await self.mcp_client.call_tool(tool_name, request.params)

            # Extract text from MCP response
            content = result.get("content", [])
            if content and len(content) > 0:
                text = content[0].get("text", "")
            else:
                text = str(result)

            return ClaudeCodeResponse(
                success=True,
                result=text,
                metadata={
                    "action": request.action.value,
                    "tool": tool_name
                }
            )

        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return ClaudeCodeResponse(
                success=False,
                error=f"MCP error: {str(e)}"
            )

    def _response_to_dict(self, response: ClaudeCodeResponse) -> dict[str, Any]:
        """Convert response to dictionary"""
        result = {
            "success": response.success
        }

        if response.result:
            result["result"] = response.result

        if response.metadata:
            result["metadata"] = response.metadata

        if response.error:
            result["error"] = response.error

        return result

    async def _handle_file_change(self, params: dict[str, Any]):
        """Handle file change notification from VS Code"""
        logger.info(f"File changed in VS Code: {params.get('uri')}")
        # Could trigger Matrix notifications here

    async def _handle_diagnostics(self, params: dict[str, Any]):
        """Handle diagnostics notification from VS Code"""
        uri = params.get("uri", "")
        diagnostics = params.get("diagnostics", [])

        logger.info(f"Diagnostics updated for {uri}: {len(diagnostics)} issues")
        # Could trigger Matrix alerts for critical issues

    def get_stats(self) -> dict[str, Any]:
        """Get department statistics"""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()

        return {
            **self.stats,
            "uptime_seconds": uptime,
            "connected": self.mcp_client.connected if self.mcp_client else False,
            "server_url": self.mcp_server_url
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_claude_code_department(
    mcp_server_url: str = "ws://localhost:9001",
    auto_reconnect: bool = True
) -> ClaudeCodeDepartment:
    """
    Factory function to create Claude Code Department

    Args:
        mcp_server_url: WebSocket URL of VS Code MCP server
        auto_reconnect: Automatically reconnect on disconnect

    Returns:
        Configured ClaudeCodeDepartment instance
    """
    return ClaudeCodeDepartment(
        mcp_server_url=mcp_server_url,
        auto_reconnect=auto_reconnect
    )

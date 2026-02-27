#!/usr/bin/env python3
"""
MCP Server for Infrastructure Department

Part 2: Foundation Infrastructure (Days 6-10)

Exposes SQL backend via Model Context Protocol (MCP).
Handles requests from Context Department and returns QueryResults.

Features:
- query_sql tool for precision data queries
- Session management and tracking
- Error escalation to Context/Orchestration
- Performance monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import hashlib

from HoloLoom.infrastructure.sql import SQLBackend, SQLConfig, create_sql_backend
from HoloLoom.infrastructure.mcp.protocol import (
    MCPRequest,
    MCPResponse,
    MCPError,
    Session,
    RequestType,
    ResponseStatus,
    ErrorCode,
    ToolName,
    AVAILABLE_TOOLS,
    QUERY_SQL_TOOL,
    create_success_response,
    create_error_response,
    validate_query_sql_params,
    generate_session_id
)


logger = logging.getLogger(__name__)


# ============================================================================
# MCP Server
# ============================================================================

class MCPServer:
    """
    MCP Server for Infrastructure Department

    Exposes SQL backend via Model Context Protocol
    """

    def __init__(self, sql_backend: SQLBackend):
        """
        Initialize MCP server

        Args:
            sql_backend: Connected SQL backend
        """
        self.sql_backend = sql_backend
        self.sessions: Dict[str, Session] = {}
        self.request_count = 0
        self.error_count = 0
        self._cleanup_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        """Async context manager entry"""
        # Start session cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
        logger.info("MCP server started")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("MCP server stopped")

    # ========================================================================
    # Request Handling
    # ========================================================================

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle MCP request from Context Department

        Args:
            request: MCP request

        Returns:
            MCP response
        """
        self.request_count += 1

        # Update session
        session = self._get_or_create_session(request.session_id)
        session.update()

        logger.info(
            f"Handling request {request.request_id} "
            f"(session: {request.session_id}, type: {request.request_type})"
        )

        try:
            # Route by request type
            if request.request_type == RequestType.TOOL_LIST:
                return await self._handle_tool_list(request)
            elif request.request_type == RequestType.TOOL_CALL:
                return await self._handle_tool_call(request)
            elif request.request_type == RequestType.HEALTH_CHECK:
                return await self._handle_health_check(request)
            else:
                return create_error_response(
                    request.request_id,
                    request.session_id,
                    MCPError(
                        code=ErrorCode.INVALID_REQUEST,
                        message=f"Unknown request type: {request.request_type}"
                    )
                )

        except Exception as e:
            logger.error(f"Request {request.request_id} crashed: {e}", exc_info=True)
            self.error_count += 1

            # Escalate unexpected errors to Context Department
            return create_error_response(
                request.request_id,
                request.session_id,
                MCPError(
                    code=ErrorCode.UNKNOWN_ERROR,
                    message=f"Internal server error: {str(e)}",
                    escalate_to="context"
                )
            )

    async def _handle_tool_list(self, request: MCPRequest) -> MCPResponse:
        """Handle tool list request"""
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                        "default": p.default
                    }
                    for p in tool.parameters
                ],
                "returns": tool.returns
            }
            for tool in AVAILABLE_TOOLS
        ]

        return create_success_response(
            request.request_id,
            request.session_id,
            {"tools": tools, "count": len(tools)},
            metadata={"mcp_version": "1.0.0"}
        )

    async def _handle_tool_call(self, request: MCPRequest) -> MCPResponse:
        """Handle tool call request"""
        if not request.tool_name:
            return create_error_response(
                request.request_id,
                request.session_id,
                MCPError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="Missing tool_name for tool_call request"
                )
            )

        # Route to tool handler
        if request.tool_name == ToolName.QUERY_SQL:
            return await self._handle_query_sql(request)
        else:
            return create_error_response(
                request.request_id,
                request.session_id,
                MCPError(
                    code=ErrorCode.TOOL_NOT_FOUND,
                    message=f"Tool not found: {request.tool_name}",
                    details={"available_tools": [t.name for t in AVAILABLE_TOOLS]}
                )
            )

    async def _handle_health_check(self, request: MCPRequest) -> MCPResponse:
        """Handle health check request"""
        return create_success_response(
            request.request_id,
            request.session_id,
            {
                "status": "healthy",
                "backend_type": self.sql_backend.backend_type,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "session_count": len(self.sessions)
            }
        )

    # ========================================================================
    # Tool Handlers
    # ========================================================================

    async def _handle_query_sql(self, request: MCPRequest) -> MCPResponse:
        """
        Handle query_sql tool call

        Parameters:
            - query: SQL query string (SELECT only)
            - params: Query parameters (optional)
            - timeout_ms: Query timeout in milliseconds (optional)

        Returns:
            QueryResult with rows, row_count, latency_ms
        """
        params = request.parameters

        # Validate parameters
        validation_error = validate_query_sql_params(params)
        if validation_error:
            return create_error_response(
                request.request_id,
                request.session_id,
                validation_error
            )

        # Extract parameters
        query = params["query"]
        query_params = params.get("params", [])
        timeout_ms = params.get("timeout_ms", 30000)

        # Execute query with timeout
        try:
            result = await asyncio.wait_for(
                self.sql_backend.execute_query(query, query_params),
                timeout=timeout_ms / 1000.0
            )

            if not result.success:
                # Query failed - escalate if critical
                error = MCPError(
                    code=ErrorCode.SQL_ERROR,
                    message=result.error or "Unknown SQL error",
                    details={
                        "query": query,
                        "backend": result.backend
                    }
                )

                # Escalate connection errors to Context Department
                if "connection" in (result.error or "").lower():
                    error.escalate_to = "context"

                return create_error_response(
                    request.request_id,
                    request.session_id,
                    error
                )

            # Success - return query result
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]

            return create_success_response(
                request.request_id,
                request.session_id,
                {
                    "rows": result.rows,
                    "row_count": result.row_count,
                    "latency_ms": result.latency_ms,
                    "backend": result.backend
                },
                metadata={
                    "query_hash": query_hash,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        except asyncio.TimeoutError:
            # Query timeout - escalate to Context for rerouting
            return create_error_response(
                request.request_id,
                request.session_id,
                MCPError(
                    code=ErrorCode.TIMEOUT,
                    message=f"Query timeout after {timeout_ms}ms",
                    details={"query": query, "timeout_ms": timeout_ms},
                    escalate_to="context"
                )
            )

    # ========================================================================
    # Session Management
    # ========================================================================

    def _get_or_create_session(self, session_id: str) -> Session:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
            logger.info(f"Created new session: {session_id}")

        return self.sessions[session_id]

    async def _cleanup_sessions(self):
        """Background task to clean up stale sessions"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                now = datetime.utcnow()
                stale_sessions = []

                for session_id, session in self.sessions.items():
                    # Remove sessions inactive for >1 hour
                    if now - session.last_request_at > timedelta(hours=1):
                        stale_sessions.append(session_id)

                for session_id in stale_sessions:
                    del self.sessions[session_id]
                    logger.info(f"Cleaned up stale session: {session_id}")

                if stale_sessions:
                    logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}", exc_info=True)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        active_sessions = sum(
            1 for s in self.sessions.values()
            if datetime.utcnow() - s.last_request_at < timedelta(minutes=5)
        )

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1)
        }


# ============================================================================
# Factory Function
# ============================================================================

async def create_mcp_server(
    sql_config: Optional[SQLConfig] = None
) -> MCPServer:
    """
    Create MCP server with SQL backend

    Args:
        sql_config: SQL configuration (uses defaults if None)

    Returns:
        MCPServer instance (use as async context manager)
    """
    if sql_config is None:
        sql_config = SQLConfig()

    # Create SQL backend
    sql_backend = create_sql_backend(sql_config)
    await sql_backend.connect()

    # Create MCP server
    return MCPServer(sql_backend)

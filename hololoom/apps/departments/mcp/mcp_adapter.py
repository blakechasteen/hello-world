#!/usr/bin/env python3
"""
MCP Adapter — Wraps a BaseDepartment as an MCP endpoint.

Takes an MCPRequest, translates it to a DepartmentRequest, calls the
department, and wraps the result as an MCPResponse. Handles tool_list
and health_check request types natively.

This is the bridge between the wire protocol (MCPRequest/MCPResponse)
and the domain protocol (DepartmentRequest/DepartmentResponse).
"""

from __future__ import annotations

import logging
import time

from hololoom.alignment.mcp_department_registry import (
    DepartmentCharter,
    MCPTool,
)
from hololoom.infrastructure.mcp.protocol import (
    ErrorCode,
    MCPError,
    MCPRequest,
    MCPResponse,
    RequestType,
    create_error_response,
    create_success_response,
)
from hololoom.protocols.department import (
    DepartmentRequest,
    DepartmentResponse,
)

from ..base import BaseDepartment

logger = logging.getLogger(__name__)


class DepartmentMCPAdapter:
    """
    Wraps a BaseDepartment as an MCP-addressable endpoint.

    Responsibilities:
    - Translate MCPRequest → DepartmentRequest → MCPResponse
    - Expose charter tools via tool_list
    - Report health via health_check
    - Track per-tool metrics

    The adapter does NOT enforce permissions — that's the router's job.
    It trusts that if a request reaches it, permission was already checked.
    """

    def __init__(self, department: BaseDepartment, charter: DepartmentCharter):
        self.department = department
        self.charter = charter
        self._tool_map: dict[str, MCPTool] = {
            tool.name: tool for tool in charter.tools
        }

    @property
    def name(self) -> str:
        return self.charter.name

    async def handle(self, request: MCPRequest) -> MCPResponse:
        """
        Handle an MCP request by dispatching to the appropriate handler.

        Supported request types:
        - TOOL_CALL: translate to DepartmentRequest, call execute()
        - TOOL_LIST: return charter tools
        - HEALTH_CHECK: call department.health_check()
        """
        match request.request_type:
            case RequestType.TOOL_CALL:
                return await self._handle_tool_call(request)
            case RequestType.TOOL_LIST:
                return self._handle_tool_list(request)
            case RequestType.HEALTH_CHECK:
                return await self._handle_health_check(request)
            case _:
                error = MCPError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"Unknown request type: {request.request_type}",
                )
                return create_error_response(
                    request.request_id, request.session_id, error
                )

    async def _handle_tool_call(self, request: MCPRequest) -> MCPResponse:
        """Translate MCPRequest to DepartmentRequest, execute, wrap result."""
        # Validate tool exists
        if request.tool_name not in self._tool_map:
            error = MCPError(
                code=ErrorCode.TOOL_NOT_FOUND,
                message=f"Tool '{request.tool_name}' not found in {self.name}",
                details={"available_tools": list(self._tool_map.keys())},
            )
            return create_error_response(
                request.request_id, request.session_id, error
            )

        # Build DepartmentRequest
        dept_request = DepartmentRequest(
            task_type=request.tool_name,
            parameters=request.parameters,
            context={"session_id": request.session_id, **request.metadata},
        )

        # Execute
        start = time.monotonic()
        try:
            dept_response: DepartmentResponse = await self.department.execute(
                dept_request
            )
        except Exception as e:
            logger.error(f"{self.name}: tool_call failed: {e}")
            self.department._record_error(e)
            error = MCPError(
                code=ErrorCode.UNKNOWN_ERROR,
                message=str(e),
                escalate_to="Orchestration",
            )
            return create_error_response(
                request.request_id, request.session_id, error
            )

        latency_ms = (time.monotonic() - start) * 1000
        self.department._record_request(
            success=dept_response.error is None,
            latency_ms=latency_ms,
            confidence=dept_response.confidence.score,
        )

        # Wrap as MCPResponse
        return create_success_response(
            request_id=request.request_id,
            session_id=request.session_id,
            result={
                "result": dept_response.result,
                "confidence": dept_response.confidence.score,
                "confidence_level": dept_response.confidence.level.value,
                "task_id": str(dept_response.task_id),
                "latency_ms": latency_ms,
            },
            metadata={
                "department": self.name,
                "tool": request.tool_name,
            },
        )

    def _handle_tool_list(self, request: MCPRequest) -> MCPResponse:
        """Return the charter's tool definitions."""
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "returns": tool.returns,
            }
            for tool in self.charter.tools
        ]
        return create_success_response(
            request_id=request.request_id,
            session_id=request.session_id,
            result={"tools": tools, "department": self.name},
        )

    async def _handle_health_check(self, request: MCPRequest) -> MCPResponse:
        """Delegate to department.health_check()."""
        try:
            health = await self.department.health_check()
            return create_success_response(
                request_id=request.request_id,
                session_id=request.session_id,
                result=health,
            )
        except Exception as e:
            error = MCPError(
                code=ErrorCode.UNKNOWN_ERROR,
                message=f"Health check failed: {e}",
            )
            return create_error_response(
                request.request_id, request.session_id, error
            )

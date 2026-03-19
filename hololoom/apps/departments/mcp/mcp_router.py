#!/usr/bin/env python3
"""
MCP Router — In-process department federation.

Routes MCPRequests to the correct DepartmentMCPAdapter, enforcing
charter permissions and propagating session context. Supports
inter-department messaging (e.g., Verification requesting a re-run
from MasterWeaver).

This is the "Orchestration" runtime — the thing that makes departments
talk to each other. Starts in-process; can graduate to network later
by swapping adapters for WebSocket clients (mcp_client.py already exists).
"""

from __future__ import annotations

import logging
from typing import Any

from hololoom.alignment.mcp_department_registry import (
    PermissionLevel,
    PermissionMode,
)
from hololoom.infrastructure.mcp.protocol import (
    ErrorCode,
    MCPError,
    MCPRequest,
    MCPResponse,
    RequestType,
    ResponseStatus,
    create_error_response,
    generate_request_id,
)

from .mcp_adapter import DepartmentMCPAdapter

logger = logging.getLogger(__name__)


class MCPRouter:
    """
    In-process MCP router for department federation.

    Responsibilities:
    - Register/discover department adapters
    - Route MCPRequests to the correct adapter
    - Enforce charter permissions (HARD/SOFT)
    - Enable inter-department tool calls
    - Track audit log for cross-department requests
    """

    def __init__(self):
        self._adapters: dict[str, DepartmentMCPAdapter] = {}
        self._audit_log: list[dict[str, Any]] = []

    def register(self, adapter: DepartmentMCPAdapter) -> None:
        """Register a department adapter for routing."""
        if adapter.name in self._adapters:
            logger.warning(f"Replacing existing adapter for {adapter.name}")
        self._adapters[adapter.name] = adapter
        logger.info(f"Router: registered {adapter.name}")

    def unregister(self, name: str) -> None:
        """Remove a department adapter."""
        self._adapters.pop(name, None)

    @property
    def departments(self) -> list[str]:
        """List registered department names."""
        return list(self._adapters.keys())

    async def handle(
        self,
        request: MCPRequest,
        *,
        target: str | None = None,
        caller: str | None = None,
    ) -> MCPResponse:
        """
        Route an MCPRequest to the target department.

        Args:
            request: The MCP request to route.
            target: Department name to route to. If None, uses
                    request.metadata.get("target_department").
            caller: Department name making the request (for permission
                    checks on inter-department calls). None = external.

        Returns:
            MCPResponse from the target department.
        """
        dept_name = target or request.metadata.get("target_department")

        if not dept_name:
            return create_error_response(
                request.request_id,
                request.session_id,
                MCPError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="No target department specified",
                ),
            )

        # Check adapter exists
        adapter = self._adapters.get(dept_name)
        if adapter is None:
            return create_error_response(
                request.request_id,
                request.session_id,
                MCPError(
                    code=ErrorCode.TOOL_NOT_FOUND,
                    message=f"Department '{dept_name}' not registered",
                    details={"available": self.departments},
                ),
            )

        # Permission check (only for tool calls)
        if request.request_type == RequestType.TOOL_CALL and request.tool_name:
            denied = self._check_permission(adapter, request, caller)
            if denied is not None:
                return denied

        # Audit inter-department calls
        if caller:
            self._audit_log.append({
                "caller": caller,
                "target": dept_name,
                "tool": request.tool_name,
                "request_id": request.request_id,
                "session_id": request.session_id,
            })

        # Dispatch
        return await adapter.handle(request)

    async def call_tool(
        self,
        target: str,
        tool_name: str,
        parameters: dict[str, Any],
        session_id: str,
        *,
        caller: str | None = None,
    ) -> MCPResponse:
        """
        Convenience method for inter-department tool calls.

        This is the primary API for departments calling each other:

            # Verification asks MasterWeaver to re-extract
            response = await router.call_tool(
                target="MasterWeaver",
                tool_name="extract_entities",
                parameters={"input_data": text, "domain": "beekeeping"},
                session_id=session_id,
                caller="Verification",
            )
        """
        request = MCPRequest(
            request_id=generate_request_id(),
            session_id=session_id,
            request_type=RequestType.TOOL_CALL,
            tool_name=tool_name,
            parameters=parameters,
        )
        return await self.handle(request, target=target, caller=caller)

    async def broadcast_health(self, session_id: str) -> dict[str, Any]:
        """Check health of all registered departments."""
        results = {}
        for name, adapter in self._adapters.items():
            request = MCPRequest(
                request_id=generate_request_id(),
                session_id=session_id,
                request_type=RequestType.HEALTH_CHECK,
            )
            response = await adapter.handle(request)
            results[name] = (
                response.result
                if response.status == ResponseStatus.SUCCESS
                else {"status": "unreachable", "error": response.error}
            )
        return results

    def get_audit_log(
        self, caller: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return recent audit entries, optionally filtered by caller."""
        entries = self._audit_log
        if caller:
            entries = [e for e in entries if e["caller"] == caller]
        return entries[-limit:]

    # ------------------------------------------------------------------
    # Permission enforcement
    # ------------------------------------------------------------------

    def _check_permission(
        self,
        adapter: DepartmentMCPAdapter,
        request: MCPRequest,
        caller: str | None,
    ) -> MCPResponse | None:
        """
        Check whether the caller has permission to invoke this tool.

        Returns an error MCPResponse if denied, None if allowed.
        """
        tool_def = adapter._tool_map.get(request.tool_name)
        if tool_def is None:
            return None  # unknown tool — adapter will handle the 404

        # Determine caller's permissions
        caller_permissions = self._get_caller_permissions(caller)

        # Check each required permission
        for required in tool_def.required_permissions:
            if required not in caller_permissions:
                msg = (
                    f"{caller or 'external'} lacks {required.value} "
                    f"permission for {adapter.name}.{request.tool_name}"
                )
                if tool_def.permission_mode == PermissionMode.HARD:
                    return create_error_response(
                        request.request_id,
                        request.session_id,
                        MCPError(
                            code=ErrorCode.PERMISSION_DENIED,
                            message=msg,
                        ),
                    )
                else:
                    # SOFT: log but allow
                    logger.warning(f"SOFT permission violation: {msg}")

        return None

    def _get_caller_permissions(
        self, caller: str | None
    ) -> list[PermissionLevel]:
        """
        Resolve the permission set for a caller.

        External callers (None) = the application/user layer. They get
        all permissions since they represent the top-level orchestrator.
        Inter-department callers get their charter permissions.
        Unknown callers (not registered) get READ only.
        """
        if caller is None:
            # Application/user — full access
            return list(PermissionLevel)

        adapter = self._adapters.get(caller)
        if adapter is None:
            return [PermissionLevel.READ]

        return list(adapter.charter.permissions)

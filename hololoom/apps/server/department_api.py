"""
Department Federation API — HTTP bridge for MCP department routing.

Exposes the MCPRouter as REST endpoints so that external callers
(nanoclaw, OpenClaw agents, curl) can invoke department tools over HTTP.

Endpoints:
    POST /departments/{name}/{tool}     — Call a specific tool
    GET  /departments                   — List all departments
    GET  /departments/{name}/tools      — List tools for a department
    GET  /departments/{name}/health     — Health check one department
    GET  /departments/health            — Health check all departments

The router boots a lightweight federation on first request (zero-config).
Real departments can be registered later via register_department().

Created: 2026-03-12
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hololoom.apps.departments.mcp.bootstrap import (
    create_lightweight_federation,
)
from hololoom.apps.departments.mcp.mcp_router import MCPRouter
from hololoom.infrastructure.mcp.protocol import (
    RequestType,
    ResponseStatus,
    generate_session_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/departments", tags=["departments"])

# Module-level singleton — lazy-initialized on first request
_router: MCPRouter | None = None


async def _get_router() -> MCPRouter:
    """Get or create the department federation router."""
    global _router
    if _router is None:
        logger.info("Booting lightweight department federation...")
        _router = await create_lightweight_federation()
        logger.info(
            f"Federation ready: {_router.departments}"
        )
    return _router


def get_federation_router() -> MCPRouter | None:
    """Access the live MCPRouter (None if not yet booted). For Python callers."""
    return _router


async def register_department(name: str, department) -> None:
    """
    Register a real department into the running federation.

    Replaces the lightweight stand-in if one exists.
    """
    from hololoom.alignment.mcp_department_registry import DEPARTMENT_REGISTRY
    from hololoom.apps.departments.mcp.mcp_adapter import DepartmentMCPAdapter

    mcp_router = await _get_router()
    charter = DEPARTMENT_REGISTRY.get(name)
    if charter is None:
        raise ValueError(
            f"No charter for '{name}'. "
            f"Available: {list(DEPARTMENT_REGISTRY.keys())}"
        )
    adapter = DepartmentMCPAdapter(department, charter)
    mcp_router.register(adapter)
    logger.info(f"Registered real department: {name}")


# ============================================================================
# Request/Response models
# ============================================================================


class ToolCallRequest(BaseModel):
    """Request body for calling a department tool."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = Field(
        default=None,
        description="Session ID for context continuity. Auto-generated if omitted.",
    )
    caller: str | None = Field(
        default=None,
        description="Calling department name (for inter-department permission checks).",
    )


class ToolCallResponse(BaseModel):
    """Response from a department tool call."""
    status: str
    result: Any | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    request_id: str
    session_id: str


class DepartmentInfo(BaseModel):
    """Summary of a registered department."""
    name: str
    role: str
    tools: list[str]


# ============================================================================
# Routes
# ============================================================================


@router.get("", response_model=list[DepartmentInfo])
async def list_departments():
    """List all registered departments and their tools."""
    mcp_router = await _get_router()
    result = []
    for name in mcp_router.departments:
        adapter = mcp_router._adapters[name]
        result.append(DepartmentInfo(
            name=name,
            role=adapter.charter.role,
            tools=[t.name for t in adapter.charter.tools],
        ))
    return result


@router.get("/health")
async def health_all():
    """Health check all departments."""
    mcp_router = await _get_router()
    session_id = generate_session_id()
    return await mcp_router.broadcast_health(session_id)


@router.get("/{name}/tools")
async def list_tools(name: str):
    """List available tools for a department."""
    mcp_router = await _get_router()
    adapter = mcp_router._adapters.get(name)
    if adapter is None:
        raise HTTPException(
            status_code=404,
            detail=f"Department '{name}' not found. Available: {mcp_router.departments}",
        )
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
            "returns": t.returns,
        }
        for t in adapter.charter.tools
    ]


@router.get("/{name}/health")
async def health_one(name: str):
    """Health check a single department."""
    mcp_router = await _get_router()
    if name not in mcp_router._adapters:
        raise HTTPException(
            status_code=404,
            detail=f"Department '{name}' not found. Available: {mcp_router.departments}",
        )
    session_id = generate_session_id()
    from hololoom.infrastructure.mcp.protocol import MCPRequest, generate_request_id
    req = MCPRequest(
        request_id=generate_request_id(),
        session_id=session_id,
        request_type=RequestType.HEALTH_CHECK,
    )
    resp = await mcp_router._adapters[name].handle(req)
    if resp.status == ResponseStatus.SUCCESS:
        return resp.result
    raise HTTPException(status_code=500, detail=str(resp.error))


@router.post("/{name}/{tool}", response_model=ToolCallResponse)
async def call_tool(name: str, tool: str, body: ToolCallRequest):
    """
    Call a tool on a department.

    Examples:
        POST /departments/MasterWeaver/extract_entities
        {"parameters": {"input_data": "Bees pollinate flowers", "domain": "beekeeping"}}

        POST /departments/Infrastructure/diagnose_performance
        {"parameters": {}}

        POST /departments/Verification/validate_confidence_claim
        {"parameters": {"claim": {"confidence": 0.99}, "evidence": {}}}
    """
    mcp_router = await _get_router()
    session_id = body.session_id or generate_session_id()

    resp = await mcp_router.call_tool(
        target=name,
        tool_name=tool,
        parameters=body.parameters,
        session_id=session_id,
        caller=body.caller,
    )

    return ToolCallResponse(
        status=resp.status.value if hasattr(resp.status, "value") else resp.status,
        result=resp.result,
        error=resp.error if resp.error else None,
        metadata=resp.metadata or {},
        request_id=resp.request_id,
        session_id=resp.session_id,
    )

"""
MCP Federation — Inter-department communication layer.

Bridges the Department protocol (DepartmentRequest/DepartmentResponse)
with the MCP wire protocol (MCPRequest/MCPResponse), adding:
- Permission enforcement (HARD/SOFT from charters)
- Session ID propagation
- Error escalation with department-aware routing
- Health-based discovery

Usage:
    from hololoom.apps.departments.mcp import DepartmentMCPAdapter, MCPRouter

    adapter = DepartmentMCPAdapter(my_department, charter)
    router = MCPRouter()
    router.register(adapter)
    response = await router.handle(mcp_request)
"""

from .mcp_adapter import DepartmentMCPAdapter  # noqa: F401
from .mcp_router import MCPRouter  # noqa: F401

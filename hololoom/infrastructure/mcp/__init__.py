"""
Model Context Protocol (MCP) Server

Part 2: Foundation Infrastructure (Days 6-10)

Public API:
- MCPServer: Main server class
- MCPRequest/MCPResponse: Protocol types
- create_mcp_server: Factory function
"""

from hololoom.infrastructure.mcp.protocol import (
    AVAILABLE_TOOLS,
    QUERY_SQL_TOOL,
    ErrorCode,
    MCPError,
    # Request/Response
    MCPRequest,
    MCPResponse,
    # Enums
    RequestType,
    ResponseStatus,
    # Tools
    Tool,
    ToolName,
    ToolParameter,
    create_error_response,
    create_success_response,
    # Helpers
    generate_request_id,
    generate_session_id,
    validate_query_sql_params,
)
from hololoom.infrastructure.mcp.server import MCPServer, create_mcp_server

__all__ = [
    # Server
    "MCPServer",
    "create_mcp_server",

    # Protocol
    "MCPRequest",
    "MCPResponse",
    "MCPError",

    # Enums
    "RequestType",
    "ResponseStatus",
    "ErrorCode",
    "ToolName",

    # Tools
    "Tool",
    "ToolParameter",
    "QUERY_SQL_TOOL",
    "AVAILABLE_TOOLS",

    # Helpers
    "generate_request_id",
    "generate_session_id",
    "create_success_response",
    "create_error_response",
    "validate_query_sql_params",
]

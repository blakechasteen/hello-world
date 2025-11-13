#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Types

Part 2: Foundation Infrastructure (Days 6-10)

Defines MCP protocol types for inter-department communication.
Based on: https://modelcontextprotocol.io/specification

MCP Departments:
- Infrastructure: SQL backend (this server)
- Context: Query routing and classification
- Orchestration: Overall coordination
- MasterWeaver: Response synthesis
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


# ============================================================================
# MCP Protocol Version
# ============================================================================

MCP_VERSION = "1.0.0"


# ============================================================================
# Tool Definitions
# ============================================================================

class ToolName(str, Enum):
    """Available MCP tools"""
    QUERY_SQL = "query_sql"


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Optional[Any] = None


@dataclass
class Tool:
    """MCP tool definition"""
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: str  # Return type description


# ============================================================================
# MCP Request/Response
# ============================================================================

class RequestType(str, Enum):
    """MCP request types"""
    TOOL_CALL = "tool_call"
    TOOL_LIST = "tool_list"
    HEALTH_CHECK = "health_check"


class ResponseStatus(str, Enum):
    """MCP response status"""
    SUCCESS = "success"
    ERROR = "error"
    ESCALATE = "escalate"  # Escalate to higher department


@dataclass
class MCPRequest:
    """
    MCP request from Context Department

    Example:
        {
            "request_id": "req_123",
            "session_id": "session_456",
            "request_type": "tool_call",
            "tool_name": "query_sql",
            "parameters": {
                "query": "SELECT * FROM policy_rules WHERE domain = ?",
                "params": ["beekeeping"]
            },
            "metadata": {
                "user_id": "user_123",
                "timestamp": "2025-11-12T10:30:00Z"
            }
        }
    """
    request_id: str
    session_id: str
    request_type: RequestType
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPRequest":
        """Parse MCP request from dict"""
        return cls(
            request_id=data["request_id"],
            session_id=data["session_id"],
            request_type=RequestType(data["request_type"]),
            tool_name=data.get("tool_name"),
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "request_type": self.request_type.value,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MCPResponse:
    """
    MCP response to Context Department

    Example (Success):
        {
            "request_id": "req_123",
            "session_id": "session_456",
            "status": "success",
            "result": {
                "rows": [...],
                "row_count": 5,
                "latency_ms": 12.5
            },
            "error": null,
            "metadata": {
                "backend": "postgresql",
                "query_hash": "abc123"
            }
        }

    Example (Error):
        {
            "request_id": "req_123",
            "session_id": "session_456",
            "status": "error",
            "result": null,
            "error": {
                "code": "SQL_ERROR",
                "message": "no such table: nonexistent_table",
                "details": {...}
            },
            "metadata": {...}
        }

    Example (Escalate):
        {
            "request_id": "req_123",
            "session_id": "session_456",
            "status": "escalate",
            "result": null,
            "error": {
                "code": "TIMEOUT",
                "message": "Query timeout after 30s",
                "escalate_to": "context"
            },
            "metadata": {...}
        }
    """
    request_id: str
    session_id: str
    status: ResponseStatus
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResponse":
        """Parse MCP response from dict"""
        return cls(
            request_id=data["request_id"],
            session_id=data["session_id"],
            status=ResponseStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow()
        )


# ============================================================================
# Error Codes
# ============================================================================

class ErrorCode(str, Enum):
    """MCP error codes"""
    # Infrastructure errors (500-599)
    SQL_ERROR = "SQL_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    TIMEOUT = "TIMEOUT"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"

    # Request errors (400-499)
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    MISSING_PARAMETER = "MISSING_PARAMETER"

    # Permission errors (403)
    PERMISSION_DENIED = "PERMISSION_DENIED"

    # Unknown
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class MCPError:
    """MCP error details"""
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    escalate_to: Optional[str] = None  # Department to escalate to

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        result = {
            "code": self.code.value,
            "message": self.message
        }
        if self.details:
            result["details"] = self.details
        if self.escalate_to:
            result["escalate_to"] = self.escalate_to
        return result


# ============================================================================
# Session Management
# ============================================================================

@dataclass
class Session:
    """MCP session tracking"""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_request_at: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self):
        """Update session activity"""
        self.last_request_at = datetime.utcnow()
        self.request_count += 1


# ============================================================================
# Tool Registry
# ============================================================================

# SQL query tool definition
QUERY_SQL_TOOL = Tool(
    name=ToolName.QUERY_SQL,
    description="Execute SQL query on precision data tables (policy_rules, transaction_logs, audit_trails, user_permissions)",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="SQL query string (SELECT only for safety)",
            required=True
        ),
        ToolParameter(
            name="params",
            type="array",
            description="Query parameters for SQL injection prevention",
            required=False,
            default=[]
        ),
        ToolParameter(
            name="timeout_ms",
            type="number",
            description="Query timeout in milliseconds (default: 30000)",
            required=False,
            default=30000
        )
    ],
    returns="QueryResult with rows, row_count, latency_ms, success, error"
)


# All available tools
AVAILABLE_TOOLS = [
    QUERY_SQL_TOOL
]


# ============================================================================
# Helper Functions
# ============================================================================

def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{uuid.uuid4().hex[:12]}"


def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"session_{uuid.uuid4().hex[:12]}"


def create_success_response(
    request_id: str,
    session_id: str,
    result: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> MCPResponse:
    """Create success response"""
    return MCPResponse(
        request_id=request_id,
        session_id=session_id,
        status=ResponseStatus.SUCCESS,
        result=result,
        metadata=metadata or {}
    )


def create_error_response(
    request_id: str,
    session_id: str,
    error: MCPError,
    metadata: Optional[Dict[str, Any]] = None
) -> MCPResponse:
    """Create error response"""
    status = ResponseStatus.ESCALATE if error.escalate_to else ResponseStatus.ERROR

    return MCPResponse(
        request_id=request_id,
        session_id=session_id,
        status=status,
        error=error.to_dict(),
        metadata=metadata or {}
    )


def validate_query_sql_params(params: Dict[str, Any]) -> Optional[MCPError]:
    """
    Validate query_sql parameters

    Returns:
        MCPError if validation fails, None if valid
    """
    # Check required parameter
    if "query" not in params:
        return MCPError(
            code=ErrorCode.MISSING_PARAMETER,
            message="Missing required parameter: 'query'",
            details={"parameter": "query"}
        )

    # Check query type
    query = params["query"].strip().upper()
    if not query.startswith("SELECT"):
        return MCPError(
            code=ErrorCode.INVALID_PARAMETERS,
            message="Only SELECT queries are allowed for safety",
            details={"query": params["query"]}
        )

    # Check timeout
    if "timeout_ms" in params:
        timeout = params["timeout_ms"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            return MCPError(
                code=ErrorCode.INVALID_PARAMETERS,
                message="timeout_ms must be a positive number",
                details={"timeout_ms": timeout}
            )

    return None

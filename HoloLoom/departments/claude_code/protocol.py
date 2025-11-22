"""
Protocol definitions for Claude Code Department

Defines request/response types for Matrix → VS Code integration
"""

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum


class CodeAction(str, Enum):
    """Supported code actions"""
    QUERY = "query"
    REFACTOR = "refactor"
    EXPLAIN = "explain"
    TEST = "test"
    FIX = "fix"
    CONTEXT = "context"


class ReasoningMode(str, Enum):
    """HoloLoom reasoning modes"""
    DIRECT = "direct"
    VERIFY = "verify"
    RESEARCH = "research"
    PLAN_EXECUTE = "plan_execute"


@dataclass
class ClaudeCodeRequest:
    """
    Request to Claude Code Department

    Attributes:
        action: What action to perform
        params: Action-specific parameters
        user_id: Matrix user ID (for authorization)
        room_id: Matrix room ID (for context)
    """
    action: CodeAction
    params: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    room_id: Optional[str] = None


@dataclass
class ClaudeCodeResponse:
    """
    Response from Claude Code Department

    Attributes:
        success: Whether action succeeded
        result: Action result (formatted text)
        metadata: Additional metadata (confidence, duration, etc.)
        error: Error message if failed
    """
    success: bool
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class MCPToolCall:
    """
    MCP protocol tool invocation

    Attributes:
        name: Tool name (e.g., "code/query")
        arguments: Tool arguments
        id: Request ID for tracking
    """
    name: str
    arguments: Dict[str, Any]
    id: str


@dataclass
class MCPRequest:
    """MCP JSON-RPC 2.0 request"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: str = ""
    method: str = ""
    params: Optional[Dict[str, Any]] = None


@dataclass
class MCPResponse:
    """MCP JSON-RPC 2.0 response"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class MCPNotification:
    """MCP notification (no response expected)"""
    jsonrpc: Literal["2.0"] = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None


# ============================================================================
# Helper Functions
# ============================================================================

def create_query_request(
    question: str,
    mode: ReasoningMode = ReasoningMode.VERIFY,
    include_context: bool = True,
    user_id: Optional[str] = None,
    room_id: Optional[str] = None
) -> ClaudeCodeRequest:
    """Create a code query request"""
    return ClaudeCodeRequest(
        action=CodeAction.QUERY,
        params={
            "question": question,
            "mode": mode.value,
            "includeContext": include_context
        },
        user_id=user_id,
        room_id=room_id
    )


def create_refactor_request(
    instruction: str,
    code: Optional[str] = None,
    user_id: Optional[str] = None,
    room_id: Optional[str] = None
) -> ClaudeCodeRequest:
    """Create a refactor request"""
    params = {"instruction": instruction}
    if code:
        params["code"] = code

    return ClaudeCodeRequest(
        action=CodeAction.REFACTOR,
        params=params,
        user_id=user_id,
        room_id=room_id
    )


def create_explain_request(
    target: Optional[str] = None,
    depth: Literal["brief", "detailed", "comprehensive"] = "detailed",
    user_id: Optional[str] = None,
    room_id: Optional[str] = None
) -> ClaudeCodeRequest:
    """Create an explain request"""
    params = {"depth": depth}
    if target:
        params["target"] = target

    return ClaudeCodeRequest(
        action=CodeAction.EXPLAIN,
        params=params,
        user_id=user_id,
        room_id=room_id
    )


def create_test_request(
    code: Optional[str] = None,
    test_type: Literal["unit", "integration", "edge", "all"] = "unit",
    user_id: Optional[str] = None,
    room_id: Optional[str] = None
) -> ClaudeCodeRequest:
    """Create a test generation request"""
    params = {"testType": test_type}
    if code:
        params["code"] = code

    return ClaudeCodeRequest(
        action=CodeAction.TEST,
        params=params,
        user_id=user_id,
        room_id=room_id
    )


def create_fix_request(
    code: Optional[str] = None,
    include_diagnostics: bool = True,
    user_id: Optional[str] = None,
    room_id: Optional[str] = None
) -> ClaudeCodeRequest:
    """Create a fix request"""
    params = {"includeDiagnostics": include_diagnostics}
    if code:
        params["code"] = code

    return ClaudeCodeRequest(
        action=CodeAction.FIX,
        params=params,
        user_id=user_id,
        room_id=room_id
    )


def create_context_request(
    include_selection: bool = True,
    include_diagnostics: bool = True,
    user_id: Optional[str] = None,
    room_id: Optional[str] = None
) -> ClaudeCodeRequest:
    """Create a context request"""
    return ClaudeCodeRequest(
        action=CodeAction.CONTEXT,
        params={
            "includeSelection": include_selection,
            "includeDiagnostics": include_diagnostics
        },
        user_id=user_id,
        room_id=room_id
    )

"""
Protocol definitions for Claude Code Department

Defines request/response types for Matrix → VS Code integration
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


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
    params: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    room_id: str | None = None


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
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


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
    arguments: dict[str, Any]
    id: str


@dataclass
class MCPRequest:
    """MCP JSON-RPC 2.0 request"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: str = ""
    method: str = ""
    params: dict[str, Any] | None = None


@dataclass
class MCPResponse:
    """MCP JSON-RPC 2.0 response"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: str = ""
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


@dataclass
class MCPNotification:
    """MCP notification (no response expected)"""
    jsonrpc: Literal["2.0"] = "2.0"
    method: str = ""
    params: dict[str, Any] | None = None


# ============================================================================
# Helper Functions
# ============================================================================

def create_query_request(
    question: str,
    mode: ReasoningMode = ReasoningMode.VERIFY,
    include_context: bool = True,
    user_id: str | None = None,
    room_id: str | None = None
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
    code: str | None = None,
    user_id: str | None = None,
    room_id: str | None = None
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
    target: str | None = None,
    depth: Literal["brief", "detailed", "comprehensive"] = "detailed",
    user_id: str | None = None,
    room_id: str | None = None
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
    code: str | None = None,
    test_type: Literal["unit", "integration", "edge", "all"] = "unit",
    user_id: str | None = None,
    room_id: str | None = None
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
    code: str | None = None,
    include_diagnostics: bool = True,
    user_id: str | None = None,
    room_id: str | None = None
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
    user_id: str | None = None,
    room_id: str | None = None
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

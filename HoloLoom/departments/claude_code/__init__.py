"""
Claude Code Department
======================

Bridges Matrix ChatOps to VS Code Claude Code extension via MCP protocol.

Architecture:
- MCP Client connects to VS Code MCP Server (WebSocket port 9001)
- Exposes tools: query, refactor, explain, test, fix
- Provides department interface for HoloLoom integration
- Enables Matrix → VS Code → Matrix bidirectional communication

Usage:
    from HoloLoom.departments.claude_code import ClaudeCodeDepartment

    dept = ClaudeCodeDepartment()
    await dept.start()

    result = await dept.process({
        "action": "query",
        "question": "How does the WeavingOrchestrator work?"
    })

    await dept.stop()
"""

from .department import ClaudeCodeDepartment
from .protocol import (
    ClaudeCodeRequest,
    ClaudeCodeResponse,
    MCPToolCall,
    CodeAction
)

__all__ = [
    'ClaudeCodeDepartment',
    'ClaudeCodeRequest',
    'ClaudeCodeResponse',
    'MCPToolCall',
    'CodeAction'
]

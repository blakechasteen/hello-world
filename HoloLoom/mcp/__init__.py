"""
MCP - Model Context Protocol Integration
==========================================
Tool ecosystem for HoloLoom using MCP protocol.
"""

from .server import MCPServer, MCPConfig
from .protocol import ToolDefinition, ToolResult

__all__ = ['MCPServer', 'MCPConfig', 'ToolDefinition', 'ToolResult']

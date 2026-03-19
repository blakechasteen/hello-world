"""
MCP Base — Shared utilities for HoloLoom MCP servers.
======================================================

Provides:
- Safe MCP import with graceful fallback
- Unified tool builder (Tool from name + description + schema dict)
- Standard error wrapping (TextContent with error prefix)
- Standard JSON response formatting

All HoloLoom MCP servers should import from here instead of
reimplementing MCP boilerplate.

Usage:
    from hololoom.mcp_tools.mcp_base import (
        MCP_AVAILABLE,
        create_server,
        error_content,
        json_content,
        tool,
    )

    if not MCP_AVAILABLE:
        sys.exit("MCP not installed. Run: pip install mcp")

    server = create_server("hololoom-lite")

    TOOLS = [
        tool("loom_recall", "Search memories by query", {
            "query": {"type": "string", "description": "Search text"},
            "top_k": {"type": "integer", "default": 5},
        }, required=["query"]),
    ]
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Safe MCP import
# ============================================================================

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import ImageContent, Resource, TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not installed. Run: pip install mcp")

    # Placeholder types so modules can define signatures without crashing
    class Server:  # type: ignore[no-redef]
        def __init__(self, name: str) -> None:
            self.name = name

    class Tool:  # type: ignore[no-redef]
        pass

    class TextContent:  # type: ignore[no-redef]
        pass

    class ImageContent:  # type: ignore[no-redef]
        pass

    class Resource:  # type: ignore[no-redef]
        pass

    async def stdio_server(*a: Any, **kw: Any) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("MCP not installed")


# Fix Windows encoding (common across all MCP servers)
if sys.platform == "win32":
    import io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


# ============================================================================
# Server factory
# ============================================================================

def create_server(name: str) -> Server:
    """Create an MCP server with the given name."""
    return Server(name)


# ============================================================================
# Tool builder
# ============================================================================

def tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
) -> Tool:
    """
    Build an MCP Tool with JSON Schema input.

    Args:
        name: Tool name (use hololoom_ prefix for consistency)
        description: What the tool does (shown to the LLM)
        properties: JSON Schema properties dict
        required: List of required property names
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return Tool(name=name, description=description, inputSchema=schema)


# ============================================================================
# Response helpers
# ============================================================================

def json_content(data: Any, **kwargs: Any) -> list[TextContent]:
    """Wrap a Python object as a JSON TextContent response."""
    return [TextContent(type="text", text=json.dumps(data, indent=2, default=str, **kwargs))]


def text_content(text: str) -> list[TextContent]:
    """Wrap a string as a TextContent response."""
    return [TextContent(type="text", text=text)]


def error_content(message: str) -> list[TextContent]:
    """Wrap an error message as a TextContent response with [ERROR] prefix."""
    return [TextContent(type="text", text=f"[ERROR] {message}")]

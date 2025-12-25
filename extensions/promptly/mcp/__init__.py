"""
Promptly MCP Server - Model Context Protocol Integration

Exposes 13 core tools for Claude Desktop integration:
- Prompt management (add, get, list, history)
- Branch management (create, switch, list)
- Skill management (add, get, list)
- Chain management (create, run)
- LLM Judge evaluation

Usage:
    # Run as MCP server (for Claude Desktop)
    python -m promptly.mcp.server

    # Or import and run programmatically
    from promptly.mcp import run_server
    import asyncio
    asyncio.run(run_server())
"""

from .server import (
    app,
    main as run_server,
)

__all__ = [
    'app',
    'run_server',
]

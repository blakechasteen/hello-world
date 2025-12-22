"""
HoloLoom Integrations - Nexus Layer
====================================
Exposing HoloLoom to external ecosystems.

Integrations:
- MCP Server: Model Context Protocol server for Claude Desktop
- Dark Trace: Interpretability system integration with weaving orchestrator
"""

from .mcp_server import server as mcp_server

# Dark Trace Integration (Phase 10 - December 2025)
from .dark_trace_integration import (
    DarkTraceIntegration,
    IntegrationConfig,
    IntegrationMode,
    IntegrationResult,
    create_integration,
    enable_dark_trace,
    get_dark_trace,
    with_interpretability,
)

__all__ = [
    # MCP Server
    "mcp_server",
    # Dark Trace Integration
    "DarkTraceIntegration",
    "IntegrationConfig",
    "IntegrationMode",
    "IntegrationResult",
    "create_integration",
    "enable_dark_trace",
    "get_dark_trace",
    "with_interpretability",
]

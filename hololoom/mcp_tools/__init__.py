"""
HoloLoom MCP Tools
==================

Model Context Protocol (MCP) server integration for HoloLoom.
Exposes HoloLoom's memory, reasoning, and learning capabilities to Claude Desktop.

**Status**: Production Ready (November 2025)
**Integration**: From Promptly platform (27+ tools)
**Documentation**: See README.md

Core tool categories:
- Memory tools: Store and retrieve from hololoom knowledge graph
- Reasoning tools: Execute agentic reasoning modes
- Learning tools: Access learning statistics and patterns
- Skill tools: Execute skill templates (after skills integration)
- Evaluation tools: A/B testing, LLM-as-judge (after Week 2)
- Analytics tools: Query performance and usage stats
"""

from .server import app, create_hololoom_mcp_server

__all__ = ['app', 'create_hololoom_mcp_server']

__version__ = '1.0.0'

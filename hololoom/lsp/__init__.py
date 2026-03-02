"""
HoloLoom LSP Server Package
===========================

Language Server Protocol implementation for HoloLoom neural memory system.

This package provides LSP server capabilities that integrate HoloLoom's
neural architecture with code editors and IDEs.

Features:
- Code completion from semantic memory
- Hover information from knowledge graph
- Go-to-definition for entity relationships
- Workspace symbol search

Quick Start:
    from hololoom.lsp import server
    server.main()

Or run as module:
    python -m HoloLoom.lsp.server [--port 8080] [--log-level DEBUG]
"""

from .server import (
    HoloLoomLanguageServer,
    server,
    setup_logging,
)

__version__ = "0.1.0"
__author__ = "HoloLoom Contributors"

__all__ = [
    "HoloLoomLanguageServer",
    "server",
    "setup_logging",
]

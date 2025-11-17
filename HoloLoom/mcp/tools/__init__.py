"""
MCP Tools - Built-in Tool Collection
=====================================
Standard tools for HoloLoom MCP server.
"""

from .calculator import register_calculator
from .search import register_search
from .text_tools import register_text_tools

__all__ = ['register_calculator', 'register_search', 'register_text_tools']


def register_all_tools(server):
    """Register all built-in tools with MCP server."""
    register_calculator(server)
    register_search(server)
    register_text_tools(server)

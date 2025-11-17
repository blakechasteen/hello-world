"""
HoloLoom Query Interface
========================

Natural language queries over org-mode knowledge graphs.

Usage:
    from HoloLoom.query import query

    result = await query("What should I work on?")
    print(result)
"""

from .engine import QueryEngine, query

__all__ = ['QueryEngine', 'query']

"""
HoloLoom Performance Optimizations
===================================
Caching, compression, and optimization utilities.
"""

from .cache import CacheEntry, QueryCache

__all__ = ["QueryCache", "CacheEntry"]

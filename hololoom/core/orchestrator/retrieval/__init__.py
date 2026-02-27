"""
Retrieval Module
================

Memory retrieval functions for the WeavingOrchestrator.

This module provides multi-pass memory crawling with Matryoshka importance gating
and backend memory querying capabilities.

Exported Functions:
    multipass_memory_crawl: Recursive gated multipass crawling
    query_memory_backend: Simple memory backend querying

Author: Claude Code (Elegance Pass Refactoring - Phase 6)
Date: 2025-11-22
"""

from hololoom.core.orchestrator.retrieval.multipass_retrieval import (
    multipass_memory_crawl,
    query_memory_backend,
)

__all__ = [
    'multipass_memory_crawl',
    'query_memory_backend',
]

#!/usr/bin/env python3
"""
Infrastructure Department - Shared data layer and performance diagnostics.

This package provides low-level infrastructure services for all departments:
- Zero-copy memory-mapped embeddings
- Neo4j + Qdrant integration
- Performance diagnostics and monitoring
- Resource management (connection pooling, caching)
"""

from .infrastructure import InfrastructureDepartment
from .zero_copy import EmbeddingView, ZeroCopyEmbeddingStore

__all__ = [
    'InfrastructureDepartment',
    'ZeroCopyEmbeddingStore',
    'EmbeddingView'
]

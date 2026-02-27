"""
Memory Store Implementations
=============================
Protocol-based memory backends following HoloLoom standards.
"""

from .in_memory_store import InMemoryStore

__all__ = ['InMemoryStore']

# Optional imports (graceful degradation)
try:
    from .mem0_store import Mem0MemoryStore
    __all__.append('Mem0MemoryStore')
except ImportError:
    pass

try:
    from .neo4j_store import Neo4jMemoryStore
    __all__.append('Neo4jMemoryStore')
except ImportError:
    pass

try:
    from .qdrant_store import QdrantMemoryStore
    __all__.append('QdrantMemoryStore')
except ImportError:
    pass

try:
    from .hybrid_store import HybridMemoryStore
    __all__.append('HybridMemoryStore')
except ImportError:
    pass

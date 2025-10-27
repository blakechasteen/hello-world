#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weaving Memory Adapter - Bridge between WeavingShuttle and Memory Backends
===========================================================================

Adapts various memory backends (UnifiedMemory, backend_factory, etc.) to work
seamlessly with WeavingShuttle's YarnGraph interface.

This adapter provides:
- Unified API for WeavingShuttle regardless of backend
- Automatic conversion between MemoryShard and backend Memory types
- Thread selection from temporal windows
- Graceful degradation when backends unavailable

Philosophy:
The WeavingShuttle should work with ANY memory backend without knowing
the implementation details. This adapter is the "translation layer".
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib

# Shared types
from HoloLoom.Documentation.types import MemoryShard, Query

# Try importing various memory backends
try:
    from HoloLoom.memory.unified import UnifiedMemory, RecallStrategy, Memory as UnifiedMemoryObj
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError:
    UNIFIED_MEMORY_AVAILABLE = False
    UnifiedMemory = None

try:
    from HoloLoom.memory.backend_factory import create_memory_backend
    from HoloLoom.memory.protocol import Memory, MemoryQuery, Strategy
    BACKEND_FACTORY_AVAILABLE = True
except ImportError:
    BACKEND_FACTORY_AVAILABLE = False
    create_memory_backend = None

try:
    from HoloLoom.chrono.trigger import TemporalWindow
    CHRONO_AVAILABLE = True
except ImportError:
    CHRONO_AVAILABLE = False
    TemporalWindow = None

logger = logging.getLogger(__name__)


# ============================================================================
# Conversion Utilities
# ============================================================================

def memoryshard_to_protocol_memory(shard: MemoryShard) -> 'Memory':
    """
    Convert MemoryShard to protocol Memory object.

    Args:
        shard: WeavingShuttle MemoryShard

    Returns:
        Protocol Memory object for backend storage
    """
    if not BACKEND_FACTORY_AVAILABLE:
        raise RuntimeError("Backend factory not available")

    return Memory(
        id=shard.id,
        text=shard.text,
        timestamp=datetime.now(),  # Could extract from shard metadata
        context={
            'episode': shard.episode,
            'entities': shard.entities,
            'motifs': shard.motifs
        },
        metadata=getattr(shard, 'metadata', {})
    )


def protocol_memory_to_memoryshard(memory: 'Memory') -> MemoryShard:
    """
    Convert protocol Memory to MemoryShard.

    Args:
        memory: Protocol Memory object from backend

    Returns:
        MemoryShard for WeavingShuttle
    """
    return MemoryShard(
        id=memory.id,
        text=memory.text,
        episode=memory.context.get('episode', 'unknown'),
        entities=memory.context.get('entities', []),
        motifs=memory.context.get('motifs', []),
        embedding=getattr(memory, 'embedding', []),
        metadata=memory.metadata
    )


def unified_memory_to_memoryshard(memory: 'UnifiedMemoryObj') -> MemoryShard:
    """
    Convert UnifiedMemory Memory to MemoryShard.

    Args:
        memory: UnifiedMemory Memory object

    Returns:
        MemoryShard for WeavingShuttle
    """
    return MemoryShard(
        id=memory.id,
        text=memory.text,
        episode=memory.context.get('episode', 'unknown') if memory.context else 'unknown',
        entities=memory.context.get('entities', []) if memory.context else [],
        motifs=memory.tags or [],
        embedding=[],  # UnifiedMemory handles this internally
        metadata={'relevance': memory.relevance}
    )


# ============================================================================
# Weaving Memory Adapter
# ============================================================================

class WeavingMemoryAdapter:
    """
    Adapter that makes any memory backend work with WeavingShuttle.

    Provides YarnGraph-compatible interface while delegating to various backends:
    - UnifiedMemory
    - Backend factory (Neo4j, Qdrant, Hybrid)
    - In-memory fallback

    Usage:
        # With UnifiedMemory
        adapter = WeavingMemoryAdapter.from_unified_memory(
            user_id="blake",
            enable_neo4j=True,
            enable_qdrant=True
        )

        # With backend factory
        adapter = WeavingMemoryAdapter.from_backend_factory(
            backend_type="hybrid",
            neo4j_config={...},
            qdrant_config={...}
        )

        # With initial shards (in-memory fallback)
        adapter = WeavingMemoryAdapter.from_shards(shards)

        # Use in WeavingShuttle
        shuttle = WeavingShuttle(cfg=config, memory_adapter=adapter)
    """

    def __init__(
        self,
        backend: Optional[Any] = None,
        backend_type: str = "in_memory",
        initial_shards: Optional[List[MemoryShard]] = None
    ):
        """
        Initialize memory adapter.

        Args:
            backend: Memory backend (UnifiedMemory, protocol backend, etc.)
            backend_type: Type identifier (unified, factory, in_memory)
            initial_shards: Optional initial memory shards
        """
        self.backend = backend
        self.backend_type = backend_type
        self.logger = logging.getLogger(__name__)

        # In-memory fallback
        self.shards: Dict[str, MemoryShard] = {}
        if initial_shards:
            self.shards = {shard.id: shard for shard in initial_shards}

        self.logger.info(f"WeavingMemoryAdapter initialized: {backend_type}")

    # ========================================================================
    # Factory Methods
    # ========================================================================

    @classmethod
    def from_unified_memory(
        cls,
        user_id: str = "default",
        enable_mem0: bool = False,
        enable_neo4j: bool = True,
        enable_qdrant: bool = True,
        enable_hofstadter: bool = False
    ) -> 'WeavingMemoryAdapter':
        """
        Create adapter from UnifiedMemory backend.

        Args:
            user_id: User identifier
            enable_*: Backend feature flags

        Returns:
            WeavingMemoryAdapter with UnifiedMemory backend
        """
        if not UNIFIED_MEMORY_AVAILABLE:
            logger.warning("UnifiedMemory not available, falling back to in-memory")
            return cls(backend_type="in_memory")

        backend = UnifiedMemory(
            user_id=user_id,
            enable_mem0=enable_mem0,
            enable_neo4j=enable_neo4j,
            enable_qdrant=enable_qdrant,
            enable_hofstadter=enable_hofstadter
        )

        return cls(backend=backend, backend_type="unified")

    @classmethod
    def from_backend_factory(
        cls,
        backend_type: str = "hybrid",
        neo4j_config: Optional[Dict] = None,
        qdrant_config: Optional[Dict] = None
    ) -> 'WeavingMemoryAdapter':
        """
        Create adapter from backend factory.

        Args:
            backend_type: Backend type (neo4j, qdrant, hybrid, networkx)
            neo4j_config: Neo4j configuration
            qdrant_config: Qdrant configuration

        Returns:
            WeavingMemoryAdapter with factory backend
        """
        if not BACKEND_FACTORY_AVAILABLE:
            logger.warning("Backend factory not available, falling back to in-memory")
            return cls(backend_type="in_memory")

        try:
            backend = create_memory_backend(
                backend_type=backend_type,
                neo4j_config=neo4j_config,
                qdrant_config=qdrant_config
            )
            return cls(backend=backend, backend_type="factory")
        except Exception as e:
            logger.error(f"Failed to create backend: {e}, falling back to in-memory")
            return cls(backend_type="in_memory")

    @classmethod
    def from_shards(cls, shards: List[MemoryShard]) -> 'WeavingMemoryAdapter':
        """
        Create adapter with in-memory shard storage (fallback/testing).

        Args:
            shards: Initial memory shards

        Returns:
            WeavingMemoryAdapter with in-memory backend
        """
        return cls(backend_type="in_memory", initial_shards=shards)

    # ========================================================================
    # YarnGraph-Compatible Interface
    # ========================================================================

    def select_threads(
        self,
        temporal_window: Optional['TemporalWindow'],
        query: Query
    ) -> List[MemoryShard]:
        """
        Select threads based on temporal window and query.

        Compatible with YarnGraph.select_threads() signature.

        Args:
            temporal_window: Time bounds for selection
            query: Query for relevance filtering

        Returns:
            List of relevant memory shards
        """
        if self.backend_type == "unified" and self.backend:
            return self._select_via_unified(query)
        elif self.backend_type == "factory" and self.backend:
            return self._select_via_factory(query)
        else:
            return self._select_via_in_memory(query)

    def _select_via_unified(self, query: Query) -> List[MemoryShard]:
        """Select threads via UnifiedMemory backend."""
        try:
            # Use balanced strategy for best results
            memories = self.backend.recall(
                query=query.text,
                strategy=RecallStrategy.BALANCED,
                limit=10
            )

            # Convert to MemoryShards
            shards = [unified_memory_to_memoryshard(mem) for mem in memories]
            self.logger.debug(f"Selected {len(shards)} threads via UnifiedMemory")
            return shards

        except Exception as e:
            self.logger.error(f"UnifiedMemory recall failed: {e}")
            return []

    def _select_via_factory(self, query: Query) -> List[MemoryShard]:
        """Select threads via backend factory."""
        try:
            # Create query object
            query_obj = MemoryQuery(
                text=query.text,
                strategy=Strategy.BALANCED,
                limit=10
            )

            # Async recall
            result = asyncio.run(self.backend.recall(query_obj))

            # Convert to MemoryShards
            shards = [protocol_memory_to_memoryshard(mem) for mem in result.memories]
            self.logger.debug(f"Selected {len(shards)} threads via backend factory")
            return shards

        except Exception as e:
            self.logger.error(f"Backend factory recall failed: {e}")
            return []

    def _select_via_in_memory(self, query: Query) -> List[MemoryShard]:
        """Select threads from in-memory shards (simple fallback)."""
        # Simple implementation: return all shards
        # In production, would do BM25 or similarity search
        shards = list(self.shards.values())
        self.logger.debug(f"Selected {len(shards)} threads from in-memory store")
        return shards

    # ========================================================================
    # Storage Operations
    # ========================================================================

    async def add_shard(self, shard: MemoryShard) -> str:
        """
        Add a new memory shard to the backend.

        Args:
            shard: Memory shard to add

        Returns:
            Shard ID
        """
        if self.backend_type == "unified" and self.backend:
            # Store via UnifiedMemory
            memory_id = self.backend.store(
                text=shard.text,
                context={
                    'episode': shard.episode,
                    'entities': shard.entities,
                    'motifs': shard.motifs
                },
                importance=0.5
            )
            return memory_id

        elif self.backend_type == "factory" and self.backend:
            # Store via backend factory
            memory = memoryshard_to_protocol_memory(shard)
            memory_id = await self.backend.store(memory)
            return memory_id

        else:
            # Store in-memory
            self.shards[shard.id] = shard
            return shard.id

    def add_shard_sync(self, shard: MemoryShard) -> str:
        """Synchronous version of add_shard."""
        return asyncio.run(self.add_shard(shard))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory backend statistics.

        Returns:
            Dict with backend stats
        """
        if self.backend_type == "unified" and self.backend:
            return {
                'backend_type': 'unified_memory',
                'user_id': self.backend.user_id,
                'total_memories': 'unknown'  # UnifiedMemory doesn't expose this yet
            }
        elif self.backend_type == "factory" and self.backend:
            return {
                'backend_type': 'factory',
                'backends': getattr(self.backend, 'backends', []),
                'total_memories': 'unknown'
            }
        else:
            return {
                'backend_type': 'in_memory',
                'total_shards': len(self.shards)
            }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_weaving_memory(
    mode: str = "in_memory",
    shards: Optional[List[MemoryShard]] = None,
    **kwargs
) -> WeavingMemoryAdapter:
    """
    Convenience function to create memory adapter.

    Args:
        mode: Memory mode (in_memory, unified, hybrid)
        shards: Optional initial shards
        **kwargs: Additional config for backend

    Returns:
        WeavingMemoryAdapter

    Examples:
        # In-memory (fast, testing)
        memory = create_weaving_memory("in_memory", shards=shards)

        # UnifiedMemory (intelligent extraction)
        memory = create_weaving_memory("unified", user_id="blake")

        # Hybrid (production)
        memory = create_weaving_memory(
            "hybrid",
            neo4j_config={'url': 'bolt://localhost:7687'},
            qdrant_config={'url': 'http://localhost:6333'}
        )
    """
    if mode == "in_memory":
        return WeavingMemoryAdapter.from_shards(shards or [])
    elif mode == "unified":
        return WeavingMemoryAdapter.from_unified_memory(**kwargs)
    elif mode == "hybrid":
        return WeavingMemoryAdapter.from_backend_factory(
            backend_type="hybrid",
            **kwargs
        )
    else:
        logger.warning(f"Unknown mode: {mode}, defaulting to in-memory")
        return WeavingMemoryAdapter.from_shards(shards or [])


if __name__ == "__main__":
    print("""
Weaving Memory Adapter
======================

Bridges WeavingShuttle with various memory backends:
- UnifiedMemory (intelligent extraction)
- Backend factory (Neo4j, Qdrant, Hybrid)
- In-memory (fast fallback)

Usage:
    # In-memory mode
    adapter = WeavingMemoryAdapter.from_shards(shards)

    # UnifiedMemory mode
    adapter = WeavingMemoryAdapter.from_unified_memory(
        user_id="blake",
        enable_neo4j=True,
        enable_qdrant=True
    )

    # Hybrid backend mode
    adapter = WeavingMemoryAdapter.from_backend_factory(
        backend_type="hybrid",
        neo4j_config={'url': 'bolt://localhost:7687'},
        qdrant_config={'url': 'http://localhost:6333'}
    )

    # Use in WeavingShuttle
    shuttle = WeavingShuttle(cfg=config, memory_adapter=adapter)
""")
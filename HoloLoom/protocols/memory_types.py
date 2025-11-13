"""
Memory Types - Data types for memory systems
=============================================

All memory-related data types (Memory, MemoryQuery, retrieval results, etc.)

This module contains the data structures used by memory protocols.
Separated from protocols to avoid circular imports.

Author: HoloLoom Architecture Team
Date: 2025-01-12 (Phase 0, Task 7: Protocol Consolidation)
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


# ============================================================================
# Core Memory Types
# ============================================================================

@dataclass
class Memory:
    """
    Single memory unit.

    Compatible with MemoryShard from SpinningWheel.
    Used by all memory backends (Mem0, Neo4j, Qdrant, HoloLoom).
    """
    id: str
    text: str
    timestamp: datetime
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    embedding: Optional[Any] = None  # Vector embedding (numpy array or list)

    @classmethod
    def from_shard(cls, shard: Any, timestamp: Optional[datetime] = None) -> 'Memory':
        """Create Memory from MemoryShard (SpinningWheel output)."""
        return cls(
            id=shard.id,
            text=shard.text,
            timestamp=timestamp or datetime.now(),
            context={
                'episode': getattr(shard, 'episode', None),
                'entities': getattr(shard, 'entities', []),
                'motifs': getattr(shard, 'motifs', []),
            },
            metadata=getattr(shard, 'metadata', None) or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'id': self.id,
            'text': self.text,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'metadata': self.metadata
        }
        # Convert embedding to list for JSON serialization
        if self.embedding is not None:
            import numpy as np
            if isinstance(self.embedding, np.ndarray):
                result['embedding'] = self.embedding.tolist()
            else:
                result['embedding'] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create Memory from dictionary."""
        data = data.copy()
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        # Convert embedding back to numpy array
        if 'embedding' in data and data['embedding'] is not None:
            import numpy as np
            if isinstance(data['embedding'], list):
                data['embedding'] = np.array(data['embedding'])
        return cls(**data)


@dataclass
class MemoryQuery:
    """
    Query for retrieving memories.

    Used by all memory backends to specify search criteria.
    """
    text: str
    user_id: str = "default"
    limit: int = 5
    filters: Optional[Dict[str, Any]] = None
    strategy: Optional['Strategy'] = None


# ============================================================================
# Retrieval Results
# ============================================================================

@dataclass
class MemoryRetrievalResult:
    """
    Retrieval results from memory backend.

    Used by memory backend's retrieve() method.
    Contains memories plus metadata about the retrieval.

    Note: This is different from StrategyRetrievalResult which is used
    by RetrievalStrategy protocol for strategy pattern.
    """
    memories: List[Memory]
    scores: List[float]
    strategy_used: str
    metadata: Dict[str, Any]


# ============================================================================
# Enums
# ============================================================================

class Strategy(Enum):
    """Retrieval strategies for memory queries."""
    TEMPORAL = "temporal"      # Time-based retrieval
    SEMANTIC = "semantic"      # Similarity-based retrieval
    GRAPH = "graph"           # Graph traversal
    PATTERN = "pattern"       # Pattern matching
    FUSED = "fused"           # Multi-strategy fusion
    BALANCED = "balanced"     # Balanced approach


class QueryMode(Enum):
    """Query complexity modes."""
    FAST = "fast"                      # Quick, shallow search
    BALANCED = "balanced"              # Standard search
    COMPREHENSIVE = "comprehensive"    # Deep, thorough search
    RESEARCH = "research"              # Maximum depth research mode


# ============================================================================
# Helper Functions
# ============================================================================

def shards_to_memories(shards: List[Any], timestamp: Optional[datetime] = None) -> List[Memory]:
    """
    Convert SpinningWheel shards to Memory objects.

    Args:
        shards: List of MemoryShard objects
        timestamp: Optional timestamp (defaults to now)

    Returns:
        List of Memory objects
    """
    return [Memory.from_shard(shard, timestamp) for shard in shards]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core Types
    'Memory',
    'MemoryQuery',

    # Retrieval Results
    'MemoryRetrievalResult',

    # Enums
    'Strategy',
    'QueryMode',

    # Utilities
    'shards_to_memories',
]

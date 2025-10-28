"""
Memory Protocols - Protocol-based interfaces for memory backends.
All backends implement MemoryStore protocol for easy extension.
"""

from typing import List, Dict, Optional, Protocol, runtime_checkable, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class Memory:
    """Single memory unit. Compatible with MemoryShard from SpinningWheel."""
    id: str
    text: str
    timestamp: datetime
    context: Dict[str, Any]
    metadata: Dict[str, Any]

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
        return {
            'id': self.id,
            'text': self.text,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        data = data.copy()
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class MemoryQuery:
    """Memory query."""
    text: str
    user_id: str = "default"
    limit: int = 5
    filters: Optional[Dict[str, Any]] = None
    strategy: Optional['Strategy'] = None


@dataclass
class RetrievalResult:
    """Retrieval results with scores and metadata."""
    memories: List[Memory]
    scores: List[float]
    strategy_used: str
    metadata: Dict[str, Any]


class Strategy(Enum):
    """Retrieval strategies."""
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    GRAPH = "graph"
    PATTERN = "pattern"
    FUSED = "fused"
    BALANCED = "balanced"


class QueryMode(Enum):
    """Query complexity modes."""
    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"
    RESEARCH = "research"


# ============================================================================
# Core Protocol
# ============================================================================

try:
    from HoloLoom.protocols import MemoryStore
except ImportError:
    # Fallback if canonical protocol unavailable
    @runtime_checkable
    class MemoryStore(Protocol):
        """Memory storage backend protocol."""

        async def store(self, memory: Memory, user_id: str = "default") -> str: ...
        async def store_many(self, memories: List[Memory], user_id: str = "default") -> List[str]: ...
        async def get_by_id(self, memory_id: str) -> Optional[Memory]: ...
        async def retrieve(self, query: MemoryQuery, strategy: Strategy = Strategy.FUSED) -> RetrievalResult: ...
        async def delete(self, memory_id: str) -> bool: ...
        async def health_check(self) -> Dict[str, Any]: ...


# ============================================================================
# Helper Functions
# ============================================================================

def shards_to_memories(shards: List[Any], timestamp: Optional[datetime] = None) -> List[Memory]:
    """Convert SpinningWheel shards to Memory objects."""
    return [Memory.from_shard(shard, timestamp) for shard in shards]
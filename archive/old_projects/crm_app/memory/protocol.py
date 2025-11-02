"""
Memory System Protocol

Defines the interface that all memory subsystems must implement.
Following HoloLoom's protocol-based design pattern.

Every memory subsystem (Symbolic, Semantic, Episodic, Relational, Working)
implements this protocol to enable unified memory operations.
"""

from typing import Protocol, List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np


class MemoryType(str, Enum):
    """Types of memory subsystems"""
    SYMBOLIC = "symbolic"      # Exact entity storage
    SEMANTIC = "semantic"      # Continuous embeddings
    EPISODIC = "episodic"      # Time-ordered events
    RELATIONAL = "relational"  # Entity relationships
    WORKING = "working"        # Active context
    META = "meta"              # Memory about memories


@dataclass
class MemoryAddress:
    """
    Unique address for any memory in the system.

    Attributes:
        subsystem: Which memory subsystem (symbolic, semantic, etc.)
        id: Unique identifier within subsystem
        version: Version number for time-travel/replay
        timestamp: When memory was created
    """
    subsystem: MemoryType
    id: str
    version: int = 1
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def __str__(self) -> str:
        return f"{self.subsystem.value}://{self.id}@v{self.version}"


@dataclass
class Memory:
    """
    Universal memory representation.

    All memories (contacts, embeddings, activities, edges) are represented
    as Memory objects with subsystem-specific data.

    Attributes:
        address: Unique memory address
        content: Primary content (entity data, embedding vector, etc.)
        metadata: Additional information about the memory
        confidence: Confidence in this memory (0-1)
        source: Where this memory came from
    """
    address: MemoryAddress
    content: Any  # Flexible: Contact, np.ndarray, Activity, Edge, etc.
    metadata: Dict[str, Any]
    confidence: float = 1.0
    source: Optional[str] = None

    def __post_init__(self):
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.utcnow().isoformat()


@dataclass
class MemoryQuery:
    """
    Query specification for memory retrieval.

    Attributes:
        subsystem: Which memory subsystem to query
        criteria: Query criteria (filters, similarity target, etc.)
        limit: Maximum results to return
        min_confidence: Minimum confidence threshold
        time_range: Optional temporal filter
    """
    subsystem: MemoryType
    criteria: Dict[str, Any]
    limit: int = 10
    min_confidence: float = 0.0
    time_range: Optional[tuple] = None  # (start, end)


@dataclass
class MemoryResult:
    """
    Result from memory query.

    Attributes:
        memories: Retrieved memories
        scores: Relevance scores for each memory
        metadata: Query execution metadata
        total_found: Total memories matching (before limit)
    """
    memories: List[Memory]
    scores: List[float]
    metadata: Dict[str, Any]
    total_found: int


class MemoryProtocol(Protocol):
    """
    Protocol that all memory subsystems must implement.

    This enables:
    - Unified memory operations across all subsystems
    - Swappable implementations
    - Consistent interface for reads, writes, queries

    Every subsystem (Symbolic, Semantic, Episodic, Relational, Working)
    implements these methods in a subsystem-appropriate way.
    """

    def write(self, memory: Memory) -> MemoryAddress:
        """
        Write memory to subsystem.

        Args:
            memory: Memory object to store

        Returns:
            Address where memory was written

        Examples:
            # Symbolic: Store contact entity
            addr = symbolic.write(Memory(content=contact, ...))

            # Semantic: Store embedding
            addr = semantic.write(Memory(content=embedding_vector, ...))

            # Episodic: Store activity
            addr = episodic.write(Memory(content=activity, ...))
        """
        ...

    def read(self, address: MemoryAddress) -> Optional[Memory]:
        """
        Read memory by address.

        Args:
            address: Memory address to retrieve

        Returns:
            Memory object or None if not found
        """
        ...

    def query(self, query: MemoryQuery) -> MemoryResult:
        """
        Query memories by criteria.

        Args:
            query: Query specification

        Returns:
            MemoryResult with matching memories

        Examples:
            # Symbolic: Filter by attributes
            result = symbolic.query(MemoryQuery(
                subsystem=MemoryType.SYMBOLIC,
                criteria={"lead_score__gt": 0.8}
            ))

            # Semantic: Find similar
            result = semantic.query(MemoryQuery(
                subsystem=MemoryType.SEMANTIC,
                criteria={"similar_to": embedding_vector}
            ))

            # Episodic: Time range
            result = episodic.query(MemoryQuery(
                subsystem=MemoryType.EPISODIC,
                criteria={"type": "call"},
                time_range=(start_date, end_date)
            ))
        """
        ...

    def update(self, address: MemoryAddress, updates: Dict[str, Any]) -> Memory:
        """
        Update existing memory.

        Args:
            address: Memory to update
            updates: Fields to update

        Returns:
            Updated memory
        """
        ...

    def delete(self, address: MemoryAddress) -> bool:
        """
        Delete memory (or mark inactive).

        Args:
            address: Memory to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    def associate(self, addr1: MemoryAddress, addr2: MemoryAddress, relation: str) -> MemoryAddress:
        """
        Create association between memories.

        Args:
            addr1: First memory
            addr2: Second memory
            relation: Relationship type

        Returns:
            Address of the association (edge) memory

        Note:
            This is primarily for Relational memory, but all subsystems
            support it for cross-subsystem associations.
        """
        ...

    def compress(self, address: MemoryAddress) -> Memory:
        """
        Compress symbolic memory to semantic representation.

        Args:
            address: Symbolic memory to compress

        Returns:
            Semantic memory (embedding)

        Note:
            Used by Semantic subsystem to create embeddings from
            symbolic entities.
        """
        ...

    def stats(self) -> Dict[str, Any]:
        """
        Get subsystem statistics.

        Returns:
            Stats dict with counts, sizes, performance metrics
        """
        ...


# Convenience types for common memory operations
SymbolicMemory = Memory  # For contacts, companies, deals
SemanticMemory = Memory  # For embeddings (content is np.ndarray)
EpisodicMemory = Memory  # For activities (time-ordered)
RelationalMemory = Memory  # For edges/relationships
WorkingMemory = Memory  # For active context
MetaMemory = Memory  # For metadata about memories

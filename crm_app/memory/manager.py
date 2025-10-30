"""
Unified Memory Manager

Coordinates all memory subsystems and provides a single interface for
all memory operations in the CRM.

"Everything is a memory operation" - actualized.
"""

from typing import Dict, List, Optional, Any
import time
from datetime import datetime

from .protocol import (
    MemoryProtocol,
    MemoryType,
    MemoryAddress,
    Memory,
    MemoryQuery,
    MemoryResult
)


class UnifiedMemory:
    """
    Unified interface to all memory subsystems.

    This is the single entry point for all memory operations in the CRM.
    Routes operations to appropriate subsystems based on memory type.

    Architecture:
        UnifiedMemory coordinates 6 subsystems:
        - Symbolic: Contacts, Companies, Deals (exact storage)
        - Semantic: Embeddings (continuous representations)
        - Episodic: Activities (time-ordered events)
        - Relational: Knowledge Graph (entity relationships)
        - Working: Active context during queries
        - Meta: Metadata about all memories

    Usage:
        ```python
        memory = UnifiedMemory()

        # Write to symbolic memory
        contact_addr = memory.write_symbolic(contact)

        # Compress to semantic memory
        embedding_addr = memory.compress(contact_addr)

        # Query semantic memory
        similar = memory.query_semantic({"similar_to": contact_addr})

        # Associate memories
        memory.associate(contact_addr, company_addr, "WORKS_AT")
        ```

    Every CRM operation becomes a memory operation:
    - create_contact() → write_symbolic()
    - embed_contact() → compress()
    - find_similar() → query_semantic()
    - log_activity() → write_episodic()
    - add_relationship() → associate()
    """

    def __init__(self):
        """
        Initialize unified memory system.

        Subsystems are registered lazily to avoid circular dependencies.
        """
        self._subsystems: Dict[MemoryType, MemoryProtocol] = {}
        self._meta_memory: Dict[str, Any] = {
            'total_writes': 0,
            'total_reads': 0,
            'total_queries': 0,
            'subsystem_stats': {}
        }

    def register_subsystem(self, memory_type: MemoryType, subsystem: MemoryProtocol):
        """
        Register a memory subsystem.

        Args:
            memory_type: Type of memory (SYMBOLIC, SEMANTIC, etc.)
            subsystem: Implementation of MemoryProtocol
        """
        self._subsystems[memory_type] = subsystem
        self._meta_memory['subsystem_stats'][memory_type.value] = {}
        print(f"[UnifiedMemory] Registered {memory_type.value} subsystem")

    def _get_subsystem(self, memory_type: MemoryType) -> MemoryProtocol:
        """Get subsystem by type, with error handling."""
        if memory_type not in self._subsystems:
            raise ValueError(
                f"Memory subsystem {memory_type.value} not registered. "
                f"Available: {list(self._subsystems.keys())}"
            )
        return self._subsystems[memory_type]

    # ========================================================================
    # Core Memory Operations (Subsystem-Agnostic)
    # ========================================================================

    def write(self, memory: Memory) -> MemoryAddress:
        """
        Write memory to appropriate subsystem.

        Args:
            memory: Memory object to store

        Returns:
            Address where memory was written
        """
        start_time = time.time()

        subsystem = self._get_subsystem(memory.address.subsystem)
        address = subsystem.write(memory)

        # Update meta-memory
        self._meta_memory['total_writes'] += 1
        elapsed_ms = (time.time() - start_time) * 1000

        # Track write to meta-memory
        self._update_meta_memory('write', memory.address.subsystem, elapsed_ms)

        return address

    def read(self, address: MemoryAddress) -> Optional[Memory]:
        """
        Read memory by address from any subsystem.

        Args:
            address: Memory address to retrieve

        Returns:
            Memory object or None
        """
        start_time = time.time()

        subsystem = self._get_subsystem(address.subsystem)
        memory = subsystem.read(address)

        # Update meta-memory
        self._meta_memory['total_reads'] += 1
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_meta_memory('read', address.subsystem, elapsed_ms)

        return memory

    def query(self, query: MemoryQuery) -> MemoryResult:
        """
        Query memories from specified subsystem.

        Args:
            query: Query specification

        Returns:
            MemoryResult with matching memories
        """
        start_time = time.time()

        subsystem = self._get_subsystem(query.subsystem)
        result = subsystem.query(query)

        # Update meta-memory
        self._meta_memory['total_queries'] += 1
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_meta_memory('query', query.subsystem, elapsed_ms)

        # Enrich result metadata
        result.metadata['query_time_ms'] = elapsed_ms
        result.metadata['subsystem'] = query.subsystem.value

        return result

    def update(self, address: MemoryAddress, updates: Dict[str, Any]) -> Memory:
        """
        Update existing memory.

        Args:
            address: Memory to update
            updates: Fields to update

        Returns:
            Updated memory
        """
        subsystem = self._get_subsystem(address.subsystem)
        return subsystem.update(address, updates)

    def delete(self, address: MemoryAddress) -> bool:
        """
        Delete memory from subsystem.

        Args:
            address: Memory to delete

        Returns:
            True if deleted, False if not found
        """
        subsystem = self._get_subsystem(address.subsystem)
        return subsystem.delete(address)

    def associate(
        self,
        addr1: MemoryAddress,
        addr2: MemoryAddress,
        relation: str
    ) -> MemoryAddress:
        """
        Create association between memories.

        This primarily writes to Relational memory but can work
        across any subsystems.

        Args:
            addr1: First memory
            addr2: Second memory
            relation: Relationship type (e.g., "WORKS_AT", "SIMILAR_TO")

        Returns:
            Address of the association (edge) memory
        """
        # Get relational subsystem
        relational = self._get_subsystem(MemoryType.RELATIONAL)
        return relational.associate(addr1, addr2, relation)

    def compress(self, address: MemoryAddress) -> Memory:
        """
        Compress symbolic memory to semantic representation.

        This reads from Symbolic memory and writes to Semantic memory.

        Args:
            address: Symbolic memory address (e.g., contact)

        Returns:
            Semantic memory (embedding)
        """
        semantic = self._get_subsystem(MemoryType.SEMANTIC)
        return semantic.compress(address)

    # ========================================================================
    # Convenience Methods (Typed by Subsystem)
    # ========================================================================

    def write_symbolic(self, content: Any, metadata: Optional[Dict] = None) -> MemoryAddress:
        """Write to symbolic memory (contacts, companies, deals)."""
        address = MemoryAddress(subsystem=MemoryType.SYMBOLIC, id=str(content.id))
        memory = Memory(
            address=address,
            content=content,
            metadata=metadata or {}
        )
        return self.write(memory)

    def write_semantic(self, embedding: Any, source_address: MemoryAddress,
                      metadata: Optional[Dict] = None) -> MemoryAddress:
        """Write to semantic memory (embeddings)."""
        address = MemoryAddress(
            subsystem=MemoryType.SEMANTIC,
            id=f"emb_{source_address.id}"
        )
        memory = Memory(
            address=address,
            content=embedding,
            metadata={
                'source_address': str(source_address),
                **(metadata or {})
            }
        )
        return self.write(memory)

    def write_episodic(self, activity: Any, metadata: Optional[Dict] = None) -> MemoryAddress:
        """Write to episodic memory (activities)."""
        address = MemoryAddress(subsystem=MemoryType.EPISODIC, id=str(activity.id))
        memory = Memory(
            address=address,
            content=activity,
            metadata=metadata or {}
        )
        return self.write(memory)

    def query_symbolic(self, criteria: Dict[str, Any], limit: int = 10) -> MemoryResult:
        """Query symbolic memory with filters."""
        query = MemoryQuery(
            subsystem=MemoryType.SYMBOLIC,
            criteria=criteria,
            limit=limit
        )
        return self.query(query)

    def query_semantic(self, criteria: Dict[str, Any], limit: int = 10,
                      min_similarity: float = 0.3) -> MemoryResult:
        """Query semantic memory (similarity search)."""
        query = MemoryQuery(
            subsystem=MemoryType.SEMANTIC,
            criteria=criteria,
            limit=limit,
            min_confidence=min_similarity
        )
        return self.query(query)

    def query_episodic(self, criteria: Dict[str, Any], time_range: Optional[tuple] = None,
                      limit: int = 50) -> MemoryResult:
        """Query episodic memory (activity history)."""
        query = MemoryQuery(
            subsystem=MemoryType.EPISODIC,
            criteria=criteria,
            time_range=time_range,
            limit=limit
        )
        return self.query(query)

    # ========================================================================
    # Meta-Memory Operations
    # ========================================================================

    def _update_meta_memory(self, operation: str, subsystem: MemoryType, elapsed_ms: float):
        """Update meta-memory statistics."""
        subsystem_stats = self._meta_memory['subsystem_stats'][subsystem.value]

        if operation not in subsystem_stats:
            subsystem_stats[operation] = {
                'count': 0,
                'total_time_ms': 0.0,
                'avg_time_ms': 0.0
            }

        stats = subsystem_stats[operation]
        stats['count'] += 1
        stats['total_time_ms'] += elapsed_ms
        stats['avg_time_ms'] = stats['total_time_ms'] / stats['count']

    def stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory system statistics.

        Returns:
            Statistics from all subsystems plus meta-memory
        """
        stats = {
            'meta': self._meta_memory.copy(),
            'subsystems': {}
        }

        for memory_type, subsystem in self._subsystems.items():
            stats['subsystems'][memory_type.value] = subsystem.stats()

        return stats

    def status(self) -> str:
        """
        Get human-readable status of memory system.

        Returns:
            Status string showing all subsystems and their states
        """
        lines = ["UnifiedMemory Status:"]
        lines.append(f"  Total Writes: {self._meta_memory['total_writes']}")
        lines.append(f"  Total Reads: {self._meta_memory['total_reads']}")
        lines.append(f"  Total Queries: {self._meta_memory['total_queries']}")
        lines.append("")
        lines.append("  Registered Subsystems:")

        for memory_type, subsystem in self._subsystems.items():
            subsystem_stats = subsystem.stats()
            lines.append(f"    - {memory_type.value}: {subsystem_stats}")

        return "\n".join(lines)


# Factory function for easy creation
def create_unified_memory() -> UnifiedMemory:
    """
    Create and initialize unified memory system.

    Returns:
        UnifiedMemory instance

    Note:
        Subsystems must be registered after creation:
        ```python
        memory = create_unified_memory()
        memory.register_subsystem(MemoryType.SYMBOLIC, symbolic_impl)
        memory.register_subsystem(MemoryType.SEMANTIC, semantic_impl)
        # ... etc
        ```
    """
    return UnifiedMemory()

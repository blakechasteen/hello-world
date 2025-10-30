"""
Unified Memory System

Everything is a memory operation. This module provides the unified interface
to all memory subsystems in the CRM.

Memory Subsystems:
- Symbolic Memory: Exact entity storage (Contacts, Companies, Deals)
- Semantic Memory: Continuous embeddings for approximate retrieval
- Episodic Memory: Time-ordered interaction history (Activities)
- Relational Memory: Entity relationships (Knowledge Graph)
- Working Memory: Active context during processing
- Meta Memory: Metadata about memories (quality, confidence, lineage)

Architecture:
    All CRM operations are memory operations:
    - Creating contact = Memory write (symbolic subsystem)
    - Computing embedding = Memory compression (semantic subsystem)
    - Finding similar = Memory query (semantic subsystem)
    - Logging activity = Memory write (episodic subsystem)
    - Adding relationship = Memory association (relational subsystem)

Usage:
    ```python
    from crm_app.memory import UnifiedMemory, MemoryType

    memory = UnifiedMemory()

    # Write to symbolic memory
    address = memory.write(MemoryType.SYMBOLIC, contact)

    # Query semantic memory
    similar = memory.query(MemoryType.SEMANTIC, similarity_criteria)

    # Read from any subsystem
    entity = memory.read(address)
    ```
"""

from .protocol import (
    MemoryProtocol,
    MemoryType,
    MemoryAddress,
    Memory,
    MemoryQuery,
    MemoryResult
)

from .manager import UnifiedMemory

__all__ = [
    'MemoryProtocol',
    'MemoryType',
    'MemoryAddress',
    'Memory',
    'MemoryQuery',
    'MemoryResult',
    'UnifiedMemory'
]

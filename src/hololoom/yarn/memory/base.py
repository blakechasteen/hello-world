"""Compatibility shim for memory package.

This module re-exports the factory functions and protocols from
`holoLoom.memory.cache` so other modules can import from
`holoLoom.memory.base` as expected by orchestrator.
"""
from .cache import (
    Retriever,
    MemoryShard,
    RetrieverMS,
    MemoryManager,
    PDVClient,
    MemoAIClient,
    create_retriever,
    create_memory_manager,
)

__all__ = [
    'Retriever', 'MemoryShard', 'RetrieverMS', 'MemoryManager',
    'PDVClient', 'MemoAIClient', 'create_retriever', 'create_memory_manager'
]

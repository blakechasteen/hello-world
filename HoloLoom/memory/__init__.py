from .cache import (
    Retriever,
    MemoryShard,
    RetrieverMS,
    MemoryManager,
    PDVClient,
    MemoAIClient,
    create_retriever,
    create_memory_manager
)

from .graph import (
    KGStore,
    KGEdge,
    KG,
    extract_entities_simple,
    build_kg_from_text
)

# Weaving Metaphor Aliases
YarnGraph = KG  # The persistent symbolic memory - discrete thread structure
ReflectionBuffer = MemoryManager  # Learning loop - stores outcomes for improvement

__all__ = [
    # Cache
    'Retriever',
    'MemoryShard',
    'RetrieverMS',
    'MemoryManager',
    'ReflectionBuffer',  # Weaving alias
    'PDVClient',
    'MemoAIClient',
    'create_retriever',
    'create_memory_manager',
    # Graph
    'KGStore',
    'KGEdge',
    'KG',
    'YarnGraph',  # Weaving alias
    'extract_entities_simple',
    'build_kg_from_text'
]
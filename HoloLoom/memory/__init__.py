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
    LegacyShardsAdapter,
    extract_entities_simple,
    build_kg_from_text
)

# Weaving Metaphor Aliases
YarnGraph = KG  # The persistent symbolic memory - discrete thread structure
ReflectionBuffer = MemoryManager  # Learning loop - stores outcomes for improvement

# =============================================================================
# Backward Compatibility Re-exports (December 2025 Consolidation)
# =============================================================================
# These directories were consolidated into memory/:
#   - HoloLoom/awareness/ → HoloLoom/memory/awareness/
#   - HoloLoom/memory_symphony/ → HoloLoom/memory/symphony/
#   - HoloLoom/yarn/ → HoloLoom/memory/yarn/
#
# Old import paths still work via HoloLoom/__init__.py shims.
# =============================================================================

# Re-export awareness module
from .awareness import (
    BetaWaveContextPacker,
    BetaWavePacker,  # Backward compat alias
    CompositionalAwarenessLayer,
    SmartContextPacker,
    ContextPacker,  # Backward compat alias
    DualStreamGenerator,
    MemoryFusion,
    MetaAwarenessLayer,
)

# Re-export symphony (memory conductor) module
from .symphony import (
    MemoryConductor,
    MemoryQuery,
    MemoryStrategy,
    MemoryCoordinationResult,
    create_memory_conductor,
)

# Re-export yarn module
from .yarn import Yarn, EggrollYarn

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
    'LegacyShardsAdapter',  # Backward compatibility for deprecated shards
    'extract_entities_simple',
    'build_kg_from_text',
    # Awareness (consolidated Dec 2025)
    'BetaWaveContextPacker',
    'BetaWavePacker',  # Backward compat alias
    'CompositionalAwarenessLayer',
    'SmartContextPacker',
    'ContextPacker',  # Backward compat alias
    'DualStreamGenerator',
    'MemoryFusion',
    'MetaAwarenessLayer',
    # Symphony (consolidated Dec 2025)
    'MemoryConductor',
    'MemoryQuery',
    'MemoryStrategy',
    'MemoryCoordinationResult',
    'create_memory_conductor',
    # Yarn (consolidated Dec 2025)
    'Yarn',
    'EggrollYarn',  # Backward compatibility alias
]
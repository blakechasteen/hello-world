import warnings as _warnings

# Core memory imports — graceful degradation if numpy/networkx not installed
try:
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
except ImportError as e:
    _warnings.warn(
        f"Memory core unavailable (missing dependency: {e}). "
        f"Install with: pip install hololoom"
    )

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
try:
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
except ImportError:
    pass

# Re-export symphony (memory conductor) module
try:
    from .symphony import (
        MemoryConductor,
        MemoryQuery,
        MemoryStrategy,
        MemoryCoordinationResult,
        create_memory_conductor,
    )
except ImportError:
    pass

# Re-export yarn module
try:
    from .yarn import Yarn, EggrollYarn
except ImportError:
    pass

# Borrowed Features (March 2026) — graceful degradation if deps missing
try:
    from .retrieval_planner import RetrievalPlanner, RetrievalPlan, RetrievalComplexity
    from .version_log import VersionLog, MemoryDelta
    from .synthesis import SynthesisResult, default_synthesis_fn
    from .memfs import MemFS, MemoryFile
    from .sleep_consolidator import SleepConsolidator
    from .security_screen import SecurityScreenError, default_security_screen, create_security_screen
except ImportError as e:
    import warnings
    warnings.warn(f"Memory borrowed features unavailable: {e}")

# Token Maximization (March 2026) — graceful degradation if deps missing
try:
    from .query_cache import QueryCache, QueryCacheEntry
    from .multi_resolution import MultiResolutionRenderer, ResolutionPolicy, RenderResolution
    from .repo_cache import RepoCache, RepoCacheEntry
    from .render_cache import RenderCache, RenderCacheEntry
except ImportError as e:
    import warnings
    warnings.warn(f"Memory token maximization features unavailable: {e}")

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
    # Borrowed Features (March 2026)
    'RetrievalPlanner',
    'RetrievalPlan',
    'RetrievalComplexity',
    'VersionLog',
    'MemoryDelta',
    'SynthesisResult',
    'default_synthesis_fn',
    'MemFS',
    'MemoryFile',
    'SleepConsolidator',
    'SecurityScreenError',
    'default_security_screen',
    'create_security_screen',
    # Token Maximization (March 2026)
    'QueryCache',
    'QueryCacheEntry',
    'MultiResolutionRenderer',
    'ResolutionPolicy',
    'RenderResolution',
    'RepoCache',
    'RepoCacheEntry',
    'RenderCache',
    'RenderCacheEntry',
]
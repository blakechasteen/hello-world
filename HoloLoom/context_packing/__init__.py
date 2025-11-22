"""
HoloLoom Context Packing System
================================

Physics-based context compression with 40-90% token savings.

Combines:
- Beta wave activation spreading (neuroscience-inspired)
- Multi-signal importance scoring (6 signals)
- Matryoshka-aware compression (multi-scale embeddings)

Author: Claude Code
Date: 2025-11-22

Quick Start:
    >>> from HoloLoom.context_packing import ContextPacker, ContextPackerConfig
    >>>
    >>> config = ContextPackerConfig.balanced()  # 40-60% savings
    >>> packer = ContextPacker(config)
    >>>
    >>> result = packer.pack(
    ...     query="What is Thompson Sampling?",
    ...     candidate_nodes=memory_nodes,
    ...     graph=knowledge_graph,
    ...     target_tokens=2000
    ... )
    >>>
    >>> print(f"Compressed: {result.original_count} -> {result.compressed_count}")
    >>> print(f"Token savings: {result.token_savings}")
    >>> print(f"Compression ratio: {result.compression_ratio:.1%}")
"""

# Protocol definitions
from .protocol import (
    ImportanceSignal,
    ActivationState,
    CompressionResult,
    BetaWave,
    ActivationSpreaderProtocol,
    ImportanceScorerProtocol,
    ContextCompressorProtocol,
    ContextPackerProtocol,
    ActivationMap,
    ImportanceMap,
)

# Configuration
from .config import (
    BetaWaveConfig,
    ImportanceScorerConfig,
    CompressionConfig,
    ContextPackerConfig,
)

# Core implementations (lazy loading for faster imports)
__all__ = [
    # Protocols
    "ImportanceSignal",
    "ActivationState",
    "CompressionResult",
    "BetaWave",
    "ActivationSpreaderProtocol",
    "ImportanceScorerProtocol",
    "ContextCompressorProtocol",
    "ContextPackerProtocol",
    "ActivationMap",
    "ImportanceMap",
    # Configuration
    "BetaWaveConfig",
    "ImportanceScorerConfig",
    "CompressionConfig",
    "ContextPackerConfig",
    # Core implementations (lazy loaded)
    "ActivationSpreader",
    "ImportanceScorer",
    "ContextCompressor",
    "ContextPacker",
]


def __getattr__(name):
    """Lazy loading for core implementations."""
    if name == "ActivationSpreader":
        from .activation_spreader import ActivationSpreader
        globals()[name] = ActivationSpreader
        return ActivationSpreader

    elif name == "ImportanceScorer":
        from .importance_scorer import ImportanceScorer
        globals()[name] = ImportanceScorer
        return ImportanceScorer

    elif name == "ContextCompressor":
        from .context_compressor import ContextCompressor
        globals()[name] = ContextCompressor
        return ContextCompressor

    elif name == "ContextPacker":
        from .packer import ContextPacker
        globals()[name] = ContextPacker
        return ContextPacker

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
HoloLoom Context Packing System
================================

Physics-based context compression with 40-90% token savings.

Combines:
- Beta wave activation spreading (neuroscience-inspired)
- Multi-signal importance scoring (7 signals including MI - Phase 5)
- Matryoshka-aware compression (multi-scale embeddings)
- Information budget compression (Phase 5 - Tishby's Information Bottleneck)

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

Phase 5 Information-Theoretic Packing:
    >>> from HoloLoom.context_packing import information_budget_pack
    >>>
    >>> nodes, scales, mi_scores = information_budget_pack(
    ...     query="What is Thompson Sampling?",
    ...     candidate_nodes=memory_nodes,
    ...     graph=knowledge_graph,
    ...     node_contents=contents,
    ...     information_budget=5.0  # bits
    ... )
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
    # Convenience functions
    "pack_context",
    "adaptive_pack_context",
    "information_budget_pack",
    # Phase 6.1: Adaptive strategy (lazy loaded)
    "AdaptiveComplexity",
    "AdaptiveCompressionStrategy",
    "StrategySelection",
    "select_adaptive_strategy",
    "get_adaptive_mi_budget",
    "get_adaptive_config",
]


def __getattr__(name):
    """Lazy loading for core implementations and convenience functions."""
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

    elif name == "pack_context":
        from .packer import pack_context
        globals()[name] = pack_context
        return pack_context

    elif name == "adaptive_pack_context":
        from .packer import adaptive_pack_context
        globals()[name] = adaptive_pack_context
        return adaptive_pack_context

    elif name == "information_budget_pack":
        from .packer import information_budget_pack
        globals()[name] = information_budget_pack
        return information_budget_pack

    # Phase 6.1: Adaptive strategy
    elif name == "AdaptiveComplexity":
        from .adaptive_strategy import AdaptiveComplexity
        globals()[name] = AdaptiveComplexity
        return AdaptiveComplexity

    elif name == "AdaptiveCompressionStrategy":
        from .adaptive_strategy import AdaptiveCompressionStrategy
        globals()[name] = AdaptiveCompressionStrategy
        return AdaptiveCompressionStrategy

    elif name == "StrategySelection":
        from .adaptive_strategy import StrategySelection
        globals()[name] = StrategySelection
        return StrategySelection

    elif name == "select_adaptive_strategy":
        from .adaptive_strategy import select_adaptive_strategy
        globals()[name] = select_adaptive_strategy
        return select_adaptive_strategy

    elif name == "get_adaptive_mi_budget":
        from .adaptive_strategy import get_adaptive_mi_budget
        globals()[name] = get_adaptive_mi_budget
        return get_adaptive_mi_budget

    elif name == "get_adaptive_config":
        from .adaptive_strategy import get_adaptive_config
        globals()[name] = get_adaptive_config
        return get_adaptive_config

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

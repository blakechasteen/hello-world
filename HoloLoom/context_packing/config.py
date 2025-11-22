"""
Awareness System Configuration
===============================

Configuration classes for Context Packing system.

Author: Claude Code
Date: 2025-11-22
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from .protocol import ImportanceSignal


@dataclass
class BetaWaveConfig:
    """
    Configuration for beta wave activation spreading.

    Beta waves (12-30 Hz) represent focused attention in neuroscience.
    Higher frequency = faster activation spread, less decay.
    """
    frequency: float = 20.0      # Hz, beta range: 12-30
    amplitude: float = 1.0        # Initial activation strength
    decay_rate: float = 0.7       # Decay per hop (0.7 = 30% loss)
    max_hops: int = 3             # Maximum propagation distance
    min_activation: float = 0.1   # Stop propagating below this

    def validate(self):
        """Validate configuration values."""
        assert 12.0 <= self.frequency <= 30.0, "Beta frequency must be 12-30 Hz"
        assert 0.0 < self.amplitude <= 1.0, "Amplitude must be in (0, 1]"
        assert 0.0 < self.decay_rate < 1.0, "Decay rate must be in (0, 1)"
        assert self.max_hops > 0, "Max hops must be positive"
        assert 0.0 < self.min_activation < 1.0, "Min activation must be in (0, 1)"


@dataclass
class ImportanceScorerConfig:
    """
    Configuration for multi-signal importance scoring.

    Weights determine how much each signal contributes to final importance.
    Default weights are balanced for general-purpose use.
    """
    # Signal weights (should sum to 1.0)
    weights: Dict[ImportanceSignal, float] = field(default_factory=lambda: {
        ImportanceSignal.RECENCY: 0.20,        # Recent = important
        ImportanceSignal.RELEVANCE: 0.30,      # Relevant = most important
        ImportanceSignal.CENTRALITY: 0.15,     # Central = moderately important
        ImportanceSignal.ACCESS_FREQUENCY: 0.10,  # Frequent = somewhat important
        ImportanceSignal.CONFIDENCE: 0.15,     # High confidence = important
        ImportanceSignal.HEAT: 0.10            # Hot patterns = important
    })

    # Recency decay (how quickly importance decays with time)
    recency_half_life: float = 3600.0  # 1 hour in seconds

    # Minimum confidence to consider
    min_confidence: float = 0.3

    # Centrality algorithm
    centrality_algorithm: str = "pagerank"  # or "betweenness", "closeness"

    def validate(self):
        """Validate configuration values."""
        total_weight = sum(self.weights.values())
        assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"
        assert self.recency_half_life > 0, "Recency half-life must be positive"
        assert 0.0 <= self.min_confidence <= 1.0, "Min confidence must be in [0, 1]"
        assert self.centrality_algorithm in ["pagerank", "betweenness", "closeness"]


@dataclass
class CompressionConfig:
    """
    Configuration for context compression.

    Controls how aggressively to compress context and which
    Matryoshka scales to use.
    """
    # Target compression ratio (0.5 = keep 50%, drop 50%)
    target_ratio: float = 0.5

    # Matryoshka scales (descending order)
    matryoshka_scales: list = field(default_factory=lambda: [384, 256, 128])

    # Importance thresholds for scale assignment
    # Nodes above high_threshold get full 384D
    # Nodes between mid_threshold and high_threshold get 256D
    # Nodes between low_threshold and mid_threshold get 128D
    # Nodes below low_threshold are dropped
    high_importance_threshold: float = 0.75
    mid_importance_threshold: float = 0.50
    low_importance_threshold: float = 0.25

    # Always preserve top K most important nodes (even if below threshold)
    preserve_top_k: int = 10

    # Estimated tokens per node (for budget calculation)
    avg_tokens_per_node: int = 50

    def validate(self):
        """Validate configuration values."""
        assert 0.0 < self.target_ratio <= 1.0, "Target ratio must be in (0, 1]"
        assert len(self.matryoshka_scales) > 0, "Must have at least one scale"
        assert all(s > 0 for s in self.matryoshka_scales), "Scales must be positive"
        assert self.matryoshka_scales == sorted(self.matryoshka_scales, reverse=True), \
            "Scales must be in descending order"
        assert 0.0 < self.low_importance_threshold < self.mid_importance_threshold < \
               self.high_importance_threshold <= 1.0, \
            "Importance thresholds must be ordered: low < mid < high"
        assert self.preserve_top_k > 0, "preserve_top_k must be positive"
        assert self.avg_tokens_per_node > 0, "avg_tokens_per_node must be positive"


@dataclass
class ContextPackerConfig:
    """
    Main configuration for Context Packer.

    Combines all sub-configurations and adds global settings.
    """
    # Sub-configurations
    beta_wave: BetaWaveConfig = field(default_factory=BetaWaveConfig)
    importance_scorer: ImportanceScorerConfig = field(default_factory=ImportanceScorerConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    # Global settings
    enable_activation_spreading: bool = True
    enable_matryoshka_compression: bool = True
    enable_adaptive_packing: bool = True

    # Token budgets
    default_token_budget: int = 2000
    min_token_budget: int = 500
    max_token_budget: int = 8000

    # Logging and debugging
    verbose: bool = False
    track_metrics: bool = True

    def validate(self):
        """Validate all configurations."""
        self.beta_wave.validate()
        self.importance_scorer.validate()
        self.compression.validate()

        assert self.min_token_budget > 0, "min_token_budget must be positive"
        assert self.min_token_budget <= self.default_token_budget <= self.max_token_budget, \
            "Token budgets must be ordered: min <= default <= max"

    @classmethod
    def aggressive(cls) -> "ContextPackerConfig":
        """
        Aggressive compression preset (60-90% token savings).

        Use when: Token budget is tight, can tolerate some information loss.
        """
        config = cls()
        config.compression.target_ratio = 0.3  # Keep only 30%
        config.compression.high_importance_threshold = 0.85
        config.compression.mid_importance_threshold = 0.60
        config.compression.low_importance_threshold = 0.35
        config.beta_wave.max_hops = 2  # Less spreading
        return config

    @classmethod
    def balanced(cls) -> "ContextPackerConfig":
        """
        Balanced compression preset (40-60% token savings).

        Use when: Good tradeoff between compression and quality.
        Default configuration.
        """
        return cls()  # Uses default values

    @classmethod
    def conservative(cls) -> "ContextPackerConfig":
        """
        Conservative compression preset (20-40% token savings).

        Use when: Quality is paramount, token budget is generous.
        """
        config = cls()
        config.compression.target_ratio = 0.7  # Keep 70%
        config.compression.high_importance_threshold = 0.65
        config.compression.mid_importance_threshold = 0.40
        config.compression.low_importance_threshold = 0.15
        config.beta_wave.max_hops = 4  # More spreading
        config.compression.preserve_top_k = 15
        return config

    @classmethod
    def research(cls) -> "ContextPackerConfig":
        """
        Research preset (maximum spreading, minimal compression).

        Use when: Research mode queries needing maximum context.
        """
        config = cls()
        config.compression.target_ratio = 0.9  # Keep 90%
        config.beta_wave.max_hops = 5
        config.beta_wave.decay_rate = 0.8  # Less decay
        config.compression.preserve_top_k = 20
        config.default_token_budget = 4000
        config.verbose = True
        return config

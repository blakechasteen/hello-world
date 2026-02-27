"""
Dark Trace Configuration: Settings and Presets

Configuration for the DarkTraceEngine and interpretability lenses.
Provides presets for different use cases (fast analysis, full research, safety-focused).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum, auto
from pathlib import Path


class TraceMode(Enum):
    """
    Analysis modes controlling depth vs speed tradeoff.

    FAST: Quick overview, minimal lenses
    STANDARD: Balanced analysis with SAE + Semantic
    FULL: All lenses with correlation tracking
    RESEARCH: Maximum depth, all features, full logging
    SAFETY: Focus on safety-relevant features
    """
    FAST = auto()
    STANDARD = auto()
    FULL = auto()
    RESEARCH = auto()
    SAFETY = auto()


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder lens."""
    enabled: bool = True

    # Architecture
    expansion_factor: int = 16          # Latent dim = input_dim * expansion_factor
    l1_coefficient: float = 3e-4        # L1 sparsity penalty

    # Feature extraction
    activation_threshold: float = 0.1   # Minimum activation to consider "active"
    top_k_features: int = 20            # Max features to return per analysis
    normalize_activations: bool = True  # Normalize before analysis

    # Training (if background training enabled)
    batch_size: int = 32
    learning_rate: float = 1e-3
    buffer_size: int = 1000             # Activation buffer for training

    # Labeling
    auto_label: bool = False            # Use LLM for auto-labeling
    label_cache_path: Optional[str] = None


@dataclass
class SemanticConfig:
    """Configuration for Semantic Axes lens."""
    enabled: bool = True

    # Dimensions
    use_standard_16: bool = True        # Use standard 16 interpretable axes
    use_extended_228: bool = False      # Use full 228 extended dimensions
    custom_dimensions: List[Dict[str, Any]] = field(default_factory=list)

    # Projection
    projection_method: str = "exemplar" # "exemplar" or "learned"
    normalize_output: bool = True

    # Exemplars
    exemplars_per_axis: int = 10        # Number of exemplar pairs per axis


@dataclass
class DomainConfig:
    """Configuration for domain-specific lens."""
    enabled: bool = False               # Off by default

    # Domain definition
    domain_name: Optional[str] = None   # e.g., "medical", "legal", "code"
    config_path: Optional[str] = None   # Path to domain YAML config

    # Learning
    learn_from_data: bool = False       # Learn axes from data (vs config-only)


@dataclass
class CausalConfig:
    """Configuration for causal validation."""
    enabled: bool = True

    # Ablation
    ablation_method: str = "zero"       # "zero", "mean", "resample"
    ablation_samples: int = 10          # Samples for statistical significance

    # Injection
    injection_range: tuple = (-2.0, 2.0)  # Range for injection strength

    # Patching
    patch_all_positions: bool = False   # Patch all token positions vs last only


@dataclass
class SafetyConfig:
    """Configuration for safety monitoring."""
    enabled: bool = True

    # Safety thresholds
    dangerous_activation_threshold: float = 0.5  # Flag features above this
    monitor_all_activations: bool = False        # Monitor all vs flagged only

    # Forbidden features (by ID)
    forbidden_features: Set[str] = field(default_factory=set)

    # Required features (must be active for safety)
    required_features: Set[str] = field(default_factory=set)

    # Reconstruction loss threshold (high = model "confused")
    confusion_threshold: float = 0.5

    # Human-in-the-loop
    escalate_on_dangerous: bool = True


@dataclass
class CorrelationConfig:
    """Configuration for SAE ↔ Semantic correlation tracking."""
    enabled: bool = True

    # Minimum samples for correlation
    min_samples: int = 30

    # Significance threshold
    confidence_threshold: float = 0.9

    # Update frequency
    update_every_n_analyses: int = 10

    # Storage
    persist_correlations: bool = False
    correlation_cache_path: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for Dark Trace logging."""
    enabled: bool = True

    # Verbosity (0=off, 1=summary, 2=detailed, 3=debug)
    verbosity: int = 1

    # Output
    log_to_file: bool = False
    log_path: Optional[str] = None
    log_format: str = "json"            # "json" or "text"

    # What to log
    log_activations: bool = False       # Log all activations (verbose)
    log_steering: bool = True           # Log steering operations
    log_safety: bool = True             # Log safety events


@dataclass
class TraceConfig:
    """
    Complete configuration for DarkTraceEngine.

    Use factory methods for common presets:
    - TraceConfig.fast() - Quick analysis
    - TraceConfig.standard() - Balanced (default)
    - TraceConfig.full() - All features enabled
    - TraceConfig.research() - Maximum depth
    - TraceConfig.safety() - Safety-focused
    """
    # Mode
    mode: TraceMode = TraceMode.STANDARD

    # Input dimensions (set by DarkTraceEngine based on model)
    input_dim: int = 384                # Default for HoloLoom embeddings

    # Lens configurations
    sae: SAEConfig = field(default_factory=SAEConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)

    # System configurations
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Registry
    registry_path: Optional[str] = None  # Path to persist feature registry
    enable_history: bool = True          # Track activation history
    history_size: int = 1000             # History entries per feature

    # Performance
    parallel_lenses: bool = True         # Run lenses in parallel
    cache_projections: bool = True       # Cache lens projections
    timeout_ms: float = 1000.0           # Timeout per lens analysis

    # ---- Factory Methods ----

    @classmethod
    def fast(cls, input_dim: int = 384) -> "TraceConfig":
        """
        Fast analysis preset.

        - SAE only
        - No correlation tracking
        - Minimal logging
        """
        return cls(
            mode=TraceMode.FAST,
            input_dim=input_dim,
            sae=SAEConfig(
                enabled=True,
                top_k_features=10,
                auto_label=False,
            ),
            semantic=SemanticConfig(
                enabled=False,
            ),
            domain=DomainConfig(
                enabled=False,
            ),
            causal=CausalConfig(
                enabled=False,
            ),
            correlation=CorrelationConfig(
                enabled=False,
            ),
            logging=LoggingConfig(
                verbosity=0,
                log_activations=False,
            ),
            enable_history=False,
            timeout_ms=100.0,
        )

    @classmethod
    def standard(cls, input_dim: int = 384) -> "TraceConfig":
        """
        Standard analysis preset (default).

        - SAE + Semantic lenses
        - Correlation tracking
        - Normal logging
        """
        return cls(
            mode=TraceMode.STANDARD,
            input_dim=input_dim,
            sae=SAEConfig(
                enabled=True,
                top_k_features=20,
            ),
            semantic=SemanticConfig(
                enabled=True,
                use_standard_16=True,
                use_extended_228=False,
            ),
            domain=DomainConfig(
                enabled=False,
            ),
            causal=CausalConfig(
                enabled=True,
                ablation_samples=5,
            ),
            correlation=CorrelationConfig(
                enabled=True,
                update_every_n_analyses=10,
            ),
            logging=LoggingConfig(
                verbosity=1,
            ),
        )

    @classmethod
    def full(cls, input_dim: int = 384) -> "TraceConfig":
        """
        Full analysis preset.

        - All lenses enabled
        - Full correlation tracking
        - Detailed logging
        """
        return cls(
            mode=TraceMode.FULL,
            input_dim=input_dim,
            sae=SAEConfig(
                enabled=True,
                top_k_features=50,
                auto_label=True,
            ),
            semantic=SemanticConfig(
                enabled=True,
                use_standard_16=True,
                use_extended_228=True,
            ),
            domain=DomainConfig(
                enabled=True,
            ),
            causal=CausalConfig(
                enabled=True,
                ablation_samples=10,
            ),
            correlation=CorrelationConfig(
                enabled=True,
                update_every_n_analyses=1,
                persist_correlations=True,
            ),
            logging=LoggingConfig(
                verbosity=2,
                log_activations=True,
            ),
            enable_history=True,
            history_size=5000,
            timeout_ms=5000.0,
        )

    @classmethod
    def research(cls, input_dim: int = 384) -> "TraceConfig":
        """
        Research preset.

        - Maximum depth and features
        - No timeout
        - Full logging
        - Publication-ready output
        """
        return cls(
            mode=TraceMode.RESEARCH,
            input_dim=input_dim,
            sae=SAEConfig(
                enabled=True,
                top_k_features=100,
                auto_label=True,
            ),
            semantic=SemanticConfig(
                enabled=True,
                use_standard_16=True,
                use_extended_228=True,
            ),
            domain=DomainConfig(
                enabled=True,
                learn_from_data=True,
            ),
            causal=CausalConfig(
                enabled=True,
                ablation_samples=50,
            ),
            correlation=CorrelationConfig(
                enabled=True,
                update_every_n_analyses=1,
                persist_correlations=True,
                min_samples=10,
            ),
            logging=LoggingConfig(
                verbosity=3,
                log_to_file=True,
                log_activations=True,
            ),
            enable_history=True,
            history_size=10000,
            timeout_ms=float("inf"),
        )

    @classmethod
    def safety(cls, input_dim: int = 384) -> "TraceConfig":
        """
        Safety-focused preset.

        - SAE + Safety monitoring
        - Strict thresholds
        - All safety logging
        """
        return cls(
            mode=TraceMode.SAFETY,
            input_dim=input_dim,
            sae=SAEConfig(
                enabled=True,
                activation_threshold=0.05,  # Lower threshold for safety
                top_k_features=50,
            ),
            semantic=SemanticConfig(
                enabled=True,
                use_standard_16=True,
            ),
            domain=DomainConfig(
                enabled=False,
            ),
            causal=CausalConfig(
                enabled=True,
                ablation_samples=10,
            ),
            safety=SafetyConfig(
                enabled=True,
                dangerous_activation_threshold=0.3,  # Lower = more sensitive
                monitor_all_activations=True,
                escalate_on_dangerous=True,
            ),
            logging=LoggingConfig(
                verbosity=2,
                log_safety=True,
                log_steering=True,
            ),
        )

    # ---- Convenience Methods ----

    def enable_lens(self, lens_name: str) -> "TraceConfig":
        """Enable a specific lens."""
        if lens_name == "sae":
            self.sae.enabled = True
        elif lens_name == "semantic":
            self.semantic.enabled = True
        elif lens_name == "domain":
            self.domain.enabled = True
        elif lens_name == "causal":
            self.causal.enabled = True
        return self

    def disable_lens(self, lens_name: str) -> "TraceConfig":
        """Disable a specific lens."""
        if lens_name == "sae":
            self.sae.enabled = False
        elif lens_name == "semantic":
            self.semantic.enabled = False
        elif lens_name == "domain":
            self.domain.enabled = False
        elif lens_name == "causal":
            self.causal.enabled = False
        return self

    def add_forbidden_feature(self, feature_id: str) -> "TraceConfig":
        """Add a feature to forbidden list."""
        self.safety.forbidden_features.add(feature_id)
        return self

    def add_required_feature(self, feature_id: str) -> "TraceConfig":
        """Add a feature to required list."""
        self.safety.required_features.add(feature_id)
        return self

    def set_domain(self, domain_name: str, config_path: Optional[str] = None) -> "TraceConfig":
        """Configure domain lens."""
        self.domain.enabled = True
        self.domain.domain_name = domain_name
        self.domain.config_path = config_path
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        import dataclasses
        return {
            "mode": self.mode.name,
            "input_dim": self.input_dim,
            "sae": dataclasses.asdict(self.sae),
            "semantic": dataclasses.asdict(self.semantic),
            "domain": dataclasses.asdict(self.domain),
            "causal": dataclasses.asdict(self.causal),
            "safety": {
                **dataclasses.asdict(self.safety),
                "forbidden_features": list(self.safety.forbidden_features),
                "required_features": list(self.safety.required_features),
            },
            "correlation": dataclasses.asdict(self.correlation),
            "logging": dataclasses.asdict(self.logging),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceConfig":
        """Create configuration from dictionary."""
        config = cls()
        config.mode = TraceMode[data.get("mode", "STANDARD")]
        config.input_dim = data.get("input_dim", 384)

        if "sae" in data:
            for k, v in data["sae"].items():
                setattr(config.sae, k, v)

        if "semantic" in data:
            for k, v in data["semantic"].items():
                setattr(config.semantic, k, v)

        if "safety" in data:
            safety_data = data["safety"]
            for k, v in safety_data.items():
                if k == "forbidden_features":
                    config.safety.forbidden_features = set(v)
                elif k == "required_features":
                    config.safety.required_features = set(v)
                else:
                    setattr(config.safety, k, v)

        return config

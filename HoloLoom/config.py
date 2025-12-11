"""
HoloLoom Configuration (Backward Compatible)
============================================

This module provides backward compatibility with the legacy config system
while exposing the new zero-config architecture.

New (Recommended):
    from HoloLoom.config import Config

    # Zero-config - just works
    config = Config()

    # Presets for common use cases
    config = Config.fast()       # Balanced (default)
    config = Config.fused()      # Highest quality
    config = Config.research()   # Experimental features

    # With expansion bundles for research features
    from HoloLoom.expansions.physics import PhysicsConfig

    config = Config.research()
    config.load_expansion(PhysicsConfig(use_gp_bandits=True))

Legacy (Still Works, But Deprecated):
    config = Config(use_gp_bandits=True)  # Auto-loads PhysicsConfig with warning

Migration Guide:
    Old: Config(use_gp_bandits=True, gp_acquisition="thompson")
    New: config = Config.research()
         config.load_expansion(PhysicsConfig(use_gp_bandits=True, gp_acquisition="thompson"))

Date: December 2025
"""

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Import from shared types
from HoloLoom.protocols.types import BanditStrategy


# =============================================================================
# ENUMS (Keep for backward compatibility - these are widely used)
# =============================================================================

class KGBackend(Enum):
    """
    Knowledge Graph backend selection (DEPRECATED - use MemoryBackend).

    - NETWORKX: In-memory NetworkX graph (default, no persistence)
    - NEO4J: Neo4j graph database (persistent, scalable, production-grade)
    """
    NETWORKX = "networkx"
    NEO4J = "neo4j"


class MemoryBackend(Enum):
    """
    Memory backend: INMEMORY (dev), HYBRID (prod), HYPERSPACE (research).

    - INMEMORY: NetworkX in-memory, no deps, <10ms
    - HYBRID: Neo4j+Qdrant, auto-fallback to INMEMORY, ~50ms (DEFAULT)
    - HYPERSPACE: Gated multipass, optional, ~150ms
    """
    INMEMORY = "inmemory"
    HYBRID = "hybrid"
    HYPERSPACE = "hyperspace"


class Environment(Enum):
    """
    Deployment environment - controls safety, logging, and performance settings.

    - DEVELOPMENT: Local dev, auto-approve all, verbose logging, no persistence
    - STAGING: Pre-prod testing, selective approval, moderate logging
    - PRODUCTION: Live deployment, require approval for high-risk, minimal logging
    """
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ExecutionMode(Enum):
    """
    Execution modes for HoloLoom.

    - BARE: Minimal processing (fastest, lowest quality)
    - FAST: Balanced processing (good speed/quality tradeoff)
    - FUSED: Full processing (highest quality)
    - RESEARCH: Experimental features enabled
    """
    BARE = "bare"
    FAST = "fast"
    FUSED = "fused"
    RESEARCH = "research"


# =============================================================================
# LEGACY FIELD MAPPING (for auto-migration)
# =============================================================================

# Maps legacy field names to their expansion bundle
_LEGACY_EXPANSION_MAP = {
    # Physics bundle (GP Bandits + PDE Flow)
    "use_gp_bandits": "physics",
    "gp_acquisition": "physics",
    "gp_kernel_type": "physics",
    "gp_kernel_length_scale": "physics",
    "gp_kernel_variance": "physics",
    "gp_matern_nu": "physics",
    "gp_noise_variance": "physics",
    "gp_ucb_beta": "physics",
    "gp_ucb_adaptive_beta": "physics",
    "gp_n_candidates_per_dim": "physics",
    "gp_update_interval": "physics",
    "gp_warmup_samples": "physics",
    "use_semantic_flow": "physics",
    "pde_type": "physics",
    "flow_dt": "physics",
    "flow_steps": "physics",
    "flow_reaction_type": "physics",
    "flow_diffusion_coef": "physics",
    "flow_wave_speed": "physics",

    # Bayesian bundle
    "use_bayesian": "bayesian",
    "bayesian_samples": "bayesian",
    "bayesian_kl_weight": "bayesian",
    "bayesian_prior_std": "bayesian",

    # Geometry bundle (Riemannian embeddings)
    "use_riemannian": "geometry",
    "riemannian_hyperbolic_dim": "geometry",
    "riemannian_spherical_dim": "geometry",
    "riemannian_euclidean_dim": "geometry",
    "riemannian_hyperbolic_curvature": "geometry",
    "riemannian_spherical_curvature": "geometry",

    # Advanced Spectral bundle
    "use_wavelets": "advanced_spectral",
    "wavelet_scales": "advanced_spectral",
    "wavelet_type": "advanced_spectral",
    "use_diffusion_maps": "advanced_spectral",
    "diffusion_map_dims": "advanced_spectral",
    "diffusion_time": "advanced_spectral",
    "use_multiscale_spectral": "advanced_spectral",
    "multiscale_spectral_scales": "advanced_spectral",
}


def _detect_environment() -> Environment:
    """Auto-detect environment from HOLOLOOM_ENV."""
    env = os.getenv("HOLOLOOM_ENV", "development").lower()
    try:
        return Environment(env)
    except ValueError:
        return Environment.DEVELOPMENT


def _detect_llm() -> tuple:
    """Auto-detect LLM from available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-3-5-sonnet-20241022"
    if os.getenv("OPENAI_API_KEY"):
        return "openai", "gpt-4"
    if os.getenv("OLLAMA_HOST"):
        return "ollama", "llama3.2:3b"
    # Check if Ollama is running locally
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        if result == 0:
            return "ollama", "llama3.2:3b"
    except Exception:
        pass
    return None, None


# =============================================================================
# MODE DEFAULTS (internal - settings derived from execution mode)
# =============================================================================

_MODE_DEFAULTS: Dict[ExecutionMode, Dict[str, Any]] = {
    ExecutionMode.BARE: {
        "scales": [768],
        "fusion_weights": {768: 1.0},
        "n_transformer_layers": 1,
        "n_attention_heads": 2,
        "enable_linguistic_gate": False,
        "enable_zero_copy_embeddings": False,
        "enable_semantic_calculus": False,
        "fast_mode": True,
    },
    ExecutionMode.FAST: {
        "scales": [768],
        "fusion_weights": {768: 1.0},
        "n_transformer_layers": 2,
        "n_attention_heads": 4,
        "enable_linguistic_gate": True,
        "linguistic_mode": "both",
        "use_compositional_cache": True,
        "enable_zero_copy_embeddings": True,
        "enable_semantic_calculus": False,
        "fast_mode": True,
    },
    ExecutionMode.FUSED: {
        "scales": [768],
        "fusion_weights": {768: 1.0},
        "n_transformer_layers": 2,
        "n_attention_heads": 4,
        "enable_linguistic_gate": True,
        "linguistic_mode": "both",
        "use_compositional_cache": True,
        "enable_zero_copy_embeddings": True,
        "enable_semantic_calculus": True,
        "fast_mode": False,
    },
    ExecutionMode.RESEARCH: {
        "scales": [768],
        "fusion_weights": {768: 1.0},
        "n_transformer_layers": 2,
        "n_attention_heads": 4,
        "enable_linguistic_gate": True,
        "linguistic_mode": "both",
        "use_compositional_cache": True,
        "enable_zero_copy_embeddings": True,
        "enable_semantic_calculus": True,
        "use_spring_activation": True,
        "enable_spring_activation": True,  # Alias for use_spring_activation
        "enable_recursive_learning": True,
        "enable_jenny": True,
        "fast_mode": False,
    },
}


# =============================================================================
# MAIN CONFIG CLASS (Zero-Config Architecture)
# =============================================================================

@dataclass
class Config:
    """
    HoloLoom Configuration - Zero-config by default.

    Philosophy: "Convention over Configuration"
    - Tier 0: Config() just works with sensible defaults
    - Tier 1: Presets (fast, fused, research) for common use cases
    - Tier 2: Fine-tuning individual fields only if needed

    Usage:
        # Tier 0: Just works
        config = Config()

        # Tier 1: Named presets
        config = Config.fast()
        config = Config.research()

        # Tier 2: Fine-tuning
        config = Config(retrieval_k=10, pipeline_timeout=10.0)

        # With expansion bundles (research features)
        from HoloLoom.expansions.physics import PhysicsConfig
        config = Config.research()
        config.load_expansion(PhysicsConfig(use_gp_bandits=True))
    """

    # =========================================================================
    # PRIMARY SETTING (the one choice that matters)
    # =========================================================================
    mode: ExecutionMode = ExecutionMode.FAST

    # =========================================================================
    # TIER 1: COMMONLY ADJUSTED (5-10 fields)
    # =========================================================================
    retrieval_k: int = 6
    pipeline_timeout: float = 5.0
    memory_path: str = "data"

    # LLM - auto-detected but overridable
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    # =========================================================================
    # TIER 2: AUTO-CONFIGURED (from environment)
    # =========================================================================
    environment: Environment = field(default_factory=_detect_environment)
    memory_backend: MemoryBackend = MemoryBackend.HYBRID

    # =========================================================================
    # INTERNAL STATE (not user-facing)
    # =========================================================================
    _mode_settings: Dict[str, Any] = field(default_factory=dict, repr=False)
    _expansion_settings: Dict[str, Any] = field(default_factory=dict, repr=False)
    _expansions: List[Any] = field(default_factory=list, repr=False)
    _legacy_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    # =========================================================================
    # BACKWARD COMPATIBILITY FIELDS (deprecated, but preserved)
    # These fields are kept for existing code but should migrate to expansions
    # =========================================================================

    # Basic model/policy settings (still commonly used)
    base_model_name: Optional[str] = None
    n_transformer_layers: int = 2
    n_attention_heads: int = 4
    n_tools: int = 4
    n_adapters: int = 4
    bandit_strategy: BanditStrategy = BanditStrategy.EPSILON_GREEDY
    epsilon: float = 0.1
    blend_neural_weight: float = 0.7

    # Embedding configuration
    scales: List[int] = field(default_factory=lambda: [768])
    fusion_weights: Dict[int, float] = field(default_factory=lambda: {768: 1.0})
    enable_zero_copy_embeddings: bool = False
    zero_copy_cache_path: str = '.cache/embeddings.mmap'
    zero_copy_cache_size: int = 10000

    # Smart Query Routing
    enable_smart_routing: bool = True
    routing_classifier: str = "moonshot"
    enable_semantic_tier: bool = False
    enable_adaptive_learning: bool = True
    enable_classification_telemetry: bool = True
    classification_telemetry_path: str = "./classification_logs"

    # Retrieval settings
    bm25_weight: float = 0.15
    fast_mode: bool = False
    retrieval_timeout: float = 0.2

    # Spring Activation
    use_spring_activation: bool = False
    enable_spring_activation: bool = False  # Alias for use_spring_activation (backward compat)
    spring_stiffness: float = 0.15
    spring_damping: float = 0.85
    spring_decay: float = 0.98
    spring_iterations: int = 200
    spring_convergence_epsilon: float = 1e-4
    spring_activation_threshold: float = 0.1
    spring_seed_count: int = 3

    # Feature extraction
    spectral_k_eigen: int = 4
    svd_components: int = 2

    # Semantic Calculus
    enable_semantic_calculus: bool = False
    semantic_dimensions: int = 16
    semantic_cache_size: int = 10000
    semantic_dt: float = 1.0
    semantic_framework: str = "compassionate"
    semantic_trajectory: bool = True
    semantic_ethics: bool = True

    # Phase 5: Linguistic Gate
    enable_linguistic_gate: bool = False
    linguistic_mode: str = "disabled"
    use_compositional_cache: bool = True
    parse_cache_size: int = 10000
    merge_cache_size: int = 50000
    linguistic_weight: float = 0.3
    prefilter_similarity_threshold: float = 0.3
    prefilter_keep_ratio: float = 0.7

    # Shuttle Integration
    enable_shuttle: bool = True
    shuttle_mode: str = "auto"

    # Beta Wave Context Packing
    enable_beta_wave_packing: bool = False
    packing_token_budget: int = 4000
    packing_query_reserve: int = 400
    packing_response_reserve: int = 1000
    packing_activation_threshold: float = 0.3
    packing_compression_threshold: float = 0.7

    # Recursive Learning
    enable_recursive_learning: bool = False
    recursive_learning_update_interval: float = 60.0
    recursive_learning_refinement_threshold: float = 0.75
    recursive_learning_max_iterations: int = 3
    recursive_learning_enable_background: bool = True
    recursive_learning_enable_hot_patterns: bool = True
    recursive_learning_enable_scratchpad: bool = True

    # Unified Physics
    enable_unified_physics: bool = False
    physics_enable_routing: bool = True
    physics_enable_packing: bool = True
    physics_enable_thermodynamics: bool = True
    physics_enable_wave_mechanics: bool = True
    physics_mode: str = "adaptive"
    physics_track_provenance: bool = True

    # Safety & Environment
    layer6_enabled: bool = False
    enable_safety_guardrails: bool = True
    safety_log_all_decisions: bool = True

    # Conscience Architecture
    enable_conscience: bool = True
    conscience_preset: str = "standard"
    conscience_fail_open: bool = True
    conscience_auto_learn: bool = True
    conscience_learning_interval: float = 60.0
    conscience_persist_path: Optional[str] = None

    # Memory management
    working_memory_size: int = 100
    episodic_buffer_size: int = 100

    # Prometheus Metrics
    enable_prometheus_metrics: bool = True
    prometheus_metrics_port: int = 8001

    # Jenny UI
    enable_jenny: bool = False
    jenny_persist_path: str = "./jenny_specs"
    jenny_default_renderer: str = "html"
    jenny_max_panels_per_query: int = 6
    jenny_auto_lifecycle: bool = True
    jenny_cleanup_interval: float = 60.0
    jenny_enable_mrf: bool = True
    jenny_enable_learning: bool = True
    jenny_learning_persist_path: str = "./jenny_learning"

    # WeaveHouse
    use_weave_house: bool = False
    weave_house_exploration_depth: int = 2
    weave_house_tension_threshold: float = 0.3

    # Dreaming
    enable_dreaming: bool = True
    dream_consolidation_interval: float = 3600.0
    dream_math_bleed_rate: float = 0.3
    dream_pattern_bleed_rate: float = 0.2

    # Neo4j Configuration (auto-read from env)
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "hololoom123"))
    neo4j_database: str = "neo4j"

    # Qdrant Configuration (auto-read from env)
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_collection: str = "hololoom_memories"
    qdrant_use_https: bool = False

    # Hyperspace Configuration
    hyperspace_depth: int = 3
    hyperspace_thresholds: List[float] = field(default_factory=lambda: [0.6, 0.75, 0.85])
    hyperspace_breadth: int = 10

    # Legacy deprecated fields (kept for compatibility)
    kg_backend: Optional[KGBackend] = None
    mem0_api_key: Optional[str] = None
    mem0_org_id: Optional[str] = None
    mem0_project_id: Optional[str] = None

    # =========================================================================
    # EXPANSION BUNDLE FIELDS (backward compatibility)
    # These fields can be set directly or via load_expansion().
    # Recommended: Use expansion bundles instead of setting directly.
    # =========================================================================

    # Physics Bundle (GP Bandits)
    use_gp_bandits: bool = False
    gp_acquisition: str = "thompson"
    gp_kernel_type: str = "matern"
    gp_kernel_length_scale: float = 0.3
    gp_kernel_variance: float = 1.0
    gp_matern_nu: float = 2.5
    gp_noise_variance: float = 0.01
    gp_ucb_beta: float = 2.0
    gp_ucb_adaptive_beta: bool = True
    gp_n_candidates_per_dim: int = 5
    gp_update_interval: int = 10
    gp_warmup_samples: int = 10

    # Physics Bundle (PDE Semantic Flow)
    use_semantic_flow: bool = False
    pde_type: str = "heat"
    flow_dt: float = 0.01
    flow_steps: int = 10
    flow_reaction_type: str = "competitive"
    flow_diffusion_coef: float = 1.0
    flow_wave_speed: float = 1.0

    # Bayesian Bundle
    use_bayesian: bool = False
    bayesian_samples: int = 10
    bayesian_kl_weight: float = 1.0
    bayesian_prior_std: float = 1.0

    # Geometry Bundle (Riemannian Embeddings)
    use_riemannian: bool = False
    riemannian_hyperbolic_dim: int = 256
    riemannian_spherical_dim: int = 256
    riemannian_euclidean_dim: int = 256
    riemannian_hyperbolic_curvature: float = -1.0
    riemannian_spherical_curvature: float = 1.0

    # Advanced Spectral Bundle
    use_wavelets: bool = False
    wavelet_scales: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    wavelet_type: str = "mexican_hat"
    use_diffusion_maps: bool = False
    diffusion_map_dims: int = 32
    diffusion_time: float = 1.0
    use_multiscale_spectral: bool = False
    multiscale_spectral_scales: List[int] = field(default_factory=lambda: [96, 192, 384])

    def __post_init__(self):
        """Initialize mode defaults and auto-detection."""
        # Auto-detect LLM if not specified
        if self.llm_provider is None:
            self.llm_provider, self.llm_model = _detect_llm()

        # Apply mode-specific defaults
        if self.mode in _MODE_DEFAULTS:
            self._mode_settings = _MODE_DEFAULTS[self.mode].copy()
            # Apply mode defaults to actual fields
            for key, value in self._mode_settings.items():
                if hasattr(self, key):
                    # Only apply if not explicitly set (still at default)
                    current = getattr(self, key)
                    default = self.__dataclass_fields__[key].default
                    if current == default or (callable(default) and current == default()):
                        setattr(self, key, value)

        # Validate mode
        if isinstance(self.mode, str):
            self.mode = ExecutionMode(self.mode.lower())

        # Validate scales
        if sorted(self.scales) != self.scales:
            raise ValueError("scales must be in ascending order")

        # Normalize fusion weights
        if self.fusion_weights:
            total = sum(self.fusion_weights.values())
            if not (0.95 <= total <= 1.05):
                for k in self.fusion_weights:
                    self.fusion_weights[k] /= total

    # =========================================================================
    # EXPANSION BUNDLE SUPPORT
    # =========================================================================

    def load_expansion(self, expansion) -> "Config":
        """
        Load an expansion bundle into the config.

        Args:
            expansion: An ExpansionBundle instance (PhysicsConfig, BayesianConfig, etc.)

        Returns:
            self (for chaining)

        Example:
            config = Config.research()
            config.load_expansion(PhysicsConfig(use_gp_bandits=True))
        """
        self._expansions.append(expansion)
        settings = expansion.get_settings()
        self._expansion_settings.update(settings)

        # Also set on actual fields for backward compat
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return self

    def _get(self, key: str, default: Any = None) -> Any:
        """
        Get setting with priority: expansion > mode > explicit > default.

        Internal helper for computed properties.
        """
        if key in self._expansion_settings:
            return self._expansion_settings[key]
        if key in self._mode_settings:
            return self._mode_settings[key]
        if hasattr(self, key):
            return getattr(self, key)
        return default

    # =========================================================================
    # SMART PROPERTIES (Environment-Aware)
    # =========================================================================

    @property
    def safety_testing_mode(self) -> bool:
        """Whether to bypass approval requirements (testing mode)."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def safety_auto_approve_categories(self) -> set:
        """Action categories to auto-approve without human intervention."""
        if self.environment == Environment.DEVELOPMENT:
            return {"query", "retrieval", "analysis", "storage", "modification", "execution", "external"}
        elif self.environment == Environment.STAGING:
            return {"query", "retrieval", "analysis"}
        else:
            return set()

    @property
    def logging_level(self) -> str:
        """Logging verbosity level based on environment."""
        if self.environment == Environment.DEVELOPMENT:
            return "DEBUG"
        elif self.environment == Environment.STAGING:
            return "INFO"
        return "WARNING"

    # =========================================================================
    # PRESETS (Tier 1)
    # =========================================================================

    @classmethod
    def bare(cls) -> "Config":
        """Fastest execution, minimal features."""
        return cls(mode=ExecutionMode.BARE)

    @classmethod
    def fast(cls) -> "Config":
        """Balanced speed and quality (DEFAULT)."""
        return cls(mode=ExecutionMode.FAST)

    @classmethod
    def fused(cls) -> "Config":
        """Highest quality, all standard features enabled."""
        return cls(mode=ExecutionMode.FUSED)

    @classmethod
    def research(cls) -> "Config":
        """
        Experimental features for research.

        Enables semantic calculus, spring activation, recursive learning.
        Use load_expansion() to add research bundles (physics, bayesian, etc.)
        """
        return cls(mode=ExecutionMode.RESEARCH)

    @classmethod
    def multi_perspective(cls) -> "Config":
        """Multi-perspective WeaveHouse system."""
        cfg = cls.fused()
        cfg.use_weave_house = True
        cfg.enable_dreaming = True
        cfg.weave_house_exploration_depth = 2
        cfg.weave_house_tension_threshold = 0.3
        return cfg

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict:
        """Serialize config to dictionary."""
        return {
            'scales': self.scales,
            'fusion_weights': self.fusion_weights,
            'base_model_name': self.base_model_name,
            'mode': self.mode.value,
            'fast_mode': self.fast_mode,
            'memory_path': self.memory_path,
            'n_transformer_layers': self.n_transformer_layers,
            'n_attention_heads': self.n_attention_heads,
            'n_tools': self.n_tools,
            'n_adapters': self.n_adapters,
            'retrieval_k': self.retrieval_k,
            'bm25_weight': self.bm25_weight,
            'spectral_k_eigen': self.spectral_k_eigen,
            'svd_components': self.svd_components,
            'working_memory_size': self.working_memory_size,
            'episodic_buffer_size': self.episodic_buffer_size,
            'pipeline_timeout': self.pipeline_timeout,
            'retrieval_timeout': self.retrieval_timeout,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Config":
        """Deserialize config from dictionary."""
        # Handle mode enum
        if 'mode' in data and isinstance(data['mode'], str):
            data['mode'] = ExecutionMode(data['mode'])
        return cls(**data)


# =============================================================================
# LEGACY COMPATIBILITY EXPORTS
# =============================================================================

# These are re-exported for backward compatibility
__all__ = [
    "Config",
    "ExecutionMode",
    "Environment",
    "MemoryBackend",
    "KGBackend",
    "BanditStrategy",
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=== HoloLoom Configuration Examples ===\n")

    # Zero-config (just works)
    print("1. Zero-Config (Default):")
    cfg = Config()
    print(f"   Mode: {cfg.mode.value}")
    print(f"   LLM: {cfg.llm_provider} / {cfg.llm_model}")
    print(f"   Environment: {cfg.environment.value}")

    # Presets
    print("\n2. Preset: Fast Mode:")
    cfg_fast = Config.fast()
    print(f"   Mode: {cfg_fast.mode.value}")
    print(f"   Linguistic Gate: {cfg_fast.enable_linguistic_gate}")
    print(f"   Zero-Copy: {cfg_fast.enable_zero_copy_embeddings}")

    print("\n3. Preset: Research Mode:")
    cfg_research = Config.research()
    print(f"   Mode: {cfg_research.mode.value}")
    print(f"   Semantic Calculus: {cfg_research.enable_semantic_calculus}")

    # With expansion bundles
    print("\n4. Research with Expansion Bundle:")
    try:
        from HoloLoom.expansions.physics import PhysicsConfig
        cfg_physics = Config.research()
        cfg_physics.load_expansion(PhysicsConfig(use_gp_bandits=True))
        print(f"   GP Bandits: {cfg_physics.use_gp_bandits}")
        print(f"   GP Acquisition: {cfg_physics.gp_acquisition}")
    except ImportError:
        print("   (Expansion bundle not available)")

    # Fine-tuning
    print("\n5. Fine-Tuning:")
    cfg_custom = Config(
        retrieval_k=10,
        pipeline_timeout=10.0,
        mode=ExecutionMode.FUSED
    )
    print(f"   Retrieval K: {cfg_custom.retrieval_k}")
    print(f"   Timeout: {cfg_custom.pipeline_timeout}s")

    print("\n✓ All config examples complete!")

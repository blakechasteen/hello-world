"""
HoloLoom Configuration
======================
Configuration settings for the HoloLoom system.

Defines execution modes, model settings, and system parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# Import BanditStrategy from shared types (no circular dependency!)
from HoloLoom.protocols.types import BanditStrategy


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
      - Regex-only motif detection
      - No spectral features
      - Fast single-scale retrieval
      - Simple policy
    
    - FAST: Balanced processing (good speed/quality tradeoff)
      - Hybrid motif detection (regex + spaCy if available)
      - Spectral features enabled
      - Fast retrieval with smallest scale
      - Neural policy
    
    - FUSED: Full processing (highest quality)
      - Full hybrid motif detection
      - All spectral features
      - Multi-scale fused retrieval
      - Full neural policy with all adapters
    """
    BARE = "bare"
    FAST = "fast"
    FUSED = "fused"


@dataclass
class Config:
    """
    Configuration for HoloLoom orchestrator.
    
    Controls all aspects of system behavior including:
    - Embedding dimensions (Matryoshka scales)
    - Fusion weights for multi-scale retrieval
    - Model selection
    - Execution mode
    - Persistence settings
    """
    
    # Embedding configuration
    scales: List[int] = field(default_factory=lambda: [768])
    fusion_weights: Dict[int, float] = field(default_factory=lambda: {
        768: 1.0   # Single scale: 100% weight (simplified from multi-scale)
    })

    # Zero-Copy Embeddings (November 2025) - 30-50x faster, 50% memory savings
    enable_zero_copy_embeddings: bool = False  # Enable zero-copy memory-mapped embeddings
    zero_copy_cache_path: str = '.cache/embeddings.mmap'  # Persistent cache location
    zero_copy_cache_size: int = 10000  # Maximum embeddings to cache

    # Smart Query Routing (November 2025) - 100% accuracy, <1ms latency
    enable_smart_routing: bool = True  # Enable smart query routing with fast paths
    routing_classifier: str = "moonshot"  # "baseline" or "moonshot"
    enable_semantic_tier: bool = False  # Tier 3 semantic embeddings (adds 20ms, 98% accuracy)
    enable_adaptive_learning: bool = True  # Learn from production misclassifications
    enable_classification_telemetry: bool = True  # Log classifications for monitoring
    classification_telemetry_path: str = "./classification_logs"  # Telemetry log directory

    # Model selection
    base_model_name: Optional[str] = None  # Uses env var HOLOLOOM_BASE_ENCODER if None

    # LLM Provider Configuration (for metaprompting, agentic reasoning, LLM-enhanced features)
    llm_provider: Optional[str] = None  # LLM provider: 'anthropic', 'google', 'openai', 'ollama'
    llm_model: Optional[str] = None  # Model name (e.g., 'claude-3-5-sonnet-20241022', 'gpt-4', 'gemini-pro')

    # Execution mode
    mode: ExecutionMode = ExecutionMode.FUSED
    fast_mode: bool = False  # If True, force fast retrieval regardless of mode
    
    # Persistence
    memory_path: Optional[str] = "data"  # Root directory for memory storage

    # Memory Backend Selection (Unified)
    memory_backend: 'MemoryBackend' = None  # Set in __post_init__ to avoid forward reference

    # Legacy: Knowledge Graph backend (deprecated, use memory_backend)
    kg_backend: Optional[KGBackend] = None

    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "hololoom123"
    neo4j_database: str = "neo4j"

    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "hololoom_memories"
    qdrant_use_https: bool = False

    # Mem0 Configuration
    mem0_api_key: Optional[str] = None  # If using Mem0 cloud
    mem0_org_id: Optional[str] = None
    mem0_project_id: Optional[str] = None

    # Hyperspace Configuration (for HYPERSPACE backend)
    hyperspace_depth: int = 3  # Max recursion depth
    hyperspace_thresholds: List[float] = field(default_factory=lambda: [0.6, 0.75, 0.85])
    hyperspace_breadth: int = 10  # Links per level
    
    # Neural network settings
    n_transformer_layers: int = 2
    n_attention_heads: int = 4
    
    # Policy settings
    n_tools: int = 4  # answer, search, notion_write, calc
    n_adapters: int = 4  # general, farm, brewing, mirrorcore
    bandit_strategy: BanditStrategy = BanditStrategy.EPSILON_GREEDY
    epsilon: float = 0.1  # Exploration rate for epsilon-greedy (10%)
    blend_neural_weight: float = 0.7  # Neural weight in Bayesian blend (30% bandit)

    # Bayesian Policy Settings (Priority 5 - Variational Inference)
    use_bayesian: bool = False  # Enable Bayesian uncertainty quantification
    bayesian_samples: int = 10  # MC samples for uncertainty estimation (10× overhead)
    bayesian_kl_weight: float = 1.0  # KL divergence weight in ELBO
    bayesian_prior_std: float = 1.0  # Prior weight standard deviation

    # GP Bandit Settings (for continuous action spaces)
    use_gp_bandits: bool = False  # Enable Gaussian Process bandits
    gp_acquisition: str = "thompson"  # GP acquisition: "thompson" or "ucb"
    gp_kernel_type: str = "matern"  # GP kernel: "matern" or "rbf"
    gp_kernel_length_scale: float = 0.3  # GP kernel length scale
    gp_kernel_variance: float = 1.0  # GP kernel variance
    gp_matern_nu: float = 2.5  # Matérn kernel smoothness (1.5, 2.5, 5.0)
    gp_noise_variance: float = 0.01  # GP observation noise
    gp_ucb_beta: float = 2.0  # UCB exploration parameter
    gp_ucb_adaptive_beta: bool = True  # Use adaptive β = √(2 log(t))
    gp_n_candidates_per_dim: int = 5  # Discretization resolution
    gp_update_interval: int = 10  # Retrain GP every N observations
    
    # Retrieval settings
    retrieval_k: int = 6  # Number of shards to retrieve
    bm25_weight: float = 0.15  # Weight of BM25 in fused retrieval

    # Spring Activation Retrieval (optional, modular)
    use_spring_activation: bool = False  # Enable physics-based spreading activation
    spring_stiffness: float = 0.15  # k: Spring stiffness (0.05-0.5 typical)
    spring_damping: float = 0.85  # c: Damping coefficient (0.5-0.95 typical)
    spring_decay: float = 0.98  # Activation decay per step (0.90-0.99)
    spring_iterations: int = 200  # Max propagation steps
    spring_convergence_epsilon: float = 1e-4  # Energy change threshold
    spring_activation_threshold: float = 0.1  # Min activation to retrieve
    spring_seed_count: int = 3  # Number of seed nodes from embedding
    
    # Feature extraction
    spectral_k_eigen: int = 4  # Number of Laplacian eigenvalues
    svd_components: int = 2  # Number of SVD topic components

    # Advanced Spectral Methods (Priority 4 - Mathematical Moonshot)
    use_wavelets: bool = False  # Enable multi-scale wavelet features (adds ~10-50ms, O(n³) complexity)
    wavelet_scales: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])  # Coarse → Fine
    use_diffusion_maps: bool = False  # Enable diffusion geometry (adds ~20-100ms, cached)
    diffusion_map_dims: int = 32  # Diffusion embedding dimension
    use_multiscale_spectral: bool = False  # Enable hierarchical spectral analysis (experimental)
    multiscale_spectral_scales: List[int] = field(default_factory=lambda: [96, 192, 384])  # Match Matryoshka scales

    # Semantic Calculus (optional)
    enable_semantic_calculus: bool = False  # Enable semantic flow analysis
    semantic_dimensions: int = 16  # Number of semantic dimension pairs
    semantic_cache_size: int = 10000  # Embedding cache size
    semantic_dt: float = 1.0  # Time step for calculus
    semantic_framework: str = "compassionate"  # Ethical framework: compassionate, scientific, therapeutic
    semantic_trajectory: bool = True  # Compute velocity/acceleration/curvature
    semantic_ethics: bool = True  # Run ethical analysis

    # PDE Semantic Flow (Priority 6 - Temporal Dynamics)
    use_semantic_flow: bool = False  # Enable PDE-based temporal evolution (research mode only - expensive!)
    pde_type: str = "heat"  # PDE type: heat (diffusion), wave (oscillation), reaction_diffusion (competition), hamilton_jacobi (optimal paths)
    flow_dt: float = 0.01  # PDE timestep (smaller = more accurate, more expensive)
    flow_steps: int = 10  # Evolution steps between queries
    flow_reaction_type: str = "competitive"  # Reaction type for reaction_diffusion: logistic, competitive, cubic
    flow_diffusion_coef: float = 1.0  # Diffusion coefficient for reaction_diffusion
    flow_wave_speed: float = 1.0  # Wave speed for wave equation

    # Phase 5: Universal Grammar + Compositional Cache (optional)
    enable_linguistic_gate: bool = False  # Enable Phase 5 linguistic matryoshka gate
    linguistic_mode: str = "disabled"  # Linguistic filter mode: disabled, prefilter, embedding, both
    use_compositional_cache: bool = True  # 3-tier compositional cache (when linguistic_gate enabled)
    parse_cache_size: int = 10000  # X-bar structure cache size
    merge_cache_size: int = 50000  # Compositional embedding cache size
    linguistic_weight: float = 0.3  # Weight for linguistic features (0-1)
    prefilter_similarity_threshold: float = 0.3  # Min syntactic similarity for pre-filter
    prefilter_keep_ratio: float = 0.7  # Keep top 70% of candidates after linguistic filter

    # Shuttle Integration (January 2025) - MCTS-powered Warp↔Yarn intersection
    enable_shuttle: bool = True  # Enable Shuttle for intelligent thread selection at Step 3
    shuttle_mode: str = "auto"  # Shuttle mode: "auto" (derive from execution mode), "full", "lite", "minimal"

    # Priority 2: Riemannian Embeddings (Mathematical Moonshot)
    use_riemannian: bool = False  # Enable Riemannian manifold structure for embeddings
    riemannian_hyperbolic_dim: int = 256  # Dimension for hierarchical concepts (K < 0)
    riemannian_spherical_dim: int = 256   # Dimension for clustered concepts (K > 0)
    riemannian_euclidean_dim: int = 256   # Dimension for linear features (K = 0)
    riemannian_hyperbolic_curvature: float = -1.0  # Negative curvature for hierarchies
    riemannian_spherical_curvature: float = 1.0    # Positive curvature for clusters

    # Beta Wave Context Packing (optional - requires MultiWaveMemoryEngine)
    enable_beta_wave_packing: bool = False  # Enable physics-based context optimization
    packing_token_budget: int = 4000  # Total token budget for packed context
    packing_query_reserve: int = 400  # Tokens reserved for query
    packing_response_reserve: int = 1000  # Tokens reserved for LLM response
    packing_activation_threshold: float = 0.3  # Min activation to include (low filtered out)
    packing_compression_threshold: float = 0.7  # Activation threshold for compression vs full content

    # Recursive Learning System (Phase 1-5) - Self-Improving Intelligence
    enable_recursive_learning: bool = False  # Enable integrated recursive learning (pattern learning, refinement, etc.)
    recursive_learning_update_interval: float = 60.0  # Background learning update interval (seconds)
    recursive_learning_refinement_threshold: float = 0.75  # Confidence threshold below which to trigger refinement
    recursive_learning_max_iterations: int = 3  # Maximum refinement iterations for low-confidence queries
    recursive_learning_enable_background: bool = True  # Enable background learning thread (Thompson Sampling, policy updates)
    recursive_learning_enable_hot_patterns: bool = True  # Enable hot pattern tracking and adaptive retrieval
    recursive_learning_enable_scratchpad: bool = True  # Enable provenance tracking via scratchpad

    # Unified Physics Engine (Phases 1-4) - Physics-Based Intelligence
    enable_unified_physics: bool = False  # Enable unified physics engine (routing, packing, thermodynamics, wave mechanics)
    physics_enable_routing: bool = True  # Phase 1: Gradient flow routing (optimal tool selection)
    physics_enable_packing: bool = True  # Phase 2: Fluid dynamics packing (context optimization)
    physics_enable_thermodynamics: bool = True  # Phase 3: Thermodynamic exploration (F = E - T*S)
    physics_enable_wave_mechanics: bool = True  # Phase 4: Wave mechanics (pattern interference detection)
    physics_mode: str = "adaptive"  # Physics integration mode: "sequential", "parallel", "adaptive"
    physics_track_provenance: bool = True  # Track complete physics provenance in spacetime metadata

    # Layer 6: Self-Modification (LOCKED - requires research environment)
    # See README_SAFETY.md for unlock instructions
    # Requires BOTH environment variable AND this flag:
    #   export HOLOLOOM_LAYER6_UNLOCK=research
    #   config.layer6_enabled = True
    layer6_enabled: bool = False  # DO NOT enable without proper research infrastructure

    # Deployment Environment (controls safety, logging, performance)
    # Set via: config.environment = Environment.DEVELOPMENT (or STAGING, PRODUCTION)
    # Or via env var: HOLOLOOM_ENV=development (default if not set)
    environment: Environment = Environment.DEVELOPMENT

    # Layer 6 Safety Guardrails (environment-aware)
    # These properties automatically adjust based on environment:
    # - DEVELOPMENT: Auto-approve all, verbose logging
    # - STAGING: Auto-approve safe actions, require approval for risky
    # - PRODUCTION: Require approval for all high-risk actions
    enable_safety_guardrails: bool = True  # Master switch for safety system
    safety_log_all_decisions: bool = True  # Log every safety decision (for audit)

    # Conscience Architecture (December 2025) - Unified Safety with consider/witness/learn
    # Modular conscience integration for agentic reasoning and weaving orchestrator
    # Toggle at: config, constructor, per-request, or preset level
    enable_conscience: bool = True  # Master switch for conscience safety layer
    conscience_preset: str = "standard"  # Lens preset: "standard", "paranoid", "research"
    conscience_fail_open: bool = True  # Graceful degradation (fail-open for availability)
    conscience_auto_learn: bool = True  # Auto-learn from execution outcomes
    conscience_learning_interval: float = 60.0  # Background learning interval (seconds)
    conscience_persist_path: Optional[str] = None  # Path for conscience wisdom persistence

    # Memory management
    working_memory_size: int = 100  # Cache size for recent queries
    episodic_buffer_size: int = 100  # Size of recent interaction buffer
    
    # Timeouts (seconds)
    pipeline_timeout: float = 5.0  # Max time for full pipeline
    retrieval_timeout: float = 0.2  # Max time for retrieval (200ms - reduced from 2s)

    # Prometheus Metrics (production monitoring)
    enable_prometheus_metrics: bool = True  # Enable Prometheus metrics collection
    prometheus_metrics_port: int = 8001  # Port for metrics HTTP endpoint

    # Jenny Generative UI Runtime (December 2025)
    enable_jenny: bool = False  # Enable Jenny UI panel generation
    jenny_persist_path: str = "./jenny_specs"  # SpecLedger persistence directory
    jenny_default_renderer: str = "html"  # Default renderer: "html", "terminal", "json"
    jenny_max_panels_per_query: int = 6  # Maximum panels per weave
    jenny_auto_lifecycle: bool = True  # Auto-transition NASCENT → STABLE
    jenny_cleanup_interval: float = 60.0  # Cleanup interval for DISSOLVING panels (seconds)

    # Jenny MRF Integration (Phase 2.1-2.2: Thompson Sampling panel learning - December 2025)
    jenny_enable_mrf: bool = True  # Enable MRF-enhanced panel compilation
    jenny_enable_learning: bool = True  # Enable Thompson Sampling learning from user actions
    jenny_learning_persist_path: str = "./jenny_learning"  # Learning state persistence directory

    def __post_init__(self):
        """Validate configuration."""
        # Set defaults
        if self.memory_backend is None:
            # Default to HYBRID for all modes (production-ready with auto-fallback)
            self.memory_backend = MemoryBackend.HYBRID

        # Ensure scales are sorted
        if sorted(self.scales) != self.scales:
            raise ValueError("scales must be in ascending order")

        # Ensure fusion weights sum to approximately 1.0
        if self.fusion_weights:
            total_weight = sum(self.fusion_weights.values())
            if not (0.95 <= total_weight <= 1.05):
                import warnings
                warnings.warn(
                    f"Fusion weights sum to {total_weight:.3f}, expected ~1.0. "
                    "Weights will be automatically normalized."
                )
                # Normalize
                for k in self.fusion_weights:
                    self.fusion_weights[k] /= total_weight

        # Validate mode
        if isinstance(self.mode, str):
            self.mode = ExecutionMode(self.mode.lower())

        # Validate hyperspace thresholds
        if self.memory_backend == MemoryBackend.HYPERSPACE:
            if len(self.hyperspace_thresholds) != self.hyperspace_depth:
                import warnings
                warnings.warn(
                    f"hyperspace_thresholds length ({len(self.hyperspace_thresholds)}) "
                    f"should match hyperspace_depth ({self.hyperspace_depth})"
                )

    # ============================================================================
    # Smart Properties (Environment-Aware)
    # ============================================================================

    @property
    def safety_testing_mode(self) -> bool:
        """
        Whether to bypass approval requirements (testing mode).

        Auto-determined by environment:
        - DEVELOPMENT: True (auto-approve everything)
        - STAGING: False (require approval for high-risk)
        - PRODUCTION: False (require approval for high-risk)

        Returns:
            True if in development mode
        """
        return self.environment == Environment.DEVELOPMENT

    @property
    def safety_auto_approve_categories(self) -> set:
        """
        Action categories to auto-approve without human intervention.

        Environment-specific:
        - DEVELOPMENT: All categories (full auto-approve)
        - STAGING: Read-only categories (query, retrieval, analysis)
        - PRODUCTION: None (require approval for all)

        Returns:
            Set of ActionCategory names to auto-approve
        """
        if self.environment == Environment.DEVELOPMENT:
            # Development: Auto-approve EVERYTHING for fast iteration
            return {"query", "retrieval", "analysis", "storage", "modification", "execution", "external"}
        elif self.environment == Environment.STAGING:
            # Staging: Auto-approve safe read-only operations
            return {"query", "retrieval", "analysis"}
        else:  # PRODUCTION
            # Production: Require approval for ALL high-risk actions
            return set()

    @property
    def logging_level(self) -> str:
        """
        Logging verbosity level based on environment.

        - DEVELOPMENT: DEBUG (show everything)
        - STAGING: INFO (moderate detail)
        - PRODUCTION: WARNING (errors and warnings only)

        Returns:
            Logging level string
        """
        if self.environment == Environment.DEVELOPMENT:
            return "DEBUG"
        elif self.environment == Environment.STAGING:
            return "INFO"
        else:  # PRODUCTION
            return "WARNING"

    @classmethod
    def bare(cls) -> 'Config':
        """Create a bare-mode configuration (fastest)."""
        return cls(
            scales=[768],
            fusion_weights={768: 1.0},
            mode=ExecutionMode.BARE,
            fast_mode=True,
            n_transformer_layers=1,
            n_attention_heads=2,
            enable_semantic_calculus=False  # Disabled for speed
        )

    @classmethod
    def fast(cls) -> 'Config':
        """Create a fast-mode configuration (balanced)."""
        return cls(
            scales=[768],
            fusion_weights={768: 1.0},
            mode=ExecutionMode.FAST,
            fast_mode=True,
            n_transformer_layers=2,
            n_attention_heads=4,
            enable_semantic_calculus=False,  # Disabled by default (user can enable)
            semantic_dimensions=8,  # Fewer dimensions if enabled
            semantic_ethics=False,  # Skip ethics for speed
            # Phase 5: Enable compositional cache (10-300× speedup)
            enable_linguistic_gate=True,
            linguistic_mode="both",
            use_compositional_cache=True,
            # Zero-copy embeddings (1.4x speedup, 50% memory savings)
            enable_zero_copy_embeddings=True
        )

    @classmethod
    def fused(cls) -> 'Config':
        """Create a fused-mode configuration (highest quality)."""
        return cls(
            scales=[768],
            fusion_weights={768: 1.0},
            mode=ExecutionMode.FUSED,
            fast_mode=False,
            n_transformer_layers=2,
            n_attention_heads=4,
            enable_semantic_calculus=False,  # Disabled by default (user can enable)
            semantic_dimensions=16,  # Full dimensions if enabled
            semantic_ethics=True,  # Full ethics if enabled
            # Phase 5: Enable compositional cache (10-300× speedup)
            enable_linguistic_gate=True,
            linguistic_mode="both",
            use_compositional_cache=True,
            # Zero-copy embeddings (1.4x speedup, 50% memory savings)
            enable_zero_copy_embeddings=True
        )
    
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
            'retrieval_timeout': self.retrieval_timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Config':
        """Deserialize config from dictionary."""
        return cls(**data)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== HoloLoom Configuration Examples ===\n")
    
    # Default (fused mode)
    print("1. Default Config (Fused):")
    cfg_default = Config()
    print(f"   Mode: {cfg_default.mode.value}")
    print(f"   Scales: {cfg_default.scales}")
    print(f"   Fusion weights: {cfg_default.fusion_weights}")
    
    # Bare mode (fastest)
    print("\n2. Bare Mode (Fastest):")
    cfg_bare = Config.bare()
    print(f"   Mode: {cfg_bare.mode.value}")
    print(f"   Scales: {cfg_bare.scales}")
    print(f"   Layers: {cfg_bare.n_transformer_layers}")
    
    # Fast mode (balanced)
    print("\n3. Fast Mode (Balanced):")
    cfg_fast = Config.fast()
    print(f"   Mode: {cfg_fast.mode.value}")
    print(f"   Scales: {cfg_fast.scales}")
    print(f"   Fusion weights: {cfg_fast.fusion_weights}")
    
    # Custom config
    print("\n4. Custom Config:")
    cfg_custom = Config(
        scales=[128, 256],
        mode=ExecutionMode.FAST,
        memory_path="custom_data",
        retrieval_k=10
    )
    print(f"   Mode: {cfg_custom.mode.value}")
    print(f"   Scales: {cfg_custom.scales}")
    print(f"   Retrieval K: {cfg_custom.retrieval_k}")
    
    # Serialization
    print("\n5. Serialization:")
    data = cfg_default.to_dict()
    print(f"   Serialized keys: {list(data.keys())[:5]}...")
    cfg_restored = Config.from_dict(data)
    print(f"   Restored mode: {cfg_restored.mode.value}")
    
    print("\n✓ All config examples complete!")
"""
HoloLoom Configuration
======================
Configuration settings for the HoloLoom system.

Defines execution modes, model settings, and system parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# Import BanditStrategy from policy to avoid duplication
try:
    from HoloLoom.policy.unified import BanditStrategy as PolicyBanditStrategy
    _POLICY_AVAILABLE = True
except ImportError:
    _POLICY_AVAILABLE = False


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


# Use BanditStrategy from policy module to avoid duplication
# If policy is not available (rare edge case), define a local version
if _POLICY_AVAILABLE:
    BanditStrategy = PolicyBanditStrategy
else:
    class BanditStrategy(Enum):
        """
        Bandit exploration strategies for tool selection.

        - EPSILON_GREEDY: Explore with probability epsilon (default 10%)
        - BAYESIAN_BLEND: Blend neural predictions with bandit priors
        - PURE_THOMPSON: Use Thompson Sampling exclusively
        """
        EPSILON_GREEDY = "epsilon_greedy"
        BAYESIAN_BLEND = "bayesian_blend"
        PURE_THOMPSON = "pure_thompson"


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
    
    # Model selection
    base_model_name: Optional[str] = None  # Uses env var HOLOLOOM_BASE_ENCODER if None
    
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

    # Semantic Calculus (optional)
    enable_semantic_calculus: bool = False  # Enable semantic flow analysis
    semantic_dimensions: int = 16  # Number of semantic dimension pairs
    semantic_cache_size: int = 10000  # Embedding cache size
    semantic_dt: float = 1.0  # Time step for calculus
    semantic_framework: str = "compassionate"  # Ethical framework: compassionate, scientific, therapeutic
    semantic_trajectory: bool = True  # Compute velocity/acceleration/curvature
    semantic_ethics: bool = True  # Run ethical analysis

    # Phase 5: Universal Grammar + Compositional Cache (optional)
    enable_linguistic_gate: bool = False  # Enable Phase 5 linguistic matryoshka gate
    linguistic_mode: str = "disabled"  # Linguistic filter mode: disabled, prefilter, embedding, both
    use_compositional_cache: bool = True  # 3-tier compositional cache (when linguistic_gate enabled)
    parse_cache_size: int = 10000  # X-bar structure cache size
    merge_cache_size: int = 50000  # Compositional embedding cache size
    linguistic_weight: float = 0.3  # Weight for linguistic features (0-1)
    prefilter_similarity_threshold: float = 0.3  # Min syntactic similarity for pre-filter
    prefilter_keep_ratio: float = 0.7  # Keep top 70% of candidates after linguistic filter

    # Memory management
    working_memory_size: int = 100  # Cache size for recent queries
    episodic_buffer_size: int = 100  # Size of recent interaction buffer
    
    # Timeouts (seconds)
    pipeline_timeout: float = 5.0  # Max time for full pipeline
    retrieval_timeout: float = 2.0  # Max time for retrieval
    
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
            semantic_ethics=False  # Skip ethics for speed
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
            semantic_ethics=True  # Full ethics if enabled
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
"""
HoloLoom Configuration
======================
Configuration settings for the HoloLoom system.

Defines execution modes, model settings, and system parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class KGBackend(Enum):
    """
    Knowledge Graph backend selection.

    - NETWORKX: In-memory NetworkX graph (default, no persistence)
    - NEO4J: Neo4j graph database (persistent, scalable, production-grade)
    """
    NETWORKX = "networkx"
    NEO4J = "neo4j"


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
    scales: List[int] = field(default_factory=lambda: [96, 192, 384])
    fusion_weights: Dict[int, float] = field(default_factory=lambda: {
        96: 0.25,   # Smallest scale: 25% weight
        192: 0.35,  # Medium scale: 35% weight
        384: 0.40   # Largest scale: 40% weight
    })
    
    # Model selection
    base_model_name: Optional[str] = None  # Uses env var HOLOLOOM_BASE_ENCODER if None
    
    # Execution mode
    mode: ExecutionMode = ExecutionMode.FUSED
    fast_mode: bool = False  # If True, force fast retrieval regardless of mode
    
    # Persistence
    memory_path: Optional[str] = "data"  # Root directory for memory storage

    # Knowledge Graph backend
    kg_backend: KGBackend = KGBackend.NETWORKX
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "hololoom123"
    neo4j_database: str = "neo4j"
    
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
    
    # Feature extraction
    spectral_k_eigen: int = 4  # Number of Laplacian eigenvalues
    svd_components: int = 2  # Number of SVD topic components
    
    # Memory management
    working_memory_size: int = 100  # Cache size for recent queries
    episodic_buffer_size: int = 100  # Size of recent interaction buffer
    
    # Timeouts (seconds)
    pipeline_timeout: float = 5.0  # Max time for full pipeline
    retrieval_timeout: float = 2.0  # Max time for retrieval
    
    def __post_init__(self):
        """Validate configuration."""
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
    
    @classmethod
    def bare(cls) -> 'Config':
        """Create a bare-mode configuration (fastest)."""
        return cls(
            scales=[96],
            fusion_weights={96: 1.0},
            mode=ExecutionMode.BARE,
            fast_mode=True,
            n_transformer_layers=1,
            n_attention_heads=2
        )
    
    @classmethod
    def fast(cls) -> 'Config':
        """Create a fast-mode configuration (balanced)."""
        return cls(
            scales=[96, 192],
            fusion_weights={96: 0.4, 192: 0.6},
            mode=ExecutionMode.FAST,
            fast_mode=True,
            n_transformer_layers=2,
            n_attention_heads=4
        )
    
    @classmethod
    def fused(cls) -> 'Config':
        """Create a fused-mode configuration (highest quality)."""
        return cls(
            scales=[96, 192, 384],
            fusion_weights={96: 0.25, 192: 0.35, 384: 0.40},
            mode=ExecutionMode.FUSED,
            fast_mode=False,
            n_transformer_layers=2,
            n_attention_heads=4
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
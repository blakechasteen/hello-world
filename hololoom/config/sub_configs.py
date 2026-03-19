"""
Config Sub-Models
=================

Frozen dataclasses for each configuration domain.
Created in PR 5 (B-1). Wired into Config in PR 7 (B-3).

Each sub-config groups related settings with stripped prefixes.
For example, ``jenny_persist_path`` becomes ``JennyConfig.persist_path``.

Naming convention:
    - ``enable: bool`` is always the first field (where applicable)
    - Field names drop the category prefix from the flat config
    - Defaults match the current Config() defaults exactly
"""

from dataclasses import dataclass

from hololoom.protocols.types import BanditStrategy

# =========================================================================
# Embedding & Feature Extraction
# =========================================================================

@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding pipeline settings."""
    enable_zero_copy: bool = False
    zero_copy_cache_path: str = '.cache/embeddings.mmap'
    zero_copy_cache_size: int = 10000
    spectral_k_eigen: int = 4
    svd_components: int = 2


# =========================================================================
# Smart Query Routing
# =========================================================================

@dataclass(frozen=True)
class RoutingConfig:
    """Smart query routing settings."""
    enable: bool = True
    enable_semantic_tier: bool = False
    enable_adaptive_learning: bool = True
    enable_classification_telemetry: bool = True
    classification_telemetry_path: str = "./classification_logs"


# =========================================================================
# Retrieval
# =========================================================================

@dataclass(frozen=True)
class RetrievalConfig:
    """Memory retrieval settings."""
    k: int = 6
    bm25_weight: float = 0.15
    timeout: float = 0.2


# =========================================================================
# Recursive Learning
# =========================================================================

@dataclass(frozen=True)
class RecursiveLearningConfig:
    """Recursive learning loop settings."""
    enable: bool = False
    update_interval: float = 60.0
    refinement_threshold: float = 0.75
    max_iterations: int = 3
    enable_background: bool = True
    enable_hot_patterns: bool = True
    enable_scratchpad: bool = True


# =========================================================================
# Safety
# =========================================================================

@dataclass(frozen=True)
class SafetyConfig:
    """Safety guardrail settings."""
    enable_guardrails: bool = True
    log_all_decisions: bool = True


# =========================================================================
# Jenny UI
# =========================================================================

@dataclass(frozen=True)
class JennyConfig:
    """Jenny visual compilation panel settings."""
    enable: bool = False
    persist_path: str = "./jenny_specs"
    default_renderer: str = "html"
    max_panels_per_query: int = 6
    auto_lifecycle: bool = True
    cleanup_interval: float = 60.0
    enable_mrf: bool = True
    enable_learning: bool = True
    learning_persist_path: str = "./jenny_learning"


# =========================================================================
# Policy / Bandit
# =========================================================================

@dataclass(frozen=True)
class PolicyConfig:
    """Bandit strategy and model architecture settings."""
    n_transformer_layers: int = 2
    n_attention_heads: int = 4
    n_tools: int = 4
    n_adapters: int = 4
    bandit_strategy: BanditStrategy = BanditStrategy.EPSILON_GREEDY
    epsilon: float = 0.1
    blend_neural_weight: float = 0.7


# =========================================================================
# Metrics
# =========================================================================

@dataclass(frozen=True)
class MetricsConfig:
    """Prometheus metrics export settings."""
    enable: bool = True
    port: int = 8001


# =========================================================================
# Spring Activation
# =========================================================================

@dataclass(frozen=True)
class SpringConfig:
    """Spring-based activation propagation settings."""
    enable: bool = False
    stiffness: float = 0.15
    damping: float = 0.85
    decay: float = 0.98
    iterations: int = 200
    convergence_epsilon: float = 1e-4
    activation_threshold: float = 0.1
    seed_count: int = 3


# =========================================================================
# Semantic Calculus
# =========================================================================

@dataclass(frozen=True)
class SemanticCalculusConfig:
    """Semantic calculus framework settings."""
    enable: bool = False
    dimensions: int = 16
    cache_size: int = 10000
    dt: float = 1.0
    framework: str = "compassionate"
    trajectory: bool = True
    ethics: bool = True


# =========================================================================
# Linguistic Gate
# =========================================================================

@dataclass(frozen=True)
class LinguisticConfig:
    """Linguistic gate and compositional analysis settings."""
    enable: bool = False
    mode: str = "disabled"
    use_compositional_cache: bool = True
    parse_cache_size: int = 10000
    merge_cache_size: int = 50000
    weight: float = 0.3
    prefilter_similarity_threshold: float = 0.3
    prefilter_keep_ratio: float = 0.7


# =========================================================================
# Unified Physics
# =========================================================================

@dataclass(frozen=True)
class PhysicsConfig:
    """Unified physics engine settings."""
    enable: bool = False
    enable_routing: bool = True
    enable_packing: bool = True
    enable_thermodynamics: bool = True
    enable_wave_mechanics: bool = True
    mode: str = "adaptive"
    track_provenance: bool = True


# =========================================================================
# Conscience Architecture
# =========================================================================

@dataclass(frozen=True)
class ConscienceConfig:
    """Conscience ethical evaluation settings."""
    enable: bool = True
    preset: str = "standard"
    fail_open: bool = True
    auto_learn: bool = True
    learning_interval: float = 60.0
    persist_path: str | None = None


# =========================================================================
# Beta Wave Context Packing
# =========================================================================

@dataclass(frozen=True)
class BetaWavePackingConfig:
    """Beta wave context packing settings."""
    enable: bool = False
    token_budget: int = 4000
    query_reserve: int = 400
    response_reserve: int = 1000
    activation_threshold: float = 0.3
    compression_threshold: float = 0.7


# =========================================================================
# Backend / Infrastructure
# =========================================================================

@dataclass(frozen=True)
class BackendConfig:
    """LLM, memory backend, and infrastructure connection settings."""
    # LLM
    llm_provider: str | None = None
    llm_model: str | None = None

    # Memory
    memory_backend: str = "hybrid"
    memory_path: str = "data"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "hololoom_memories"
    qdrant_use_https: bool = False

    # Hyperspace
    hyperspace_depth: int = 3
    hyperspace_thresholds: tuple[float, ...] = (0.6, 0.75, 0.85)
    hyperspace_breadth: int = 10


# =========================================================================
# Semantic Bandit
# =========================================================================

@dataclass(frozen=True)
class SemanticBanditConfig:
    """Thompson Sampling semantic bandit settings."""
    enable: bool = True
    weight: float = 0.3
    topic_shift_threshold: float = 2.0
    success_threshold: float = 0.75


# =========================================================================
# Compat mapping: flat config field name → (sub_config_attr, field_name)
# Used by Config.__getattr__ bridge in B-3 (PR 7).
# =========================================================================

COMPAT_MAP = {
    # EmbeddingConfig
    'enable_zero_copy_embeddings': ('embedding', 'enable_zero_copy'),
    'zero_copy_cache_path': ('embedding', 'zero_copy_cache_path'),
    'zero_copy_cache_size': ('embedding', 'zero_copy_cache_size'),
    'spectral_k_eigen': ('embedding', 'spectral_k_eigen'),
    'svd_components': ('embedding', 'svd_components'),

    # RoutingConfig
    'enable_smart_routing': ('routing', 'enable'),
    'enable_semantic_tier': ('routing', 'enable_semantic_tier'),
    'enable_adaptive_learning': ('routing', 'enable_adaptive_learning'),
    'enable_classification_telemetry': ('routing', 'enable_classification_telemetry'),
    'classification_telemetry_path': ('routing', 'classification_telemetry_path'),

    # RetrievalConfig
    'retrieval_k': ('retrieval', 'k'),
    'bm25_weight': ('retrieval', 'bm25_weight'),
    'retrieval_timeout': ('retrieval', 'timeout'),

    # RecursiveLearningConfig
    'enable_recursive_learning': ('recursive_learning', 'enable'),
    'recursive_learning_update_interval': ('recursive_learning', 'update_interval'),
    'recursive_learning_refinement_threshold': ('recursive_learning', 'refinement_threshold'),
    'recursive_learning_max_iterations': ('recursive_learning', 'max_iterations'),
    'recursive_learning_enable_background': ('recursive_learning', 'enable_background'),
    'recursive_learning_enable_hot_patterns': ('recursive_learning', 'enable_hot_patterns'),
    'recursive_learning_enable_scratchpad': ('recursive_learning', 'enable_scratchpad'),

    # SafetyConfig
    'enable_safety_guardrails': ('safety', 'enable_guardrails'),
    'safety_log_all_decisions': ('safety', 'log_all_decisions'),

    # JennyConfig
    'enable_jenny': ('jenny', 'enable'),
    'jenny_persist_path': ('jenny', 'persist_path'),
    'jenny_default_renderer': ('jenny', 'default_renderer'),
    'jenny_max_panels_per_query': ('jenny', 'max_panels_per_query'),
    'jenny_auto_lifecycle': ('jenny', 'auto_lifecycle'),
    'jenny_cleanup_interval': ('jenny', 'cleanup_interval'),
    'jenny_enable_mrf': ('jenny', 'enable_mrf'),
    'jenny_enable_learning': ('jenny', 'enable_learning'),
    'jenny_learning_persist_path': ('jenny', 'learning_persist_path'),

    # PolicyConfig
    'n_transformer_layers': ('policy', 'n_transformer_layers'),
    'n_attention_heads': ('policy', 'n_attention_heads'),
    'n_tools': ('policy', 'n_tools'),
    'n_adapters': ('policy', 'n_adapters'),
    'bandit_strategy': ('policy', 'bandit_strategy'),
    'epsilon': ('policy', 'epsilon'),
    'blend_neural_weight': ('policy', 'blend_neural_weight'),

    # MetricsConfig
    'enable_prometheus_metrics': ('metrics', 'enable'),
    'prometheus_metrics_port': ('metrics', 'port'),

    # SpringConfig
    'use_spring_activation': ('spring', 'enable'),
    'enable_spring_activation': ('spring', 'enable'),
    'spring_stiffness': ('spring', 'stiffness'),
    'spring_damping': ('spring', 'damping'),
    'spring_decay': ('spring', 'decay'),
    'spring_iterations': ('spring', 'iterations'),
    'spring_convergence_epsilon': ('spring', 'convergence_epsilon'),
    'spring_activation_threshold': ('spring', 'activation_threshold'),
    'spring_seed_count': ('spring', 'seed_count'),

    # SemanticCalculusConfig
    'enable_semantic_calculus': ('semantic_calculus', 'enable'),
    'semantic_dimensions': ('semantic_calculus', 'dimensions'),
    'semantic_cache_size': ('semantic_calculus', 'cache_size'),
    'semantic_dt': ('semantic_calculus', 'dt'),
    'semantic_framework': ('semantic_calculus', 'framework'),
    'semantic_trajectory': ('semantic_calculus', 'trajectory'),
    'semantic_ethics': ('semantic_calculus', 'ethics'),

    # LinguisticConfig
    'enable_linguistic_gate': ('linguistic', 'enable'),
    'linguistic_mode': ('linguistic', 'mode'),
    'use_compositional_cache': ('linguistic', 'use_compositional_cache'),
    'parse_cache_size': ('linguistic', 'parse_cache_size'),
    'merge_cache_size': ('linguistic', 'merge_cache_size'),
    'linguistic_weight': ('linguistic', 'weight'),
    'prefilter_similarity_threshold': ('linguistic', 'prefilter_similarity_threshold'),
    'prefilter_keep_ratio': ('linguistic', 'prefilter_keep_ratio'),

    # PhysicsConfig
    'enable_unified_physics': ('physics', 'enable'),
    'physics_enable_routing': ('physics', 'enable_routing'),
    'physics_enable_packing': ('physics', 'enable_packing'),
    'physics_enable_thermodynamics': ('physics', 'enable_thermodynamics'),
    'physics_enable_wave_mechanics': ('physics', 'enable_wave_mechanics'),
    'physics_mode': ('physics', 'mode'),
    'physics_track_provenance': ('physics', 'track_provenance'),

    # ConscienceConfig
    'enable_conscience': ('conscience', 'enable'),
    'conscience_preset': ('conscience', 'preset'),
    'conscience_fail_open': ('conscience', 'fail_open'),
    'conscience_auto_learn': ('conscience', 'auto_learn'),
    'conscience_learning_interval': ('conscience', 'learning_interval'),
    'conscience_persist_path': ('conscience', 'persist_path'),

    # BetaWavePackingConfig
    'enable_beta_wave_packing': ('beta_wave_packing', 'enable'),
    'packing_token_budget': ('beta_wave_packing', 'token_budget'),
    'packing_query_reserve': ('beta_wave_packing', 'query_reserve'),
    'packing_response_reserve': ('beta_wave_packing', 'response_reserve'),
    'packing_activation_threshold': ('beta_wave_packing', 'activation_threshold'),
    'packing_compression_threshold': ('beta_wave_packing', 'compression_threshold'),

    # BackendConfig
    'llm_provider': ('backend', 'llm_provider'),
    'llm_model': ('backend', 'llm_model'),
    'memory_path': ('backend', 'memory_path'),
    'neo4j_uri': ('backend', 'neo4j_uri'),
    'neo4j_username': ('backend', 'neo4j_username'),
    'neo4j_password': ('backend', 'neo4j_password'),
    'neo4j_database': ('backend', 'neo4j_database'),
    'qdrant_host': ('backend', 'qdrant_host'),
    'qdrant_port': ('backend', 'qdrant_port'),
    'qdrant_collection': ('backend', 'qdrant_collection'),
    'qdrant_use_https': ('backend', 'qdrant_use_https'),
    'hyperspace_depth': ('backend', 'hyperspace_depth'),
    'hyperspace_breadth': ('backend', 'hyperspace_breadth'),

    # SemanticBanditConfig
    'enable_semantic_bandit': ('semantic_bandit', 'enable'),
    'semantic_bandit_weight': ('semantic_bandit', 'weight'),
    'semantic_topic_shift_threshold': ('semantic_bandit', 'topic_shift_threshold'),
    'semantic_bandit_success_threshold': ('semantic_bandit', 'success_threshold'),
}

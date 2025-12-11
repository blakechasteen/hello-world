"""
HoloLoom Configuration v2 - Zero-Config Architecture
====================================================

Philosophy: "Convention over Configuration" - Good software needs very little config.

3-Tier Configuration System:
- Tier 0: Config() - Just works with sensible defaults
- Tier 1: Config.fast() / Config.research() - Named presets
- Tier 2: Config(retrieval_k=10) - Fine-tuning for power users

Date: December 2025
Author: Claude Opus 4.5
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import os
import subprocess


# =============================================================================
# ENUMS (Tier 0 - The ONE choice that matters)
# =============================================================================

class ExecutionMode(Enum):
    """
    Execution modes - the ONE choice that matters.

    - BARE: Fastest, minimal features (~50ms)
    - FAST: Balanced speed/quality (~150ms) - DEFAULT
    - FUSED: Highest quality (~300ms)
    - RESEARCH: Experimental features, no time limits
    """
    BARE = "bare"
    FAST = "fast"
    FUSED = "fused"
    RESEARCH = "research"


class Environment(Enum):
    """
    Deployment environment - auto-detected from HOLOLOOM_ENV.

    Controls safety, logging, and performance settings.
    """
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class MemoryBackend(Enum):
    """
    Memory backend - auto-fallback from HYBRID to INMEMORY if Docker unavailable.

    - INMEMORY: NetworkX in-memory, no deps, <10ms
    - HYBRID: Neo4j+Qdrant, auto-fallback to INMEMORY, ~50ms (DEFAULT)
    - HYPERSPACE: Gated multipass, optional, ~150ms
    """
    INMEMORY = "inmemory"
    HYBRID = "hybrid"
    HYPERSPACE = "hyperspace"


# Import BanditStrategy from shared types (no circular dependency)
try:
    from HoloLoom.protocols.types import BanditStrategy
except ImportError:
    # Fallback if protocols not available
    class BanditStrategy(Enum):
        EPSILON_GREEDY = "epsilon_greedy"
        BAYESIAN_BLEND = "bayesian_blend"
        PURE_THOMPSON = "pure_thompson"


# =============================================================================
# AUTO-DETECTION HELPERS
# =============================================================================

def _detect_environment() -> Environment:
    """Auto-detect environment from HOLOLOOM_ENV."""
    env = os.getenv("HOLOLOOM_ENV", "development").lower()
    try:
        return Environment(env)
    except ValueError:
        return Environment.DEVELOPMENT


def _check_ollama_running() -> bool:
    """Check if Ollama is running locally."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _detect_llm() -> tuple[Optional[str], Optional[str]]:
    """Auto-detect LLM from available API keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-3-5-sonnet-20241022"
    if os.getenv("OPENAI_API_KEY"):
        return "openai", "gpt-4"
    if os.getenv("OLLAMA_HOST") or _check_ollama_running():
        return "ollama", "llama3.2:3b"
    return None, None


def _detect_neo4j() -> tuple[str, str, str]:
    """Auto-detect Neo4j settings from environment."""
    return (
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "hololoom123")
    )


def _detect_qdrant() -> tuple[str, int]:
    """Auto-detect Qdrant settings from environment."""
    return (
        os.getenv("QDRANT_HOST", "localhost"),
        int(os.getenv("QDRANT_PORT", "6333"))
    )


# =============================================================================
# MODE DEFAULTS (Internal - derived from mode)
# =============================================================================

_MODE_DEFAULTS: Dict[ExecutionMode, Dict[str, Any]] = {
    ExecutionMode.BARE: {
        # Embeddings
        "scales": [768],
        "fusion_weights": {768: 1.0},
        "enable_zero_copy_embeddings": False,

        # Neural network
        "n_transformer_layers": 1,
        "n_attention_heads": 2,
        "fast_mode": True,

        # Features (minimal)
        "enable_linguistic_gate": False,
        "enable_semantic_calculus": False,
        "enable_smart_routing": True,

        # Retrieval
        "bm25_weight": 0.15,
        "spectral_k_eigen": 4,
        "svd_components": 2,
    },

    ExecutionMode.FAST: {
        # Embeddings
        "scales": [768],
        "fusion_weights": {768: 1.0},
        "enable_zero_copy_embeddings": True,

        # Neural network
        "n_transformer_layers": 2,
        "n_attention_heads": 4,
        "fast_mode": True,

        # Features (balanced)
        "enable_linguistic_gate": True,
        "linguistic_mode": "both",
        "use_compositional_cache": True,
        "enable_semantic_calculus": False,
        "enable_smart_routing": True,

        # Retrieval
        "bm25_weight": 0.15,
        "spectral_k_eigen": 4,
        "svd_components": 2,
    },

    ExecutionMode.FUSED: {
        # Embeddings
        "scales": [768],
        "fusion_weights": {768: 1.0},
        "enable_zero_copy_embeddings": True,

        # Neural network
        "n_transformer_layers": 2,
        "n_attention_heads": 4,
        "fast_mode": False,

        # Features (full)
        "enable_linguistic_gate": True,
        "linguistic_mode": "both",
        "use_compositional_cache": True,
        "enable_semantic_calculus": True,
        "semantic_dimensions": 16,
        "semantic_ethics": True,
        "enable_smart_routing": True,

        # Retrieval
        "bm25_weight": 0.15,
        "spectral_k_eigen": 4,
        "svd_components": 2,
    },

    ExecutionMode.RESEARCH: {
        # Embeddings
        "scales": [768],
        "fusion_weights": {768: 1.0},
        "enable_zero_copy_embeddings": True,

        # Neural network
        "n_transformer_layers": 2,
        "n_attention_heads": 4,
        "fast_mode": False,

        # Features (all enabled)
        "enable_linguistic_gate": True,
        "linguistic_mode": "both",
        "use_compositional_cache": True,
        "enable_semantic_calculus": True,
        "semantic_dimensions": 16,
        "semantic_ethics": True,
        "enable_smart_routing": True,

        # Research-specific features
        "enable_recursive_learning": True,
        "enable_jenny": True,
        "use_spring_activation": True,
        "enable_conscience": True,
        "enable_dreaming": True,

        # Extended timeouts
        "pipeline_timeout": 30.0,
        "retrieval_timeout": 5.0,
    },
}


# =============================================================================
# EXPANSION BUNDLE PROTOCOL
# =============================================================================

class ExpansionBundle:
    """
    Protocol for expansion bundles (physics, bayesian, geometry, etc.).

    Expansion bundles contain optional research features that can be
    loaded into a Config after construction.
    """

    def get_settings(self) -> Dict[str, Any]:
        """Return dictionary of settings to merge into config."""
        raise NotImplementedError


# =============================================================================
# CONFIG DATACLASS
# =============================================================================

@dataclass
class Config:
    """
    HoloLoom Configuration - Zero-config by default.

    Usage:
        # Tier 0: Just works
        config = Config()

        # Tier 1: Named presets
        config = Config.fast()
        config = Config.research()

        # Tier 2: Fine-tuning
        config = Config(retrieval_k=10, pipeline_timeout=10.0)

        # With expansion bundles
        from HoloLoom.expansions.physics import PhysicsConfig
        config = Config.research()
        config.load_expansion(PhysicsConfig())
    """

    # =========================================================================
    # TIER 0: PRIMARY SETTING (the ONE choice that matters)
    # =========================================================================
    mode: ExecutionMode = ExecutionMode.FAST

    # =========================================================================
    # TIER 1: COMMONLY ADJUSTED (5-10 fields)
    # =========================================================================

    # Retrieval
    retrieval_k: int = 6  # Number of memories to retrieve

    # Timeouts
    pipeline_timeout: float = 5.0  # Max pipeline time (seconds)
    retrieval_timeout: float = 0.2  # Max retrieval time (seconds)

    # Persistence
    memory_path: str = "data"  # Root directory for memory storage
    memory_backend: MemoryBackend = MemoryBackend.HYBRID

    # LLM (optional - auto-detected if None)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    # Policy
    n_tools: int = 4  # Number of available tools
    n_adapters: int = 4  # Number of domain adapters
    bandit_strategy: BanditStrategy = BanditStrategy.EPSILON_GREEDY
    epsilon: float = 0.1  # Exploration rate (10%)

    # Safety (always enabled by default)
    enable_safety_guardrails: bool = True
    enable_conscience: bool = True

    # =========================================================================
    # TIER 2: AUTO-CONFIGURED (from environment)
    # =========================================================================
    environment: Environment = field(default_factory=_detect_environment)

    # Internal - derived from mode (not user-facing)
    _mode_settings: Dict[str, Any] = field(default_factory=dict, repr=False)
    _expansions: List[ExpansionBundle] = field(default_factory=list, repr=False)
    _expansion_settings: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Apply mode-specific defaults and auto-detection."""
        # Convert string mode to enum if needed
        if isinstance(self.mode, str):
            self.mode = ExecutionMode(self.mode.lower())

        # Auto-detect LLM if not specified
        if self.llm_provider is None:
            self.llm_provider, self.llm_model = _detect_llm()

        # Apply mode-specific settings
        self._mode_settings = _MODE_DEFAULTS.get(self.mode, _MODE_DEFAULTS[ExecutionMode.FAST]).copy()

        # Apply research-specific timeout adjustments
        if self.mode == ExecutionMode.RESEARCH:
            if self.pipeline_timeout == 5.0:  # Default unchanged
                self.pipeline_timeout = 30.0

    # =========================================================================
    # EXPANSION BUNDLE SUPPORT
    # =========================================================================

    def load_expansion(self, expansion: ExpansionBundle) -> "Config":
        """
        Load an expansion bundle's settings into this config.

        Args:
            expansion: ExpansionBundle instance (e.g., PhysicsConfig())

        Returns:
            Self for method chaining
        """
        self._expansions.append(expansion)
        settings = expansion.get_settings()
        self._expansion_settings.update(settings)
        return self

    # =========================================================================
    # COMPUTED PROPERTIES (derived from mode + expansions)
    # =========================================================================

    def _get(self, key: str, default: Any = None) -> Any:
        """Get setting from expansion > mode > default."""
        if key in self._expansion_settings:
            return self._expansion_settings[key]
        if key in self._mode_settings:
            return self._mode_settings[key]
        return default

    # --- Embedding Settings ---

    @property
    def scales(self) -> List[int]:
        """Matryoshka embedding scales."""
        return self._get("scales", [768])

    @property
    def fusion_weights(self) -> Dict[int, float]:
        """Multi-scale fusion weights."""
        return self._get("fusion_weights", {768: 1.0})

    @property
    def enable_zero_copy_embeddings(self) -> bool:
        """Enable zero-copy memory-mapped embeddings (30-50x faster)."""
        return self._get("enable_zero_copy_embeddings", False)

    @property
    def zero_copy_cache_path(self) -> str:
        """Zero-copy embeddings cache path."""
        return self._get("zero_copy_cache_path", ".cache/embeddings.mmap")

    @property
    def zero_copy_cache_size(self) -> int:
        """Maximum embeddings to cache."""
        return self._get("zero_copy_cache_size", 10000)

    # --- Neural Network Settings ---

    @property
    def n_transformer_layers(self) -> int:
        """Number of transformer layers."""
        return self._get("n_transformer_layers", 2)

    @property
    def n_attention_heads(self) -> int:
        """Number of attention heads."""
        return self._get("n_attention_heads", 4)

    @property
    def fast_mode(self) -> bool:
        """Fast mode flag for retrieval."""
        return self._get("fast_mode", True)

    @property
    def blend_neural_weight(self) -> float:
        """Neural weight in Bayesian blend (30% bandit)."""
        return self._get("blend_neural_weight", 0.7)

    # --- Feature Settings ---

    @property
    def enable_linguistic_gate(self) -> bool:
        """Enable Phase 5 linguistic matryoshka gate."""
        return self._get("enable_linguistic_gate", False)

    @property
    def linguistic_mode(self) -> str:
        """Linguistic filter mode: disabled, prefilter, embedding, both."""
        return self._get("linguistic_mode", "disabled")

    @property
    def use_compositional_cache(self) -> bool:
        """Enable 3-tier compositional cache."""
        return self._get("use_compositional_cache", True)

    @property
    def parse_cache_size(self) -> int:
        """X-bar structure cache size."""
        return self._get("parse_cache_size", 10000)

    @property
    def merge_cache_size(self) -> int:
        """Compositional embedding cache size."""
        return self._get("merge_cache_size", 50000)

    @property
    def linguistic_weight(self) -> float:
        """Weight for linguistic features (0-1)."""
        return self._get("linguistic_weight", 0.3)

    @property
    def prefilter_similarity_threshold(self) -> float:
        """Min syntactic similarity for pre-filter."""
        return self._get("prefilter_similarity_threshold", 0.3)

    @property
    def prefilter_keep_ratio(self) -> float:
        """Keep top X% of candidates after linguistic filter."""
        return self._get("prefilter_keep_ratio", 0.7)

    # --- Semantic Calculus ---

    @property
    def enable_semantic_calculus(self) -> bool:
        """Enable semantic flow analysis."""
        return self._get("enable_semantic_calculus", False)

    @property
    def semantic_dimensions(self) -> int:
        """Number of semantic dimension pairs."""
        return self._get("semantic_dimensions", 16)

    @property
    def semantic_cache_size(self) -> int:
        """Embedding cache size for semantic calculus."""
        return self._get("semantic_cache_size", 10000)

    @property
    def semantic_dt(self) -> float:
        """Time step for calculus."""
        return self._get("semantic_dt", 1.0)

    @property
    def semantic_framework(self) -> str:
        """Ethical framework: compassionate, scientific, therapeutic."""
        return self._get("semantic_framework", "compassionate")

    @property
    def semantic_trajectory(self) -> bool:
        """Compute velocity/acceleration/curvature."""
        return self._get("semantic_trajectory", True)

    @property
    def semantic_ethics(self) -> bool:
        """Run ethical analysis."""
        return self._get("semantic_ethics", True)

    # --- Routing ---

    @property
    def enable_smart_routing(self) -> bool:
        """Enable smart query routing with fast paths."""
        return self._get("enable_smart_routing", True)

    @property
    def routing_classifier(self) -> str:
        """Routing classifier: baseline or moonshot."""
        return self._get("routing_classifier", "moonshot")

    @property
    def enable_semantic_tier(self) -> bool:
        """Tier 3 semantic embeddings (adds 20ms, 98% accuracy)."""
        return self._get("enable_semantic_tier", False)

    @property
    def enable_adaptive_learning(self) -> bool:
        """Learn from production misclassifications."""
        return self._get("enable_adaptive_learning", True)

    @property
    def enable_classification_telemetry(self) -> bool:
        """Log classifications for monitoring."""
        return self._get("enable_classification_telemetry", True)

    @property
    def classification_telemetry_path(self) -> str:
        """Telemetry log directory."""
        return self._get("classification_telemetry_path", "./classification_logs")

    # --- Retrieval ---

    @property
    def bm25_weight(self) -> float:
        """Weight of BM25 in fused retrieval."""
        return self._get("bm25_weight", 0.15)

    @property
    def spectral_k_eigen(self) -> int:
        """Number of Laplacian eigenvalues."""
        return self._get("spectral_k_eigen", 4)

    @property
    def svd_components(self) -> int:
        """Number of SVD topic components."""
        return self._get("svd_components", 2)

    # --- Spring Activation (via expansion or RESEARCH mode) ---

    @property
    def use_spring_activation(self) -> bool:
        """Enable physics-based spreading activation."""
        return self._get("use_spring_activation", False)

    @property
    def spring_stiffness(self) -> float:
        """Spring stiffness (k)."""
        return self._get("spring_stiffness", 0.15)

    @property
    def spring_damping(self) -> float:
        """Damping coefficient (c)."""
        return self._get("spring_damping", 0.85)

    @property
    def spring_decay(self) -> float:
        """Activation decay per step."""
        return self._get("spring_decay", 0.98)

    @property
    def spring_iterations(self) -> int:
        """Max propagation steps."""
        return self._get("spring_iterations", 200)

    @property
    def spring_convergence_epsilon(self) -> float:
        """Energy change threshold."""
        return self._get("spring_convergence_epsilon", 1e-4)

    @property
    def spring_activation_threshold(self) -> float:
        """Min activation to retrieve."""
        return self._get("spring_activation_threshold", 0.1)

    @property
    def spring_seed_count(self) -> int:
        """Number of seed nodes from embedding."""
        return self._get("spring_seed_count", 3)

    # --- Recursive Learning ---

    @property
    def enable_recursive_learning(self) -> bool:
        """Enable integrated recursive learning."""
        return self._get("enable_recursive_learning", False)

    @property
    def recursive_learning_update_interval(self) -> float:
        """Background learning update interval (seconds)."""
        return self._get("recursive_learning_update_interval", 60.0)

    @property
    def recursive_learning_refinement_threshold(self) -> float:
        """Confidence threshold for refinement trigger."""
        return self._get("recursive_learning_refinement_threshold", 0.75)

    @property
    def recursive_learning_max_iterations(self) -> int:
        """Maximum refinement iterations."""
        return self._get("recursive_learning_max_iterations", 3)

    @property
    def recursive_learning_enable_background(self) -> bool:
        """Enable background learning thread."""
        return self._get("recursive_learning_enable_background", True)

    @property
    def recursive_learning_enable_hot_patterns(self) -> bool:
        """Enable hot pattern tracking."""
        return self._get("recursive_learning_enable_hot_patterns", True)

    @property
    def recursive_learning_enable_scratchpad(self) -> bool:
        """Enable provenance tracking."""
        return self._get("recursive_learning_enable_scratchpad", True)

    # --- Jenny UI ---

    @property
    def enable_jenny(self) -> bool:
        """Enable Jenny UI panel generation."""
        return self._get("enable_jenny", False)

    @property
    def jenny_persist_path(self) -> str:
        """SpecLedger persistence directory."""
        return self._get("jenny_persist_path", "./jenny_specs")

    @property
    def jenny_default_renderer(self) -> str:
        """Default renderer: html, terminal, json."""
        return self._get("jenny_default_renderer", "html")

    @property
    def jenny_max_panels_per_query(self) -> int:
        """Maximum panels per weave."""
        return self._get("jenny_max_panels_per_query", 6)

    @property
    def jenny_enable_mrf(self) -> bool:
        """Enable MRF-enhanced panel compilation."""
        return self._get("jenny_enable_mrf", True)

    @property
    def jenny_enable_learning(self) -> bool:
        """Enable Thompson Sampling learning."""
        return self._get("jenny_enable_learning", True)

    # --- Conscience ---

    @property
    def conscience_preset(self) -> str:
        """Lens preset: standard, paranoid, research."""
        return self._get("conscience_preset", "standard")

    @property
    def conscience_fail_open(self) -> bool:
        """Graceful degradation (fail-open for availability)."""
        return self._get("conscience_fail_open", True)

    @property
    def conscience_auto_learn(self) -> bool:
        """Auto-learn from execution outcomes."""
        return self._get("conscience_auto_learn", True)

    @property
    def conscience_learning_interval(self) -> float:
        """Background learning interval (seconds)."""
        return self._get("conscience_learning_interval", 60.0)

    @property
    def conscience_persist_path(self) -> Optional[str]:
        """Path for conscience wisdom persistence."""
        return self._get("conscience_persist_path", None)

    # --- Dreaming ---

    @property
    def enable_dreaming(self) -> bool:
        """Enable background dream consolidation."""
        return self._get("enable_dreaming", True)

    @property
    def dream_consolidation_interval(self) -> float:
        """Seconds between dream cycles."""
        return self._get("dream_consolidation_interval", 3600.0)

    @property
    def dream_math_bleed_rate(self) -> float:
        """Math insight sharing rate."""
        return self._get("dream_math_bleed_rate", 0.3)

    @property
    def dream_pattern_bleed_rate(self) -> float:
        """Pattern insight sharing rate."""
        return self._get("dream_pattern_bleed_rate", 0.2)

    # --- Memory Management ---

    @property
    def working_memory_size(self) -> int:
        """Cache size for recent queries."""
        return self._get("working_memory_size", 100)

    @property
    def episodic_buffer_size(self) -> int:
        """Size of recent interaction buffer."""
        return self._get("episodic_buffer_size", 100)

    # --- Database (auto-detected from environment) ---

    @property
    def neo4j_uri(self) -> str:
        """Neo4j connection URI."""
        uri, _, _ = _detect_neo4j()
        return self._get("neo4j_uri", uri)

    @property
    def neo4j_username(self) -> str:
        """Neo4j username."""
        _, username, _ = _detect_neo4j()
        return self._get("neo4j_username", username)

    @property
    def neo4j_password(self) -> str:
        """Neo4j password."""
        _, _, password = _detect_neo4j()
        return self._get("neo4j_password", password)

    @property
    def neo4j_database(self) -> str:
        """Neo4j database name."""
        return self._get("neo4j_database", "neo4j")

    @property
    def qdrant_host(self) -> str:
        """Qdrant host."""
        host, _ = _detect_qdrant()
        return self._get("qdrant_host", host)

    @property
    def qdrant_port(self) -> int:
        """Qdrant port."""
        _, port = _detect_qdrant()
        return self._get("qdrant_port", port)

    @property
    def qdrant_collection(self) -> str:
        """Qdrant collection name."""
        return self._get("qdrant_collection", "hololoom_memories")

    @property
    def qdrant_use_https(self) -> bool:
        """Use HTTPS for Qdrant."""
        return self._get("qdrant_use_https", False)

    # --- Shuttle ---

    @property
    def enable_shuttle(self) -> bool:
        """Enable Shuttle for intelligent thread selection."""
        return self._get("enable_shuttle", True)

    @property
    def shuttle_mode(self) -> str:
        """Shuttle mode: auto, full, lite, minimal."""
        return self._get("shuttle_mode", "auto")

    # --- Hyperspace ---

    @property
    def hyperspace_depth(self) -> int:
        """Max recursion depth for Hyperspace."""
        return self._get("hyperspace_depth", 3)

    @property
    def hyperspace_thresholds(self) -> List[float]:
        """Thresholds per recursion level."""
        return self._get("hyperspace_thresholds", [0.6, 0.75, 0.85])

    @property
    def hyperspace_breadth(self) -> int:
        """Links per level."""
        return self._get("hyperspace_breadth", 10)

    # --- Prometheus ---

    @property
    def enable_prometheus_metrics(self) -> bool:
        """Enable Prometheus metrics collection."""
        return self._get("enable_prometheus_metrics", True)

    @property
    def prometheus_metrics_port(self) -> int:
        """Port for metrics HTTP endpoint."""
        return self._get("prometheus_metrics_port", 8001)

    # --- Beta Wave Context Packing ---

    @property
    def enable_beta_wave_packing(self) -> bool:
        """Enable physics-based context optimization."""
        return self._get("enable_beta_wave_packing", False)

    @property
    def packing_token_budget(self) -> int:
        """Total token budget for packed context."""
        return self._get("packing_token_budget", 4000)

    @property
    def packing_query_reserve(self) -> int:
        """Tokens reserved for query."""
        return self._get("packing_query_reserve", 400)

    @property
    def packing_response_reserve(self) -> int:
        """Tokens reserved for LLM response."""
        return self._get("packing_response_reserve", 1000)

    @property
    def packing_activation_threshold(self) -> float:
        """Min activation to include."""
        return self._get("packing_activation_threshold", 0.3)

    @property
    def packing_compression_threshold(self) -> float:
        """Activation threshold for compression vs full content."""
        return self._get("packing_compression_threshold", 0.7)

    # --- WeaveHouse ---

    @property
    def use_weave_house(self) -> bool:
        """Enable WeaveHouse multi-perspective system."""
        return self._get("use_weave_house", False)

    @property
    def weave_house_exploration_depth(self) -> int:
        """How deep to explore disagreement zones."""
        return self._get("weave_house_exploration_depth", 2)

    @property
    def weave_house_tension_threshold(self) -> float:
        """Minimum tension to trigger exploration."""
        return self._get("weave_house_tension_threshold", 0.3)

    # =========================================================================
    # ENVIRONMENT-AWARE PROPERTIES (Smart Defaults)
    # =========================================================================

    @property
    def safety_testing_mode(self) -> bool:
        """Whether to bypass approval requirements (dev mode only)."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def safety_auto_approve_categories(self) -> set:
        """Action categories to auto-approve."""
        if self.environment == Environment.DEVELOPMENT:
            return {"query", "retrieval", "analysis", "storage", "modification", "execution", "external"}
        elif self.environment == Environment.STAGING:
            return {"query", "retrieval", "analysis"}
        else:  # PRODUCTION
            return set()

    @property
    def logging_level(self) -> str:
        """Logging verbosity based on environment."""
        if self.environment == Environment.DEVELOPMENT:
            return "DEBUG"
        elif self.environment == Environment.STAGING:
            return "INFO"
        else:  # PRODUCTION
            return "WARNING"

    # =========================================================================
    # PRESETS (Tier 1)
    # =========================================================================

    @classmethod
    def bare(cls) -> "Config":
        """Fastest execution, minimal features (~50ms)."""
        return cls(mode=ExecutionMode.BARE)

    @classmethod
    def fast(cls) -> "Config":
        """Balanced speed and quality (~150ms) - DEFAULT."""
        return cls(mode=ExecutionMode.FAST)

    @classmethod
    def fused(cls) -> "Config":
        """Highest quality, all features enabled (~300ms)."""
        return cls(mode=ExecutionMode.FUSED)

    @classmethod
    def research(cls) -> "Config":
        """Experimental features for research."""
        return cls(mode=ExecutionMode.RESEARCH)

    @classmethod
    def multi_perspective(cls) -> "Config":
        """Multi-perspective WeaveHouse system."""
        cfg = cls(mode=ExecutionMode.FUSED)
        cfg._mode_settings["use_weave_house"] = True
        cfg._mode_settings["enable_dreaming"] = True
        cfg._mode_settings["weave_house_exploration_depth"] = 2
        cfg._mode_settings["weave_house_tension_threshold"] = 0.3
        return cfg

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "mode": self.mode.value,
            "retrieval_k": self.retrieval_k,
            "pipeline_timeout": self.pipeline_timeout,
            "retrieval_timeout": self.retrieval_timeout,
            "memory_path": self.memory_path,
            "memory_backend": self.memory_backend.value,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "n_tools": self.n_tools,
            "n_adapters": self.n_adapters,
            "bandit_strategy": self.bandit_strategy.value,
            "epsilon": self.epsilon,
            "enable_safety_guardrails": self.enable_safety_guardrails,
            "enable_conscience": self.enable_conscience,
            "environment": self.environment.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Deserialize config from dictionary."""
        # Convert string enums back
        if "mode" in data:
            data["mode"] = ExecutionMode(data["mode"])
        if "memory_backend" in data:
            data["memory_backend"] = MemoryBackend(data["memory_backend"])
        if "environment" in data:
            data["environment"] = Environment(data["environment"])
        if "bandit_strategy" in data:
            data["bandit_strategy"] = BanditStrategy(data["bandit_strategy"])
        return cls(**data)


# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

# Re-export for backward compatibility
__all__ = [
    "Config",
    "ExecutionMode",
    "Environment",
    "MemoryBackend",
    "BanditStrategy",
    "ExpansionBundle",
]


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=== HoloLoom Zero-Config Architecture ===\n")

    # Tier 0: Just works
    print("1. Tier 0 - Zero Config:")
    cfg = Config()
    print(f"   Mode: {cfg.mode.value}")
    print(f"   Environment: {cfg.environment.value}")
    print(f"   LLM: {cfg.llm_provider} / {cfg.llm_model}")

    # Tier 1: Presets
    print("\n2. Tier 1 - Presets:")
    for preset_name in ["bare", "fast", "fused", "research"]:
        preset = getattr(Config, preset_name)()
        print(f"   {preset_name.upper()}: linguistic_gate={preset.enable_linguistic_gate}, "
              f"semantic_calculus={preset.enable_semantic_calculus}")

    # Tier 2: Fine-tuning
    print("\n3. Tier 2 - Fine-tuning:")
    cfg = Config(retrieval_k=10, pipeline_timeout=10.0)
    print(f"   retrieval_k: {cfg.retrieval_k}")
    print(f"   pipeline_timeout: {cfg.pipeline_timeout}")

    print("\n✓ Zero-config architecture working!")

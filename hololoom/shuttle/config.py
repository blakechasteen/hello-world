"""
HoloLoom Shuttle - Configuration and Validation

Provides configuration management with automatic validation and
graceful degradation support.

Created: 2025-01-21
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import BackendUnavailableError, ConfigurationError

logger = logging.getLogger(__name__)


class ShuttleMode(Enum):
    """
    Shuttle operation modes with automatic degradation.

    FULL: Neo4j + Qdrant + MCTS (production)
    LITE: NetworkX + NumPy + simplified MCTS (development)
    MINIMAL: Pure Python + greedy BFS (always works)
    AUTO: Automatically detect best available mode
    """
    FULL = "full"
    LITE = "lite"
    MINIMAL = "minimal"
    AUTO = "auto"


@dataclass
class ShuttleConfig:
    """
    Configuration for Shuttle operation.

    Attributes:
        mode: Operation mode (FULL/LITE/MINIMAL/AUTO)
        mcts_simulations: Number of MCTS iterations (10-100)
        mcts_timeout_ms: MCTS timeout in milliseconds
        warp_top_k: Number of Warp results to retrieve
        max_graph_depth: Maximum graph traversal depth
        max_graph_nodes: Maximum nodes to retrieve
        exploration_constant: UCB1 exploration parameter (1.0-2.0)
        enable_entity_extraction: Whether to extract entities from Warp
        entity_extraction_method: Method to use (spacy/regex/payload)
        enable_graceful_degradation: Fall back on errors
        fallback_chain: Ordered list of modes to try
        validate_backends: Check backend connectivity on init
        timeout_budget_ms: Total timeout budget for intersect()
    """

    # Operation mode
    mode: ShuttleMode = ShuttleMode.AUTO

    # MCTS parameters
    mcts_simulations: int = 32
    mcts_timeout_ms: int = 5000
    exploration_constant: float = 1.4

    # Warp parameters
    warp_top_k: int = 10
    warp_timeout_ms: int = 3000

    # Yarn parameters
    max_graph_depth: int = 2
    max_graph_nodes: int = 40
    yarn_timeout_ms: int = 5000

    # Entity extraction
    enable_entity_extraction: bool = True
    entity_extraction_method: str = "payload"  # "spacy", "regex", "payload"
    entity_extraction_fallback: bool = True

    # Error handling
    enable_graceful_degradation: bool = True
    fallback_chain: list[ShuttleMode] = field(default_factory=lambda: [
        ShuttleMode.FULL,
        ShuttleMode.LITE,
        ShuttleMode.MINIMAL
    ])
    max_retries: int = 2
    retry_delay_ms: int = 100

    # Validation
    validate_backends: bool = True
    validate_trajectory_registry: bool = True

    # Performance
    timeout_budget_ms: int = 10000  # Total timeout for intersect()
    enable_caching: bool = True
    cache_size: int = 1000

    # Logging
    log_level: str = "INFO"
    log_performance_metrics: bool = True
    log_trajectory_selection: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """
        Validate configuration parameters.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate MCTS parameters
        if self.mcts_simulations < 1:
            raise ConfigurationError(
                "mcts_simulations must be >= 1",
                details={"value": self.mcts_simulations}
            )

        if self.mcts_simulations > 1000:
            logger.warning(
                f"mcts_simulations={self.mcts_simulations} is very high, "
                "may cause performance issues"
            )

        if not (1.0 <= self.exploration_constant <= 3.0):
            raise ConfigurationError(
                "exploration_constant should be in [1.0, 3.0]",
                details={"value": self.exploration_constant}
            )

        # Validate graph parameters
        if self.max_graph_depth < 1:
            raise ConfigurationError(
                "max_graph_depth must be >= 1",
                details={"value": self.max_graph_depth}
            )

        if self.max_graph_nodes < 1:
            raise ConfigurationError(
                "max_graph_nodes must be >= 1",
                details={"value": self.max_graph_nodes}
            )

        # Validate timeout budget
        total_component_timeouts = (
            self.warp_timeout_ms +
            self.yarn_timeout_ms +
            self.mcts_timeout_ms
        )

        if total_component_timeouts > self.timeout_budget_ms:
            logger.warning(
                f"Component timeouts ({total_component_timeouts}ms) exceed "
                f"total budget ({self.timeout_budget_ms}ms)"
            )

        # Validate entity extraction
        valid_methods = ["spacy", "regex", "payload"]
        if self.entity_extraction_method not in valid_methods:
            raise ConfigurationError(
                f"Invalid entity_extraction_method: {self.entity_extraction_method}",
                details={"valid_methods": valid_methods}
            )

        # Validate fallback chain
        if self.enable_graceful_degradation and not self.fallback_chain:
            raise ConfigurationError(
                "fallback_chain must not be empty when graceful degradation enabled"
            )

    @classmethod
    def for_mode(cls, mode: ShuttleMode, **overrides) -> 'ShuttleConfig':
        """
        Create config optimized for a specific mode.

        Args:
            mode: Target operation mode
            **overrides: Override default parameters

        Returns:
            ShuttleConfig instance
        """
        if mode == ShuttleMode.FULL:
            defaults = {
                "mcts_simulations": 32,
                "mcts_timeout_ms": 5000,
                "max_graph_depth": 2,
                "max_graph_nodes": 40,
                "entity_extraction_method": "spacy",
            }
        elif mode == ShuttleMode.LITE:
            defaults = {
                "mcts_simulations": 16,
                "mcts_timeout_ms": 2000,
                "max_graph_depth": 2,
                "max_graph_nodes": 30,
                "entity_extraction_method": "regex",
            }
        elif mode == ShuttleMode.MINIMAL:
            defaults = {
                "mcts_simulations": 0,  # Use greedy BFS instead
                "mcts_timeout_ms": 1000,
                "max_graph_depth": 1,
                "max_graph_nodes": 20,
                "entity_extraction_method": "payload",
            }
        else:  # AUTO
            defaults = {
                "mcts_simulations": 32,
                "mcts_timeout_ms": 5000,
            }

        # Merge defaults with overrides
        merged = {**defaults, **overrides, "mode": mode}
        return cls(**merged)


def validate_config(config: ShuttleConfig) -> None:
    """
    Validate Shuttle configuration.

    Checks:
    - Parameter ranges
    - Backend availability (if validate_backends=True)
    - Trajectory registry (if validate_trajectory_registry=True)

    Args:
        config: Configuration to validate

    Raises:
        ConfigurationError: If validation fails
    """
    # Config __post_init__ already validates parameters
    # This function adds runtime checks

    if config.validate_backends:
        _validate_backends(config)

    if config.validate_trajectory_registry:
        _validate_trajectory_registry()

    logger.info(f"Shuttle configuration validated: mode={config.mode.value}")


def _validate_backends(config: ShuttleConfig) -> None:
    """
    Check backend connectivity.

    Args:
        config: Shuttle configuration

    Raises:
        BackendUnavailableError: If required backend is down
    """
    errors = []

    if config.mode == ShuttleMode.FULL:
        # Check Neo4j
        try:
            _check_neo4j()
        except Exception as e:
            errors.append(("Neo4j", str(e)))

        # Check Qdrant
        try:
            _check_qdrant()
        except Exception as e:
            errors.append(("Qdrant", str(e)))

        if errors and not config.enable_graceful_degradation:
            backend_names = ", ".join(name for name, _ in errors)
            raise BackendUnavailableError(
                backend_names,
                "Required backends unavailable in FULL mode",
                fallback_mode=None
            )

    logger.debug("Backend validation passed")


def _check_neo4j() -> None:
    """
    Check Neo4j connectivity.

    Raises:
        Exception: If Neo4j is unavailable
    """
    try:
        from neo4j import GraphDatabase
        # Try to import - actual connection check happens in adapter
        logger.debug("Neo4j driver available")
    except ImportError:
        raise ImportError("neo4j package not installed")


def _check_qdrant() -> None:
    """
    Check Qdrant connectivity.

    Raises:
        Exception: If Qdrant is unavailable
    """
    try:
        from qdrant_client import QdrantClient
        # Try to import - actual connection check happens in adapter
        logger.debug("Qdrant client available")
    except ImportError:
        raise ImportError("qdrant-client package not installed")


def _validate_trajectory_registry() -> None:
    """
    Validate trajectory registry has required trajectories.

    Raises:
        ConfigurationError: If registry is invalid
    """
    try:
        from .trajectories import TRAJECTORY_BY_NAME

        if not TRAJECTORY_BY_NAME:
            raise ConfigurationError("Trajectory registry is empty")

        logger.debug(
            f"Trajectory registry validated: "
            f"{len(TRAJECTORY_BY_NAME)} trajectories"
        )

    except ImportError as e:
        raise ConfigurationError(
            f"Failed to import trajectory registry: {e}"
        )


# Preset configurations

def default_config() -> ShuttleConfig:
    """Default configuration (AUTO mode)."""
    return ShuttleConfig()


def production_config() -> ShuttleConfig:
    """Production configuration (FULL mode, optimized for quality)."""
    return ShuttleConfig.for_mode(
        ShuttleMode.FULL,
        mcts_simulations=50,
        warp_top_k=15,
        max_graph_depth=3,
        max_graph_nodes=50,
        enable_graceful_degradation=True,
        validate_backends=True,
    )


def development_config() -> ShuttleConfig:
    """Development configuration (LITE mode, optimized for speed)."""
    return ShuttleConfig.for_mode(
        ShuttleMode.LITE,
        mcts_simulations=10,
        warp_top_k=5,
        max_graph_depth=1,
        max_graph_nodes=20,
        enable_graceful_degradation=False,
        validate_backends=False,
    )


def minimal_config() -> ShuttleConfig:
    """Minimal configuration (MINIMAL mode, zero dependencies)."""
    return ShuttleConfig.for_mode(
        ShuttleMode.MINIMAL,
        mcts_simulations=0,
        warp_top_k=5,
        max_graph_depth=1,
        max_graph_nodes=10,
        entity_extraction_method="payload",
        enable_graceful_degradation=False,
        validate_backends=False,
    )

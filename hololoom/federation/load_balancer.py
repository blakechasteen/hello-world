#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Balancer for Distributed Inference
========================================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 19: Distributed Inference

Weighted least-connections load balancer for routing inference
requests across federation nodes.

Author: Claude Code (Phase 4 - December 2025)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from hololoom.federation.types import FederationNode, GuildTrustLevel

logger = logging.getLogger(__name__)


# ============================================================================
#  Constants
# ============================================================================

# Default weights for load balancer scoring
DEFAULT_LOAD_WEIGHT = 0.4      # Weight for current load
DEFAULT_TRUST_WEIGHT = 0.3    # Weight for trust score
DEFAULT_LATENCY_WEIGHT = 0.2  # Weight for historical latency
DEFAULT_SUCCESS_WEIGHT = 0.1  # Weight for success rate

# Threshold for considering a node overloaded
OVERLOAD_THRESHOLD = 0.9  # 90% load

# Maximum connections per node before rejecting
MAX_CONNECTIONS_PER_NODE = 100

# Decay factor for exponential moving average of latency
LATENCY_EMA_ALPHA = 0.3


# ============================================================================
#  Enums
# ============================================================================

class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = auto()           # Simple rotation
    LEAST_CONNECTIONS = auto()     # Prefer nodes with fewer active connections
    WEIGHTED_LEAST_CONN = auto()   # Weighted by load + trust + latency
    RANDOM = auto()                # Random selection
    TRUST_FIRST = auto()           # Prefer highest trust nodes


class NodeHealth(Enum):
    """Node health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ============================================================================
#  Data Classes
# ============================================================================

@dataclass
class NodeStats:
    """Statistics for a single node."""

    node_id: str
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    current_load: float = 0.0
    health: NodeHealth = NodeHealth.UNKNOWN
    last_health_check: float = 0.0
    models: Set[str] = field(default_factory=set)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)."""
        if self.total_requests == 0:
            return 1.0  # Assume good until proven otherwise
        return self.successful_requests / self.total_requests

    @property
    def is_available(self) -> bool:
        """Check if node is available for requests."""
        return (
            self.health in (NodeHealth.HEALTHY, NodeHealth.UNKNOWN) and
            self.active_connections < MAX_CONNECTIONS_PER_NODE and
            self.current_load < OVERLOAD_THRESHOLD
        )


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""

    strategy: LoadBalanceStrategy = LoadBalanceStrategy.WEIGHTED_LEAST_CONN
    load_weight: float = DEFAULT_LOAD_WEIGHT
    trust_weight: float = DEFAULT_TRUST_WEIGHT
    latency_weight: float = DEFAULT_LATENCY_WEIGHT
    success_weight: float = DEFAULT_SUCCESS_WEIGHT
    health_check_interval_seconds: float = 30.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5  # Failures before opening
    circuit_breaker_reset_seconds: float = 60.0


@dataclass
class SelectionResult:
    """Result of node selection."""

    node: Optional[FederationNode]
    node_id: Optional[str]
    score: float
    reason: str
    alternatives: List[str] = field(default_factory=list)


# ============================================================================
#  Circuit Breaker
# ============================================================================

@dataclass
class CircuitBreakerState:
    """State for a circuit breaker."""

    failures: int = 0
    last_failure: float = 0.0
    is_open: bool = False
    opened_at: float = 0.0


class CircuitBreaker:
    """Circuit breaker for individual nodes."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_seconds: float = 60.0,
    ):
        self._failure_threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._states: Dict[str, CircuitBreakerState] = {}

    def is_open(self, node_id: str) -> bool:
        """Check if circuit is open (blocking requests)."""
        state = self._states.get(node_id)
        if state is None:
            return False

        if not state.is_open:
            return False

        # Check if reset time has passed
        if time.time() - state.opened_at >= self._reset_seconds:
            # Half-open: allow one request through
            state.is_open = False
            return False

        return True

    def record_success(self, node_id: str) -> None:
        """Record successful request."""
        state = self._states.get(node_id)
        if state is not None:
            state.failures = 0
            state.is_open = False

    def record_failure(self, node_id: str) -> bool:
        """
        Record failed request.

        Returns:
            True if circuit is now open.
        """
        if node_id not in self._states:
            self._states[node_id] = CircuitBreakerState()

        state = self._states[node_id]
        state.failures += 1
        state.last_failure = time.time()

        if state.failures >= self._failure_threshold:
            state.is_open = True
            state.opened_at = time.time()
            logger.warning(
                f"Circuit breaker opened for node {node_id} "
                f"after {state.failures} failures"
            )
            return True

        return False

    def get_state(self, node_id: str) -> Optional[CircuitBreakerState]:
        """Get circuit breaker state for a node."""
        return self._states.get(node_id)


# ============================================================================
#  Load Balancer
# ============================================================================

class LoadBalancer:
    """
    Weighted least-connections load balancer.

    Routes inference requests to federation nodes based on:
    - Current load (active connections / capacity)
    - Trust score (guild trust level)
    - Historical latency (exponential moving average)
    - Success rate (successful / total requests)

    Example:
        >>> balancer = LoadBalancer()
        >>> balancer.register_node(node, models=["llama3", "mistral"])
        >>>
        >>> # Select best node for inference
        >>> result = balancer.select_node(
        ...     model="llama3",
        ...     required_trust=GuildTrustLevel.MEMBER
        ... )
        >>> if result.node:
        ...     print(f"Selected: {result.node_id} (score: {result.score:.2f})")
        ...     # Use node for inference
        ...     balancer.acquire_connection(result.node_id)
        ...     # ... perform inference ...
        ...     balancer.release_connection(result.node_id, latency_ms=150.0)
    """

    def __init__(
        self,
        config: Optional[LoadBalancerConfig] = None,
    ):
        """
        Initialize load balancer.

        Args:
            config: Load balancer configuration (uses defaults if None)
        """
        self._config = config or LoadBalancerConfig()
        self._nodes: Dict[str, FederationNode] = {}
        self._stats: Dict[str, NodeStats] = {}
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._round_robin_index = 0

        if self._config.enable_circuit_breaker:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=self._config.circuit_breaker_threshold,
                reset_seconds=self._config.circuit_breaker_reset_seconds,
            )

    def register_node(
        self,
        node: FederationNode,
        models: Optional[Set[str]] = None,
        initial_load: float = 0.0,
    ) -> None:
        """
        Register a federation node with the balancer.

        Args:
            node: Federation node to register
            models: Set of model names this node supports
            initial_load: Initial load value (0.0-1.0)
        """
        self._nodes[node.node_id] = node
        self._stats[node.node_id] = NodeStats(
            node_id=node.node_id,
            current_load=initial_load,
            models=models or set(),
            health=NodeHealth.HEALTHY,
        )
        logger.debug(f"Registered node {node.node_id} with models {models}")

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from the balancer."""
        self._nodes.pop(node_id, None)
        self._stats.pop(node_id, None)
        logger.debug(f"Unregistered node {node_id}")

    def update_node_load(self, node_id: str, load: float) -> None:
        """
        Update a node's current load.

        Args:
            node_id: Node identifier
            load: Current load (0.0-1.0)
        """
        stats = self._stats.get(node_id)
        if stats:
            stats.current_load = max(0.0, min(1.0, load))

    def update_node_health(
        self,
        node_id: str,
        health: NodeHealth,
        models: Optional[Set[str]] = None,
    ) -> None:
        """
        Update a node's health status.

        Args:
            node_id: Node identifier
            health: New health status
            models: Updated models (if provided)
        """
        stats = self._stats.get(node_id)
        if stats:
            stats.health = health
            stats.last_health_check = time.time()
            if models is not None:
                stats.models = models

    def acquire_connection(self, node_id: str) -> bool:
        """
        Acquire a connection slot for a node.

        Args:
            node_id: Node identifier

        Returns:
            True if connection acquired, False if at capacity
        """
        stats = self._stats.get(node_id)
        if not stats:
            return False

        if stats.active_connections >= MAX_CONNECTIONS_PER_NODE:
            return False

        stats.active_connections += 1
        stats.total_requests += 1
        return True

    def release_connection(
        self,
        node_id: str,
        success: bool = True,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Release a connection slot and record metrics.

        Args:
            node_id: Node identifier
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
        """
        stats = self._stats.get(node_id)
        if not stats:
            return

        stats.active_connections = max(0, stats.active_connections - 1)
        stats.last_latency_ms = latency_ms

        # Update exponential moving average of latency
        if stats.avg_latency_ms == 0.0:
            stats.avg_latency_ms = latency_ms
        else:
            stats.avg_latency_ms = (
                LATENCY_EMA_ALPHA * latency_ms +
                (1 - LATENCY_EMA_ALPHA) * stats.avg_latency_ms
            )

        if success:
            stats.successful_requests += 1
            if self._circuit_breaker:
                self._circuit_breaker.record_success(node_id)
        else:
            stats.failed_requests += 1
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(node_id)

    def select_node(
        self,
        model: Optional[str] = None,
        required_trust: Optional[GuildTrustLevel] = None,
        exclude_nodes: Optional[Set[str]] = None,
    ) -> SelectionResult:
        """
        Select the best node for a request.

        Args:
            model: Required model name (None = any model)
            required_trust: Minimum trust level required
            exclude_nodes: Nodes to exclude from selection

        Returns:
            SelectionResult with selected node and score
        """
        exclude = exclude_nodes or set()
        candidates = []

        # Filter candidates
        for node_id, node in self._nodes.items():
            if node_id in exclude:
                continue

            stats = self._stats.get(node_id)
            if not stats or not stats.is_available:
                continue

            # Check circuit breaker
            if self._circuit_breaker and self._circuit_breaker.is_open(node_id):
                continue

            # Check model capability
            if model and model not in stats.models:
                continue

            # Check trust level
            if required_trust:
                trust_map = {
                    GuildTrustLevel.STARTER: 0.3,
                    GuildTrustLevel.ESTABLISHED: 0.6,
                    GuildTrustLevel.VETERAN: 1.0,
                }
                required_score = trust_map.get(required_trust, 0.3)
                if node.trust_score < required_score:
                    continue

            candidates.append((node_id, node, stats))

        if not candidates:
            return SelectionResult(
                node=None,
                node_id=None,
                score=0.0,
                reason="No available nodes matching criteria",
            )

        # Select based on strategy
        if self._config.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            selected = self._select_round_robin(candidates)
        elif self._config.strategy == LoadBalanceStrategy.RANDOM:
            selected = self._select_random(candidates)
        elif self._config.strategy == LoadBalanceStrategy.TRUST_FIRST:
            selected = self._select_trust_first(candidates)
        elif self._config.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            selected = self._select_least_connections(candidates)
        else:  # WEIGHTED_LEAST_CONN (default)
            selected = self._select_weighted(candidates)

        node_id, node, stats, score = selected
        alternatives = [
            c[0] for c in candidates
            if c[0] != node_id
        ][:5]  # Top 5 alternatives

        return SelectionResult(
            node=node,
            node_id=node_id,
            score=score,
            reason=f"Selected via {self._config.strategy.name}",
            alternatives=alternatives,
        )

    def _select_round_robin(
        self,
        candidates: List[tuple],
    ) -> tuple:
        """Round robin selection."""
        self._round_robin_index = (
            (self._round_robin_index + 1) % len(candidates)
        )
        node_id, node, stats = candidates[self._round_robin_index]
        return (node_id, node, stats, 1.0)

    def _select_random(
        self,
        candidates: List[tuple],
    ) -> tuple:
        """Random selection."""
        import random
        node_id, node, stats = random.choice(candidates)
        return (node_id, node, stats, 1.0)

    def _select_trust_first(
        self,
        candidates: List[tuple],
    ) -> tuple:
        """Select highest trust node."""
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x[1].trust_score,
            reverse=True
        )
        node_id, node, stats = sorted_candidates[0]
        return (node_id, node, stats, node.trust_score)

    def _select_least_connections(
        self,
        candidates: List[tuple],
    ) -> tuple:
        """Select node with fewest active connections."""
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x[2].active_connections
        )
        node_id, node, stats = sorted_candidates[0]
        score = 1.0 - (stats.active_connections / MAX_CONNECTIONS_PER_NODE)
        return (node_id, node, stats, score)

    def _select_weighted(
        self,
        candidates: List[tuple],
    ) -> tuple:
        """
        Weighted selection based on load, trust, latency, and success rate.

        Score formula:
            score = (1 - load) * load_weight
                  + trust_score * trust_weight
                  + (1 - norm_latency) * latency_weight
                  + success_rate * success_weight
        """
        scored_candidates = []

        # Find max latency for normalization
        max_latency = max(
            stats.avg_latency_ms for _, _, stats in candidates
        ) or 1.0

        for node_id, node, stats in candidates:
            # Calculate component scores
            load_score = 1.0 - stats.current_load
            trust_score = node.trust_score

            # Normalize latency (lower is better)
            if max_latency > 0:
                latency_score = 1.0 - (stats.avg_latency_ms / max_latency)
            else:
                latency_score = 1.0

            success_score = stats.success_rate

            # Weighted sum
            total_score = (
                load_score * self._config.load_weight +
                trust_score * self._config.trust_weight +
                latency_score * self._config.latency_weight +
                success_score * self._config.success_weight
            )

            scored_candidates.append((node_id, node, stats, total_score))

        # Select highest scoring node
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x[3],
            reverse=True
        )

        return sorted_candidates[0]

    def get_stats(self, node_id: str) -> Optional[NodeStats]:
        """Get statistics for a node."""
        return self._stats.get(node_id)

    def get_all_stats(self) -> Dict[str, NodeStats]:
        """Get statistics for all nodes."""
        return dict(self._stats)

    def get_available_nodes(
        self,
        model: Optional[str] = None,
    ) -> List[str]:
        """
        Get list of available node IDs.

        Args:
            model: Optional model filter

        Returns:
            List of available node IDs
        """
        available = []
        for node_id, stats in self._stats.items():
            if not stats.is_available:
                continue
            if self._circuit_breaker and self._circuit_breaker.is_open(node_id):
                continue
            if model and model not in stats.models:
                continue
            available.append(node_id)
        return available

    def get_models(self) -> Set[str]:
        """Get set of all available models across nodes."""
        models = set()
        for stats in self._stats.values():
            if stats.is_available:
                models.update(stats.models)
        return models


# ============================================================================
#  Factory Function
# ============================================================================

def create_load_balancer(
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.WEIGHTED_LEAST_CONN,
    enable_circuit_breaker: bool = True,
    **kwargs,
) -> LoadBalancer:
    """
    Create a LoadBalancer instance.

    Args:
        strategy: Load balancing strategy to use
        enable_circuit_breaker: Enable circuit breaker protection
        **kwargs: Additional config options

    Returns:
        Configured LoadBalancer
    """
    config = LoadBalancerConfig(
        strategy=strategy,
        enable_circuit_breaker=enable_circuit_breaker,
        **kwargs,
    )
    return LoadBalancer(config)


# ============================================================================
#  Exports
# ============================================================================

__all__ = [
    # Enums
    "LoadBalanceStrategy",
    "NodeHealth",
    # Data classes
    "NodeStats",
    "LoadBalancerConfig",
    "SelectionResult",
    "CircuitBreakerState",
    # Classes
    "CircuitBreaker",
    "LoadBalancer",
    # Factory
    "create_load_balancer",
    # Constants
    "DEFAULT_LOAD_WEIGHT",
    "DEFAULT_TRUST_WEIGHT",
    "DEFAULT_LATENCY_WEIGHT",
    "DEFAULT_SUCCESS_WEIGHT",
    "OVERLOAD_THRESHOLD",
    "MAX_CONNECTIONS_PER_NODE",
]

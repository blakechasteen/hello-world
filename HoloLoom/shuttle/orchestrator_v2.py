"""
HoloLoom Shuttle - Main Orchestrator (v2 with Error Handling)

Production-ready orchestrator with comprehensive error handling,
timeout enforcement, and graceful degradation.

Created: 2025-01-21
"""

from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Optional
import time
import logging

from .config import ShuttleConfig, ShuttleMode
from .exceptions import (
    ShuttleError,
    TimeoutError,
    WarpSearchError,
    YarnTraversalError,
    EntityExtractionError,
    BackendUnavailableError,
)
from .entity_extraction import Anchor, EntityExtractionFactory
from .trajectories import (
    TraversalConfig,
    TrajectoryStrategy,
    ALL_TRAJECTORIES,
    TRAJECTORY_BY_NAME,
)
from .trajectory_bandit import TrajectoryBandit, RewardCalculator
from .mcts import MCTSState, NeighborMap, run_mcts_with_timeout

logger = logging.getLogger(__name__)


# ============================================================================
# Interface Protocols
# ============================================================================

class WarpInterface(Protocol):
    """
    Interface for vector search backend (Qdrant).

    Warp provides semantic/fuzzy search over embedded content.
    """

    def search(
        self,
        query: str,
        top_k: int = 10,
        timeout_ms: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for anchor points.

        Args:
            query: Natural language query
            top_k: Number of results to return
            timeout_ms: Optional timeout in milliseconds

        Returns:
            List of dicts with keys: 'id', 'text', 'score', 'metadata'

        Raises:
            WarpSearchError: If search fails
            TimeoutError: If search exceeds timeout
        """
        ...


class YarnInterface(Protocol):
    """
    Interface for knowledge graph backend (Neo4j).

    Yarn provides structured traversal of symbolic relationships.
    """

    def build_neighbor_map(
        self,
        anchors: List[str],
        allowed_edge_types: List[str],
        max_depth: int,
        max_nodes: int,
        timeout_ms: Optional[int] = None
    ) -> tuple[NeighborMap, List[str]]:
        """
        Build neighbor map for MCTS from anchor nodes.

        Args:
            anchors: List of anchor node IDs
            allowed_edge_types: Edge types to traverse
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to include
            timeout_ms: Optional timeout in milliseconds

        Returns:
            (neighbor_map, all_node_ids)

        Raises:
            YarnTraversalError: If graph traversal fails
            TimeoutError: If traversal exceeds timeout
        """
        ...

    def describe_nodes(
        self,
        node_ids: List[str],
        timeout_ms: Optional[int] = None
    ) -> str:
        """
        Get human-readable description of nodes.

        Args:
            node_ids: List of node IDs to describe
            timeout_ms: Optional timeout in milliseconds

        Returns:
            Formatted string describing nodes and relationships

        Raises:
            YarnTraversalError: If description fails
        """
        ...


# ============================================================================
# Result Types
# ============================================================================

@dataclass
class WeaveResult:
    """
    Result from shuttle query execution.

    Contains both fuzzy (Warp) and structural (Yarn) evidence,
    along with complete metadata for debugging and analysis.
    """
    query: str
    fuzzy_evidence: List[Dict[str, Any]]  # Warp search results
    structural_claims: str  # Yarn graph description
    selected_nodes: List[str]  # Nodes selected by MCTS
    trajectory_used: str  # Which trajectory was selected
    reward: float  # Calculated reward
    search_time_ms: float  # Total execution time

    # Detailed timing breakdown
    timing: Dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Error tracking
    errors: List[str] = field(default_factory=list)
    degraded: bool = False  # True if graceful degradation occurred

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging."""
        return {
            "query": self.query,
            "trajectory_used": self.trajectory_used,
            "reward": self.reward,
            "search_time_ms": self.search_time_ms,
            "num_fuzzy_results": len(self.fuzzy_evidence),
            "num_selected_nodes": len(self.selected_nodes),
            "timing": self.timing,
            "metadata": self.metadata,
            "errors": self.errors,
            "degraded": self.degraded,
        }


# ============================================================================
# Main Orchestrator
# ============================================================================

class Shuttle:
    """
    Main orchestrator for Warp ↔ Yarn intersection with error handling.

    Coordinates:
    1. Warp search (vector/semantic)
    2. Entity extraction (anchor identification)
    3. Trajectory selection (Thompson Sampling)
    4. Yarn graph expansion (MCTS)
    5. Result synthesis

    Features:
    - Comprehensive error handling
    - Timeout enforcement at each stage
    - Graceful degradation on failures
    - Detailed timing and metadata tracking
    """

    def __init__(
        self,
        warp: WarpInterface,
        yarn: YarnInterface,
        config: Optional[ShuttleConfig] = None,
    ):
        """
        Args:
            warp: Vector search backend
            yarn: Knowledge graph backend
            config: Optional configuration (uses default if not provided)
        """
        self.warp = warp
        self.yarn = yarn
        self.config = config or ShuttleConfig()

        # Initialize trajectory bandit
        trajectory_names = [t.name for t in ALL_TRAJECTORIES]
        self.trajectory_bandit = TrajectoryBandit(trajectory_names)

        # Initialize entity extractor
        self.entity_extractor = EntityExtractionFactory.create(
            method=self.config.entity_extraction_method,
            enable_fallback=self.config.entity_extraction_fallback
        )

        logger.info(
            f"Shuttle initialized: mode={self.config.mode.value}, "
            f"mcts_sims={self.config.mcts_simulations}"
        )

    def intersect(
        self,
        query: str,
        trajectory_name: Optional[str] = None,
    ) -> WeaveResult:
        """
        Main query execution with error handling.

        Flow:
        1. Warp search → semantic anchors
        2. Entity extraction → anchor points
        3. Trajectory selection → graph traversal strategy
        4. Yarn neighbor map → graph structure
        5. MCTS → optimal path finding
        6. Result synthesis → WeaveResult

        Each stage has timeout enforcement and error handling.

        Args:
            query: Natural language query
            trajectory_name: Optional specific trajectory (overrides bandit)

        Returns:
            WeaveResult with fuzzy + structural evidence

        Raises:
            ShuttleError: If all fallback mechanisms fail
        """
        start_time = time.time()
        timing = {}
        errors = []
        degraded = False

        # Initialize empty result (fallback)
        result = WeaveResult(
            query=query,
            fuzzy_evidence=[],
            structural_claims="",
            selected_nodes=[],
            trajectory_used="none",
            reward=0.0,
            search_time_ms=0.0,
            timing=timing,
            errors=errors,
            degraded=degraded,
        )

        try:
            # ================================================================
            # Stage 1: Warp Search
            # ================================================================
            warp_start = time.time()
            try:
                warp_results = self.warp.search(
                    query,
                    top_k=self.config.warp_top_k,
                    timeout_ms=self.config.warp_timeout_ms
                )
                timing["warp_search_ms"] = (time.time() - warp_start) * 1000
                logger.debug(f"Warp search found {len(warp_results)} results")

            except Exception as e:
                errors.append(f"Warp search failed: {e}")
                logger.error(f"Warp search failed: {e}")

                if not self.config.enable_graceful_degradation:
                    raise WarpSearchError(str(e))

                # Graceful degradation: return empty result
                degraded = True
                result.errors = errors
                result.degraded = degraded
                result.search_time_ms = (time.time() - start_time) * 1000
                return result

            # ================================================================
            # Stage 2: Entity Extraction
            # ================================================================
            extract_start = time.time()
            try:
                anchors = self.entity_extractor.extract(
                    warp_results,
                    max_anchors=10
                )
                timing["entity_extraction_ms"] = (time.time() - extract_start) * 1000
                logger.debug(f"Extracted {len(anchors)} anchors")

            except EntityExtractionError as e:
                errors.append(f"Entity extraction failed: {e}")
                logger.warning(f"Entity extraction failed: {e}")

                # Fallback: use Warp results directly as anchors
                anchors = [
                    Anchor(
                        name=r.get("text", "")[:50],
                        type="Unknown",
                        node_id=r.get("id"),
                        source="warp_fallback"
                    )
                    for r in warp_results[:5]
                ]
                degraded = True

            if not anchors:
                errors.append("No anchors extracted")
                result.fuzzy_evidence = warp_results
                result.errors = errors
                result.degraded = degraded
                result.search_time_ms = (time.time() - start_time) * 1000
                return result

            # ================================================================
            # Stage 3: Trajectory Selection
            # ================================================================
            if trajectory_name is not None:
                selected_trajectory = trajectory_name
            else:
                selected_trajectory = self.trajectory_bandit.choose_trajectory()

            trajectory = TRAJECTORY_BY_NAME[selected_trajectory]
            traversal_config = trajectory.build_config(anchors)

            logger.debug(
                f"Selected trajectory: {selected_trajectory}, "
                f"depth={traversal_config.max_depth}, "
                f"max_nodes={traversal_config.max_nodes}"
            )

            # ================================================================
            # Stage 4: Yarn Neighbor Map
            # ================================================================
            yarn_start = time.time()
            try:
                anchor_ids = [a.node_id for a in anchors if a.node_id is not None]

                if not anchor_ids:
                    errors.append("No valid anchor node IDs")
                    result.fuzzy_evidence = warp_results
                    result.trajectory_used = selected_trajectory
                    result.errors = errors
                    result.degraded = True
                    result.search_time_ms = (time.time() - start_time) * 1000
                    return result

                neighbor_map, all_nodes = self.yarn.build_neighbor_map(
                    anchors=anchor_ids,
                    allowed_edge_types=traversal_config.allowed_edge_types,
                    max_depth=traversal_config.max_depth,
                    max_nodes=traversal_config.max_nodes,
                    timeout_ms=self.config.yarn_timeout_ms
                )
                timing["yarn_neighbor_map_ms"] = (time.time() - yarn_start) * 1000
                logger.debug(f"Yarn neighbor map: {len(neighbor_map)} nodes")

            except Exception as e:
                errors.append(f"Yarn neighbor map failed: {e}")
                logger.error(f"Yarn neighbor map failed: {e}")

                if not self.config.enable_graceful_degradation:
                    raise YarnTraversalError(str(e))

                # Graceful degradation: return Warp-only result
                result.fuzzy_evidence = warp_results
                result.structural_claims = "Graph traversal unavailable"
                result.trajectory_used = selected_trajectory
                result.errors = errors
                result.degraded = True
                result.search_time_ms = (time.time() - start_time) * 1000
                return result

            # ================================================================
            # Stage 5: MCTS (if enabled)
            # ================================================================
            if self.config.mcts_simulations > 0 and neighbor_map:
                mcts_start = time.time()
                try:
                    # Define rollout function
                    def rollout_fn(state: MCTSState) -> float:
                        node_count = len(state.selected_nodes)
                        if node_count == 0:
                            return 0.0
                        elif node_count < 5:
                            return 0.3
                        elif node_count < 15:
                            return 0.7
                        else:
                            return 0.9

                    mcts_result = run_mcts_with_timeout(
                        initial_node_id=anchor_ids[0],
                        neighbor_map=neighbor_map,
                        max_depth=traversal_config.max_depth,
                        num_simulations=self.config.mcts_simulations,
                        rollout_fn=rollout_fn,
                        timeout_ms=self.config.mcts_timeout_ms
                    )

                    selected_nodes = mcts_result.state.selected_nodes
                    timing["mcts_search_ms"] = (time.time() - mcts_start) * 1000
                    logger.debug(f"MCTS selected {len(selected_nodes)} nodes")

                except TimeoutError as e:
                    errors.append(f"MCTS timeout: {e}")
                    logger.warning(f"MCTS timeout: {e}")
                    # Fallback: use anchors only
                    selected_nodes = anchor_ids[:5]
                    degraded = True

                except Exception as e:
                    errors.append(f"MCTS failed: {e}")
                    logger.error(f"MCTS failed: {e}")
                    # Fallback: use anchors only
                    selected_nodes = anchor_ids[:5]
                    degraded = True

            else:
                # MCTS disabled or no neighbor map: use anchors
                selected_nodes = anchor_ids[:5]

            # ================================================================
            # Stage 6: Node Description
            # ================================================================
            describe_start = time.time()
            try:
                structural_claims = self.yarn.describe_nodes(
                    selected_nodes,
                    timeout_ms=self.config.yarn_timeout_ms
                )
                timing["yarn_describe_ms"] = (time.time() - describe_start) * 1000

            except Exception as e:
                errors.append(f"Node description failed: {e}")
                logger.error(f"Node description failed: {e}")
                structural_claims = f"Description unavailable ({len(selected_nodes)} nodes selected)"
                degraded = True

            # ================================================================
            # Stage 7: Reward Calculation
            # ================================================================
            reward = RewardCalculator.from_node_count(
                num_nodes=len(selected_nodes),
                max_nodes=traversal_config.max_nodes
            )

            # ================================================================
            # Stage 8: Bandit Update
            # ================================================================
            self.trajectory_bandit.update(selected_trajectory, reward)

            # ================================================================
            # Assemble Result
            # ================================================================
            search_time = (time.time() - start_time) * 1000

            result = WeaveResult(
                query=query,
                fuzzy_evidence=warp_results,
                structural_claims=structural_claims,
                selected_nodes=selected_nodes,
                trajectory_used=selected_trajectory,
                reward=reward,
                search_time_ms=search_time,
                timing=timing,
                metadata={
                    "mode": self.config.mode.value,
                    "mcts_simulations": self.config.mcts_simulations,
                    "num_anchors": len(anchors),
                    "traversal_config": {
                        "max_depth": traversal_config.max_depth,
                        "max_nodes": traversal_config.max_nodes,
                        "edge_types": traversal_config.allowed_edge_types,
                    },
                },
                errors=errors,
                degraded=degraded,
            )

            if self.config.log_performance_metrics:
                logger.info(
                    f"Shuttle query completed: "
                    f"trajectory={selected_trajectory}, "
                    f"nodes={len(selected_nodes)}, "
                    f"time={search_time:.1f}ms, "
                    f"degraded={degraded}"
                )

            return result

        except Exception as e:
            # Final catch-all handler
            errors.append(f"Shuttle execution failed: {e}")
            logger.error(f"Shuttle execution failed: {e}", exc_info=True)

            if not self.config.enable_graceful_degradation:
                raise ShuttleError(str(e))

            # Return minimal result
            result.errors = errors
            result.degraded = True
            result.search_time_ms = (time.time() - start_time) * 1000
            return result

    def get_trajectory_statistics(self) -> Dict[str, Dict]:
        """Get Thompson Sampling statistics for all trajectories."""
        return self.trajectory_bandit.get_statistics()

    def get_best_trajectories(self, top_k: int = 3) -> List[tuple[str, float]]:
        """Get top-k trajectories by mean reward."""
        return self.trajectory_bandit.get_best_policies(top_k)

    def save_state(self, filepath: str):
        """Save trajectory bandit state for persistence."""
        self.trajectory_bandit.save(filepath)

    @classmethod
    def load_state(
        cls,
        warp: WarpInterface,
        yarn: YarnInterface,
        filepath: str,
        config: Optional[ShuttleConfig] = None,
    ) -> "Shuttle":
        """
        Create shuttle with loaded bandit state.

        Args:
            warp: Warp interface
            yarn: Yarn interface
            filepath: Path to saved state
            config: Optional configuration

        Returns:
            Shuttle with loaded state
        """
        shuttle = cls(warp=warp, yarn=yarn, config=config)

        trajectory_names = [t.name for t in ALL_TRAJECTORIES]
        shuttle.trajectory_bandit = TrajectoryBandit.load(filepath, trajectory_names)

        return shuttle


# ============================================================================
# Factory Functions
# ============================================================================

def create_shuttle(
    warp: WarpInterface,
    yarn: YarnInterface,
    config: Optional[ShuttleConfig] = None,
) -> Shuttle:
    """
    Convenience function to create a Shuttle instance.

    Args:
        warp: Vector search backend
        yarn: Knowledge graph backend
        config: Optional configuration

    Returns:
        Configured Shuttle instance
    """
    return Shuttle(warp=warp, yarn=yarn, config=config)
"""
HoloLoom Shuttle - Main Orchestrator

Ties together Warp (vector search), Yarn (knowledge graph), policies, bandits, and MCTS
to provide intelligent context retrieval.
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol

from .bandits import PolicySelector, RewardCalculator
from .mcts import MCTSState, NeighborMap, run_mcts_search
from .policies import ALL_POLICIES, POLICY_BY_NAME, Anchor

# ============================================================================
# Interface Protocols
# ============================================================================

class WarpInterface(Protocol):
    """
    Interface for vector search backend (Qdrant).

    Warp provides fuzzy, semantic search over embedded content.
    """

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Semantic search for anchor points.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of dicts with keys: 'id', 'text', 'score', 'metadata'
        """
        ...


class YarnInterface(Protocol):
    """
    Interface for knowledge graph backend (Neo4j).

    Yarn provides structured, symbolic traversal of relationships.
    """

    def build_neighbor_map(
        self,
        anchors: list[str],
        allowed_edge_types: list[str],
        max_depth: int,
        max_nodes: int,
    ) -> tuple[NeighborMap, list[str]]:
        """
        Build neighbor map for MCTS from anchor nodes.

        Args:
            anchors: List of anchor node IDs
            allowed_edge_types: Edge types to traverse
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to include

        Returns:
            (neighbor_map, all_node_ids)
            - neighbor_map: Dict[node_id, List[neighbor_ids]]
            - all_node_ids: List of all nodes in the map
        """
        ...

    def describe_nodes(self, node_ids: list[str]) -> str:
        """
        Get human-readable description of nodes.

        Args:
            node_ids: List of node IDs to describe

        Returns:
            Formatted string describing the nodes and their relationships
        """
        ...


# ============================================================================
# Result Types
# ============================================================================

@dataclass
class ShuttleResult:
    """Result from shuttle query execution."""
    query: str
    fuzzy_evidence: list[dict[str, Any]]  # Warp search results
    structural_claims: str  # Yarn graph description
    selected_nodes: list[str]  # Nodes selected by MCTS
    policy_used: str  # Which policy was selected
    reward: float  # Calculated reward for this result
    search_time_ms: float  # Total execution time
    metadata: dict[str, Any]  # Additional metadata


# ============================================================================
# Main Orchestrator
# ============================================================================

class Shuttle:
    """
    Main orchestrator for Warp ↔ Yarn intersection.

    Coordinates:
    1. Warp search (vector/semantic)
    2. Policy selection (Thompson Sampling)
    3. Yarn graph expansion (MCTS)
    4. Result synthesis
    """

    def __init__(
        self,
        warp: WarpInterface,
        yarn: YarnInterface,
        num_mcts_simulations: int = 50,
        enable_learning: bool = True,
    ):
        """
        Args:
            warp: Vector search backend
            yarn: Knowledge graph backend
            num_mcts_simulations: MCTS iterations per query
            enable_learning: Whether to learn from query outcomes
        """
        self.warp = warp
        self.yarn = yarn
        self.num_mcts_simulations = num_mcts_simulations
        self.enable_learning = enable_learning

        # Initialize policy selector (Thompson Sampling bandit)
        policy_names = [p.name for p in ALL_POLICIES]
        self.policy_selector = PolicySelector(policy_names)

    def intersect(
        self,
        query: str,
        top_k_warp: int = 10,
        policy_name: str | None = None,
    ) -> ShuttleResult:
        """
        Main query execution: Warp search → Policy selection → Yarn expansion → MCTS.

        Args:
            query: Natural language query
            top_k_warp: Number of Warp results to retrieve
            policy_name: Optional specific policy to use (overrides Thompson Sampling)

        Returns:
            ShuttleResult with fuzzy + structural evidence
        """
        start_time = time.time()

        # Step 1: Warp search (semantic/fuzzy)
        warp_results = self.warp.search(query, top_k=top_k_warp)

        # Convert Warp results to anchors
        anchors = self._warp_results_to_anchors(warp_results)

        # Step 2: Policy selection (Thompson Sampling or manual override)
        if policy_name is not None:
            selected_policy_name = policy_name
        else:
            selected_policy_name = self.policy_selector.select([p.name for p in ALL_POLICIES])

        policy = POLICY_BY_NAME[selected_policy_name]

        # Step 3: Get expansion config from policy
        expansion_config = policy.build_config(anchors)

        # Step 4: Build neighbor map from Yarn
        anchor_ids = [a.node_id for a in anchors if a.node_id is not None]

        if not anchor_ids:
            # No anchors have node IDs - fallback to empty result
            return ShuttleResult(
                query=query,
                fuzzy_evidence=warp_results,
                structural_claims="No anchor nodes found in knowledge graph",
                selected_nodes=[],
                policy_used=selected_policy_name,
                reward=0.0,
                search_time_ms=(time.time() - start_time) * 1000,
                metadata={"mcts_simulations": 0},
            )

        neighbor_map, all_nodes = self.yarn.build_neighbor_map(
            anchors=anchor_ids,
            allowed_edge_types=expansion_config.allowed_edge_types,
            max_depth=expansion_config.max_depth,
            max_nodes=expansion_config.max_nodes,
        )

        # Step 5: Run MCTS to find best expansion
        if not neighbor_map:
            # No neighbors found - return just anchors
            selected_nodes = anchor_ids[:1] if anchor_ids else []
        else:
            # Define rollout function (simple heuristic for now)
            def rollout_fn(state: MCTSState) -> float:
                # Reward based on number of nodes discovered and depth
                # More nodes = more context (up to a point)
                node_count = len(state.selected_nodes)
                depth_penalty = state.depth * 0.05  # Slight penalty for deep paths

                if node_count == 0:
                    return 0.0
                elif node_count < 5:
                    return 0.3 - depth_penalty
                elif node_count < 15:
                    return 0.7 - depth_penalty
                else:
                    return 0.9 - depth_penalty

            mcts_result = run_mcts_search(
                initial_node_id=anchor_ids[0],
                neighbor_map=neighbor_map,
                max_depth=expansion_config.max_depth,
                num_simulations=self.num_mcts_simulations,
                rollout_fn=rollout_fn,
            )

            selected_nodes = mcts_result.state.selected_nodes

        # Step 6: Get descriptions of selected nodes from Yarn
        structural_claims = self.yarn.describe_nodes(selected_nodes)

        # Step 7: Calculate reward
        reward = RewardCalculator.from_answer_quality(
            answer_generated=bool(structural_claims),
            answer_length=len(structural_claims),
            retrieval_count=len(selected_nodes),
        )

        # Step 8: Update policy bandit (if learning enabled)
        if self.enable_learning:
            self.policy_selector.update(selected_policy_name, reward)

        search_time = (time.time() - start_time) * 1000

        return ShuttleResult(
            query=query,
            fuzzy_evidence=warp_results,
            structural_claims=structural_claims,
            selected_nodes=selected_nodes,
            policy_used=selected_policy_name,
            reward=reward,
            search_time_ms=search_time,
            metadata={
                "mcts_simulations": self.num_mcts_simulations,
                "policy_config": {
                    "max_depth": expansion_config.max_depth,
                    "max_nodes": expansion_config.max_nodes,
                    "allowed_edge_types": expansion_config.allowed_edge_types,
                },
            },
        )

    def _warp_results_to_anchors(self, warp_results: list[dict[str, Any]]) -> list[Anchor]:
        """
        Convert Warp search results to Anchor objects.

        Args:
            warp_results: List of dicts from Warp.search()

        Returns:
            List of Anchor objects
        """
        anchors = []
        for result in warp_results:
            # Extract metadata
            metadata = result.get('metadata', {})

            anchor = Anchor(
                name=result.get('text', '')[:50],  # First 50 chars as name
                type=metadata.get('type', 'Unknown'),
                node_id=result.get('id'),  # Warp ID should map to Yarn node ID
            )
            anchors.append(anchor)

        return anchors

    def get_policy_statistics(self) -> dict[str, dict]:
        """Get Thompson Sampling statistics for all policies."""
        return self.policy_selector.get_statistics()

    def get_best_policies(self, top_k: int = 3) -> list[tuple[str, float]]:
        """Get top-k policies by mean reward."""
        return self.policy_selector.get_best_policies(top_k)

    def save_bandit_state(self, filepath: str):
        """Save policy bandit state for persistence."""
        self.policy_selector.save(filepath)

    @classmethod
    def load_bandit_state(
        cls,
        warp: WarpInterface,
        yarn: YarnInterface,
        filepath: str,
        **kwargs
    ) -> "Shuttle":
        """
        Create shuttle with loaded bandit state.

        Args:
            warp: Warp interface instance
            yarn: Yarn interface instance
            filepath: Path to saved bandit state
            **kwargs: Additional arguments for Shuttle constructor

        Returns:
            Shuttle instance with loaded state
        """
        shuttle = cls(warp=warp, yarn=yarn, **kwargs)

        policy_names = [p.name for p in ALL_POLICIES]
        shuttle.policy_selector = PolicySelector.load(filepath, policy_names)

        return shuttle


# ============================================================================
# Convenience Functions
# ============================================================================

def create_shuttle(
    warp: WarpInterface,
    yarn: YarnInterface,
    num_mcts_simulations: int = 50,
    enable_learning: bool = True,
) -> Shuttle:
    """
    Convenience function to create a Shuttle instance.

    Args:
        warp: Vector search backend
        yarn: Knowledge graph backend
        num_mcts_simulations: MCTS iterations per query
        enable_learning: Whether to learn from query outcomes

    Returns:
        Configured Shuttle instance
    """
    return Shuttle(
        warp=warp,
        yarn=yarn,
        num_mcts_simulations=num_mcts_simulations,
        enable_learning=enable_learning,
    )

"""
HoloLoom Shuttle - Monte Carlo Tree Search (MCTS)

Treats graph expansion as a sequential decision problem.
Explores different expansion paths to find high-quality context for answering queries.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

# ============================================================================
# Type Aliases
# ============================================================================

# Maps node_id -> list of neighbor node_ids
NeighborMap = dict[str, list[str]]


# ============================================================================
# MCTS State Representation
# ============================================================================

@dataclass
class MCTSState:
    """
    Abstract state for graph expansion.

    Represents the current set of selected nodes and depth in the search tree.
    In real usage, this tracks which nodes have been included in the context.
    """
    selected_nodes: list[str]
    depth: int

    def is_terminal(self, max_depth: int) -> bool:
        """Check if this is a terminal state (can't expand further)."""
        return self.depth >= max_depth

    def available_actions(self, neighbor_map: NeighborMap) -> list[str]:
        """
        Return a list of candidate node_ids we could expand next.

        Args:
            neighbor_map: Maps node_id -> list of neighbor node_ids

        Returns:
            List of node_ids that can be added next
        """
        if not self.selected_nodes:
            return []

        # Get neighbors of the most recently added node
        frontier_node = self.selected_nodes[-1]
        neighbors = neighbor_map.get(frontier_node, [])

        # Filter to nodes we haven't already selected
        selected_set = set(self.selected_nodes)
        return [n for n in neighbors if n not in selected_set]


# ============================================================================
# MCTS Node
# ============================================================================

@dataclass
class MCTSNode:
    """
    A node in the MCTS search tree.

    Tracks visit statistics and children for the UCB1 selection algorithm.
    """
    state: MCTSState
    parent: MCTSNode | None = None
    action_from_parent: str | None = None  # node_id that was added

    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    def ucb1(self, exploration_constant: float = 1.4) -> float:
        """
        Calculate UCB1 score for this node.

        UCB1 = exploitation + exploration
             = (average_value) + C * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_constant: C parameter (sqrt(2) ≈ 1.414 is theoretically optimal)

        Returns:
            UCB1 score (higher = more promising to explore)
        """
        if self.visits == 0:
            return float("inf")  # Unvisited nodes have infinite priority

        # Exploitation: average value from this node
        exploitation = self.value_sum / self.visits

        # Exploration: bonus for less-visited nodes
        if self.parent is None or self.parent.visits == 0:
            exploration = 0.0
        else:
            exploration = exploration_constant * math.sqrt(
                math.log(self.parent.visits + 1) / (self.visits + 1e-9)
            )

        return exploitation + exploration


# ============================================================================
# MCTS Algorithm
# ============================================================================

class MCTS:
    """
    Monte Carlo Tree Search for graph expansion.

    Explores different ways to expand the graph from anchors,
    trying to find the expansion that provides the best context.
    """

    def __init__(
        self,
        max_depth: int,
        num_simulations: int,
        exploration_constant: float = 1.4,
    ):
        """
        Args:
            max_depth: Maximum depth for graph expansion
            num_simulations: Number of MCTS iterations to run
            exploration_constant: UCB1 exploration parameter
        """
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant

    def search(
        self,
        root_state: MCTSState,
        neighbor_map: NeighborMap,
        rollout_fn: Callable[[MCTSState], float],
    ) -> MCTSState:
        """
        Perform MCTS and return the best final state found.

        Args:
            root_state: Initial state (anchor nodes)
            neighbor_map: Maps node_id -> list of neighbor node_ids
            rollout_fn: Function that evaluates a state and returns reward in [0, 1]

        Returns:
            Best MCTSState found
        """
        root = MCTSNode(state=root_state)

        for _ in range(self.num_simulations):
            # 1. Selection: traverse tree using UCB1
            node = self._select(root, neighbor_map)

            # 2. Expansion: add a new child if possible
            if not node.state.is_terminal(self.max_depth):
                node = self._expand(node, neighbor_map)

            # 3. Simulation: evaluate the node
            reward = self._rollout(node.state, neighbor_map, rollout_fn)

            # 4. Backpropagation: update statistics
            self._backpropagate(node, reward)

        # Choose the child with highest visit count (most promising)
        if not root.children:
            return root.state

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.state

    def _select(self, node: MCTSNode, neighbor_map: NeighborMap) -> MCTSNode:
        """
        Selection phase: traverse tree using UCB1 until we reach a node
        that can be expanded or is terminal.

        Args:
            node: Starting node (typically root)
            neighbor_map: Graph neighbor information

        Returns:
            Node to expand or simulate from
        """
        while not node.state.is_terminal(self.max_depth):
            actions = node.state.available_actions(neighbor_map)

            if not actions:
                # No more neighbors to explore from this node
                return node

            if len(node.children) < len(actions):
                # There are still unexplored actions from this node
                return node

            # All actions explored: pick the best child using UCB1
            node = max(node.children, key=lambda c: c.ucb1(self.exploration_constant))

        return node

    def _expand(self, node: MCTSNode, neighbor_map: NeighborMap) -> MCTSNode:
        """
        Expansion phase: try one untried action and create a child node.

        Args:
            node: Node to expand from
            neighbor_map: Graph neighbor information

        Returns:
            Newly created child node
        """
        actions = node.state.available_actions(neighbor_map)

        if not actions:
            return node  # Can't expand

        # Find actions we haven't tried yet
        used_actions = {child.action_from_parent for child in node.children}
        untried_actions = [a for a in actions if a not in used_actions]

        if not untried_actions:
            return node  # All actions already tried

        # Pick a random untried action
        action = random.choice(untried_actions)

        # Create new state by adding this node
        new_state = MCTSState(
            selected_nodes=node.state.selected_nodes + [action],
            depth=node.state.depth + 1,
        )

        # Create child node
        child = MCTSNode(
            state=new_state,
            parent=node,
            action_from_parent=action,
        )
        node.children.append(child)

        return child

    def _rollout(
        self,
        state: MCTSState,
        neighbor_map: NeighborMap,
        rollout_fn: Callable[[MCTSState], float],
    ) -> float:
        """
        Simulation phase: evaluate the quality of this state.

        For now, just call the provided rollout_fn.

        In the future, we could simulate additional random steps here
        using neighbor_map and a stochastic policy.

        Args:
            state: State to evaluate
            neighbor_map: Graph neighbor information
            rollout_fn: Evaluation function

        Returns:
            Reward in [0, 1]
        """
        return rollout_fn(state)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: update statistics up the tree.

        Args:
            node: Leaf node where simulation occurred
            reward: Observed reward
        """
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent


# ============================================================================
# Result Type
# ============================================================================

@dataclass
class MCTSResult:
    """Result from MCTS search."""
    state: MCTSState
    search_time: float
    iterations_run: int
    best_reward: float

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "selected_nodes": self.state.selected_nodes,
            "depth": self.state.depth,
            "total_nodes": len(self.state.selected_nodes),
            "search_time_ms": self.search_time * 1000,
            "iterations": self.iterations_run,
            "best_reward": self.best_reward,
        }


# ============================================================================
# Convenience Function
# ============================================================================

def run_mcts_search(
    initial_node_id: str,
    neighbor_map: NeighborMap,
    max_depth: int,
    num_simulations: int,
    rollout_fn: Callable[[MCTSState], float],
) -> MCTSResult:
    """
    Convenience function to run MCTS search.

    Args:
        initial_node_id: Starting node (anchor from Warp)
        neighbor_map: Maps node_id -> list of neighbor node_ids
        max_depth: Maximum expansion depth
        num_simulations: Number of MCTS iterations
        rollout_fn: Function to evaluate state quality

    Returns:
        MCTSResult with best state found
    """
    import time

    start_time = time.time()

    # Create initial state
    root_state = MCTSState(
        selected_nodes=[initial_node_id],
        depth=0,
    )

    # Run MCTS
    mcts = MCTS(
        max_depth=max_depth,
        num_simulations=num_simulations,
    )

    best_state = mcts.search(root_state, neighbor_map, rollout_fn)

    search_time = time.time() - start_time

    # Evaluate final state
    final_reward = rollout_fn(best_state)

    return MCTSResult(
        state=best_state,
        search_time=search_time,
        iterations_run=num_simulations,
        best_reward=final_reward,
    )

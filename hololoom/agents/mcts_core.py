"""
Monte Carlo Tree Search - Core Engine

MCTS all the way down - powers everything:
- Working memory focus exploration
- Pattern validation and selection
- Multi-step query planning
- Decision making under uncertainty

Key Innovation: Hierarchical MCTS at multiple scales:
- Micro: Focus vector adjustments (milliseconds)
- Meso: Pattern/tool selection (100ms)
- Macro: Multi-query planning (seconds)

Philosophy: Don't guess the best path - simulate thousands and learn.
"""

from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
import math
import time
from abc import ABC, abstractmethod


# ============================================================================
# MCTS Node - Universal Tree Node
# ============================================================================

@dataclass
class MCTSNode:
    """
    Universal MCTS node for any state space.

    Tracks:
    - State: Current configuration (working memory, pattern, etc.)
    - Statistics: Visits, value, UCT score
    - Tree structure: Parent, children
    - Action: What led to this state
    """
    state: Any  # Can be WorkingMemoryState, Pattern, Query, etc.
    parent: Optional['MCTSNode'] = None
    action: Optional[Any] = None  # Action that led here

    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0

    # Tree structure
    children: List['MCTSNode'] = field(default_factory=list)

    # Metadata
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def value(self) -> float:
        """Average value (exploitation)"""
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def uct_score(self, exploration_weight: float = 1.414, breakthrough_bias: float = 0.0) -> float:
        """
        Upper Confidence Bound for Trees (UCT) with breakthrough bias.

        Balance exploitation (known good) vs exploration (unknown potential),
        plus bias toward breakthrough regions.

        Args:
            exploration_weight: C_p constant (sqrt(2) ≈ 1.414 is standard)
            breakthrough_bias: Bonus for breakthrough paths (0.0 - 1.0)

        Returns:
            UCT score (higher = should explore this node)
        """
        if self.visits == 0:
            return float('inf')  # Always explore unvisited nodes

        if self.parent is None or self.parent.visits == 0:
            return self.value()

        exploitation = self.value()
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        # Add breakthrough bias
        return exploitation + exploration + breakthrough_bias

    def best_child(self, exploration_weight: float = 1.414) -> Optional['MCTSNode']:
        """Get child with highest UCT score"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.uct_score(exploration_weight))

    def most_visited_child(self) -> Optional['MCTSNode']:
        """Get most visited child (final decision)"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)


# ============================================================================
# MCTS State Space - Abstract Interface
# ============================================================================

class MCTSStateSpace(ABC):
    """
    Abstract state space for MCTS.

    Implement this to use MCTS with any domain:
    - Working memory exploration
    - Pattern selection
    - Query planning
    - Tool selection
    """

    @abstractmethod
    def get_legal_actions(self, state: Any) -> List[Any]:
        """Get all legal actions from this state"""
        pass

    @abstractmethod
    def apply_action(self, state: Any, action: Any) -> Any:
        """Apply action to state, return new state"""
        pass

    @abstractmethod
    async def evaluate(self, state: Any) -> float:
        """
        Evaluate state quality (0.0 to 1.0).

        This is the reward signal that guides MCTS.
        Higher = better state.
        """
        pass

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Check if state is terminal (end of episode)"""
        pass

    @abstractmethod
    def copy_state(self, state: Any) -> Any:
        """Deep copy state for simulation"""
        pass


# ============================================================================
# MCTS Engine - Core Algorithm
# ============================================================================

class MCTSEngine:
    """
    Universal Monte Carlo Tree Search engine.

    Implements the four phases:
    1. Selection: Traverse tree using UCT
    2. Expansion: Add new child node
    3. Simulation: Rollout to terminal state
    4. Backpropagation: Update statistics

    Usage:
        engine = MCTSEngine(state_space)
        best_action = await engine.search(initial_state, n_simulations=1000)
    """

    def __init__(
        self,
        state_space: MCTSStateSpace,
        exploration_weight: float = 1.414,
        discount_factor: float = 0.95,
        max_depth: int = 10,
        breakthrough_detector: Optional[Any] = None  # BreakthroughDetector
    ):
        self.state_space = state_space
        self.exploration_weight = exploration_weight
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.breakthrough_detector = breakthrough_detector

        # Statistics
        self.total_simulations = 0
        self.total_time = 0.0

        # Breakthrough tracking
        self.previous_reward = 0.0
        self.previous_confidence = 0.0
        self.received_breakthroughs: List[Any] = []  # From parallel searches

    async def search(
        self,
        initial_state: Any,
        n_simulations: int = 100,
        time_budget: Optional[float] = None
    ) -> Tuple[Any, MCTSNode]:
        """
        Run MCTS search from initial state.

        Args:
            initial_state: Starting state
            n_simulations: Number of MCTS iterations
            time_budget: Optional time limit (seconds)

        Returns:
            (best_action, root_node)
        """
        start_time = time.time()

        # Create root node
        root = MCTSNode(state=initial_state, depth=0)

        # Run simulations
        for i in range(n_simulations):
            # Check time budget
            if time_budget and (time.time() - start_time) > time_budget:
                break

            # MCTS iteration
            await self._iterate(root)

            self.total_simulations += 1

        self.total_time += time.time() - start_time

        # Select best action (most visited child)
        best_child = root.most_visited_child()
        best_action = best_child.action if best_child else None

        return best_action, root

    async def _iterate(self, root: MCTSNode):
        """Single MCTS iteration (select, expand, simulate, backprop)"""

        # PHASE 1: SELECTION (with breakthrough bias)
        node = self._select(root)

        # PHASE 2: EXPANSION
        if not self.state_space.is_terminal(node.state) and node.visits > 0:
            node = self._expand(node)

        # PHASE 3: SIMULATION (ROLLOUT)
        value = await self._simulate(node)

        # PHASE 4: BACKPROPAGATION
        self._backpropagate(node, value)

        # PHASE 5: BREAKTHROUGH DETECTION (if enabled)
        if self.breakthrough_detector and self.total_simulations > 0:
            breakthrough = self.breakthrough_detector.detect_breakthrough(
                reward=value,
                previous_reward=self.previous_reward,
                confidence=node.value(),
                previous_confidence=self.previous_confidence,
                action_sequence=self._get_action_sequence(node),
                state_signature=self._get_state_signature(node.state),
                visits=node.visits,
                search_id=f"search_{id(self)}"
            )

            # Update tracking
            self.previous_reward = value
            self.previous_confidence = node.value()

    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Select node to expand using UCT with breakthrough bias.

        Traverse tree selecting best children until we reach:
        - Unvisited node, or
        - Terminal node, or
        - Node with unexpanded actions
        """
        node = root

        while node.children and not self.state_space.is_terminal(node.state):
            # Get breakthrough bias for each child
            if self.breakthrough_detector:
                best_child = None
                best_score = -float('inf')

                for child in node.children:
                    # Get breakthrough bias
                    state_sig = self._get_state_signature(child.state)
                    bias = self.breakthrough_detector.get_breakthrough_bias(state_sig)

                    # UCT score with breakthrough bias
                    score = child.uct_score(self.exploration_weight, bias)

                    if score > best_score:
                        best_score = score
                        best_child = child

                node = best_child
            else:
                # Standard UCT selection
                node = node.best_child(self.exploration_weight)

            if node is None:
                break

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand node by adding one child.

        Strategy: Add children for all legal actions (eager expansion).
        Then select one child to simulate.
        """
        # Get legal actions
        legal_actions = self.state_space.get_legal_actions(node.state)

        # Filter out actions we've already expanded
        existing_actions = set(child.action for child in node.children if child.action is not None)
        unexpanded_actions = [a for a in legal_actions if a not in existing_actions]

        if not unexpanded_actions:
            # All actions expanded, select best child
            return node.best_child(self.exploration_weight) or node

        # Choose random unexpanded action
        action = np.random.choice(unexpanded_actions)

        # Apply action
        new_state = self.state_space.apply_action(node.state, action)

        # Create child node
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action,
            depth=node.depth + 1
        )

        node.children.append(child)

        return child

    async def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate from node to terminal state (rollout).

        Uses random policy for speed. Could use learned policy for better results.

        Returns:
            Discounted cumulative reward
        """
        state = self.state_space.copy_state(node.state)
        depth = node.depth
        total_value = 0.0

        # Rollout with random policy
        while not self.state_space.is_terminal(state) and depth < self.max_depth:
            # Get legal actions
            actions = self.state_space.get_legal_actions(state)

            if not actions:
                break

            # Random action
            action = np.random.choice(actions)

            # Apply action
            state = self.state_space.apply_action(state, action)

            # Evaluate state
            value = await self.state_space.evaluate(state)

            # Discount
            discount = self.discount_factor ** (depth - node.depth)
            total_value += discount * value

            depth += 1

        # Final state evaluation
        final_value = await self.state_space.evaluate(state)
        discount = self.discount_factor ** (depth - node.depth)
        total_value += discount * final_value

        return total_value

    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate value up the tree.

        Updates visit counts and values for node and all ancestors.
        """
        current = node

        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _get_action_sequence(self, node: MCTSNode) -> List[Any]:
        """Get sequence of actions from root to node"""
        sequence = []
        current = node

        while current and current.action is not None:
            sequence.insert(0, current.action)
            current = current.parent

        return sequence

    def _get_state_signature(self, state: Any) -> str:
        """Get unique signature for state (for breakthrough detection)"""
        # Simple hash-based signature
        # Could be overridden for domain-specific signatures
        return str(hash(str(state)))

    def receive_breakthrough(self, breakthrough: Any):
        """
        Receive breakthrough from parallel search.

        Called by FeedForwardBroadcaster.
        """
        self.received_breakthroughs.append(breakthrough)

        # Update detector baseline if available
        if self.breakthrough_detector:
            self.breakthrough_detector.update_baseline(breakthrough.reward)

    def get_statistics(self) -> Dict[str, Any]:
        """Get MCTS engine statistics"""
        stats = {
            'total_simulations': self.total_simulations,
            'total_time': self.total_time,
            'avg_time_per_simulation': self.total_time / self.total_simulations if self.total_simulations > 0 else 0.0,
            'simulations_per_second': self.total_simulations / self.total_time if self.total_time > 0 else 0.0
        }

        # Add breakthrough stats if detector present
        if self.breakthrough_detector:
            stats['breakthrough'] = self.breakthrough_detector.get_stats()
            stats['received_breakthroughs'] = len(self.received_breakthroughs)

        return stats


# ============================================================================
# Hierarchical MCTS - Multi-Scale Planning
# ============================================================================

class HierarchicalMCTS:
    """
    Hierarchical MCTS for multi-scale planning.

    Scales:
    - Macro: High-level goal decomposition (multi-query sequences)
    - Meso: Mid-level action selection (tool choice, pattern selection)
    - Micro: Low-level parameter tuning (focus adjustment)

    Each level uses MCTS independently but passes goals down and results up.
    """

    def __init__(
        self,
        macro_state_space: MCTSStateSpace,
        meso_state_space: MCTSStateSpace,
        micro_state_space: MCTSStateSpace,
        macro_budget: int = 50,
        meso_budget: int = 100,
        micro_budget: int = 200
    ):
        self.macro_engine = MCTSEngine(macro_state_space, max_depth=5)
        self.meso_engine = MCTSEngine(meso_state_space, max_depth=10)
        self.micro_engine = MCTSEngine(micro_state_space, max_depth=20)

        self.macro_budget = macro_budget
        self.meso_budget = meso_budget
        self.micro_budget = micro_budget

    async def plan(self, initial_state: Any) -> List[Any]:
        """
        Hierarchical planning.

        1. Macro: Plan high-level goals
        2. For each goal:
           a. Meso: Plan actions
           b. For each action:
              i. Micro: Optimize parameters

        Returns:
            Complete action sequence with optimized parameters
        """
        # MACRO: Plan high-level goals
        macro_action, macro_root = await self.macro_engine.search(
            initial_state,
            n_simulations=self.macro_budget
        )

        # Extract goal sequence from macro tree
        goals = self._extract_sequence(macro_root)

        # MESO + MICRO: Plan each goal
        complete_plan = []

        for goal in goals:
            # MESO: Plan actions for this goal
            meso_action, meso_root = await self.meso_engine.search(
                goal,
                n_simulations=self.meso_budget
            )

            actions = self._extract_sequence(meso_root)

            # MICRO: Optimize each action
            for action in actions:
                optimized_action, _ = await self.micro_engine.search(
                    action,
                    n_simulations=self.micro_budget
                )

                complete_plan.append(optimized_action)

        return complete_plan

    def _extract_sequence(self, root: MCTSNode) -> List[Any]:
        """Extract action sequence from MCTS tree (follow most visited path)"""
        sequence = []
        node = root

        while node.children:
            node = node.most_visited_child()
            if node and node.action is not None:
                sequence.append(node.action)

        return sequence


# ============================================================================
# MCTS Utilities
# ============================================================================

def visualize_mcts_tree(root: MCTSNode, max_depth: int = 3) -> str:
    """
    Visualize MCTS tree (ASCII art).

    Shows visit counts and values for debugging.
    """
    lines = []

    def _traverse(node: MCTSNode, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return

        # Node info
        value = node.value()
        action_str = str(node.action) if node.action else "ROOT"
        lines.append(f"{prefix}{action_str} [visits={node.visits}, value={value:.3f}]")

        # Children
        for i, child in enumerate(sorted(node.children, key=lambda c: c.visits, reverse=True)):
            is_last = i == len(node.children) - 1
            child_prefix = prefix + ("└─ " if is_last else "├─ ")
            next_prefix = prefix + ("   " if is_last else "│  ")
            _traverse(child, child_prefix, depth + 1)

    _traverse(root)
    return "\n".join(lines)

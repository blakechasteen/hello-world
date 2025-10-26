"""
MCTS-Enhanced Convergence Engine - The Flux Capacitor
=====================================================
Monte Carlo Tree Search integrated with Thompson Sampling for decision-making.

Philosophy:
The "flux capacitor" performs temporal search across possible decisions:
- MCTS explores the decision tree
- Thompson Sampling guides exploration at each node
- Simulations predict future outcomes
- Best path collapses to final decision

This is Thompson Sampling ALL THE WAY DOWN with MCTS search on top!

Architecture:
1. Each decision is a tree node
2. MCTS runs simulations to evaluate paths
3. TS samples from bandit priors at each node
4. UCB1 balances exploration/exploitation
5. Best action emerges from tree statistics
"""

import logging
import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# MCTS Node
# ============================================================================

@dataclass
class MCTSNode:
    """
    Monte Carlo Tree Search Node.

    Represents a decision point in the search tree.
    Uses Thompson Sampling to guide rollouts.
    """
    tool_idx: int  # Which tool this node represents
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)

    # Statistics
    visits: int = 0
    value_sum: float = 0.0

    # Thompson Sampling priors (inherited from bandit)
    alpha: float = 1.0  # Beta distribution success parameter
    beta: float = 1.0   # Beta distribution failure parameter

    @property
    def average_value(self) -> float:
        """Average value from visits."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def thompson_prior(self) -> float:
        """Thompson Sampling prior (expected value of Beta)."""
        return self.alpha / (self.alpha + self.beta)

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """
        UCB1 score for tree policy.

        Balances exploitation (average value) and exploration (visit count).

        Args:
            exploration_constant: C parameter for UCB1 (sqrt(2) is standard)

        Returns:
            UCB1 score
        """
        if self.visits == 0:
            return float('inf')  # Always explore unvisited nodes

        if self.parent is None or self.parent.visits == 0:
            return self.average_value

        # UCB1 formula
        exploitation = self.average_value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def thompson_sample(self) -> float:
        """
        Sample from Thompson Sampling prior.

        Returns:
            Sampled value from Beta(alpha, beta)
        """
        return np.random.beta(self.alpha, self.beta)

    def update(self, value: float) -> None:
        """
        Update node statistics.

        Args:
            value: Outcome value from simulation
        """
        self.visits += 1
        self.value_sum += value

        # Update Thompson Sampling priors
        if value > 0:
            self.alpha += value
        else:
            self.beta += abs(value)

    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0


# ============================================================================
# MCTS Flux Capacitor
# ============================================================================

class MCTSFluxCapacitor:
    """
    Monte Carlo Tree Search decision engine.

    Performs temporal search through decision space using:
    - UCB1 for tree policy
    - Thompson Sampling for rollout policy
    - Statistical aggregation for final decision

    The "flux capacitor" searches across time (simulations) to find
    the best decision path.
    """

    def __init__(
        self,
        n_tools: int,
        n_simulations: int = 100,
        exploration_constant: float = 1.414,
        thompson_priors: Optional[np.ndarray] = None
    ):
        """
        Initialize MCTS Flux Capacitor.

        Args:
            n_tools: Number of tools (actions)
            n_simulations: Number of MCTS simulations to run
            exploration_constant: UCB1 exploration parameter (sqrt(2) default)
            thompson_priors: Initial Thompson Sampling priors [n_tools, 2] (alpha, beta)
        """
        self.n_tools = n_tools
        self.n_simulations = n_simulations
        self.exploration_constant = exploration_constant

        # Initialize root with Thompson priors
        if thompson_priors is not None:
            self.alpha_priors = thompson_priors[:, 0]
            self.beta_priors = thompson_priors[:, 1]
        else:
            self.alpha_priors = np.ones(n_tools)
            self.beta_priors = np.ones(n_tools)

        # Statistics
        self.total_simulations = 0
        self.best_tool_history = []

        logger.info(f"MCTS Flux Capacitor initialized: {n_tools} tools, {n_simulations} sims")

    def search(
        self,
        neural_probs: Optional[np.ndarray] = None,
        features: Optional[Dict] = None
    ) -> Tuple[int, float, Dict]:
        """
        Run MCTS search to find best tool.

        Args:
            neural_probs: Neural network probabilities (optional, for blending)
            features: Feature dict for reward estimation

        Returns:
            Tuple of (tool_idx, confidence, stats)
        """
        # Create root node
        root = MCTSNode(tool_idx=-1)  # Root has no tool

        # Create child nodes for each tool
        for i in range(self.n_tools):
            child = MCTSNode(
                tool_idx=i,
                parent=root,
                alpha=self.alpha_priors[i],
                beta=self.beta_priors[i]
            )
            root.children.append(child)

        # Run simulations
        for sim_idx in range(self.n_simulations):
            # 1. Selection: Walk down tree using UCB1
            node = self._select(root)

            # 2. Expansion: Add children if not leaf (for deeper trees later)
            # Currently single-level tree, skip expansion

            # 3. Simulation (Rollout): Use Thompson Sampling to estimate value
            value = self._simulate(node, neural_probs, features)

            # 4. Backpropagation: Update all nodes in path
            self._backpropagate(node, value)

        # Select best tool based on visit counts (most robust)
        tool_idx = max(root.children, key=lambda n: n.visits).tool_idx

        # Calculate confidence based on visit distribution
        visits = np.array([child.visits for child in root.children])
        confidence = visits[tool_idx] / visits.sum()

        # Statistics
        stats = {
            "simulations": self.n_simulations,
            "visit_counts": visits.tolist(),
            "average_values": [child.average_value for child in root.children],
            "thompson_priors": [child.thompson_prior for child in root.children],
            "ucb1_scores": [child.ucb1_score(self.exploration_constant) for child in root.children]
        }

        self.total_simulations += self.n_simulations
        self.best_tool_history.append(tool_idx)

        logger.info(f"MCTS search complete: tool={tool_idx}, confidence={confidence:.1%}, visits={visits.tolist()}")

        return tool_idx, confidence, stats

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Walk down tree using UCB1.

        Args:
            node: Current node

        Returns:
            Selected leaf node
        """
        while not node.is_leaf():
            # Select child with highest UCB1 score
            node = max(node.children, key=lambda n: n.ucb1_score(self.exploration_constant))

        return node

    def _simulate(
        self,
        node: MCTSNode,
        neural_probs: Optional[np.ndarray],
        features: Optional[Dict]
    ) -> float:
        """
        Simulation phase: Use Thompson Sampling to estimate value.

        This is where TS ALL THE WAY DOWN happens!

        Args:
            node: Node to simulate from
            neural_probs: Neural network probabilities (optional)
            features: Features for reward estimation

        Returns:
            Estimated value
        """
        # Sample from Thompson prior
        ts_sample = node.thompson_sample()

        # Blend with neural probabilities if available
        if neural_probs is not None:
            neural_value = neural_probs[node.tool_idx]
            # Weighted blend: 70% TS, 30% neural
            value = 0.7 * ts_sample + 0.3 * neural_value
        else:
            value = ts_sample

        # Add small reward bonus for exploration (encourages diversity)
        if node.visits == 0:
            value += 0.1

        # Feature-based reward shaping (optional)
        if features is not None:
            # Example: Bonus for motifs
            motifs = features.get("motifs", [])
            if len(motifs) > 0:
                value += 0.05 * min(len(motifs), 3)  # Cap bonus

        return value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagation phase: Update all nodes in path.

        Args:
            node: Leaf node to backpropagate from
            value: Value to propagate
        """
        while node is not None:
            node.update(value)
            node = node.parent

    def update_priors(self, tool_idx: int, reward: float) -> None:
        """
        Update Thompson Sampling priors based on actual outcome.

        Args:
            tool_idx: Tool that was used
            reward: Observed reward
        """
        if reward > 0:
            self.alpha_priors[tool_idx] += reward
        else:
            self.beta_priors[tool_idx] += abs(reward)

        logger.debug(f"Updated priors: tool={tool_idx}, reward={reward:.3f}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get flux capacitor statistics."""
        return {
            "total_simulations": self.total_simulations,
            "n_decisions": len(self.best_tool_history),
            "tool_distribution": np.bincount(self.best_tool_history, minlength=self.n_tools).tolist(),
            "alpha_priors": self.alpha_priors.tolist(),
            "beta_priors": self.beta_priors.tolist(),
            "thompson_priors": (self.alpha_priors / (self.alpha_priors + self.beta_priors)).tolist()
        }


# ============================================================================
# Enhanced Convergence Engine with MCTS
# ============================================================================

@dataclass
class MCTSCollapseResult:
    """Result of MCTS collapse."""
    tool: str
    tool_idx: int
    confidence: float
    strategy_used: str
    mcts_stats: Dict[str, Any]
    ucb1_scores: List[float]
    visit_counts: List[int]


class MCTSConvergenceEngine:
    """
    Convergence Engine with MCTS Flux Capacitor.

    Uses Monte Carlo Tree Search with Thompson Sampling to make
    decisions with full temporal lookahead.
    """

    def __init__(
        self,
        tools: List[str],
        n_simulations: int = 100,
        exploration_constant: float = 1.414
    ):
        """
        Initialize MCTS Convergence Engine.

        Args:
            tools: List of tool names
            n_simulations: MCTS simulations per decision
            exploration_constant: UCB1 exploration parameter
        """
        self.tools = tools
        self.n_tools = len(tools)

        # Create MCTS Flux Capacitor
        self.flux = MCTSFluxCapacitor(
            n_tools=self.n_tools,
            n_simulations=n_simulations,
            exploration_constant=exploration_constant
        )

        # Statistics
        self.decision_count = 0
        self.tool_usage = {tool: 0 for tool in tools}

        logger.info(f"MCTS ConvergenceEngine initialized: {self.n_tools} tools, {n_simulations} sims/decision")

    def collapse(
        self,
        neural_probs: Optional[np.ndarray] = None,
        features: Optional[Dict] = None
    ) -> MCTSCollapseResult:
        """
        Collapse continuous to discrete using MCTS.

        Args:
            neural_probs: Neural network probabilities (optional)
            features: Feature dict

        Returns:
            MCTSCollapseResult with decision and full trace
        """
        # Run MCTS search
        tool_idx, confidence, stats = self.flux.search(neural_probs, features)

        # Get tool name
        tool = self.tools[tool_idx]

        # Update statistics
        self.decision_count += 1
        self.tool_usage[tool] += 1

        # Create result
        result = MCTSCollapseResult(
            tool=tool,
            tool_idx=tool_idx,
            confidence=confidence,
            strategy_used=f"mcts_{self.flux.n_simulations}_sims",
            mcts_stats=stats,
            ucb1_scores=stats["ucb1_scores"],
            visit_counts=stats["visit_counts"]
        )

        logger.info(f"MCTS collapse: {tool} (confidence={confidence:.1%})")

        return result

    def update_from_outcome(self, tool_idx: int, reward: float) -> None:
        """
        Update MCTS priors from actual outcome.

        Args:
            tool_idx: Tool that was used
            reward: Observed reward (>0 for success)
        """
        self.flux.update_priors(tool_idx, reward)

    def get_statistics(self) -> Dict[str, Any]:
        """Get convergence engine statistics."""
        return {
            "decision_count": self.decision_count,
            "tool_usage": self.tool_usage,
            "flux_stats": self.flux.get_statistics()
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MCTS FLUX CAPACITOR DEMO")
    print("="*80)

    # Create engine
    tools = ["search", "summarize", "extract", "respond", "clarify"]
    engine = MCTSConvergenceEngine(
        tools=tools,
        n_simulations=50,
        exploration_constant=1.414
    )

    # Test decisions
    print("\nRunning 5 test decisions with MCTS...\n")

    for i in range(5):
        # Create mock neural probs
        neural_probs = np.random.dirichlet(np.ones(len(tools)))

        # Create mock features
        features = {"motifs": ["test_motif"]}

        # Collapse
        result = engine.collapse(neural_probs, features)

        print(f"Decision {i+1}:")
        print(f"  Tool: {result.tool}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Visit counts: {result.visit_counts}")
        print(f"  UCB1 scores: {[f'{s:.3f}' for s in result.ucb1_scores]}")
        print()

        # Simulate outcome
        reward = 1.0 if result.confidence > 0.3 else 0.5
        engine.update_from_outcome(result.tool_idx, reward)

    # Statistics
    print("="*80)
    print("STATISTICS")
    print("="*80)
    stats = engine.get_statistics()
    print(f"Total decisions: {stats['decision_count']}")
    print(f"Tool usage: {stats['tool_usage']}")
    print(f"Thompson priors: {[f'{p:.3f}' for p in stats['flux_stats']['thompson_priors']]}")
    print(f"Total MCTS simulations: {stats['flux_stats']['total_simulations']}")

    print("\nFlux Capacitor operational!")

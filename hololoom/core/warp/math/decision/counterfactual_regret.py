"""
Counterfactual Regret Minimization — Solving Extensive-Form Games
=================================================================

CFR finds Nash equilibria in sequential (extensive-form) games by
minimizing counterfactual regret at each information set.

Core idea: at each information set, track regret for each action
(how much better would it have been to always play that action?).
Use regret-matching to convert regrets into strategies. The average
strategy over all iterations converges to a Nash equilibrium.

Key insight: "counterfactual" value is the expected payoff if the
player *tries* to reach this information set (reach probability = 1
for this player, actual reach for opponents).

Variants:
- Vanilla CFR: O(1/sqrt(T)) convergence to Nash
- CFR+: Floor negative regrets at 0, faster in practice
- Linear CFR: Time-weighted averaging, best empirical convergence

Mathematical Foundation:
In a two-player zero-sum game, if both players use regret-minimizing
strategies, the average strategy profile converges to a Nash equilibrium.
Exploitability (distance to Nash) decreases as O(1/sqrt(T)).

Imports from regret_minimization.py: RegretStats

Applications to Warp Space:
- Multi-agent strategy learning (tool selection under competition)
- Adversarial robustness (find worst-case opponent strategies)
- Sequential decision-making with information asymmetry
- Resource allocation games between weaving components

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class InfoSet:
    """An information set in an extensive-form game.

    An information set groups game states that a player cannot
    distinguish (they have the same observation history).

    Attributes:
        player: Which player owns this info set (0 or 1).
        observations: Hashable observation tuple identifying this info set.
        legal_actions: List of action indices available here.
    """
    player: int
    observations: tuple
    legal_actions: list[int]


@dataclass
class CFRResult:
    """Result of CFR training.

    Attributes:
        strategy: Average strategy at each info set.
            Maps info set key (tuple) -> action probabilities (np.ndarray).
        exploitability: How far from Nash equilibrium (sum of both players'
            best-response improvements). 0 = perfect Nash.
        iterations: Number of CFR iterations completed.
    """
    strategy: dict[tuple, np.ndarray]
    exploitability: float
    iterations: int


@dataclass
class GameNode:
    """A node in the game tree.

    Attributes:
        player: Which player acts (-1 for terminal, -2 for chance).
        actions: List of action indices available.
        children: Dict mapping action -> child GameNode.
        payoffs: Terminal payoffs array (one per player). None if non-terminal.
        info_set_key: Hashable key for this node's information set.
        chance_probs: For chance nodes, probability of each action.
    """
    player: int = 0
    actions: list[int] = field(default_factory=list)
    children: dict[int, 'GameNode'] = field(default_factory=dict)
    payoffs: np.ndarray | None = None
    info_set_key: tuple | None = None
    chance_probs: np.ndarray | None = None

    @property
    def is_terminal(self) -> bool:
        return self.payoffs is not None

    @property
    def is_chance(self) -> bool:
        return self.player == -2

    @staticmethod
    def from_matrix(
        payoff_A: np.ndarray,
        payoff_B: np.ndarray | None = None,
    ) -> 'GameNode':
        """Convert a normal-form game to extensive form.

        For a 2-player game with payoff matrices A (player 0) and B (player 1):
        Player 0 chooses row, then player 1 chooses column (sequential).

        In zero-sum games, B = -A.

        Args:
            payoff_A: Payoff matrix for player 0, shape (n_actions_0, n_actions_1).
            payoff_B: Payoff matrix for player 1. If None, uses -payoff_A (zero-sum).

        Returns:
            Root GameNode of the extensive-form representation.
        """
        payoff_A = np.asarray(payoff_A)
        if payoff_B is None:
            payoff_B = -payoff_A
        else:
            payoff_B = np.asarray(payoff_B)

        n_rows, n_cols = payoff_A.shape

        # Root: player 0 chooses row
        root = GameNode(
            player=0,
            actions=list(range(n_rows)),
            info_set_key=(0,),  # Player 0 has one info set
        )

        for row in range(n_rows):
            # Player 1 chooses column (doesn't know row in simultaneous game)
            p1_node = GameNode(
                player=1,
                actions=list(range(n_cols)),
                info_set_key=(1,),  # Player 1 has one info set (simultaneous)
            )

            for col in range(n_cols):
                # Terminal node
                terminal = GameNode(
                    player=-1,
                    payoffs=np.array([payoff_A[row, col], payoff_B[row, col]]),
                )
                p1_node.children[col] = terminal

            root.children[row] = p1_node

        return root


# ============================================================================
# VANILLA CFR
# ============================================================================

class CounterfactualRegretMinimization:
    """Vanilla Counterfactual Regret Minimization.

    Iteratively computes counterfactual values and updates regrets
    at each information set. The average strategy converges to Nash.
    """

    def __init__(self):
        # cumulative_regret[info_set_key] = np.ndarray of regrets per action
        self._cumulative_regret: dict[tuple, np.ndarray] = {}
        # strategy_sum[info_set_key] = sum of strategies over iterations
        self._strategy_sum: dict[tuple, np.ndarray] = {}
        self._iterations: int = 0

    def _get_strategy(self, info_set_key: tuple, n_actions: int) -> np.ndarray:
        """Regret-matching: convert regrets to strategy.

        sigma(a) = max(R(a), 0) / sum_a' max(R(a'), 0)
        If all regrets <= 0, play uniform.
        """
        if info_set_key not in self._cumulative_regret:
            self._cumulative_regret[info_set_key] = np.zeros(n_actions)

        regrets = self._cumulative_regret[info_set_key]
        positive = np.maximum(regrets, 0)
        total = positive.sum()

        if total > 0:
            return positive / total
        else:
            return np.ones(n_actions) / n_actions

    def _cfr_recursive(
        self,
        node: GameNode,
        reach_probs: np.ndarray,
        n_players: int = 2,
    ) -> np.ndarray:
        """Recursive counterfactual value computation.

        Args:
            node: Current game tree node.
            reach_probs: Reach probabilities for each player, shape (n_players,).
            n_players: Number of players.

        Returns:
            Expected utility for each player, shape (n_players,).
        """
        if node.is_terminal:
            return node.payoffs.copy()

        if node.is_chance:
            # Average over chance outcomes
            ev = np.zeros(n_players)
            for action in node.actions:
                prob = node.chance_probs[action]
                child = node.children[action]
                ev += prob * self._cfr_recursive(child, reach_probs, n_players)
            return ev

        player = node.player
        n_actions = len(node.actions)
        strategy = self._get_strategy(node.info_set_key, n_actions)

        # Compute counterfactual values for each action
        action_values = np.zeros((n_actions, n_players))
        for i, action in enumerate(node.actions):
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[i]
            action_values[i] = self._cfr_recursive(
                node.children[action], new_reach, n_players
            )

        # Node value = strategy-weighted sum
        node_value = np.zeros(n_players)
        for i in range(n_actions):
            node_value += strategy[i] * action_values[i]

        # Counterfactual reach: product of other players' reach probs
        cf_reach = 1.0
        for p in range(n_players):
            if p != player:
                cf_reach *= reach_probs[p]

        # Update regrets
        if node.info_set_key not in self._cumulative_regret:
            self._cumulative_regret[node.info_set_key] = np.zeros(n_actions)

        for i in range(n_actions):
            regret = action_values[i, player] - node_value[player]
            self._cumulative_regret[node.info_set_key][i] += cf_reach * regret

        # Accumulate strategy (weighted by player's reach)
        if node.info_set_key not in self._strategy_sum:
            self._strategy_sum[node.info_set_key] = np.zeros(n_actions)

        self._strategy_sum[node.info_set_key] += reach_probs[player] * strategy

        return node_value

    def train(
        self,
        root: GameNode,
        n_iterations: int = 1000,
        n_players: int = 2,
    ) -> CFRResult:
        """Run vanilla CFR training.

        Args:
            root: Root of the game tree.
            n_iterations: Number of CFR iterations.
            n_players: Number of players.

        Returns:
            CFRResult with average strategy and exploitability.
        """
        self._cumulative_regret.clear()
        self._strategy_sum.clear()

        for i in range(n_iterations):
            reach_probs = np.ones(n_players)
            self._cfr_recursive(root, reach_probs, n_players)
            self._iterations = i + 1

        # Extract average strategy
        strategy = {}
        for key, s_sum in self._strategy_sum.items():
            total = s_sum.sum()
            if total > 0:
                strategy[key] = s_sum / total
            else:
                strategy[key] = np.ones(len(s_sum)) / len(s_sum)

        # Compute exploitability
        exploit = self.exploitability(root, strategy, n_players)

        return CFRResult(
            strategy=strategy,
            exploitability=exploit,
            iterations=self._iterations,
        )

    def get_strategy(self, info_set_key: tuple) -> np.ndarray:
        """Get the average strategy at an information set.

        Args:
            info_set_key: Information set key.

        Returns:
            Action probability distribution.
        """
        if info_set_key not in self._strategy_sum:
            raise KeyError(f"Info set {info_set_key} not found")

        s_sum = self._strategy_sum[info_set_key]
        total = s_sum.sum()
        if total > 0:
            return s_sum / total
        return np.ones(len(s_sum)) / len(s_sum)

    @staticmethod
    def exploitability(
        root: GameNode,
        strategy: dict[tuple, np.ndarray],
        n_players: int = 2,
    ) -> float:
        """Compute exploitability of a strategy profile.

        Exploitability = sum over players of (best_response_value - strategy_value).
        A Nash equilibrium has exploitability 0.

        Args:
            root: Root of game tree.
            strategy: Strategy profile (info_set_key -> action probs).
            n_players: Number of players.

        Returns:
            Exploitability (>= 0, 0 means Nash equilibrium).
        """
        total_exploit = 0.0

        for exploiting_player in range(n_players):
            # Value when everyone plays strategy
            strategy_value = CounterfactualRegretMinimization._tree_value(
                root, strategy, n_players
            )[exploiting_player]

            # Best response value for exploiting_player
            br_value = CounterfactualRegretMinimization._best_response_value(
                root, strategy, exploiting_player, n_players
            )

            total_exploit += max(0.0, br_value - strategy_value)

        return total_exploit

    @staticmethod
    def _tree_value(
        node: GameNode,
        strategy: dict[tuple, np.ndarray],
        n_players: int,
    ) -> np.ndarray:
        """Compute expected payoff under given strategy profile."""
        if node.is_terminal:
            return node.payoffs.copy()

        if node.is_chance:
            ev = np.zeros(n_players)
            for action in node.actions:
                ev += node.chance_probs[action] * CounterfactualRegretMinimization._tree_value(
                    node.children[action], strategy, n_players
                )
            return ev

        n_actions = len(node.actions)
        if node.info_set_key in strategy:
            sigma = strategy[node.info_set_key]
        else:
            sigma = np.ones(n_actions) / n_actions

        ev = np.zeros(n_players)
        for i, action in enumerate(node.actions):
            ev += sigma[i] * CounterfactualRegretMinimization._tree_value(
                node.children[action], strategy, n_players
            )
        return ev

    @staticmethod
    def _best_response_value(
        node: GameNode,
        strategy: dict[tuple, np.ndarray],
        exploiting_player: int,
        n_players: int,
    ) -> float:
        """Compute the best-response value for one player against fixed opponents."""
        if node.is_terminal:
            return float(node.payoffs[exploiting_player])

        if node.is_chance:
            val = 0.0
            for action in node.actions:
                val += node.chance_probs[action] * CounterfactualRegretMinimization._best_response_value(
                    node.children[action], strategy, exploiting_player, n_players
                )
            return val

        n_actions = len(node.actions)

        if node.player == exploiting_player:
            # Best response: pick action maximizing value
            best = -np.inf
            for action in node.actions:
                val = CounterfactualRegretMinimization._best_response_value(
                    node.children[action], strategy, exploiting_player, n_players
                )
                best = max(best, val)
            return best
        else:
            # Other player plays their strategy
            if node.info_set_key in strategy:
                sigma = strategy[node.info_set_key]
            else:
                sigma = np.ones(n_actions) / n_actions

            val = 0.0
            for i, action in enumerate(node.actions):
                val += sigma[i] * CounterfactualRegretMinimization._best_response_value(
                    node.children[action], strategy, exploiting_player, n_players
                )
            return val


# ============================================================================
# CFR+ (Regret Matching+)
# ============================================================================

class CFRPlus:
    """CFR+ variant with regret matching+.

    Differences from vanilla CFR:
    1. Floor cumulative regrets at 0 after each iteration
    2. Use current strategy weighted by iteration number (linear weighting)

    Empirically converges faster, especially in large games.
    """

    def __init__(self):
        self._cumulative_regret: dict[tuple, np.ndarray] = {}
        self._strategy_sum: dict[tuple, np.ndarray] = {}
        self._iterations: int = 0

    def _get_strategy(self, info_set_key: tuple, n_actions: int) -> np.ndarray:
        """Regret matching (same as vanilla)."""
        if info_set_key not in self._cumulative_regret:
            self._cumulative_regret[info_set_key] = np.zeros(n_actions)

        regrets = self._cumulative_regret[info_set_key]
        positive = np.maximum(regrets, 0)
        total = positive.sum()

        if total > 0:
            return positive / total
        return np.ones(n_actions) / n_actions

    def _cfr_plus_recursive(
        self,
        node: GameNode,
        reach_probs: np.ndarray,
        iteration: int,
        n_players: int = 2,
    ) -> np.ndarray:
        """CFR+ recursive traversal."""
        if node.is_terminal:
            return node.payoffs.copy()

        if node.is_chance:
            ev = np.zeros(n_players)
            for action in node.actions:
                ev += node.chance_probs[action] * self._cfr_plus_recursive(
                    node.children[action], reach_probs, iteration, n_players
                )
            return ev

        player = node.player
        n_actions = len(node.actions)
        strategy = self._get_strategy(node.info_set_key, n_actions)

        action_values = np.zeros((n_actions, n_players))
        for i, action in enumerate(node.actions):
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[i]
            action_values[i] = self._cfr_plus_recursive(
                node.children[action], new_reach, iteration, n_players
            )

        node_value = np.zeros(n_players)
        for i in range(n_actions):
            node_value += strategy[i] * action_values[i]

        cf_reach = 1.0
        for p in range(n_players):
            if p != player:
                cf_reach *= reach_probs[p]

        if node.info_set_key not in self._cumulative_regret:
            self._cumulative_regret[node.info_set_key] = np.zeros(n_actions)

        for i in range(n_actions):
            regret = action_values[i, player] - node_value[player]
            self._cumulative_regret[node.info_set_key][i] += cf_reach * regret

        # CFR+ key difference: floor regrets at 0
        self._cumulative_regret[node.info_set_key] = np.maximum(
            self._cumulative_regret[node.info_set_key], 0
        )

        # Linear weighting for strategy averaging
        if node.info_set_key not in self._strategy_sum:
            self._strategy_sum[node.info_set_key] = np.zeros(n_actions)

        weight = max(iteration, 1)
        self._strategy_sum[node.info_set_key] += weight * reach_probs[player] * strategy

        return node_value

    def train(
        self,
        root: GameNode,
        n_iterations: int = 1000,
        n_players: int = 2,
    ) -> CFRResult:
        """Run CFR+ training.

        Args:
            root: Root of the game tree.
            n_iterations: Number of iterations.
            n_players: Number of players.

        Returns:
            CFRResult with average strategy and exploitability.
        """
        self._cumulative_regret.clear()
        self._strategy_sum.clear()

        for i in range(n_iterations):
            reach_probs = np.ones(n_players)
            self._cfr_plus_recursive(root, reach_probs, i + 1, n_players)
            self._iterations = i + 1

        strategy = {}
        for key, s_sum in self._strategy_sum.items():
            total = s_sum.sum()
            if total > 0:
                strategy[key] = s_sum / total
            else:
                strategy[key] = np.ones(len(s_sum)) / len(s_sum)

        exploit = CounterfactualRegretMinimization.exploitability(
            root, strategy, n_players
        )

        return CFRResult(
            strategy=strategy,
            exploitability=exploit,
            iterations=self._iterations,
        )


# ============================================================================
# LINEAR CFR
# ============================================================================

class LinearCFR:
    """Linear CFR: time-weighted regret and strategy averaging.

    Both regrets and strategies are weighted linearly by iteration number.
    This gives more weight to recent iterations when the strategy is
    better, leading to faster convergence in practice.
    """

    def __init__(self):
        self._cumulative_regret: dict[tuple, np.ndarray] = {}
        self._strategy_sum: dict[tuple, np.ndarray] = {}
        self._iterations: int = 0

    def _get_strategy(self, info_set_key: tuple, n_actions: int) -> np.ndarray:
        """Regret matching."""
        if info_set_key not in self._cumulative_regret:
            self._cumulative_regret[info_set_key] = np.zeros(n_actions)

        regrets = self._cumulative_regret[info_set_key]
        positive = np.maximum(regrets, 0)
        total = positive.sum()

        if total > 0:
            return positive / total
        return np.ones(n_actions) / n_actions

    def _linear_cfr_recursive(
        self,
        node: GameNode,
        reach_probs: np.ndarray,
        iteration: int,
        n_players: int = 2,
    ) -> np.ndarray:
        """Linear CFR recursive traversal."""
        if node.is_terminal:
            return node.payoffs.copy()

        if node.is_chance:
            ev = np.zeros(n_players)
            for action in node.actions:
                ev += node.chance_probs[action] * self._linear_cfr_recursive(
                    node.children[action], reach_probs, iteration, n_players
                )
            return ev

        player = node.player
        n_actions = len(node.actions)
        strategy = self._get_strategy(node.info_set_key, n_actions)

        action_values = np.zeros((n_actions, n_players))
        for i, action in enumerate(node.actions):
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[i]
            action_values[i] = self._linear_cfr_recursive(
                node.children[action], new_reach, iteration, n_players
            )

        node_value = np.zeros(n_players)
        for i in range(n_actions):
            node_value += strategy[i] * action_values[i]

        cf_reach = 1.0
        for p in range(n_players):
            if p != player:
                cf_reach *= reach_probs[p]

        if node.info_set_key not in self._cumulative_regret:
            self._cumulative_regret[node.info_set_key] = np.zeros(n_actions)

        # Linear CFR: weight regrets by iteration number
        weight = max(iteration, 1)
        for i in range(n_actions):
            regret = action_values[i, player] - node_value[player]
            self._cumulative_regret[node.info_set_key][i] += weight * cf_reach * regret

        # Floor at 0 (like CFR+)
        self._cumulative_regret[node.info_set_key] = np.maximum(
            self._cumulative_regret[node.info_set_key], 0
        )

        # Linear weighting for strategy
        if node.info_set_key not in self._strategy_sum:
            self._strategy_sum[node.info_set_key] = np.zeros(n_actions)

        self._strategy_sum[node.info_set_key] += weight * reach_probs[player] * strategy

        return node_value

    def train(
        self,
        root: GameNode,
        n_iterations: int = 1000,
        n_players: int = 2,
    ) -> CFRResult:
        """Run Linear CFR training.

        Args:
            root: Root of the game tree.
            n_iterations: Number of iterations.
            n_players: Number of players.

        Returns:
            CFRResult with average strategy and exploitability.
        """
        self._cumulative_regret.clear()
        self._strategy_sum.clear()

        for i in range(n_iterations):
            reach_probs = np.ones(n_players)
            self._linear_cfr_recursive(root, reach_probs, i + 1, n_players)
            self._iterations = i + 1

        strategy = {}
        for key, s_sum in self._strategy_sum.items():
            total = s_sum.sum()
            if total > 0:
                strategy[key] = s_sum / total
            else:
                strategy[key] = np.ones(len(s_sum)) / len(s_sum)

        exploit = CounterfactualRegretMinimization.exploitability(
            root, strategy, n_players
        )

        return CFRResult(
            strategy=strategy,
            exploitability=exploit,
            iterations=self._iterations,
        )


if __name__ == "__main__":
    print("Counterfactual Regret Minimization Module Demo")
    print("=" * 50)

    # Rock-Paper-Scissors
    # Payoff matrix for player 0 (zero-sum)
    rps = np.array([
        [0, -1, 1],   # Rock vs Rock, Paper, Scissors
        [1, 0, -1],   # Paper vs ...
        [-1, 1, 0],   # Scissors vs ...
    ])

    root = GameNode.from_matrix(rps)

    # Vanilla CFR
    cfr = CounterfactualRegretMinimization()
    result = cfr.train(root, n_iterations=1000)
    print(f"Vanilla CFR ({result.iterations} iters):")
    for key, strat in result.strategy.items():
        print(f"  Info set {key}: {strat.round(3)}")
    print(f"  Exploitability: {result.exploitability:.6f}")

    # CFR+
    cfr_plus = CFRPlus()
    result_plus = cfr_plus.train(root, n_iterations=1000)
    print(f"\nCFR+ ({result_plus.iterations} iters):")
    for key, strat in result_plus.strategy.items():
        print(f"  Info set {key}: {strat.round(3)}")
    print(f"  Exploitability: {result_plus.exploitability:.6f}")

    # Linear CFR
    lcfr = LinearCFR()
    result_lcfr = lcfr.train(root, n_iterations=1000)
    print(f"\nLinear CFR ({result_lcfr.iterations} iters):")
    for key, strat in result_lcfr.strategy.items():
        print(f"  Info set {key}: {strat.round(3)}")
    print(f"  Exploitability: {result_lcfr.exploitability:.6f}")

    # Expected: all converge to (1/3, 1/3, 1/3) for both players
    print("\nExpected: uniform (1/3, 1/3, 1/3) for both players (Nash of RPS)")

    print("\n" + "=" * 50)
    print("Counterfactual regret minimization module ready!")

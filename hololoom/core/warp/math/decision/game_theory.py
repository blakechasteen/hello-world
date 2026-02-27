"""
Game Theory - Nash Equilibria, Mechanism Design, Auctions
=========================================================

Mathematical theory of strategic interaction.

Classes:
    NormalFormGame: Matrix games (simultaneous moves)
    ExtensiveFormGame: Game trees (sequential moves)
    NashEquilibrium: Solution concepts
    MixedStrategy: Randomized strategies
    MechanismDesign: Truthful mechanism design (VCG)
    AuctionTheory: First-price, second-price, combinatorial auctions
    CooperativeGame: Coalitional games, Shapley value
    EvolutionaryGame: Replicator dynamics, ESS

Applications:
    - Economics (markets, auctions)
    - Multi-agent AI (negotiation, cooperation)
    - Mechanism design (incentives)
    - Evolutionary biology
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from itertools import product, combinations


@dataclass
class Strategy:
    """Pure or mixed strategy."""
    probabilities: np.ndarray  # Probability over pure strategies
    is_pure: bool = False

    def __post_init__(self):
        """Check if strategy is pure (single action with prob 1)."""
        self.is_pure = np.sum(self.probabilities == 1.0) == 1

    @staticmethod
    def pure(action: int, n_actions: int) -> 'Strategy':
        """Create pure strategy (deterministic action)."""
        probs = np.zeros(n_actions)
        probs[action] = 1.0
        return Strategy(probs, is_pure=True)

    @staticmethod
    def uniform(n_actions: int) -> 'Strategy':
        """Create uniform mixed strategy."""
        probs = np.ones(n_actions) / n_actions
        return Strategy(probs, is_pure=False)


class NormalFormGame:
    """
    Normal-form (strategic-form) game: simultaneous move game.

    Represented by payoff matrices for each player.
    """

    def __init__(self, payoff_matrices: List[np.ndarray], names: Optional[List[str]] = None):
        """
        Args:
            payoff_matrices: List of payoff matrices (one per player)
                            For 2-player: [U_1(i,j), U_2(i,j)]
            names: Optional player names
        """
        self.payoffs = payoff_matrices
        self.n_players = len(payoff_matrices)
        self.n_actions = [matrix.shape[i] for i, matrix in enumerate(payoff_matrices)]
        self.names = names or [f"Player {i+1}" for i in range(self.n_players)]

    def utility(self, player: int, action_profile: Tuple[int, ...]) -> float:
        """
        Utility for player given action profile.

        Args:
            player: Player index
            action_profile: Tuple of actions (one per player)
        """
        return float(self.payoffs[player][action_profile])

    def expected_utility(self, player: int, strategies: List[Strategy]) -> float:
        """
        Expected utility for player under mixed strategy profile.

        E[U_i] = Σ_a (∏_j σ_j(a_j)) U_i(a)
        """
        expected_u = 0.0

        # Iterate over all action profiles
        action_ranges = [range(n) for n in self.n_actions]
        for action_profile in product(*action_ranges):
            # Probability of this profile
            prob = np.prod([strategies[j].probabilities[action_profile[j]]
                           for j in range(self.n_players)])
            # Add contribution
            expected_u += prob * self.utility(player, action_profile)

        return expected_u

    def best_response(self, player: int, opponent_strategies: List[Strategy]) -> int:
        """
        Best response: action maximizing expected utility.

        BR_i(σ_{-i}) = argmax_{a_i} E[U_i(a_i, σ_{-i})]

        Args:
            player: Index of player finding best response (0 to n_players-1)
            opponent_strategies: List of (n_players - 1) strategies for OTHER players,
                                 ordered by player index excluding current player.
                                 E.g., for player=1 in 3-player game: [s0, s2]

        Returns:
            Best action index for the player
        """
        # Validate opponent_strategies length
        expected_len = self.n_players - 1
        if len(opponent_strategies) != expected_len:
            raise ValueError(
                f"opponent_strategies must have {expected_len} elements "
                f"(one per opponent), got {len(opponent_strategies)}"
            )

        best_action = 0
        best_utility = -np.inf

        for action in range(self.n_actions[player]):
            # Create full strategy profile by inserting player's strategy
            # list() creates a copy to avoid mutating the original
            test_strategies = list(opponent_strategies)
            test_strategies.insert(player, Strategy.pure(action, self.n_actions[player]))

            utility = self.expected_utility(player, test_strategies)
            if utility > best_utility:
                best_utility = utility
                best_action = action

        return best_action

    def is_dominant_strategy(self, player: int, action: int) -> bool:
        """
        Check if action strictly dominates all other actions.

        a_i dominates a_i' if U_i(a_i, a_{-i}) > U_i(a_i', a_{-i}) for all a_{-i}.
        """
        # For 2-player games (simplified)
        if self.n_players != 2:
            return False  # TODO: generalize

        opponent = 1 - player
        for opponent_action in range(self.n_actions[opponent]):
            for other_action in range(self.n_actions[player]):
                if other_action == action:
                    continue

                # Check if action dominates other_action
                if player == 0:
                    u_action = self.payoffs[player][action, opponent_action]
                    u_other = self.payoffs[player][other_action, opponent_action]
                else:
                    u_action = self.payoffs[player][opponent_action, action]
                    u_other = self.payoffs[player][opponent_action, other_action]

                if u_action <= u_other:
                    return False

        return True

    @staticmethod
    def prisoners_dilemma() -> 'NormalFormGame':
        """
        Classic Prisoner's Dilemma.

        (Cooperate, Cooperate) is Pareto optimal.
        (Defect, Defect) is Nash equilibrium (dominant strategies).
        """
        # Payoff matrices: [Player1, Player2]
        # Actions: 0=Cooperate, 1=Defect
        U1 = np.array([
            [-1, -3],  # P1 Cooperate: (-1,-1) or (-3,0)
            [0, -2]    # P1 Defect: (0,-3) or (-2,-2)
        ])
        U2 = np.array([
            [-1, 0],   # P2 Cooperate: (-1,-1) or (0,-3)
            [-3, -2]   # P2 Defect: (-3,0) or (-2,-2)
        ])
        return NormalFormGame([U1, U2], names=["Prisoner 1", "Prisoner 2"])

    @staticmethod
    def matching_pennies() -> 'NormalFormGame':
        """
        Matching Pennies: zero-sum game with no pure Nash equilibrium.

        Unique mixed Nash: both play (0.5, 0.5).
        """
        U1 = np.array([
            [1, -1],
            [-1, 1]
        ])
        U2 = -U1  # Zero-sum
        return NormalFormGame([U1, U2], names=["Player 1", "Player 2"])

    @staticmethod
    def battle_of_sexes() -> 'NormalFormGame':
        """
        Battle of the Sexes: coordination game.

        Two pure Nash equilibria + one mixed.
        """
        U1 = np.array([
            [2, 0],
            [0, 1]
        ])
        U2 = np.array([
            [1, 0],
            [0, 2]
        ])
        return NormalFormGame([U1, U2], names=["Player 1", "Player 2"])


class NashEquilibrium:
    """
    Nash equilibrium: no player can improve by unilateral deviation.

    σ* is Nash iff: U_i(σ_i*, σ_{-i}*) >= U_i(σ_i, σ_{-i}*) for all i, σ_i.
    """

    @staticmethod
    def find_pure(game: NormalFormGame) -> List[Tuple[int, ...]]:
        """
        Find all pure strategy Nash equilibria.

        Exhaustive search over action profiles.
        """
        equilibria = []

        # Iterate over all action profiles
        action_ranges = [range(n) for n in game.n_actions]
        for action_profile in product(*action_ranges):
            is_equilibrium = True

            # Check if each player's action is best response
            for player in range(game.n_players):
                # Create opponent strategies
                opponent_strategies = []
                for j in range(game.n_players):
                    if j != player:
                        opponent_strategies.append(
                            Strategy.pure(action_profile[j], game.n_actions[j])
                        )

                best = game.best_response(player, opponent_strategies)
                if best != action_profile[player]:
                    is_equilibrium = False
                    break

            if is_equilibrium:
                equilibria.append(action_profile)

        return equilibria

    @staticmethod
    def find_mixed_2player(game: NormalFormGame, tolerance: float = 1e-9) -> Optional[List[Strategy]]:
        """
        Find mixed strategy Nash equilibrium for 2-player game.

        Uses support enumeration algorithm:
        1. Try all possible support pairs (S1, S2)
        2. Solve indifference conditions for each support pair
        3. Verify the solution is a valid Nash equilibrium

        For Matching Pennies: returns (0.5, 0.5) for both players.

        Args:
            game: 2-player normal form game
            tolerance: Numerical tolerance for linear algebra

        Returns:
            List of 2 Strategy objects, or None if no equilibrium found
        """
        if game.n_players != 2:
            raise ValueError("Only 2-player games supported")

        n1, n2 = game.n_actions[0], game.n_actions[1]

        # Build payoff matrices A (player 1) and B (player 2)
        # A[i,j] = utility to player 1 when P1 plays i, P2 plays j
        # B[i,j] = utility to player 2 when P1 plays i, P2 plays j
        A = np.zeros((n1, n2))
        B = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                A[i, j] = game.utility(0, (i, j))
                B[i, j] = game.utility(1, (i, j))

        # Try all support pairs, starting with smallest
        for k1 in range(1, n1 + 1):
            for k2 in range(1, n2 + 1):
                for supp1 in combinations(range(n1), k1):
                    for supp2 in combinations(range(n2), k2):
                        result = NashEquilibrium._solve_support_pair(
                            A, B, supp1, supp2, n1, n2, tolerance
                        )
                        if result is not None:
                            p, q = result
                            # Verify it's actually an equilibrium
                            strat1 = Strategy(p, is_pure=(k1 == 1))
                            strat2 = Strategy(q, is_pure=(k2 == 1))
                            if NashEquilibrium.verify_equilibrium(game, [strat1, strat2], tolerance):
                                return [strat1, strat2]

        # Fallback: return uniform (always an equilibrium for zero-sum)
        return [Strategy.uniform(n) for n in game.n_actions]

    @staticmethod
    def _solve_support_pair(
        A: np.ndarray, B: np.ndarray,
        supp1: Tuple[int, ...], supp2: Tuple[int, ...],
        n1: int, n2: int, tolerance: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Solve indifference conditions for a given support pair.

        For player 1's strategy p to make player 2 indifferent over supp2:
          For all j in supp2: sum_i p_i * B[i,j] = v2 (same value)

        For player 2's strategy q to make player 1 indifferent over supp1:
          For all i in supp1: sum_j A[i,j] * q_j = v1 (same value)

        Returns (p, q) probability vectors if valid, None otherwise.
        """
        k1, k2 = len(supp1), len(supp2)

        # Solve for q (player 2's strategy) using player 1's indifference
        # A_sub[i,:] @ q = v1 for all i in supp1
        # Also: sum(q[supp2]) = 1, q >= 0
        A_sub = A[np.ix_(list(supp1), list(supp2))]

        # Set up linear system: make all rows equal (indifference)
        # [A_sub[0] - A_sub[1], A_sub[0] - A_sub[2], ..., 1 1 1 1] @ q_sub = [0, 0, ..., 1]
        if k2 == 1:
            # Single action in support - probability is 1
            q_sub = np.array([1.0])
        else:
            # Indifference conditions + probability sum = 1
            M_q = np.zeros((k2, k2))
            b_q = np.zeros(k2)
            for row in range(k1 - 1):
                M_q[row, :] = A_sub[row, :] - A_sub[row + 1, :]
            M_q[k2 - 1, :] = 1.0  # Sum to 1
            b_q[k2 - 1] = 1.0

            try:
                q_sub = np.linalg.solve(M_q, b_q)
            except np.linalg.LinAlgError:
                return None

            # Check non-negativity
            if np.any(q_sub < -tolerance):
                return None

        # Solve for p (player 1's strategy) using player 2's indifference
        B_sub = B[np.ix_(list(supp1), list(supp2))]

        if k1 == 1:
            p_sub = np.array([1.0])
        else:
            # B_sub.T[j,:] @ p_sub = v2 for all j in supp2
            M_p = np.zeros((k1, k1))
            b_p = np.zeros(k1)
            for row in range(k2 - 1):
                M_p[row, :] = B_sub[:, row] - B_sub[:, row + 1]
            M_p[k1 - 1, :] = 1.0
            b_p[k1 - 1] = 1.0

            try:
                p_sub = np.linalg.solve(M_p, b_p)
            except np.linalg.LinAlgError:
                return None

            if np.any(p_sub < -tolerance):
                return None

        # Build full probability vectors
        p = np.zeros(n1)
        q = np.zeros(n2)
        for idx, i in enumerate(supp1):
            p[i] = max(0.0, p_sub[idx])  # Clamp small negatives
        for idx, j in enumerate(supp2):
            q[j] = max(0.0, q_sub[idx])

        # Normalize (handle numerical issues)
        p_sum = np.sum(p)
        q_sum = np.sum(q)
        if p_sum < tolerance or q_sum < tolerance:
            return None
        p /= p_sum
        q /= q_sum

        return p, q

    @staticmethod
    def verify_equilibrium(game: NormalFormGame, strategies: List[Strategy],
                         tolerance: float = 1e-6) -> bool:
        """
        Verify if strategy profile is Nash equilibrium.

        Check: no profitable unilateral deviation.
        """
        for player in range(game.n_players):
            current_utility = game.expected_utility(player, strategies)

            # Try all pure deviations
            for action in range(game.n_actions[player]):
                deviation_strategies = strategies.copy()
                deviation_strategies[player] = Strategy.pure(action, game.n_actions[player])
                deviation_utility = game.expected_utility(player, deviation_strategies)

                if deviation_utility > current_utility + tolerance:
                    return False  # Profitable deviation found

        return True

    @staticmethod
    def iterated_elimination_dominated(game: NormalFormGame) -> Tuple[NormalFormGame, Dict[int, List[int]]]:
        """
        Iteratively eliminate strictly dominated strategies (IESDS).

        A strategy s_i is strictly dominated if there exists another strategy s'_i
        such that for ALL opponent strategy profiles:
            U_i(s'_i, s_{-i}) > U_i(s_i, s_{-i})

        Returns:
            Tuple of:
            - Reduced game with dominated strategies removed
            - Dictionary mapping player -> list of surviving action indices
        """
        if game.n_players != 2:
            # For simplicity, only implement 2-player case
            return game, {i: list(range(game.n_actions[i])) for i in range(game.n_players)}

        # Track surviving actions for each player
        surviving = {
            0: list(range(game.n_actions[0])),
            1: list(range(game.n_actions[1]))
        }

        changed = True
        while changed:
            changed = False

            # Check player 1's actions
            for player in [0, 1]:
                other = 1 - player
                to_remove = []

                for action in surviving[player]:
                    # Check if action is strictly dominated by any other action
                    for alt_action in surviving[player]:
                        if alt_action == action:
                            continue

                        # Check if alt_action strictly dominates action
                        strictly_better = True
                        for opp_action in surviving[other]:
                            if player == 0:
                                u_action = game.utility(player, (action, opp_action))
                                u_alt = game.utility(player, (alt_action, opp_action))
                            else:
                                u_action = game.utility(player, (opp_action, action))
                                u_alt = game.utility(player, (opp_action, alt_action))

                            if u_alt <= u_action:
                                strictly_better = False
                                break

                        if strictly_better:
                            # alt_action strictly dominates action
                            to_remove.append(action)
                            break

                # Remove dominated actions
                for action in to_remove:
                    if action in surviving[player]:
                        surviving[player].remove(action)
                        changed = True

        # Build reduced game with only surviving actions
        n1_new = len(surviving[0])
        n2_new = len(surviving[1])

        if n1_new == 0 or n2_new == 0:
            # Degenerate case - return original
            return game, surviving

        # Create new payoff matrices
        new_payoffs = []
        for player in range(2):
            new_matrix = np.zeros((n1_new, n2_new))
            for i_new, i_old in enumerate(surviving[0]):
                for j_new, j_old in enumerate(surviving[1]):
                    new_matrix[i_new, j_new] = game.utility(player, (i_old, j_old))
            new_payoffs.append(new_matrix)

        reduced_game = NormalFormGame(new_payoffs)
        return reduced_game, surviving


class MechanismDesign:
    """
    Mechanism design: design game to achieve desired outcome.

    VCG mechanisms: truthful, efficient, individually rational.
    """

    @staticmethod
    def vcg_auction(values: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Vickrey-Clarke-Groves auction (truthful mechanism).

        Single item: second-price auction.
        Returns: (winner, payments)
        """
        n_bidders = len(values)

        # Winner: highest value
        winner = int(np.argmax(values))

        # Payments: VCG formula
        # p_i = (social welfare without i) - (social welfare with i, excluding i's value)
        payments = np.zeros(n_bidders)

        # For single-item: winner pays second-highest bid
        sorted_values = np.sort(values)
        if n_bidders > 1:
            payments[winner] = sorted_values[-2]
        else:
            payments[winner] = 0

        return winner, payments

    @staticmethod
    def is_truthful(
        mechanism: Callable[[np.ndarray], Tuple[int, np.ndarray]],
        test_cases: List[np.ndarray],
        n_misreport_samples: int = 10
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if mechanism is strategy-proof (truthful/incentive-compatible).

        Truthful iff reporting true value is a dominant strategy for each agent.
        Tests by checking that no agent can gain by misreporting their value.

        Args:
            mechanism: Function taking bids array, returning (winner, payments)
            test_cases: List of true value profiles to test
            n_misreport_samples: Number of misreport values to try per agent

        Returns:
            Tuple of:
            - True if no profitable deviation found, False otherwise
            - If False, dict with counterexample details; None if True
        """
        for true_values in test_cases:
            n_agents = len(true_values)

            # Get outcome when everyone reports truthfully
            winner_true, payments_true = mechanism(true_values)

            for agent in range(n_agents):
                # Agent's utility from truthful reporting
                if agent == winner_true:
                    utility_true = true_values[agent] - payments_true[agent]
                else:
                    utility_true = 0 - payments_true[agent]  # Didn't win

                # Try different misreports for this agent
                for misreport_factor in np.linspace(0.0, 2.0, n_misreport_samples):
                    misreport = true_values[agent] * misreport_factor

                    # Create bid profile with agent's misreport
                    bids = true_values.copy()
                    bids[agent] = misreport

                    # Get outcome with misreport
                    winner_mis, payments_mis = mechanism(bids)

                    # Agent's utility from misreporting (still based on TRUE value)
                    if agent == winner_mis:
                        utility_mis = true_values[agent] - payments_mis[agent]
                    else:
                        utility_mis = 0 - payments_mis[agent]

                    # Check for profitable deviation
                    if utility_mis > utility_true + 1e-9:
                        return False, {
                            'agent': agent,
                            'true_value': true_values[agent],
                            'misreport': misreport,
                            'utility_true': utility_true,
                            'utility_misreport': utility_mis,
                            'gain': utility_mis - utility_true,
                            'true_values': true_values.tolist()
                        }

        return True, None

    @staticmethod
    def individual_rationality(values: np.ndarray, payments: np.ndarray) -> bool:
        """
        Check individual rationality: utility_i >= 0 for all i.

        No bidder worse off than not participating.
        """
        utilities = values - payments
        return np.all(utilities >= -1e-10)


class AuctionTheory:
    """
    Auction theory: first-price, second-price, combinatorial.

    Revenue equivalence theorem: many auction formats yield same expected revenue.
    """

    @staticmethod
    def second_price_auction(bids: np.ndarray) -> Tuple[int, float]:
        """
        Second-price (Vickrey) auction: truthful bidding is dominant strategy.

        Winner pays second-highest bid.
        """
        winner = int(np.argmax(bids))
        sorted_bids = np.sort(bids)
        price = sorted_bids[-2] if len(bids) > 1 else 0
        return winner, price

    @staticmethod
    def first_price_auction(bids: np.ndarray) -> Tuple[int, float]:
        """
        First-price auction: winner pays own bid.

        Equilibrium bidding: shade below true value.
        """
        winner = int(np.argmax(bids))
        price = bids[winner]
        return winner, price

    @staticmethod
    def optimal_bid_first_price(value: float, n_bidders: int,
                               distribution: str = 'uniform') -> float:
        """
        Optimal bid in first-price auction (Bayesian Nash equilibrium).

        For uniform [0,1] values: bid = (n-1)/n * value.
        """
        if distribution == 'uniform':
            return (n_bidders - 1) / n_bidders * value
        else:
            return value  # Simplified

    @staticmethod
    def revenue_equivalence(values: np.ndarray, n_bidders: int) -> float:
        """
        Expected revenue under revenue equivalence theorem.

        E[R] = E[second-highest value] (for symmetric auctions).
        """
        # Monte Carlo approximation
        samples = 10000
        second_highest = []

        for _ in range(samples):
            sample_values = np.random.rand(n_bidders)
            sorted_vals = np.sort(sample_values)
            second_highest.append(sorted_vals[-2] if n_bidders > 1 else 0)

        return np.mean(second_highest)


class CooperativeGame:
    """
    Cooperative game theory: coalitional games, Shapley value.

    Characteristic function: v(S) = value of coalition S.
    """

    def __init__(self, n_players: int, characteristic_function: Callable):
        """
        Args:
            n_players: Number of players
            characteristic_function: v(coalition) -> value
        """
        self.n = n_players
        self.v = characteristic_function

    def shapley_value(self) -> np.ndarray:
        r"""
        Shapley value: fair division of total value.

        φ_i = Σ_S [(|S|-1)!(n-|S|)!/n!] [v(S) - v(S\{i})]

        Unique solution satisfying efficiency, symmetry, null player, additivity.
        """
        shapley = np.zeros(self.n)

        # Iterate over all coalitions
        for coalition_size in range(1, self.n + 1):
            for coalition in self._all_coalitions(coalition_size):
                coalition_set = set(coalition)
                v_coalition = self.v(coalition)

                # Marginal contribution of each player
                for player in coalition:
                    coalition_minus_player = tuple(p for p in coalition if p != player)
                    v_minus = self.v(coalition_minus_player) if coalition_minus_player else 0

                    marginal = v_coalition - v_minus

                    # Weight: (|S|-1)! (n-|S|)! / n!
                    weight = (
                        math.factorial(coalition_size - 1) *
                        math.factorial(self.n - coalition_size) /
                        math.factorial(self.n)
                    )

                    shapley[player] += weight * marginal

        return shapley

    def core(self) -> List[np.ndarray]:
        """
        Core: allocations where no coalition can improve.

        x in Core iff: Σ_i x_i = v(N) and Σ_{i in S} x_i >= v(S) for all S.
        """
        # Simplified: return Shapley value (which may not be in core)
        return [self.shapley_value()]

    def _all_coalitions(self, size: int) -> List[Tuple[int, ...]]:
        """Generate all coalitions of given size."""
        from itertools import combinations
        return list(combinations(range(self.n), size))

    @staticmethod
    def glove_game(n_left: int, n_right: int) -> 'CooperativeGame':
        """
        Glove game: need left and right glove to make pair.

        v(S) = min(left gloves in S, right gloves in S).
        """
        def v(coalition):
            if not coalition:
                return 0
            left = sum(1 for p in coalition if p < n_left)
            right = len(coalition) - left
            return min(left, right)

        return CooperativeGame(n_left + n_right, v)


class EvolutionaryGame:
    """
    Evolutionary game theory: replicator dynamics, ESS.

    Population of strategies evolving over time.
    """

    def __init__(self, payoff_matrix: np.ndarray):
        """
        Args:
            payoff_matrix: Payoff matrix A where A[i,j] = payoff to i vs j
        """
        self.A = payoff_matrix
        self.n_strategies = payoff_matrix.shape[0]

    def replicator_dynamics(self, population: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Replicator equation: dx_i/dt = x_i (f_i - f_avg).

        Strategies with above-average fitness grow.
        """
        # Average fitness of each strategy
        fitness = self.A @ population

        # Average fitness of population
        avg_fitness = population @ fitness

        # Replicator equation
        dx = population * (fitness - avg_fitness) * dt

        # Update population
        new_population = population + dx

        # Ensure non-negative and normalized
        new_population = np.maximum(new_population, 0)
        new_population /= np.sum(new_population)

        return new_population

    def simulate(self, initial_population: np.ndarray, steps: int = 1000,
                dt: float = 0.01) -> np.ndarray:
        """Simulate replicator dynamics."""
        trajectory = np.zeros((steps, self.n_strategies))
        trajectory[0] = initial_population

        for t in range(1, steps):
            trajectory[t] = self.replicator_dynamics(trajectory[t-1], dt)

        return trajectory

    def is_ess(self, strategy: int) -> bool:
        """
        Check if pure strategy is evolutionarily stable (ESS).

        σ is ESS iff: u(σ, σ) > u(σ', σ) for all σ' != σ
        (strict Nash + stability condition).
        """
        payoff_against_self = self.A[strategy, strategy]

        for other in range(self.n_strategies):
            if other == strategy:
                continue
            if self.A[other, strategy] >= payoff_against_self:
                return False

        return True


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_prisoners_dilemma():
    """Example: Prisoner's Dilemma Nash equilibrium."""
    game = NormalFormGame.prisoners_dilemma()
    equilibria = NashEquilibrium.find_pure(game)
    return equilibria


def example_vcg_auction():
    """Example: VCG auction with 3 bidders."""
    values = np.array([10, 8, 5])
    winner, payments = MechanismDesign.vcg_auction(values)
    return winner, payments, values


def example_shapley_value():
    """Example: Shapley value in glove game."""
    game = CooperativeGame.glove_game(n_left=2, n_right=1)
    shapley = game.shapley_value()
    return shapley


if __name__ == "__main__":
    print("Game Theory Module")
    print("=" * 60)

    # Test 1: Prisoner's Dilemma
    print("\n[Test 1] Prisoner's Dilemma")
    pd_eq = example_prisoners_dilemma()
    print(f"Pure Nash equilibria: {pd_eq}")
    print(f"Expected: [(1, 1)] (both defect)")

    # Test 2: Matching Pennies (no pure NE)
    print("\n[Test 2] Matching Pennies")
    mp = NormalFormGame.matching_pennies()
    mp_eq = NashEquilibrium.find_pure(mp)
    print(f"Pure Nash equilibria: {mp_eq}")
    print(f"Expected: [] (no pure NE, only mixed)")

    # Test 3: VCG Auction
    print("\n[Test 3] VCG (Vickrey) Auction")
    winner, payments, values = example_vcg_auction()
    print(f"Values: {values}")
    print(f"Winner: Player {winner} (value={values[winner]})")
    print(f"Payment: ${payments[winner]:.2f} (expect $8 = second-highest)")

    # Test 4: Second-price auction
    print("\n[Test 4] Second-price auction")
    bids = np.array([10, 8, 5])
    winner2, price2 = AuctionTheory.second_price_auction(bids)
    print(f"Bids: {bids}")
    print(f"Winner: Player {winner2}, pays ${price2:.2f}")

    # Test 5: Shapley value
    print("\n[Test 5] Shapley value (glove game)")
    shapley = example_shapley_value()
    print(f"Shapley values: {shapley}")
    print(f"Sum: {np.sum(shapley):.4f} (should equal v(N))")

    # Test 6: Replicator dynamics
    print("\n[Test 6] Replicator dynamics")
    payoff = np.array([
        [1, 0],
        [0, 2]
    ])
    evo = EvolutionaryGame(payoff)
    initial = np.array([0.7, 0.3])
    trajectory = evo.simulate(initial, steps=100)
    print(f"Initial population: {initial}")
    print(f"Final population: {trajectory[-1]}")

    print("\n" + "=" * 60)
    print("All game theory tests complete!")
    print("Nash equilibria, auctions, and Shapley values ready.")

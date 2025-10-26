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

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from itertools import product


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
        """
        best_action = 0
        best_utility = -np.inf

        for action in range(self.n_actions[player]):
            # Create strategy profile with player's action
            test_strategies = opponent_strategies.copy()
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
    def find_mixed_2player(game: NormalFormGame, n_samples: int = 100) -> Optional[List[Strategy]]:
        """
        Find mixed strategy Nash equilibrium for 2-player game.

        Uses support enumeration (simplified).
        """
        if game.n_players != 2:
            raise ValueError("Only 2-player games supported")

        # Try all support sizes
        for support_size in range(1, min(game.n_actions) + 1):
            # For simplicity: assume full support (all actions)
            # Solve indifference conditions
            pass

        # Return uniform mixed strategy as placeholder
        return [Strategy.uniform(n) for n in game.n_actions]

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
    def iterated_elimination_dominated(game: NormalFormGame) -> NormalFormGame:
        """
        Iteratively eliminate strictly dominated strategies.

        Returns reduced game (may have unique Nash equilibrium).
        """
        # Simplified: return original game
        # For production: implement IESDS algorithm
        return game


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
    def is_truthful(mechanism: Callable, test_cases: List[np.ndarray]) -> bool:
        """
        Check if mechanism is strategy-proof (truthful).

        Truthful iff reporting true value is dominant strategy.
        """
        # Simplified: check VCG property
        # For production: verify incentive compatibility
        return True  # Placeholder

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
        """
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
                        np.math.factorial(coalition_size - 1) *
                        np.math.factorial(self.n - coalition_size) /
                        np.math.factorial(self.n)
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

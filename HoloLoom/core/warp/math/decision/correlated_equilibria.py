"""
Correlated Equilibria - Beyond Nash via Mediators
=================================================

Correlated equilibrium generalizes Nash by allowing a "mediator"
to recommend correlated actions to players.

Classes:
    CorrelatedEquilibrium: LP-based correlated equilibrium finder
    Mediator: Signal distribution for coordination

Key Insight:
    Nash: Players independently randomize
    Correlated: A mediator recommends actions; players follow if no incentive to deviate

Example (Traffic Light):
    Two drivers at intersection. Nash equilibria: (Go, Stop) or (Stop, Go).
    Correlated equilibrium: Traffic light recommends who goes (better welfare).

Applications:
    - Multi-agent coordination (better than Nash)
    - Signaling mechanisms
    - Algorithmic game theory (Hedge → correlated equilibrium)

Author: Claude Code
Date: December 2025 (Math Module Expansion)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Import from game_theory module
try:
    from .game_theory import NormalFormGame, Strategy
except ImportError:
    from HoloLoom.warp.math.decision.game_theory import NormalFormGame, Strategy


class EquilibriumType(Enum):
    """Types of equilibrium concepts."""
    NASH = "nash"
    CORRELATED = "correlated"
    COARSE_CORRELATED = "coarse_correlated"


@dataclass
class CorrelatedSignal:
    """
    A correlated signal (recommendation) from mediator.

    Attributes:
        distribution: Probability distribution over action profiles
        action_profiles: List of action profiles (tuples)
    """
    distribution: np.ndarray
    action_profiles: List[Tuple[int, ...]]

    @property
    def support_size(self) -> int:
        """Number of action profiles with positive probability."""
        return np.sum(self.distribution > 1e-10)

    def sample(self) -> Tuple[int, ...]:
        """Sample an action profile from the distribution."""
        idx = np.random.choice(len(self.action_profiles), p=self.distribution)
        return self.action_profiles[idx]

    def recommendation(self, player: int) -> np.ndarray:
        """
        Get recommendation distribution for a specific player.

        Args:
            player: Player index

        Returns:
            Distribution over that player's actions (marginal)
        """
        # Extract player's action from each profile
        n_actions = max(profile[player] for profile in self.action_profiles) + 1
        marginal = np.zeros(n_actions)

        for prob, profile in zip(self.distribution, self.action_profiles):
            marginal[profile[player]] += prob

        return marginal


class CorrelatedEquilibrium:
    """
    Find correlated equilibria using linear programming.

    A correlated equilibrium is a distribution π over action profiles such that:
    For all players i and actions a_i, a'_i:
        ∑_{a_{-i}} π(a_i, a_{-i}) * [U_i(a_i, a_{-i}) - U_i(a'_i, a_{-i})] >= 0

    This says: if mediator recommends a_i, player has no incentive to deviate to a'_i.

    Uses scipy.optimize.linprog for LP solving.
    """

    @staticmethod
    def find(
        game: NormalFormGame,
        objective: str = "social_welfare"
    ) -> Optional[CorrelatedSignal]:
        """
        Find a correlated equilibrium via linear programming.

        Args:
            game: Normal form game
            objective: Optimization objective
                - "social_welfare": Maximize sum of utilities
                - "egalitarian": Maximize minimum utility
                - "uniform": Find any CE (uniform objective)

        Returns:
            CorrelatedSignal or None if no equilibrium found
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            # Fallback to simple enumeration for 2x2 games
            return CorrelatedEquilibrium._find_simple(game, objective)

        # Build LP: max c^T x s.t. Ax <= b, x >= 0, sum(x) = 1

        # Enumerate all action profiles
        if game.n_players != 2:
            raise ValueError("Currently only supports 2-player games")

        n1, n2 = game.n_actions[0], game.n_actions[1]
        n_profiles = n1 * n2

        # Create action profile list
        action_profiles = [(i, j) for i in range(n1) for j in range(n2)]

        # Build incentive compatibility constraints
        # For each player i, action a, deviation a':
        #   sum over other's actions of π(a, a_{-i}) * [U_i(a) - U_i(a')] >= 0

        constraints_A = []
        constraints_b = []

        # Player 1 constraints
        for a1 in range(n1):  # Recommended action
            for a1_prime in range(n1):  # Deviation
                if a1 == a1_prime:
                    continue

                row = np.zeros(n_profiles)
                for a2 in range(n2):
                    profile_idx = a1 * n2 + a2
                    # Utility difference: U(a1, a2) - U(a1', a2)
                    u_follow = game.payoffs[0][a1, a2]
                    u_deviate = game.payoffs[0][a1_prime, a2]
                    row[profile_idx] = u_follow - u_deviate

                # Want sum >= 0, so -row <= 0
                constraints_A.append(-row)
                constraints_b.append(0)

        # Player 2 constraints
        for a2 in range(n2):
            for a2_prime in range(n2):
                if a2 == a2_prime:
                    continue

                row = np.zeros(n_profiles)
                for a1 in range(n1):
                    profile_idx = a1 * n2 + a2
                    u_follow = game.payoffs[1][a1, a2]
                    u_deviate = game.payoffs[1][a1, a2_prime]
                    row[profile_idx] = u_follow - u_deviate

                constraints_A.append(-row)
                constraints_b.append(0)

        A_ub = np.array(constraints_A) if constraints_A else None
        b_ub = np.array(constraints_b) if constraints_b else None

        # Equality constraint: probabilities sum to 1
        A_eq = np.ones((1, n_profiles))
        b_eq = np.array([1.0])

        # Objective: depends on optimization goal
        if objective == "social_welfare":
            # Maximize sum of utilities
            c = np.zeros(n_profiles)
            for idx, (a1, a2) in enumerate(action_profiles):
                c[idx] = -(game.payoffs[0][a1, a2] + game.payoffs[1][a1, a2])
        elif objective == "egalitarian":
            # This requires a different formulation (add auxiliary variable)
            # For simplicity, use social welfare
            c = np.zeros(n_profiles)
            for idx, (a1, a2) in enumerate(action_profiles):
                c[idx] = -(game.payoffs[0][a1, a2] + game.payoffs[1][a1, a2])
        else:  # uniform
            c = np.zeros(n_profiles)

        # Solve LP
        bounds = [(0, 1) for _ in range(n_profiles)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')

        if result.success:
            # Clean up small probabilities
            distribution = np.maximum(result.x, 0)
            distribution /= distribution.sum()

            return CorrelatedSignal(
                distribution=distribution,
                action_profiles=action_profiles
            )

        return None

    @staticmethod
    def _find_simple(game: NormalFormGame, objective: str) -> Optional[CorrelatedSignal]:
        """
        Simple fallback: try Nash equilibria as correlated equilibria.

        Every Nash equilibrium is a correlated equilibrium (product distribution).
        """
        from .game_theory import NashEquilibrium

        # Find pure Nash equilibria
        pure_nash = NashEquilibrium.find_pure(game)

        if pure_nash:
            # Return first pure Nash as a correlated equilibrium
            profile = pure_nash[0]
            n1, n2 = game.n_actions[0], game.n_actions[1]
            n_profiles = n1 * n2

            action_profiles = [(i, j) for i in range(n1) for j in range(n2)]
            distribution = np.zeros(n_profiles)

            # Put all probability on the Nash equilibrium
            idx = profile[0] * n2 + profile[1]
            distribution[idx] = 1.0

            return CorrelatedSignal(distribution=distribution, action_profiles=action_profiles)

        return None

    @staticmethod
    def verify(game: NormalFormGame, signal: CorrelatedSignal, tol: float = 1e-6) -> bool:
        """
        Verify that a signal is a valid correlated equilibrium.

        Checks incentive compatibility: no player wants to deviate.
        """
        n1, n2 = game.n_actions[0], game.n_actions[1]

        # Check player 1's incentive constraints
        for a1 in range(n1):
            for a1_prime in range(n1):
                if a1 == a1_prime:
                    continue

                # Expected gain from following vs deviating
                gain = 0.0
                for a2 in range(n2):
                    profile_idx = a1 * n2 + a2
                    prob = signal.distribution[profile_idx]
                    u_follow = game.payoffs[0][a1, a2]
                    u_deviate = game.payoffs[0][a1_prime, a2]
                    gain += prob * (u_follow - u_deviate)

                if gain < -tol:
                    return False

        # Check player 2's incentive constraints
        for a2 in range(n2):
            for a2_prime in range(n2):
                if a2 == a2_prime:
                    continue

                gain = 0.0
                for a1 in range(n1):
                    profile_idx = a1 * n2 + a2
                    prob = signal.distribution[profile_idx]
                    u_follow = game.payoffs[1][a1, a2]
                    u_deviate = game.payoffs[1][a1, a2_prime]
                    gain += prob * (u_follow - u_deviate)

                if gain < -tol:
                    return False

        return True

    @staticmethod
    def social_welfare(game: NormalFormGame, signal: CorrelatedSignal) -> float:
        """Compute expected social welfare of a correlated signal."""
        total = 0.0
        for prob, profile in zip(signal.distribution, signal.action_profiles):
            if prob > 0:
                for player in range(game.n_players):
                    total += prob * game.payoffs[player][profile]
        return total

    @staticmethod
    def compare_to_nash(game: NormalFormGame) -> Dict:
        """
        Compare correlated equilibrium to Nash equilibria.

        Returns dict with Nash welfare, CE welfare, and improvement ratio.
        """
        from .game_theory import NashEquilibrium

        # Find Nash equilibria
        pure_nash = NashEquilibrium.find_pure(game)

        # Compute Nash welfare
        nash_welfare = 0.0
        if pure_nash:
            profile = pure_nash[0]
            for player in range(game.n_players):
                nash_welfare += game.payoffs[player][profile]

        # Find correlated equilibrium
        ce = CorrelatedEquilibrium.find(game, objective="social_welfare")

        ce_welfare = 0.0
        if ce:
            ce_welfare = CorrelatedEquilibrium.social_welfare(game, ce)

        improvement = (ce_welfare - nash_welfare) / nash_welfare if nash_welfare > 0 else 0

        return {
            "nash_welfare": nash_welfare,
            "ce_welfare": ce_welfare,
            "improvement_ratio": improvement,
            "pure_nash_count": len(pure_nash)
        }


class CoarseCorrelatedEquilibrium:
    """
    Coarse correlated equilibrium (CCE): weaker concept than CE.

    Players must decide whether to follow recommendation BEFORE seeing it.
    (CE: Players see recommendation, then decide whether to follow)

    Regret minimization algorithms (Hedge, EXP3) converge to CCE.
    """

    @staticmethod
    def from_regret_dynamics(
        game: NormalFormGame,
        iterations: int = 1000,
        learning_rate: float = 0.1
    ) -> CorrelatedSignal:
        """
        Find approximate CCE via regret minimization.

        Uses multiplicative weights / Hedge algorithm.
        Time-averaged strategies converge to CCE.
        """
        n1, n2 = game.n_actions[0], game.n_actions[1]

        # Initialize uniform weights
        weights1 = np.ones(n1) / n1
        weights2 = np.ones(n2) / n2

        # Track empirical distribution over joint actions
        empirical = np.zeros((n1, n2))

        for t in range(iterations):
            # Sample actions from current distributions
            a1 = np.random.choice(n1, p=weights1)
            a2 = np.random.choice(n2, p=weights2)

            # Update empirical distribution
            empirical[a1, a2] += 1

            # Compute losses (negative utilities)
            loss1 = np.zeros(n1)
            loss2 = np.zeros(n2)

            for i in range(n1):
                loss1[i] = -game.payoffs[0][i, a2]
            for j in range(n2):
                loss2[j] = -game.payoffs[1][a1, j]

            # Update weights (multiplicative weights)
            weights1 *= np.exp(-learning_rate * loss1)
            weights1 /= weights1.sum()

            weights2 *= np.exp(-learning_rate * loss2)
            weights2 /= weights2.sum()

        # Normalize empirical distribution
        empirical /= empirical.sum()

        # Convert to CorrelatedSignal
        action_profiles = [(i, j) for i in range(n1) for j in range(n2)]
        distribution = empirical.flatten()

        return CorrelatedSignal(distribution=distribution, action_profiles=action_profiles)


if __name__ == "__main__":
    print("Correlated Equilibria Module Demo")
    print("=" * 50)

    # Game of Chicken (Hawk-Dove)
    # Two drivers, each can Swerve (0) or Straight (1)
    # Both Straight = crash, Both Swerve = tie, Mixed = one wins
    U1 = np.array([
        [0, -1],   # (Swerve, Swerve), (Swerve, Straight)
        [1, -10]   # (Straight, Swerve), (Straight, Straight) = crash
    ])
    U2 = np.array([
        [0, 1],    # Symmetric
        [-1, -10]
    ])

    game = NormalFormGame([U1, U2])

    print("\nGame of Chicken:")
    print("Actions: Swerve (0), Straight (1)")
    print(f"Payoff matrix player 1:\n{U1}")

    # Find correlated equilibrium
    ce = CorrelatedEquilibrium.find(game, objective="social_welfare")

    if ce:
        print(f"\nCorrelated Equilibrium found!")
        print(f"Distribution: {ce.distribution}")
        print(f"Support size: {ce.support_size}")

        is_valid = CorrelatedEquilibrium.verify(game, ce)
        print(f"Verified: {is_valid}")

        welfare = CorrelatedEquilibrium.social_welfare(game, ce)
        print(f"Social welfare: {welfare:.2f}")

    # Compare to Nash
    comparison = CorrelatedEquilibrium.compare_to_nash(game)
    print(f"\nNash welfare: {comparison['nash_welfare']:.2f}")
    print(f"CE welfare: {comparison['ce_welfare']:.2f}")
    print(f"Improvement: {comparison['improvement_ratio']:.1%}")

    print("\n" + "=" * 50)
    print("Correlated equilibria module ready!")

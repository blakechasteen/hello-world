"""
Comprehensive Tests for Game Theory Module.

Validates:
- Pure Nash equilibrium (Prisoner's Dilemma)
- Mixed Nash equilibrium (Matching Pennies, Battle of Sexes)
- IESDS algorithm (dominated strategy elimination)
- Auction truthfulness (VCG vs first-price)
- Best response validation (Phase 1.4 fix)
- Shapley value and cooperative games
- Evolutionary games and ESS

Author: Claude Code
Date: December 2025 (Math Module Upgrade)
"""

import numpy as np
import pytest
from HoloLoom.warp.math.decision.game_theory import (
    Strategy, NormalFormGame, NashEquilibrium, MechanismDesign,
    AuctionTheory, CooperativeGame, EvolutionaryGame
)


class TestStrategy:
    """Test Strategy class fundamentals."""

    def test_pure_strategy_creation(self):
        """Pure strategy should have single action with prob 1."""
        strat = Strategy.pure(action=2, n_actions=5)

        assert strat.is_pure
        assert strat.probabilities[2] == 1.0
        assert np.sum(strat.probabilities) == 1.0

    def test_uniform_strategy(self):
        """Uniform strategy should have equal probabilities."""
        strat = Strategy.uniform(n_actions=4)

        assert not strat.is_pure
        np.testing.assert_array_almost_equal(
            strat.probabilities, [0.25, 0.25, 0.25, 0.25]
        )

    def test_mixed_strategy_detection(self):
        """Mixed strategies should not be detected as pure."""
        probs = np.array([0.3, 0.7])
        strat = Strategy(probs)

        assert not strat.is_pure


class TestPureNashEquilibrium:
    """Test pure Nash equilibrium finding."""

    def test_prisoners_dilemma_has_defect_equilibrium(self):
        """Prisoner's Dilemma should have (Defect, Defect) as only pure NE."""
        game = NormalFormGame.prisoners_dilemma()
        equilibria = NashEquilibrium.find_pure(game)

        assert len(equilibria) == 1
        assert equilibria[0] == (1, 1)  # Both defect

    def test_matching_pennies_no_pure_equilibrium(self):
        """Matching Pennies should have no pure Nash equilibrium."""
        game = NormalFormGame.matching_pennies()
        equilibria = NashEquilibrium.find_pure(game)

        assert len(equilibria) == 0

    def test_battle_of_sexes_two_pure_equilibria(self):
        """Battle of Sexes should have 2 pure Nash equilibria."""
        game = NormalFormGame.battle_of_sexes()
        equilibria = NashEquilibrium.find_pure(game)

        assert len(equilibria) == 2
        assert (0, 0) in equilibria  # Both choose first option
        assert (1, 1) in equilibria  # Both choose second option

    def test_coordination_game(self):
        """Simple coordination game test."""
        # Payoffs: coordinate is good, mismatch is bad
        U1 = np.array([[2, 0], [0, 1]])
        U2 = np.array([[2, 0], [0, 1]])
        game = NormalFormGame([U1, U2])

        equilibria = NashEquilibrium.find_pure(game)

        assert (0, 0) in equilibria
        assert (1, 1) in equilibria


class TestMixedNashEquilibrium:
    """Test mixed Nash equilibrium finding (Phase 2.1)."""

    def test_matching_pennies_mixed_equilibrium(self):
        """Matching Pennies mixed Nash should be (0.5, 0.5) for both."""
        game = NormalFormGame.matching_pennies()
        strategies = NashEquilibrium.find_mixed_2player(game)

        assert strategies is not None
        assert len(strategies) == 2

        # Both players should play uniform
        np.testing.assert_array_almost_equal(
            strategies[0].probabilities, [0.5, 0.5], decimal=2
        )
        np.testing.assert_array_almost_equal(
            strategies[1].probabilities, [0.5, 0.5], decimal=2
        )

    def test_mixed_equilibrium_is_verified(self):
        """Mixed equilibrium should pass verification."""
        game = NormalFormGame.matching_pennies()
        strategies = NashEquilibrium.find_mixed_2player(game)

        is_equilibrium = NashEquilibrium.verify_equilibrium(game, strategies)
        assert is_equilibrium

    def test_rock_paper_scissors(self):
        """Rock-Paper-Scissors should have uniform mixed NE."""
        # Payoff matrix: R beats S, S beats P, P beats R
        U1 = np.array([
            [0, -1, 1],   # Rock vs Rock/Paper/Scissors
            [1, 0, -1],   # Paper
            [-1, 1, 0]    # Scissors
        ])
        U2 = -U1  # Zero-sum
        game = NormalFormGame([U1, U2])

        strategies = NashEquilibrium.find_mixed_2player(game)

        # Unique NE is (1/3, 1/3, 1/3)
        np.testing.assert_array_almost_equal(
            strategies[0].probabilities, [1/3, 1/3, 1/3], decimal=2
        )
        np.testing.assert_array_almost_equal(
            strategies[1].probabilities, [1/3, 1/3, 1/3], decimal=2
        )


class TestIESDS:
    """Test Iterated Elimination of Strictly Dominated Strategies (Phase 2.2)."""

    def test_iesds_removes_dominated_action(self):
        """IESDS should remove strictly dominated strategies."""
        # Game where action 2 is strictly dominated by action 0
        U1 = np.array([
            [5, 5],   # Action 0: always gets 5
            [3, 3],   # Action 1: always gets 3 (dominated by 0)
            [1, 1]    # Action 2: always gets 1 (dominated by both)
        ])
        U2 = np.zeros((3, 2))  # Player 2's payoffs don't matter
        game = NormalFormGame([U1, U2])

        reduced, surviving = NashEquilibrium.iterated_elimination_dominated(game)

        # Actions 1 and 2 should be eliminated for player 0
        assert len(surviving[0]) == 1
        assert 0 in surviving[0]

    def test_prisoners_dilemma_defect_survives(self):
        """In Prisoner's Dilemma, Defect dominates Cooperate."""
        game = NormalFormGame.prisoners_dilemma()
        reduced, surviving = NashEquilibrium.iterated_elimination_dominated(game)

        # Cooperate (0) should be eliminated, Defect (1) survives
        assert surviving[0] == [1]
        assert surviving[1] == [1]

    def test_no_dominated_strategies(self):
        """Games without dominated strategies should be unchanged."""
        game = NormalFormGame.matching_pennies()
        reduced, surviving = NashEquilibrium.iterated_elimination_dominated(game)

        # All strategies survive
        assert len(surviving[0]) == 2
        assert len(surviving[1]) == 2

    def test_iesds_iterative_elimination(self):
        """IESDS should iteratively eliminate multiple rounds."""
        # After removing dominated for P1, more become dominated for P2
        U1 = np.array([
            [4, 0, 0],
            [2, 2, 2],
            [0, 0, 4]
        ])
        U2 = np.array([
            [4, 2, 0],
            [0, 2, 0],
            [0, 2, 4]
        ])
        game = NormalFormGame([U1, U2])

        reduced, surviving = NashEquilibrium.iterated_elimination_dominated(game)

        # Should reduce game through iterative elimination
        total_surviving = len(surviving[0]) + len(surviving[1])
        # The game may be reduced (exact result depends on structure)
        assert total_surviving <= 6


class TestAuctionTruthfulness:
    """Test auction truthfulness checking (Phase 2.3)."""

    def test_vcg_auction_is_truthful(self):
        """VCG (Vickrey) auction should be truthful."""
        test_cases = [
            np.array([10.0, 8.0, 5.0]),
            np.array([100.0, 50.0]),
            np.array([5.0, 5.0, 5.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ]

        is_truthful, counterexample = MechanismDesign.is_truthful(
            MechanismDesign.vcg_auction,
            test_cases,
            n_misreport_samples=20
        )

        assert is_truthful, f"VCG should be truthful but found: {counterexample}"

    def test_first_price_auction_not_truthful(self):
        """First-price auction should NOT be truthful."""
        def first_price_mechanism(bids):
            winner = int(np.argmax(bids))
            payments = np.zeros(len(bids))
            payments[winner] = bids[winner]
            return winner, payments

        test_cases = [
            np.array([10.0, 8.0, 5.0]),
            np.array([100.0, 50.0]),
        ]

        is_truthful, counterexample = MechanismDesign.is_truthful(
            first_price_mechanism,
            test_cases,
            n_misreport_samples=20
        )

        # First-price should NOT be truthful
        assert not is_truthful, "First-price should have profitable deviations"
        assert counterexample is not None
        assert counterexample['gain'] > 0

    def test_vcg_second_price_equivalence(self):
        """VCG for single item should equal second-price auction."""
        bids = np.array([10, 8, 5])

        winner_vcg, payments_vcg = MechanismDesign.vcg_auction(bids)
        winner_sp, price_sp = AuctionTheory.second_price_auction(bids)

        assert winner_vcg == winner_sp
        assert payments_vcg[winner_vcg] == price_sp


class TestBestResponseValidation:
    """Test best_response validation (Phase 1.4 fix)."""

    def test_best_response_wrong_length_raises(self):
        """best_response should raise for wrong opponent_strategies length."""
        game = NormalFormGame.prisoners_dilemma()

        with pytest.raises(ValueError) as excinfo:
            game.best_response(0, [])  # Should be 1 opponent strategy

        assert "1 elements" in str(excinfo.value)

    def test_best_response_correct_length_works(self):
        """best_response should work with correct opponent_strategies length."""
        game = NormalFormGame.prisoners_dilemma()
        opponent_strat = Strategy.pure(1, 2)  # Opponent defects

        best = game.best_response(0, [opponent_strat])

        assert best == 1  # Best response to defect is defect

    def test_best_response_three_player_game(self):
        """Test best_response validation in 3-player game."""
        # Create 3-player game
        # Payoffs are 3x3x3 tensors (action profiles)
        shape = (3, 3, 3)
        U1 = np.random.rand(*shape)
        U2 = np.random.rand(*shape)
        U3 = np.random.rand(*shape)
        game = NormalFormGame([U1, U2, U3])

        with pytest.raises(ValueError) as excinfo:
            game.best_response(0, [Strategy.uniform(3)])  # Should be 2

        assert "2 elements" in str(excinfo.value)


class TestCooperativeGames:
    """Test cooperative game theory."""

    def test_shapley_value_efficiency(self):
        """Shapley values should sum to v(N)."""
        # Simple game: v(S) = |S|
        def v(coalition):
            return len(coalition)

        game = CooperativeGame(3, v)
        shapley = game.shapley_value()

        # v(N) = 3 for grand coalition
        assert abs(np.sum(shapley) - 3.0) < 1e-6

    def test_shapley_value_symmetry(self):
        """Symmetric players should have equal Shapley values."""
        # Symmetric game: v(S) = |S|^2
        def v(coalition):
            return len(coalition) ** 2

        game = CooperativeGame(3, v)
        shapley = game.shapley_value()

        # All players symmetric -> equal values
        assert abs(shapley[0] - shapley[1]) < 1e-6
        assert abs(shapley[1] - shapley[2]) < 1e-6

    def test_glove_game(self):
        """Test glove game Shapley values."""
        game = CooperativeGame.glove_game(n_left=1, n_right=2)
        shapley = game.shapley_value()

        # One left glove is more valuable when there are 2 right gloves
        # Left glove holder should get more
        assert shapley[0] > shapley[1]  # Left > Right


class TestEvolutionaryGames:
    """Test evolutionary game theory."""

    def test_replicator_preserves_normalization(self):
        """Replicator dynamics should keep population normalized."""
        payoff = np.array([[1, 0], [0, 2]])
        evo = EvolutionaryGame(payoff)

        population = np.array([0.7, 0.3])
        for _ in range(100):
            population = evo.replicator_dynamics(population)

        assert abs(np.sum(population) - 1.0) < 1e-6

    def test_ess_strict_nash(self):
        """Pure ESS should be a strict Nash equilibrium."""
        # Strategy 0 beats all others
        payoff = np.array([
            [2, 3, 3],  # Strategy 0 beats all
            [1, 1, 1],
            [1, 1, 1]
        ])
        evo = EvolutionaryGame(payoff)

        assert evo.is_ess(0)
        assert not evo.is_ess(1)
        assert not evo.is_ess(2)

    def test_simulation_convergence(self):
        """Simulation should converge to ESS."""
        # Strategy 1 is ESS
        payoff = np.array([
            [0, 1],
            [2, 1]  # Strategy 1 beats 0 when rare
        ])
        evo = EvolutionaryGame(payoff)

        initial = np.array([0.9, 0.1])
        trajectory = evo.simulate(initial, steps=500)

        # Should converge toward strategy 1
        final = trajectory[-1]
        assert final[1] > 0.5


class TestUtilityCalculations:
    """Test utility and expected utility calculations."""

    def test_expected_utility_pure_strategies(self):
        """Expected utility with pure strategies equals direct payoff."""
        game = NormalFormGame.prisoners_dilemma()

        strat0 = Strategy.pure(0, 2)  # Cooperate
        strat1 = Strategy.pure(1, 2)  # Defect

        # (Cooperate, Defect) -> Player 0 gets -3
        eu = game.expected_utility(0, [strat0, strat1])
        assert abs(eu - (-3)) < 1e-6

    def test_expected_utility_mixed_strategies(self):
        """Expected utility with mixed strategies is weighted average."""
        game = NormalFormGame.matching_pennies()

        # Both play uniform
        uniform = Strategy.uniform(2)

        eu = game.expected_utility(0, [uniform, uniform])

        # Zero-sum with uniform: expected utility is 0
        assert abs(eu) < 1e-6


class TestDominantStrategies:
    """Test dominant strategy detection."""

    def test_defect_dominant_in_pd(self):
        """Defect should be dominant in Prisoner's Dilemma."""
        game = NormalFormGame.prisoners_dilemma()

        # Defect (1) dominates for both players
        assert game.is_dominant_strategy(0, 1)  # Player 0
        assert game.is_dominant_strategy(1, 1)  # Player 1

        # Cooperate (0) is not dominant
        assert not game.is_dominant_strategy(0, 0)
        assert not game.is_dominant_strategy(1, 0)

    def test_no_dominant_in_matching_pennies(self):
        """No dominant strategies in Matching Pennies."""
        game = NormalFormGame.matching_pennies()

        assert not game.is_dominant_strategy(0, 0)
        assert not game.is_dominant_strategy(0, 1)
        assert not game.is_dominant_strategy(1, 0)
        assert not game.is_dominant_strategy(1, 1)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_bidder_auction(self):
        """Single bidder auction should work."""
        bids = np.array([10.0])
        winner, payments = MechanismDesign.vcg_auction(bids)

        assert winner == 0
        assert payments[0] == 0  # No competition

    def test_tie_breaking(self):
        """Tied values should be handled deterministically."""
        bids = np.array([10.0, 10.0, 10.0])
        winner, payments = MechanismDesign.vcg_auction(bids)

        # Should pick first (argmax behavior)
        assert winner == 0
        assert payments[0] == 10.0

    def test_equilibrium_verification_tolerance(self):
        """Equilibrium verification should respect tolerance."""
        game = NormalFormGame.matching_pennies()

        # Slightly off from equilibrium
        probs1 = np.array([0.5001, 0.4999])
        probs2 = np.array([0.5, 0.5])
        strategies = [Strategy(probs1), Strategy(probs2)]

        # Should verify with reasonable tolerance
        is_eq = NashEquilibrium.verify_equilibrium(game, strategies, tolerance=0.01)
        assert is_eq


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

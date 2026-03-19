"""
Unit tests for warp math decision and data modules.

Covers:
- decision/game_theory.py (Nash equilibria, payoff matrices, auctions, Shapley)
- decision/operations_research.py (LP, network flows, knapsack, scheduling, EOQ)
- data_understanding.py (DataInterpreter, EnhancedMathResult)
- meaning_synthesizer.py (MeaningTemplates, MeaningResult, MeaningSynthesizer)
- smart_operation_selector.py (ThompsonSampling, OperationComposer, RigorousTester)
- causal/causal_inference.py (CausalGraph, SCM, interventions, counterfactuals)
"""

import numpy as np
import pytest

# ============================================================================
# Game Theory
# ============================================================================

from hololoom.core.warp.math.decision.game_theory import (
    Strategy,
    NormalFormGame,
    NashEquilibrium,
    MechanismDesign,
    AuctionTheory,
    CooperativeGame,
    EvolutionaryGame,
)


class TestStrategy:
    def test_pure_strategy(self):
        s = Strategy.pure(1, 3)
        assert s.is_pure
        np.testing.assert_array_equal(s.probabilities, [0, 1, 0])

    def test_uniform_strategy(self):
        s = Strategy.uniform(4)
        assert not s.is_pure
        np.testing.assert_allclose(s.probabilities, [0.25, 0.25, 0.25, 0.25])

    def test_mixed_strategy_not_pure(self):
        s = Strategy(np.array([0.5, 0.5]))
        assert not s.is_pure


class TestNormalFormGame:
    def test_prisoners_dilemma_utility(self):
        game = NormalFormGame.prisoners_dilemma()
        assert game.n_players == 2
        # (Cooperate, Cooperate) -> (-1, -1)
        assert game.utility(0, (0, 0)) == -1
        assert game.utility(1, (0, 0)) == -1
        # (Defect, Defect) -> (-2, -2)
        assert game.utility(0, (1, 1)) == -2
        assert game.utility(1, (1, 1)) == -2

    def test_expected_utility_pure(self):
        game = NormalFormGame.prisoners_dilemma()
        strats = [Strategy.pure(1, 2), Strategy.pure(1, 2)]
        eu = game.expected_utility(0, strats)
        assert eu == pytest.approx(-2.0)

    def test_best_response_prisoners_dilemma(self):
        game = NormalFormGame.prisoners_dilemma()
        # Opponent cooperates -> best response is Defect (1)
        br = game.best_response(0, [Strategy.pure(0, 2)])
        assert br == 1
        # Opponent defects -> best response is still Defect (1)
        br = game.best_response(0, [Strategy.pure(1, 2)])
        assert br == 1

    def test_is_dominant_strategy_prisoners_dilemma(self):
        game = NormalFormGame.prisoners_dilemma()
        # Defect (1) is dominant for player 0
        assert game.is_dominant_strategy(0, 1) is True
        # Cooperate (0) is not dominant
        assert game.is_dominant_strategy(0, 0) is False

    def test_matching_pennies_no_dominant(self):
        game = NormalFormGame.matching_pennies()
        assert game.is_dominant_strategy(0, 0) is False
        assert game.is_dominant_strategy(0, 1) is False

    def test_battle_of_sexes(self):
        game = NormalFormGame.battle_of_sexes()
        assert game.n_players == 2
        # Both coordinate on action 0 -> (2, 1)
        assert game.utility(0, (0, 0)) == 2
        assert game.utility(1, (0, 0)) == 1

    def test_custom_game(self):
        U1 = np.array([[3, 0], [5, 1]])
        U2 = np.array([[3, 5], [0, 1]])
        game = NormalFormGame([U1, U2], names=["A", "B"])
        assert game.names == ["A", "B"]
        assert game.utility(0, (0, 0)) == 3


class TestNashEquilibrium:
    def test_pure_nash_prisoners_dilemma(self):
        game = NormalFormGame.prisoners_dilemma()
        eq = NashEquilibrium.find_pure(game)
        assert (1, 1) in eq  # Both defect
        assert len(eq) == 1

    def test_pure_nash_matching_pennies(self):
        game = NormalFormGame.matching_pennies()
        eq = NashEquilibrium.find_pure(game)
        assert len(eq) == 0  # No pure NE

    def test_pure_nash_battle_of_sexes(self):
        game = NormalFormGame.battle_of_sexes()
        eq = NashEquilibrium.find_pure(game)
        assert (0, 0) in eq
        assert (1, 1) in eq
        assert len(eq) == 2

    def test_mixed_nash_matching_pennies(self):
        game = NormalFormGame.matching_pennies()
        result = NashEquilibrium.find_mixed_2player(game)
        assert result is not None
        s1, s2 = result
        np.testing.assert_allclose(s1.probabilities, [0.5, 0.5], atol=0.01)
        np.testing.assert_allclose(s2.probabilities, [0.5, 0.5], atol=0.01)

    def test_verify_equilibrium(self):
        game = NormalFormGame.prisoners_dilemma()
        strats = [Strategy.pure(1, 2), Strategy.pure(1, 2)]
        assert NashEquilibrium.verify_equilibrium(game, strats) is True
        # Not an equilibrium: both cooperate (either can deviate)
        strats_coop = [Strategy.pure(0, 2), Strategy.pure(0, 2)]
        assert NashEquilibrium.verify_equilibrium(game, strats_coop) is False

    def test_iterated_elimination(self):
        game = NormalFormGame.prisoners_dilemma()
        reduced, surviving = NashEquilibrium.iterated_elimination_dominated(game)
        # Defect dominates cooperate for both players
        assert surviving[0] == [1]
        assert surviving[1] == [1]


class TestMechanismDesign:
    def test_vcg_auction_winner(self):
        values = np.array([10.0, 8.0, 5.0])
        winner, payments = MechanismDesign.vcg_auction(values)
        assert winner == 0
        assert payments[winner] == pytest.approx(8.0)
        assert payments[1] == 0
        assert payments[2] == 0

    def test_vcg_single_bidder(self):
        values = np.array([10.0])
        winner, payments = MechanismDesign.vcg_auction(values)
        assert winner == 0
        assert payments[0] == 0

    def test_vcg_is_truthful(self):
        test_cases = [
            np.array([10.0, 8.0, 5.0]),
            np.array([5.0, 5.1]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        ]
        is_truthful, counter = MechanismDesign.is_truthful(
            MechanismDesign.vcg_auction, test_cases, n_misreport_samples=10
        )
        assert is_truthful is True
        assert counter is None

    def test_individual_rationality(self):
        values = np.array([10.0, 8.0, 5.0])
        payments = np.array([8.0, 0.0, 0.0])
        assert MechanismDesign.individual_rationality(values, payments) == True
        # Overpayment
        payments_bad = np.array([11.0, 0.0, 0.0])
        assert MechanismDesign.individual_rationality(values, payments_bad) == False


class TestAuctionTheory:
    def test_second_price_auction(self):
        bids = np.array([10, 8, 5])
        winner, price = AuctionTheory.second_price_auction(bids)
        assert winner == 0
        assert price == 8

    def test_first_price_auction(self):
        bids = np.array([7, 9, 5])
        winner, price = AuctionTheory.first_price_auction(bids)
        assert winner == 1
        assert price == 9

    def test_optimal_bid_first_price(self):
        # With n=2, optimal is value/2
        bid = AuctionTheory.optimal_bid_first_price(10.0, 2)
        assert bid == pytest.approx(5.0)
        # With n=3, optimal is (2/3)*value
        bid = AuctionTheory.optimal_bid_first_price(9.0, 3)
        assert bid == pytest.approx(6.0)


class TestCooperativeGame:
    def test_shapley_value_glove_game(self):
        # 2 left gloves, 1 right glove -> right is scarce
        game = CooperativeGame.glove_game(n_left=2, n_right=1)
        shapley = game.shapley_value()
        # Sum of Shapley values = v(grand coalition) = min(2,1) = 1
        assert np.sum(shapley) == pytest.approx(1.0)
        # Right glove player (index 2) gets more
        assert shapley[2] > shapley[0]
        assert shapley[2] > shapley[1]

    def test_shapley_value_symmetric(self):
        # Symmetric game: v(S) = |S|
        def v(coalition):
            return len(coalition)
        game = CooperativeGame(3, v)
        shapley = game.shapley_value()
        # Symmetric: each gets 1
        np.testing.assert_allclose(shapley, [1.0, 1.0, 1.0], atol=1e-10)

    def test_core(self):
        game = CooperativeGame.glove_game(n_left=1, n_right=1)
        core = game.core()
        assert len(core) >= 1
        # Core should sum to v(N) = 1
        assert np.sum(core[0]) == pytest.approx(1.0)


class TestEvolutionaryGame:
    def test_replicator_dynamics_step(self):
        payoff = np.array([[1, 0], [0, 2]])
        evo = EvolutionaryGame(payoff)
        pop = np.array([0.3, 0.7])
        new_pop = evo.replicator_dynamics(pop)
        assert np.sum(new_pop) == pytest.approx(1.0)
        assert all(new_pop >= 0)

    def test_simulate_convergence(self):
        # Strategy 1 dominates
        payoff = np.array([[3, 3], [0, 0]])
        evo = EvolutionaryGame(payoff)
        trajectory = evo.simulate(np.array([0.6, 0.4]), steps=500)
        # Should converge to strategy 0
        assert trajectory[-1, 0] > 0.99

    def test_is_ess(self):
        # ESS check: A[strategy, strategy] > A[other, strategy] for all other
        # Strategy 0: A[0,0]=3 > A[1,0]=2 -> ESS
        payoff = np.array([[3, 0], [2, 1]])
        evo = EvolutionaryGame(payoff)
        assert evo.is_ess(0) is True
        # Strategy 1: A[1,1]=1, A[0,1]=0. 1 > 0, so strategy 1 IS ESS too.
        # Both strategies are ESS in this game.
        # Use a game where strategy 1 is NOT ESS:
        payoff2 = np.array([[3, 0], [4, 1]])
        evo2 = EvolutionaryGame(payoff2)
        # Strategy 0: A[0,0]=3, A[1,0]=4. 4 >= 3 -> NOT ESS
        assert evo2.is_ess(0) is False
        # Strategy 1: A[1,1]=1, A[0,1]=0. 1 > 0 -> ESS
        assert evo2.is_ess(1) is True


# ============================================================================
# Operations Research
# ============================================================================

from hololoom.core.warp.math.decision.operations_research import (
    LinearProgramming,
    NetworkFlows,
    IntegerProgramming,
    Scheduling,
    DynamicProgramming,
    InventoryTheory,
)


class TestLinearProgramming:
    def test_solve_2d_graphical(self):
        # min -x - y s.t. x + y <= 4, x <= 3, y <= 3 (maximize x+y within feasible)
        # Minimum of -x-y at vertex (3,1) or (1,3) -> obj = -4
        c = np.array([-1.0, -1.0])
        A = np.array([[1, 1], [1, 0], [0, 1]])
        b = np.array([4.0, 3.0, 3.0])
        x, obj = LinearProgramming.solve_2d_graphical(c, A, b)
        assert obj == pytest.approx(-4.0, abs=0.1)

    def test_duality_theorem(self):
        result = LinearProgramming.duality_theorem(None, None, None)
        assert "Strong Duality" in result


class TestNetworkFlows:
    def test_max_flow_simple(self):
        net = NetworkFlows(4)
        net.add_edge(0, 1, 10)
        net.add_edge(0, 2, 10)
        net.add_edge(1, 3, 10)
        net.add_edge(2, 3, 10)
        assert net.max_flow(0, 3) == 20

    def test_max_flow_bottleneck(self):
        net = NetworkFlows(3)
        net.add_edge(0, 1, 5)
        net.add_edge(1, 2, 3)
        assert net.max_flow(0, 2) == 3

    def test_max_flow_complex(self):
        # From the module example
        net = NetworkFlows(6)
        net.add_edge(0, 1, 10)
        net.add_edge(0, 2, 10)
        net.add_edge(1, 3, 4)
        net.add_edge(1, 4, 8)
        net.add_edge(2, 4, 9)
        net.add_edge(3, 5, 10)
        net.add_edge(4, 3, 6)
        net.add_edge(4, 5, 10)
        flow = net.max_flow(0, 5)
        assert flow == 19

    def test_min_cut(self):
        net = NetworkFlows(3)
        net.add_edge(0, 1, 5)
        net.add_edge(1, 2, 3)
        cut_val, _ = net.min_cut(0, 2)
        assert cut_val == 3

    def test_max_flow_min_cut_theorem(self):
        result = NetworkFlows.max_flow_min_cut_theorem()
        assert "Max-Flow Min-Cut" in result


class TestScheduling:
    def test_earliest_deadline_first(self):
        jobs = [(5, 10), (3, 6), (2, 8), (4, 12)]
        schedule = Scheduling.earliest_deadline_first(jobs)
        # Sorted by deadline: 6, 8, 10, 12 -> indices 1, 2, 0, 3
        assert schedule == [1, 2, 0, 3]

    def test_shortest_processing_time(self):
        times = np.array([5, 3, 2, 4])
        order = Scheduling.shortest_processing_time(times)
        np.testing.assert_array_equal(order, [2, 1, 3, 0])

    def test_johnsons_algorithm(self):
        # 3 jobs, 2 machines
        times = np.array([[3, 4], [1, 5], [6, 2]])
        order = Scheduling.johnsons_algorithm(times)
        # Job 1: m1=1 < m2=5 -> set A
        # Job 0: m1=3 < m2=4 -> set A
        # Job 2: m1=6 >= m2=2 -> set B
        # A sorted by m1 asc: [1, 0], B sorted by m2 desc: [2]
        np.testing.assert_array_equal(order, [1, 0, 2])


class TestDynamicProgramming:
    def test_knapsack_01(self):
        values = np.array([60, 100, 120])
        weights = np.array([10, 20, 30])
        max_val, items = DynamicProgramming.knapsack_01(values, weights, 50)
        assert max_val == 220  # Items 1 and 2
        assert set(items) == {1, 2}

    def test_knapsack_single_item(self):
        values = np.array([10])
        weights = np.array([5])
        max_val, items = DynamicProgramming.knapsack_01(values, weights, 5)
        assert max_val == 10
        assert items == [0]

    def test_knapsack_no_fit(self):
        values = np.array([10, 20])
        weights = np.array([10, 20])
        max_val, items = DynamicProgramming.knapsack_01(values, weights, 5)
        assert max_val == 0
        assert items == []

    def test_shortest_path_floyd_warshall(self):
        INF = 1e9
        graph = np.array([
            [0, 3, INF],
            [INF, 0, 1],
            [INF, INF, 0]
        ])
        dist = DynamicProgramming.shortest_path(graph)
        assert dist[0, 2] == 4  # 0->1->2: 3+1
        assert dist[0, 1] == 3
        assert dist[1, 2] == 1


class TestInventoryTheory:
    def test_eoq(self):
        # D=1000, K=50, h=2 -> Q*=sqrt(2*1000*50/2) = sqrt(50000) ~ 223.6
        Q, TC = InventoryTheory.eoq(1000, 50, 2)
        assert Q == pytest.approx(223.6, abs=0.1)
        # TC = sqrt(2*1000*50*2) = sqrt(200000) ~ 447.2
        assert TC == pytest.approx(447.2, abs=0.1)


# ============================================================================
# Data Understanding
# ============================================================================

from hololoom.core.warp.math.data_understanding import (
    ValueSignificance,
    TrendType,
    InterpretedValue,
    PatternInterpretation,
    DataInterpreter,
    EnhancedMathResult,
)


class TestDataInterpreter:
    def setup_method(self):
        self.interp = DataInterpreter()

    # --- Scalar interpretation ---

    def test_interpret_similarity_high(self):
        result = self.interp.interpret_scalar(0.95, "similarity")
        assert result.significance == ValueSignificance.VERY_HIGH
        assert "high" in result.label.lower()
        assert result.anomaly is False

    def test_interpret_similarity_low(self):
        result = self.interp.interpret_scalar(0.1, "similarity")
        assert result.significance == ValueSignificance.VERY_LOW

    def test_interpret_similarity_negative(self):
        # [-1,1] range gets normalized to [0,1]
        result = self.interp.interpret_scalar(-0.5, "similarity")
        assert result.normalized_value == pytest.approx(0.25)

    def test_interpret_distance_close(self):
        result = self.interp.interpret_scalar(0.05, "distance", reference_range=(0, 1))
        assert result.significance == ValueSignificance.VERY_HIGH
        assert "close" in result.label.lower()

    def test_interpret_distance_far(self):
        result = self.interp.interpret_scalar(0.9, "distance", reference_range=(0, 1))
        assert result.significance == ValueSignificance.VERY_LOW

    def test_interpret_probability_high(self):
        result = self.interp.interpret_scalar(0.96, "probability")
        assert result.significance == ValueSignificance.VERY_HIGH
        assert "likely" in result.label.lower()

    def test_interpret_probability_low(self):
        result = self.interp.interpret_scalar(0.1, "probability")
        assert result.significance == ValueSignificance.VERY_LOW
        assert "unlikely" in result.label.lower()

    def test_interpret_generic_extreme(self):
        result = self.interp.interpret_scalar(4.0, "generic")
        assert result.significance == ValueSignificance.EXTREME

    def test_interpret_generic_moderate(self):
        result = self.interp.interpret_scalar(0.7, "generic")
        assert result.significance == ValueSignificance.MODERATE

    def test_interpret_generic_low(self):
        result = self.interp.interpret_scalar(0.2, "generic")
        assert result.significance == ValueSignificance.LOW

    # --- Anomaly detection ---

    def test_anomaly_probability_out_of_range(self):
        result = self.interp.interpret_scalar(1.5, "probability")
        assert result.anomaly is True

    def test_anomaly_nan(self):
        result = self.interp.interpret_scalar(float('nan'), "generic")
        assert result.anomaly is True

    def test_anomaly_inf(self):
        result = self.interp.interpret_scalar(float('inf'), "generic")
        assert result.anomaly is True

    def test_anomaly_outside_ref_range(self):
        result = self.interp.interpret_scalar(100.0, "generic", reference_range=(0, 1))
        assert result.anomaly is True

    def test_no_anomaly_in_range(self):
        result = self.interp.interpret_scalar(0.5, "similarity")
        assert result.anomaly is False

    # --- Array interpretation ---

    def test_interpret_array_empty(self):
        result = self.interp.interpret_array(np.array([]))
        assert "error" in result

    def test_interpret_array_stats(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.interp.interpret_array(values)
        assert result["statistics"]["mean"] == pytest.approx(3.0)
        assert result["statistics"]["min"] == pytest.approx(1.0)
        assert result["statistics"]["max"] == pytest.approx(5.0)
        assert result["statistics"]["count"] == 5

    def test_interpret_array_trend_increasing(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = self.interp.interpret_array(values)
        assert len(result["trends"]) == 1
        trend = result["trends"][0]
        assert trend.pattern_type == "trend"
        assert "increasing" in trend.description.lower()

    def test_interpret_array_anomaly_detection(self):
        values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100], dtype=float)
        result = self.interp.interpret_array(values)
        assert len(result["anomalies"]) >= 1
        assert result["anomalies"][0]["index"] == 9

    def test_interpret_array_no_anomaly_constant(self):
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = self.interp.interpret_array(values)
        assert len(result["anomalies"]) == 0

    def test_interpret_array_concentration_pattern(self):
        # Tight cluster with outlier spread
        values = np.array([5.0, 5.1, 5.05, 4.95, 5.02, 0.0, 10.0])
        result = self.interp.interpret_array(values)
        # Should detect concentration (IQR/range < 0.3)
        patterns = [p for p in result["patterns"] if p.pattern_type == "concentration"]
        assert len(patterns) >= 1

    # --- Matrix interpretation ---

    def test_interpret_matrix_symmetric(self):
        matrix = np.array([[1, 0.5], [0.5, 1]])
        result = self.interp.interpret_matrix(matrix)
        assert result["structure"]["is_symmetric"] == True

    def test_interpret_matrix_asymmetric(self):
        matrix = np.array([[1, 0.9], [0.1, 1]])
        result = self.interp.interpret_matrix(matrix)
        assert result["structure"]["is_symmetric"] == False

    def test_interpret_matrix_sparse(self):
        matrix = np.zeros((5, 5))
        matrix[0, 0] = 1
        result = self.interp.interpret_matrix(matrix)
        assert result["structure"]["sparsity"] > 0.9

    def test_interpret_matrix_correlations(self):
        matrix = np.array([[1, 0.95], [0.95, 1]])
        result = self.interp.interpret_matrix(matrix)
        assert len(result["correlations"]) >= 1
        assert result["correlations"][0]["strength"] == "strong"

    def test_interpret_matrix_diagonal_dominant(self):
        matrix = np.array([[10, 1], [1, 10]])
        result = self.interp.interpret_matrix(matrix)
        assert result["structure"]["diagonal_dominant"] is True


class TestEnhancedMathResult:
    def test_basic_dict_result(self):
        raw = {"avg_similarity": 0.85, "count": 5}
        enhanced = EnhancedMathResult("inner_product", raw)
        assert enhanced.operation == "inner_product"
        assert len(enhanced.interpretation["interpreted_values"]) >= 1

    def test_array_result(self):
        raw = {"similarities": np.array([0.9, 0.8, 0.7])}
        enhanced = EnhancedMathResult("inner_product", raw)
        assert "similarities" in enhanced.interpretation["interpreted_arrays"]

    def test_matrix_result(self):
        raw = {"correlation_matrix": np.eye(3)}
        enhanced = EnhancedMathResult("analysis", raw)
        assert "correlation_matrix" in enhanced.interpretation["interpreted_arrays"]

    def test_infer_value_type(self):
        enhanced = EnhancedMathResult("test", {})
        assert enhanced._infer_value_type("similarity_score") == "similarity"
        assert enhanced._infer_value_type("distance_metric") == "distance"
        assert enhanced._infer_value_type("probability") == "probability"
        assert enhanced._infer_value_type("random_key") == "generic"

    def test_insights_for_anomaly(self):
        raw = {"probability": 1.5}  # anomalous
        enhanced = EnhancedMathResult("test", raw)
        # Should detect the anomalous probability
        anomalous_values = [
            v for v in enhanced.interpretation["interpreted_values"]
            if v["interpretation"].anomaly
        ]
        assert len(anomalous_values) >= 1


# ============================================================================
# Meaning Synthesizer
# ============================================================================

from hololoom.core.warp.math.meaning_synthesizer import (
    MeaningTemplate,
    MeaningTemplates,
    MeaningResult,
    MeaningSynthesizer,
)
from hololoom.core.warp.math.operation_selector import (
    QueryIntent,
    MathOperation,
    OperationPlan,
    MathDomain,
    OperationLevel,
)


class TestMeaningTemplates:
    def test_templates_loaded(self):
        templates = MeaningTemplates.get_templates()
        assert len(templates) > 10
        assert "inner_product" in templates
        assert "eigenvalues" in templates
        assert "gradient" in templates

    def test_template_formatter_inner_product(self):
        templates = MeaningTemplates.get_templates()
        t = templates["inner_product"]
        formatted = t.formatter({"similarities": [0.95, 0.8, 0.7]})
        assert "0.95" in formatted["top_scores"]

    def test_template_formatter_eigenvalues_stable(self):
        templates = MeaningTemplates.get_templates()
        t = templates["eigenvalues"]
        formatted = t.formatter({"max_eigenvalue": 0.5})
        assert formatted["stability"] == "stable behavior"

    def test_template_formatter_eigenvalues_unstable(self):
        templates = MeaningTemplates.get_templates()
        t = templates["eigenvalues"]
        formatted = t.formatter({"max_eigenvalue": 1.5})
        assert formatted["stability"] == "unstable dynamics"


class TestMeaningResult:
    def test_to_text_concise(self):
        mr = MeaningResult(
            summary="Found 3 items",
            details=["Detail 1"],
            key_insights=["Insight 1"],
        )
        assert mr.to_text("concise") == "Found 3 items"

    def test_to_text_detailed(self):
        mr = MeaningResult(
            summary="Summary here",
            details=["Detail 1", "Detail 2"],
            key_insights=["Insight A"],
            recommendations=["Rec 1"],
        )
        text = mr.to_text("detailed")
        assert "Summary here" in text
        assert "Detail 1" in text
        assert "Insight A" in text
        assert "Rec 1" in text

    def test_to_text_technical(self):
        mr = MeaningResult(
            summary="Summary",
            details=["Math detail"],
            key_insights=[],
            provenance={"op": "inner_product"},
        )
        text = mr.to_text("technical")
        assert "Provenance" in text
        assert "inner_product" in text

    def test_to_text_unknown_style(self):
        mr = MeaningResult(summary="Fallback", details=[], key_insights=[])
        assert mr.to_text("something_else") == "Fallback"


def _make_plan(op_names):
    """Helper to create a simple OperationPlan for testing."""
    ops = [
        MathOperation(
            name=name,
            domain=MathDomain.ANALYSIS,
            level=OperationLevel.BASIC,
            description=f"Test {name}",
            use_cases=[QueryIntent.SIMILARITY],
            estimated_cost=5,
        )
        for name in op_names
    ]
    return OperationPlan(
        operations=ops,
        justifications={name: "test" for name in op_names},
        total_cost=5 * len(ops),
        domains_used={MathDomain.ANALYSIS},
        metadata={"primary_intent": "similarity"},
    )


class TestMeaningSynthesizer:
    def setup_method(self):
        self.synth = MeaningSynthesizer(use_data_understanding=False)

    def test_synthesize_similarity(self):
        results = {"inner_product": {"similarities": [0.9, 0.8]}}
        plan = _make_plan(["inner_product"])
        meaning = self.synth.synthesize(results, QueryIntent.SIMILARITY, plan)
        assert "similar" in meaning.summary.lower() or "Found" in meaning.summary
        assert meaning.confidence > 0

    def test_synthesize_optimization(self):
        results = {"gradient": {"improvement": 0.15}}
        plan = _make_plan(["gradient"])
        meaning = self.synth.synthesize(results, QueryIntent.OPTIMIZATION, plan)
        assert "Optimized" in meaning.summary or "improvement" in meaning.summary.lower()

    def test_synthesize_verification(self):
        results = {"metric_verification": {"is_valid": True, "all_passed": True}}
        plan = _make_plan(["metric_verification"])
        meaning = self.synth.synthesize(results, QueryIntent.VERIFICATION, plan)
        assert "Verified" in meaning.summary

    def test_confidence_reduces_on_failure(self):
        results = {"op1": {"success": False}}
        plan = _make_plan(["op1"])
        meaning = self.synth.synthesize(results, QueryIntent.ANALYSIS, plan)
        assert meaning.confidence < 1.0

    def test_confidence_reduces_on_low_cost(self):
        results = {"op1": {"success": True}}
        plan = _make_plan(["op1"])
        plan.total_cost = 5  # low cost
        meaning = self.synth.synthesize(results, QueryIntent.ANALYSIS, plan)
        assert meaning.confidence <= 0.95

    def test_recommendations_high_cost(self):
        results = {"op1": {"success": True}}
        plan = _make_plan(["op1"])
        plan.total_cost = 50
        meaning = self.synth.synthesize(results, QueryIntent.ANALYSIS, plan)
        cost_recs = [r for r in meaning.recommendations if "cost" in r.lower()]
        assert len(cost_recs) >= 1

    def test_insights_low_similarity(self):
        results = {"inner_product": {"similarities": [0.1, 0.2]}}
        plan = _make_plan(["inner_product"])
        meaning = self.synth.synthesize(results, QueryIntent.SIMILARITY, plan)
        low_insights = [i for i in meaning.key_insights if "low" in i.lower() or "Low" in i]
        assert len(low_insights) >= 1

    def test_provenance_tracking(self):
        results = {"gradient": {"success": True}}
        plan = _make_plan(["gradient"])
        meaning = self.synth.synthesize(results, QueryIntent.OPTIMIZATION, plan)
        assert "gradient" in meaning.provenance["operations_executed"]
        assert meaning.provenance["intent"] == "optimization"

    def test_synthesis_history(self):
        results = {"op1": {"success": True}}
        plan = _make_plan(["op1"])
        self.synth.synthesize(results, QueryIntent.ANALYSIS, plan)
        self.synth.synthesize(results, QueryIntent.ANALYSIS, plan)
        assert len(self.synth.synthesis_history) == 2


# ============================================================================
# Smart Operation Selector
# ============================================================================

from hololoom.core.warp.math.smart_operation_selector import (
    ComposedOperation,
    OperationComposer,
    OperationStatistics,
    ThompsonSamplingLearner,
    RigorousTester,
    SmartMathOperationSelector,
)


class TestOperationStatistics:
    def test_initial_success_rate(self):
        stats = OperationStatistics("op1", "similarity")
        assert stats.success_rate == 0.5  # No data

    def test_update_success(self):
        stats = OperationStatistics("op1", "similarity")
        stats.update(True, 5, 0.1)
        assert stats.successes == 1
        assert stats.failures == 0
        assert stats.success_rate == 1.0

    def test_update_failure(self):
        stats = OperationStatistics("op1", "similarity")
        stats.update(False, 5, 0.1)
        assert stats.successes == 0
        assert stats.failures == 1
        assert stats.success_rate == 0.0

    def test_running_average_time(self):
        stats = OperationStatistics("op1", "similarity")
        stats.update(True, 5, 1.0)
        stats.update(True, 5, 3.0)
        assert stats.avg_execution_time == pytest.approx(2.0)


class TestOperationComposer:
    def _make_ops(self, names):
        return [
            MathOperation(
                name=name,
                domain=MathDomain.ANALYSIS,
                level=OperationLevel.BASIC,
                description=f"Test {name}",
                use_cases=[QueryIntent.SIMILARITY],
                estimated_cost=10,
            )
            for name in names
        ]

    def test_compose_sequential(self):
        composer = OperationComposer()
        ops = self._make_ops(["op1", "op2"])
        composed = composer.compose_sequential(ops)
        assert composed.composition_type == "sequential"
        assert composed.estimated_cost == 20  # sum
        assert "op1" in composed.name
        assert "op2" in composed.name

    def test_compose_parallel(self):
        composer = OperationComposer()
        ops = self._make_ops(["op1", "op2", "op3"])
        composed = composer.compose_parallel(ops)
        assert composed.composition_type == "parallel"
        assert composed.estimated_cost == 10  # max

    def test_suggest_compositions(self):
        composer = OperationComposer()
        ops = [
            MathOperation("inner_product", MathDomain.ANALYSIS, OperationLevel.BASIC,
                          "sim", [QueryIntent.SIMILARITY], estimated_cost=5),
            MathOperation("metric_distance", MathDomain.GEOMETRY, OperationLevel.BASIC,
                          "dist", [QueryIntent.SIMILARITY], estimated_cost=5),
        ]
        suggestions = composer.suggest_compositions(ops)
        assert len(suggestions) >= 1
        assert any("similarity_pipeline" in s.name for s in suggestions)

    def test_compose_custom_name(self):
        composer = OperationComposer()
        ops = self._make_ops(["a", "b"])
        composed = composer.compose_sequential(ops, name="custom_pipe")
        assert composed.name == "custom_pipe"


class TestThompsonSamplingLearner:
    def test_get_stats_creates(self):
        learner = ThompsonSamplingLearner()
        stats = learner.get_stats("op1", "similarity")
        assert stats.operation_name == "op1"
        assert stats.intent == "similarity"

    def test_select_operation(self):
        learner = ThompsonSamplingLearner()
        ops = [
            MathOperation("op1", MathDomain.ANALYSIS, OperationLevel.BASIC,
                          "d", [QueryIntent.SIMILARITY], estimated_cost=5),
            MathOperation("op2", MathDomain.ANALYSIS, OperationLevel.BASIC,
                          "d", [QueryIntent.SIMILARITY], estimated_cost=5),
        ]
        selected = learner.select_operation(ops, QueryIntent.SIMILARITY)
        assert selected.name in ["op1", "op2"]

    def test_learning_improves_selection(self):
        np.random.seed(42)
        learner = ThompsonSamplingLearner()
        ops = [
            MathOperation("good", MathDomain.ANALYSIS, OperationLevel.BASIC,
                          "d", [QueryIntent.SIMILARITY], estimated_cost=5),
            MathOperation("bad", MathDomain.ANALYSIS, OperationLevel.BASIC,
                          "d", [QueryIntent.SIMILARITY], estimated_cost=5),
        ]
        # Train: "good" always succeeds, "bad" always fails
        for _ in range(20):
            learner.record_feedback("good", "similarity", True, 5, 0.1)
            learner.record_feedback("bad", "similarity", False, 5, 0.1)

        # After training, "good" should be selected most of the time
        selections = [learner.select_operation(ops, QueryIntent.SIMILARITY).name for _ in range(50)]
        good_count = selections.count("good")
        assert good_count > 40  # Should be heavily biased toward "good"

    def test_get_leaderboard(self):
        learner = ThompsonSamplingLearner()
        learner.record_feedback("op1", "sim", True, 5, 0.1)
        learner.record_feedback("op2", "sim", False, 5, 0.1)
        board = learner.get_leaderboard()
        assert len(board) == 2
        assert board[0]["operation_name"] == "op1"  # Higher success rate

    def test_get_leaderboard_filtered(self):
        learner = ThompsonSamplingLearner()
        learner.record_feedback("op1", "sim", True, 5, 0.1)
        learner.record_feedback("op2", "opt", True, 5, 0.1)
        board = learner.get_leaderboard(intent="sim")
        assert len(board) == 1


class TestRigorousTester:
    def test_numerical_stability_pass(self):
        tester = RigorousTester()
        op = MathOperation("test_op", MathDomain.ANALYSIS, OperationLevel.BASIC,
                           "d", [QueryIntent.ANALYSIS], estimated_cost=5)
        result = tester.verify_operation(
            op, {"result": np.array([1.0, 2.0, 3.0])},
            properties=["numerical_stability"]
        )
        assert result["all_passed"] is True

    def test_numerical_stability_fail_nan(self):
        tester = RigorousTester()
        op = MathOperation("test_op", MathDomain.ANALYSIS, OperationLevel.BASIC,
                           "d", [QueryIntent.ANALYSIS], estimated_cost=5)
        result = tester.verify_operation(
            op, {"result": np.array([1.0, float('nan')])},
            properties=["numerical_stability"]
        )
        assert result["all_passed"] is False

    def test_metric_symmetry(self):
        tester = RigorousTester()
        op = MathOperation("metric_distance", MathDomain.GEOMETRY, OperationLevel.BASIC,
                           "d", [QueryIntent.SIMILARITY], estimated_cost=5)
        result = tester.verify_operation(
            op,
            {
                "distance_function": lambda x, y: np.linalg.norm(x - y),
                "samples": [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
                "result": 1.0,
            },
            properties=["metric_symmetry", "metric_triangle_inequality", "metric_identity"]
        )
        assert result["all_passed"] is True

    def test_orthogonality_check(self):
        tester = RigorousTester()
        op = MathOperation("gram_schmidt", MathDomain.ALGEBRA, OperationLevel.BASIC,
                           "d", [QueryIntent.TRANSFORMATION], estimated_cost=5)
        # Orthonormal basis
        vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        result = tester.verify_operation(
            op,
            {"vectors": vectors, "result": vectors},
            properties=["orthogonality", "normalization"]
        )
        assert result["all_passed"] is True

    def test_gradient_descent_convergence(self):
        tester = RigorousTester()
        op = MathOperation("gradient", MathDomain.OPTIMIZATION, OperationLevel.BASIC,
                           "d", [QueryIntent.OPTIMIZATION], estimated_cost=5)
        result = tester.verify_operation(
            op,
            {"objective_values": [10.0, 8.0, 5.0, 3.0, 2.5], "result": 2.5},
            properties=["gradient_descent_convergence"]
        )
        assert result["all_passed"] is True

    def test_gradient_descent_divergence(self):
        tester = RigorousTester()
        op = MathOperation("gradient", MathDomain.OPTIMIZATION, OperationLevel.BASIC,
                           "d", [QueryIntent.OPTIMIZATION], estimated_cost=5)
        result = tester.verify_operation(
            op,
            {"objective_values": [10.0, 8.0, 12.0], "result": 12.0},
            properties=["gradient_descent_convergence"]
        )
        assert result["results"]["gradient_descent_convergence"]["passed"] is False

    def test_get_test_report_empty(self):
        tester = RigorousTester()
        report = tester.get_test_report()
        assert report["total_tests"] == 0

    def test_get_test_report(self):
        tester = RigorousTester()
        op = MathOperation("test_op", MathDomain.ANALYSIS, OperationLevel.BASIC,
                           "d", [QueryIntent.ANALYSIS], estimated_cost=5)
        tester.verify_operation(op, {"result": 1.0}, properties=["numerical_stability"])
        report = tester.get_test_report()
        assert report["total_tests"] == 1
        assert report["pass_rate"] == 1.0

    def test_auto_select_properties(self):
        tester = RigorousTester()
        metric_op = MathOperation("metric_distance", MathDomain.GEOMETRY, OperationLevel.BASIC,
                                  "d", [QueryIntent.SIMILARITY], estimated_cost=5)
        props = tester._select_properties_for_operation(metric_op)
        assert "numerical_stability" in props
        assert "metric_symmetry" in props


class TestSmartMathOperationSelector:
    def test_init(self):
        selector = SmartMathOperationSelector(load_state=False, use_contextual=False)
        assert selector.learner is not None
        assert selector.composer is not None
        assert selector.tester is not None

    def test_plan_operations_smart(self):
        selector = SmartMathOperationSelector(load_state=False, use_contextual=False)
        plan = selector.plan_operations_smart(
            query_text="Find similar documents",
            enable_learning=True,
            enable_composition=True,
            budget=50,
        )
        assert len(plan.operations) > 0
        assert plan.total_cost <= 50

    def test_execute_plan_with_verification(self):
        selector = SmartMathOperationSelector(load_state=False, use_contextual=False)
        plan = selector.plan_operations_smart(
            query_text="Analyze convergence",
            enable_learning=True,
            budget=30,
        )
        result = selector.execute_plan_with_verification(plan, {})
        assert "operations_executed" in result
        assert "total_time" in result

    def test_record_feedback(self):
        selector = SmartMathOperationSelector(load_state=False, use_contextual=False)
        plan = selector.plan_operations_smart(
            query_text="Find similar items",
            enable_learning=True,
            budget=50,
        )
        selector.record_feedback(plan, success=True, quality=0.9, execution_time=0.1)
        stats = selector.get_smart_statistics()
        assert stats["rl_learning"]["total_feedback"] > 0

    def test_state_save_load(self, tmp_path):
        selector = SmartMathOperationSelector(load_state=False, use_contextual=False)
        selector.state_file = tmp_path / "state.json"

        # Record some feedback
        selector.learner.record_feedback("op1", "similarity", True, 5, 0.1)
        selector._save_state()

        # Load in new selector
        selector2 = SmartMathOperationSelector(load_state=False, use_contextual=False)
        selector2.state_file = tmp_path / "state.json"
        selector2._load_state()
        assert ("op1", "similarity") in selector2.learner.stats


# ============================================================================
# Causal Inference
# ============================================================================

from hololoom.core.warp.math.causal.causal_inference import (
    CausalQueryType,
    CausalQuery,
    CausalGraph,
    StructuralEquation,
    StructuralCausalModel,
    IdentificationResult,
    CausalReasoner,
    CausalDiscovery,
    identify_effect,
    is_identifiable,
    do_intervention,
    counterfactual,
)


class TestCausalGraph:
    def test_add_edge_and_nodes(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        assert "X" in g.nodes
        assert "Y" in g.nodes

    def test_parents_children(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        g.add_edge("Z", "Y")
        assert g.parents("Y") == {"X", "Z"}
        assert g.children("X") == {"Y"}

    def test_ancestors(self):
        g = CausalGraph()
        g.add_edges([("A", "B"), ("B", "C"), ("A", "C")])
        assert g.ancestors("C") == {"A", "B"}
        assert g.ancestors("A") == set()

    def test_descendants(self):
        g = CausalGraph()
        g.add_edges([("A", "B"), ("B", "C")])
        assert g.descendants("A") == {"B", "C"}
        assert g.descendants("C") == set()

    def test_do_removes_incoming_edges(self):
        g = CausalGraph()
        g.add_edges([("Z", "X"), ("X", "Y"), ("Z", "Y")])
        mutilated = g.do({"X"})
        # do(X) removes incoming edges to X (Z->X removed)
        assert mutilated.parents("X") == set()
        # Outgoing edges from X are kept (X->Y remains)
        assert mutilated.children("X") == {"Y"}
        # Y's parents: X->Y and Z->Y both remain
        assert mutilated.parents("Y") == {"X", "Z"}

    def test_d_separation_chain(self):
        # A -> B -> C: A _||_ C | B
        g = CausalGraph()
        g.add_edges([("A", "B"), ("B", "C")])
        assert g.d_separated({"A"}, {"C"}, {"B"}) is True
        assert g.d_separated({"A"}, {"C"}, set()) is False

    def test_d_separation_fork(self):
        # B <- A -> C: B _||_ C | A
        g = CausalGraph()
        g.add_edges([("A", "B"), ("A", "C")])
        assert g.d_separated({"B"}, {"C"}, {"A"}) is True
        assert g.d_separated({"B"}, {"C"}, set()) is False

    def test_find_adjustment_set(self):
        # Z -> X -> Y, Z -> Y (confounder)
        g = CausalGraph()
        g.add_edges([("Z", "X"), ("X", "Y"), ("Z", "Y")])
        adj = g.find_adjustment_set("X", "Y")
        assert adj is not None
        assert "Z" in adj

    def test_is_valid_adjustment_set(self):
        g = CausalGraph()
        g.add_edges([("Z", "X"), ("X", "Y"), ("Z", "Y")])
        assert g.is_valid_adjustment_set("X", "Y", {"Z"}) is True
        # Descendant of X can't be in adjustment set
        g.add_edge("X", "M")
        assert g.is_valid_adjustment_set("X", "Y", {"M"}) is False


class TestStructuralEquation:
    def test_evaluate_linear(self):
        eq = StructuralEquation(
            variable="Y",
            parents=["X"],
            function=lambda p: 2 * p["X"],
        )
        val = eq.evaluate({"X": 3.0}, noise=0.0)
        assert val == pytest.approx(6.0)

    def test_evaluate_with_noise(self):
        eq = StructuralEquation(
            variable="Y",
            parents=["X"],
            function=lambda p: p["X"],
            noise_dist=lambda: 1.0,
        )
        val = eq.evaluate({"X": 5.0})
        assert val == pytest.approx(6.0)


class TestStructuralCausalModel:
    def _make_simple_scm(self):
        """Z -> X -> Y, Z -> Y (confounder)."""
        scm = StructuralCausalModel()
        scm.add_equation("Z", [], lambda _: 0.0,
                         noise_dist=lambda: np.random.normal(0, 1))
        scm.add_equation("X", ["Z"], lambda p: 0.5 * p["Z"],
                         noise_dist=lambda: 0.0)
        scm.add_equation("Y", ["X", "Z"], lambda p: 2 * p["X"] + p["Z"],
                         noise_dist=lambda: 0.0)
        return scm

    def test_topological_order(self):
        scm = self._make_simple_scm()
        order = scm.topological_order
        assert order.index("Z") < order.index("X")
        assert order.index("X") < order.index("Y")

    def test_sample(self):
        scm = self._make_simple_scm()
        samples = scm.sample(100)
        assert "Z" in samples
        assert "X" in samples
        assert "Y" in samples
        assert len(samples["Y"]) == 100

    def test_intervention_do_x(self):
        scm = self._make_simple_scm()
        np.random.seed(42)
        samples = scm.intervene({"X": 2.0}, n_samples=500)
        # Y = 2*X + Z = 2*2 + Z = 4 + Z, E[Z]=0 -> E[Y]~4
        mean_y = np.mean(samples["Y"])
        assert mean_y == pytest.approx(4.0, abs=0.3)
        # X should be constant at 2.0
        np.testing.assert_allclose(samples["X"], 2.0)

    def test_counterfactual(self):
        scm = self._make_simple_scm()
        # Observed: Z=1, X=0.5, Y=2*0.5+1=2
        observed = {"Z": 1.0, "X": 0.5, "Y": 2.0}
        # Counterfactual: what if X had been 0?
        cf_y = scm.counterfactual(observed, {"X": 0.0}, "Y")
        # Y = 2*0 + Z = Z; noise for Z = 1 - 0 = 1, so Z=1, Y=1
        assert cf_y == pytest.approx(1.0)


class TestCausalQuery:
    def test_interventional_str(self):
        q = CausalQuery(
            outcome=["Y"],
            treatment=["X"],
            query_type=CausalQueryType.INTERVENTIONAL,
        )
        s = str(q)
        assert "do(X)" in s
        assert "Y" in s

    def test_observational_str(self):
        q = CausalQuery(
            outcome=["Y"],
            treatment=["X"],
            query_type=CausalQueryType.OBSERVATIONAL,
        )
        s = str(q)
        assert "P(Y | X)" in s


class TestIdentifyEffect:
    def test_identifiable_with_confounder(self):
        g = CausalGraph()
        g.add_edges([("Z", "X"), ("X", "Y"), ("Z", "Y")])
        result = identify_effect(g, "X", "Y")
        assert result.identifiable is True
        assert result.adjustment_set is not None
        assert "Z" in result.adjustment_set

    def test_is_identifiable_simple(self):
        g = CausalGraph()
        g.add_edges([("X", "Y")])
        assert is_identifiable(g, "X", "Y") is True

    def test_identifiable_no_adjustment_needed(self):
        g = CausalGraph()
        g.add_edge("X", "Y")
        result = identify_effect(g, "X", "Y")
        assert result.identifiable is True
        assert result.adjustment_set == set()


class TestCausalReasoner:
    def test_what_if(self):
        scm = StructuralCausalModel()
        scm.add_equation("X", [], lambda _: 0.0,
                         noise_dist=lambda: 0.0)
        scm.add_equation("Y", ["X"], lambda p: 3 * p["X"],
                         noise_dist=lambda: 0.0)
        reasoner = CausalReasoner(scm)
        result = reasoner.what_if({"X": 2.0}, "Y")
        assert result["mean"] == pytest.approx(6.0)

    def test_what_if_all(self):
        scm = StructuralCausalModel()
        scm.add_equation("X", [], lambda _: 0.0, noise_dist=lambda: 0.0)
        scm.add_equation("Y", ["X"], lambda p: p["X"], noise_dist=lambda: 0.0)
        reasoner = CausalReasoner(scm)
        result = reasoner.what_if({"X": 5.0}, "all")
        assert "X" in result
        assert "Y" in result
        assert result["Y"]["mean"] == pytest.approx(5.0)

    def test_would_have_been(self):
        scm = StructuralCausalModel()
        scm.add_equation("X", [], lambda _: 0.0, noise_dist=lambda: 0.0)
        scm.add_equation("Y", ["X"], lambda p: 2 * p["X"], noise_dist=lambda: 0.0)
        reasoner = CausalReasoner(scm)
        val = reasoner.would_have_been(
            observed={"X": 3.0, "Y": 6.0},
            counterfactual_action={"X": 0.0},
            query="Y"
        )
        assert val == pytest.approx(0.0)

    def test_add_causal_mechanism(self):
        reasoner = CausalReasoner()
        reasoner.add_causal_mechanism("Y", ["X"], lambda p: p["X"] + 1)
        assert "Y" in reasoner.scm.equations


class TestCausalDiscovery:
    def test_from_correlation(self):
        np.random.seed(42)
        # X causes Y
        x = np.random.normal(0, 1, 100)
        y = 2 * x + np.random.normal(0, 0.1, 100)
        data = np.column_stack([x, y])
        graph = CausalDiscovery.from_correlation(data, ["X", "Y"], threshold=0.3)
        assert "X" in graph.nodes
        assert "Y" in graph.nodes


class TestConvenienceFunctions:
    def test_do_intervention_fn(self):
        scm = StructuralCausalModel()
        scm.add_equation("X", [], lambda _: 0.0, noise_dist=lambda: 0.0)
        scm.add_equation("Y", ["X"], lambda p: p["X"], noise_dist=lambda: 0.0)
        samples = do_intervention(scm, {"X": 3.0}, n_samples=10)
        np.testing.assert_allclose(samples["Y"], 3.0)

    def test_counterfactual_fn(self):
        scm = StructuralCausalModel()
        scm.add_equation("X", [], lambda _: 0.0, noise_dist=lambda: 0.0)
        scm.add_equation("Y", ["X"], lambda p: p["X"], noise_dist=lambda: 0.0)
        val = counterfactual(scm, {"X": 5.0, "Y": 5.0}, {"X": 0.0}, "Y")
        assert val == pytest.approx(0.0)

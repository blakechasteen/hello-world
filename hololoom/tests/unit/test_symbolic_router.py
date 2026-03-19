"""
Tests for symbolic_router.py — Adaptive dispatch to optimal solver.

Wire 11 of the structured AI integration roadmap.
"""

import numpy as np
import pytest

from hololoom.core.convergence.symbolic_router import (
    SymbolicRouter,
    ProblemType,
    RoutingDecision,
    RoutingOutcome,
    Solver,
    classify_problem,
    create_symbolic_router,
)


# ============================================================================
# Problem classification
# ============================================================================

class TestClassifyProblem:
    def test_causal_query(self):
        pt, signals = classify_problem("What causes inflation to rise?")
        assert pt == ProblemType.CAUSAL

    def test_intervention_query(self):
        pt, _ = classify_problem("What happens if we increase the dosage?")
        assert pt == ProblemType.CAUSAL

    def test_counterfactual_query(self):
        pt, _ = classify_problem("Why did the system fail?")
        assert pt == ProblemType.CAUSAL

    def test_constraint_query(self):
        pt, _ = classify_problem("Given the constraints, find a valid solution")
        assert pt == ProblemType.CONSTRAINT

    def test_graph_query(self):
        pt, _ = classify_problem("What is the path between quantum computing and cryptography in the graph?")
        assert pt == ProblemType.GRAPH_TRAVERSAL

    def test_ranking_query(self):
        pt, _ = classify_problem("Which tool is best for this task? Rank the options.")
        assert pt == ProblemType.RANKING

    def test_prediction_query(self):
        pt, _ = classify_problem("What will happen to memory usage next week?")
        assert pt == ProblemType.PREDICTION

    def test_uncertainty_query(self):
        pt, _ = classify_problem("How confident are we in this probability estimate?")
        assert pt == ProblemType.UNCERTAINTY

    def test_deliberation_query(self):
        pt, _ = classify_problem("Should we refactor the module? Consider the tradeoffs.")
        assert pt == ProblemType.DELIBERATION

    def test_unknown_falls_to_factual(self):
        pt, _ = classify_problem("Hello world")
        assert pt == ProblemType.FACTUAL

    def test_signals_returned(self):
        _, signals = classify_problem("What causes X and what is the effect?")
        assert len(signals) > 0
        assert "causal" in signals


# ============================================================================
# Router basics
# ============================================================================

class TestSymbolicRouterBasics:
    def test_default_solvers_registered(self):
        router = SymbolicRouter()
        names = router.solver_names()
        assert "causal_engine" in names
        assert "kg_reasoner" in names
        assert "convergence_engine" in names
        assert "world_model" in names
        assert "deliberation" in names

    def test_route_returns_decision(self):
        router = SymbolicRouter()
        decision = router.route("What causes the system to slow down?")
        assert isinstance(decision, RoutingDecision)
        assert decision.solver_name in router.solver_names()
        assert 0.0 <= decision.confidence <= 1.0

    def test_causal_routes_to_causal_engine(self):
        router = SymbolicRouter()
        # Run multiple times — Thompson Sampling is stochastic
        causal_count = 0
        for _ in range(20):
            decision = router.route("What causes X to lead to Y?")
            if decision.solver_name in ("causal_engine", "causal_bandit"):
                causal_count += 1
        assert causal_count > 5  # Should route to causal most of the time

    def test_graph_routes_to_kg_reasoner(self):
        router = SymbolicRouter()
        kg_count = 0
        for _ in range(20):
            decision = router.route("How is node A connected to node B in the graph?")
            if decision.solver_name == "kg_reasoner":
                kg_count += 1
        assert kg_count > 5

    def test_prediction_routes_to_world_model(self):
        router = SymbolicRouter()
        wm_count = 0
        for _ in range(20):
            decision = router.route("Predict what will happen to the activation next")
            if decision.solver_name in ("world_model", "free_energy"):
                wm_count += 1
        assert wm_count > 5

    def test_alternatives_provided(self):
        router = SymbolicRouter()
        decision = router.route("Rank these options and compare them")
        # Should have at least one alternative
        assert isinstance(decision.alternatives, list)

    def test_unknown_query_gets_routed(self):
        router = SymbolicRouter()
        decision = router.route("asdfghjkl 12345")
        assert decision.solver_name  # Should still pick something


# ============================================================================
# Thompson Sampling learning
# ============================================================================

class TestRouterLearning:
    def test_outcome_updates_prior(self):
        router = SymbolicRouter()
        # Report good outcome for causal_engine on causal problems
        for _ in range(10):
            router.report_outcome(RoutingOutcome(
                solver_name="causal_engine",
                problem_type=ProblemType.CAUSAL,
                quality=0.9,
            ))

        rankings = router.get_solver_rankings(ProblemType.CAUSAL)
        # causal_engine should be ranked high
        solver_names = [name for name, _ in rankings]
        assert solver_names[0] == "causal_engine"

    def test_poor_outcomes_lower_ranking(self):
        router = SymbolicRouter()
        # Report bad outcomes for convergence_engine on ranking
        for _ in range(10):
            router.report_outcome(RoutingOutcome(
                solver_name="convergence_engine",
                problem_type=ProblemType.RANKING,
                quality=0.1,
            ))
        # Report good outcomes for minimax_gate on ranking
        for _ in range(10):
            router.report_outcome(RoutingOutcome(
                solver_name="minimax_gate",
                problem_type=ProblemType.RANKING,
                quality=0.9,
            ))

        rankings = router.get_solver_rankings(ProblemType.RANKING)
        names = [name for name, _ in rankings]
        # minimax should rank above convergence
        assert names.index("minimax_gate") < names.index("convergence_engine")


# ============================================================================
# Custom solvers
# ============================================================================

class TestCustomSolvers:
    def test_register_custom_solver(self):
        router = SymbolicRouter()
        custom = Solver(
            name="my_solver",
            problem_types=[ProblemType.FACTUAL],
            priority=2.0,
            description="Custom factual solver",
        )
        router.register_solver(custom)
        assert "my_solver" in router.solver_names()

    def test_custom_solver_gets_routed(self):
        router = SymbolicRouter()
        custom = Solver(
            name="super_causal",
            problem_types=[ProblemType.CAUSAL],
            priority=10.0,  # Very high priority
        )
        router.register_solver(custom)

        # With high priority, should get selected often
        super_count = 0
        for _ in range(20):
            decision = router.route("What causes this effect?")
            if decision.solver_name == "super_causal":
                super_count += 1
        assert super_count > 5


# ============================================================================
# Factory
# ============================================================================

class TestFactory:
    def test_create(self):
        router = create_symbolic_router()
        assert isinstance(router, SymbolicRouter)
        assert len(router.solver_names()) >= 9

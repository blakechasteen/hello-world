"""
Tests for scoring_evolver.py — AlphaEvolve pattern for self-improving scoring.

Wire 12 of the structured AI integration roadmap.
"""

import numpy as np
import pytest

from hololoom.core.convergence.scoring_evolver import (
    ScoringEvolver,
    ScoringConfig,
    EvolutionStats,
    create_scoring_evolver,
)


# ============================================================================
# ScoringConfig
# ============================================================================

class TestScoringConfig:
    def test_score_computation(self):
        config = ScoringConfig(weights={"w_a": 2.0, "w_b": 3.0})
        score = config.score({"w_a": 1.0, "w_b": 2.0})
        assert score == pytest.approx(8.0)  # 2*1 + 3*2

    def test_score_missing_terms(self):
        config = ScoringConfig(weights={"w_a": 2.0, "w_b": 3.0})
        score = config.score({"w_a": 1.0})  # w_b missing in values
        assert score == pytest.approx(2.0)  # 2*1 + 3*0

    def test_score_extra_terms_ignored(self):
        config = ScoringConfig(weights={"w_a": 2.0})
        score = config.score({"w_a": 1.0, "w_extra": 100.0})
        assert score == pytest.approx(2.0)


# ============================================================================
# Population seeding
# ============================================================================

class TestSeeding:
    def test_seed_creates_population(self):
        evolver = ScoringEvolver(
            term_names=["w_tc", "w_urgency", "w_variety"],
            initial_weights={"w_tc": 25, "w_urgency": 15, "w_variety": 5},
        )
        configs = evolver.seed_population(n=10)
        assert len(configs) == 10

    def test_first_seed_matches_initial(self):
        initial = {"w_tc": 25, "w_urgency": 15, "w_variety": 5}
        evolver = ScoringEvolver(term_names=list(initial.keys()), initial_weights=initial)
        configs = evolver.seed_population(n=5)
        # First config should be exact initial weights
        assert configs[0].weights == initial

    def test_seeds_are_diverse(self):
        evolver = ScoringEvolver(
            term_names=["a", "b", "c"],
            mutation_scale=10.0,
        )
        configs = evolver.seed_population(n=20)
        # Not all configs should have the same weights
        weight_sets = [tuple(c.weights.values()) for c in configs]
        assert len(set(weight_sets)) > 1

    def test_seeds_have_non_negative_weights(self):
        evolver = ScoringEvolver(term_names=["a", "b"], mutation_scale=50.0)
        configs = evolver.seed_population(n=20)
        for config in configs:
            for w in config.weights.values():
                assert w >= 0.0


# ============================================================================
# Mutation
# ============================================================================

class TestMutation:
    def test_mutate_returns_child(self):
        evolver = ScoringEvolver(term_names=["a", "b", "c"])
        parent = ScoringConfig(weights={"a": 10, "b": 20, "c": 30}, config_id=99)
        child = evolver.mutate(parent)
        assert child.parent_id == 99
        assert child.config_id != parent.config_id
        assert child.generation > 0

    def test_mutate_changes_some_weights(self):
        evolver = ScoringEvolver(
            term_names=["a", "b", "c", "d", "e"],
            mutation_rate=0.8,  # High mutation rate
            mutation_scale=10.0,
        )
        parent = ScoringConfig(weights={t: 10.0 for t in evolver.term_names})

        # Over many mutations, at least some weights should change
        any_changed = False
        for _ in range(10):
            child = evolver.mutate(parent)
            if any(child.weights[t] != parent.weights[t] for t in evolver.term_names):
                any_changed = True
                break
        assert any_changed

    def test_mutate_preserves_non_negativity(self):
        evolver = ScoringEvolver(
            term_names=["a", "b"],
            mutation_rate=1.0,
            mutation_scale=100.0,
        )
        parent = ScoringConfig(weights={"a": 1.0, "b": 1.0})
        for _ in range(50):
            child = evolver.mutate(parent)
            for w in child.weights.values():
                assert w >= 0.0


# ============================================================================
# Evaluation
# ============================================================================

class TestEvaluation:
    def _make_historical_data(self):
        """Create historical data where option 0 is always best with high w_a."""
        data = []
        truth = []
        for _ in range(10):
            example = {
                "option_0_w_a": 0.9, "option_0_w_b": 0.1,
                "option_1_w_a": 0.1, "option_1_w_b": 0.9,
            }
            data.append(example)
            truth.append(0)  # Option 0 is correct
        return data, truth

    def test_perfect_config_scores_high(self):
        data, truth = self._make_historical_data()
        evolver = ScoringEvolver(term_names=["w_a", "w_b"])

        # Config that weights w_a highly should score well
        good_config = ScoringConfig(weights={"w_a": 100.0, "w_b": 1.0})
        fitness = evolver.evaluate(good_config, data, truth)
        assert fitness == 1.0

    def test_bad_config_scores_low(self):
        data, truth = self._make_historical_data()
        evolver = ScoringEvolver(term_names=["w_a", "w_b"])

        # Config that weights w_b highly should score poorly
        bad_config = ScoringConfig(weights={"w_a": 1.0, "w_b": 100.0})
        fitness = evolver.evaluate(bad_config, data, truth)
        assert fitness == 0.0

    def test_empty_data_returns_neutral(self):
        evolver = ScoringEvolver(term_names=["w_a"])
        fitness = evolver.evaluate(ScoringConfig(weights={"w_a": 1.0}), [], [])
        assert fitness == 0.5

    def test_fitness_stored_on_config(self):
        data, truth = self._make_historical_data()
        evolver = ScoringEvolver(term_names=["w_a", "w_b"])
        config = ScoringConfig(weights={"w_a": 100.0, "w_b": 1.0})
        evolver.evaluate(config, data, truth)
        assert config.fitness == 1.0


# ============================================================================
# Evolution loop
# ============================================================================

class TestEvolutionLoop:
    def test_full_loop(self):
        evolver = ScoringEvolver(
            term_names=["w_a", "w_b"],
            initial_weights={"w_a": 10.0, "w_b": 10.0},
            mutation_rate=0.5,
            mutation_scale=5.0,
        )
        evolver.seed_population(n=5)

        # Historical data where w_a matters
        data = [
            {"option_0_w_a": 0.9, "option_0_w_b": 0.1,
             "option_1_w_a": 0.1, "option_1_w_b": 0.9}
        ] * 10
        truth = [0] * 10

        # Run 50 generations
        for _ in range(50):
            parent = evolver.select_parent()
            child = evolver.mutate(parent)
            fitness = evolver.evaluate(child, data, truth)
            evolver.record(child, fitness)

        best = evolver.best_config()
        assert best is not None
        assert best.fitness > 0.5

    def test_stats_tracked(self):
        evolver = ScoringEvolver(term_names=["a", "b"])
        evolver.seed_population(n=5)

        for _ in range(10):
            parent = evolver.select_parent()
            child = evolver.mutate(parent)
            child.fitness = np.random.random()
            evolver.record(child, child.fitness)

        stats = evolver.stats()
        assert isinstance(stats, EvolutionStats)
        assert stats.generations == 10
        assert stats.population_size > 0
        assert len(stats.fitness_trajectory) == 10

    def test_evolution_improves_fitness(self):
        """Over many generations, best fitness should improve."""
        evolver = ScoringEvolver(
            term_names=["w_a", "w_b", "w_c"],
            initial_weights={"w_a": 5.0, "w_b": 5.0, "w_c": 5.0},
            mutation_rate=0.5,
            mutation_scale=3.0,
        )
        evolver.seed_population(n=10)

        # Data where w_a=high, w_b=low, w_c=medium is optimal
        data = [
            {"option_0_w_a": 0.8, "option_0_w_b": 0.2, "option_0_w_c": 0.5,
             "option_1_w_a": 0.2, "option_1_w_b": 0.8, "option_1_w_c": 0.5}
        ] * 20
        truth = [0] * 20

        initial_best = evolver.best_config()
        initial_fitness = evolver.evaluate(initial_best, data, truth) if initial_best else 0.0

        for _ in range(100):
            parent = evolver.select_parent()
            child = evolver.mutate(parent)
            fitness = evolver.evaluate(child, data, truth)
            evolver.record(child, fitness)

        final_best = evolver.best_config()
        assert final_best.fitness >= initial_fitness

    def test_select_parent_without_population(self):
        evolver = ScoringEvolver(term_names=["a"])
        parent = evolver.select_parent()
        assert isinstance(parent, ScoringConfig)


# ============================================================================
# MAP-Elites diversity
# ============================================================================

class TestDiversity:
    def test_population_maintains_diversity(self):
        evolver = ScoringEvolver(
            term_names=["a", "b"],
            mutation_rate=1.0,
            mutation_scale=20.0,
        )
        evolver.seed_population(n=20)

        stats = evolver.stats()
        assert stats.diversity > 0  # Some cells occupied
        assert stats.population_size > 1


# ============================================================================
# Factory
# ============================================================================

class TestFactory:
    def test_create(self):
        evolver = create_scoring_evolver(
            term_names=["x", "y"],
            initial_weights={"x": 5.0, "y": 10.0},
            mutation_rate=0.4,
        )
        assert isinstance(evolver, ScoringEvolver)
        assert evolver.mutation_rate == 0.4

    def test_defaults(self):
        evolver = create_scoring_evolver(term_names=["a", "b", "c"])
        assert evolver.mutation_rate == 0.3
        assert evolver.mutation_scale == 5.0

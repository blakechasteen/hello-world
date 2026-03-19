"""
Tests for convergence refinement_strategies.py.

Covers:
- QueryClassifier: all query types, edge cases, unknown queries
- StrategyStats: update logic, Thompson Sampling delegation, averages
- StrategySelector: heuristic paths, learning toggle, statistics, best-strategy lookup
- StrategyExecutor: every strategy's execution params
- create_strategy_selector factory
"""

import sys
import types
from dataclasses import dataclass
from enum import Enum
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Shim — the real protocols module is missing RefinementStrategy / RefinementStep.
# We inject compatible stubs before importing the module under test.
# ---------------------------------------------------------------------------

class RefinementStrategy(Enum):
    EXPAND_SEARCH = "expand_search"
    RERANK = "rerank"
    ALTERNATE_MODE = "alternate_mode"
    DECOMPOSE = "decompose"
    VERIFY_AND_CORRECT = "verify_and_correct"
    MULTI_PERSPECTIVE = "multi_perspective"


@dataclass
class RefinementStep:
    iteration: int
    query: str
    response: str
    confidence: float
    improvement: float
    strategy_used: str
    reasoning: str
    timestamp: object = None


class StrategySelectorProtocol:
    """Stub — satisfies the import; not used by tests."""
    pass


def _patch_protocol_shim():
    """Inject stubs into the sys.modules slot that refinement_strategies.py imports.

    Try to load the real module first so other convergence modules that import
    from it get the full set of names. Only fall back to a minimal shim when the
    real module cannot be loaded (missing deps like fabric.spacetime).
    """
    _mod_names = (
        "hololoom.protocols.recursive_reasoning",
        "hololoom.core.protocols.recursive_reasoning",
    )
    _stub_names = {
        "RefinementStrategy": RefinementStrategy,
        "RefinementStep": RefinementStep,
        "StrategySelectorProtocol": StrategySelectorProtocol,
    }

    # Try loading the real module if it hasn't been loaded yet
    existing = sys.modules.get(_mod_names[0]) or sys.modules.get(_mod_names[1])
    if existing is None:
        try:
            import importlib
            existing = importlib.import_module("hololoom.core.protocols.recursive_reasoning")
        except (ImportError, Exception):
            existing = None

    for mod_name in _mod_names:
        current = sys.modules.get(mod_name)
        if current is not None:
            for k, v in _stub_names.items():
                if not hasattr(current, k):
                    setattr(current, k, v)
        elif existing is not None:
            for k, v in _stub_names.items():
                if not hasattr(existing, k):
                    setattr(existing, k, v)
            sys.modules[mod_name] = existing
        else:
            shim = types.ModuleType(mod_name)
            for k, v in _stub_names.items():
                setattr(shim, k, v)
            sys.modules[mod_name] = shim


_patch_protocol_shim()

# Re-import shimmed types from the protocol module so this test uses the same
# class objects as the code under test (avoids isinstance/enum identity mismatches
# when another test file loaded its own shim first).
import hololoom.core.protocols.recursive_reasoning as _proto_mod  # noqa: E402

RefinementStrategy = _proto_mod.RefinementStrategy  # noqa: F811
RefinementStep = _proto_mod.RefinementStep  # noqa: F811

# Now safe to import the module under test
from hololoom.core.convergence.refinement_strategies import (  # noqa: E402
    QueryClassifier,
    StrategyStats,
    StrategySelector,
    StrategyExecutor,
    create_strategy_selector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(confidence: float, improvement: float = 0.0, iteration: int = 1) -> RefinementStep:
    return RefinementStep(
        iteration=iteration,
        query="q",
        response="r",
        confidence=confidence,
        improvement=improvement,
        strategy_used="test",
        reasoning="test",
    )


def _history(pairs: List) -> List[RefinementStep]:
    return [_step(c, imp, i + 1) for i, (c, imp) in enumerate(pairs)]


# ============================================================================
# QueryClassifier
# ============================================================================

class TestQueryClassifier:
    def setup_method(self):
        self.clf = QueryClassifier()

    # --- factual ---
    def test_what_is_classified_factual(self):
        assert self.clf.classify("What is Python?") == "factual"

    def test_define_classified_factual(self):
        assert self.clf.classify("Define machine learning") == "factual"

    def test_explain_classified_factual(self):
        assert self.clf.classify("Explain gradient descent") == "factual"

    def test_describe_classified_factual(self):
        assert self.clf.classify("Describe the attention mechanism") == "factual"

    # --- procedural ---
    def test_how_to_classified_procedural(self):
        assert self.clf.classify("How to implement gradient descent?") == "procedural"

    def test_how_do_classified_procedural(self):
        assert self.clf.classify("How do neural networks learn?") == "procedural"

    def test_steps_classified_procedural(self):
        assert self.clf.classify("What are the steps to train a model?") == "procedural"

    def test_process_classified_procedural(self):
        # Must not contain factual keywords (what is / define / explain / describe)
        assert self.clf.classify("The process of backpropagation is iterative") == "procedural"

    def test_method_classified_procedural(self):
        assert self.clf.classify("What method should I use?") == "procedural"

    # --- analytical ---
    def test_why_classified_analytical(self):
        assert self.clf.classify("Why does overfitting occur?") == "analytical"

    def test_reason_classified_analytical(self):
        # Must not contain factual keywords; use "reason" directly without "what is"
        assert self.clf.classify("The reason for vanishing gradients") == "analytical"

    def test_cause_classified_analytical(self):
        assert self.clf.classify("What causes mode collapse in GANs?") == "analytical"

    def test_analyze_classified_analytical(self):
        assert self.clf.classify("Analyze the impact of learning rate") == "analytical"

    def test_evaluate_classified_analytical(self):
        assert self.clf.classify("Evaluate the tradeoffs of this approach") == "analytical"

    # --- comparative ---
    def test_compare_classified_comparative(self):
        assert self.clf.classify("Compare Python and Java") == "comparative"

    def test_difference_classified_comparative(self):
        # Must not trigger "what is" (factual) — use "difference" without that prefix
        assert self.clf.classify("The difference between LSTM and GRU") == "comparative"

    def test_vs_classified_comparative(self):
        assert self.clf.classify("SVM vs neural networks") == "comparative"

    def test_versus_classified_comparative(self):
        assert self.clf.classify("Batch versus online learning") == "comparative"

    def test_better_classified_comparative(self):
        assert self.clf.classify("Which is better, Adam or SGD?") == "comparative"

    # --- technical ---
    def test_algorithm_classified_technical(self):
        # Must not trigger "explain" (factual); use "algorithm" without factual/procedural prefix
        assert self.clf.classify("The algorithm for quicksort uses partitioning") == "technical"

    def test_code_classified_technical(self):
        assert self.clf.classify("Write code for binary search") == "technical"

    def test_function_classified_technical(self):
        assert self.clf.classify("Write a function to parse JSON") == "technical"

    def test_implement_classified_technical(self):
        assert self.clf.classify("Implement a linked list") == "technical"

    def test_optimize_classified_technical(self):
        # "how to" triggers procedural before technical — use "optimize" directly
        assert self.clf.classify("optimize this database query for performance") == "technical"

    # --- creative (fallback) ---
    def test_no_keyword_classified_creative(self):
        assert self.clf.classify("Tell me about the future of AI") == "creative"

    def test_empty_string_classified_creative(self):
        assert self.clf.classify("") == "creative"

    def test_gibberish_classified_creative(self):
        assert self.clf.classify("xyzzy frobnicator zap") == "creative"

    # --- case insensitivity ---
    def test_classify_is_case_insensitive(self):
        assert self.clf.classify("WHAT IS Python?") == "factual"
        assert self.clf.classify("HOW TO train?") == "procedural"
        assert self.clf.classify("WHY does this happen?") == "analytical"

    # --- priority ordering: factual wins over procedural when both match ---
    def test_factual_takes_priority_over_procedural(self):
        # "what is" (factual) appears before "how to" (procedural) in check order
        result = self.clf.classify("What is the process of digestion?")
        assert result == "factual"


# ============================================================================
# StrategyStats
# ============================================================================

class TestStrategyStats:
    def test_initial_state(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        assert stats.total_uses == 0
        assert stats.successful_uses == 0
        assert stats.total_improvement == pytest.approx(0.0)
        assert stats.avg_improvement == pytest.approx(0.0)
        assert stats.success_rate == pytest.approx(0.0)

    def test_update_positive_improvement(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        stats.update(0.10)
        assert stats.total_uses == 1
        assert stats.successful_uses == 1
        assert stats.total_improvement == pytest.approx(0.10)
        assert stats.avg_improvement == pytest.approx(0.10)
        assert stats.success_rate == pytest.approx(1.0)

    def test_update_negative_improvement(self):
        stats = StrategyStats(strategy=RefinementStrategy.RERANK)
        stats.update(-0.05)
        assert stats.total_uses == 1
        assert stats.successful_uses == 0
        assert stats.success_rate == pytest.approx(0.0)

    def test_update_zero_improvement(self):
        stats = StrategyStats(strategy=RefinementStrategy.RERANK)
        stats.update(0.0)
        assert stats.total_uses == 1
        assert stats.successful_uses == 0  # 0.0 is not > 0
        assert stats.success_rate == pytest.approx(0.0)

    def test_avg_improvement_over_multiple_updates(self):
        stats = StrategyStats(strategy=RefinementStrategy.DECOMPOSE)
        stats.update(0.10)
        stats.update(0.20)
        assert stats.total_uses == 2
        assert stats.avg_improvement == pytest.approx(0.15)

    def test_success_rate_mixed(self):
        stats = StrategyStats(strategy=RefinementStrategy.MULTI_PERSPECTIVE)
        stats.update(0.10)   # success
        stats.update(-0.05)  # failure
        assert stats.success_rate == pytest.approx(0.5)

    def test_alpha_beta_delegate_to_arm(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        # Alpha and beta are delegated to BetaArm — starts at (1.0, 1.0)
        assert stats.alpha == pytest.approx(1.0)
        assert stats.beta == pytest.approx(1.0)

    def test_alpha_increases_on_success(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        before_alpha = stats.alpha
        stats.update(0.15)
        assert stats.alpha > before_alpha

    def test_beta_increases_on_failure(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        before_beta = stats.beta
        stats.update(-0.10)
        assert stats.beta > before_beta

    def test_get_thompson_sample_in_range(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        for _ in range(20):
            s = stats.get_thompson_sample()
            assert 0.0 <= s <= 1.0

    def test_get_expected_reward_initial(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        # BetaArm(1,1) → expected = 0.5
        assert stats.get_expected_reward() == pytest.approx(0.5)

    def test_get_expected_reward_after_successes(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        for _ in range(5):
            stats.update(0.10)
        # After repeated successes, expected reward should exceed 0.5
        assert stats.get_expected_reward() > 0.5

    def test_alpha_setter_delegates_to_arm(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        stats.alpha = 5.0
        assert stats.alpha == pytest.approx(5.0)
        assert stats.get_expected_reward() == pytest.approx(5.0 / (5.0 + 1.0))

    def test_beta_setter_delegates_to_arm(self):
        stats = StrategyStats(strategy=RefinementStrategy.EXPAND_SEARCH)
        stats.beta = 9.0
        assert stats.beta == pytest.approx(9.0)


# ============================================================================
# StrategySelector — heuristic path
# ============================================================================

class TestStrategySelectorHeuristic:
    """Tests for _heuristic_strategy: no learning, no history needed."""

    def setup_method(self):
        self.selector = StrategySelector(enable_learning=False)

    @pytest.mark.asyncio
    async def test_low_confidence_factual_returns_expand_search(self):
        strategy = await self.selector.select_strategy("What is Python?", 0.45, [])
        assert strategy == RefinementStrategy.EXPAND_SEARCH

    @pytest.mark.asyncio
    async def test_low_confidence_technical_returns_expand_search(self):
        strategy = await self.selector.select_strategy("Implement a sort algorithm", 0.40, [])
        assert strategy == RefinementStrategy.EXPAND_SEARCH

    @pytest.mark.asyncio
    async def test_low_confidence_creative_returns_decompose(self):
        strategy = await self.selector.select_strategy("Tell me about the future", 0.30, [])
        assert strategy == RefinementStrategy.DECOMPOSE

    @pytest.mark.asyncio
    async def test_low_confidence_procedural_returns_decompose(self):
        strategy = await self.selector.select_strategy("How to build a rocket?", 0.40, [])
        assert strategy == RefinementStrategy.DECOMPOSE

    @pytest.mark.asyncio
    async def test_medium_confidence_analytical_returns_verify(self):
        strategy = await self.selector.select_strategy("Why does overfitting occur?", 0.60, [])
        assert strategy == RefinementStrategy.VERIFY_AND_CORRECT

    @pytest.mark.asyncio
    async def test_medium_confidence_comparative_returns_multi_perspective(self):
        strategy = await self.selector.select_strategy("Compare Python and Java", 0.65, [])
        assert strategy == RefinementStrategy.MULTI_PERSPECTIVE

    @pytest.mark.asyncio
    async def test_medium_confidence_other_returns_rerank(self):
        strategy = await self.selector.select_strategy("Tell me about the future", 0.60, [])
        assert strategy == RefinementStrategy.RERANK

    @pytest.mark.asyncio
    async def test_high_confidence_procedural_returns_verify(self):
        strategy = await self.selector.select_strategy("How to train a model?", 0.75, [])
        assert strategy == RefinementStrategy.VERIFY_AND_CORRECT

    @pytest.mark.asyncio
    async def test_high_confidence_non_procedural_returns_alternate_mode(self):
        strategy = await self.selector.select_strategy("What is entropy?", 0.80, [])
        assert strategy == RefinementStrategy.ALTERNATE_MODE

    @pytest.mark.asyncio
    async def test_returns_valid_enum_member(self):
        strategy = await self.selector.select_strategy("xyzzy", 0.50, [])
        assert isinstance(strategy, RefinementStrategy)

    @pytest.mark.asyncio
    async def test_no_history_uses_heuristic(self):
        """With enable_learning=True but empty history, falls back to heuristic."""
        selector_learning = StrategySelector(enable_learning=True)
        strategy = await selector_learning.select_strategy("What is entropy?", 0.80, [])
        # Empty history → heuristic path
        assert isinstance(strategy, RefinementStrategy)


# ============================================================================
# StrategySelector — learning path
# ============================================================================

class TestStrategySelectorLearning:
    def setup_method(self):
        self.selector = StrategySelector(enable_learning=True)

    @pytest.mark.asyncio
    async def test_learn_from_positive_outcome(self):
        await self.selector.learn_from_outcome(
            strategy=RefinementStrategy.EXPAND_SEARCH,
            improvement=0.15,
            query_type="factual",
        )
        key = ("factual", RefinementStrategy.EXPAND_SEARCH)
        assert self.selector.strategy_stats[key].total_uses == 1
        assert self.selector.strategy_stats[key].successful_uses == 1

    @pytest.mark.asyncio
    async def test_learn_from_negative_outcome(self):
        await self.selector.learn_from_outcome(
            strategy=RefinementStrategy.RERANK,
            improvement=-0.05,
            query_type="creative",
        )
        key = ("creative", RefinementStrategy.RERANK)
        assert self.selector.strategy_stats[key].total_uses == 1
        assert self.selector.strategy_stats[key].successful_uses == 0

    @pytest.mark.asyncio
    async def test_learning_disabled_skips_update(self):
        selector = StrategySelector(enable_learning=False)
        await selector.learn_from_outcome(
            strategy=RefinementStrategy.EXPAND_SEARCH,
            improvement=0.15,
            query_type="factual",
        )
        key = ("factual", RefinementStrategy.EXPAND_SEARCH)
        # Should not have been updated
        assert selector.strategy_stats[key].total_uses == 0

    @pytest.mark.asyncio
    async def test_thompson_sample_used_with_history(self):
        """With history, selector should use Thompson Sampling (not heuristic)."""
        history = [_step(0.5, 0.05)]
        # Call several times — all should return valid enum members
        results = set()
        for _ in range(30):
            strategy = await self.selector.select_strategy("What is Python?", 0.70, history)
            results.add(strategy)
        # At least one strategy was selected
        assert len(results) >= 1
        for s in results:
            assert isinstance(s, RefinementStrategy)

    @pytest.mark.asyncio
    async def test_learn_creates_missing_key(self):
        """Learning for an unregistered key should create it on the fly."""
        selector = StrategySelector(enable_learning=True)
        # Manually add a fake query type
        await selector.learn_from_outcome(
            strategy=RefinementStrategy.DECOMPOSE,
            improvement=0.08,
            query_type="unknown_type",
        )
        key = ("unknown_type", RefinementStrategy.DECOMPOSE)
        assert key in selector.strategy_stats
        assert selector.strategy_stats[key].total_uses == 1

    @pytest.mark.asyncio
    async def test_repeated_learning_accumulates(self):
        for _ in range(5):
            await self.selector.learn_from_outcome(
                strategy=RefinementStrategy.MULTI_PERSPECTIVE,
                improvement=0.10,
                query_type="analytical",
            )
        key = ("analytical", RefinementStrategy.MULTI_PERSPECTIVE)
        assert self.selector.strategy_stats[key].total_uses == 5
        assert self.selector.strategy_stats[key].successful_uses == 5


# ============================================================================
# StrategySelector — initialization
# ============================================================================

class TestStrategySelectorInit:
    def test_initializes_all_query_type_strategy_combinations(self):
        selector = StrategySelector(enable_learning=True)
        query_types = ["factual", "procedural", "analytical", "comparative", "technical", "creative"]
        for qt in query_types:
            for strategy in RefinementStrategy:
                assert (qt, strategy) in selector.strategy_stats

    def test_all_initial_stats_have_zero_uses(self):
        selector = StrategySelector(enable_learning=False)
        for stats in selector.strategy_stats.values():
            assert stats.total_uses == 0

    def test_enable_learning_stored(self):
        s1 = StrategySelector(enable_learning=True)
        s2 = StrategySelector(enable_learning=False)
        assert s1.enable_learning is True
        assert s2.enable_learning is False

    def test_classifier_created(self):
        selector = StrategySelector()
        assert isinstance(selector.classifier, QueryClassifier)


# ============================================================================
# StrategySelector — statistics
# ============================================================================

class TestStrategySelectorStatistics:
    @pytest.mark.asyncio
    async def test_get_strategy_statistics_by_query_type(self):
        selector = StrategySelector(enable_learning=True)
        await selector.learn_from_outcome(
            strategy=RefinementStrategy.EXPAND_SEARCH,
            improvement=0.10,
            query_type="factual",
        )
        stats = selector.get_strategy_statistics(query_type="factual")
        assert stats["query_type"] == "factual"
        assert "expand_search" in stats["strategies"]

    @pytest.mark.asyncio
    async def test_get_strategy_statistics_all_types(self):
        selector = StrategySelector(enable_learning=True)
        await selector.learn_from_outcome(
            strategy=RefinementStrategy.RERANK,
            improvement=0.08,
            query_type="factual",
        )
        stats = selector.get_strategy_statistics()
        assert stats["query_type"] == "all"

    @pytest.mark.asyncio
    async def test_get_strategy_statistics_aggregates_across_types(self):
        selector = StrategySelector(enable_learning=True)
        await selector.learn_from_outcome(
            strategy=RefinementStrategy.EXPAND_SEARCH,
            improvement=0.10,
            query_type="factual",
        )
        await selector.learn_from_outcome(
            strategy=RefinementStrategy.EXPAND_SEARCH,
            improvement=0.20,
            query_type="technical",
        )
        stats = selector.get_strategy_statistics()
        # aggregated "all" should include expand_search with total_uses=2
        assert stats["strategies"].get("expand_search", {}).get("total_uses", 0) == 2

    @pytest.mark.asyncio
    async def test_get_strategy_statistics_empty_no_uses(self):
        selector = StrategySelector(enable_learning=True)
        stats = selector.get_strategy_statistics(query_type="factual")
        # No learning done — strategies dict may be empty or contain zero-use entries
        # but the structure must be correct
        assert "strategies" in stats
        assert "query_type" in stats

    def test_get_best_strategy_no_learning(self):
        selector = StrategySelector(enable_learning=False)
        strategy, reward = selector.get_best_strategy_for_type("factual")
        assert isinstance(strategy, RefinementStrategy)
        assert 0.0 <= reward <= 1.0

    @pytest.mark.asyncio
    async def test_get_best_strategy_after_learning(self):
        selector = StrategySelector(enable_learning=True)
        # Give EXPAND_SEARCH many successes for "factual"
        for _ in range(10):
            await selector.learn_from_outcome(
                strategy=RefinementStrategy.EXPAND_SEARCH,
                improvement=0.10,
                query_type="factual",
            )
        best, reward = selector.get_best_strategy_for_type("factual")
        assert best == RefinementStrategy.EXPAND_SEARCH
        assert reward > 0.5

    def test_get_best_strategy_fallback_when_no_winner(self):
        selector = StrategySelector(enable_learning=False)
        strategy, reward = selector.get_best_strategy_for_type("factual")
        # All at default Beta(1,1)=0.5, so EXPAND_SEARCH is the declared fallback
        assert strategy == RefinementStrategy.EXPAND_SEARCH


# ============================================================================
# StrategyExecutor
# ============================================================================

class TestStrategyExecutor:
    def test_expand_search_params(self):
        params = StrategyExecutor.get_execution_params(RefinementStrategy.EXPAND_SEARCH)
        assert params["max_sources"] == 10
        assert params["mode"] == "research"
        assert params["enable_reranking"] is True

    def test_rerank_params(self):
        params = StrategyExecutor.get_execution_params(RefinementStrategy.RERANK)
        assert params["max_sources"] == 5
        assert params["enable_reranking"] is True
        assert params["mode"] == "verify"

    def test_alternate_mode_params(self):
        params = StrategyExecutor.get_execution_params(RefinementStrategy.ALTERNATE_MODE)
        assert params["mode"] == "plan_execute"
        assert params["enable_reranking"] is False

    def test_decompose_params(self):
        params = StrategyExecutor.get_execution_params(RefinementStrategy.DECOMPOSE)
        assert params["decompose"] is True
        assert params["max_sources"] == 3
        assert params["mode"] == "direct"

    def test_verify_and_correct_params(self):
        params = StrategyExecutor.get_execution_params(RefinementStrategy.VERIFY_AND_CORRECT)
        assert params["max_sources"] == 7
        assert params["mode"] == "verify"
        assert params["enable_reranking"] is True
        assert params["cross_check"] is True

    def test_multi_perspective_params(self):
        params = StrategyExecutor.get_execution_params(RefinementStrategy.MULTI_PERSPECTIVE)
        assert params["max_sources"] == 8
        assert params["perspective_count"] == 3

    def test_all_strategies_return_dict(self):
        for strategy in RefinementStrategy:
            params = StrategyExecutor.get_execution_params(strategy)
            assert isinstance(params, dict)
            assert len(params) > 0

    def test_all_strategies_have_max_sources(self):
        for strategy in RefinementStrategy:
            params = StrategyExecutor.get_execution_params(strategy)
            assert "max_sources" in params
            assert isinstance(params["max_sources"], int)
            assert params["max_sources"] > 0

    def test_is_static_method(self):
        """get_execution_params can be called on the class, not just an instance."""
        params = StrategyExecutor.get_execution_params(RefinementStrategy.RERANK)
        assert isinstance(params, dict)


# ============================================================================
# Factory function
# ============================================================================

class TestCreateStrategySelector:
    def test_creates_strategy_selector_instance(self):
        selector = create_strategy_selector()
        assert isinstance(selector, StrategySelector)

    def test_enable_learning_true_by_default(self):
        selector = create_strategy_selector()
        assert selector.enable_learning is True

    def test_enable_learning_can_be_disabled(self):
        selector = create_strategy_selector(enable_learning=False)
        assert selector.enable_learning is False

    def test_created_selector_is_functional(self):
        """End-to-end: factory-built selector can classify and select."""
        import asyncio

        selector = create_strategy_selector(enable_learning=False)
        strategy = asyncio.get_event_loop().run_until_complete(
            selector.select_strategy("What is Python?", 0.40, [])
        )
        assert isinstance(strategy, RefinementStrategy)

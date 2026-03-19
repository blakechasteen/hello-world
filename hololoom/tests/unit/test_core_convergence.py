"""
Tests for convergence subsystem files:
- mcts_engine.py (MCTSNode, MCTSFluxCapacitor, MCTSConvergenceEngine)
- query_decomposition.py (ComplexityDetector, QueryDecomposer, factory)
- recursive_reasoner_enhanced.py (EnhancedRecursiveReasoner, factory)
- __init__.py (re-exports from engine.py)

Uses shimming for missing protocol types, same pattern as sibling test files.
"""

import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shim — inject missing protocol types before importing modules under test.
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


@dataclass
class DecompositionTree:
    root_query: str
    sub_queries: List[str]
    depth: int
    complexity_score: float
    synthesis_strategy: str


class ReasoningStrategy(Enum):
    DIRECT = "direct"
    VERIFY = "verify"
    REFINE = "refine"
    CRITIQUE = "critique"
    DECOMPOSE = "decompose"
    EXPLORE = "explore"
    HOFSTADTER = "hofstadter"
    ADAPTIVE = "adaptive"


class StopCondition(Enum):
    MAX_ITERATIONS = "max_iterations"
    QUALITY_THRESHOLD = "quality_threshold"
    NO_IMPROVEMENT = "no_improvement"
    CONVERGENCE = "convergence"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIME_BUDGET = "time_budget"


@dataclass
class ReasoningJournal:
    strategy_used: Optional[Any] = None
    traces: list = field(default_factory=list)


@dataclass
class RecursiveConfig:
    strategy: ReasoningStrategy = ReasoningStrategy.REFINE
    max_iterations: int = 5
    quality_threshold: float = 0.9
    min_improvement: float = 0.05
    time_budget_ms: Optional[float] = None
    cost_budget: Optional[float] = None
    enable_journal: bool = True
    parallel_explore_branches: int = 3
    hofstadter_depth: int = 3
    enable_decomposition: bool = True
    enable_learning: bool = True
    refinement_threshold: float = 0.75
    max_refinement_iterations: int = 5


@dataclass
class RecursiveResult:
    final_response: str
    final_confidence: float
    iterations: int
    refinement_steps: List[RefinementStep]
    convergence_reason: str
    total_latency_ms: float
    improvement_trajectory: List[float]
    journal: Any = None
    decomposition: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategySelectorProtocol:
    pass


class QueryDecomposerProtocol:
    pass


class ConvergenceDetectorProtocol:
    pass


_SHIM_NAMES = {
    "RefinementStrategy": RefinementStrategy,
    "RefinementStep": RefinementStep,
    "DecompositionTree": DecompositionTree,
    "ReasoningStrategy": ReasoningStrategy,
    "StopCondition": StopCondition,
    "ReasoningJournal": ReasoningJournal,
    "RecursiveConfig": RecursiveConfig,
    "RecursiveResult": RecursiveResult,
    "StrategySelectorProtocol": StrategySelectorProtocol,
    "QueryDecomposerProtocol": QueryDecomposerProtocol,
    "ConvergenceDetectorProtocol": ConvergenceDetectorProtocol,
}

_MOD_NAMES = (
    "hololoom.protocols.recursive_reasoning",
    "hololoom.core.protocols.recursive_reasoning",
)

# Save original state so we can restore on cleanup
_original_modules: dict[str, Any] = {}


def _patch_protocol_shim():
    """Inject stubs into protocol module slots.

    Try to load the real module first so other convergence modules that import
    from it get the full set of names. Only fall back to a minimal shim when the
    real module cannot be loaded (missing deps like fabric.spacetime).
    """
    # Try loading the real module if not already in sys.modules
    existing = sys.modules.get(_MOD_NAMES[0]) or sys.modules.get(_MOD_NAMES[1])
    if existing is None:
        try:
            import importlib
            existing = importlib.import_module("hololoom.core.protocols.recursive_reasoning")
        except (ImportError, Exception):
            existing = None

    for mod_name in _MOD_NAMES:
        _original_modules[mod_name] = sys.modules.get(mod_name)
        current = sys.modules.get(mod_name)
        if current is not None:
            for k, v in _SHIM_NAMES.items():
                if not hasattr(current, k):
                    setattr(current, k, v)
        elif existing is not None:
            for k, v in _SHIM_NAMES.items():
                if not hasattr(existing, k):
                    setattr(existing, k, v)
            sys.modules[mod_name] = existing
        else:
            shim = types.ModuleType(mod_name)
            for k, v in _SHIM_NAMES.items():
                setattr(shim, k, v)
            sys.modules[mod_name] = shim


def _unpatch_protocol_shim():
    """Restore sys.modules to pre-shim state."""
    for mod_name in _MOD_NAMES:
        original = _original_modules.get(mod_name)
        if original is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original


_patch_protocol_shim()

# Re-import shimmed types from the protocol module so this test uses the same
# class objects as the code under test (avoids isinstance mismatches when another
# test file loaded its own shim first).
import hololoom.core.protocols.recursive_reasoning as _proto_mod  # noqa: E402

RefinementStrategy = _proto_mod.RefinementStrategy  # noqa: F811
RefinementStep = _proto_mod.RefinementStep  # noqa: F811
RecursiveConfig = _proto_mod.RecursiveConfig  # noqa: F811
RecursiveResult = getattr(_proto_mod, "RecursiveResult", RecursiveResult)  # noqa: F811
ReasoningStrategy = _proto_mod.ReasoningStrategy  # noqa: F811
StopCondition = _proto_mod.StopCondition  # noqa: F811
ReasoningJournal = _proto_mod.ReasoningJournal  # noqa: F811


# Now safe to import modules under test
from hololoom.core.convergence.mcts_engine import (  # noqa: E402
    MCTSNode,
    MCTSFluxCapacitor,
    MCTSCollapseResult,
    MCTSConvergenceEngine,
)
from hololoom.core.convergence.query_decomposition import (  # noqa: E402
    ComplexityDetector,
    QueryDecomposer,
    create_query_decomposer,
)
from hololoom.core.convergence import (  # noqa: E402
    ConvergenceEngine,
    CollapseStrategy,
    CollapseResult,
    ThompsonBandit,
    create_convergence_engine,
)
from hololoom.core.convergence.drift_detector import (  # noqa: E402
    DriftDetector,
    DriftResult,
    create_drift_detector,
)
from hololoom.core.convergence.minimax_gate import (  # noqa: E402
    MinimaxGate,
    GateResult,
    create_minimax_gate,
)


# ============================================================================
# MCTSNode
# ============================================================================

class TestMCTSNode:
    def test_initial_state(self):
        node = MCTSNode(tool_idx=0)
        assert node.tool_idx == 0
        assert node.visits == 0
        assert node.value_sum == 0.0
        assert node.alpha == 1.0
        assert node.beta == 1.0

    def test_average_value_zero_visits(self):
        node = MCTSNode(tool_idx=0)
        assert node.average_value == 0.0

    def test_average_value_after_updates(self):
        node = MCTSNode(tool_idx=0)
        node.update(0.6)
        node.update(0.8)
        assert node.average_value == pytest.approx(0.7)

    def test_thompson_prior(self):
        node = MCTSNode(tool_idx=0, alpha=3.0, beta=1.0)
        assert node.thompson_prior == pytest.approx(0.75)

    def test_thompson_prior_uniform(self):
        node = MCTSNode(tool_idx=0)
        assert node.thompson_prior == pytest.approx(0.5)

    def test_ucb1_score_unvisited_is_inf(self):
        node = MCTSNode(tool_idx=0)
        assert node.ucb1_score() == float("inf")

    def test_ucb1_score_no_parent(self):
        node = MCTSNode(tool_idx=0)
        node.visits = 5
        node.value_sum = 3.0
        # No parent → returns average_value
        assert node.ucb1_score() == pytest.approx(0.6)

    def test_ucb1_score_with_parent(self):
        parent = MCTSNode(tool_idx=-1)
        parent.visits = 10
        child = MCTSNode(tool_idx=0, parent=parent)
        child.visits = 4
        child.value_sum = 2.0
        score = child.ucb1_score(exploration_constant=1.414)
        # exploitation = 0.5, exploration > 0
        assert score > 0.5

    def test_ucb1_score_parent_zero_visits(self):
        parent = MCTSNode(tool_idx=-1)
        parent.visits = 0
        child = MCTSNode(tool_idx=0, parent=parent)
        child.visits = 3
        child.value_sum = 1.5
        # parent.visits==0 → returns average_value
        assert child.ucb1_score() == pytest.approx(0.5)

    def test_thompson_sample_in_range(self):
        node = MCTSNode(tool_idx=0, alpha=2.0, beta=3.0)
        for _ in range(50):
            s = node.thompson_sample()
            assert 0.0 <= s <= 1.0

    def test_update_positive(self):
        node = MCTSNode(tool_idx=0)
        node.update(0.7)
        assert node.visits == 1
        assert node.value_sum == pytest.approx(0.7)
        assert node.alpha == pytest.approx(1.7)
        assert node.beta == pytest.approx(1.0)

    def test_update_negative(self):
        node = MCTSNode(tool_idx=0)
        node.update(-0.3)
        assert node.visits == 1
        assert node.value_sum == pytest.approx(-0.3)
        assert node.alpha == pytest.approx(1.0)
        assert node.beta == pytest.approx(1.3)

    def test_update_zero(self):
        node = MCTSNode(tool_idx=0)
        node.update(0.0)
        assert node.visits == 1
        # value 0 is not > 0, so beta gets abs(0) = 0
        assert node.alpha == pytest.approx(1.0)
        assert node.beta == pytest.approx(1.0)

    def test_is_leaf_no_children(self):
        node = MCTSNode(tool_idx=0)
        assert node.is_leaf() is True

    def test_is_leaf_with_children(self):
        node = MCTSNode(tool_idx=0)
        node.children.append(MCTSNode(tool_idx=1, parent=node))
        assert node.is_leaf() is False


# ============================================================================
# MCTSFluxCapacitor
# ============================================================================

class TestMCTSFluxCapacitor:
    def test_init_defaults(self):
        fc = MCTSFluxCapacitor(n_tools=3)
        assert fc.n_tools == 3
        assert fc.n_simulations == 100
        assert fc.total_simulations == 0
        assert len(fc.best_tool_history) == 0
        np.testing.assert_array_equal(fc.alpha_priors, np.ones(3))
        np.testing.assert_array_equal(fc.beta_priors, np.ones(3))

    def test_init_with_priors(self):
        priors = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        fc = MCTSFluxCapacitor(n_tools=3, thompson_priors=priors)
        np.testing.assert_array_equal(fc.alpha_priors, [3.0, 1.0, 2.0])
        np.testing.assert_array_equal(fc.beta_priors, [1.0, 3.0, 2.0])

    def test_search_returns_valid_tool(self):
        fc = MCTSFluxCapacitor(n_tools=4, n_simulations=20)
        tool_idx, confidence, stats = fc.search()
        assert 0 <= tool_idx < 4
        assert 0.0 < confidence <= 1.0
        assert "simulations" in stats
        assert "visit_counts" in stats
        assert len(stats["visit_counts"]) == 4
        assert sum(stats["visit_counts"]) == 20
        assert fc.total_simulations == 20

    def test_search_with_neural_probs(self):
        fc = MCTSFluxCapacitor(n_tools=3, n_simulations=30)
        neural = np.array([0.1, 0.8, 0.1])
        tool_idx, confidence, stats = fc.search(neural_probs=neural)
        assert 0 <= tool_idx < 3

    def test_search_with_features(self):
        fc = MCTSFluxCapacitor(n_tools=3, n_simulations=30)
        features = {"motifs": ["a", "b"]}
        tool_idx, confidence, stats = fc.search(features=features)
        assert 0 <= tool_idx < 3

    def test_search_with_neural_and_features(self):
        fc = MCTSFluxCapacitor(n_tools=2, n_simulations=20)
        tool_idx, _, _ = fc.search(
            neural_probs=np.array([0.9, 0.1]),
            features={"motifs": ["x"]},
        )
        assert 0 <= tool_idx < 2

    def test_search_records_history(self):
        fc = MCTSFluxCapacitor(n_tools=3, n_simulations=10)
        fc.search()
        fc.search()
        assert len(fc.best_tool_history) == 2
        assert fc.total_simulations == 20

    def test_update_priors_positive(self):
        fc = MCTSFluxCapacitor(n_tools=3)
        fc.update_priors(1, 0.5)
        assert fc.alpha_priors[1] == pytest.approx(1.5)
        assert fc.beta_priors[1] == pytest.approx(1.0)

    def test_update_priors_negative(self):
        fc = MCTSFluxCapacitor(n_tools=3)
        fc.update_priors(0, -0.3)
        assert fc.alpha_priors[0] == pytest.approx(1.0)
        assert fc.beta_priors[0] == pytest.approx(1.3)

    def test_get_statistics(self):
        fc = MCTSFluxCapacitor(n_tools=3, n_simulations=10)
        fc.search()
        stats = fc.get_statistics()
        assert stats["total_simulations"] == 10
        assert stats["n_decisions"] == 1
        assert len(stats["tool_distribution"]) == 3
        assert len(stats["alpha_priors"]) == 3
        assert len(stats["thompson_priors"]) == 3

    def test_stats_empty_before_search(self):
        fc = MCTSFluxCapacitor(n_tools=2, n_simulations=5)
        stats = fc.get_statistics()
        assert stats["total_simulations"] == 0
        assert stats["n_decisions"] == 0

    def test_heavily_biased_priors_converge(self):
        """With very strong priors, search should favor the dominant tool."""
        priors = np.array([[50.0, 1.0], [1.0, 50.0]])
        fc = MCTSFluxCapacitor(n_tools=2, n_simulations=50, thompson_priors=priors)
        tool_idx, confidence, _ = fc.search()
        # Tool 0 has alpha=50, beta=1 — extremely strong prior
        assert tool_idx == 0
        assert confidence > 0.5


# ============================================================================
# MCTSConvergenceEngine
# ============================================================================

class TestMCTSConvergenceEngine:
    def test_init(self):
        engine = MCTSConvergenceEngine(
            tools=["search", "summarize", "respond"],
            n_simulations=20,
        )
        assert engine.n_tools == 3
        assert engine.decision_count == 0
        assert all(v == 0 for v in engine.tool_usage.values())

    def test_collapse_returns_result(self):
        engine = MCTSConvergenceEngine(
            tools=["a", "b", "c"],
            n_simulations=15,
        )
        result = engine.collapse()
        assert isinstance(result, MCTSCollapseResult)
        assert result.tool in ["a", "b", "c"]
        assert 0 <= result.tool_idx < 3
        assert 0.0 < result.confidence <= 1.0
        assert "mcts_15_sims" == result.strategy_used
        assert len(result.visit_counts) == 3
        assert engine.decision_count == 1
        assert engine.tool_usage[result.tool] == 1

    def test_collapse_with_neural_probs(self):
        engine = MCTSConvergenceEngine(tools=["x", "y"], n_simulations=10)
        result = engine.collapse(neural_probs=np.array([0.9, 0.1]))
        assert result.tool in ["x", "y"]

    def test_collapse_with_features(self):
        engine = MCTSConvergenceEngine(tools=["a", "b"], n_simulations=10)
        result = engine.collapse(features={"motifs": ["m1"]})
        assert result.tool in ["a", "b"]

    def test_update_from_outcome(self):
        engine = MCTSConvergenceEngine(tools=["a", "b"], n_simulations=10)
        engine.update_from_outcome(0, 0.8)
        assert engine.flux.alpha_priors[0] == pytest.approx(1.8)

    def test_get_statistics(self):
        engine = MCTSConvergenceEngine(tools=["a", "b"], n_simulations=10)
        engine.collapse()
        stats = engine.get_statistics()
        assert stats["decision_count"] == 1
        assert "tool_usage" in stats
        assert "flux_stats" in stats

    def test_multiple_collapses_track_usage(self):
        engine = MCTSConvergenceEngine(tools=["a", "b"], n_simulations=10)
        for _ in range(5):
            engine.collapse()
        assert engine.decision_count == 5
        assert sum(engine.tool_usage.values()) == 5


# ============================================================================
# ComplexityDetector
# ============================================================================

class TestComplexityDetector:
    def setup_method(self):
        self.detector = ComplexityDetector()

    def test_simple_query_low_score(self):
        score = self.detector.detect("What is X?")
        assert score < 0.3

    def test_long_query_higher_score(self):
        long_q = " ".join(["word"] * 60)
        score = self.detector.detect(long_q)
        # Length alone contributes up to 0.3
        assert score >= 0.3

    def test_conjunctions_increase_score(self):
        base = self.detector.detect("Tell me about X")
        conj = self.detector.detect("Tell me about X and Y and Z also W")
        assert conj > base

    def test_multiple_questions_increase_score(self):
        single = self.detector.detect("What is X?")
        multi = self.detector.detect("What is X? How does Y work? Why is Z important?")
        assert multi > single

    def test_complexity_keywords_increase_score(self):
        plain = self.detector.detect("Tell me about X")
        complex_q = self.detector.detect(
            "Compare and analyze the tradeoffs and implications of X"
        )
        assert complex_q > plain

    def test_nesting_increases_score(self):
        plain = self.detector.detect("Tell me about X")
        nested = self.detector.detect("Tell me about X (including Y); also Z")
        assert nested > plain

    def test_score_capped_at_1(self):
        # Throw everything at it
        monster = (
            "Compare and analyze the tradeoffs and implications of X and Y "
            "and Z and also furthermore? What is it? How does it work? Why? "
            "(nested (deeply)); more; " + " ".join(["word"] * 60)
        )
        score = self.detector.detect(monster)
        assert score <= 1.0

    def test_empty_query(self):
        score = self.detector.detect("")
        assert score == pytest.approx(0.0)


# ============================================================================
# QueryDecomposer
# ============================================================================

class TestQueryDecomposer:
    def setup_method(self):
        self.decomposer = QueryDecomposer(max_depth=3, max_sub_queries=5)

    def test_detect_complexity(self):
        score = self.decomposer.detect_complexity("What is X?")
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_simple_query_no_decomposition(self):
        tree = await self.decomposer.decompose("What is X?", complexity_threshold=0.9)
        assert tree.root_query == "What is X?"
        assert tree.sub_queries == ["What is X?"]
        assert tree.synthesis_strategy == "direct"
        assert tree.depth == 1

    @pytest.mark.asyncio
    async def test_conjunction_splitting(self):
        query = "What is machine learning and how does deep learning work and why is regularization important"
        tree = await self.decomposer.decompose(query, complexity_threshold=0.1)
        # Should split on "and"
        assert len(tree.sub_queries) > 1

    @pytest.mark.asyncio
    async def test_multiple_questions_extraction(self):
        query = "What is X? How does Y work? Why is Z important?"
        tree = await self.decomposer.decompose(query, complexity_threshold=0.1)
        assert len(tree.sub_queries) >= 2

    @pytest.mark.asyncio
    async def test_comparison_splitting_via_split_comparison(self):
        # _split_comparison is called when "compare" present but no " and "/" or " —
        # however _decompose_strategy checks conjunctions first. Test _split_comparison directly.
        result = self.decomposer._split_comparison("Compare apples and oranges")
        assert len(result) == 3  # What is X? What is Y? How do X and Y differ?

    @pytest.mark.asyncio
    async def test_vs_splitting(self):
        # "vs" triggers comparison path (no "and"/"or" conjunction)
        query = "Python vs Java for backend development"
        tree = await self.decomposer.decompose(query, complexity_threshold=0.1)
        assert len(tree.sub_queries) >= 2

    def test_versus_splitting_direct(self):
        """Test _split_comparison handles 'versus' correctly."""
        result = self.decomposer._split_comparison("Python versus Java")
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_comparison_synthesis_strategy(self):
        """Queries with 'compare' get comparison_synthesis strategy."""
        # "compare" without "and" so conjunction split doesn't fire first
        query = "Compare these two approaches: microservices, monolith; furthermore evaluate tradeoffs"
        tree = await self.decomposer.decompose(query, complexity_threshold=0.1)
        assert tree.synthesis_strategy == "comparison_synthesis"

    @pytest.mark.asyncio
    async def test_aspect_based_decomposition(self):
        query = "Analyze the impact of containerization on modern deployments with many implications"
        tree = await self.decomposer.decompose(query, complexity_threshold=0.1)
        assert len(tree.sub_queries) >= 2
        assert tree.synthesis_strategy == "analytical_synthesis"

    @pytest.mark.asyncio
    async def test_hierarchical_fallback(self):
        # A complex query that doesn't match other strategies
        query = (
            "Tell me everything about the architectural evolution of microservices "
            "including the history context implications tradeoffs furthermore also"
        )
        tree = await self.decomposer.decompose(query, complexity_threshold=0.1)
        assert len(tree.sub_queries) >= 2

    @pytest.mark.asyncio
    async def test_max_sub_queries_limit(self):
        decomposer = QueryDecomposer(max_sub_queries=2)
        query = "What is A and what is B and what is C and what is D"
        tree = await decomposer.decompose(query, complexity_threshold=0.1)
        assert len(tree.sub_queries) <= 2

    def test_split_conjunctions_short_parts_filtered(self):
        # Parts with < 3 words are filtered out
        result = self.decomposer._split_conjunctions("A and B and explain the full concept")
        # "A" and "B" should be filtered; only longer parts kept
        for sq in result:
            if sq != "A and B and explain the full concept":
                assert len(sq.split()) >= 3 or sq.endswith("?")

    def test_extract_questions_single(self):
        result = self.decomposer._extract_questions("What is X?")
        assert result == ["What is X?"]

    def test_extract_questions_multiple(self):
        result = self.decomposer._extract_questions("What is X? How about Y?")
        assert len(result) == 2

    def test_extract_topic(self):
        topic = self.decomposer._extract_topic("What is the meaning of recursion?")
        assert len(topic) > 0
        # Question words and "is/are/the" should be stripped
        assert "what" not in topic.lower().split()

    def test_extract_topic_empty(self):
        topic = self.decomposer._extract_topic("is the?")
        # Should return the original query as fallback when all words stripped
        assert len(topic) > 0

    def test_select_synthesis_strategy_compare(self):
        strategy = self.decomposer._select_synthesis_strategy("Compare X and Y", ["q1", "q2"])
        assert strategy == "comparison_synthesis"

    def test_select_synthesis_strategy_vs(self):
        strategy = self.decomposer._select_synthesis_strategy("X vs Y", ["q1", "q2"])
        assert strategy == "comparison_synthesis"

    def test_select_synthesis_strategy_analyze(self):
        strategy = self.decomposer._select_synthesis_strategy("Analyze the impact", ["q1", "q2"])
        assert strategy == "analytical_synthesis"

    def test_select_synthesis_strategy_many_subqueries(self):
        strategy = self.decomposer._select_synthesis_strategy("something", ["q"] * 4)
        assert strategy == "hierarchical_synthesis"

    def test_select_synthesis_strategy_default(self):
        strategy = self.decomposer._select_synthesis_strategy("something", ["q1", "q2"])
        assert strategy == "sequential_synthesis"

    # --- Synthesis ---

    @pytest.mark.asyncio
    async def test_synthesize_sequential(self):
        result = await self.decomposer.synthesize_sub_answers(
            ["Answer 1", "Answer 2"],
            synthesis_strategy="sequential_synthesis",
        )
        assert "Part 1" in result
        assert "Part 2" in result

    @pytest.mark.asyncio
    async def test_synthesize_comparison_enough_answers(self):
        result = await self.decomposer.synthesize_sub_answers(
            ["Entity A info", "Entity B info", "Differences"],
            synthesis_strategy="comparison_synthesis",
        )
        assert "Entity 1" in result
        assert "Entity 2" in result
        assert "Key Differences" in result

    @pytest.mark.asyncio
    async def test_synthesize_comparison_too_few_falls_back(self):
        result = await self.decomposer.synthesize_sub_answers(
            ["Only one"],
            synthesis_strategy="comparison_synthesis",
        )
        # Falls back to sequential
        assert "Part 1" in result

    @pytest.mark.asyncio
    async def test_synthesize_analytical(self):
        result = await self.decomposer.synthesize_sub_answers(
            ["Overview text", "Advantages text", "Disadvantages text", "Use cases text"],
            synthesis_strategy="analytical_synthesis",
        )
        assert "Overview" in result
        assert "Advantages" in result

    @pytest.mark.asyncio
    async def test_synthesize_hierarchical_enough(self):
        result = await self.decomposer.synthesize_sub_answers(
            ["Overview", "Details", "Examples"],
            synthesis_strategy="hierarchical_synthesis",
        )
        assert "Overview" in result
        assert "Details" in result
        assert "Examples" in result

    @pytest.mark.asyncio
    async def test_synthesize_hierarchical_too_few_falls_back(self):
        result = await self.decomposer.synthesize_sub_answers(
            ["Only one"],
            synthesis_strategy="hierarchical_synthesis",
        )
        assert "Part 1" in result

    @pytest.mark.asyncio
    async def test_synthesize_with_objects_having_response_attr(self):
        """Sub-results with .response attribute are handled."""
        obj = MagicMock()
        obj.response = "Mocked response"
        result = await self.decomposer.synthesize_sub_answers(
            [obj, "plain string"],
            synthesis_strategy="sequential_synthesis",
        )
        assert "Mocked response" in result
        assert "plain string" in result

    @pytest.mark.asyncio
    async def test_synthesize_with_non_string_objects(self):
        result = await self.decomposer.synthesize_sub_answers(
            [42, None],
            synthesis_strategy="sequential_synthesis",
        )
        assert "42" in result
        assert "None" in result


# ============================================================================
# create_query_decomposer factory
# ============================================================================

class TestCreateQueryDecomposer:
    def test_creates_instance(self):
        d = create_query_decomposer()
        assert isinstance(d, QueryDecomposer)

    def test_custom_params(self):
        d = create_query_decomposer(max_depth=2, max_sub_queries=3)
        assert d.max_depth == 2
        assert d.max_sub_queries == 3

    def test_weaving_fn_stored(self):
        fn = lambda q: q
        d = create_query_decomposer(weaving_fn=fn)
        assert d.weaving_fn is fn


# ============================================================================
# EnhancedRecursiveReasoner
# ============================================================================

class TestEnhancedRecursiveReasoner:
    """Tests for recursive_reasoner_enhanced.py.

    The reasoner depends on a department (LLM backend) which we fully mock.
    """

    def _make_department_result(self, answer="test answer", confidence=0.85):
        """Create a mock department execution result."""
        result = MagicMock()
        result.response = {"answer": answer}
        result.confidence = MagicMock()
        result.confidence.score = confidence
        return result

    def _make_department(self, results=None):
        """Create a mock department that returns sequential results."""
        dept = MagicMock()
        if results is None:
            results = [self._make_department_result()]
        dept.execute = AsyncMock(side_effect=results)
        return dept

    @pytest.mark.asyncio
    async def test_reason_no_decomposition_high_confidence(self):
        """High initial confidence -> no refinement needed."""
        from hololoom.core.convergence.recursive_reasoner_enhanced import (
            EnhancedRecursiveReasoner,
        )

        dept = self._make_department([self._make_department_result(confidence=0.9)])
        config = RecursiveConfig(
            enable_decomposition=False,
            enable_learning=False,
            refinement_threshold=0.75,
        )
        reasoner = EnhancedRecursiveReasoner(department=dept, config=config)
        result = await reasoner.reason("What is X?")

        assert isinstance(result, RecursiveResult)
        assert result.final_confidence == pytest.approx(0.9)
        assert result.iterations == 1
        assert dept.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_reason_refinement_loop(self):
        """Low initial confidence triggers refinement."""
        from hololoom.core.convergence.recursive_reasoner_enhanced import (
            EnhancedRecursiveReasoner,
        )

        results = [
            self._make_department_result(confidence=0.5),   # initial
            self._make_department_result(confidence=0.7),   # refinement 1
            self._make_department_result(confidence=0.85),  # refinement 2
        ]
        dept = self._make_department(results)
        config = RecursiveConfig(
            enable_decomposition=False,
            enable_learning=False,
            refinement_threshold=0.75,
            max_refinement_iterations=5,
            quality_threshold=0.8,
            min_improvement=0.01,
        )
        reasoner = EnhancedRecursiveReasoner(department=dept, config=config)
        result = await reasoner.reason("What is X?")

        assert result.iterations >= 2
        assert result.final_confidence >= 0.7

    @pytest.mark.asyncio
    async def test_reason_with_decomposition(self):
        """Complex query with decomposition enabled should decompose."""
        from hololoom.core.convergence.recursive_reasoner_enhanced import (
            EnhancedRecursiveReasoner,
        )

        # The decomposer needs a complex enough query
        results = [
            self._make_department_result(answer="sub1", confidence=0.8),
            self._make_department_result(answer="sub2", confidence=0.7),
            self._make_department_result(answer="sub3", confidence=0.9),
        ]
        dept = self._make_department(results)
        config = RecursiveConfig(
            enable_decomposition=True,
            enable_learning=False,
        )

        # Mock the decomposer to always decompose
        mock_decomposer = MagicMock()
        mock_decomposition = DecompositionTree(
            root_query="Compare X and Y",
            sub_queries=["What is X?", "What is Y?", "How do X and Y differ?"],
            depth=1,
            complexity_score=0.8,
            synthesis_strategy="comparison_synthesis",
        )
        mock_decomposer.decompose = AsyncMock(return_value=mock_decomposition)
        mock_decomposer.synthesize_sub_answers = AsyncMock(return_value="Synthesized answer")

        reasoner = EnhancedRecursiveReasoner(
            department=dept,
            config=config,
            decomposer=mock_decomposer,
        )
        result = await reasoner.reason("Compare X and Y")

        assert result.final_response == "Synthesized answer"
        assert dept.execute.call_count == 3  # 3 sub-queries

    @pytest.mark.asyncio
    async def test_reason_time_budget_exceeded(self):
        """Time budget should stop refinement."""
        from hololoom.core.convergence.recursive_reasoner_enhanced import (
            EnhancedRecursiveReasoner,
        )

        results = [
            self._make_department_result(confidence=0.4),
            self._make_department_result(confidence=0.45),
        ]
        dept = self._make_department(results)
        config = RecursiveConfig(
            enable_decomposition=False,
            enable_learning=False,
            refinement_threshold=0.75,
            time_budget_ms=0.001,  # Effectively zero
            max_refinement_iterations=10,
        )
        reasoner = EnhancedRecursiveReasoner(department=dept, config=config)
        result = await reasoner.reason("What is X?")

        # Should stop early due to time budget
        assert "time_budget" in result.convergence_reason.lower() or result.iterations <= 2

    @pytest.mark.asyncio
    async def test_build_result_metadata(self):
        """_build_result produces correct metadata."""
        from hololoom.core.convergence.recursive_reasoner_enhanced import (
            EnhancedRecursiveReasoner,
        )

        dept = self._make_department([self._make_department_result(confidence=0.9)])
        config = RecursiveConfig(enable_decomposition=False, enable_learning=False, refinement_threshold=0.75)
        reasoner = EnhancedRecursiveReasoner(department=dept, config=config)
        result = await reasoner.reason("What is X?")

        assert "query" in result.metadata
        assert "stop_condition" in result.metadata
        assert "initial_confidence" in result.metadata
        assert "final_confidence" in result.metadata
        assert result.total_latency_ms >= 0


# ============================================================================
# create_recursive_reasoner factory
# ============================================================================

class TestCreateRecursiveReasoner:
    def test_factory_creates_instance(self):
        from hololoom.core.convergence.recursive_reasoner_enhanced import (
            create_recursive_reasoner,
        )

        dept = MagicMock()
        reasoner = create_recursive_reasoner(dept)
        assert hasattr(reasoner, "reason")

    def test_factory_with_config(self):
        from hololoom.core.convergence.recursive_reasoner_enhanced import (
            create_recursive_reasoner,
        )

        dept = MagicMock()
        config = RecursiveConfig(max_iterations=3)
        reasoner = create_recursive_reasoner(dept, config=config)
        assert reasoner.config.max_iterations == 3

    def test_factory_without_config_sets_flags(self):
        from hololoom.core.convergence.recursive_reasoner_enhanced import (
            create_recursive_reasoner,
        )

        dept = MagicMock()
        reasoner = create_recursive_reasoner(dept, enable_decomposition=False, enable_learning=False)
        assert reasoner.config.enable_decomposition is False
        assert reasoner.config.enable_learning is False


# ============================================================================
# __init__.py re-exports
# ============================================================================

# ============================================================================
# ThompsonBandit (engine.py)
# ============================================================================

class TestThompsonBandit:
    def test_init(self):
        bandit = ThompsonBandit(n_tools=4)
        assert bandit.n_tools == 4
        np.testing.assert_array_equal(bandit.successes, np.ones(4))
        np.testing.assert_array_equal(bandit.failures, np.ones(4))
        np.testing.assert_array_equal(bandit.pulls, np.zeros(4))

    def test_sample_returns_valid_index(self):
        bandit = ThompsonBandit(n_tools=5)
        for _ in range(20):
            idx = bandit.sample()
            assert 0 <= idx < 5
        # Pulls should be tracked
        assert bandit.pulls.sum() == 20

    def test_get_priors_uniform(self):
        bandit = ThompsonBandit(n_tools=3)
        priors = bandit.get_priors()
        np.testing.assert_array_almost_equal(priors, [0.5, 0.5, 0.5])

    def test_update_positive_reward(self):
        bandit = ThompsonBandit(n_tools=3)
        bandit.update(0, 0.8)
        assert bandit.successes[0] == pytest.approx(1.8)
        assert bandit.total_reward[0] == pytest.approx(0.8)

    def test_update_negative_reward(self):
        bandit = ThompsonBandit(n_tools=3)
        bandit.update(1, -0.5)
        assert bandit.failures[1] == pytest.approx(1.5)
        assert bandit.total_reward[1] == pytest.approx(-0.5)

    def test_get_statistics(self):
        bandit = ThompsonBandit(n_tools=2)
        bandit.update(0, 1.0)
        stats = bandit.get_statistics()
        assert "tool_priors" in stats
        assert "pulls" in stats
        assert "total_rewards" in stats
        assert "success_counts" in stats
        assert "failure_counts" in stats
        assert len(stats["tool_priors"]) == 2

    def test_priors_shift_with_updates(self):
        bandit = ThompsonBandit(n_tools=2)
        for _ in range(10):
            bandit.update(0, 1.0)
        priors = bandit.get_priors()
        assert priors[0] > priors[1]


# ============================================================================
# ConvergenceEngine (engine.py)
# ============================================================================

class TestConvergenceEngine:
    def test_init(self):
        engine = ConvergenceEngine(tools=["a", "b", "c"])
        assert engine.n_tools == 3
        assert engine.default_strategy == CollapseStrategy.EPSILON_GREEDY
        assert len(engine.collapse_history) == 0

    def test_collapse_argmax(self):
        engine = ConvergenceEngine(tools=["a", "b", "c"], entropy_temperature=0.0)
        probs = np.array([0.1, 0.8, 0.1])
        result = engine.collapse(probs, strategy=CollapseStrategy.ARGMAX, inject_entropy=False)
        assert result.tool == "b"
        assert result.tool_idx == 1
        assert "argmax" in result.strategy_used

    def test_collapse_epsilon_greedy_exploit(self):
        engine = ConvergenceEngine(tools=["a", "b"], epsilon=0.0, entropy_temperature=0.0)
        probs = np.array([0.9, 0.1])
        result = engine.collapse(probs, strategy=CollapseStrategy.EPSILON_GREEDY, inject_entropy=False)
        assert result.tool == "a"
        assert "exploit" in result.strategy_used

    def test_collapse_epsilon_greedy_explore(self):
        engine = ConvergenceEngine(tools=["a", "b"], epsilon=1.0, entropy_temperature=0.0)
        probs = np.array([0.9, 0.1])
        result = engine.collapse(probs, strategy=CollapseStrategy.EPSILON_GREEDY, inject_entropy=False)
        # epsilon=1.0 always explores via Thompson
        assert "explore" in result.strategy_used

    def test_collapse_bayesian_blend(self):
        engine = ConvergenceEngine(tools=["a", "b", "c"], entropy_temperature=0.0)
        probs = np.array([0.6, 0.3, 0.1])
        result = engine.collapse(probs, strategy=CollapseStrategy.BAYESIAN_BLEND, inject_entropy=False)
        assert result.tool in ["a", "b", "c"]
        assert "bayesian_blend" in result.strategy_used

    def test_collapse_pure_thompson(self):
        engine = ConvergenceEngine(tools=["a", "b"], entropy_temperature=0.0)
        probs = np.array([0.5, 0.5])
        result = engine.collapse(probs, strategy=CollapseStrategy.PURE_THOMPSON, inject_entropy=False)
        assert result.tool in ["a", "b"]
        assert "pure_thompson" in result.strategy_used

    def test_collapse_records_history(self):
        engine = ConvergenceEngine(tools=["a", "b"], entropy_temperature=0.0)
        probs = np.array([0.5, 0.5])
        engine.collapse(probs, inject_entropy=False)
        engine.collapse(probs, inject_entropy=False)
        assert len(engine.collapse_history) == 2

    def test_inject_entropy_zero_temp(self):
        engine = ConvergenceEngine(tools=["a"], entropy_temperature=0.0)
        probs = np.array([1.0])
        result = engine.inject_entropy(probs, temperature=0)
        np.testing.assert_array_equal(result, probs)

    def test_inject_entropy_nonzero_temp(self):
        engine = ConvergenceEngine(tools=["a", "b", "c"], entropy_temperature=0.5)
        probs = np.array([0.5, 0.3, 0.2])
        noisy = engine.inject_entropy(probs)
        # Should still sum to 1
        assert noisy.sum() == pytest.approx(1.0)
        assert len(noisy) == 3

    def test_collapse_with_entropy_injection(self):
        engine = ConvergenceEngine(tools=["a", "b"], entropy_temperature=0.1)
        probs = np.array([0.8, 0.2])
        result = engine.collapse(probs, inject_entropy=True)
        assert result.tool in ["a", "b"]

    def test_update_from_outcome_success(self):
        engine = ConvergenceEngine(tools=["a", "b"])
        engine.update_from_outcome(0, success=True)
        assert engine.bandit.successes[0] == pytest.approx(2.0)

    def test_update_from_outcome_failure(self):
        engine = ConvergenceEngine(tools=["a", "b"])
        engine.update_from_outcome(1, success=False)
        assert engine.bandit.failures[1] == pytest.approx(2.0)

    def test_update_from_outcome_explicit_reward(self):
        engine = ConvergenceEngine(tools=["a", "b"])
        engine.update_from_outcome(0, success=True, reward=0.5)
        assert engine.bandit.successes[0] == pytest.approx(1.5)

    def test_get_trace(self):
        engine = ConvergenceEngine(tools=["a", "b"], entropy_temperature=0.0)
        engine.collapse(np.array([0.5, 0.5]), inject_entropy=False)
        trace = engine.get_trace()
        assert trace["total_collapses"] == 1
        assert "tools" in trace
        assert "bandit_stats" in trace
        assert len(trace["recent_collapses"]) == 1

    def test_bregman_blend_weight(self):
        engine = ConvergenceEngine(tools=["a", "b"])
        # When neural and bandit agree perfectly, weight should be close to 1.0
        w = engine._bregman_blend_weight(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        assert 0.3 <= w <= 0.9

    def test_bregman_blend_weight_disagreement(self):
        engine = ConvergenceEngine(tools=["a", "b"])
        # When they disagree strongly
        w = engine._bregman_blend_weight(np.array([0.99, 0.01]), np.array([0.01, 0.99]))
        assert 0.3 <= w <= 0.9
        # Should be lower weight than agreement
        w_agree = engine._bregman_blend_weight(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        assert w < w_agree

    @patch.dict("os.environ", {"HOLOLOOM_BREGMAN_BLEND": "true"})
    def test_collapse_bayesian_blend_with_bregman(self):
        """Test Bregman adaptive blending when feature flag is on."""
        # Reimport to pick up env var — but the flag is set at module level.
        # Instead, test the method directly.
        engine = ConvergenceEngine(tools=["a", "b"])
        w = engine._bregman_blend_weight(np.array([0.7, 0.3]), np.array([0.6, 0.4]))
        assert 0.3 <= w <= 0.9


# ============================================================================
# ConvergenceEngine factory
# ============================================================================

class TestCreateConvergenceEngineFactory:
    def test_creates_engine(self):
        engine = create_convergence_engine(tools=["a", "b", "c"])
        assert isinstance(engine, ConvergenceEngine)

    def test_custom_strategy(self):
        engine = create_convergence_engine(
            tools=["a", "b"],
            strategy=CollapseStrategy.ARGMAX,
        )
        assert engine.default_strategy == CollapseStrategy.ARGMAX

    def test_custom_epsilon(self):
        engine = create_convergence_engine(tools=["a"], epsilon=0.5)
        assert engine.epsilon == pytest.approx(0.5)


# ============================================================================
# __init__.py re-exports
# ============================================================================

class TestConvergenceInit:
    def test_convergence_engine_importable(self):
        assert ConvergenceEngine is not None

    def test_collapse_strategy_importable(self):
        assert CollapseStrategy is not None

    def test_collapse_result_importable(self):
        assert CollapseResult is not None

    def test_thompson_bandit_importable(self):
        assert ThompsonBandit is not None

    def test_create_convergence_engine_importable(self):
        assert callable(create_convergence_engine)


# ============================================================================
# Additional refinement_strategies coverage
# ============================================================================

class TestRefinementStrategiesExtra:
    """Cover uncovered paths in refinement_strategies.py via the reasoner integration."""

    def setup_method(self):
        from hololoom.core.convergence.refinement_strategies import (
            StrategySelector,
            StrategyExecutor,
            QueryClassifier,
        )
        self.StrategySelector = StrategySelector
        self.StrategyExecutor = StrategyExecutor
        self.QueryClassifier = QueryClassifier

    @pytest.mark.asyncio
    async def test_thompson_path_with_history(self):
        """Exercise the Thompson Sampling selection path (line 199+)."""
        selector = self.StrategySelector(enable_learning=True)
        # Pre-load some learning data
        await selector.learn_from_outcome(
            RefinementStrategy.EXPAND_SEARCH, 0.2, "factual"
        )
        # Now call with non-empty history to trigger Thompson path
        history = [RefinementStep(1, "q", "r", 0.5, 0.0, "initial", "test")]
        strategy = await selector.select_strategy("What is Python?", 0.6, history)
        assert isinstance(strategy, RefinementStrategy)

    @pytest.mark.asyncio
    async def test_learn_updates_stats(self):
        selector = self.StrategySelector(enable_learning=True)
        await selector.learn_from_outcome(RefinementStrategy.RERANK, 0.1, "factual")
        await selector.learn_from_outcome(RefinementStrategy.RERANK, -0.05, "factual")
        key = ("factual", RefinementStrategy.RERANK)
        assert selector.strategy_stats[key].total_uses == 2
        assert selector.strategy_stats[key].successful_uses == 1

    @pytest.mark.asyncio
    async def test_learn_disabled_noop(self):
        selector = self.StrategySelector(enable_learning=False)
        await selector.learn_from_outcome(RefinementStrategy.DECOMPOSE, 0.5, "analytical")
        key = ("analytical", RefinementStrategy.DECOMPOSE)
        assert selector.strategy_stats[key].total_uses == 0

    def test_get_statistics_aggregated(self):
        import asyncio
        selector = self.StrategySelector(enable_learning=True)
        asyncio.get_event_loop().run_until_complete(
            selector.learn_from_outcome(RefinementStrategy.EXPAND_SEARCH, 0.1, "factual")
        )
        asyncio.get_event_loop().run_until_complete(
            selector.learn_from_outcome(RefinementStrategy.EXPAND_SEARCH, 0.2, "technical")
        )
        stats = selector.get_strategy_statistics()
        assert stats["query_type"] == "all"
        assert "expand_search" in stats["strategies"]
        assert stats["strategies"]["expand_search"]["total_uses"] == 2

    def test_get_statistics_filtered(self):
        import asyncio
        selector = self.StrategySelector(enable_learning=True)
        asyncio.get_event_loop().run_until_complete(
            selector.learn_from_outcome(RefinementStrategy.RERANK, 0.15, "factual")
        )
        stats = selector.get_strategy_statistics(query_type="factual")
        assert stats["query_type"] == "factual"

    def test_get_best_strategy(self):
        import asyncio
        selector = self.StrategySelector(enable_learning=True)
        for _ in range(10):
            asyncio.get_event_loop().run_until_complete(
                selector.learn_from_outcome(RefinementStrategy.VERIFY_AND_CORRECT, 0.15, "analytical")
            )
        best, reward = selector.get_best_strategy_for_type("analytical")
        assert best == RefinementStrategy.VERIFY_AND_CORRECT
        assert reward > 0.5

    def test_executor_all_strategies(self):
        for strategy in RefinementStrategy:
            params = self.StrategyExecutor.get_execution_params(strategy)
            assert isinstance(params, dict)
            assert "max_sources" in params


# ============================================================================
# DriftDetector (drift_detector.py)
# ============================================================================

class TestDriftDetector:
    def test_init(self):
        detector = DriftDetector(threshold=0.4)
        assert detector.threshold == 0.4

    def test_check_no_context(self):
        detector = DriftDetector()
        result = detector.check(np.array([1.0, 0.0, 0.0]), [])
        assert isinstance(result, DriftResult)
        assert result.drifted is False
        assert result.context_size == 0

    def test_check_identical_embeddings_no_drift(self):
        detector = DriftDetector(threshold=0.3)
        emb = np.array([0.5, 0.3, 0.2])
        result = detector.check(emb, [emb.copy()])
        assert result.drifted is False
        assert result.divergence < 0.01
        assert result.context_size == 1

    def test_check_very_different_embeddings_drift(self):
        detector = DriftDetector(threshold=0.01)
        response = np.array([10.0, 0.0, 0.0, 0.0])
        context = [np.array([0.0, 0.0, 0.0, 10.0])]
        result = detector.check(response, context)
        assert result.drifted is True
        assert result.divergence > 0.01

    def test_check_multiple_context_embeddings(self):
        detector = DriftDetector(threshold=0.3)
        response = np.array([0.5, 0.3, 0.2])
        contexts = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.4, 0.4, 0.2]),
        ]
        result = detector.check(response, contexts)
        assert result.context_size == 2
        assert isinstance(result.divergence, float)

    def test_check_dimension_mismatch(self):
        detector = DriftDetector()
        response = np.array([0.5, 0.3])
        context = [np.array([0.5, 0.3, 0.2])]  # Different dim
        result = detector.check(response, context)
        assert result.drifted is False
        assert "mismatch" in result.detail.lower()

    def test_softmax(self):
        detector = DriftDetector()
        probs = detector._softmax(np.array([1.0, 2.0, 3.0]))
        assert probs.sum() == pytest.approx(1.0)
        # Highest input → highest probability
        assert probs[2] > probs[1] > probs[0]

    def test_jsd_identical(self):
        detector = DriftDetector()
        p = np.array([0.5, 0.3, 0.2])
        assert detector._jsd(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_jsd_different(self):
        detector = DriftDetector()
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        jsd = detector._jsd(p, q)
        assert jsd > 0.0
        assert jsd <= 1.0

    def test_factory(self):
        detector = create_drift_detector(threshold=0.5)
        assert isinstance(detector, DriftDetector)
        assert detector.threshold == 0.5


# ============================================================================
# MinimaxGate (minimax_gate.py)
# ============================================================================

class TestMinimaxGate:
    def test_init(self):
        gate = MinimaxGate(regret_threshold=0.1)
        assert gate.regret_threshold == 0.1

    def test_verify_matching_selection(self):
        gate = MinimaxGate()
        # Tool 0 has best worst-case payoff
        payoffs = np.array([
            [0.5, 0.4],  # tool 0: worst = 0.4
            [0.8, 0.1],  # tool 1: worst = 0.1
        ])
        result = gate.verify(selected_tool=0, tool_payoffs=payoffs)
        assert result.verified is True
        assert result.minimax_tool == 0
        assert result.regret == pytest.approx(0.0)

    def test_verify_divergent_selection(self):
        gate = MinimaxGate()
        payoffs = np.array([
            [0.5, 0.4],  # tool 0: worst = 0.4 (minimax optimal)
            [0.8, 0.1],  # tool 1: worst = 0.1
        ])
        result = gate.verify(selected_tool=1, tool_payoffs=payoffs)
        assert result.verified is False
        assert result.minimax_tool == 0
        assert result.ts_tool == 1
        assert result.regret == pytest.approx(0.3)

    def test_verify_with_regret_threshold(self):
        gate = MinimaxGate(regret_threshold=0.5)
        payoffs = np.array([
            [0.5, 0.4],  # worst = 0.4
            [0.8, 0.1],  # worst = 0.1
        ])
        # Regret = 0.3 < threshold 0.5 → verified
        result = gate.verify(selected_tool=1, tool_payoffs=payoffs)
        assert result.verified is True

    def test_verify_invalid_matrix(self):
        gate = MinimaxGate()
        result = gate.verify(selected_tool=0, tool_payoffs=np.array([1, 2, 3]))
        assert result.verified is True
        assert "Invalid" in result.detail

    def test_verify_tool_out_of_range(self):
        gate = MinimaxGate()
        payoffs = np.array([[0.5, 0.4]])
        result = gate.verify(selected_tool=5, tool_payoffs=payoffs)
        assert result.verified is True
        assert "out of range" in result.detail

    def test_verify_negative_tool_index(self):
        gate = MinimaxGate()
        payoffs = np.array([[0.5, 0.4]])
        result = gate.verify(selected_tool=-1, tool_payoffs=payoffs)
        assert result.verified is True

    def test_verify_with_state_priors(self):
        gate = MinimaxGate()
        payoffs = np.array([
            [0.5, 0.4],
            [0.8, 0.1],
        ])
        priors = np.array([0.6, 0.4])
        result = gate.verify(selected_tool=0, tool_payoffs=payoffs, state_priors=priors)
        assert result.verified is True
        assert "expected under priors" in result.detail

    def test_verify_with_wrong_size_priors(self):
        gate = MinimaxGate()
        payoffs = np.array([
            [0.5, 0.4],
            [0.8, 0.1],
        ])
        # Wrong number of states
        priors = np.array([0.5])
        result = gate.verify(selected_tool=0, tool_payoffs=payoffs, state_priors=priors)
        # Should still work, just no expected payoff in detail
        assert "expected under priors" not in result.detail

    def test_factory(self):
        gate = create_minimax_gate(regret_threshold=0.2)
        assert isinstance(gate, MinimaxGate)
        assert gate.regret_threshold == 0.2

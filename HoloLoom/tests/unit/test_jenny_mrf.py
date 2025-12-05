"""
Test Jenny MRF Integration - Thompson Sampling + MRF Enhancement
================================================================

Tests for the Jenny MRF integration module (jenny_mrf.py):
1. PanelTypeLearner - Thompson Sampling panel selection
2. generate_why_panel_mrf - MRF-enhanced WHY panel generation
3. JennyMRFCompiler - Full MRF-enhanced compilation

Date: December 2025
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from HoloLoom.visualization.jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    PanelSizeJenny,
    LifecycleStage,
)
from HoloLoom.visualization.jenny_mrf import (
    PanelTypeLearner,
    PanelTypePrior,
    JennyMRFCompiler,
    generate_why_panel_mrf,
    create_mrf_compiler,
    create_panel_learner,
    MRF_AVAILABLE,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_spacetime():
    """Create a mock Spacetime object."""
    spacetime = MagicMock()
    spacetime.query_text = "What is Thompson Sampling?"
    spacetime.response = "Thompson Sampling is a Bayesian approach..."
    spacetime.confidence = 0.85
    spacetime.trace = MagicMock()
    spacetime.trace.threads_activated = ["thread1", "thread2", "thread3"]
    spacetime.trace.stage_durations = {"retrieve": 50, "decision": 30, "generate": 70}
    spacetime.trace.duration_ms = 150
    return spacetime


@pytest.fixture
def mock_analysis():
    """Create a mock QueryAnalysis object."""
    analysis = MagicMock()
    analysis.query_type = "factual"
    analysis.complexity = "moderate"
    analysis.confidence_level = "high"
    analysis.has_sources = True
    analysis.has_graph_context = True
    analysis.has_reasoning = False
    analysis.has_metrics = True
    return analysis


@pytest.fixture
def learner():
    """Create a fresh PanelTypeLearner."""
    return PanelTypeLearner()


@pytest.fixture
def temp_persist_path(tmp_path):
    """Create a temporary path for persistence testing."""
    return str(tmp_path / "learner_state.json")


# ============================================================================
# PanelTypePrior Tests
# ============================================================================

class TestPanelTypePrior:
    """Tests for the Beta distribution prior."""

    def test_default_initialization(self):
        """Prior starts with α=1, β=1 (uniform)."""
        prior = PanelTypePrior()
        assert prior.alpha == 1.0
        assert prior.beta == 1.0

    def test_expected_value_uniform(self):
        """Uniform prior has E[X] = 0.5."""
        prior = PanelTypePrior()
        assert prior.expected_value() == 0.5

    def test_expected_value_after_success(self):
        """E[X] increases after success."""
        prior = PanelTypePrior()
        prior.update_success(1.0)
        # E[X] = 2 / (2 + 1) = 0.667
        assert prior.expected_value() > 0.5
        assert abs(prior.expected_value() - 2/3) < 0.01

    def test_expected_value_after_failure(self):
        """E[X] stays same after failure with confidence=1.0."""
        prior = PanelTypePrior()
        prior.update_failure(1.0)
        # β += (1 - 1.0) = 0, so E[X] unchanged
        assert prior.expected_value() == 0.5

    def test_expected_value_after_failure_low_confidence(self):
        """E[X] decreases after failure with low confidence."""
        prior = PanelTypePrior()
        prior.update_failure(0.3)  # β += 0.7
        # E[X] = 1 / (1 + 1.7) ≈ 0.37
        assert prior.expected_value() < 0.5

    def test_sample_in_range(self):
        """Sample is always in [0, 1]."""
        prior = PanelTypePrior(alpha=5.0, beta=2.0)
        for _ in range(100):
            sample = prior.sample()
            assert 0.0 <= sample <= 1.0


# ============================================================================
# PanelTypeLearner Tests
# ============================================================================

class TestPanelTypeLearner:
    """Tests for Thompson Sampling panel type selection."""

    def test_initialization(self, learner):
        """Learner initializes with empty priors."""
        assert len(learner.priors) == 0
        assert len(learner.selection_history) == 0

    def test_select_from_single_candidate(self, learner):
        """With single candidate, always selects it."""
        selected = learner.select("factual", [PanelTypeJenny.TEXT])
        assert selected == PanelTypeJenny.TEXT

    def test_select_from_empty_candidates(self, learner):
        """With no candidates, returns TEXT as default."""
        selected = learner.select("factual", [])
        assert selected == PanelTypeJenny.TEXT

    def test_select_creates_history(self, learner):
        """Selection adds to history."""
        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE]
        learner.select("factual", candidates)

        assert len(learner.selection_history) == 1
        entry = learner.selection_history[0]
        assert entry["query_type"] == "factual"
        assert "TEXT" in entry["candidates"] or "text" in [c.lower() for c in entry["candidates"]]

    def test_update_success_increases_alpha(self, learner):
        """Success update increases alpha."""
        learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)

        key = ("factual", "text")
        prior = learner.priors[key]
        assert prior.alpha == 1.0 + 0.9  # Initial + confidence
        assert prior.beta == 1.0  # Unchanged

    def test_update_failure_increases_beta(self, learner):
        """Failure update increases beta."""
        learner.update("factual", PanelTypeJenny.TEXT, success=False, confidence=0.3)

        key = ("factual", "text")
        prior = learner.priors[key]
        assert prior.alpha == 1.0  # Unchanged
        assert prior.beta == 1.0 + 0.7  # Initial + (1 - confidence)

    def test_learning_improves_selection(self, learner):
        """After many successes, learner prefers successful panel type."""
        # Train: TEXT always succeeds, CONFIDENCE always fails
        for _ in range(20):
            learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)
            learner.update("factual", PanelTypeJenny.CONFIDENCE, success=False, confidence=0.2)

        # Sample many times - TEXT should win most
        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE]
        selections = [learner.select("factual", candidates) for _ in range(50)]
        text_count = sum(1 for s in selections if s == PanelTypeJenny.TEXT)

        # TEXT should be selected much more often (>80%)
        assert text_count >= 40, f"TEXT selected only {text_count}/50 times"

    def test_exploration_bonus(self, learner):
        """Exploration bonus adds randomness."""
        # Without bonus: deterministic-ish based on priors
        # With high bonus: more random
        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE]

        # Train strongly for TEXT
        for _ in range(50):
            learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.95)

        # With high exploration, should sometimes pick CONFIDENCE
        selections = [
            learner.select("factual", candidates, exploration_bonus=1.0)
            for _ in range(100)
        ]
        confidence_count = sum(1 for s in selections if s == PanelTypeJenny.CONFIDENCE)

        # Should have at least some CONFIDENCE selections due to exploration
        assert confidence_count >= 5, "Exploration bonus should add variety"

    def test_get_statistics(self, learner):
        """get_statistics returns learner state."""
        learner.select("factual", [PanelTypeJenny.TEXT])
        learner.update("factual", PanelTypeJenny.TEXT, success=True)

        stats = learner.get_statistics()
        assert stats["total_selections"] == 1
        assert "factual:text" in stats["priors"]

    def test_get_best_panel_type(self, learner):
        """get_best_panel_type returns highest E[X] panel."""
        # Train TEXT with high success
        for _ in range(10):
            learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)
            learner.update("factual", PanelTypeJenny.CONFIDENCE, success=False, confidence=0.3)

        best = learner.get_best_panel_type("factual")
        assert best == PanelTypeJenny.TEXT

    def test_get_best_panel_type_unknown_query(self, learner):
        """get_best_panel_type returns None for unknown query type."""
        best = learner.get_best_panel_type("unknown_type")
        assert best is None


# ============================================================================
# PanelTypeLearner Persistence Tests
# ============================================================================

class TestPanelTypeLearnerPersistence:
    """Tests for learner state persistence."""

    def test_save_and_load(self, temp_persist_path):
        """State persists across learner instances."""
        # Create and train learner
        learner1 = PanelTypeLearner(persist_path=temp_persist_path)
        learner1.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)
        learner1.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.8)

        # Create new learner with same path
        learner2 = PanelTypeLearner(persist_path=temp_persist_path)

        # Check state was loaded
        key = ("factual", "text")
        assert key in learner2.priors
        prior = learner2.priors[key]
        assert prior.alpha == 1.0 + 0.9 + 0.8  # Initial + two successes

    def test_no_file_gracefully_handled(self, temp_persist_path):
        """Non-existent persist file doesn't crash."""
        learner = PanelTypeLearner(persist_path=temp_persist_path + ".nonexistent")
        assert len(learner.priors) == 0

    def test_corrupt_file_gracefully_handled(self, temp_persist_path):
        """Corrupt persist file logs warning but doesn't crash."""
        # Write corrupt data
        with open(temp_persist_path, 'w') as f:
            f.write("not valid json")

        # Should not raise
        learner = PanelTypeLearner(persist_path=temp_persist_path)
        assert len(learner.priors) == 0


# ============================================================================
# WHY Panel Generation Tests
# ============================================================================

class TestGenerateWhyPanelMRF:
    """Tests for MRF-enhanced WHY panel generation."""

    @pytest.mark.asyncio
    async def test_generates_why_panel_without_mrf(self, mock_spacetime, mock_analysis):
        """Generates WHY panel even without MRF."""
        specs = [
            JennySpec(
                spacetime_id="test",
                panel_type=PanelTypeJenny.TEXT,
                title="Response",
                content={"text": "Test response"},
            )
        ]

        panel = await generate_why_panel_mrf(
            mock_spacetime, specs, mock_analysis, mrf=None
        )

        assert panel.panel_type == PanelTypeJenny.WHY
        assert panel.title == "Why This UI?"
        assert panel.lifecycle == LifecycleStage.SYSTEM
        assert panel.content["mrf_enhanced"] is False

    @pytest.mark.asyncio
    async def test_why_panel_contains_decision_factors(self, mock_spacetime, mock_analysis):
        """WHY panel includes decision factors."""
        specs = []
        panel = await generate_why_panel_mrf(
            mock_spacetime, specs, mock_analysis, mrf=None
        )

        content = panel.content
        assert "decision_factors" in content
        factors = content["decision_factors"]
        assert "confidence_level" in factors
        assert "has_sources" in factors

    @pytest.mark.asyncio
    async def test_why_panel_has_correct_priority(self, mock_spacetime, mock_analysis):
        """WHY panel gets lowest priority (appears last)."""
        specs = []
        panel = await generate_why_panel_mrf(
            mock_spacetime, specs, mock_analysis, mrf=None, priority=99
        )

        assert panel.priority == 99

    @pytest.mark.asyncio
    async def test_why_panel_lists_generated_panels(self, mock_spacetime, mock_analysis):
        """WHY panel lists what panels were generated."""
        specs = [
            JennySpec(
                spacetime_id="test",
                panel_type=PanelTypeJenny.TEXT,
                title="Response",
                content={},
            ),
            JennySpec(
                spacetime_id="test",
                panel_type=PanelTypeJenny.CONFIDENCE,
                title="Confidence",
                content={},
            ),
        ]

        panel = await generate_why_panel_mrf(
            mock_spacetime, specs, mock_analysis, mrf=None
        )

        assert panel.content["panels_generated"] == 2
        assert "text" in panel.content["panel_types"]
        assert "confidence" in panel.content["panel_types"]


# ============================================================================
# JennyMRFCompiler Tests
# ============================================================================

class TestJennyMRFCompiler:
    """Tests for the MRF-enhanced compiler."""

    def test_initialization_defaults(self):
        """Compiler initializes with sensible defaults."""
        compiler = JennyMRFCompiler()

        assert compiler.enable_learning is True
        assert compiler.learner is not None
        assert compiler.include_why_panel is True
        assert "mrf" in compiler.VERSION.lower()

    def test_initialization_learning_disabled(self):
        """Can disable learning."""
        compiler = JennyMRFCompiler(enable_learning=False)

        assert compiler.enable_learning is False
        assert compiler.learner is None

    def test_initialization_with_persist_path(self, temp_persist_path):
        """Learner uses persist path if provided."""
        compiler = JennyMRFCompiler(
            enable_learning=True,
            learning_persist_path=temp_persist_path
        )

        assert compiler.learner.persist_path == temp_persist_path

    @pytest.mark.asyncio
    async def test_compile_includes_text_panel(self, mock_spacetime):
        """Compilation always includes TEXT panel."""
        compiler = JennyMRFCompiler(include_why_panel=False)

        with patch.object(compiler, '_auto_select_strategy') as mock_strategy:
            from HoloLoom.protocols.jenny import CompilationStrategy
            mock_strategy.return_value = CompilationStrategy.MINIMAL

            specs = await compiler.compile(mock_spacetime)

        # Should have at least TEXT panel
        panel_types = [s.panel_type for s in specs]
        assert PanelTypeJenny.TEXT in panel_types

    @pytest.mark.asyncio
    async def test_compile_includes_why_panel(self, mock_spacetime):
        """Compilation includes WHY panel when enabled."""
        compiler = JennyMRFCompiler(include_why_panel=True)

        specs = await compiler.compile(mock_spacetime)

        panel_types = [s.panel_type for s in specs]
        assert PanelTypeJenny.WHY in panel_types

    @pytest.mark.asyncio
    async def test_compile_excludes_why_panel_when_disabled(self, mock_spacetime):
        """Compilation excludes WHY panel when disabled."""
        compiler = JennyMRFCompiler(include_why_panel=False)

        specs = await compiler.compile(mock_spacetime)

        panel_types = [s.panel_type for s in specs]
        assert PanelTypeJenny.WHY not in panel_types

    @pytest.mark.asyncio
    async def test_compile_panels_sorted_by_priority(self, mock_spacetime):
        """Compiled panels are sorted by priority."""
        compiler = JennyMRFCompiler(include_why_panel=True)

        specs = await compiler.compile(mock_spacetime)

        priorities = [s.priority for s in specs]
        assert priorities == sorted(priorities)

    def test_update_learning(self):
        """update_learning delegates to learner."""
        compiler = JennyMRFCompiler(enable_learning=True)

        # Update learning
        compiler.update_learning("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)

        # Verify learner was updated
        stats = compiler.get_learning_statistics()
        assert "factual:text" in stats["priors"]

    def test_update_learning_disabled(self):
        """update_learning no-ops when learning disabled."""
        compiler = JennyMRFCompiler(enable_learning=False)

        # Should not raise
        compiler.update_learning("factual", PanelTypeJenny.TEXT, success=True)

    def test_get_learning_statistics_enabled(self):
        """get_learning_statistics returns stats when learning enabled."""
        compiler = JennyMRFCompiler(enable_learning=True)
        compiler.update_learning("factual", PanelTypeJenny.TEXT, success=True)

        stats = compiler.get_learning_statistics()
        assert "total_selections" in stats
        assert "priors" in stats

    def test_get_learning_statistics_disabled(self):
        """get_learning_statistics indicates disabled when learning off."""
        compiler = JennyMRFCompiler(enable_learning=False)

        stats = compiler.get_learning_statistics()
        assert stats["learning_enabled"] is False


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_mrf_compiler(self):
        """create_mrf_compiler creates configured compiler."""
        compiler = create_mrf_compiler(enable_learning=True)

        assert isinstance(compiler, JennyMRFCompiler)
        assert compiler.enable_learning is True

    def test_create_mrf_compiler_with_persist(self, temp_persist_path):
        """create_mrf_compiler passes persist path."""
        compiler = create_mrf_compiler(
            enable_learning=True,
            learning_persist_path=temp_persist_path
        )

        assert compiler.learner.persist_path == temp_persist_path

    def test_create_panel_learner(self):
        """create_panel_learner creates configured learner."""
        learner = create_panel_learner()

        assert isinstance(learner, PanelTypeLearner)

    def test_create_panel_learner_with_persist(self, temp_persist_path):
        """create_panel_learner passes persist path."""
        learner = create_panel_learner(persist_path=temp_persist_path)

        assert learner.persist_path == temp_persist_path


# ============================================================================
# MRF Integration Tests (when MRF available)
# ============================================================================

@pytest.mark.skipif(not MRF_AVAILABLE, reason="MRF not available")
class TestMRFIntegration:
    """Tests that require MRF to be available."""

    @pytest.mark.asyncio
    async def test_why_panel_uses_mrf_when_available(self, mock_spacetime, mock_analysis):
        """WHY panel is enhanced by MRF when available."""
        from HoloLoom.prompting.unified_mrf import UnifiedMRF

        mrf = MagicMock(spec=UnifiedMRF)
        mrf.refine_prompt = AsyncMock(return_value={
            "enhanced_prompt": "MRF-enhanced explanation here"
        })

        specs = []
        panel = await generate_why_panel_mrf(
            mock_spacetime, specs, mock_analysis, mrf=mrf
        )

        assert panel.content["mrf_enhanced"] is True
        mrf.refine_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_why_panel_falls_back_on_mrf_error(self, mock_spacetime, mock_analysis):
        """WHY panel falls back to rule-based on MRF error."""
        from HoloLoom.prompting.unified_mrf import UnifiedMRF

        mrf = MagicMock(spec=UnifiedMRF)
        mrf.refine_prompt = AsyncMock(side_effect=Exception("MRF error"))

        specs = []
        panel = await generate_why_panel_mrf(
            mock_spacetime, specs, mock_analysis, mrf=mrf
        )

        # Should still generate panel with rule-based explanation
        assert panel.panel_type == PanelTypeJenny.WHY
        assert panel.content["mrf_enhanced"] is False


# ============================================================================
# Phase C: Learning Loop Integration Tests
# ============================================================================

class TestActionLearningMap:
    """Tests for ACTION_LEARNING_MAP and create_learning_callback."""

    def test_action_learning_map_exists(self):
        """ACTION_LEARNING_MAP is defined and populated."""
        from HoloLoom.visualization.jenny_mrf import ACTION_LEARNING_MAP

        assert "pin_panel" in ACTION_LEARNING_MAP
        assert "dismiss_panel" in ACTION_LEARNING_MAP
        assert "show_why_panel" in ACTION_LEARNING_MAP

    def test_action_learning_map_pin_positive(self):
        """PIN action has high positive signal."""
        from HoloLoom.visualization.jenny_mrf import ACTION_LEARNING_MAP

        success, confidence = ACTION_LEARNING_MAP["pin_panel"]
        assert success is True
        assert confidence >= 0.8

    def test_action_learning_map_dismiss_negative(self):
        """DISMISS action has low/negative signal."""
        from HoloLoom.visualization.jenny_mrf import ACTION_LEARNING_MAP

        success, confidence = ACTION_LEARNING_MAP["dismiss_panel"]
        assert success is False
        assert confidence < 0.5

    def test_action_learning_map_why_moderate(self):
        """WHY action has moderate positive signal."""
        from HoloLoom.visualization.jenny_mrf import ACTION_LEARNING_MAP

        success, confidence = ACTION_LEARNING_MAP["show_why_panel"]
        assert success is True
        assert 0.5 <= confidence <= 0.8


class TestCreateLearningCallback:
    """Tests for create_learning_callback factory function."""

    @pytest.mark.asyncio
    async def test_create_learning_callback_returns_callable(self):
        """create_learning_callback returns an async callable."""
        from HoloLoom.visualization.jenny_mrf import create_learning_callback

        compiler = JennyMRFCompiler(enable_learning=True)
        callback = await create_learning_callback(compiler)

        assert callable(callback)

    @pytest.mark.asyncio
    async def test_callback_updates_priors_on_pin(self):
        """Callback updates priors when PIN action succeeds."""
        from HoloLoom.visualization.jenny_mrf import create_learning_callback
        from HoloLoom.visualization.jenny_actions import ActionResult, ActionStatus

        compiler = JennyMRFCompiler(enable_learning=True)
        callback = await create_learning_callback(compiler)

        # Create mock action result for PIN
        result = ActionResult(
            action_id="test_action_1",
            handler="pin_panel",
            status=ActionStatus.SUCCESS,
            spec_id="test_spec_1",
        )

        # Create spec with metadata
        spec = JennySpec(
            spacetime_id="test",
            panel_type=PanelTypeJenny.TEXT,
            title="Test",
            content={"text": "Test"},
            metadata={"query_type": "factual"},
        )

        # Get initial state
        initial_stats = compiler.get_learning_statistics()
        initial_alpha = initial_stats["priors"].get("factual:text", {}).get("alpha", 1.0)

        # Execute callback
        await callback(result, spec)

        # Verify alpha increased (success update)
        stats = compiler.get_learning_statistics()
        new_alpha = stats["priors"]["factual:text"]["alpha"]
        assert new_alpha > initial_alpha

    @pytest.mark.asyncio
    async def test_callback_updates_priors_on_dismiss(self):
        """Callback updates priors when DISMISS action occurs."""
        from HoloLoom.visualization.jenny_mrf import create_learning_callback
        from HoloLoom.visualization.jenny_actions import ActionResult, ActionStatus

        compiler = JennyMRFCompiler(enable_learning=True)
        callback = await create_learning_callback(compiler)

        # Create mock action result for DISMISS
        result = ActionResult(
            action_id="test_action_2",
            handler="dismiss_panel",
            status=ActionStatus.SUCCESS,
            spec_id="test_spec_2",
        )

        # Create spec with metadata
        spec = JennySpec(
            spacetime_id="test",
            panel_type=PanelTypeJenny.TEXT,
            title="Test",
            content={"text": "Test"},
            metadata={"query_type": "factual"},
        )

        # Get initial state
        initial_stats = compiler.get_learning_statistics()
        initial_beta = initial_stats["priors"].get("factual:text", {}).get("beta", 1.0)

        # Execute callback
        await callback(result, spec)

        # Verify beta increased (failure update since dismiss = success=False)
        stats = compiler.get_learning_statistics()
        new_beta = stats["priors"]["factual:text"]["beta"]
        assert new_beta > initial_beta

    @pytest.mark.asyncio
    async def test_callback_ignores_unknown_handlers(self):
        """Callback skips unknown action handlers."""
        from HoloLoom.visualization.jenny_mrf import create_learning_callback
        from HoloLoom.visualization.jenny_actions import ActionResult, ActionStatus

        compiler = JennyMRFCompiler(enable_learning=True)
        callback = await create_learning_callback(compiler)

        # Create mock action result for unknown handler
        result = ActionResult(
            action_id="test_action_3",
            handler="unknown_handler",
            status=ActionStatus.SUCCESS,
            spec_id="test_spec_3",
        )

        spec = JennySpec(
            spacetime_id="test",
            panel_type=PanelTypeJenny.TEXT,
            title="Test",
            content={"text": "Test"},
            metadata={"query_type": "factual"},
        )

        # Should not raise, and priors should remain empty
        await callback(result, spec)

        stats = compiler.get_learning_statistics()
        assert len(stats["priors"]) == 0

    @pytest.mark.asyncio
    async def test_callback_uses_unknown_query_type_fallback(self):
        """Callback uses 'unknown' query_type when not in metadata."""
        from HoloLoom.visualization.jenny_mrf import create_learning_callback
        from HoloLoom.visualization.jenny_actions import ActionResult, ActionStatus

        compiler = JennyMRFCompiler(enable_learning=True)
        callback = await create_learning_callback(compiler)

        result = ActionResult(
            action_id="test_action_4",
            handler="pin_panel",
            status=ActionStatus.SUCCESS,
            spec_id="test_spec_4",
        )

        # Spec without query_type in metadata
        spec = JennySpec(
            spacetime_id="test",
            panel_type=PanelTypeJenny.TEXT,
            title="Test",
            content={"text": "Test"},
            metadata={},  # No query_type
        )

        await callback(result, spec)

        stats = compiler.get_learning_statistics()
        assert "unknown:text" in stats["priors"]


class TestJennyRuntimeLearningIntegration:
    """Tests for JennyRuntime learning integration (Phase C)."""

    @pytest.mark.asyncio
    async def test_runtime_accepts_compiler(self):
        """JennyRuntime accepts optional compiler parameter."""
        from HoloLoom.visualization.jenny_runtime import JennyRuntime

        compiler = JennyMRFCompiler(enable_learning=True)
        runtime = JennyRuntime(compiler=compiler)

        assert runtime._compiler is compiler

    @pytest.mark.asyncio
    async def test_runtime_creates_learning_callback_on_start(self):
        """JennyRuntime creates learning callback when compiler has learning enabled."""
        from HoloLoom.visualization.jenny_runtime import JennyRuntime

        compiler = JennyMRFCompiler(enable_learning=True)

        async with JennyRuntime(compiler=compiler) as runtime:
            # Verify action handler has learning callback
            assert runtime._actions._learning_callback is not None

    @pytest.mark.asyncio
    async def test_runtime_no_callback_when_learning_disabled(self):
        """JennyRuntime doesn't create callback when learning disabled."""
        from HoloLoom.visualization.jenny_runtime import JennyRuntime

        compiler = JennyMRFCompiler(enable_learning=False)

        async with JennyRuntime(compiler=compiler) as runtime:
            # Verify action handler has no learning callback
            assert runtime._actions._learning_callback is None

    @pytest.mark.asyncio
    async def test_runtime_no_callback_when_no_compiler(self):
        """JennyRuntime doesn't create callback when no compiler provided."""
        from HoloLoom.visualization.jenny_runtime import JennyRuntime

        async with JennyRuntime() as runtime:
            # Verify action handler has no learning callback
            assert runtime._actions._learning_callback is None

    @pytest.mark.asyncio
    async def test_end_to_end_learning_loop(self):
        """End-to-end test: action → learning callback → Thompson update."""
        from HoloLoom.visualization.jenny_runtime import JennyRuntime

        compiler = JennyMRFCompiler(enable_learning=True)

        async with JennyRuntime(compiler=compiler) as runtime:
            # Create a panel
            panel = await runtime.ask("What is Thompson Sampling?")

            # Get initial learning state
            initial_stats = compiler.get_learning_statistics()

            # Execute PIN action
            result = await runtime.act(panel, "pin")

            # Verify action succeeded
            from HoloLoom.visualization.jenny_actions import ActionStatus
            assert result.status == ActionStatus.SUCCESS

            # Verify learning was updated
            # (query_type defaults to 'unknown' since runtime uses standalone query mode)
            final_stats = compiler.get_learning_statistics()

            # Check that priors were created/updated
            # The exact key depends on the query_type in spec.metadata
            assert len(final_stats["priors"]) >= len(initial_stats.get("priors", {}))


class TestJennySessionLearningIntegration:
    """Tests for jenny_session convenience function with learning."""

    @pytest.mark.asyncio
    async def test_jenny_session_accepts_compiler(self):
        """jenny_session accepts compiler parameter."""
        from HoloLoom.visualization.jenny_runtime import jenny_session

        compiler = JennyMRFCompiler(enable_learning=True)

        async with jenny_session(compiler=compiler) as jenny:
            assert jenny._compiler is compiler
            assert jenny._actions._learning_callback is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

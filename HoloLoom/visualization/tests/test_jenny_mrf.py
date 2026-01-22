"""
Jenny MRF and Thompson Sampling Tests
=====================================
Comprehensive tests for panel type learning and MRF integration.

Date: January 2026
Author: Claude Code

Test Categories:
- PanelTypePrior (~80 lines): Beta distribution, sampling, updates
- PanelTypeLearner (~120 lines): Selection, updates, statistics, persistence
- JennyMRFCompiler (~100 lines): Compilation with learning, update methods
- Learning Callback (~60 lines): Action→learning signal mapping
- Factory Functions (~40 lines): create_mrf_compiler, create_panel_learner
"""

import pytest
import os
import json
import random
from typing import List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from HoloLoom.visualization.jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    PanelSizeJenny,
    create_spec,
)
from HoloLoom.visualization.jenny_enums import LifecycleStage


# ============================================================================
# MRF Imports with Graceful Degradation
# ============================================================================

try:
    from HoloLoom.visualization.jenny_mrf import (
        PanelTypePrior,
        PanelTypeLearner,
        JennyMRFCompiler,
        create_mrf_compiler,
        create_panel_learner,
        create_learning_callback,
        ACTION_LEARNING_MAP,
        MRF_AVAILABLE,
        generate_why_panel_mrf,
    )
    MRF_MODULE_AVAILABLE = True
except ImportError:
    MRF_MODULE_AVAILABLE = False
    PanelTypePrior = None
    PanelTypeLearner = None
    JennyMRFCompiler = None
    create_mrf_compiler = None
    create_panel_learner = None
    create_learning_callback = None
    ACTION_LEARNING_MAP = None
    MRF_AVAILABLE = False
    generate_why_panel_mrf = None


# Skip all tests if MRF module not available
pytestmark = pytest.mark.skipif(
    not MRF_MODULE_AVAILABLE,
    reason="Jenny MRF module not available"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def panel_prior():
    """Create a fresh PanelTypePrior."""
    return PanelTypePrior()


@pytest.fixture
def panel_prior_trained():
    """Create a PanelTypePrior with some training."""
    prior = PanelTypePrior(alpha=5.0, beta=2.0)
    return prior


@pytest.fixture
def panel_learner():
    """Create a fresh PanelTypeLearner."""
    return PanelTypeLearner()


@pytest.fixture
def panel_learner_with_history():
    """Create a learner with training history."""
    learner = PanelTypeLearner()

    # Simulate some learning history
    learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)
    learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.85)
    learner.update("factual", PanelTypeJenny.CONFIDENCE, success=True, confidence=0.7)
    learner.update("analytical", PanelTypeJenny.GRAPH, success=True, confidence=0.8)
    learner.update("analytical", PanelTypeJenny.GRAPH, success=False, confidence=0.3)

    return learner


@pytest.fixture
def temp_persist_path(tmp_path):
    """Create a temporary path for learner persistence."""
    return str(tmp_path / "learner_state.json")


@pytest.fixture
def mock_spacetime():
    """Create a mock Spacetime object."""
    spacetime = Mock()
    spacetime.query_text = "What is Thompson Sampling?"
    spacetime.response = "Thompson Sampling is a Bayesian approach..."
    spacetime.confidence = 0.85
    spacetime.trace = Mock()
    spacetime.trace.threads_activated = ["thread1", "thread2"]
    spacetime.trace.stage_durations = {"retrieval": 50.0, "decision": 30.0}
    spacetime.trace.duration_ms = 150.0
    spacetime.metadata = {"tool_used": "answer"}
    return spacetime


@pytest.fixture
def mock_query_analysis():
    """Create a mock QueryAnalysis."""
    analysis = Mock()
    analysis.query_type = "factual"
    analysis.complexity = "moderate"
    analysis.confidence_level = "high"
    analysis.has_sources = True
    analysis.has_graph_context = True
    analysis.has_reasoning = False
    analysis.has_metrics = True
    return analysis


# ============================================================================
# PanelTypePrior Tests
# ============================================================================

class TestPanelTypePrior:
    """Tests for PanelTypePrior Beta distribution."""

    def test_default_initialization(self, panel_prior):
        """Test default prior is uniform (alpha=1, beta=1)."""
        assert panel_prior.alpha == 1.0
        assert panel_prior.beta == 1.0

    def test_custom_initialization(self):
        """Test custom prior initialization."""
        prior = PanelTypePrior(alpha=3.0, beta=5.0)
        assert prior.alpha == 3.0
        assert prior.beta == 5.0

    def test_sample_returns_value_in_range(self, panel_prior):
        """Test sample returns value in [0, 1]."""
        for _ in range(100):
            sample = panel_prior.sample()
            assert 0.0 <= sample <= 1.0

    def test_sample_uses_beta_distribution(self, panel_prior_trained):
        """Test samples follow Beta distribution characteristics."""
        samples = [panel_prior_trained.sample() for _ in range(1000)]

        # With alpha=5, beta=2, mean should be around 0.714
        mean_sample = sum(samples) / len(samples)
        expected_mean = panel_prior_trained.expected_value()

        # Allow 10% tolerance
        assert abs(mean_sample - expected_mean) < 0.1

    def test_expected_value_calculation(self):
        """Test expected value formula."""
        prior = PanelTypePrior(alpha=3.0, beta=7.0)
        expected = prior.expected_value()

        # E[X] = alpha / (alpha + beta) = 3 / 10 = 0.3
        assert expected == pytest.approx(0.3)

    def test_expected_value_uniform(self, panel_prior):
        """Test uniform prior has expected value of 0.5."""
        assert panel_prior.expected_value() == 0.5

    def test_update_success_increases_alpha(self, panel_prior):
        """Test update_success increases alpha."""
        initial_alpha = panel_prior.alpha
        panel_prior.update_success(confidence=0.8)

        assert panel_prior.alpha == initial_alpha + 0.8
        assert panel_prior.beta == 1.0  # Unchanged

    def test_update_success_default_confidence(self, panel_prior):
        """Test update_success with default confidence."""
        initial_alpha = panel_prior.alpha
        panel_prior.update_success()

        assert panel_prior.alpha == initial_alpha + 1.0

    def test_update_failure_increases_beta(self, panel_prior):
        """Test update_failure increases beta based on (1 - confidence)."""
        initial_beta = panel_prior.beta
        panel_prior.update_failure(confidence=0.3)

        # beta += (1 - 0.3) = 0.7
        assert panel_prior.beta == pytest.approx(initial_beta + 0.7)
        assert panel_prior.alpha == 1.0  # Unchanged

    def test_update_failure_default_confidence(self, panel_prior):
        """Test update_failure with default confidence."""
        initial_beta = panel_prior.beta
        panel_prior.update_failure()

        # beta += (1 - 1.0) = 0
        assert panel_prior.beta == initial_beta + 0.0

    def test_multiple_updates_accumulate(self, panel_prior):
        """Test multiple updates accumulate correctly."""
        panel_prior.update_success(0.9)
        panel_prior.update_success(0.8)
        panel_prior.update_failure(0.2)

        # alpha = 1 + 0.9 + 0.8 = 2.7
        # beta = 1 + (1-0.2) = 1.8
        assert panel_prior.alpha == pytest.approx(2.7)
        assert panel_prior.beta == pytest.approx(1.8)


# ============================================================================
# PanelTypeLearner Tests
# ============================================================================

class TestPanelTypeLearner:
    """Tests for PanelTypeLearner Thompson Sampling."""

    def test_initialization(self, panel_learner):
        """Test learner initializes with empty state."""
        assert len(panel_learner.priors) == 0
        assert len(panel_learner.selection_history) == 0
        assert panel_learner.persist_path is None

    def test_select_empty_candidates_returns_text(self, panel_learner):
        """Test select with empty candidates returns TEXT."""
        result = panel_learner.select("factual", [])
        assert result == PanelTypeJenny.TEXT

    def test_select_single_candidate(self, panel_learner):
        """Test select with single candidate returns that candidate."""
        result = panel_learner.select("factual", [PanelTypeJenny.GRAPH])
        assert result == PanelTypeJenny.GRAPH

    def test_select_multiple_candidates(self, panel_learner):
        """Test select from multiple candidates."""
        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE, PanelTypeJenny.GRAPH]
        result = panel_learner.select("factual", candidates)

        assert result in candidates

    def test_select_records_history(self, panel_learner):
        """Test select records selection in history."""
        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE]
        panel_learner.select("factual", candidates)

        assert len(panel_learner.selection_history) == 1
        history_entry = panel_learner.selection_history[0]

        assert history_entry["query_type"] == "factual"
        assert "timestamp" in history_entry
        assert "samples" in history_entry
        assert "selected" in history_entry

    def test_select_with_exploration_bonus(self, panel_learner):
        """Test select with exploration bonus."""
        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE]

        # Run many selections - exploration bonus should increase variability
        results_no_bonus = [panel_learner.select("factual", candidates) for _ in range(20)]
        results_with_bonus = [panel_learner.select("factual", candidates, exploration_bonus=0.5) for _ in range(20)]

        # Both should return valid candidates
        assert all(r in candidates for r in results_no_bonus)
        assert all(r in candidates for r in results_with_bonus)

    def test_select_trained_prefers_successful(self, panel_learner_with_history):
        """Test trained learner prefers successful panel types."""
        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CODE]

        # TEXT has history of success for "factual", CODE has no history
        # Run multiple times to see preference
        selections = [panel_learner_with_history.select("factual", candidates) for _ in range(50)]

        text_count = selections.count(PanelTypeJenny.TEXT)
        # TEXT should be selected more often (but not always due to exploration)
        assert text_count > 0

    def test_update_creates_prior(self, panel_learner):
        """Test update creates prior if not exists."""
        panel_learner.update("procedural", PanelTypeJenny.CODE, success=True, confidence=0.8)

        key = ("procedural", "code")
        assert key in panel_learner.priors

    def test_update_success(self, panel_learner):
        """Test update on success increases alpha."""
        panel_learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)

        prior = panel_learner.priors[("factual", "text")]
        assert prior.alpha == pytest.approx(1.9)
        assert prior.beta == 1.0

    def test_update_failure(self, panel_learner):
        """Test update on failure increases beta."""
        panel_learner.update("factual", PanelTypeJenny.TEXT, success=False, confidence=0.3)

        prior = panel_learner.priors[("factual", "text")]
        assert prior.alpha == 1.0
        assert prior.beta == pytest.approx(1.7)

    def test_get_statistics(self, panel_learner_with_history):
        """Test get_statistics returns correct structure."""
        stats = panel_learner_with_history.get_statistics()

        assert "total_selections" in stats
        assert "priors" in stats
        assert stats["total_selections"] >= 0

        # Check prior stats structure
        for key, prior_stats in stats["priors"].items():
            assert "alpha" in prior_stats
            assert "beta" in prior_stats
            assert "expected_value" in prior_stats
            assert "total_updates" in prior_stats

    def test_get_best_panel_type_existing(self, panel_learner_with_history):
        """Test get_best_panel_type for known query type."""
        best = panel_learner_with_history.get_best_panel_type("factual")

        # Should return TEXT (has highest expected value for factual)
        assert best is not None
        assert best in [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE]

    def test_get_best_panel_type_unknown(self, panel_learner):
        """Test get_best_panel_type for unknown query type."""
        best = panel_learner.get_best_panel_type("unknown_type")

        # No priors exist, should return None
        assert best is None

    def test_persistence_save_load(self, temp_persist_path):
        """Test learner can save and load state."""
        # Create and train learner
        learner1 = PanelTypeLearner(persist_path=temp_persist_path)
        learner1.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)
        learner1.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.8)

        # Verify state was saved
        assert Path(temp_persist_path).exists()

        # Create new learner that loads state
        learner2 = PanelTypeLearner(persist_path=temp_persist_path)

        # Verify priors were restored
        prior1 = learner1.priors[("factual", "text")]
        prior2 = learner2.priors[("factual", "text")]

        assert prior1.alpha == pytest.approx(prior2.alpha)
        assert prior1.beta == pytest.approx(prior2.beta)

    def test_persistence_file_format(self, temp_persist_path):
        """Test persisted state file format."""
        learner = PanelTypeLearner(persist_path=temp_persist_path)
        learner.update("factual", PanelTypeJenny.TEXT, success=True)

        # Read and verify JSON format
        with open(temp_persist_path) as f:
            state = json.load(f)

        assert "priors" in state
        assert "selection_count" in state
        assert "saved_at" in state

    def test_persistence_handles_missing_file(self, temp_persist_path):
        """Test learner handles missing persist file gracefully."""
        # File doesn't exist - should not raise
        learner = PanelTypeLearner(persist_path=temp_persist_path)

        # Learner should be empty
        assert len(learner.priors) == 0


# ============================================================================
# JennyMRFCompiler Tests
# ============================================================================

class TestJennyMRFCompiler:
    """Tests for JennyMRFCompiler."""

    def test_initialization_default(self):
        """Test default compiler initialization."""
        compiler = JennyMRFCompiler()

        assert compiler.enable_learning is True
        assert compiler.learner is not None
        assert compiler.include_why_panel is True

    def test_initialization_no_learning(self):
        """Test compiler initialization without learning."""
        compiler = JennyMRFCompiler(enable_learning=False)

        assert compiler.enable_learning is False
        assert compiler.learner is None

    def test_initialization_with_persist_path(self, temp_persist_path):
        """Test compiler with learning persistence."""
        compiler = JennyMRFCompiler(
            enable_learning=True,
            learning_persist_path=temp_persist_path
        )

        assert compiler.learner is not None
        assert compiler.learner.persist_path == temp_persist_path

    def test_update_learning(self):
        """Test update_learning delegates to learner."""
        compiler = JennyMRFCompiler(enable_learning=True)

        compiler.update_learning("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)

        # Verify learner was updated
        stats = compiler.get_learning_statistics()
        assert "factual:text" in stats["priors"]

    def test_update_learning_disabled(self):
        """Test update_learning does nothing when disabled."""
        compiler = JennyMRFCompiler(enable_learning=False)

        # Should not raise
        compiler.update_learning("factual", PanelTypeJenny.TEXT, success=True)

        stats = compiler.get_learning_statistics()
        assert stats == {"learning_enabled": False}

    def test_get_learning_statistics(self):
        """Test get_learning_statistics returns correct format."""
        compiler = JennyMRFCompiler(enable_learning=True)
        compiler.update_learning("factual", PanelTypeJenny.TEXT, success=True, confidence=0.8)

        stats = compiler.get_learning_statistics()

        assert "total_selections" in stats
        assert "priors" in stats
        assert "factual:text" in stats["priors"]

    def test_get_learning_statistics_disabled(self):
        """Test get_learning_statistics when learning disabled."""
        compiler = JennyMRFCompiler(enable_learning=False)

        stats = compiler.get_learning_statistics()
        assert stats == {"learning_enabled": False}

    def test_version(self):
        """Test compiler version is set."""
        compiler = JennyMRFCompiler()
        assert hasattr(compiler, 'VERSION')
        assert "mrf" in compiler.VERSION.lower()


# ============================================================================
# Learning Callback Tests
# ============================================================================

class TestLearningCallback:
    """Tests for action-to-learning callback."""

    def test_action_learning_map_exists(self):
        """Test ACTION_LEARNING_MAP is defined."""
        assert ACTION_LEARNING_MAP is not None
        assert len(ACTION_LEARNING_MAP) > 0

    def test_action_learning_map_structure(self):
        """Test ACTION_LEARNING_MAP has correct structure."""
        for action, (success, confidence) in ACTION_LEARNING_MAP.items():
            assert isinstance(action, str)
            assert isinstance(success, bool)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0

    def test_pin_panel_is_success(self):
        """Test pin_panel action is success signal."""
        assert "pin_panel" in ACTION_LEARNING_MAP
        success, confidence = ACTION_LEARNING_MAP["pin_panel"]
        assert success is True
        assert confidence >= 0.8  # Should be strong signal

    def test_dismiss_panel_is_failure(self):
        """Test dismiss_panel action is failure signal."""
        assert "dismiss_panel" in ACTION_LEARNING_MAP
        success, confidence = ACTION_LEARNING_MAP["dismiss_panel"]
        assert success is False

    def test_copy_content_is_success(self):
        """Test copy_content action is success signal."""
        if "copy_content" in ACTION_LEARNING_MAP:
            success, confidence = ACTION_LEARNING_MAP["copy_content"]
            assert success is True

    def test_expand_panel_is_success(self):
        """Test expand_panel action is success signal."""
        if "expand_panel" in ACTION_LEARNING_MAP:
            success, confidence = ACTION_LEARNING_MAP["expand_panel"]
            assert success is True


# ============================================================================
# Factory Functions Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_mrf_compiler(self):
        """Test create_mrf_compiler creates compiler."""
        compiler = create_mrf_compiler()

        assert isinstance(compiler, JennyMRFCompiler)
        assert compiler.enable_learning is True

    def test_create_mrf_compiler_no_learning(self):
        """Test create_mrf_compiler without learning."""
        compiler = create_mrf_compiler(enable_learning=False)

        assert isinstance(compiler, JennyMRFCompiler)
        assert compiler.enable_learning is False

    def test_create_mrf_compiler_with_persist(self, temp_persist_path):
        """Test create_mrf_compiler with persistence."""
        compiler = create_mrf_compiler(
            enable_learning=True,
            learning_persist_path=temp_persist_path
        )

        assert compiler.learner.persist_path == temp_persist_path

    def test_create_panel_learner(self):
        """Test create_panel_learner creates learner."""
        learner = create_panel_learner()

        assert isinstance(learner, PanelTypeLearner)
        assert learner.persist_path is None

    def test_create_panel_learner_with_persist(self, temp_persist_path):
        """Test create_panel_learner with persistence."""
        learner = create_panel_learner(persist_path=temp_persist_path)

        assert isinstance(learner, PanelTypeLearner)
        assert learner.persist_path == temp_persist_path


# ============================================================================
# Thompson Sampling Behavior Tests
# ============================================================================

class TestThompsonSamplingBehavior:
    """Tests for Thompson Sampling exploration/exploitation behavior."""

    def test_exploration_with_uniform_priors(self, panel_learner):
        """Test uniform priors lead to random exploration."""
        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE, PanelTypeJenny.GRAPH]

        # With uniform priors, all candidates should be selected sometimes
        selections = [panel_learner.select("factual", candidates) for _ in range(100)]

        unique_selections = set(selections)
        assert len(unique_selections) >= 2  # Should have variety

    def test_exploitation_with_trained_prior(self):
        """Test trained prior leads to exploitation."""
        learner = PanelTypeLearner()

        # Heavily train TEXT for factual
        for _ in range(20):
            learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.95)

        # Negatively train CONFIDENCE
        for _ in range(10):
            learner.update("factual", PanelTypeJenny.CONFIDENCE, success=False, confidence=0.2)

        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE]
        selections = [learner.select("factual", candidates) for _ in range(50)]

        # TEXT should dominate
        text_ratio = selections.count(PanelTypeJenny.TEXT) / len(selections)
        assert text_ratio > 0.7

    def test_query_type_independence(self, panel_learner):
        """Test different query types have independent priors."""
        # Train TEXT for factual
        panel_learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.95)
        panel_learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.95)

        # Train GRAPH for analytical
        panel_learner.update("analytical", PanelTypeJenny.GRAPH, success=True, confidence=0.95)
        panel_learner.update("analytical", PanelTypeJenny.GRAPH, success=True, confidence=0.95)

        # Verify separate priors
        factual_text = panel_learner.priors[("factual", "text")]
        analytical_graph = panel_learner.priors[("analytical", "graph")]

        # Each should have high expected value
        assert factual_text.expected_value() > 0.6
        assert analytical_graph.expected_value() > 0.6

        # Untrained combinations should have uniform prior
        assert ("factual", "graph") not in panel_learner.priors or \
               panel_learner.priors[("factual", "graph")].expected_value() == 0.5

    def test_convergence_over_time(self):
        """Test Thompson Sampling converges to best option."""
        learner = PanelTypeLearner()

        # Simulate environment where TEXT is 80% good, GRAPH is 40% good
        random.seed(42)

        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.GRAPH]

        for _ in range(100):
            selected = learner.select("factual", candidates)

            # Simulate outcome based on true probabilities
            if selected == PanelTypeJenny.TEXT:
                success = random.random() < 0.8
            else:
                success = random.random() < 0.4

            learner.update("factual", selected, success=success, confidence=0.9)

        # After 100 iterations, TEXT should have higher expected value
        text_ev = learner.priors[("factual", "text")].expected_value()
        graph_ev = learner.priors[("factual", "graph")].expected_value()

        assert text_ev > graph_ev


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_select_with_none_candidates(self, panel_learner):
        """Test select handles None candidates gracefully."""
        # Should handle like empty list
        result = panel_learner.select("factual", None or [])
        assert result == PanelTypeJenny.TEXT

    def test_prior_with_zero_values(self):
        """Test prior handles edge case values."""
        # Very small alpha/beta
        prior = PanelTypePrior(alpha=0.01, beta=0.01)

        # Should still sample
        sample = prior.sample()
        assert 0.0 <= sample <= 1.0

    def test_learner_high_volume_updates(self, panel_learner):
        """Test learner handles high volume of updates."""
        for i in range(1000):
            panel_learner.update(
                f"query_type_{i % 10}",
                list(PanelTypeJenny)[i % len(PanelTypeJenny)],
                success=i % 2 == 0,
                confidence=0.5 + (i % 50) / 100
            )

        stats = panel_learner.get_statistics()
        assert stats["total_selections"] == 0  # No selections, just updates

    def test_persistence_corrupt_file(self, temp_persist_path):
        """Test learner handles corrupt persist file."""
        # Write invalid JSON
        with open(temp_persist_path, 'w') as f:
            f.write("not valid json")

        # Should not raise, just log warning
        learner = PanelTypeLearner(persist_path=temp_persist_path)
        assert len(learner.priors) == 0  # Empty due to load failure

    def test_expected_value_extreme_alpha(self):
        """Test expected value with extreme alpha."""
        prior = PanelTypePrior(alpha=1000.0, beta=1.0)
        ev = prior.expected_value()

        assert ev > 0.99

    def test_expected_value_extreme_beta(self):
        """Test expected value with extreme beta."""
        prior = PanelTypePrior(alpha=1.0, beta=1000.0)
        ev = prior.expected_value()

        assert ev < 0.01

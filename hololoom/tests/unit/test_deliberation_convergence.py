"""
Tests for deliberation_convergence.py — EVINCE-inspired MI control for deliberation.

Wire 7 of the structured AI integration roadmap.
"""

import numpy as np
import pytest

from hololoom.core.convergence.deliberation_convergence import (
    DeliberationConvergence,
    RoundMetrics,
    ConvergenceState,
    create_deliberation_convergence,
    _text_to_distribution,
    _align_and_jsd,
    _entropy,
)


# ============================================================================
# Pure function tests
# ============================================================================

class TestPureFunctions:
    def test_text_to_distribution_nonempty(self):
        dist = _text_to_distribution("hello world")
        assert len(dist) > 0
        assert abs(dist.sum() - 1.0) < 1e-9

    def test_text_to_distribution_empty(self):
        dist = _text_to_distribution("")
        assert len(dist) == 1
        assert dist[0] == 1.0

    def test_text_to_distribution_short(self):
        dist = _text_to_distribution("ab")  # Too short for trigrams
        assert len(dist) == 1

    def test_jsd_identical(self):
        p = np.array([0.5, 0.3, 0.2])
        assert _align_and_jsd(p, p.copy()) == pytest.approx(0.0, abs=1e-9)

    def test_jsd_bounded(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        jsd = _align_and_jsd(p, q)
        assert 0.0 <= jsd <= 1.0

    def test_jsd_different_lengths(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.3, 0.4])
        jsd = _align_and_jsd(p, q)
        assert 0.0 <= jsd <= 1.0

    def test_entropy_uniform(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        h = _entropy(p)
        assert h == pytest.approx(2.0, abs=0.01)  # log2(4) = 2 bits

    def test_entropy_certain(self):
        p = np.array([1.0, 0.0, 0.0])
        h = _entropy(p)
        assert h == pytest.approx(0.0, abs=1e-9)


# ============================================================================
# DeliberationConvergence basics
# ============================================================================

class TestDeliberationConvergenceBasics:
    def test_initial_state(self):
        tracker = DeliberationConvergence()
        assert tracker.rounds == 0
        assert tracker.latest_mi() == 1.0
        converged, _ = tracker.has_converged()
        assert converged is False

    def test_record_round(self):
        tracker = DeliberationConvergence()
        metrics = tracker.record_round(
            "The plan is to build a search engine.",
            "The plan has flaws in scalability.",
            "Revised plan addresses scalability concerns.",
        )
        assert isinstance(metrics, RoundMetrics)
        assert metrics.round_num == 1
        assert tracker.rounds == 1

    def test_round_metrics_fields(self):
        tracker = DeliberationConvergence()
        m = tracker.record_round("proposal", "critique", "consensus")
        assert m.proposal_entropy >= 0
        assert m.critique_entropy >= 0
        assert m.consensus_entropy >= 0
        assert 0.0 <= m.proposal_critique_jsd <= 1.0
        assert 0.0 <= m.argument_diversity <= 1.0

    def test_first_round_cross_jsd_is_max(self):
        """First round has no previous consensus — cross-round JSD should be 1.0."""
        tracker = DeliberationConvergence()
        m = tracker.record_round("a", "b", "c")
        assert m.cross_round_jsd == 1.0


# ============================================================================
# Convergence detection
# ============================================================================

class TestConvergenceDetection:
    def test_identical_consensus_converges(self):
        """Repeating the same consensus should trigger MI plateau."""
        tracker = DeliberationConvergence(mi_threshold=0.05, min_rounds=1)
        tracker.record_round("proposal A", "critique A", "consensus X")
        tracker.record_round("proposal B", "critique B", "consensus X")
        converged, reason = tracker.has_converged()
        assert converged is True
        assert "MI plateau" in reason or "stabilized" in reason

    def test_very_different_consensus_does_not_converge(self):
        tracker = DeliberationConvergence(mi_threshold=0.01, max_rounds=10)
        tracker.record_round(
            "Build a bridge",
            "The bridge will collapse",
            "We need stronger materials for the bridge",
        )
        tracker.record_round(
            "Launch a rocket to Mars",
            "The fuel calculations are wrong",
            "Complete redesign of propulsion system needed",
        )
        converged, _ = tracker.has_converged()
        assert converged is False

    def test_max_rounds_triggers(self):
        tracker = DeliberationConvergence(max_rounds=3, mi_threshold=0.001)
        for i in range(3):
            tracker.record_round(f"proposal {i}", f"critique {i}", f"consensus {i}")
        converged, reason = tracker.has_converged()
        assert converged is True
        assert "Max rounds" in reason

    def test_min_rounds_respected(self):
        """Even with perfect agreement, min_rounds must be met."""
        tracker = DeliberationConvergence(min_rounds=3, mi_threshold=0.5)
        tracker.record_round("same", "same", "same")
        tracker.record_round("same", "same", "same")
        converged, reason = tracker.has_converged()
        assert converged is False
        assert "Minimum" in reason

    def test_agreement_triggers(self):
        """When proposal and critique are very similar, agreement is reached."""
        tracker = DeliberationConvergence(agreement_threshold=0.3, min_rounds=1)
        tracker.record_round(
            "The answer is Thompson Sampling with Beta distributions",
            "The answer is Thompson Sampling using Beta distributions",
            "Thompson Sampling with Beta distributions is the answer",
        )
        converged, reason = tracker.has_converged()
        # The similar proposal and critique should trigger agreement
        assert converged is True
        assert "Agreement" in reason or "MI plateau" in reason


# ============================================================================
# Trajectories
# ============================================================================

class TestTrajectories:
    def test_mi_trajectory(self):
        tracker = DeliberationConvergence(max_rounds=10)
        for i in range(4):
            tracker.record_round(f"proposal {i}", f"critique {i}", f"consensus {i}")
        traj = tracker.mi_trajectory()
        assert len(traj) == 4
        assert traj[0] == 1.0  # First round

    def test_diversity_trajectory(self):
        tracker = DeliberationConvergence(max_rounds=10)
        tracker.record_round("aaa", "bbb", "ccc")
        tracker.record_round("xxx", "yyy", "zzz")
        traj = tracker.diversity_trajectory()
        assert len(traj) == 2
        assert all(0.0 <= d <= 1.0 for d in traj)

    def test_convergence_decreasing_mi(self):
        """Simulate a converging debate: consensus stabilizes over rounds."""
        tracker = DeliberationConvergence(mi_threshold=0.05, max_rounds=10, min_rounds=2)

        # Round 1: very different
        tracker.record_round(
            "We should build a monolith",
            "Monoliths are terrible for scaling",
            "Consider a hybrid approach with clear module boundaries",
        )

        # Round 2: closer — both discuss the hybrid
        tracker.record_round(
            "The hybrid approach needs protocol-first design",
            "Agreed on protocols, but monitoring is missing",
            "Consider a hybrid approach with clear module boundaries and monitoring",
        )

        # Round 3: consensus stabilizes
        tracker.record_round(
            "Protocol-first hybrid with monitoring",
            "Looks good, minor concern about latency",
            "Consider a hybrid approach with clear module boundaries and monitoring",
        )

        # MI should be decreasing
        traj = tracker.mi_trajectory()
        assert traj[0] > traj[-1], f"MI should decrease: {traj}"


# ============================================================================
# Custom embedding function
# ============================================================================

class TestCustomEmbedding:
    def test_embedding_fn_used(self):
        call_count = [0]

        def mock_embedding(text):
            call_count[0] += 1
            return np.array([hash(text) % 100, len(text), 42.0])

        tracker = DeliberationConvergence(embedding_fn=mock_embedding)
        tracker.record_round("a", "b", "c")
        assert call_count[0] == 3  # Called for proposal, critique, consensus


# ============================================================================
# State export
# ============================================================================

class TestState:
    def test_state_export(self):
        tracker = DeliberationConvergence(max_rounds=5)
        tracker.record_round("p", "c", "s")
        tracker.record_round("p2", "c2", "s2")

        state = tracker.state()
        assert isinstance(state, ConvergenceState)
        assert state.rounds_completed == 2
        assert len(state.mi_trajectory) == 2

    def test_state_reflects_convergence(self):
        tracker = DeliberationConvergence(max_rounds=2)
        tracker.record_round("a", "b", "c")
        tracker.record_round("d", "e", "f")
        state = tracker.state()
        assert state.converged is True
        assert "Max rounds" in state.convergence_reason


# ============================================================================
# Reset
# ============================================================================

class TestReset:
    def test_reset_clears_state(self):
        tracker = DeliberationConvergence()
        tracker.record_round("a", "b", "c")
        assert tracker.rounds == 1

        tracker.reset()
        assert tracker.rounds == 0
        assert tracker.latest_mi() == 1.0


# ============================================================================
# Factory
# ============================================================================

class TestFactory:
    def test_create_deliberation_convergence(self):
        tracker = create_deliberation_convergence(mi_threshold=0.1, max_rounds=3)
        assert isinstance(tracker, DeliberationConvergence)
        assert tracker.mi_threshold == 0.1
        assert tracker.max_rounds == 3

    def test_create_defaults(self):
        tracker = create_deliberation_convergence()
        assert tracker.mi_threshold == 0.05
        assert tracker.max_rounds == 5

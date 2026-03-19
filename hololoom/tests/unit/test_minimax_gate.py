"""
Tests for minimax_gate.py — Game-theoretic decision verification.

Wire 3 of the structured AI integration roadmap.
"""

import numpy as np
import pytest

from hololoom.core.convergence.minimax_gate import (
    MinimaxGate,
    GateResult,
    MinimaxGateProtocol,
    create_minimax_gate,
)


# ============================================================================
# GateResult
# ============================================================================

class TestGateResult:
    def test_fields(self):
        r = GateResult(
            verified=True, ts_tool=0, minimax_tool=0,
            ts_payoff=1.0, minimax_payoff=1.0, regret=0.0, detail="ok"
        )
        assert r.verified is True
        assert r.regret == 0.0


# ============================================================================
# MinimaxGate basics
# ============================================================================

class TestMinimaxGateBasics:
    def test_protocol_compliance(self):
        g = MinimaxGate()
        assert isinstance(g, MinimaxGateProtocol)

    def test_invalid_payoff_matrix(self):
        g = MinimaxGate()
        result = g.verify(0, np.array([1, 2, 3]))  # 1D, not 2D
        assert result.verified is True
        assert "Invalid" in result.detail

    def test_tool_index_out_of_range(self):
        g = MinimaxGate()
        payoffs = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = g.verify(5, payoffs)
        assert result.verified is True
        assert "out of range" in result.detail


# ============================================================================
# Minimax computation
# ============================================================================

class TestMinimaxComputation:
    def test_ts_matches_minimax_verified(self):
        """When TS picks the minimax-optimal tool, gate verifies."""
        g = MinimaxGate()
        # Tool 0: worst case = 1, Tool 1: worst case = 2
        # Minimax picks tool 1
        payoffs = np.array([
            [1.0, 3.0],  # tool 0: min=1
            [2.0, 2.0],  # tool 1: min=2  <-- minimax
        ])
        result = g.verify(selected_tool=1, tool_payoffs=payoffs)
        assert result.verified is True
        assert result.minimax_tool == 1
        assert result.regret == pytest.approx(0.0)

    def test_ts_diverges_from_minimax(self):
        """When TS picks a non-minimax tool, gate flags divergence."""
        g = MinimaxGate()
        payoffs = np.array([
            [1.0, 3.0],  # tool 0: min=1
            [2.0, 2.0],  # tool 1: min=2  <-- minimax
        ])
        result = g.verify(selected_tool=0, tool_payoffs=payoffs)
        assert result.verified is False
        assert result.ts_tool == 0
        assert result.minimax_tool == 1
        assert result.regret == pytest.approx(1.0)  # 2 - 1

    def test_regret_threshold_allows_small_divergence(self):
        """With regret_threshold > 0, small divergences pass."""
        g = MinimaxGate(regret_threshold=0.5)
        payoffs = np.array([
            [1.5, 3.0],  # tool 0: min=1.5
            [2.0, 2.0],  # tool 1: min=2  <-- minimax
        ])
        result = g.verify(selected_tool=0, tool_payoffs=payoffs)
        # Regret = 2.0 - 1.5 = 0.5, equals threshold → verified
        assert result.verified is True

    def test_regret_threshold_blocks_large_divergence(self):
        g = MinimaxGate(regret_threshold=0.3)
        payoffs = np.array([
            [1.0, 3.0],  # tool 0: min=1
            [2.0, 2.0],  # tool 1: min=2
        ])
        result = g.verify(selected_tool=0, tool_payoffs=payoffs)
        # Regret = 1.0 > 0.3
        assert result.verified is False

    def test_three_tools(self):
        """Works with more than 2 tools."""
        g = MinimaxGate()
        payoffs = np.array([
            [0.5, 3.0, 1.0],  # tool 0: min=0.5
            [1.0, 1.0, 1.0],  # tool 1: min=1.0  <-- minimax
            [0.0, 5.0, 0.0],  # tool 2: min=0.0
        ])
        result = g.verify(selected_tool=2, tool_payoffs=payoffs)
        assert result.verified is False
        assert result.minimax_tool == 1
        assert result.regret == pytest.approx(1.0)

    def test_dominant_strategy(self):
        """When one tool dominates all others in all states."""
        g = MinimaxGate()
        payoffs = np.array([
            [5.0, 5.0, 5.0],  # tool 0: dominant
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ])
        result = g.verify(selected_tool=0, tool_payoffs=payoffs)
        assert result.verified is True
        assert result.minimax_tool == 0

    def test_all_tools_equal(self):
        """When all tools have the same worst case, any selection is verified."""
        g = MinimaxGate()
        payoffs = np.array([
            [2.0, 2.0],
            [2.0, 2.0],
        ])
        result = g.verify(selected_tool=1, tool_payoffs=payoffs)
        assert result.verified is True
        assert result.regret == pytest.approx(0.0)


# ============================================================================
# State priors (optional enrichment)
# ============================================================================

class TestStatePriors:
    def test_state_priors_included_in_detail(self):
        g = MinimaxGate()
        payoffs = np.array([
            [1.0, 3.0],
            [2.0, 2.0],
        ])
        priors = np.array([0.7, 0.3])  # State 0 more likely
        result = g.verify(selected_tool=0, tool_payoffs=payoffs, state_priors=priors)
        assert "expected under priors" in result.detail

    def test_state_priors_wrong_length_ignored(self):
        g = MinimaxGate()
        payoffs = np.array([
            [1.0, 3.0],
            [2.0, 2.0],
        ])
        priors = np.array([0.5])  # Wrong length
        result = g.verify(selected_tool=0, tool_payoffs=payoffs, state_priors=priors)
        assert "expected under priors" not in result.detail

    def test_priors_do_not_affect_minimax(self):
        """Priors are informational only — minimax is worst-case, not expected-case."""
        g = MinimaxGate()
        payoffs = np.array([
            [0.0, 10.0],  # tool 0: great in state 1, terrible in state 0
            [4.0, 4.0],   # tool 1: stable
        ])
        priors = np.array([0.01, 0.99])  # State 1 almost certain
        result = g.verify(selected_tool=0, tool_payoffs=payoffs, state_priors=priors)
        # Minimax still picks tool 1 (worst-case 4 > worst-case 0)
        assert result.minimax_tool == 1
        assert result.verified is False


# ============================================================================
# Prisoners dilemma style scenarios
# ============================================================================

class TestGameScenarios:
    def test_risk_averse_vs_risk_seeking(self):
        """Classic risk scenario: safe tool vs high-variance tool."""
        g = MinimaxGate()
        payoffs = np.array([
            [3.0, 3.0],   # tool 0: safe, consistent
            [0.0, 10.0],  # tool 1: risky, high ceiling
        ])
        # Minimax picks safe tool (worst-case 3 > worst-case 0)
        result = g.verify(selected_tool=1, tool_payoffs=payoffs)
        assert result.verified is False
        assert result.minimax_tool == 0
        assert result.detail  # Has explanation

    def test_no_good_options(self):
        """All tools have poor worst-case — minimax picks least-bad."""
        g = MinimaxGate()
        payoffs = np.array([
            [-5.0, -1.0],   # tool 0: min=-5
            [-3.0, -3.0],   # tool 1: min=-3  <-- least bad
            [-10.0, 0.0],   # tool 2: min=-10
        ])
        result = g.verify(selected_tool=1, tool_payoffs=payoffs)
        assert result.verified is True
        assert result.minimax_tool == 1


# ============================================================================
# Factory
# ============================================================================

class TestFactory:
    def test_create_minimax_gate(self):
        g = create_minimax_gate(regret_threshold=0.5)
        assert isinstance(g, MinimaxGate)
        assert g.regret_threshold == 0.5

    def test_create_minimax_gate_default(self):
        g = create_minimax_gate()
        assert g.regret_threshold == 0.0

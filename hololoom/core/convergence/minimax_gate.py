"""
Minimax Verification Gate — Game-Theoretic Decision Verification
================================================================

Sits between the convergence engine (Step 7) and tool execution (Step 8).
Constructs a game matrix of tools × environment states, solves for the
minimax-optimal tool. If it diverges from the Thompson Sampling selection,
flags the decision for review.

This is Wire 3 of the structured AI integration roadmap.

The gate is CHEAP: for 10 tools × 20 states, the computation is <1ms.
It only fires (logs a warning) when minimax and TS disagree.

Usage:
    gate = MinimaxGate()
    result = gate.verify(
        selected_tool=2,
        tool_payoffs=payoff_matrix,  # (n_tools, n_states)
    )
    if not result.verified:
        logger.warning(f"Minimax disagrees: TS chose {result.ts_tool}, minimax prefers {result.minimax_tool}")

Author: Claude Code (Structured AI Wiring - Wire 3)
Date: 2026-03-18
"""

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol
# ============================================================================

@runtime_checkable
class MinimaxGateProtocol(Protocol):
    """Protocol for minimax verification. Implementations are swappable."""

    def verify(
        self,
        selected_tool: int,
        tool_payoffs: np.ndarray,
        state_priors: np.ndarray | None,
    ) -> 'GateResult':
        ...


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class GateResult:
    """Result of minimax verification."""
    verified: bool          # True if TS selection matches minimax recommendation
    ts_tool: int            # Tool selected by Thompson Sampling
    minimax_tool: int       # Tool recommended by minimax analysis
    ts_payoff: float        # Worst-case payoff for TS selection
    minimax_payoff: float   # Worst-case payoff for minimax selection
    regret: float           # Difference in worst-case payoffs (minimax - ts)
    detail: str             # Human-readable explanation


# ============================================================================
# Minimax Gate
# ============================================================================

class MinimaxGate:
    """
    Game-theoretic verification gate for tool selection decisions.

    Given a payoff matrix (tools × environment states), computes the
    maximin tool — the tool that maximizes the minimum payoff across
    all possible environment states. This is the "safest" choice under
    worst-case uncertainty.

    If the Thompson Sampling selection matches the maximin tool, the
    decision is "verified." If they disagree, the gate reports the
    divergence and the regret (how much worse the TS choice could be
    in the worst case).

    The gate does NOT override Thompson Sampling — it provides an
    independent signal. TS optimizes expected value; minimax optimizes
    worst-case value. Divergence between them is informative.
    """

    def __init__(self, regret_threshold: float = 0.0):
        """
        Args:
            regret_threshold: Minimum regret to flag as divergence.
                            0.0 = flag any disagreement (default).
                            Higher values ignore small divergences.
        """
        self.regret_threshold = regret_threshold
        logger.info(f"MinimaxGate initialized (regret_threshold={regret_threshold})")

    def verify(
        self,
        selected_tool: int,
        tool_payoffs: np.ndarray,
        state_priors: np.ndarray | None = None,
    ) -> GateResult:
        """
        Verify tool selection against minimax analysis.

        Args:
            selected_tool: Tool index chosen by Thompson Sampling
            tool_payoffs: Payoff matrix of shape (n_tools, n_states).
                         Entry [i, j] = expected payoff of tool i in state j.
            state_priors: Optional prior distribution over states.
                         If provided, computes expected payoff under priors
                         as a secondary check. Does not affect minimax computation.

        Returns:
            GateResult with verification status and analysis
        """
        tool_payoffs = np.asarray(tool_payoffs, dtype=np.float64)

        if tool_payoffs.ndim != 2:
            return GateResult(
                verified=True,
                ts_tool=selected_tool,
                minimax_tool=selected_tool,
                ts_payoff=0.0,
                minimax_payoff=0.0,
                regret=0.0,
                detail="Invalid payoff matrix (need 2D), skipping verification",
            )

        n_tools, n_states = tool_payoffs.shape

        if selected_tool < 0 or selected_tool >= n_tools:
            return GateResult(
                verified=True,
                ts_tool=selected_tool,
                minimax_tool=selected_tool,
                ts_payoff=0.0,
                minimax_payoff=0.0,
                regret=0.0,
                detail=f"Tool index {selected_tool} out of range [0, {n_tools}), skipping",
            )

        # Maximin: for each tool, find its worst-case payoff, then pick the
        # tool with the best worst-case payoff.
        worst_case_payoffs = tool_payoffs.min(axis=1)  # (n_tools,)
        minimax_tool = int(np.argmax(worst_case_payoffs))
        minimax_payoff = float(worst_case_payoffs[minimax_tool])

        ts_payoff = float(worst_case_payoffs[selected_tool])
        regret = minimax_payoff - ts_payoff

        verified = (selected_tool == minimax_tool) or (regret <= self.regret_threshold)

        detail_parts = [
            f"TS selected tool {selected_tool} (worst-case={ts_payoff:.3f})",
            f"minimax recommends tool {minimax_tool} (worst-case={minimax_payoff:.3f})",
            f"regret={regret:.3f}",
        ]

        if state_priors is not None:
            state_priors = np.asarray(state_priors, dtype=np.float64)
            if len(state_priors) == n_states:
                expected_ts = float(tool_payoffs[selected_tool] @ state_priors)
                expected_mm = float(tool_payoffs[minimax_tool] @ state_priors)
                detail_parts.append(
                    f"expected under priors: TS={expected_ts:.3f}, minimax={expected_mm:.3f}"
                )

        if not verified:
            logger.warning(f"Minimax gate divergence: {'; '.join(detail_parts)}")
        else:
            logger.debug(f"Minimax gate verified: {'; '.join(detail_parts)}")

        return GateResult(
            verified=verified,
            ts_tool=selected_tool,
            minimax_tool=minimax_tool,
            ts_payoff=ts_payoff,
            minimax_payoff=minimax_payoff,
            regret=regret,
            detail="; ".join(detail_parts),
        )


# ============================================================================
# Factory
# ============================================================================

def create_minimax_gate(regret_threshold: float = 0.0) -> MinimaxGate:
    """Create a minimax verification gate."""
    return MinimaxGate(regret_threshold=regret_threshold)

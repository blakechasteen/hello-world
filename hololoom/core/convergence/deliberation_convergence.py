"""
Deliberation Convergence — EVINCE-Inspired Information-Theoretic Control
========================================================================

Wire 7 of the structured AI integration roadmap.

Applies the EVINCE dual-entropy pattern to HoloLoom's deliberation cycle.
Measures information flow between Propose (explorer/high entropy) and
Challenge (confirmer/low entropy) phases. When mutual information between
them plateaus, the deliberation has converged — further rounds won't
produce new insights.

Based on: EVINCE (arXiv 2408.14575, 2024) — improved Top-1 accuracy
by ~7% on medical diagnosis via MI-controlled multi-LLM debate.

Key insight: the Planner is naturally high-entropy (exploratory, temp=0.7)
and the Critic is naturally low-entropy (focused, temp=0.4). This maps
directly to EVINCE's explorer/confirmer roles.

Usage:
    tracker = DeliberationConvergence(threshold=0.05)

    # After each deliberation round:
    tracker.record_round(proposal_text, critique_text, consensus_text)

    if tracker.has_converged():
        print("Deliberation converged — further rounds won't help")
        print(f"Rounds: {tracker.rounds}, Final MI: {tracker.latest_mi():.4f}")

Author: Claude Code (Structured AI Wiring - Wire 7)
Date: 2026-03-18
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class RoundMetrics:
    """Information-theoretic metrics for a single deliberation round."""
    round_num: int
    proposal_entropy: float     # H(proposal) — explorer diversity
    critique_entropy: float     # H(critique) — confirmer focus
    consensus_entropy: float    # H(consensus) — resolution diversity
    proposal_critique_jsd: float  # JSD(proposal, critique) — disagreement
    cross_round_jsd: float      # JSD(consensus_n, consensus_n-1) — convergence signal
    argument_diversity: float   # JSD across all three phases — debate health


@dataclass
class ConvergenceState:
    """Overall convergence state of the deliberation."""
    converged: bool
    rounds_completed: int
    convergence_reason: str
    mi_trajectory: list[float]  # Cross-round JSD values over time
    final_diversity: float      # How diverse the final consensus was


# ============================================================================
# Core Functions (pure, no side effects)
# ============================================================================

def _text_to_distribution(text: str) -> np.ndarray:
    """
    Convert text to a probability distribution via character trigram frequencies.

    This is a lightweight, dependency-free approach. For production use,
    pass an embedding_fn to DeliberationConvergence.
    """
    if not text or len(text) < 3:
        return np.array([1.0])

    trigrams = {}
    text_lower = text.lower()
    for i in range(len(text_lower) - 2):
        tri = text_lower[i:i+3]
        trigrams[tri] = trigrams.get(tri, 0) + 1

    if not trigrams:
        return np.array([1.0])

    total = sum(trigrams.values())
    keys = sorted(trigrams.keys())
    return np.array([trigrams[k] / total for k in keys])


def _align_and_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Compute JSD between two potentially different-length distributions."""
    if len(p) == len(q):
        pass
    else:
        max_len = max(len(p), len(q))
        eps = 1e-10
        p_new = np.full(max_len, eps)
        q_new = np.full(max_len, eps)
        p_new[:len(p)] = p
        q_new[:len(q)] = q
        p = p_new / p_new.sum()
        q = q_new / q_new.sum()

    m = 0.5 * (p + q)
    # Safe log: only compute where both numerator and denominator are positive
    eps = 1e-12
    m_safe = np.maximum(m, eps)
    kl_pm = np.sum(np.where(p > eps, p * np.log2(p / m_safe), 0.0))
    kl_qm = np.sum(np.where(q > eps, q * np.log2(q / m_safe), 0.0))
    return float(np.clip(0.5 * kl_pm + 0.5 * kl_qm, 0.0, 1.0))


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits."""
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ============================================================================
# Deliberation Convergence Tracker
# ============================================================================

class DeliberationConvergence:
    """
    Tracks information-theoretic convergence across deliberation rounds.

    The EVINCE pattern:
    1. Planner (explorer) has naturally high entropy (creative, divergent)
    2. Critic (confirmer) has naturally low entropy (focused, convergent)
    3. As rounds progress, their JSD decreases — they're agreeing more
    4. When cross-round JSD drops below threshold, deliberation has converged

    Three convergence signals (any triggers stop):
    - **MI plateau**: JSD between consecutive consensus outputs < threshold
    - **Agreement**: JSD between proposal and critique < agreement_threshold
    - **Max rounds**: Hard ceiling to prevent infinite deliberation
    """

    def __init__(
        self,
        mi_threshold: float = 0.05,
        agreement_threshold: float = 0.1,
        max_rounds: int = 5,
        min_rounds: int = 1,
        embedding_fn=None,
    ):
        """
        Args:
            mi_threshold: JSD threshold for cross-round convergence (default: 0.05)
            agreement_threshold: JSD threshold for proposal-critique agreement (default: 0.1)
            max_rounds: Maximum deliberation rounds (default: 5)
            min_rounds: Minimum rounds before checking convergence (default: 1)
            embedding_fn: Optional callable(text) -> np.ndarray for richer distributions
        """
        self.mi_threshold = mi_threshold
        self.agreement_threshold = agreement_threshold
        self.max_rounds = max_rounds
        self.min_rounds = min_rounds
        self.embedding_fn = embedding_fn

        self._rounds: list[RoundMetrics] = []
        self._prev_consensus_dist: np.ndarray | None = None

        logger.info(
            f"DeliberationConvergence initialized "
            f"(mi_threshold={mi_threshold}, agreement={agreement_threshold}, "
            f"max_rounds={max_rounds})"
        )

    @property
    def rounds(self) -> int:
        """Number of rounds recorded."""
        return len(self._rounds)

    def _to_dist(self, text: str) -> np.ndarray:
        """Convert text to distribution using embedding_fn or fallback."""
        if self.embedding_fn is not None:
            emb = np.asarray(self.embedding_fn(text), dtype=np.float64).flatten()
            exp_x = np.exp(emb - np.max(emb))
            return exp_x / exp_x.sum()
        return _text_to_distribution(text)

    def record_round(
        self,
        proposal_text: str,
        critique_text: str,
        consensus_text: str,
    ) -> RoundMetrics:
        """
        Record a deliberation round and compute information-theoretic metrics.

        Args:
            proposal_text: Planner output (Phase 1)
            critique_text: Critic output (Phase 2)
            consensus_text: Synthesizer output (Phase 3)

        Returns:
            RoundMetrics for this round
        """
        p_dist = self._to_dist(proposal_text)
        c_dist = self._to_dist(critique_text)
        s_dist = self._to_dist(consensus_text)

        # Compute metrics
        proposal_entropy = _entropy(p_dist)
        critique_entropy = _entropy(c_dist)
        consensus_entropy = _entropy(s_dist)

        # Disagreement: how different are proposal and critique?
        proposal_critique_jsd = _align_and_jsd(p_dist, c_dist)

        # Cross-round convergence: how different is this consensus from last round's?
        if self._prev_consensus_dist is not None:
            cross_round_jsd = _align_and_jsd(s_dist, self._prev_consensus_dist)
        else:
            cross_round_jsd = 1.0  # First round: maximum divergence by convention

        # Argument diversity: average pairwise JSD across all three phases
        jsd_pc = proposal_critique_jsd
        jsd_ps = _align_and_jsd(p_dist, s_dist)
        jsd_cs = _align_and_jsd(c_dist, s_dist)
        argument_diversity = (jsd_pc + jsd_ps + jsd_cs) / 3.0

        self._prev_consensus_dist = s_dist

        metrics = RoundMetrics(
            round_num=len(self._rounds) + 1,
            proposal_entropy=proposal_entropy,
            critique_entropy=critique_entropy,
            consensus_entropy=consensus_entropy,
            proposal_critique_jsd=proposal_critique_jsd,
            cross_round_jsd=cross_round_jsd,
            argument_diversity=argument_diversity,
        )

        self._rounds.append(metrics)

        logger.debug(
            f"Round {metrics.round_num}: "
            f"H(P)={proposal_entropy:.3f}, H(C)={critique_entropy:.3f}, "
            f"JSD(P,C)={proposal_critique_jsd:.4f}, "
            f"cross_round_JSD={cross_round_jsd:.4f}, "
            f"diversity={argument_diversity:.4f}"
        )

        return metrics

    def has_converged(self) -> tuple[bool, str]:
        """
        Check if deliberation has converged.

        Returns:
            (converged, reason) tuple
        """
        if not self._rounds:
            return False, "No rounds recorded"

        if len(self._rounds) < self.min_rounds:
            return False, f"Minimum rounds not reached ({len(self._rounds)}/{self.min_rounds})"

        latest = self._rounds[-1]

        # Signal 1: Max rounds
        if len(self._rounds) >= self.max_rounds:
            return True, f"Max rounds reached ({self.max_rounds})"

        # Signal 2: Cross-round MI plateau
        if len(self._rounds) >= 2 and latest.cross_round_jsd < self.mi_threshold:
            return True, (
                f"MI plateau: cross-round JSD={latest.cross_round_jsd:.4f} "
                f"< threshold={self.mi_threshold} (consensus stabilized)"
            )

        # Signal 3: Proposal-critique agreement
        if latest.proposal_critique_jsd < self.agreement_threshold:
            return True, (
                f"Agreement reached: JSD(proposal, critique)={latest.proposal_critique_jsd:.4f} "
                f"< threshold={self.agreement_threshold}"
            )

        return False, (
            f"Not converged: cross_round_JSD={latest.cross_round_jsd:.4f}, "
            f"agreement_JSD={latest.proposal_critique_jsd:.4f}"
        )

    def latest_mi(self) -> float:
        """Return the latest cross-round JSD (convergence signal)."""
        if not self._rounds:
            return 1.0
        return self._rounds[-1].cross_round_jsd

    def mi_trajectory(self) -> list[float]:
        """Return the trajectory of cross-round JSD values."""
        return [r.cross_round_jsd for r in self._rounds]

    def diversity_trajectory(self) -> list[float]:
        """Return argument diversity over rounds."""
        return [r.argument_diversity for r in self._rounds]

    def state(self) -> ConvergenceState:
        """Export full convergence state."""
        converged, reason = self.has_converged()
        return ConvergenceState(
            converged=converged,
            rounds_completed=len(self._rounds),
            convergence_reason=reason,
            mi_trajectory=self.mi_trajectory(),
            final_diversity=self._rounds[-1].argument_diversity if self._rounds else 0.0,
        )

    def reset(self) -> None:
        """Reset tracker for a new deliberation session."""
        self._rounds.clear()
        self._prev_consensus_dist = None


# ============================================================================
# Factory
# ============================================================================

def create_deliberation_convergence(
    mi_threshold: float = 0.05,
    agreement_threshold: float = 0.1,
    max_rounds: int = 5,
    embedding_fn=None,
) -> DeliberationConvergence:
    """Create a deliberation convergence tracker."""
    return DeliberationConvergence(
        mi_threshold=mi_threshold,
        agreement_threshold=agreement_threshold,
        max_rounds=max_rounds,
        embedding_fn=embedding_fn,
    )

"""
Self-Feedback — the system generates its own learning signals.

Four diagnostics that let the feedback system evaluate itself:

1. ConvergenceDiagnostic: detects stalled bandits (arms not moving)
2. ContradictionDetector: spots disagreements across timescales
3. RetroactiveUpdater: traces promotions back through causation chains
4. SelfEvaluator: re-scores past decisions for consistency checking

These run on collected signals and produce new FeedbackSignals — the system
watching itself learn, and adjusting when it notices something off.

Author: Claude Code (with HoloLoom by Blake)
Date: 2026-03-19
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

try:
    from hololoom.core.protocols.feedback import (
        FeedbackSignal,
        Timescale,
        TIMESCALE_WEIGHT,
    )
except ImportError:
    from hololoom.protocols.feedback import (
        FeedbackSignal,
        Timescale,
        TIMESCALE_WEIGHT,
    )

from hololoom.core.reflection.causation import CausationTracker

logger = logging.getLogger(__name__)


# ============================================================================
# Convergence Diagnostic
# ============================================================================

class ConvergenceDiagnostic:
    """
    Monitors bandits for stalled learning.

    Tracks arm means over a sliding window. If the mean hasn't changed by more
    than stall_threshold across the window, emit a "convergence_stalled" signal.
    This is the system noticing it stopped learning — a meta-feedback signal
    that can trigger exploration boosts.
    """

    def __init__(self, window_size: int = 20, stall_threshold: float = 0.01):
        """
        Args:
            window_size: Number of observations before checking for stalls.
            stall_threshold: Minimum mean change to consider "learning".
        """
        self._window_size = window_size
        self._stall_threshold = stall_threshold
        # arm_key -> list of (timestamp, mean) observations
        self._history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

    def observe(self, arm_key: str, mean: float) -> None:
        """Record a new observation of an arm's mean."""
        self._history[arm_key].append((datetime.now(), mean))
        # Keep only the window
        if len(self._history[arm_key]) > self._window_size * 2:
            self._history[arm_key] = self._history[arm_key][-self._window_size:]

    def check(self, bandit_arms: dict[str, Any] | None = None) -> list[FeedbackSignal]:
        """
        Check all tracked arms for convergence stalls.

        If bandit_arms is provided, observe current means before checking.
        bandit_arms should map key -> arm object with successes/failures attrs.

        Returns FeedbackSignals for any stalled arms.
        """
        if bandit_arms is not None:
            for key, arm in bandit_arms.items():
                successes = getattr(arm, "successes", getattr(arm, "alpha", 1.0))
                failures = getattr(arm, "failures", getattr(arm, "beta", 1.0))
                total = successes + failures
                if total > 2.0:
                    key_str = ":".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
                    self.observe(key_str, successes / total)

        signals: list[FeedbackSignal] = []

        for arm_key, observations in self._history.items():
            if len(observations) < self._window_size:
                continue

            recent = observations[-self._window_size:]
            means = [m for _, m in recent]

            # Check if means are stalled
            mean_range = max(means) - min(means)
            if mean_range < self._stall_threshold:
                current_mean = means[-1]
                signals.append(FeedbackSignal(
                    source="convergence_diagnostic",
                    timescale=Timescale.MINUTES,
                    subject=arm_key,
                    dimension="convergence_stalled",
                    value=-0.3,  # Negative: stalling is undesirable
                    confidence=0.7,
                    metadata={
                        "mean_range": mean_range,
                        "current_mean": current_mean,
                        "window_size": self._window_size,
                        "threshold": self._stall_threshold,
                        "observation_count": len(observations),
                    },
                ))
                logger.debug(
                    "Convergence stall detected: %s (range=%.5f, mean=%.4f)",
                    arm_key, mean_range, current_mean,
                )

        return signals

    def reset(self, arm_key: str | None = None) -> None:
        """Reset history for one arm or all arms."""
        if arm_key is None:
            self._history.clear()
        else:
            self._history.pop(arm_key, None)


# ============================================================================
# Contradiction Detector
# ============================================================================

class ContradictionDetector:
    """
    Detects conflicting signals across timescales.

    When fast timescale signals (sub_second, seconds) say "good" but slow
    timescale signals (days, weeks) say "bad" for the same subject, something
    is wrong. Maybe the tool performs well per-interaction but produces poor
    long-term outcomes. This detector emits "contradiction" signals that
    should trigger deeper investigation.
    """

    def __init__(
        self,
        fast_timescales: tuple[Timescale, ...] = (Timescale.SUB_SECOND, Timescale.SECONDS),
        slow_timescales: tuple[Timescale, ...] = (Timescale.DAYS, Timescale.WEEKS),
        contradiction_threshold: float = 0.5,
    ):
        """
        Args:
            fast_timescales: Timescales considered "fast".
            slow_timescales: Timescales considered "slow".
            contradiction_threshold: Minimum magnitude difference to flag.
                If fast says +0.6 and slow says -0.3, difference is 0.9 > 0.5.
        """
        self._fast = set(fast_timescales)
        self._slow = set(slow_timescales)
        self._threshold = contradiction_threshold

    def check(self, recent_signals: list[FeedbackSignal]) -> list[FeedbackSignal]:
        """
        Check recent signals for cross-timescale contradictions.

        Groups signals by subject, compares fast vs slow aggregate.
        Returns contradiction signals for subjects with sign disagreement
        and magnitude above threshold.
        """
        # Group by subject
        by_subject: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"fast": [], "slow": []}
        )

        for sig in recent_signals:
            if sig.timescale in self._fast:
                by_subject[sig.subject]["fast"].append(sig.value)
            elif sig.timescale in self._slow:
                by_subject[sig.subject]["slow"].append(sig.value)

        signals: list[FeedbackSignal] = []

        for subject, groups in by_subject.items():
            fast_vals = groups["fast"]
            slow_vals = groups["slow"]

            if not fast_vals or not slow_vals:
                continue

            fast_avg = sum(fast_vals) / len(fast_vals)
            slow_avg = sum(slow_vals) / len(slow_vals)

            # Check for sign disagreement with sufficient magnitude
            if fast_avg * slow_avg < 0:  # Different signs
                magnitude = abs(fast_avg - slow_avg)
                if magnitude >= self._threshold:
                    signals.append(FeedbackSignal(
                        source="contradiction_detector",
                        timescale=Timescale.HOURS,
                        subject=subject,
                        dimension="contradiction",
                        value=-abs(magnitude),  # Contradictions are negative
                        confidence=min(
                            1.0,
                            (len(fast_vals) + len(slow_vals)) / 10.0,
                        ),
                        metadata={
                            "fast_avg": fast_avg,
                            "slow_avg": slow_avg,
                            "fast_count": len(fast_vals),
                            "slow_count": len(slow_vals),
                            "magnitude": magnitude,
                        },
                    ))
                    logger.info(
                        "Contradiction detected for %s: fast=%.3f, slow=%.3f (mag=%.3f)",
                        subject, fast_avg, slow_avg, magnitude,
                    )

        return signals


# ============================================================================
# Retroactive Updater
# ============================================================================

class RetroactiveUpdater:
    """
    Generates retroactive feedback when promotion events occur.

    When a federation note gets promoted (a delayed positive signal), this
    walks the causation chain backward and generates credit signals for
    every decision that contributed to producing that note.
    """

    def __init__(self, causation_tracker: CausationTracker):
        self._causation = causation_tracker

    def on_promotion(
        self,
        agent_id: str,
        note_path: str,
        causation_chain: list[str] | None = None,
    ) -> list[FeedbackSignal]:
        """
        Generate retroactive credit signals when a note is promoted.

        Args:
            agent_id: The agent whose note was promoted.
            note_path: Path to the promoted note.
            causation_chain: Optional chain of decision IDs that led to this note.

        Returns:
            Credit-assignment FeedbackSignals for upstream decisions.
        """
        chain = causation_chain or []

        # Create the promotion outcome signal
        outcome = FeedbackSignal(
            source="retroactive_updater",
            timescale=Timescale.DAYS,
            subject=note_path,
            dimension="promotion",
            value=0.9,  # Promotions are strongly positive
            confidence=0.95,
            causation_chain=chain,
            metadata={"agent_id": agent_id},
        )

        # Walk the causation chain
        credit_signals = self._causation.assign_credit(outcome)

        if credit_signals:
            logger.info(
                "Retroactive update: promotion of %s generated %d credit signals",
                note_path, len(credit_signals),
            )

        return credit_signals

    def on_rejection(
        self,
        agent_id: str,
        note_path: str,
        reason: str = "",
        causation_chain: list[str] | None = None,
    ) -> list[FeedbackSignal]:
        """
        Generate retroactive negative credit when a note is rejected.

        Softer than promotion (lower magnitude) because rejection could be
        subjective or contextual, not necessarily a system failure.
        """
        chain = causation_chain or []

        outcome = FeedbackSignal(
            source="retroactive_updater",
            timescale=Timescale.DAYS,
            subject=note_path,
            dimension="rejection",
            value=-0.4,  # Moderate negative signal
            confidence=0.6,
            causation_chain=chain,
            metadata={"agent_id": agent_id, "reason": reason},
        )

        return self._causation.assign_credit(outcome)


# ============================================================================
# Self-Evaluator
# ============================================================================

class SelfEvaluator:
    """
    Periodically re-evaluates past decisions for consistency.

    Given an evaluation function, re-scores recent decisions and compares
    the new score to the original. If they diverge significantly, emits
    "consistency" signals — the system checking whether it would make
    the same decision again with current knowledge.

    This detects concept drift: if the system's values have changed,
    old decisions that seemed good might now score poorly.
    """

    def __init__(
        self,
        evaluation_fn: Callable[[str, dict], float],
        consistency_threshold: float = 0.3,
    ):
        """
        Args:
            evaluation_fn: (chosen, context) -> score in [-1, 1].
                Re-evaluates a past decision with current knowledge.
            consistency_threshold: Minimum divergence to emit a signal.
        """
        self._eval_fn = evaluation_fn
        self._threshold = consistency_threshold
        # Recent decisions to re-evaluate: (chosen, context, original_score, timestamp)
        self._decisions: list[tuple[str, dict, float, datetime]] = []
        self._max_stored = 100

    def record_for_evaluation(
        self, chosen: str, context: dict, original_score: float
    ) -> None:
        """Record a decision for future re-evaluation."""
        self._decisions.append((chosen, context, original_score, datetime.now()))
        if len(self._decisions) > self._max_stored:
            self._decisions = self._decisions[-self._max_stored:]

    def evaluate_recent(self, n: int = 5) -> list[FeedbackSignal]:
        """
        Re-evaluate the N most recent decisions.

        Returns consistency signals for decisions where the re-evaluation
        diverges from the original score by more than the threshold.
        """
        if not self._decisions:
            return []

        signals: list[FeedbackSignal] = []
        to_evaluate = self._decisions[-n:]

        for chosen, context, original_score, ts in to_evaluate:
            try:
                current_score = self._eval_fn(chosen, context)
            except Exception:
                logger.debug("Self-evaluation failed for %s", chosen)
                continue

            divergence = abs(current_score - original_score)

            if divergence >= self._threshold:
                # Direction of drift matters
                drift_direction = current_score - original_score
                signals.append(FeedbackSignal(
                    source="self_evaluator",
                    timescale=Timescale.HOURS,
                    subject=chosen,
                    dimension="consistency",
                    value=-divergence,  # Inconsistency is negative
                    confidence=0.6,
                    metadata={
                        "original_score": original_score,
                        "current_score": current_score,
                        "divergence": divergence,
                        "drift_direction": "improved" if drift_direction > 0 else "degraded",
                        "decision_age_hours": (datetime.now() - ts).total_seconds() / 3600.0,
                    },
                ))
                logger.info(
                    "Consistency drift: %s original=%.3f current=%.3f (divergence=%.3f)",
                    chosen, original_score, current_score, divergence,
                )

        return signals

    def clear(self) -> None:
        """Clear stored decisions."""
        self._decisions.clear()

"""
Temporal Fusion — compose feedback across timescales.

Pure math, no I/O. Takes a bag of FeedbackSignals from different timescales
and produces a fused signal per (subject, dimension) pair. The fusion weights
by timescale importance, emitter confidence, and recency decay.

The recency decay uses exponential half-life: signals older than the half-life
contribute half as much. This prevents stale feedback from dominating while
still allowing long-term trends to accumulate.

Author: Claude Code (with HoloLoom by Blake)
Date: 2026-03-19
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime

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

logger = logging.getLogger(__name__)


class TemporalFusion:
    """
    Fuse feedback signals across timescales into aggregated per-(subject, dimension) signals.

    Each input signal is weighted by:
        effective_weight = timescale_weight * confidence * recency_decay

    Signals are grouped by (subject, dimension), then a weighted average produces
    one fused signal per group. The fused signal inherits the dominant timescale
    (the timescale with highest total weight contribution in the group).

    Usage:
        fusion = TemporalFusion(recency_halflife_hours=7.0)
        fused = fusion.fuse(raw_signals)
        # fused: one FeedbackSignal per (subject, dimension) pair
    """

    def __init__(self, recency_halflife_hours: float = 7.0):
        """
        Args:
            recency_halflife_hours: Half-life for recency decay in hours.
                After this many hours, a signal contributes half its weight.
                Default 7h biases toward recent session feedback.
        """
        self.halflife = recency_halflife_hours

    def fuse(self, signals: list[FeedbackSignal]) -> list[FeedbackSignal]:
        """
        Group by (subject, dimension), produce weighted aggregate.

        Returns one FeedbackSignal per group with:
            - value = weighted average of input values
            - confidence = max confidence in the group (conservative floor)
            - timescale = dominant timescale by total weight
            - causation_chain = union of all input chains
            - metadata.fused_count = number of input signals
        """
        if not signals:
            return []

        now = datetime.now()

        # Group signals by (subject, dimension)
        groups: dict[tuple[str, str], list[FeedbackSignal]] = defaultdict(list)
        for sig in signals:
            groups[(sig.subject, sig.dimension)].append(sig)

        fused: list[FeedbackSignal] = []

        for (subject, dimension), group_signals in groups.items():
            # Compute weights
            weights: list[float] = []
            values: list[float] = []
            all_chains: list[str] = []
            timescale_weights: dict[Timescale, float] = defaultdict(float)

            for sig in group_signals:
                recency = self._recency_decay(sig.timestamp, now)
                w = sig.effective_weight * recency
                weights.append(w)
                values.append(sig.value)
                all_chains.extend(sig.causation_chain)
                timescale_weights[sig.timescale] += w

            total_weight = sum(weights)
            if total_weight < 1e-12:
                # All signals fully decayed — skip this group
                continue

            # Weighted average value
            fused_value = sum(v * w for v, w in zip(values, weights)) / total_weight

            # Clamp to [-1, 1]
            fused_value = max(-1.0, min(1.0, fused_value))

            # Fused confidence: weight-proportional average of input confidences
            fused_confidence = sum(
                sig.confidence * w for sig, w in zip(group_signals, weights)
            ) / total_weight
            fused_confidence = max(0.0, min(1.0, fused_confidence))

            # Dominant timescale
            dominant = self._dominant_timescale_from_weights(timescale_weights)

            # Deduplicate causation chain, preserve order
            seen: set[str] = set()
            unique_chains: list[str] = []
            for c in all_chains:
                if c not in seen:
                    seen.add(c)
                    unique_chains.append(c)

            fused_signal = FeedbackSignal(
                source="temporal_fusion",
                timescale=dominant,
                subject=subject,
                dimension=dimension,
                value=fused_value,
                confidence=fused_confidence,
                causation_chain=unique_chains,
                metadata={
                    "fused_count": len(group_signals),
                    "total_weight": total_weight,
                    "timescale_contributions": {
                        ts.value: round(w, 4) for ts, w in timescale_weights.items()
                    },
                },
            )
            fused.append(fused_signal)

        logger.debug(
            "Temporal fusion: %d signals -> %d fused groups", len(signals), len(fused)
        )
        return fused

    def fuse_for_subject(
        self, signals: list[FeedbackSignal], subject: str
    ) -> list[FeedbackSignal]:
        """Convenience: fuse only signals matching a specific subject."""
        filtered = [s for s in signals if s.subject == subject]
        return self.fuse(filtered)

    def _recency_decay(self, timestamp: datetime, now: datetime | None = None) -> float:
        """
        Exponential decay based on signal age.

        Returns 1.0 for now, 0.5 at halflife hours, approaches 0 for old signals.
        """
        if now is None:
            now = datetime.now()
        age_hours = (now - timestamp).total_seconds() / 3600.0
        if age_hours <= 0:
            return 1.0
        # ln(2) / halflife * age gives the decay exponent
        return math.exp(-0.693 * age_hours / max(self.halflife, 0.01))

    def _dominant_timescale(self, signals: list[FeedbackSignal]) -> Timescale:
        """Pick the timescale with the highest weight from a list of signals."""
        if not signals:
            return Timescale.SECONDS
        return max(signals, key=lambda s: TIMESCALE_WEIGHT.get(s.timescale, 0.0)).timescale

    def _dominant_timescale_from_weights(
        self, timescale_weights: dict[Timescale, float]
    ) -> Timescale:
        """Pick the timescale that contributed most total weight."""
        if not timescale_weights:
            return Timescale.SECONDS
        return max(timescale_weights, key=timescale_weights.get)

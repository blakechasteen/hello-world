"""
Feedback Adapters — thin wrappers making existing mechanisms speak FeedbackProtocol.

Each adapter wraps an existing HoloLoom mechanism (bandit, federation tracker,
reflection buffer, loom learner) and exposes it as a FeedbackSource, FeedbackConsumer,
or both. The adapters are intentionally thin — they translate between the existing
mechanism's native interface and the FeedbackSignal protocol.

All imports of wrapped mechanisms are optional (try/except). If a mechanism isn't
installed, its adapter simply won't collect or consume anything.

Author: Claude Code (with HoloLoom by Blake)
Date: 2026-03-19
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

try:
    from hololoom.core.protocols.feedback import (
        FeedbackConsumer,
        FeedbackSignal,
        FeedbackSource,
        Timescale,
    )
except ImportError:
    from hololoom.protocols.feedback import (
        FeedbackConsumer,
        FeedbackSignal,
        FeedbackSource,
        Timescale,
    )

logger = logging.getLogger(__name__)


# ============================================================================
# Bandit Adapters
# ============================================================================

class BanditSourceAdapter:
    """
    Generic adapter for any Thompson Sampling bandit.

    Exports current arm means as FeedbackSignals at the specified timescale.
    Works with any object that has an `arms` dict mapping (context, option)
    to objects with `successes` and `failures` attributes (the BanditArm pattern).

    If the bandit uses a different structure, subclass and override collect().
    """

    def __init__(
        self,
        bandit: Any,
        bandit_name: str,
        timescale: Timescale = Timescale.SECONDS,
        arms_attr: str = "arms",
    ):
        """
        Args:
            bandit: The bandit object. Must have an attribute containing arm state.
            bandit_name: Human-readable name (used as source_id and in metadata).
            timescale: Timescale for emitted signals.
            arms_attr: Attribute name on the bandit holding the arms dict.
        """
        self._bandit = bandit
        self._bandit_name = bandit_name
        self._timescale = timescale
        self._arms_attr = arms_attr
        self._last_means: dict[str, float] = {}

    @property
    def source_id(self) -> str:
        return f"bandit:{self._bandit_name}"

    @property
    def timescale(self) -> Timescale:
        return self._timescale

    def collect(self) -> list[FeedbackSignal]:
        """Export arm means as FeedbackSignals."""
        signals: list[FeedbackSignal] = []
        arms = getattr(self._bandit, self._arms_attr, None)
        if arms is None:
            return signals

        for key, arm in arms.items():
            successes = getattr(arm, "successes", getattr(arm, "alpha", 1.0))
            failures = getattr(arm, "failures", getattr(arm, "beta", 1.0))
            total = successes + failures
            if total < 2.0:
                continue  # Skip uninformed arms (still at prior)

            mean = successes / total
            # Normalize to [-1, 1]: 0.5 mean -> 0.0 signal
            value = (mean - 0.5) * 2.0

            # Determine subject from key (may be tuple or string)
            if isinstance(key, tuple):
                subject = ":".join(str(k) for k in key)
            else:
                subject = str(key)

            # Only emit if mean changed meaningfully since last collect
            prev = self._last_means.get(subject, 0.5)
            if abs(mean - prev) < 0.005:
                continue
            self._last_means[subject] = mean

            confidence = min(1.0, total / 50.0)  # More samples = more confident

            signals.append(FeedbackSignal(
                source=self.source_id,
                timescale=self._timescale,
                subject=subject,
                dimension="bandit_mean",
                value=value,
                confidence=confidence,
                metadata={
                    "bandit": self._bandit_name,
                    "successes": successes,
                    "failures": failures,
                    "total": total,
                },
            ))

        return signals


class BanditConsumerAdapter:
    """
    Generic consumer that updates a bandit from credit-assignment feedback.

    Listens for signals with dimension="outcome_credit" and updates the
    corresponding arm's successes/failures based on the credit value.
    """

    def __init__(
        self,
        bandit: Any,
        bandit_name: str,
        arms_attr: str = "arms",
        update_scale: float = 1.0,
    ):
        """
        Args:
            bandit: The bandit object to update.
            bandit_name: Used as consumer_id.
            arms_attr: Attribute name on the bandit holding the arms dict.
            update_scale: Scale factor for credit -> bandit update magnitude.
        """
        self._bandit = bandit
        self._bandit_name = bandit_name
        self._arms_attr = arms_attr
        self._update_scale = update_scale

    @property
    def consumer_id(self) -> str:
        return f"bandit_consumer:{self._bandit_name}"

    def accepts(self, signal: FeedbackSignal) -> bool:
        """Accept outcome_credit signals only."""
        return signal.dimension == "outcome_credit"

    def apply_feedback(self, signals: list[FeedbackSignal]) -> None:
        """Update bandit arms from credit signals."""
        arms = getattr(self._bandit, self._arms_attr, None)
        if arms is None:
            return

        for sig in signals:
            # Find matching arm by subject
            matching_key = None
            for key in arms:
                if isinstance(key, tuple):
                    key_str = ":".join(str(k) for k in key)
                else:
                    key_str = str(key)
                if key_str == sig.subject or str(key) == sig.subject:
                    matching_key = key
                    break

            if matching_key is None:
                continue

            arm = arms[matching_key]
            credit = sig.value * self._update_scale

            # Update arm
            if credit > 0:
                if hasattr(arm, "successes"):
                    arm.successes += credit
                elif hasattr(arm, "alpha"):
                    arm.alpha += credit
            else:
                if hasattr(arm, "failures"):
                    arm.failures += abs(credit)
                elif hasattr(arm, "beta"):
                    arm.beta += abs(credit)

            logger.debug(
                "Bandit %s arm %s updated: credit=%.3f",
                self._bandit_name, sig.subject, credit,
            )


# ============================================================================
# Federation Tracker Adapter
# ============================================================================

class FederationTrackerAdapter:
    """
    Source: emits promotion rates from federation note tracking.
    Also triggers credit assignment when notes are promoted.

    Wraps any object that tracks note proposals and their promotion status.
    Expected interface on tracker:
        - get_promotion_stats() -> dict[str, dict] mapping agent_id to stats
        - get_recent_promotions(since: datetime) -> list[dict]
    """

    def __init__(
        self,
        tracker: Any,
        causation_tracker: Any | None = None,
    ):
        self._tracker = tracker
        self._causation_tracker = causation_tracker
        self._last_collect: datetime = datetime.now()

    @property
    def source_id(self) -> str:
        return "federation_tracker"

    @property
    def timescale(self) -> Timescale:
        return Timescale.DAYS

    def collect(self) -> list[FeedbackSignal]:
        """Emit promotion rate signals at DAYS timescale."""
        signals: list[FeedbackSignal] = []

        # Collect promotion stats
        get_stats = getattr(self._tracker, "get_promotion_stats", None)
        if get_stats is None:
            return signals

        try:
            stats = get_stats()
        except Exception:
            logger.exception("Failed to get promotion stats")
            return signals

        for agent_id, agent_stats in stats.items():
            proposed = agent_stats.get("proposed", 0)
            promoted = agent_stats.get("promoted", 0)

            if proposed < 1:
                continue

            promotion_rate = promoted / proposed
            # Map promotion rate to [-1, 1]: 0% -> -0.5, 50% -> 0.5, 100% -> 1.0
            value = (promotion_rate * 1.5) - 0.5
            value = max(-1.0, min(1.0, value))

            signals.append(FeedbackSignal(
                source=self.source_id,
                timescale=Timescale.DAYS,
                subject=agent_id,
                dimension="promotion_rate",
                value=value,
                confidence=min(1.0, proposed / 20.0),
                metadata={
                    "proposed": proposed,
                    "promoted": promoted,
                    "rate": promotion_rate,
                },
            ))

        # Collect recent promotions for credit assignment
        get_recent = getattr(self._tracker, "get_recent_promotions", None)
        if get_recent is not None and self._causation_tracker is not None:
            try:
                recent = get_recent(since=self._last_collect)
                for promo in recent:
                    note_path = promo.get("note_path", "")
                    agent_id = promo.get("agent_id", "")
                    signals.append(FeedbackSignal(
                        source=self.source_id,
                        timescale=Timescale.DAYS,
                        subject=note_path,
                        dimension="promotion",
                        value=0.9,
                        confidence=0.95,
                        causation_chain=promo.get("causation_chain", []),
                        metadata={"agent_id": agent_id},
                    ))
            except Exception:
                logger.exception("Failed to get recent promotions")

        self._last_collect = datetime.now()
        return signals


# ============================================================================
# Reflection Buffer Adapter
# ============================================================================

class ReflectionBufferAdapter:
    """
    Source: emits analysis results from the ReflectionBuffer at MINUTES timescale.
    Consumer: receives tool performance feedback to augment buffer analysis.

    Wraps hololoom.core.reflection.buffer.ReflectionBuffer.
    """

    def __init__(self, buffer: Any):
        """
        Args:
            buffer: A ReflectionBuffer instance.
        """
        self._buffer = buffer
        self._last_metrics_hash: int = 0

    @property
    def source_id(self) -> str:
        return "reflection_buffer"

    @property
    def timescale(self) -> Timescale:
        return Timescale.MINUTES

    @property
    def consumer_id(self) -> str:
        return "reflection_buffer"

    def collect(self) -> list[FeedbackSignal]:
        """Emit tool success rates and pattern performance as FeedbackSignals."""
        signals: list[FeedbackSignal] = []

        metrics = getattr(self._buffer, "metrics", None)
        if metrics is None:
            return signals

        # Check if metrics changed since last collect
        metrics_hash = hash(
            (metrics.total_cycles, metrics.successful_cycles)
        )
        if metrics_hash == self._last_metrics_hash:
            return signals
        self._last_metrics_hash = metrics_hash

        # Tool success rates -> FeedbackSignals
        for tool, success_rate in getattr(metrics, "tool_success_rates", {}).items():
            usage = getattr(metrics, "tool_usage_counts", {}).get(tool, 0)
            if usage < 3:
                continue

            # Map success rate to [-1, 1]
            value = (success_rate - 0.5) * 2.0
            confidence = min(1.0, usage / 30.0)

            signals.append(FeedbackSignal(
                source=self.source_id,
                timescale=Timescale.MINUTES,
                subject=tool,
                dimension="success_rate",
                value=value,
                confidence=confidence,
                metadata={
                    "usage_count": usage,
                    "raw_success_rate": success_rate,
                    "avg_confidence": getattr(metrics, "tool_avg_confidence", {}).get(tool, 0.0),
                },
            ))

        # Pattern success rates
        for pattern, success_rate in getattr(metrics, "pattern_success_rates", {}).items():
            usage = getattr(metrics, "pattern_usage_counts", {}).get(pattern, 0)
            if usage < 5:
                continue

            value = (success_rate - 0.5) * 2.0
            confidence = min(1.0, usage / 50.0)

            signals.append(FeedbackSignal(
                source=self.source_id,
                timescale=Timescale.MINUTES,
                subject=f"pattern:{pattern}",
                dimension="success_rate",
                value=value,
                confidence=confidence,
                metadata={
                    "usage_count": usage,
                    "raw_success_rate": success_rate,
                    "avg_duration_ms": getattr(metrics, "pattern_avg_duration", {}).get(pattern, 0.0),
                },
            ))

        return signals

    def accepts(self, signal: FeedbackSignal) -> bool:
        """Accept tool-related performance feedback."""
        return signal.dimension in ("quality", "speed", "outcome_credit", "success_rate")

    def apply_feedback(self, signals: list[FeedbackSignal]) -> None:
        """
        Receive external feedback to augment buffer's own analysis.

        Currently logs for observability. Future: could adjust success_threshold
        or reward_config dynamically.
        """
        for sig in signals:
            logger.debug(
                "ReflectionBuffer received feedback: subject=%s dim=%s value=%.3f",
                sig.subject, sig.dimension, sig.value,
            )


# ============================================================================
# Loom Learner Adapter
# ============================================================================

class LoomLearnerAdapter:
    """
    Source + Consumer for the MathLoom's Thompson Sampling learner.

    Source: emits operation success rates from the learner's bandit arms.
    Consumer: updates learner arms from credit-assignment signals.

    Wraps any object with:
        - arms: dict mapping operation names to BanditArm-like objects
        - (optional) update(operation, reward): direct update method
    """

    def __init__(self, learner: Any, learner_name: str = "loom_learner"):
        self._learner = learner
        self._learner_name = learner_name
        self._source_adapter = BanditSourceAdapter(
            bandit=learner,
            bandit_name=learner_name,
            timescale=Timescale.SECONDS,
        )
        self._consumer_adapter = BanditConsumerAdapter(
            bandit=learner,
            bandit_name=learner_name,
        )

    @property
    def source_id(self) -> str:
        return f"loom_learner:{self._learner_name}"

    @property
    def timescale(self) -> Timescale:
        return Timescale.SECONDS

    @property
    def consumer_id(self) -> str:
        return f"loom_learner:{self._learner_name}"

    def collect(self) -> list[FeedbackSignal]:
        """Delegate to BanditSourceAdapter."""
        return self._source_adapter.collect()

    def accepts(self, signal: FeedbackSignal) -> bool:
        """Accept outcome_credit and quality signals for math operations."""
        return signal.dimension in ("outcome_credit", "quality")

    def apply_feedback(self, signals: list[FeedbackSignal]) -> None:
        """
        Apply feedback to the learner.

        For outcome_credit signals: delegate to BanditConsumerAdapter.
        For quality signals: use direct update if available.
        """
        credit_signals = [s for s in signals if s.dimension == "outcome_credit"]
        quality_signals = [s for s in signals if s.dimension == "quality"]

        if credit_signals:
            self._consumer_adapter.apply_feedback(credit_signals)

        # Direct update path for quality signals
        update_fn = getattr(self._learner, "update", None)
        if update_fn is not None:
            for sig in quality_signals:
                try:
                    # Map [-1, 1] signal value to reward
                    reward = (sig.value + 1.0) / 2.0  # back to [0, 1]
                    update_fn(sig.subject, reward)
                except Exception:
                    logger.debug(
                        "Could not directly update learner for %s", sig.subject
                    )


# ============================================================================
# Feedback Store Adapter
# ============================================================================

class FeedbackStoreAdapter:
    """
    Source: emits learning signals from the FeedbackStore (SQLite-backed user feedback).

    Wraps hololoom.core.reflection.feedback_store.FeedbackStore.
    Requires the store to be initialized (async). This adapter's collect()
    returns cached signals that were populated by an async refresh call.
    """

    def __init__(self, store: Any):
        self._store = store
        self._cached_signals: list[FeedbackSignal] = []

    @property
    def source_id(self) -> str:
        return "feedback_store"

    @property
    def timescale(self) -> Timescale:
        return Timescale.HOURS

    def collect(self) -> list[FeedbackSignal]:
        """Return cached signals from last async refresh."""
        signals = self._cached_signals
        self._cached_signals = []
        return signals

    async def refresh(self) -> None:
        """
        Async refresh: query the FeedbackStore for recent learning signals
        and cache them for the next collect() call.
        """
        get_signals = getattr(self._store, "get_learning_signals", None)
        if get_signals is None:
            return

        try:
            tool_signals = await get_signals(min_samples=3)
        except Exception:
            logger.exception("Failed to refresh from FeedbackStore")
            return

        signals: list[FeedbackSignal] = []
        for tool_name, ls in tool_signals.items():
            # Map expected_reward [0, 1] to [-1, 1]
            value = (ls.expected_reward - 0.5) * 2.0

            signals.append(FeedbackSignal(
                source=self.source_id,
                timescale=Timescale.HOURS,
                subject=tool_name,
                dimension="user_satisfaction",
                value=value,
                confidence=min(1.0, ls.total_samples / 50.0),
                metadata={
                    "alpha": ls.alpha,
                    "beta": ls.beta,
                    "avg_rating": ls.avg_rating,
                    "success_rate": ls.success_rate,
                    "total_samples": ls.total_samples,
                },
            ))

        self._cached_signals = signals

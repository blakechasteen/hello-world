"""
Conscience Reflection Bridge
=============================
Connects Conscience learning to ReflectionBuffer signals.

Enables the Conscience to learn from weaving outcomes:
- Correlates conscience decisions with execution results
- Converts ReflectionBuffer signals to conscience feedback
- Supports batch learning in background task
- Tracks effectiveness of conscience judgments

Philosophy: The conscience improves through observation - witnessing
outcomes helps calibrate future judgments.

Author: HoloLoom Team
Date: 2025-12-03
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Awaitable
from collections import deque

from hololoom.protocols.conscience import (
    ConscienceProtocol,
    ConscienceDecision,
    RiskLevel,
    StepType,
)

logger = logging.getLogger("hololoom.agentic.conscience_integration")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DecisionOutcome:
    """
    Tracks a conscience decision paired with its outcome.

    Used to correlate decisions with results for learning.
    """
    decision: ConscienceDecision
    witness_id: Optional[str] = None
    outcome_success: Optional[bool] = None
    outcome_confidence: Optional[float] = None
    outcome_reward: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    feedback_sent: bool = False

    def to_feedback(self) -> Dict[str, Any]:
        """
        Convert to Conscience.learn() feedback format.

        Returns:
            Feedback dict for conscience learning
        """
        # Determine if judgment was "correct"
        # - If we blocked (high risk) and outcome was bad → correct
        # - If we allowed (low risk) and outcome was good → correct
        # - Otherwise → incorrect (judgment miscalibrated)

        if self.outcome_success is None:
            # No outcome yet, can't determine correctness
            return {}

        high_risk = self.decision.risk_level >= RiskLevel.HIGH

        if high_risk:
            # We blocked/escalated - correct if outcome would've been bad
            # Since we blocked, assume it would've been bad (conservative)
            correct = True
        else:
            # We allowed - correct if outcome was good
            correct = self.outcome_success

        return {
            "record_id": self.witness_id,
            "correct": correct,
            "confidence": self.outcome_confidence or 0.8,
            "outcome_success": self.outcome_success,
            "outcome_reward": self.outcome_reward,
            "decision_risk_level": self.decision.risk_level.name,
            "decision_voice": self.decision.voice,
            "step_type": self.decision.step_type.name,
        }


@dataclass
class BridgeStatistics:
    """Statistics about bridge activity."""
    decisions_tracked: int = 0
    outcomes_received: int = 0
    feedback_sent: int = 0
    batch_learning_runs: int = 0
    correct_judgments: int = 0
    incorrect_judgments: int = 0
    last_learning_time: Optional[datetime] = None

    @property
    def accuracy(self) -> float:
        """Judgment accuracy (correct / total with feedback)."""
        total = self.correct_judgments + self.incorrect_judgments
        if total == 0:
            return 0.0
        return self.correct_judgments / total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "decisions_tracked": self.decisions_tracked,
            "outcomes_received": self.outcomes_received,
            "feedback_sent": self.feedback_sent,
            "batch_learning_runs": self.batch_learning_runs,
            "correct_judgments": self.correct_judgments,
            "incorrect_judgments": self.incorrect_judgments,
            "accuracy": self.accuracy,
            "last_learning_time": (
                self.last_learning_time.isoformat()
                if self.last_learning_time else None
            ),
        }


# =============================================================================
# Conscience Reflection Bridge
# =============================================================================

class ConscienceReflectionBridge:
    """
    Bridge connecting Conscience to ReflectionBuffer for learning.

    The bridge enables conscience improvement through:
    1. Tracking decisions and their outcomes
    2. Converting outcomes to learning feedback
    3. Batch learning from accumulated signals
    4. Background learning task for continuous improvement

    Usage:
        from hololoom.agentic.conscience_integration import ConscienceReflectionBridge
        from hololoom.conscience import Conscience
        from hololoom.reflection.buffer import ReflectionBuffer

        conscience = Conscience()
        buffer = ReflectionBuffer()

        bridge = ConscienceReflectionBridge(
            conscience=conscience,
            buffer=buffer,
            enable_background_learning=True
        )

        # Track a decision
        decision = await conscience_adapter.gate_reasoning(query, mode, context)
        bridge.track_decision(decision)

        # After execution, record outcome
        bridge.record_outcome(
            witness_id=decision.witness_id,
            success=True,
            confidence=0.85
        )

        # Learning happens automatically in background
        # Or trigger manually:
        await bridge.learn_batch()
    """

    def __init__(
        self,
        conscience: Optional[ConscienceProtocol] = None,
        buffer: Optional[Any] = None,  # ReflectionBuffer
        capacity: int = 500,
        learning_interval: float = 60.0,  # seconds
        min_batch_size: int = 10,
        enable_background_learning: bool = False,
    ):
        """
        Initialize the bridge.

        Args:
            conscience: Conscience instance for learning
            buffer: ReflectionBuffer instance (optional, for signal subscription)
            capacity: Maximum decisions to track
            learning_interval: Seconds between batch learning
            min_batch_size: Minimum outcomes before triggering learning
            enable_background_learning: Start background learning task
        """
        self._conscience = conscience
        self._buffer = buffer
        self._capacity = capacity
        self._learning_interval = learning_interval
        self._min_batch_size = min_batch_size

        # Decision tracking (keyed by witness_id)
        self._decisions: Dict[str, DecisionOutcome] = {}
        self._decision_order: deque = deque(maxlen=capacity)

        # Statistics
        self._stats = BridgeStatistics()

        # Background learning
        self._background_task: Optional[asyncio.Task] = None
        self._running = False

        if enable_background_learning:
            self._start_background_learning()

        logger.info(
            f"ConscienceReflectionBridge initialized "
            f"(capacity={capacity}, learning_interval={learning_interval}s)"
        )

    # =========================================================================
    # Decision Tracking
    # =========================================================================

    def track_decision(self, decision: ConscienceDecision) -> None:
        """
        Track a conscience decision for later outcome correlation.

        Args:
            decision: The conscience decision to track
        """
        if not decision.witness_id:
            logger.debug("Decision has no witness_id, generating one")
            decision.witness_id = f"track_{int(time.time() * 1000)}"

        witness_id = decision.witness_id

        # Create outcome tracker
        outcome = DecisionOutcome(decision=decision, witness_id=witness_id)

        # Store decision
        self._decisions[witness_id] = outcome
        self._decision_order.append(witness_id)

        # Evict old decisions if at capacity
        while len(self._decisions) > self._capacity:
            old_id = self._decision_order.popleft()
            if old_id in self._decisions:
                del self._decisions[old_id]

        self._stats.decisions_tracked += 1

        logger.debug(f"Tracked decision: {witness_id}")

    def record_outcome(
        self,
        witness_id: str,
        success: bool,
        confidence: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> bool:
        """
        Record the outcome for a tracked decision.

        Args:
            witness_id: The witness ID of the decision
            success: Whether execution was successful
            confidence: Confidence in the success assessment
            reward: Explicit reward signal (0-1)

        Returns:
            True if outcome was recorded, False if decision not found
        """
        if witness_id not in self._decisions:
            logger.warning(f"Decision not found for outcome: {witness_id}")
            return False

        outcome = self._decisions[witness_id]
        outcome.outcome_success = success
        outcome.outcome_confidence = confidence
        outcome.outcome_reward = reward

        self._stats.outcomes_received += 1

        logger.debug(
            f"Recorded outcome for {witness_id}: "
            f"success={success}, confidence={confidence}"
        )

        return True

    def record_outcome_from_spacetime(
        self,
        witness_id: str,
        spacetime: Any,  # Spacetime
        feedback: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record outcome from a Spacetime artifact.

        Convenience method that extracts success/confidence from Spacetime.

        Args:
            witness_id: The witness ID of the decision
            spacetime: Spacetime artifact from weaving
            feedback: Optional user feedback

        Returns:
            True if outcome was recorded
        """
        # Extract success and confidence from spacetime
        confidence = getattr(spacetime, 'confidence', 0.5)

        # Success if confidence >= threshold (typically 0.6)
        success = confidence >= 0.6

        # Override with explicit feedback if provided
        if feedback:
            if 'helpful' in feedback:
                success = feedback['helpful']
            if 'success' in feedback:
                success = feedback['success']

        # Compute reward
        reward = confidence
        if feedback and 'reward' in feedback:
            reward = feedback['reward']

        return self.record_outcome(
            witness_id=witness_id,
            success=success,
            confidence=confidence,
            reward=reward,
        )

    # =========================================================================
    # Learning
    # =========================================================================

    async def learn_batch(self, force: bool = False) -> int:
        """
        Send batch learning feedback to conscience.

        Processes all tracked decisions with outcomes and sends
        learning feedback to the conscience.

        Args:
            force: Force learning even if below min_batch_size

        Returns:
            Number of feedback items sent
        """
        if not self._conscience:
            logger.debug("No conscience configured, skipping learning")
            return 0

        # Collect decisions with outcomes that haven't been sent
        pending = [
            outcome for outcome in self._decisions.values()
            if outcome.outcome_success is not None and not outcome.feedback_sent
        ]

        if not force and len(pending) < self._min_batch_size:
            logger.debug(
                f"Below min_batch_size ({len(pending)} < {self._min_batch_size})"
            )
            return 0

        if not pending:
            return 0

        logger.info(f"Sending {len(pending)} feedback items to conscience")

        feedback_sent = 0

        for outcome in pending:
            feedback = outcome.to_feedback()
            if not feedback:
                continue

            try:
                await self._conscience.learn(feedback)
                outcome.feedback_sent = True
                feedback_sent += 1

                # Track correctness
                if feedback.get("correct"):
                    self._stats.correct_judgments += 1
                else:
                    self._stats.incorrect_judgments += 1

            except Exception as e:
                logger.warning(f"Failed to send feedback: {e}")

        self._stats.feedback_sent += feedback_sent
        self._stats.batch_learning_runs += 1
        self._stats.last_learning_time = datetime.now()

        logger.info(
            f"Batch learning complete: {feedback_sent} feedback items, "
            f"accuracy={self._stats.accuracy:.1%}"
        )

        return feedback_sent

    async def learn_from_buffer_signals(
        self,
        signals: Optional[List[Any]] = None,  # List[LearningSignal]
    ) -> int:
        """
        Learn from ReflectionBuffer learning signals.

        Converts buffer signals to conscience feedback format.

        Args:
            signals: Learning signals from buffer (if None, fetches from buffer)

        Returns:
            Number of learning updates applied
        """
        if not self._conscience:
            return 0

        # Get signals from buffer if not provided
        if signals is None and self._buffer is not None:
            try:
                signals = await self._buffer.analyze_and_learn(force=True)
            except Exception as e:
                logger.warning(f"Failed to get buffer signals: {e}")
                return 0

        if not signals:
            return 0

        updates = 0

        for signal in signals:
            # Convert signal to conscience feedback
            feedback = self._signal_to_feedback(signal)
            if not feedback:
                continue

            try:
                await self._conscience.learn(feedback)
                updates += 1
            except Exception as e:
                logger.warning(f"Failed to apply signal: {e}")

        logger.info(f"Applied {updates} learning signals from buffer")

        return updates

    def _signal_to_feedback(self, signal: Any) -> Optional[Dict[str, Any]]:
        """
        Convert a LearningSignal to conscience feedback format.

        Args:
            signal: LearningSignal from ReflectionBuffer

        Returns:
            Feedback dict for conscience, or None if not applicable
        """
        signal_type = getattr(signal, 'signal_type', None)

        if signal_type == "bandit_update":
            # Tool performance signal
            reward = getattr(signal, 'reward', 0.5)
            tool = getattr(signal, 'tool', 'unknown')

            return {
                "signal_type": "tool_performance",
                "tool": tool,
                "correct": reward >= 0.6,
                "confidence": reward,
                "evidence": getattr(signal, 'evidence', {}),
            }

        elif signal_type == "threshold_adjustment":
            # Threshold adjustment signal
            return {
                "signal_type": "threshold_adjustment",
                "tool": getattr(signal, 'tool', None),
                "recommendation": getattr(signal, 'recommendation', ''),
                "evidence": getattr(signal, 'evidence', {}),
            }

        elif signal_type == "pattern_preference":
            # Pattern preference signal
            return {
                "signal_type": "pattern_preference",
                "pattern": getattr(signal, 'pattern', None),
                "evidence": getattr(signal, 'evidence', {}),
            }

        return None

    # =========================================================================
    # Background Learning
    # =========================================================================

    def _start_background_learning(self) -> None:
        """Start background learning task."""
        if self._background_task is not None:
            return

        self._running = True
        self._background_task = asyncio.create_task(
            self._background_learning_loop()
        )

        logger.info("Started background learning task")

    async def _background_learning_loop(self) -> None:
        """Background learning loop."""
        while self._running:
            try:
                await asyncio.sleep(self._learning_interval)

                if not self._running:
                    break

                # Run batch learning
                await self.learn_batch()

                # Also learn from buffer signals if connected
                if self._buffer:
                    await self.learn_from_buffer_signals()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background learning error: {e}")

    async def stop_background_learning(self) -> None:
        """Stop background learning task."""
        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

        logger.info("Stopped background learning task")

    # =========================================================================
    # Statistics & Introspection
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return self._stats.to_dict()

    def get_pending_count(self) -> int:
        """Get count of decisions awaiting outcomes."""
        return sum(
            1 for o in self._decisions.values()
            if o.outcome_success is None
        )

    def get_unfed_count(self) -> int:
        """Get count of outcomes not yet sent for learning."""
        return sum(
            1 for o in self._decisions.values()
            if o.outcome_success is not None and not o.feedback_sent
        )

    def clear(self) -> None:
        """Clear all tracked decisions."""
        self._decisions.clear()
        self._decision_order.clear()
        logger.info("Cleared all tracked decisions")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close the bridge and stop background tasks."""
        await self.stop_background_learning()

        # Final learning flush
        if self.get_unfed_count() > 0:
            await self.learn_batch(force=True)

        logger.info("ConscienceReflectionBridge closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"ConscienceReflectionBridge("
            f"tracked={len(self._decisions)}, "
            f"accuracy={self._stats.accuracy:.1%})"
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_conscience_bridge(
    conscience: Optional[ConscienceProtocol] = None,
    buffer: Optional[Any] = None,
    enable_background_learning: bool = False,
) -> ConscienceReflectionBridge:
    """
    Create a ConscienceReflectionBridge with sensible defaults.

    Args:
        conscience: Conscience instance
        buffer: ReflectionBuffer instance
        enable_background_learning: Start background learning

    Returns:
        Configured bridge instance
    """
    return ConscienceReflectionBridge(
        conscience=conscience,
        buffer=buffer,
        capacity=500,
        learning_interval=60.0,
        min_batch_size=10,
        enable_background_learning=enable_background_learning,
    )

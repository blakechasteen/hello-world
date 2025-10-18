"""
Chrono Trigger - Temporal Control System
=========================================
The temporal orchestrator that manages all time-dependent aspects.

Responsibilities:
- Temporal Control: When to activate threads
- Execution: How long operations run
- Rhythm: Cadence of background processes
- Halt: Stop conditions
- Decay: Thread aging over time
- Evolve: Learning and adaptation

Philosophy:
Time is the fourth dimension of the weaving process. The Chrono Trigger
ensures threads activate at the right moment, computations complete on time,
and the Yarn Graph evolves through temporal decay and learning.
"""

import asyncio
import math
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TemporalWindow:
    """
    Defines time boundaries for thread activation.

    When Chrono Trigger fires, it creates a temporal window that determines
    which threads from the Yarn Graph are eligible for activation into Warp Space.
    """
    start: datetime
    end: datetime
    max_age: timedelta  # Maximum thread age to consider
    recency_bias: float = 0.5  # Weight boost for recent threads (0-1)
    episode_filter: Optional[str] = None  # Optional episode/session filter

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within window."""
        return self.start <= timestamp <= self.end

    def recency_weight(self, timestamp: datetime) -> float:
        """
        Calculate recency-based weight for a thread.

        More recent threads get higher weights based on recency_bias.
        Returns value between 0 and 1.
        """
        if not self.contains(timestamp):
            return 0.0

        # Calculate age as fraction of window
        window_duration = (self.end - self.start).total_seconds()
        if window_duration == 0:
            return 1.0

        age = (self.end - timestamp).total_seconds()
        age_fraction = age / window_duration

        # Apply bias: higher bias = stronger preference for recent
        weight = 1.0 - (age_fraction * self.recency_bias)
        return max(0.0, min(1.0, weight))


@dataclass
class ExecutionLimits:
    """
    Execution constraints and halt conditions.

    Defines when and why to stop ongoing computations.
    """
    max_duration: float = 5.0  # Max total pipeline duration (seconds)
    stage_timeouts: Dict[str, float] = field(default_factory=lambda: {
        'features': 2.0,
        'retrieval': 2.0,
        'decision': 2.0,
        'execution': 3.0
    })
    halt_on_low_confidence: float = 0.3  # Stop if decision confidence below this
    max_memory_mb: int = 1000  # Max memory usage
    enable_early_stopping: bool = True


# ============================================================================
# Chrono Trigger
# ============================================================================

class ChronoTrigger:
    """
    Temporal control system for HoloLoom weaving.

    The Chrono Trigger is the temporal nervous system that manages:
    1. When threads activate (temporal windows)
    2. How long operations run (execution limits)
    3. Background maintenance rhythm (heartbeat)
    4. When to stop (halt conditions)
    5. Thread aging (decay)
    6. System evolution (learning over time)

    Usage:
        chrono = ChronoTrigger(config)
        window = await chrono.fire(query_time, yarn_graph, pattern_card)
        result = await chrono.monitor(operation, timeout=2.0)
        metrics = chrono.record_completion()
        chrono.stop()
    """

    def __init__(self, config, enable_heartbeat: bool = False):
        """
        Initialize Chrono Trigger.

        Args:
            config: HoloLoom Config with timeout settings
            enable_heartbeat: Whether to start background maintenance
        """
        self.config = config
        self.limits = ExecutionLimits(
            max_duration=getattr(config, 'pipeline_timeout', 5.0),
            halt_on_low_confidence=0.3
        )

        # Temporal state
        self.start_time = None
        self.query_time = None
        self.current_window = None

        # Background processes
        self.heartbeat_task = None
        self.enable_heartbeat = enable_heartbeat

        # Decay parameters
        self.decay_rate = 0.01  # Weight decay per hour
        self.last_decay_time = datetime.now()

        # Evolution tracking
        self.execution_history = []

        logger.info("ChronoTrigger initialized")

    async def fire(
        self,
        query_time: datetime,
        pattern_card_mode: str = "fused"
    ) -> TemporalWindow:
        """
        Fire temporal activation.

        Creates a temporal window based on the Pattern Card mode.
        Starts execution timer and optional heartbeat.

        Args:
            query_time: Current query timestamp
            pattern_card_mode: Pattern Card mode (bare/fast/fused)

        Returns:
            TemporalWindow for thread selection
        """
        self.start_time = time.time()
        self.query_time = query_time

        logger.info(f"Chrono Trigger FIRED at {query_time} (mode={pattern_card_mode})")

        # Create temporal window based on Pattern Card
        if pattern_card_mode == "bare":
            # Bare: short window, high recency bias
            window = TemporalWindow(
                start=query_time - timedelta(hours=1),
                end=query_time,
                max_age=timedelta(hours=24),
                recency_bias=0.8
            )
        elif pattern_card_mode == "fast":
            # Fast: medium window, moderate bias
            window = TemporalWindow(
                start=query_time - timedelta(hours=24),
                end=query_time,
                max_age=timedelta(days=7),
                recency_bias=0.5
            )
        else:  # fused
            # Fused: full history, low bias
            window = TemporalWindow(
                start=datetime.min,
                end=query_time,
                max_age=timedelta(days=365),
                recency_bias=0.2
            )

        self.current_window = window

        # Start heartbeat if enabled
        if self.enable_heartbeat and not self.heartbeat_task:
            self.heartbeat_task = asyncio.create_task(self._heartbeat())

        return window

    async def monitor(
        self,
        operation: Callable,
        timeout: Optional[float] = None,
        stage: str = "operation",
        halt_callback: Optional[Callable[[Any], bool]] = None
    ) -> Any:
        """
        Monitor operation with timeout and halt conditions.

        Args:
            operation: Async callable to execute
            timeout: Max duration (uses stage_timeouts if None)
            stage: Stage name for timeout lookup
            halt_callback: Optional callback to check halt conditions

        Returns:
            Operation result or error dict
        """
        if timeout is None:
            timeout = self.limits.stage_timeouts.get(stage, 5.0)

        try:
            logger.debug(f"Monitoring {stage} (timeout={timeout}s)")

            result = await asyncio.wait_for(
                operation(),
                timeout=timeout
            )

            # Check halt conditions
            if self.limits.enable_early_stopping and halt_callback:
                should_halt = halt_callback(result)
                if should_halt:
                    logger.warning(f"Early halt triggered for {stage}")
                    return {"status": "halted", "reason": "early_stopping", "partial_result": result}

            return result

        except asyncio.TimeoutError:
            logger.error(f"Timeout in {stage} after {timeout}s")
            return {"status": "error", "error": "timeout", "duration": timeout, "stage": stage}
        except Exception as e:
            logger.error(f"Error in {stage}: {e}")
            return {"status": "error", "error": str(e), "stage": stage}

    def check_halt_on_confidence(self, decision: Dict) -> bool:
        """
        Check if decision confidence is too low to continue.

        Args:
            decision: Decision dict with 'confidence' field

        Returns:
            True if should halt
        """
        confidence = decision.get('confidence', 1.0)
        if confidence < self.limits.halt_on_low_confidence:
            logger.warning(f"Low confidence detected: {confidence:.3f} < {self.limits.halt_on_low_confidence}")
            return True
        return False

    async def _heartbeat(self):
        """
        Background heartbeat for maintenance.

        Runs periodically to:
        - Apply decay to thread weights
        - Prune stale cache entries
        - Update evolution metrics
        """
        logger.info("Heartbeat started")

        while True:
            try:
                await asyncio.sleep(60)  # Every minute

                logger.debug("Heartbeat pulse")

                # Apply decay if enough time has passed
                now = datetime.now()
                time_since_decay = (now - self.last_decay_time).total_seconds() / 3600

                if time_since_decay >= 1.0:  # Decay every hour
                    await self._apply_decay()
                    self.last_decay_time = now

            except asyncio.CancelledError:
                logger.info("Heartbeat stopped")
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _apply_decay(self):
        """
        Apply time-based decay to thread weights.

        In a full implementation, this would iterate through
        the Yarn Graph and decay edge weights over time.
        """
        logger.debug(f"Applying decay (rate={self.decay_rate}/hour)")

        # Placeholder: actual implementation would need yarn_graph reference
        # For now, just log the decay event
        pass

    def record_completion(self) -> Dict[str, Any]:
        """
        Record metrics on completion.

        Returns:
            Dict with execution metrics
        """
        if self.start_time is None:
            return {"error": "chrono not fired"}

        duration = time.time() - self.start_time
        budget_used = duration / self.limits.max_duration

        metrics = {
            'duration': duration,
            'timestamp': datetime.now(),
            'budget_used': budget_used,
            'within_budget': budget_used <= 1.0,
            'query_time': self.query_time
        }

        # Store in history for evolution
        self.execution_history.append(metrics)

        logger.info(f"Execution completed: {duration:.3f}s ({budget_used:.1%} of budget)")

        return metrics

    def stop(self):
        """
        Stop all temporal processes.

        Cancels heartbeat and cleans up resources.
        """
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            logger.info("Chrono Trigger stopped")

    def get_evolution_stats(self) -> Dict[str, Any]:
        """
        Get statistics for system evolution.

        Returns:
            Dict with evolution metrics
        """
        if not self.execution_history:
            return {"executions": 0}

        durations = [h['duration'] for h in self.execution_history]
        budgets = [h['budget_used'] for h in self.execution_history]

        return {
            'executions': len(self.execution_history),
            'avg_duration': sum(durations) / len(durations),
            'avg_budget_used': sum(budgets) / len(budgets),
            'success_rate': sum(1 for b in budgets if b <= 1.0) / len(budgets),
            'recent_duration': durations[-10:] if len(durations) >= 10 else durations
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("Chrono Trigger Demo")
        print("="*80 + "\n")

        # Create chrono trigger
        from holoLoom.config import Config
        config = Config.fused()
        chrono = ChronoTrigger(config)

        # Fire for query
        query_time = datetime.now()
        window = await chrono.fire(query_time, pattern_card_mode="fused")

        print(f"Temporal window:")
        print(f"  Start: {window.start}")
        print(f"  End: {window.end}")
        print(f"  Max age: {window.max_age}")
        print(f"  Recency bias: {window.recency_bias}\n")

        # Monitor an operation
        async def slow_operation():
            await asyncio.sleep(0.5)
            return {"result": "success", "confidence": 0.9}

        result = await chrono.monitor(
            slow_operation,
            timeout=2.0,
            stage="demo"
        )
        print(f"Operation result: {result}\n")

        # Record completion
        metrics = chrono.record_completion()
        print(f"Execution metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Stop
        chrono.stop()

        print("\n✓ Demo complete!")

    asyncio.run(demo())

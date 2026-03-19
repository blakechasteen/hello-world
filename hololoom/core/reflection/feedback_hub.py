"""
FeedbackHub — the loom's tension sensor.

Central orchestrator that collects feedback from all sources (bandits,
federation trackers, reflection buffers, bus signals), fuses across timescales,
assigns credit via causation chains, and distributes to consumers.

Non-blocking by design: submit() is <0.01ms (deque append). Actual processing
happens in the async flush loop at configurable intervals. The bus integration
translates bus Signals into FeedbackSignals transparently.

Architecture:
    Sources -> [submit/collect] -> pending deque
                                      |
                            flush loop (every N seconds)
                                      |
                            temporal fusion
                                      |
                            credit assignment
                                      |
                            distribute to consumers

Author: Claude Code (with HoloLoom by Blake)
Date: 2026-03-19
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
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

from hololoom.core.reflection.temporal_fusion import TemporalFusion
from hololoom.core.reflection.causation import CausationTracker

logger = logging.getLogger(__name__)


# ============================================================================
# Bus Signal Kind Constants (avoid hard import of bus module)
# ============================================================================

_BUS_KIND_EXECUTION_COMPLETE = "execution_complete"
_BUS_KIND_STEP_COMPLETE = "step_complete"
_BUS_KIND_STEP_FAILED = "step_failed"
_BUS_KIND_EXECUTION_FAILED = "execution_failed"
_BUS_KIND_ERROR = "error"
_BUS_KIND_CONVERGENCE = "convergence"
_BUS_KIND_REFINEMENT_COMPLETE = "refinement_complete"


class FeedbackHub:
    """
    Central feedback orchestrator.

    Lifecycle:
        hub = FeedbackHub(bus=my_bus, persist_path="./feedback.db")
        await hub.start()   # subscribe to bus, start flush loop
        ...
        hub.submit(signal)   # non-blocking, <0.01ms
        ...
        await hub.stop()     # clean shutdown

    Registration:
        hub.register_source(my_bandit_adapter)
        hub.register_consumer(my_bandit_consumer)

    Manual flush:
        hub.flush()  # force immediate processing
    """

    def __init__(
        self,
        bus: Any | None = None,
        persist_path: str | None = None,
        flush_interval: float = 30.0,
        max_pending: int = 10_000,
        recency_halflife_hours: float = 7.0,
    ):
        """
        Args:
            bus: Optional UnifiedBus instance. If provided, subscribes to
                 execution/step/error signals and translates them.
            persist_path: Optional path for CausationTracker SQLite persistence.
            flush_interval: Seconds between automatic flushes.
            max_pending: Maximum pending signals before oldest are dropped.
            recency_halflife_hours: Half-life for temporal fusion recency decay.
        """
        self._sources: dict[str, FeedbackSource] = {}
        self._consumers: dict[str, FeedbackConsumer] = {}
        self._causation = CausationTracker(persist_path=persist_path)
        self._fusion = TemporalFusion(recency_halflife_hours=recency_halflife_hours)
        self._pending: deque[FeedbackSignal] = deque(maxlen=max_pending)
        self._flush_interval = flush_interval
        self._bus = bus
        self._running = False
        self._flush_task: asyncio.Task | None = None
        self._flush_count = 0
        self._total_submitted = 0
        self._total_distributed = 0

    # ========================================================================
    # Registration
    # ========================================================================

    def register_source(self, source: FeedbackSource) -> None:
        """Register a feedback source. Its collect() is called each flush."""
        self._sources[source.source_id] = source
        logger.info("Registered feedback source: %s (timescale=%s)", source.source_id, source.timescale.value)

    def unregister_source(self, source_id: str) -> None:
        """Remove a registered source."""
        self._sources.pop(source_id, None)

    def register_consumer(self, consumer: FeedbackConsumer) -> None:
        """Register a feedback consumer. Receives matching signals each flush."""
        self._consumers[consumer.consumer_id] = consumer
        logger.info("Registered feedback consumer: %s", consumer.consumer_id)

    def unregister_consumer(self, consumer_id: str) -> None:
        """Remove a registered consumer."""
        self._consumers.pop(consumer_id, None)

    # ========================================================================
    # Submit (hot path — must be fast)
    # ========================================================================

    def submit(self, signal: FeedbackSignal) -> None:
        """
        Submit feedback. Non-blocking, <0.01ms.

        Signals with causation chains are also recorded for credit assignment.
        """
        self._pending.append(signal)
        self._total_submitted += 1

        if signal.causation_chain:
            self._causation.record(signal)

    def submit_batch(self, signals: list[FeedbackSignal]) -> None:
        """Submit multiple signals at once."""
        for sig in signals:
            self.submit(sig)

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def start(self) -> None:
        """Subscribe to bus signals, start flush loop."""
        if self._running:
            return

        self._running = True

        # Subscribe to bus signal kinds if bus available
        if self._bus is not None:
            self._subscribe_to_bus()

        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            "FeedbackHub started (flush_interval=%.1fs, sources=%d, consumers=%d)",
            self._flush_interval, len(self._sources), len(self._consumers),
        )

    async def stop(self) -> None:
        """Stop flush loop, do final flush, clean up."""
        self._running = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush
        self.flush()
        self._causation.close()
        logger.info(
            "FeedbackHub stopped (total submitted=%d, distributed=%d, flushes=%d)",
            self._total_submitted, self._total_distributed, self._flush_count,
        )

    # ========================================================================
    # Flush Loop
    # ========================================================================

    async def _flush_loop(self) -> None:
        """Periodic flush loop. Runs until stop() is called."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                self.flush()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in feedback flush loop")

    def flush(self) -> dict[str, Any]:
        """
        Collect from sources, fuse, assign credit, distribute to consumers.

        Returns a summary dict for observability.
        """
        self._flush_count += 1

        # 1. Collect from registered sources
        collected = 0
        for source_id, source in self._sources.items():
            try:
                source_signals = source.collect()
                for sig in source_signals:
                    self._pending.append(sig)
                    self._total_submitted += 1
                collected += len(source_signals)
            except Exception:
                logger.exception("Failed to collect from source %s", source_id)

        # 2. Drain pending into a working list
        signals: list[FeedbackSignal] = []
        while self._pending:
            signals.append(self._pending.popleft())

        if not signals:
            return {
                "flush": self._flush_count,
                "collected": collected,
                "pending": 0,
                "fused": 0,
                "credit": 0,
                "distributed": 0,
            }

        # 3. Temporal fusion
        fused = self._fusion.fuse(signals)

        # 4. Credit assignment for signals with causation chains
        credit_signals: list[FeedbackSignal] = []
        for sig in signals:
            if sig.causation_chain:
                credits = self._causation.assign_credit(sig)
                credit_signals.extend(credits)

        # Combine fused + credit signals for distribution
        distributable = fused + credit_signals

        # 5. Distribute to matching consumers
        distributed = 0
        for consumer_id, consumer in self._consumers.items():
            matching = [s for s in distributable if consumer.accepts(s)]
            if matching:
                try:
                    consumer.apply_feedback(matching)
                    distributed += len(matching)
                except Exception:
                    logger.exception("Failed to distribute to consumer %s", consumer_id)

        self._total_distributed += distributed

        summary = {
            "flush": self._flush_count,
            "collected": collected,
            "pending": len(signals),
            "fused": len(fused),
            "credit": len(credit_signals),
            "distributed": distributed,
        }
        logger.debug("Flush #%d: %s", self._flush_count, summary)
        return summary

    # ========================================================================
    # Bus Integration
    # ========================================================================

    def _subscribe_to_bus(self) -> None:
        """Subscribe to relevant bus signal kinds."""
        try:
            from hololoom.core.bus.signal import SignalKind
            bus = self._bus

            bus.on(SignalKind.EXECUTION_COMPLETE, self._on_execution_complete)
            bus.on(SignalKind.STEP_COMPLETE, self._on_step_complete)
            bus.on(SignalKind.STEP_FAILED, self._on_step_failed)
            bus.on(SignalKind.EXECUTION_FAILED, self._on_execution_failed)
            bus.on(SignalKind.ERROR, self._on_error)
            bus.on(SignalKind.CONVERGENCE, self._on_convergence)
            bus.on(SignalKind.REFINEMENT_COMPLETE, self._on_refinement_complete)
            logger.info("FeedbackHub subscribed to bus signal kinds")
        except (ImportError, AttributeError) as e:
            logger.warning("Could not subscribe to bus: %s", e)

    async def _on_execution_complete(self, signal: Any) -> None:
        """Translate EXECUTION_COMPLETE bus signal to FeedbackSignal."""
        payload = getattr(signal, "payload", {})
        confidence = payload.get("confidence", 0.5)
        tool = payload.get("tool_used", payload.get("tool", "unknown"))
        duration_ms = payload.get("duration_ms", 0)

        # Quality signal based on confidence
        self.submit(FeedbackSignal(
            source="bus:execution_complete",
            timescale=Timescale.SECONDS,
            subject=tool,
            dimension="quality",
            value=(confidence - 0.5) * 2,  # map [0,1] -> [-1,1]
            confidence=confidence,
            causation_chain=[getattr(signal, "correlation_id", "")] if getattr(signal, "correlation_id", None) else [],
            metadata={"duration_ms": duration_ms},
        ))

        # Speed signal based on duration
        if duration_ms > 0:
            # Sub-1s = good, >5s = bad
            speed_value = max(-1.0, min(1.0, 1.0 - (duration_ms / 2500.0)))
            self.submit(FeedbackSignal(
                source="bus:execution_complete",
                timescale=Timescale.SUB_SECOND,
                subject=tool,
                dimension="speed",
                value=speed_value,
                confidence=0.8,
                metadata={"duration_ms": duration_ms},
            ))

    async def _on_step_complete(self, signal: Any) -> None:
        """Translate STEP_COMPLETE bus signal."""
        payload = getattr(signal, "payload", {})
        step = payload.get("step", "unknown_step")
        confidence = payload.get("confidence", 0.5)

        self.submit(FeedbackSignal(
            source="bus:step_complete",
            timescale=Timescale.SUB_SECOND,
            subject=step,
            dimension="quality",
            value=(confidence - 0.5) * 2,
            confidence=confidence * 0.7,  # Lower confidence for individual steps
        ))

    async def _on_step_failed(self, signal: Any) -> None:
        """Translate STEP_FAILED bus signal."""
        payload = getattr(signal, "payload", {})
        step = payload.get("step", "unknown_step")

        self.submit(FeedbackSignal(
            source="bus:step_failed",
            timescale=Timescale.SUB_SECOND,
            subject=step,
            dimension="quality",
            value=-0.5,
            confidence=0.8,
            metadata={"error": payload.get("error", "")},
        ))

    async def _on_execution_failed(self, signal: Any) -> None:
        """Translate EXECUTION_FAILED bus signal."""
        payload = getattr(signal, "payload", {})
        tool = payload.get("tool_used", payload.get("tool", "unknown"))

        self.submit(FeedbackSignal(
            source="bus:execution_failed",
            timescale=Timescale.SECONDS,
            subject=tool,
            dimension="quality",
            value=-0.8,
            confidence=0.9,
            causation_chain=[getattr(signal, "correlation_id", "")] if getattr(signal, "correlation_id", None) else [],
            metadata={"error": payload.get("error", "")},
        ))

    async def _on_error(self, signal: Any) -> None:
        """Translate ERROR bus signal."""
        payload = getattr(signal, "payload", {})
        source_system = getattr(signal, "source", "unknown")

        self.submit(FeedbackSignal(
            source=f"bus:error:{source_system}",
            timescale=Timescale.SUB_SECOND,
            subject=source_system,
            dimension="reliability",
            value=-1.0,
            confidence=0.95,
            metadata={"error": payload.get("message", str(payload))},
        ))

    async def _on_convergence(self, signal: Any) -> None:
        """Translate CONVERGENCE bus signal."""
        payload = getattr(signal, "payload", {})
        subject = payload.get("subject", "convergence")
        converged = payload.get("converged", False)

        self.submit(FeedbackSignal(
            source="bus:convergence",
            timescale=Timescale.MINUTES,
            subject=subject,
            dimension="convergence",
            value=0.8 if converged else -0.3,
            confidence=0.7,
            metadata=payload,
        ))

    async def _on_refinement_complete(self, signal: Any) -> None:
        """Translate REFINEMENT_COMPLETE bus signal."""
        payload = getattr(signal, "payload", {})
        improvement = payload.get("improvement", 0.0)
        passes = payload.get("passes", 1)

        self.submit(FeedbackSignal(
            source="bus:refinement_complete",
            timescale=Timescale.SECONDS,
            subject="refinement",
            dimension="quality",
            value=max(-1.0, min(1.0, improvement)),
            confidence=min(0.9, 0.5 + passes * 0.1),
            metadata={"passes": passes, "improvement": improvement},
        ))

    # ========================================================================
    # Causation (public API for external decision recording)
    # ========================================================================

    def record_decision(
        self,
        signal_id: str,
        correlation_id: str,
        decision_type: str,
        chosen: str,
        alternatives: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a decision point for later credit assignment."""
        self._causation.record_decision(
            signal_id=signal_id,
            correlation_id=correlation_id,
            decision_type=decision_type,
            chosen=chosen,
            alternatives=alternatives,
            context=context,
        )

    # ========================================================================
    # Summary / Observability
    # ========================================================================

    def summary(self) -> dict[str, Any]:
        """Current state summary for dashboards and logging."""
        return {
            "running": self._running,
            "sources": list(self._sources.keys()),
            "consumers": list(self._consumers.keys()),
            "pending": len(self._pending),
            "flush_count": self._flush_count,
            "total_submitted": self._total_submitted,
            "total_distributed": self._total_distributed,
            "decision_count": self._causation.decision_count,
        }

    def __repr__(self) -> str:
        return (
            f"FeedbackHub(sources={len(self._sources)}, "
            f"consumers={len(self._consumers)}, "
            f"pending={len(self._pending)}, "
            f"flushes={self._flush_count})"
        )

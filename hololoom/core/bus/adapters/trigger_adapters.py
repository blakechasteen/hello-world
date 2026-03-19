"""
Trigger Adapters — Connect all four trigger modes to the UnifiedBus.
=====================================================================

Four trigger modes, each emitting signals to the bus:

1. HeartbeatTrigger: Periodic pulse (configurable interval, default 60s)
   - Emits HEARTBEAT signals on each tick
   - Handlers can react (decay memory, prune MCTS trees, health checks)

2. RitualTrigger: Wraps the existing RitualScheduler
   - Checks for cosmic/weekly ritual events on each heartbeat
   - Emits RITUAL signals when events fire

3. CronTrigger: Simple cron-like scheduler
   - Runs registered tasks at configured intervals
   - Emits CRON signals when tasks fire

4. ReactiveTrigger: Signal-driven chains
   - Already built into the bus (handler returns -> re-emit)
   - This module provides convenience for registering reactive rules

All triggers are optional. The bus works without any triggers.
Triggers add background intelligence.

Usage:
    from hololoom.core.bus import UnifiedBus
    from hololoom.core.bus.adapters.trigger_adapters import (
        HeartbeatTrigger, CronTrigger,
    )

    bus = UnifiedBus()
    heartbeat = HeartbeatTrigger(bus, interval=60.0)
    await heartbeat.start()

    cron = CronTrigger(bus)
    cron.every(300, "consolidate_memory", {"action": "consolidate"})
    await cron.start()
"""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..signal import Signal, SignalKind, SignalPriority, TriggerMode
from ..unified_bus import UnifiedBus

logger = logging.getLogger(__name__)


# ============================================================================
# Heartbeat Trigger
# ============================================================================

class HeartbeatTrigger:
    """
    Periodic pulse emitter.

    Mirrors ChronoTrigger._heartbeat() but emits signals to the bus
    instead of directly calling handlers.

    On each tick:
    - Emits a HEARTBEAT signal with tick count and health info
    - Handlers subscribed to SignalKind.HEARTBEAT react

    This replaces the need for direct heartbeat callbacks.
    """

    def __init__(
        self,
        bus: UnifiedBus,
        interval: float = 60.0,
        health_fn: Callable[[], dict[str, Any]] | None = None,
    ):
        """
        Args:
            bus: UnifiedBus to emit heartbeat signals to
            interval: Seconds between heartbeats (default: 60)
            health_fn: Optional function that returns health info for payload
        """
        self.bus = bus
        self.interval = interval
        self.health_fn = health_fn

        self._running = False
        self._task: asyncio.Task | None = None
        self._tick_count = 0

    async def start(self) -> None:
        """Start the heartbeat loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Heartbeat started (interval={self.interval}s)")

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat stopped")

    async def _loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                if not self._running:
                    break

                self._tick_count += 1

                # Build payload
                payload: dict[str, Any] = {
                    "tick": self._tick_count,
                    "timestamp": datetime.now().isoformat(),
                    "interval": self.interval,
                }

                # Add health info if available
                if self.health_fn:
                    try:
                        payload["health"] = self.health_fn()
                    except Exception as e:
                        payload["health_error"] = str(e)

                # Emit heartbeat signal
                await self.bus.emit(Signal(
                    kind=SignalKind.HEARTBEAT,
                    priority=SignalPriority.LOW,
                    source="heartbeat",
                    payload=payload,
                    trigger_mode=TriggerMode.HEARTBEAT,
                ))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def pulse(self) -> None:
        """Manually trigger a single heartbeat (for testing)."""
        self._tick_count += 1
        payload: dict[str, Any] = {
            "tick": self._tick_count,
            "timestamp": datetime.now().isoformat(),
            "manual": True,
        }
        if self.health_fn:
            try:
                payload["health"] = self.health_fn()
            except Exception as e:
                payload["health_error"] = str(e)
        await self.bus.emit(Signal(
            kind=SignalKind.HEARTBEAT,
            priority=SignalPriority.LOW,
            source="heartbeat",
            payload=payload,
            trigger_mode=TriggerMode.HEARTBEAT,
        ))

    @property
    def tick_count(self) -> int:
        return self._tick_count


# ============================================================================
# Cron Trigger
# ============================================================================

class CronJob:
    """A registered cron job."""

    def __init__(
        self,
        name: str,
        interval: float,
        payload: dict[str, Any],
    ):
        self.name = name
        self.interval = interval
        self.payload = payload
        self.last_run: float = 0.0
        self.run_count: int = 0

    def is_due(self, now: float) -> bool:
        """Check if this job is due to run."""
        return (now - self.last_run) >= self.interval

    def mark_run(self, now: float) -> None:
        """Mark this job as having run."""
        self.last_run = now
        self.run_count += 1


class CronTrigger:
    """
    Simple interval-based scheduler.

    Register jobs with every(interval, name, payload).
    The cron loop checks registered jobs and emits CRON signals
    when they're due.

    For cosmic/ritual scheduling, wrap the existing RitualScheduler
    and emit RITUAL signals instead.

    Usage:
        cron = CronTrigger(bus, check_interval=10.0)
        cron.every(300, "memory_consolidation", {"action": "consolidate"})
        cron.every(3600, "mcts_prune", {"min_visits": 2})
        await cron.start()
    """

    def __init__(
        self,
        bus: UnifiedBus,
        check_interval: float = 10.0,
    ):
        """
        Args:
            bus: UnifiedBus to emit cron signals to
            check_interval: How often to check for due jobs (seconds)
        """
        self.bus = bus
        self.check_interval = check_interval

        self._jobs: dict[str, CronJob] = {}
        self._running = False
        self._task: asyncio.Task | None = None

    def every(
        self,
        interval: float,
        name: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a cron job.

        Args:
            interval: Seconds between runs
            name: Job name (used as signal target)
            payload: Payload to include in CRON signal
        """
        self._jobs[name] = CronJob(
            name=name,
            interval=interval,
            payload=payload or {},
        )
        logger.info(f"Cron job registered: '{name}' every {interval}s")

    def remove(self, name: str) -> bool:
        """Remove a registered cron job."""
        return self._jobs.pop(name, None) is not None

    async def start(self) -> None:
        """Start the cron check loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Cron started (check_interval={self.check_interval}s)")

    async def stop(self) -> None:
        """Stop the cron loop."""
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Cron stopped")

    async def _loop(self) -> None:
        """Main cron check loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                if not self._running:
                    break
                await self._check_jobs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cron loop error: {e}")

    async def _check_jobs(self) -> None:
        """Check all jobs and emit signals for due ones."""
        now = time.time()

        for job in self._jobs.values():
            if job.is_due(now):
                job.mark_run(now)

                await self.bus.emit(Signal(
                    kind=SignalKind.CRON,
                    priority=SignalPriority.LOW,
                    source="cron",
                    target=job.name,
                    payload={
                        "job_name": job.name,
                        "run_count": job.run_count,
                        "interval": job.interval,
                        **job.payload,
                    },
                    trigger_mode=TriggerMode.CRON,
                ))

    async def tick(self) -> None:
        """Manually trigger a cron check (for testing)."""
        await self._check_jobs()

    def get_jobs(self) -> dict[str, dict[str, Any]]:
        """Get registered jobs and their status."""
        return {
            name: {
                "interval": job.interval,
                "run_count": job.run_count,
                "last_run": datetime.fromtimestamp(job.last_run).isoformat() if job.last_run else None,
                "payload": job.payload,
            }
            for name, job in self._jobs.items()
        }


# ============================================================================
# Ritual Trigger — Bus ↔ RitualRegistry Bridge
# ============================================================================


class RitualContextFactory:
    """
    Builds RitualContext from a Signal.

    This is the clean seam — the only thing that needs to know how
    Signal payloads map to RitualContext fields.

    Override this to customize context extraction for your signal schema.
    """

    def from_signal(self, signal: Signal) -> Any:
        """
        Extract RitualContext fields from signal payload/metadata.

        Returns a RitualContext (imported lazily to avoid circular deps).
        Falls back to sensible defaults for missing fields.
        """
        from ...ritual.types import RitualContext, TokenBudget

        p = signal.payload
        m = signal.metadata

        return RitualContext(
            agent_id=p.get("agent_id", m.get("agent_id", "unknown")),
            agent_role=p.get("agent_role", m.get("agent_role", "unknown")),
            domain=p.get("domain", m.get("domain", "universal")),
            task_id=p.get("task_id", signal.correlation_id or signal.id),
            task_type=p.get("task_type", signal.kind.value),
            task_intent=p.get("task_intent", p.get("text", p.get("query", ""))),
            task_complexity=p.get("task_complexity", "standard"),
            yarn=p.get("yarn"),
            warp=p.get("warp"),
            confidence=p.get("confidence", 0.8),
            confidence_history=p.get("confidence_history", [0.8]),
            ritual_outputs={},
            ritual_log=[],
            token_budget=TokenBudget(
                total=p.get("token_budget_total", 5000),
                used=p.get("token_budget_used", 0),
            ),
            time_budget=p.get("time_budget", 60.0),
            escalation_path=p.get("escalation_path", []),
            session_id=p.get("session_id", m.get("session_id", "default")),
        )


# Static mapping: SignalKind → HookEvent
# Signals not in this map are ignored by the RitualTrigger.
def _build_signal_to_hook_map() -> dict:
    """Lazy import to avoid circular deps at module level."""
    from ...ritual.hooks import HookEvent

    return {
        SignalKind.QUERY: HookEvent.TASK_RECEIVED,
        SignalKind.EXECUTION_START: HookEvent.TASK_EXECUTING,
        SignalKind.EXECUTION_COMPLETE: HookEvent.TASK_COMPLETED,
        SignalKind.EXECUTION_FAILED: HookEvent.TASK_FAILED,
        SignalKind.MEMORY_STORED: HookEvent.MEMORY_WRITE,
        SignalKind.MEMORY_QUERIED: HookEvent.MEMORY_READ,
        SignalKind.MEMORY_DECAYED: HookEvent.MEMORY_FORGET,
    }


class RitualTrigger:
    """
    Bridge between the UnifiedBus signal system and the RitualRegistry.

    Subscribes to specific SignalKind values on the bus and translates
    them into HookEvent values for the registry. On successful dispatch,
    emits a derived RITUAL signal with results for observability.

    Flat peer to HeartbeatTrigger and CronTrigger.

    Usage:
        from hololoom.core.ritual import RitualRegistry
        from hololoom.core.bus.adapters.trigger_adapters import RitualTrigger

        registry = RitualRegistry()
        # ... register bindings ...
        trigger = RitualTrigger(bus, registry)
        await trigger.start()
    """

    def __init__(
        self,
        bus: UnifiedBus,
        registry: Any,  # RitualRegistry — Any to avoid import at module level
        context_factory: RitualContextFactory | None = None,
    ):
        self.bus = bus
        self.registry = registry
        self._context_factory = context_factory or RitualContextFactory()
        self._signal_to_hook: dict | None = None
        self._running = False
        self._dispatch_count = 0

    def _get_map(self) -> dict:
        """Lazy-init the signal→hook map."""
        if self._signal_to_hook is None:
            self._signal_to_hook = _build_signal_to_hook_map()
        return self._signal_to_hook

    async def start(self) -> None:
        """Subscribe to all mapped signal kinds on the bus."""
        if self._running:
            return
        self._running = True
        for kind in self._get_map():
            self.bus.on(kind, self._handle)
        logger.info(
            "RitualTrigger started (watching %d signal kinds)",
            len(self._get_map()),
        )

    async def stop(self) -> None:
        """Mark as stopped. Bus subscriptions persist until bus is destroyed."""
        if not self._running:
            return
        self._running = False
        logger.info("RitualTrigger stopped")

    async def _handle(self, signal: Signal) -> Signal | None:
        """
        Handle an incoming bus signal:
        1. Resolve to HookEvent
        2. Build RitualContext from signal
        3. Fire the registry
        4. Emit a RITUAL signal with results
        """
        if not self._running:
            return None

        if self.registry is None:
            return None

        hook = self._get_map().get(signal.kind)
        if hook is None:
            return None

        ctx = self._context_factory.from_signal(signal)

        try:
            results = await self.registry.fire(hook, ctx)
        except Exception as e:
            logger.warning(
                "RitualTrigger dispatch failed for %s → %s: %s",
                signal.kind.value, hook.name, e,
            )
            return None

        self._dispatch_count += 1

        # Emit a RITUAL signal with results for observability
        if results:
            return signal.derive(
                kind=SignalKind.RITUAL,
                payload={
                    "hook": hook.name,
                    "results": {k: v.success for k, v in results.items()},
                    "ritual_count": len(results),
                },
                source="ritual_trigger",
            )

        return None

    @property
    def dispatch_count(self) -> int:
        return self._dispatch_count


__all__ = [
    "HeartbeatTrigger",
    "CronTrigger",
    "CronJob",
    "RitualTrigger",
    "RitualContextFactory",
]

"""
TemporalLayer — the clock.

Owns HeartbeatTrigger (60s pulse) + CronTrigger (interval jobs).
Nothing else. Each domain subscribes to clock signals in its own layer.

Cron jobs registered at startup (other layers filter by signal.target):
    memory_decay        — every 300s
    sleep_consolidation — every 1800s
    vault_promotion     — every 300s
    hot_pattern_decay   — every 3600s
    cosmic_check        — every 3600s
"""

from __future__ import annotations

import logging
from typing import Any

from .layer import LayerState

logger = logging.getLogger(__name__)


class TemporalLayer:
    """The clock. Emits HEARTBEAT and CRON signals. Nothing else."""

    name = "temporal"

    def __init__(
        self,
        heartbeat_interval: float = 60.0,
        cron_check_interval: float = 10.0,
    ):
        self._heartbeat_interval = heartbeat_interval
        self._cron_interval = cron_check_interval
        self._heartbeat: Any = None
        self._cron: Any = None
        self._state = LayerState.INIT

    async def start(self, runtime: Any) -> None:
        """Start the clock. Requires runtime.bus."""
        if runtime.bus is None:
            logger.warning("TemporalLayer: no bus — degraded")
            self._state = LayerState.DEGRADED
            return

        try:
            from ..bus.adapters.trigger_adapters import HeartbeatTrigger, CronTrigger

            self._heartbeat = HeartbeatTrigger(
                runtime.bus, interval=self._heartbeat_interval,
            )
            self._cron = CronTrigger(
                runtime.bus, check_interval=self._cron_interval,
            )

            # Standard jobs — other layers subscribe by target name
            self._cron.every(300, "memory_decay")
            self._cron.every(1800, "sleep_consolidation")
            self._cron.every(300, "vault_promotion")
            self._cron.every(3600, "hot_pattern_decay")
            self._cron.every(3600, "cosmic_check")

            await self._heartbeat.start()
            await self._cron.start()
            self._state = LayerState.RUNNING
            logger.info(
                "TemporalLayer started: heartbeat=%ss, cron=%ss, %d jobs",
                self._heartbeat_interval, self._cron_interval,
                len(self._cron.get_jobs()),
            )

        except Exception as e:
            logger.warning("TemporalLayer degraded: %s", e)
            self._state = LayerState.DEGRADED

    async def stop(self) -> None:
        """Stop the clock."""
        if self._heartbeat:
            try:
                await self._heartbeat.stop()
            except Exception as e:
                logger.warning("Heartbeat stop failed: %s", e)
        if self._cron:
            try:
                await self._cron.stop()
            except Exception as e:
                logger.warning("Cron stop failed: %s", e)
        self._state = LayerState.STOPPED

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {"state": self._state.value}
        if self._heartbeat:
            result["ticks"] = self._heartbeat.tick_count
        if self._cron:
            result["cron_jobs"] = self._cron.get_jobs()
        return result

    def state_snapshot(self) -> dict[str, Any]:
        return {"state": self._state.value}

    @property
    def heartbeat(self) -> Any:
        return self._heartbeat

    @property
    def cron(self) -> Any:
        return self._cron

    @property
    def state(self) -> LayerState:
        return self._state

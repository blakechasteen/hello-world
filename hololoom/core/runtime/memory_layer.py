"""
MemoryLayer — decay, consolidation, vault promotion.

Subscribes to clock signals from TemporalLayer:
    HEARTBEAT       → light-touch decay (every 60s)
    CRON "sleep_consolidation" → SleepConsolidator.run_cycle()
    CRON "vault_promotion"     → scan bus, propose to vault

All memory maintenance in one place. Does NOT own storage/retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from .layer import LayerState

logger = logging.getLogger(__name__)


class MemoryLayer:
    """Memory maintenance. Responds to temporal signals."""

    name = "memory"

    def __init__(
        self,
        memory_bus: Any = None,
        consolidator: Any = None,
        vault_agent: str = "hololoom",
        max_promotions_per_cycle: int = 3,
        promotion_importance: float = 0.7,
        promotion_access_count: int = 5,
    ):
        self._bus = memory_bus
        self._consolidator = consolidator
        self._vault_agent = vault_agent
        self._max_promotions = max_promotions_per_cycle
        self._promotion_importance = promotion_importance
        self._promotion_access = promotion_access_count
        self._promoted_ids: set[str] = set()
        self._state = LayerState.INIT
        self._stats = {"decayed": 0, "consolidated": 0, "promoted": 0}

    async def start(self, runtime: Any) -> None:
        """Wire to memory bus and subscribe to clock signals."""
        # Late-wire memory bus
        if self._bus is None:
            self._bus = getattr(runtime, "memory_bus", None)

        if self._bus is None:
            logger.warning("MemoryLayer: no memory bus — degraded")
            self._state = LayerState.DEGRADED
            return

        # Late-wire consolidator
        if self._consolidator is None:
            try:
                from ..memory.sleep_consolidator import SleepConsolidator
                self._consolidator = SleepConsolidator(self._bus)
            except ImportError:
                logger.debug("SleepConsolidator unavailable")

        # Subscribe to clock
        if runtime.bus is not None:
            from ..bus.signal import SignalKind
            runtime.bus.on(SignalKind.HEARTBEAT, self._handle_heartbeat)
            runtime.bus.on(SignalKind.CRON, self._handle_cron)

        self._state = LayerState.RUNNING
        logger.info("MemoryLayer started: bus=%d items", len(self._bus._items))

    async def stop(self) -> None:
        self._state = LayerState.STOPPED

    async def _handle_heartbeat(self, signal: Any) -> None:
        """Every 60s: light-touch decay."""
        if not self._bus:
            return
        try:
            forgotten = await self._bus.decay(
                max_age_seconds=86400, min_importance=0.1,
            )
            self._stats["decayed"] += forgotten
        except Exception as e:
            logger.warning("Decay failed: %s", e)

    async def _handle_cron(self, signal: Any) -> None:
        """Route cron jobs by target name."""
        target = getattr(signal, "target", "")
        if target == "sleep_consolidation":
            await self._consolidate()
        elif target == "vault_promotion":
            await self._promote_to_vault()

    async def _consolidate(self) -> None:
        """Run sleep consolidator cycle."""
        if not self._consolidator:
            return
        try:
            stats = await self._consolidator.run_cycle()
            self._stats["consolidated"] += stats.get("promoted", 0)
        except Exception as e:
            logger.warning("Consolidation failed: %s", e)

    async def _promote_to_vault(self) -> None:
        """Scan bus for high-value items, propose to PARA vault."""
        if not self._bus:
            return

        candidates = []
        for item in self._bus._items.values():
            if (
                item.importance >= self._promotion_importance
                and item.access_count >= self._promotion_access
                and len(item.content) >= 50
                and item.id not in self._promoted_ids
            ):
                candidates.append(item)

        # Best candidates first
        candidates.sort(
            key=lambda i: i.importance * i.access_count, reverse=True,
        )

        for item in candidates[: self._max_promotions]:
            try:
                from hololoom.apps.server.vault_bridge import propose_note

                result = propose_note(
                    title=f"Memory: {item.content[:60]}",
                    content=item.content,
                    links=[],
                    agent=self._vault_agent,
                )
                if result.get("status") == "proposed":
                    self._promoted_ids.add(item.id)
                    self._stats["promoted"] += 1
                    logger.info("Promoted to vault: %s", item.id)
            except ImportError:
                logger.debug("vault_bridge unavailable — skipping promotion")
                break
            except Exception as e:
                logger.warning("Vault promotion failed for %s: %s", item.id, e)

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "bus_items": len(self._bus._items) if self._bus else 0,
            "promoted_count": len(self._promoted_ids),
            "stats": dict(self._stats),
        }

    def state_snapshot(self) -> dict[str, Any]:
        return {"state": self._state.value}

    @property
    def state(self) -> LayerState:
        return self._state

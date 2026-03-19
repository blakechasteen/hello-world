"""
Sleep-Time Consolidator — Background memory maintenance (borrowed from Letta).

Composes existing MemoryConsolidator (consolidation.py) with:
1. Milestone promotion: frequently-accessed important items → PERMANENT (GCC idea)
2. Version log recording of all consolidation changes
3. MemFS tree refresh after consolidation

Integration: standalone background task, or replacement for
MemoryConsolidator.start_background_consolidation().
"""

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .lite_bus import LiteMemoryBus
    from .memfs import MemFS
    from .version_log import VersionLog

logger = logging.getLogger(__name__)

__all__ = ["SleepConsolidator"]


class SleepConsolidator:
    """Background memory maintenance composing existing consolidation
    with versioned change tracking and milestone promotion.

    Does NOT subclass MemoryConsolidator — composes it.
    Uses bus.consolidate() (the simple LiteMemoryBus method) by default.
    """

    def __init__(
        self,
        bus: "LiteMemoryBus",
        version_log: Optional["VersionLog"] = None,
        memfs: Optional["MemFS"] = None,
        milestone_access_threshold: int = 5,
        milestone_importance_threshold: float = 0.7,
        consolidate_max_items: int = 10,
    ):
        """
        Args:
            bus: The memory bus to consolidate.
            version_log: Optional version log for recording changes.
            memfs: Optional MemFS for refreshing file tree after consolidation.
            milestone_access_threshold: Min access_count to qualify for promotion.
            milestone_importance_threshold: Min importance to qualify for promotion.
            consolidate_max_items: Max items per memory_type before consolidation.
        """
        self._bus = bus
        self._version_log = version_log
        self._memfs = memfs
        self._milestone_access = milestone_access_threshold
        self._milestone_importance = milestone_importance_threshold
        self._consolidate_max = consolidate_max_items

        # Background task handle
        self._task: asyncio.Task | None = None
        self._running = False

        # Stats
        self.total_cycles = 0
        self.total_promoted = 0
        self.total_consolidated = 0

    async def run_cycle(self) -> dict[str, Any]:
        """One consolidation cycle.

        1. Consolidate each memory_type that has too many items
        2. Milestone promotion: promote qualified items to PERMANENT
        3. Record milestone in version log
        4. Return stats

        Returns:
            Dict with cycle statistics.
        """
        start = datetime.utcnow()
        cycle_stats: dict[str, Any] = {
            "consolidated": {},
            "promoted": [],
            "timestamp": start.isoformat(),
        }

        # Step 1: Consolidate memory types with too many items
        for memory_type in list(self._bus._by_type.keys()):
            item_count = len(self._bus._by_type[memory_type])
            if item_count > self._consolidate_max:
                summary_id = await self._bus.consolidate(
                    memory_type=memory_type,
                    max_items=self._consolidate_max,
                )
                if summary_id:
                    new_count = len(self._bus._by_type.get(memory_type, []))
                    cycle_stats["consolidated"][memory_type] = {
                        "before": item_count,
                        "after": new_count,
                        "summary_id": summary_id,
                    }
                    self.total_consolidated += item_count - new_count

        # Step 2: Milestone promotion (GCC idea)
        # Items with high access + high importance → make permanent (ttl=None)
        promoted = []
        for item_id, item in self._bus._items.items():
            if (
                item.access_count >= self._milestone_access
                and item.importance >= self._milestone_importance
                and item.ttl_seconds is not None  # Not already permanent
            ):
                item.ttl_seconds = None  # PERMANENT
                promoted.append(item_id)

        cycle_stats["promoted"] = promoted
        self.total_promoted += len(promoted)

        # Step 3: Record milestone in version log
        if self._version_log and (cycle_stats["consolidated"] or promoted):
            self._version_log.record_milestone(
                description=f"Sleep consolidation cycle {self.total_cycles + 1}",
                metadata={
                    "consolidated_types": list(cycle_stats["consolidated"].keys()),
                    "promoted_count": len(promoted),
                    "promoted_ids": promoted[:10],  # Cap for readability
                },
            )

        self.total_cycles += 1
        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        cycle_stats["elapsed_ms"] = elapsed_ms

        logger.info(
            "Sleep consolidation cycle %d: consolidated=%d types, promoted=%d items (%.1fms)",
            self.total_cycles,
            len(cycle_stats["consolidated"]),
            len(promoted),
            elapsed_ms,
        )
        return cycle_stats

    async def start_background(self, interval_seconds: int = 1800):
        """Start as asyncio background task."""
        if self._running:
            logger.warning("Sleep consolidator already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(interval_seconds))
        logger.info("Started sleep consolidator (interval: %ds)", interval_seconds)

    async def stop_background(self):
        """Cancel background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped sleep consolidator")

    async def _loop(self, interval_seconds: int):
        """Background loop."""
        while self._running:
            try:
                await asyncio.sleep(interval_seconds)
                await self.run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in sleep consolidation: %s", e, exc_info=True)

    def statistics(self) -> dict[str, Any]:
        """Return cumulative stats."""
        return {
            "total_cycles": self.total_cycles,
            "total_promoted": self.total_promoted,
            "total_consolidated": self.total_consolidated,
            "running": self._running,
        }

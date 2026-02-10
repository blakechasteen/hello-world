"""
Promotion Engine — 2026-02-09, updated 2026-02-10

Handles write-through and deferred promotion of memory items.

MVP rules:
- Immediate promotion (write-through):
  - explicit save requests
  - entity creation/modification
  - plan state changes
  - decisions/commitments (if tagged)
- Deferred promotion hook (interface): end-of-loop batch
- Periodic flush timer: auto-flushes deferred queue on interval
- Basic dedupe/novelty scaffold (simple heuristics)
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Optional

from .models import (
    AuditAction,
    AuditRow,
    Entity,
    EntityAlias,
    Episode,
    MemoryStoreRequest,
    PlanRow,
)
from .neo4j_adapter import Neo4jAdapter
from .postgres_adapter import PostgresAdapter

logger = logging.getLogger(__name__)


class PromotionEngine:
    """
    Manages immediate and deferred promotion of memory items.

    Immediate: write-through to Neo4j/Postgres on every call.
    Deferred: queue items for batch processing (hook for future).
    """

    def __init__(
        self,
        *,
        neo4j: Neo4jAdapter,
        pg: PostgresAdapter,
        deferred_queue_max: int = 100,
        flush_interval_seconds: float = 30.0,
    ):
        self._neo4j = neo4j
        self._pg = pg
        self._deferred_queue: deque[MemoryStoreRequest] = deque(maxlen=deferred_queue_max)
        self._flush_interval = flush_interval_seconds
        self._flush_task: Optional[asyncio.Task] = None

    async def promote_immediate(
        self,
        request: MemoryStoreRequest,
    ) -> dict[str, list[str]]:
        """
        Write-through promotion: persist entities, episodes, aliases, plans immediately.

        Returns dict of created/updated IDs by type.
        """
        result: dict[str, list[str]] = {
            "entities": [],
            "episodes": [],
            "aliases": [],
            "plans": [],
        }

        # Entities
        for entity in request.entities:
            if self._is_novel_entity(entity):
                await self._neo4j.create_entity(entity)
                result["entities"].append(entity.id)
                # Auto-create alias
                alias = EntityAlias(
                    alias=entity.name.lower(),
                    entity_id=entity.id,
                    loom_id=request.loom_id,
                    confidence=1.0,
                )
                await self._pg.upsert_alias(alias)
                result["aliases"].append(entity.name.lower())

        # Episodes
        for episode in request.episodes:
            if self._is_novel_episode(episode):
                await self._neo4j.create_episode(episode)
                result["episodes"].append(episode.id)

        # Aliases (explicit)
        for alias in request.aliases:
            await self._pg.upsert_alias(alias)
            result["aliases"].append(alias.alias)

        # Plans (always write-through per invariant #4)
        for plan in request.plans:
            await self._pg.upsert_plan(plan)
            result["plans"].append(plan.id)

        # Audit the promotion
        await self._pg.write_audit(
            AuditRow(
                loop_id=request.loop_id,
                loom_id=request.loom_id,
                action=AuditAction.PROMOTE,
                details={
                    "promoted": result,
                    "immediate": True,
                },
            )
        )

        return result

    def queue_deferred(self, request: MemoryStoreRequest) -> int:
        """
        Queue a store request for deferred/batch promotion.
        Returns current queue depth.
        """
        self._deferred_queue.append(request)
        return len(self._deferred_queue)

    async def flush_deferred(self) -> dict[str, list[str]]:
        """
        Flush the deferred queue (end-of-loop batch promotion hook).
        """
        merged: dict[str, list[str]] = {
            "entities": [],
            "episodes": [],
            "aliases": [],
            "plans": [],
        }
        while self._deferred_queue:
            request = self._deferred_queue.popleft()
            result = await self.promote_immediate(request)
            for k in merged:
                merged[k].extend(result.get(k, []))
        return merged

    @property
    def deferred_queue_depth(self) -> int:
        return len(self._deferred_queue)

    # ---- Periodic flush timer ------------------------------------------------

    def start_periodic_flush(self) -> None:
        """
        Start a background task that flushes the deferred queue
        every flush_interval_seconds.

        Call this before entering a multi-turn reasoning loop so that
        insights from turn N are persisted before turn N+1.
        """
        if self._flush_task is not None and not self._flush_task.done():
            return  # Already running
        self._flush_task = asyncio.create_task(self._periodic_flush_loop())
        logger.info(
            "Deferred promotion flush started (interval=%.0fs)",
            self._flush_interval,
        )

    async def stop_periodic_flush(self) -> None:
        """Cancel the periodic flush and do one final drain."""
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final drain
        if self._deferred_queue:
            await self.flush_deferred()
        logger.info("Deferred promotion flush stopped")

    async def _periodic_flush_loop(self) -> None:
        """Background loop: sleep → flush → repeat."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                if self._deferred_queue:
                    count = len(self._deferred_queue)
                    result = await self.flush_deferred()
                    total = sum(len(v) for v in result.values())
                    logger.info(
                        "Periodic flush: %d requests → %d items promoted",
                        count, total,
                    )
        except asyncio.CancelledError:
            return

    # ---- Novelty / Dedupe heuristics (simple MVP) -------------------------

    @staticmethod
    def _is_novel_entity(entity: Entity) -> bool:
        """
        Basic novelty check for entities.
        MVP: always novel (dedupe handled by MERGE in Neo4j).
        Hook for future: check embedding similarity, name fuzzy match.
        """
        return True

    @staticmethod
    def _is_novel_episode(episode: Episode) -> bool:
        """
        Basic novelty check for episodes.
        MVP: always novel.
        Hook for future: check summary similarity, time proximity.
        """
        return True

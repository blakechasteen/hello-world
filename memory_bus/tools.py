"""
MCP tool implementations — 2026-02-09

Tools:
1. memory_query(...)
2. memory_store(...)
3. memory_entities(...)
4. memory_resolve_entity(...)
5. memory_explain(...)

Resources:
- memory://status
- memory://pressure
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .explain import explain_item
from .formatter import format_items
from .models import (
    AuditAction,
    AuditRow,
    DetailLevel,
    Entity,
    EntityAlias,
    EntityType,
    Episode,
    EpisodeType,
    MemoryQuery,
    MemoryResult,
    MemoryStoreRequest,
    PlanRow,
    PressureStatus,
    ResolveResult,
    ExplainResult,
)
from .neo4j_adapter import Neo4jAdapter
from .postgres_adapter import PostgresAdapter
from .pressure import PressureEngine
from .promotion import PromotionEngine
from .resolve import resolve_entity
from .router import MemoryRouter
from .verify import EntityVerifier, VerificationResult

logger = logging.getLogger(__name__)


class MemoryBus:
    """
    Top-level Memory Bus facade exposing MCP tools and resources.

    Usage:
        bus = MemoryBus(neo4j_uri=..., pg_dsn=...)
        await bus.connect()
        result = await bus.memory_query(text="...", entity_names=["Aurora"])
        await bus.close()
    """

    def __init__(
        self,
        *,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "hololoom",
        pg_dsn: str = "postgresql://hololoom:hololoom@localhost:5432/hololoom",
        context_window: int = 128_000,
    ):
        self._neo4j = Neo4jAdapter(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        self._pg = PostgresAdapter(dsn=pg_dsn)
        self._pressure = PressureEngine(context_window=context_window)
        self._router: Optional[MemoryRouter] = None
        self._promotion: Optional[PromotionEngine] = None
        self._verifier: Optional[EntityVerifier] = None
        self._context_window = context_window

    # ---- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        await self._neo4j.connect()
        await self._pg.connect()
        self._router = MemoryRouter(
            neo4j=self._neo4j,
            pg=self._pg,
            pressure=self._pressure,
            context_window=self._context_window,
        )
        self._promotion = PromotionEngine(neo4j=self._neo4j, pg=self._pg)
        self._verifier = EntityVerifier(neo4j=self._neo4j, pg=self._pg)
        logger.info("MemoryBus connected")

    async def close(self) -> None:
        # Flush deferred queue before closing
        if self._promotion and self._promotion.deferred_queue_depth > 0:
            await self._promotion.flush_deferred()
        await self._neo4j.close()
        await self._pg.close()
        logger.info("MemoryBus closed")

    async def __aenter__(self) -> MemoryBus:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ---- MCP Tool 1: memory_query -----------------------------------------

    async def memory_query(
        self,
        *,
        text: str = "",
        entity_ids: list[str] | None = None,
        entity_names: list[str] | None = None,
        episode_types: list[str] | None = None,
        time_range_start: str | None = None,
        time_range_end: str | None = None,
        max_results: int = 20,
        detail_level: str = "compact",
        token_budget: int | None = None,
        loom_id: str | None = None,
        loop_id: str | None = None,
        aggressive_prime: bool = False,
    ) -> MemoryResult:
        """Query memory with structured-first routing cascade."""
        from datetime import datetime

        ep_types = [EpisodeType(t) for t in (episode_types or [])]
        t_start = datetime.fromisoformat(time_range_start) if time_range_start else None
        t_end = datetime.fromisoformat(time_range_end) if time_range_end else None

        query = MemoryQuery(
            text=text,
            entity_ids=entity_ids or [],
            entity_names=entity_names or [],
            episode_types=ep_types,
            time_range_start=t_start,
            time_range_end=t_end,
            max_results=max_results,
            detail_level=DetailLevel(detail_level),
            token_budget=token_budget,
            loom_id=loom_id,
            loop_id=loop_id,
            aggressive_prime=aggressive_prime,
        )

        assert self._router is not None, "MemoryBus not connected"
        return await self._router.route(query)

    # ---- MCP Tool 2: memory_store -----------------------------------------

    async def memory_store(
        self,
        *,
        entities: list[dict[str, Any]] | None = None,
        episodes: list[dict[str, Any]] | None = None,
        aliases: list[dict[str, Any]] | None = None,
        plans: list[dict[str, Any]] | None = None,
        loom_id: str | None = None,
        loop_id: str | None = None,
        immediate: bool = True,
    ) -> dict[str, list[str]]:
        """Store/promote memory items (entities, episodes, aliases, plans)."""
        assert self._promotion is not None, "MemoryBus not connected"

        request = MemoryStoreRequest(
            entities=[Entity(**e) for e in (entities or [])],
            episodes=[Episode(**e) for e in (episodes or [])],
            aliases=[EntityAlias(**a) for a in (aliases or [])],
            plans=[PlanRow(**p) for p in (plans or [])],
            loom_id=loom_id,
            loop_id=loop_id,
            immediate=immediate,
        )

        if immediate:
            return await self._promotion.promote_immediate(request)
        else:
            depth = self._promotion.queue_deferred(request)
            return {"deferred_queue_depth": [str(depth)]}

    # ---- MCP Tool 3: memory_entities --------------------------------------

    async def memory_entities(
        self,
        *,
        entity_id: str | None = None,
        name: str | None = None,
        loom_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List or search entities."""
        if entity_id:
            entity = await self._neo4j.get_entity(entity_id)
            if entity:
                return [entity.model_dump(mode="json")]
            return []

        if name:
            entities = await self._neo4j.search_entities_by_name(
                name, loom_origin=loom_id, limit=limit
            )
            return [e.model_dump(mode="json") for e in entities]

        # No filters — return recent (by importance)
        # For MVP, we query recent episodes to find referenced entities
        return []

    # ---- MCP Tool 4: memory_resolve_entity --------------------------------

    async def memory_resolve_entity(
        self,
        *,
        name: str,
        loom_id: str | None = None,
        max_candidates: int = 5,
        loop_id: str | None = None,
    ) -> ResolveResult:
        """Resolve a name to ranked entity candidates."""
        result = await resolve_entity(
            name,
            pg=self._pg,
            neo4j=self._neo4j,
            loom_id=loom_id,
            max_candidates=max_candidates,
        )

        # Audit
        await self._pg.write_audit(
            AuditRow(
                loop_id=loop_id,
                loom_id=loom_id,
                action=AuditAction.RESOLVE,
                details={
                    "query_name": name,
                    "candidate_count": len(result.candidates),
                    "best_match": result.best_match.entity_id if result.best_match else None,
                    "best_confidence": result.best_match.confidence if result.best_match else 0,
                },
            )
        )

        return result

    # ---- MCP Tool 5: memory_explain ---------------------------------------

    async def memory_explain(
        self,
        *,
        item_id: str,
        loom_id: str | None = None,
        loop_id: str | None = None,
    ) -> ExplainResult:
        """Return provenance chain for a memory item."""
        result = await explain_item(
            item_id,
            pg=self._pg,
            neo4j=self._neo4j,
            loom_id=loom_id,
        )

        # Audit
        await self._pg.write_audit(
            AuditRow(
                loop_id=loop_id,
                loom_id=loom_id,
                action=AuditAction.EXPLAIN,
                details={
                    "item_id": item_id,
                    "derived_from_count": len(result.derived_from),
                    "confidence": result.confidence,
                    "status": result.status,
                },
            )
        )

        return result

    # ---- MCP Resource: memory://status ------------------------------------

    async def memory_status(self) -> dict[str, Any]:
        """Return memory bus status."""
        neo4j_ok = self._neo4j._driver is not None
        pg_ok = self._pg._pool is not None

        return {
            "status": "healthy" if (neo4j_ok and pg_ok) else "degraded",
            "neo4j_connected": neo4j_ok,
            "postgres_connected": pg_ok,
            "pressure_tier": self._pressure.tier.value,
            "context_window": self._context_window,
            "deferred_queue_depth": (
                self._promotion.deferred_queue_depth if self._promotion else 0
            ),
        }

    # ---- MCP Resource: memory://pressure ----------------------------------

    def memory_pressure(self) -> PressureStatus:
        """Return current pressure tier + signals + policy."""
        return self._pressure.status()

    # ---- MCP Tool 6: memory_verify -----------------------------------------

    async def memory_verify(
        self,
        *,
        response_text: str,
        primed_entity_ids: list[str] | None = None,
        primed_entity_names: list[str] | None = None,
        loom_id: str | None = None,
        loop_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Post-response entity verification.

        Checks whether entities mentioned in model response are grounded
        in memory (via primed context, aliases, or Neo4j search).
        Returns confabulation rate and ungrounded entity list.
        """
        assert self._verifier is not None, "MemoryBus not connected"
        result = await self._verifier.verify(
            response_text=response_text,
            primed_entity_ids=primed_entity_ids,
            primed_entity_names=primed_entity_names,
            loom_id=loom_id,
            loop_id=loop_id,
        )
        return result.to_dict()

    # ---- MCP Resource: memory://confabulation_stats -------------------------

    async def confabulation_stats(
        self,
        *,
        loom_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Aggregate confabulation metrics from audit log.

        Returns overall confabulation rate, trend, and per-loop breakdown.
        """
        try:
            rows = await self._pg.get_audit_rows(
                loom_id=loom_id,
                action=AuditAction.VERIFY,
                limit=limit,
            )
        except Exception:
            return {
                "total_verifications": 0,
                "avg_confabulation_rate": 0.0,
                "avg_grounding_rate": 1.0,
                "trend": [],
            }

        if not rows:
            return {
                "total_verifications": 0,
                "avg_confabulation_rate": 0.0,
                "avg_grounding_rate": 1.0,
                "trend": [],
            }

        rates: list[float] = []
        total_grounded = 0
        total_ungrounded = 0
        trend: list[dict[str, Any]] = []

        for row in rows:
            details = row.details or {}
            rate = details.get("confabulation_rate", 0.0)
            rates.append(rate)
            total_grounded += details.get("grounded_count", 0)
            total_ungrounded += details.get("ungrounded_count", 0)
            trend.append({
                "loop_id": row.loop_id,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "confabulation_rate": rate,
                "total_mentions": details.get("total_mentions", 0),
            })

        total_mentions = total_grounded + total_ungrounded
        overall_rate = (
            total_ungrounded / total_mentions if total_mentions > 0 else 0.0
        )

        return {
            "total_verifications": len(rows),
            "total_entity_mentions": total_mentions,
            "total_grounded": total_grounded,
            "total_ungrounded": total_ungrounded,
            "avg_confabulation_rate": round(overall_rate, 4),
            "avg_grounding_rate": round(1.0 - overall_rate, 4),
            "trend": trend[:20],  # Most recent 20
        }

    # ---- Pressure management (for orchestrator use) -----------------------

    def update_pressure(self, *, context_tokens_used: int) -> None:
        """Update pressure signals (called by orchestrator)."""
        self._pressure.update(context_tokens_used=context_tokens_used)

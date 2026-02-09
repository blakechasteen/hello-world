"""
Query routing cascade — 2026-02-09

Structured-first routing:
1. EXACT: entity_id / node id direct fetch
2. STRUCTURED: Cypher/SQL deterministic queries
3. GRAPH: Neo4j traversal
4. SEMANTIC: STUB (return "not available" + guidance)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .models import (
    AuditAction,
    AuditRow,
    DetailLevel,
    MemoryItem,
    MemoryQuery,
    MemoryResult,
    PressureTier,
    ResolutionPath,
)
from .formatter import format_items
from .neo4j_adapter import Neo4jAdapter
from .postgres_adapter import PostgresAdapter
from .pressure import PressureEngine

logger = logging.getLogger(__name__)


class MemoryRouter:
    """
    Routes queries through the cascade: EXACT → STRUCTURED → GRAPH → SEMANTIC(stub).

    Applies pressure-aware policies, formats results, and writes audit rows.
    """

    def __init__(
        self,
        *,
        neo4j: Neo4jAdapter,
        pg: PostgresAdapter,
        pressure: PressureEngine,
        context_window: int = 128_000,
    ):
        self._neo4j = neo4j
        self._pg = pg
        self._pressure = pressure
        self._context_window = context_window

    async def route(self, query: MemoryQuery) -> MemoryResult:
        """
        Execute query through the routing cascade.

        Returns MemoryResult with items, formatted context, and metadata.
        """
        start = time.monotonic()
        items: list[MemoryItem] = []
        resolution_path = ResolutionPath.EXACT
        warnings: list[str] = []

        tier = self._pressure.tier
        min_importance = self._pressure.min_importance()

        # Effective token budget
        token_budget = self._pressure.get_token_cap(query.token_budget)

        try:
            # ---- EXACT: Direct entity/episode fetch by ID -----------------
            if query.entity_ids:
                resolution_path = ResolutionPath.EXACT
                for eid in query.entity_ids:
                    entity = await self._neo4j.get_entity(eid)
                    if entity and entity.importance >= min_importance:
                        items.append(self._neo4j.entity_to_item(entity))
                    episode = await self._neo4j.get_episode(eid)
                    if episode and episode.significance >= min_importance:
                        items.append(self._neo4j.episode_to_item(episode))

                # Also fetch episodes for these entities
                if items:
                    entity_items = [i for i in items if i.kind == "entity"]
                    for ei in entity_items:
                        episodes = await self._neo4j.query_episodes_for_entity(
                            ei.id,
                            episode_types=query.episode_types or None,
                            time_start=query.time_range_start,
                            time_end=query.time_range_end,
                            limit=query.max_results,
                        )
                        for ep in episodes:
                            if ep.significance >= min_importance:
                                items.append(self._neo4j.episode_to_item(ep))

            # ---- STRUCTURED: query by entity names → resolve → episodes ---
            if not items and query.entity_names:
                resolution_path = ResolutionPath.STRUCTURED
                from .resolve import resolve_entity

                for name in query.entity_names:
                    resolve_result = await resolve_entity(
                        name, pg=self._pg, neo4j=self._neo4j, loom_id=query.loom_id
                    )
                    if resolve_result.best_match:
                        eid = resolve_result.best_match.entity_id
                        entity = await self._neo4j.get_entity(eid)
                        if entity and entity.importance >= min_importance:
                            items.append(self._neo4j.entity_to_item(entity))
                        episodes = await self._neo4j.query_episodes_for_entity(
                            eid,
                            episode_types=query.episode_types or None,
                            time_start=query.time_range_start,
                            time_end=query.time_range_end,
                            limit=query.max_results,
                        )
                        for ep in episodes:
                            if ep.significance >= min_importance:
                                items.append(self._neo4j.episode_to_item(ep))

            # ---- GRAPH: recent episodes + graph traversal -----------------
            if not items and query.text:
                resolution_path = ResolutionPath.GRAPH
                episodes = await self._neo4j.query_recent_episodes(
                    loom_origin=query.loom_id,
                    limit=query.max_results,
                )
                for ep in episodes:
                    if ep.significance >= min_importance:
                        items.append(self._neo4j.episode_to_item(ep))

            # ---- SEMANTIC: STUB -------------------------------------------
            if not items:
                resolution_path = ResolutionPath.SEMANTIC
                if not self._pressure.is_semantic_allowed():
                    warnings.append(
                        "Semantic fallback disabled at pressure "
                        f"{tier.value}. Provide entity IDs or names for "
                        "structured retrieval."
                    )
                else:
                    warnings.append(
                        "Semantic search not available in MVP. "
                        "Use entity_ids or entity_names for structured queries. "
                        "Semantic (embedding-based) retrieval is deferred."
                    )

        except Exception as exc:
            logger.error("Memory query failed: %s", exc, exc_info=True)
            warnings.append(f"Query error: {exc}")
            # Audit the error
            await self._audit(
                query, resolution_path, 0, tier,
                error=str(exc),
            )
            return MemoryResult(
                items=[],
                resolution_path=resolution_path,
                pressure_tier=tier,
                warnings=warnings,
            )

        # Deduplicate by ID
        seen: set[str] = set()
        unique_items: list[MemoryItem] = []
        for item in items:
            if item.id not in seen:
                seen.add(item.id)
                item.resolution_path = resolution_path
                unique_items.append(item)

        # Trim to max_results
        unique_items = unique_items[: query.max_results]

        # Format with token budget
        formatted_text, tokens_used, actual_detail = format_items(
            unique_items,
            token_budget=token_budget,
            detail_level=query.detail_level,
            context_window=self._context_window,
            pressure_tier=tier,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Update pressure signals
        self._pressure.update(
            retrieval_result_count=len(unique_items),
            retrieval_latency_ms=elapsed_ms,
        )

        # Write audit
        await self._audit(
            query, resolution_path, tokens_used, tier,
            result_count=len(unique_items),
            latency_ms=elapsed_ms,
            result_ids=[i.id for i in unique_items[:10]],
        )

        return MemoryResult(
            items=unique_items,
            total_tokens_used=tokens_used,
            resolution_path=resolution_path,
            detail_level=actual_detail,
            pressure_tier=tier,
            formatted_context=formatted_text,
            warnings=warnings,
        )

    async def _audit(
        self,
        query: MemoryQuery,
        path: ResolutionPath,
        tokens: int,
        tier: PressureTier,
        *,
        error: str | None = None,
        result_count: int = 0,
        latency_ms: float = 0.0,
        result_ids: list[str] | None = None,
    ) -> None:
        try:
            details: dict = {
                "query_text": query.text[:200] if query.text else "",
                "entity_ids": query.entity_ids[:5],
                "entity_names": query.entity_names[:5],
                "result_count": result_count,
                "latency_ms": round(latency_ms, 2),
            }
            if error:
                details["error"] = error
            if result_ids:
                details["result_ids"] = result_ids

            await self._pg.write_audit(
                AuditRow(
                    loop_id=query.loop_id,
                    loom_id=query.loom_id,
                    action=AuditAction.ERROR if error else AuditAction.QUERY,
                    resolution_path=path,
                    tokens_used=tokens,
                    pressure_tier=tier,
                    details=details,
                )
            )
        except Exception as exc:
            logger.error("Failed to write audit row: %s", exc)

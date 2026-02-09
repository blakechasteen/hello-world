"""
Provenance / explain — 2026-02-09

memory_explain(result_id) returns provenance chain including:
- derived_from (episode ids, entity ids)
- reasoning loop id
- confidence + status (confirmed vs inferred)
- contradiction edges if present
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import AuditAction, AuditRow, ExplainResult
from .neo4j_adapter import Neo4jAdapter
from .postgres_adapter import PostgresAdapter

logger = logging.getLogger(__name__)


async def explain_item(
    item_id: str,
    *,
    pg: PostgresAdapter,
    neo4j: Neo4jAdapter,
    loom_id: Optional[str] = None,
) -> ExplainResult:
    """
    Build a provenance chain for a memory item.

    1. Look up the item in Neo4j (entity or episode).
    2. Find related audit rows from Postgres.
    3. Find derived_from relationships (ABOUT edges for episodes).
    4. Check for contradiction edges (scaffold).
    """
    derived_from: list[str] = []
    confidence = 0.0
    status = "confirmed"
    loop_id: Optional[str] = None

    # Try as episode first
    episode = await neo4j.get_episode(item_id)
    if episode:
        derived_from = list(episode.entity_ids)
        confidence = episode.significance
        status = "confirmed"
    else:
        # Try as entity
        entity = await neo4j.get_entity(item_id)
        if entity:
            confidence = entity.importance
            status = "confirmed"
            # Find episodes that reference this entity
            try:
                episodes = await neo4j.query_episodes_for_entity(entity.id, limit=5)
                derived_from = [ep.id for ep in episodes]
            except Exception:
                pass

    # Find audit trail entries for this item
    audit_rows: list[AuditRow] = []
    try:
        all_audits = await pg.get_audit_rows(loom_id=loom_id, limit=200)
        for row in all_audits:
            details = row.details or {}
            # Match audit rows that mention this item_id
            if (
                details.get("item_id") == item_id
                or details.get("entity_id") == item_id
                or details.get("episode_id") == item_id
                or item_id in details.get("entity_ids", [])
                or item_id in details.get("result_ids", [])
            ):
                audit_rows.append(row)
                if row.loop_id and not loop_id:
                    loop_id = row.loop_id
    except Exception as exc:
        logger.warning("Audit lookup failed for %r: %s", item_id, exc)

    # Contradiction detection scaffold (MVP: stub)
    contradictions: list[str] = []
    # Future: query Neo4j for :CONTRADICTS relationships

    return ExplainResult(
        item_id=item_id,
        derived_from=derived_from,
        loop_id=loop_id,
        confidence=confidence,
        status=status,
        contradictions=contradictions,
        audit_trail=audit_rows[:20],
    )

"""
Entity resolution — 2026-02-09

Prevents wrong-ID cascades by resolving names to entity IDs via:
1. Postgres entity_aliases lookup (exact match, ranked by confidence)
2. Neo4j name search fallback (CONTAINS, ranked by importance)
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import EntityType, ResolveCandidate, ResolveResult
from .neo4j_adapter import Neo4jAdapter
from .postgres_adapter import PostgresAdapter

logger = logging.getLogger(__name__)


async def resolve_entity(
    name: str,
    *,
    pg: PostgresAdapter,
    neo4j: Neo4jAdapter,
    loom_id: Optional[str] = None,
    max_candidates: int = 5,
) -> ResolveResult:
    """
    Resolve a name string to ranked entity candidates.

    Strategy:
    1. Check Postgres entity_aliases for exact alias matches.
    2. Fall back to Neo4j name search (CONTAINS) if aliases insufficient.
    3. Merge, deduplicate, rank by confidence desc.
    """
    candidates: list[ResolveCandidate] = []
    seen_ids: set[str] = set()

    # Step 1: Postgres alias lookup
    try:
        aliases = await pg.find_aliases(name, loom_id=loom_id)
        for alias in aliases:
            if alias.entity_id not in seen_ids:
                # Fetch entity details from Neo4j for name/type
                entity = await neo4j.get_entity(alias.entity_id)
                candidates.append(
                    ResolveCandidate(
                        entity_id=alias.entity_id,
                        name=entity.name if entity else alias.alias,
                        type=entity.type if entity else EntityType.OTHER,
                        confidence=alias.confidence,
                        source="alias",
                    )
                )
                seen_ids.add(alias.entity_id)
    except Exception as exc:
        logger.warning("Alias lookup failed for %r: %s", name, exc)

    # Step 2: Neo4j name search fallback
    if len(candidates) < max_candidates:
        try:
            entities = await neo4j.search_entities_by_name(
                name, loom_origin=loom_id, limit=max_candidates
            )
            for entity in entities:
                if entity.id not in seen_ids:
                    # Heuristic confidence: exact match = 0.8, partial = 0.5
                    name_lower = name.lower()
                    ent_lower = entity.name.lower()
                    if ent_lower == name_lower:
                        conf = 0.8
                    elif name_lower in ent_lower or ent_lower in name_lower:
                        conf = 0.6
                    else:
                        conf = 0.4
                    candidates.append(
                        ResolveCandidate(
                            entity_id=entity.id,
                            name=entity.name,
                            type=entity.type,
                            confidence=conf,
                            source="neo4j_name_search",
                        )
                    )
                    seen_ids.add(entity.id)
        except Exception as exc:
            logger.warning("Neo4j name search failed for %r: %s", name, exc)

    # Sort by confidence descending
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    candidates = candidates[:max_candidates]

    best = candidates[0] if candidates else None

    return ResolveResult(
        query_name=name,
        candidates=candidates,
        best_match=best,
    )

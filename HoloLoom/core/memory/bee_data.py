"""
Bee Inspection Data Loader — loads real hive entities and episodes
into Neo4j + Postgres for integration testing and validation.

Uses domain knowledge from memory/stores/beekeeping_strategy.py
(seasons, hive priority, genetics).
"""

import logging
from datetime import datetime
from typing import Optional

from .bus import MemoryBus, MemoryItem
from .neo4j_schema import (
    NODE_ENTITY,
    NODE_EPISODE,
    REL_ABOUT,
    REL_CAUSED,
    REL_FOLLOWED_BY,
    REL_LOCATED_AT,
    REL_PART_OF,
    REL_RELATED_TO,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Hive Entities
# =============================================================================

HIVE_ENTITIES = [
    {
        "id": "hive_aurora",
        "type": "Hive",
        "name": "Aurora",
        "genetics": "Carniolan",
        "location": "north_yard",
        "status": "strong",
        "frames": 8,
    },
    {
        "id": "hive_jodi",
        "type": "Hive",
        "name": "Jodi",
        "genetics": "jodi-line",
        "location": "south_yard",
        "status": "very_strong",
        "frames": 15,
    },
    {
        "id": "hive_delta",
        "type": "Hive",
        "name": "Delta",
        "genetics": "mixed",
        "location": "south_yard",
        "status": "moderate",
        "frames": 6,
    },
    {
        "id": "hive_dennis",
        "type": "Hive",
        "name": "Dennis",
        "genetics": "dennis-line",
        "location": "east_yard",
        "status": "strong",
        "frames": 10,
    },
]

LOCATION_ENTITIES = [
    {"id": "loc_north_yard", "type": "Location", "name": "North Yard"},
    {"id": "loc_south_yard", "type": "Location", "name": "South Yard"},
    {"id": "loc_east_yard", "type": "Location", "name": "East Yard"},
]

APIARY_ENTITY = {
    "id": "apiary_main",
    "type": "Apiary",
    "name": "Main Apiary",
}

# =============================================================================
# Inspection Episodes
# =============================================================================

EPISODES = [
    {
        "id": "ep_aurora_inspect_jan08",
        "type": "inspection",
        "summary": "Aurora strong, 8 frames of bees, queen present, good brood pattern",
        "timestamp": "2025-01-08T10:00:00",
        "significance": 0.7,
        "entity_ids": ["hive_aurora"],
    },
    {
        "id": "ep_aurora_treatment_jan15",
        "type": "treatment",
        "summary": "OA vaporization applied to Aurora. Mite drop count: 42",
        "timestamp": "2025-01-15T14:00:00",
        "significance": 0.8,
        "entity_ids": ["hive_aurora"],
    },
    {
        "id": "ep_aurora_inspect_jan22",
        "type": "inspection",
        "summary": "Aurora stable post-treatment, population holding, broodless period continuing",
        "timestamp": "2025-01-22T10:00:00",
        "significance": 0.6,
        "entity_ids": ["hive_aurora"],
    },
    {
        "id": "ep_jodi_inspect_jan10",
        "type": "inspection",
        "summary": "Jodi very strong, 15 frames, excellent queen, preparing for spring split",
        "timestamp": "2025-01-10T10:00:00",
        "significance": 0.8,
        "entity_ids": ["hive_jodi"],
    },
    {
        "id": "ep_delta_inspect_jan12",
        "type": "inspection",
        "summary": "Delta moderate, 6 frames, queen present but slow brood buildup",
        "timestamp": "2025-01-12T10:00:00",
        "significance": 0.5,
        "entity_ids": ["hive_delta"],
    },
    {
        "id": "ep_delta_foraging_feb01",
        "type": "observation",
        "summary": "Delta showing 30% higher foraging activity near south garden lavender/borage",
        "timestamp": "2025-02-01T11:00:00",
        "significance": 0.7,
        "entity_ids": ["hive_delta"],
    },
    {
        "id": "ep_dennis_inspect_jan14",
        "type": "inspection",
        "summary": "Dennis strong, 10 frames, good winter cluster, adequate honey stores",
        "timestamp": "2025-01-14T10:00:00",
        "significance": 0.6,
        "entity_ids": ["hive_dennis"],
    },
    {
        "id": "ep_all_winter_prep",
        "type": "planning",
        "summary": "Winter prep complete: all hives wrapped, insulation added, entrance reducers installed",
        "timestamp": "2024-11-15T09:00:00",
        "significance": 0.9,
        "entity_ids": ["hive_aurora", "hive_jodi", "hive_delta", "hive_dennis"],
    },
]

# Causal chains: treatment → observation
CAUSAL_LINKS = [
    ("ep_aurora_treatment_jan15", "ep_aurora_inspect_jan22"),   # treatment caused follow-up inspection
    ("ep_all_winter_prep", "ep_aurora_inspect_jan08"),          # winter prep led to first January checks
]

# Temporal ordering
TEMPORAL_LINKS = [
    ("ep_aurora_inspect_jan08", "ep_aurora_treatment_jan15"),
    ("ep_aurora_treatment_jan15", "ep_aurora_inspect_jan22"),
]

# =============================================================================
# Entity Aliases (for Postgres)
# =============================================================================

ENTITY_ALIASES = [
    # Aurora
    ("Aurora", "hive_aurora", "bee", 1.0),
    ("Hive Aurora", "hive_aurora", "bee", 1.0),
    ("aurora", "hive_aurora", "bee", 0.9),
    ("hive aurora", "hive_aurora", "bee", 0.9),
    # Jodi
    ("Jodi", "hive_jodi", "bee", 1.0),
    ("Hive Jodi", "hive_jodi", "bee", 1.0),
    ("jodi", "hive_jodi", "bee", 0.9),
    ("Jodi line", "hive_jodi", "bee", 0.8),
    # Delta
    ("Delta", "hive_delta", "bee", 1.0),
    ("Hive Delta", "hive_delta", "bee", 1.0),
    ("delta", "hive_delta", "bee", 0.9),
    # Dennis
    ("Dennis", "hive_dennis", "bee", 1.0),
    ("Hive Dennis", "hive_dennis", "bee", 1.0),
    ("dennis", "hive_dennis", "bee", 0.9),
    ("Dennis line", "hive_dennis", "bee", 0.8),
    # Locations
    ("North Yard", "loc_north_yard", "bee", 1.0),
    ("South Yard", "loc_south_yard", "bee", 1.0),
    ("East Yard", "loc_east_yard", "bee", 1.0),
    # Apiary
    ("Main Apiary", "apiary_main", "bee", 1.0),
]

# =============================================================================
# Facts (for Postgres fast lookup)
# =============================================================================

FACTS = [
    ("hive_aurora", "Aurora genetics: Carniolan. Typical brood break: late Dec - early Feb.", 0.95, "beekeeping"),
    ("hive_aurora", "OA vaporization schedule: every 3 weeks during broodless period.", 0.9, "treatment"),
    ("hive_jodi", "Jodi line produces excellent queens. Recommended for spring splits.", 0.85, "genetics"),
    ("hive_dennis", "Dennis line known for calm temperament and good winter hardiness.", 0.85, "genetics"),
]


# =============================================================================
# Loader
# =============================================================================

async def load_bee_data(bus: MemoryBus) -> dict:
    """Load all bee inspection data into Neo4j + Postgres.

    Args:
        bus: An initialized MemoryBus instance.

    Returns:
        Summary of what was loaded.
    """
    stats = {"entities": 0, "episodes": 0, "aliases": 0, "facts": 0, "relationships": 0}

    bus._require_neo4j()
    driver = bus._neo4j_driver
    db = bus.config.neo4j_database

    with driver.session(database=db) as session:
        # --- Load Entities ---
        for entity in HIVE_ENTITIES + LOCATION_ENTITIES + [APIARY_ENTITY]:
            props = {k: v for k, v in entity.items() if v is not None}
            now = datetime.utcnow().isoformat()
            props.setdefault("created_at", now)
            props.setdefault("updated_at", now)
            props.setdefault("importance", 0.7)
            props.setdefault("loom_origin", "bee")

            set_parts = []
            params = {}
            for k, v in props.items():
                pkey = f"p_{k}"
                set_parts.append(f"e.{k} = ${pkey}")
                params[pkey] = v

            session.run(
                f"MERGE (e:Entity {{id: $p_id}}) SET {', '.join(set_parts)}",
                **params,
            )
            stats["entities"] += 1

        # --- Load Episodes ---
        for ep in EPISODES:
            entity_ids = ep.pop("entity_ids", [])
            props = {k: v for k, v in ep.items() if v is not None}
            props.setdefault("loom_origin", "bee")

            set_parts = []
            params = {}
            for k, v in props.items():
                pkey = f"p_{k}"
                set_parts.append(f"ep.{k} = ${pkey}")
                params[pkey] = v

            session.run(
                f"MERGE (ep:Episode {{id: $p_id}}) SET {', '.join(set_parts)}",
                **params,
            )
            stats["episodes"] += 1

            # Connect episodes to entities via ABOUT
            for eid in entity_ids:
                session.run(
                    """
                    MATCH (ep:Episode {id: $ep_id}), (e:Entity {id: $e_id})
                    MERGE (ep)-[:ABOUT]->(e)
                    """,
                    ep_id=ep["id"],
                    e_id=eid,
                )
                stats["relationships"] += 1

        # --- Hive → Location relationships ---
        for hive in HIVE_ENTITIES:
            loc_id = f"loc_{hive['location']}"
            session.run(
                """
                MATCH (h:Entity {id: $hive_id}), (l:Entity {id: $loc_id})
                MERGE (h)-[:LOCATED_AT]->(l)
                """,
                hive_id=hive["id"],
                loc_id=loc_id,
            )
            stats["relationships"] += 1

        # --- All hives PART_OF apiary ---
        for hive in HIVE_ENTITIES:
            session.run(
                """
                MATCH (h:Entity {id: $hive_id}), (a:Entity {id: 'apiary_main'})
                MERGE (h)-[:PART_OF]->(a)
                """,
                hive_id=hive["id"],
            )
            stats["relationships"] += 1

        # --- Causal links ---
        for src, dst in CAUSAL_LINKS:
            session.run(
                """
                MATCH (a:Episode {id: $src}), (b:Episode {id: $dst})
                MERGE (a)-[:CAUSED]->(b)
                """,
                src=src, dst=dst,
            )
            stats["relationships"] += 1

        # --- Temporal links ---
        for src, dst in TEMPORAL_LINKS:
            session.run(
                """
                MATCH (a:Episode {id: $src}), (b:Episode {id: $dst})
                MERGE (a)-[:FOLLOWED_BY]->(b)
                """,
                src=src, dst=dst,
            )
            stats["relationships"] += 1

    # --- Load Entity Aliases into Postgres ---
    if bus._postgres:
        count = await bus._postgres.add_aliases_batch(ENTITY_ALIASES)
        stats["aliases"] = count

        # Load facts
        for neo4j_id, content, confidence, domain in FACTS:
            await bus._postgres.store_fact(
                neo4j_node_id=neo4j_id,
                content=content,
                confidence=confidence,
                domain=domain,
            )
            stats["facts"] += 1

    logger.info(
        "Bee data loaded: %d entities, %d episodes, %d aliases, %d facts, %d relationships",
        stats["entities"], stats["episodes"], stats["aliases"],
        stats["facts"], stats["relationships"],
    )
    return stats

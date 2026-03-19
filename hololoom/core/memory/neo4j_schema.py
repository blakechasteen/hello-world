"""
Neo4j Schema for Memory Bus — spec §10.1, §10.2.

Node labels, relationship types, constraints, and indexes
aligned with the architectural spec. Run ensure_schema() on startup.
"""

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Node Labels (spec §10.1)
# =============================================================================

NODE_ENTITY = "Entity"        # Named things with properties and relationships
NODE_EPISODE = "Episode"      # Timestamped events with causal chains
NODE_FACT = "Fact"            # Stable truths derived from episodes
NODE_PROCEDURE = "Procedure"  # How-to knowledge, skills, recipes
NODE_PLAN = "Plan"            # Intentions, goals, active tasks
NODE_MEMORY_QUERY = "MemoryQuery"  # Logged queries for audit
NODE_CONFLICT = "Conflict"    # Contradictions between facts
NODE_SUMMARY = "Summary"      # Compressed episode clusters

# =============================================================================
# Relationship Types (spec §10.2)
# =============================================================================

# Entity relationships
REL_RELATED_TO = "RELATED_TO"    # Generic (carries type, weight, since props)
REL_PART_OF = "PART_OF"
REL_LOCATED_AT = "LOCATED_AT"

# Temporal
REL_FOLLOWED_BY = "FOLLOWED_BY"
REL_CAUSED = "CAUSED"
REL_LED_TO = "LED_TO"

# Provenance
REL_DERIVED_FROM = "DERIVED_FROM"
REL_OBSERVED_IN = "OBSERVED_IN"  # episode → entity
REL_ABOUT = "ABOUT"              # episode → entity

# Memory meta
REL_CONTRADICTED_BY = "CONTRADICTED_BY"  # fact → fact (carries resolution, date)
REL_SUMMARIZES = "SUMMARIZES"            # summary → episodes
REL_QUERIED_FOR = "QUERIED_FOR"          # memory query → results used
REL_HAS_VERSION = "HAS_VERSION"          # entity → version snapshots

# =============================================================================
# Required Entity Properties
# =============================================================================

ENTITY_PROPS = ["id", "type", "name", "created_at", "updated_at", "importance", "loom_origin"]
EPISODE_PROPS = ["id", "type", "timestamp", "summary", "significance", "loom_origin"]
FACT_PROPS = ["id", "content", "confidence", "derived_from", "valid_from", "valid_to"]

# =============================================================================
# Schema Setup
# =============================================================================

# Cypher statements to create constraints and indexes.
# Uses CREATE ... IF NOT EXISTS for idempotent execution.
SCHEMA_STATEMENTS = [
    # Unique constraints on node IDs
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT episode_id_unique IF NOT EXISTS FOR (ep:Episode) REQUIRE ep.id IS UNIQUE",
    "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT procedure_id_unique IF NOT EXISTS FOR (p:Procedure) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT plan_id_unique IF NOT EXISTS FOR (pl:Plan) REQUIRE pl.id IS UNIQUE",
    "CREATE CONSTRAINT summary_id_unique IF NOT EXISTS FOR (s:Summary) REQUIRE s.id IS UNIQUE",

    # Performance indexes
    "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX episode_timestamp_idx IF NOT EXISTS FOR (ep:Episode) ON (ep.timestamp)",
    "CREATE INDEX episode_type_idx IF NOT EXISTS FOR (ep:Episode) ON (ep.type)",
    "CREATE INDEX fact_domain_idx IF NOT EXISTS FOR (f:Fact) ON (f.domain)",
    "CREATE INDEX plan_state_idx IF NOT EXISTS FOR (pl:Plan) ON (pl.state)",
    "CREATE INDEX plan_loom_idx IF NOT EXISTS FOR (pl:Plan) ON (pl.loom_origin)",
]

# Full-text index for entity name search (fuzzy matching).
# Separate because it uses different syntax and may already exist.
FULLTEXT_STATEMENTS = [
    """
    CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
    FOR (e:Entity) ON EACH [e.name]
    """,
]


def ensure_schema(driver, database: str = "neo4j") -> dict:
    """Create all constraints, indexes, and fulltext indexes.

    Idempotent — safe to call on every startup.

    Args:
        driver: Neo4j driver instance.
        database: Neo4j database name.

    Returns:
        Dict with counts of statements executed and any errors.
    """
    results = {"executed": 0, "errors": [], "skipped": 0}

    with driver.session(database=database) as session:
        # Core constraints and indexes
        for stmt in SCHEMA_STATEMENTS:
            try:
                session.run(stmt)
                results["executed"] += 1
            except Exception as e:
                err_msg = str(e)
                if "already exists" in err_msg.lower() or "equivalent" in err_msg.lower():
                    results["skipped"] += 1
                else:
                    results["errors"].append({"statement": stmt, "error": err_msg})
                    logger.warning("Schema statement failed: %s — %s", stmt[:60], err_msg)

        # Full-text indexes (may fail on Community Edition without APOC)
        for stmt in FULLTEXT_STATEMENTS:
            try:
                session.run(stmt)
                results["executed"] += 1
            except Exception as e:
                err_msg = str(e)
                if "already exists" in err_msg.lower():
                    results["skipped"] += 1
                else:
                    results["errors"].append({"statement": stmt.strip(), "error": err_msg})
                    logger.warning("Fulltext index skipped: %s", err_msg)

    logger.info(
        "Neo4j schema: %d executed, %d skipped, %d errors",
        results["executed"], results["skipped"], len(results["errors"])
    )
    return results

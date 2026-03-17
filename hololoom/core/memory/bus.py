"""
Memory Bus — Central coordinator for the database-backed memory system.

Spec §2.1, §4.2, §10.4. Routes queries through a 4-level cascade,
manages promotion to long-term storage, tracks audit log, and
enforces the 7 architectural invariants.

Invariants:
    1. Embeddings are an index, not a store
    2. Structured query first, semantic second
    3. Every memory injection has a token budget
    4. Plans never live only in context
    5. Context is always rebuildable from the database
    6. Scarcity triggers delegation, not compression
    7. Forgetting is a first-class capability
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .bus_config import (
    AuditAction,
    MemoryBusConfig,
    MemoryType,
    PressureTier,
    ResolutionPath,
)
from .neo4j_schema import (
    NODE_ENTITY,
    NODE_EPISODE,
    NODE_FACT,
    NODE_PLAN,
    REL_ABOUT,
    REL_CAUSED,
    REL_DERIVED_FROM,
    REL_FOLLOWED_BY,
    ensure_schema,
)

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation
try:
    import neo4j as neo4j_driver_module
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from .postgres_store import PostgresStore, PSYCOPG_AVAILABLE
except ImportError:
    PSYCOPG_AVAILABLE = False


# =============================================================================
# Data Types (spec §4.2)
# =============================================================================

@dataclass
class MemoryQuery:
    """A query against long-term memory (spec §4.2)."""
    intent: str
    entity_ids: Optional[List[str]] = None
    entity_type: Optional[str] = None
    memory_type: Optional[str] = None
    time_after: Optional[str] = None
    time_before: Optional[str] = None
    relationship: Optional[str] = None
    semantic_text: Optional[str] = None
    max_results: int = 5
    min_importance: float = 0.1
    properties_match: Optional[Dict[str, Any]] = None


@dataclass
class MemoryItem:
    """A memory item for storage (spec §2.2)."""
    content: str
    memory_type: str                         # episodic, entity, factual, etc.
    entity_ids: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None
    importance: float = 0.5
    scope: str = "local"
    loom_id: Optional[str] = None


@dataclass
class MemoryResult:
    """What comes back from memory (spec §4.2)."""
    items: List[Dict[str, Any]]
    query: MemoryQuery
    resolution_path: str
    token_estimate: int
    truncated: bool = False


# =============================================================================
# Memory Bus
# =============================================================================

class MemoryBus:
    """Central memory coordinator.

    Wraps Neo4j (graph) and Postgres (flat/fast) to provide:
    - 4-level query cascade (exact → structured → graph → semantic stub)
    - Entity resolution via alias table
    - Audit logging of every operation
    - Token-budgeted results
    - Provenance tracking
    """

    def __init__(self, config: Optional[MemoryBusConfig] = None):
        self.config = config or MemoryBusConfig()
        self._neo4j_driver = None
        self._postgres: Optional[PostgresStore] = None
        self._initialized = False

        # Runtime stats
        self._query_count = 0
        self._resolution_counts: Dict[str, int] = {
            ResolutionPath.EXACT.value: 0,
            ResolutionPath.STRUCTURED.value: 0,
            ResolutionPath.GRAPH.value: 0,
            ResolutionPath.SEMANTIC.value: 0,
        }
        self._total_tokens_used = 0

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def initialize(self) -> Dict[str, Any]:
        """Connect to backends and set up schemas.

        Returns:
            Status dict with backend connectivity info.
        """
        status = {"neo4j": "unavailable", "postgres": "unavailable", "schema": {}}

        # Neo4j
        if NEO4J_AVAILABLE:
            try:
                self._neo4j_driver = neo4j_driver_module.GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=self.config.neo4j_auth,
                )
                # Verify connectivity
                self._neo4j_driver.verify_connectivity()
                status["neo4j"] = "connected"

                # Ensure schema
                schema_result = ensure_schema(self._neo4j_driver, self.config.neo4j_database)
                status["schema"] = schema_result
                logger.info("Neo4j connected at %s", self.config.neo4j_uri)
            except Exception as e:
                status["neo4j"] = f"error: {e}"
                logger.error("Neo4j connection failed: %s", e)
        else:
            logger.warning("Neo4j driver not installed")

        # Postgres
        if PSYCOPG_AVAILABLE:
            try:
                self._postgres = PostgresStore(self.config.postgres_dsn)
                pg_result = await self._postgres.initialize()
                status["postgres"] = "connected"
                status["postgres_init"] = pg_result
                logger.info("Postgres connected")
            except Exception as e:
                status["postgres"] = f"error: {e}"
                logger.error("Postgres connection failed: %s", e)
        else:
            logger.warning("psycopg not installed — Postgres unavailable")

        self._initialized = True
        return status

    async def close(self):
        """Shut down all backend connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
            self._neo4j_driver = None
        if self._postgres:
            await self._postgres.close()
            self._postgres = None
        self._initialized = False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _require_neo4j(self):
        if not self._neo4j_driver:
            raise RuntimeError("Neo4j not available. Check connection and neo4j driver installation.")

    def _require_postgres(self):
        if not self._postgres:
            raise RuntimeError("Postgres not available. Check connection and psycopg installation.")

    # =========================================================================
    # Query — 4-Level Cascade (spec §4.2)
    # =========================================================================

    async def query(self, q: MemoryQuery) -> MemoryResult:
        """Query long-term memory through the resolution cascade.

        Level 1: EXACT — lookup by entity ID(s)
        Level 2: STRUCTURED — Cypher query with filters
        Level 3: GRAPH — Neo4j traversal within N hops
        Level 4: SEMANTIC — returns "not available" (Qdrant deferred)
        """
        self._require_neo4j()
        loop_id = str(uuid.uuid4())[:8]

        # Level 1: EXACT MATCH
        if q.entity_ids:
            items = self._query_exact(q.entity_ids)
            if items:
                result = self._build_result(items, q, ResolutionPath.EXACT)
                await self._audit_query(loop_id, q, result)
                return result

        # Level 2: STRUCTURED (Cypher with filters)
        if q.entity_type or q.memory_type or q.time_after or q.time_before:
            items = self._query_structured(q)
            if items:
                result = self._build_result(items, q, ResolutionPath.STRUCTURED)
                await self._audit_query(loop_id, q, result)
                return result

        # Level 3: GRAPH TRAVERSAL
        if q.intent:
            items = self._query_graph(q)
            if items:
                result = self._build_result(items, q, ResolutionPath.GRAPH)
                await self._audit_query(loop_id, q, result)
                return result

        # Level 4: SEMANTIC (deferred)
        if q.semantic_text:
            result = MemoryResult(
                items=[],
                query=q,
                resolution_path=ResolutionPath.SEMANTIC.value,
                token_estimate=0,
                truncated=False,
            )
            await self._audit_query(loop_id, q, result)
            return result

        # Nothing matched
        result = MemoryResult(
            items=[], query=q,
            resolution_path="none", token_estimate=0,
        )
        await self._audit_query(loop_id, q, result)
        return result

    def _query_exact(self, entity_ids: List[str]) -> List[Dict]:
        """Level 1: Exact match by entity ID."""
        items = []
        with self._neo4j_driver.session(database=self.config.neo4j_database) as session:
            for eid in entity_ids:
                result = session.run(
                    f"MATCH (e:{NODE_ENTITY} {{id: $id}}) RETURN e",
                    id=eid,
                )
                for record in result:
                    node = record["e"]
                    items.append(dict(node))
        return items

    def _query_structured(self, q: MemoryQuery) -> List[Dict]:
        """Level 2: Structured Cypher query with filters."""
        clauses = []
        params: Dict[str, Any] = {}

        # Determine which node label to query
        if q.memory_type == "episodic":
            label = NODE_EPISODE
        elif q.memory_type == "factual":
            label = NODE_FACT
        elif q.memory_type == "plan":
            label = NODE_PLAN
        else:
            label = NODE_ENTITY

        match_clause = f"MATCH (n:{label})"

        if q.entity_type:
            clauses.append("n.type = $entity_type")
            params["entity_type"] = q.entity_type

        if q.time_after:
            clauses.append("n.timestamp >= $time_after")
            params["time_after"] = q.time_after

        if q.time_before:
            clauses.append("n.timestamp <= $time_before")
            params["time_before"] = q.time_before

        where = ""
        if clauses:
            where = " WHERE " + " AND ".join(clauses)

        cypher = f"{match_clause}{where} RETURN n ORDER BY n.timestamp DESC LIMIT $limit"
        params["limit"] = q.max_results

        items = []
        with self._neo4j_driver.session(database=self.config.neo4j_database) as session:
            result = session.run(cypher, **params)
            for record in result:
                items.append(dict(record["n"]))
        return items

    def _query_graph(self, q: MemoryQuery) -> List[Dict]:
        """Level 3: Graph traversal — find entities related to the intent."""
        # Use entity name search via fulltext index if available,
        # otherwise fall back to CONTAINS match.
        items = []
        with self._neo4j_driver.session(database=self.config.neo4j_database) as session:
            # Try fulltext first
            try:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query)
                    YIELD node, score
                    RETURN node, score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    query=q.intent,
                    limit=q.max_results,
                )
                for record in result:
                    item = dict(record["node"])
                    item["_search_score"] = record["score"]
                    items.append(item)
            except Exception:
                # Fallback: CONTAINS match on name
                result = session.run(
                    f"""
                    MATCH (e:{NODE_ENTITY})
                    WHERE toLower(e.name) CONTAINS toLower($query)
                    RETURN e
                    LIMIT $limit
                    """,
                    query=q.intent,
                    limit=q.max_results,
                )
                for record in result:
                    items.append(dict(record["e"]))

            # If we found entities, also grab their connected episodes
            if items and len(items) < q.max_results:
                entity_ids = [item.get("id") for item in items if item.get("id")]
                if entity_ids:
                    ep_result = session.run(
                        f"""
                        MATCH (ep:{NODE_EPISODE})-[:{REL_ABOUT}]->(e:{NODE_ENTITY})
                        WHERE e.id IN $ids
                        RETURN ep
                        ORDER BY ep.timestamp DESC
                        LIMIT $limit
                        """,
                        ids=entity_ids,
                        limit=q.max_results - len(items),
                    )
                    for record in ep_result:
                        items.append(dict(record["ep"]))

        return items

    def _build_result(
        self, items: List[Dict], q: MemoryQuery, path: ResolutionPath
    ) -> MemoryResult:
        """Build a MemoryResult with token estimate."""
        # Invariant 3: Every memory injection has a token budget
        token_estimate = sum(self._estimate_tokens(item) for item in items)
        truncated = False

        if token_estimate > self.config.token_budget_absolute:
            # Truncate items to fit budget
            fitted_items = []
            running = 0
            for item in items:
                item_tokens = self._estimate_tokens(item)
                if running + item_tokens > self.config.token_budget_absolute:
                    truncated = True
                    break
                fitted_items.append(item)
                running += item_tokens
            items = fitted_items
            token_estimate = running

        # Update stats
        self._query_count += 1
        self._resolution_counts[path.value] = self._resolution_counts.get(path.value, 0) + 1
        self._total_tokens_used += token_estimate

        return MemoryResult(
            items=items,
            query=q,
            resolution_path=path.value,
            token_estimate=token_estimate,
            truncated=truncated,
        )

    @staticmethod
    def _estimate_tokens(item: Dict) -> int:
        """Estimate token count for a memory item (~4 chars per token)."""
        text = str(item)
        return max(1, len(text) // 4)

    # =========================================================================
    # Entity Resolution (spec §10.4)
    # =========================================================================

    async def resolve_entity(
        self,
        name_or_alias: str,
        loom_scope: Optional[str] = None,
        max_candidates: int = 3,
    ) -> List[Dict]:
        """Resolve a name or alias to entity IDs.

        Returns candidates ranked by confidence. Checks Postgres alias table
        first, then falls back to Neo4j name search.
        """
        candidates = []

        # 1. Check Postgres alias table
        if self._postgres:
            aliases = await self._postgres.resolve_alias(name_or_alias, loom_id=loom_scope)
            for alias in aliases[:max_candidates]:
                candidates.append({
                    "entity_id": alias["entity_id"],
                    "matched_alias": alias["alias"],
                    "confidence": alias["confidence"],
                    "source": "alias_table",
                })

        # 2. Fallback: search Neo4j by name
        if not candidates and self._neo4j_driver:
            with self._neo4j_driver.session(database=self.config.neo4j_database) as session:
                result = session.run(
                    f"""
                    MATCH (e:{NODE_ENTITY})
                    WHERE toLower(e.name) = toLower($name)
                    RETURN e.id AS id, e.name AS name, e.type AS type
                    LIMIT $limit
                    """,
                    name=name_or_alias,
                    limit=max_candidates,
                )
                for record in result:
                    candidates.append({
                        "entity_id": record["id"],
                        "matched_alias": record["name"],
                        "confidence": 0.8,  # direct name match, slightly less than alias
                        "source": "neo4j_name",
                    })

        # Audit
        if self._postgres:
            await self._postgres.log_audit(
                action="resolve_entity",
                details={"input": name_or_alias, "candidates": len(candidates)},
            )

        return candidates

    # =========================================================================
    # Store (spec §2.2)
    # =========================================================================

    async def store(self, item: MemoryItem) -> str:
        """Store a memory item in the appropriate backend.

        Returns the node ID.
        """
        self._require_neo4j()
        node_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat()

        props = {
            "id": node_id,
            "timestamp": now,
            "importance": item.importance,
            "loom_origin": item.loom_id,
            **(item.properties or {}),
        }

        # Determine node label from memory_type
        label_map = {
            "episodic": NODE_EPISODE,
            "entity": NODE_ENTITY,
            "factual": NODE_FACT,
            "plan": NODE_PLAN,
        }
        label = label_map.get(item.memory_type, NODE_ENTITY)

        # Add type-specific properties
        if label == NODE_EPISODE:
            props["summary"] = item.content
            props["type"] = item.properties.get("type", "observation") if item.properties else "observation"
            props["significance"] = item.importance
        elif label == NODE_ENTITY:
            props["name"] = item.content
            props["type"] = item.properties.get("type", "unknown") if item.properties else "unknown"
            props["created_at"] = now
            props["updated_at"] = now
        elif label == NODE_FACT:
            props["content"] = item.content
            props["confidence"] = item.importance
            props["domain"] = item.properties.get("domain") if item.properties else None

        # Write to Neo4j
        with self._neo4j_driver.session(database=self.config.neo4j_database) as session:
            # Build SET clause from properties
            set_parts = []
            clean_props = {}
            for k, v in props.items():
                if v is not None:
                    param_key = f"p_{k}"
                    set_parts.append(f"n.{k} = ${param_key}")
                    clean_props[param_key] = v

            set_clause = ", ".join(set_parts)
            session.run(
                f"CREATE (n:{label}) SET {set_clause}",
                **clean_props,
            )

        # Connect to existing entities if specified
        if item.entity_ids and label != NODE_ENTITY:
            with self._neo4j_driver.session(database=self.config.neo4j_database) as session:
                for eid in item.entity_ids:
                    session.run(
                        f"""
                        MATCH (n {{id: $node_id}}), (e:{NODE_ENTITY} {{id: $entity_id}})
                        CREATE (n)-[:{REL_ABOUT}]->(e)
                        """,
                        node_id=node_id,
                        entity_id=eid,
                    )

        # Mirror facts to Postgres for fast lookup
        if label == NODE_FACT and self._postgres:
            await self._postgres.store_fact(
                neo4j_node_id=node_id,
                content=item.content,
                confidence=item.importance,
                domain=item.properties.get("domain") if item.properties else None,
            )

        # Audit
        if self._postgres:
            await self._postgres.log_audit(
                action=AuditAction.PROMOTE.value,
                details={
                    "node_id": node_id,
                    "label": label,
                    "memory_type": item.memory_type,
                },
            )

        return node_id

    # =========================================================================
    # Explain (spec §10.4)
    # =========================================================================

    async def explain(self, node_id: str) -> Dict[str, Any]:
        """Return provenance chain for a memory item or relationship.

        Shows what episodes/facts it derived from, which reasoning loop
        created it, confidence level, and whether it's confirmed or inferred.
        """
        self._require_neo4j()

        explanation: Dict[str, Any] = {
            "node_id": node_id,
            "derived_from": [],
            "connected_entities": [],
            "audit_entries": [],
        }

        with self._neo4j_driver.session(database=self.config.neo4j_database) as session:
            # Get the node itself
            result = session.run(
                "MATCH (n {id: $id}) RETURN labels(n) AS labels, properties(n) AS props",
                id=node_id,
            )
            record = result.single()
            if record:
                explanation["labels"] = record["labels"]
                explanation["properties"] = dict(record["props"])

            # Get DERIVED_FROM chain
            result = session.run(
                f"""
                MATCH (n {{id: $id}})-[:{REL_DERIVED_FROM}*1..3]->(source)
                RETURN labels(source) AS labels, properties(source) AS props
                """,
                id=node_id,
            )
            for rec in result:
                explanation["derived_from"].append({
                    "labels": rec["labels"],
                    "properties": dict(rec["props"]),
                })

            # Get connected entities
            result = session.run(
                f"""
                MATCH (n {{id: $id}})-[r]->(e:{NODE_ENTITY})
                RETURN type(r) AS rel_type, e.id AS entity_id, e.name AS entity_name
                """,
                id=node_id,
            )
            for rec in result:
                explanation["connected_entities"].append({
                    "relationship": rec["rel_type"],
                    "entity_id": rec["entity_id"],
                    "entity_name": rec["entity_name"],
                })

        # Get audit entries
        if self._postgres:
            audits = await self._postgres.query_audit(limit=10)
            explanation["audit_entries"] = [
                a for a in audits
                if a.get("details") and node_id in str(a["details"])
            ]

        return explanation

    # =========================================================================
    # Entity Browsing (spec §10.4)
    # =========================================================================

    async def entities(
        self,
        query: Optional[str] = None,
        entity_type: Optional[str] = None,
        connected_to: Optional[str] = None,
        max_hops: int = 2,
        limit: int = 20,
    ) -> List[Dict]:
        """Browse the entity graph."""
        self._require_neo4j()

        with self._neo4j_driver.session(database=self.config.neo4j_database) as session:
            if connected_to:
                # Get entities within N hops of a given entity
                result = session.run(
                    f"""
                    MATCH (start:{NODE_ENTITY} {{id: $id}})-[*1..{max_hops}]-(e:{NODE_ENTITY})
                    WHERE e.id <> $id
                    RETURN DISTINCT e
                    LIMIT $limit
                    """,
                    id=connected_to,
                    limit=limit,
                )
            elif entity_type:
                result = session.run(
                    f"""
                    MATCH (e:{NODE_ENTITY} {{type: $type}})
                    RETURN e
                    LIMIT $limit
                    """,
                    type=entity_type,
                    limit=limit,
                )
            elif query:
                result = session.run(
                    f"""
                    MATCH (e:{NODE_ENTITY})
                    WHERE toLower(e.name) CONTAINS toLower($query)
                    RETURN e
                    LIMIT $limit
                    """,
                    query=query,
                    limit=limit,
                )
            else:
                result = session.run(
                    f"MATCH (e:{NODE_ENTITY}) RETURN e LIMIT $limit",
                    limit=limit,
                )

            return [dict(record["e"]) for record in result]

    # =========================================================================
    # Status & Pressure (spec §3)
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """Current memory system stats."""
        total_queries = self._query_count or 1
        semantic_pct = self._resolution_counts.get(ResolutionPath.SEMANTIC.value, 0) / total_queries

        return {
            "initialized": self._initialized,
            "neo4j": "connected" if self._neo4j_driver else "unavailable",
            "postgres": "connected" if self._postgres else "unavailable",
            "query_count": self._query_count,
            "resolution_distribution": dict(self._resolution_counts),
            "total_tokens_used": self._total_tokens_used,
            "semantic_fallback_pct": round(semantic_pct, 3),
            "semantic_target_met": semantic_pct <= self.config.max_semantic_fallback_pct,
        }

    def pressure(self) -> Dict[str, Any]:
        """Current pressure signals and active tier."""
        # In MVP, pressure is based on token usage relative to budget
        utilization = self._total_tokens_used / max(1, self.config.token_budget_absolute)
        tier = self.config.pressure_tier_for(min(1.0, utilization))

        return {
            "tier": tier.value,
            "tier_name": tier.name,
            "context_tokens_used": self._total_tokens_used,
            "token_budget": self.config.token_budget_absolute,
            "utilization_pct": round(utilization, 3),
        }

    # =========================================================================
    # Audit Helper
    # =========================================================================

    async def _audit_query(
        self, loop_id: str, q: MemoryQuery, result: MemoryResult
    ) -> None:
        """Log a query to the audit table."""
        if not self._postgres:
            return
        try:
            await self._postgres.log_audit(
                action=AuditAction.QUERY.value,
                loop_id=loop_id,
                resolution_path=result.resolution_path,
                tokens_used=result.token_estimate,
                pressure_tier=self.pressure()["tier"],
                details={
                    "intent": q.intent,
                    "result_count": len(result.items),
                    "truncated": result.truncated,
                },
            )
        except Exception as e:
            logger.warning("Audit log failed: %s", e)

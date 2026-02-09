"""
Neo4j adapter — 2026-02-09

Authority store for Entity and Episode nodes.
Uses the neo4j async driver.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
except ImportError:
    AsyncGraphDatabase = None  # type: ignore[assignment,misc]
    AsyncDriver = None  # type: ignore[assignment,misc]

from .models import (
    Entity,
    EntityType,
    Episode,
    EpisodeType,
    MemoryItem,
    ResolutionPath,
)

logger = logging.getLogger(__name__)


class Neo4jAdapter:
    """Async Neo4j adapter for entity/episode read/write + structured queries."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "hololoom",
    ):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Optional[Any] = None

    # ---- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        if AsyncGraphDatabase is None:
            raise ImportError("neo4j driver required: pip install neo4j")
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        # Ensure constraints/indexes
        async with self._driver.session() as session:
            await session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (ep:Episode) REQUIRE ep.id IS UNIQUE"
            )
            await session.run(
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            )
        logger.info("Neo4jAdapter connected and constraints ensured")

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aenter__(self) -> Neo4jAdapter:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    @property
    def driver(self) -> Any:
        if self._driver is None:
            raise RuntimeError("Neo4jAdapter not connected. Call connect() first.")
        return self._driver

    # ---- Entity CRUD ------------------------------------------------------

    async def create_entity(self, entity: Entity) -> Entity:
        async with self.driver.session() as session:
            await session.run(
                """
                MERGE (e:Entity {id: $id})
                SET e.type = $type,
                    e.name = $name,
                    e.created_at = $created_at,
                    e.updated_at = $updated_at,
                    e.importance = $importance,
                    e.loom_origin = $loom_origin,
                    e.properties = $properties
                """,
                id=entity.id,
                type=entity.type.value,
                name=entity.name,
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat(),
                importance=entity.importance,
                loom_origin=entity.loom_origin,
                properties=str(entity.properties),
            )
        return entity

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        async with self.driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity {id: $id}) RETURN e", id=entity_id
            )
            record = await result.single()
            if not record:
                return None
            return self._record_to_entity(record["e"])

    async def search_entities_by_name(
        self,
        name: str,
        *,
        loom_origin: Optional[str] = None,
        limit: int = 10,
    ) -> list[Entity]:
        name_lower = name.lower()
        if loom_origin:
            query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS $name
                  AND (e.loom_origin = $loom OR e.loom_origin IS NULL)
                RETURN e
                ORDER BY e.importance DESC
                LIMIT $limit
            """
            params = {"name": name_lower, "loom": loom_origin, "limit": limit}
        else:
            query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS $name
                RETURN e
                ORDER BY e.importance DESC
                LIMIT $limit
            """
            params = {"name": name_lower, "limit": limit}

        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._record_to_entity(r["e"]) for r in records]

    # ---- Episode CRUD -----------------------------------------------------

    async def create_episode(self, episode: Episode) -> Episode:
        async with self.driver.session() as session:
            # Create episode node
            await session.run(
                """
                MERGE (ep:Episode {id: $id})
                SET ep.type = $type,
                    ep.timestamp = $timestamp,
                    ep.summary = $summary,
                    ep.content = $content,
                    ep.significance = $significance,
                    ep.loom_origin = $loom_origin,
                    ep.properties = $properties
                """,
                id=episode.id,
                type=episode.type.value,
                timestamp=episode.timestamp.isoformat(),
                summary=episode.summary,
                content=episode.content,
                significance=episode.significance,
                loom_origin=episode.loom_origin,
                properties=str(episode.properties),
            )
            # Create ABOUT relationships
            for eid in episode.entity_ids:
                await session.run(
                    """
                    MATCH (ep:Episode {id: $ep_id})
                    MATCH (e:Entity {id: $e_id})
                    MERGE (ep)-[:ABOUT]->(e)
                    """,
                    ep_id=episode.id,
                    e_id=eid,
                )
        return episode

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (ep:Episode {id: $id})
                OPTIONAL MATCH (ep)-[:ABOUT]->(e:Entity)
                RETURN ep, collect(e.id) as entity_ids
                """,
                id=episode_id,
            )
            record = await result.single()
            if not record:
                return None
            return self._record_to_episode(record["ep"], record["entity_ids"])

    async def query_episodes_for_entity(
        self,
        entity_id: str,
        *,
        episode_types: Optional[list[EpisodeType]] = None,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 20,
    ) -> list[Episode]:
        clauses = ["(ep)-[:ABOUT]->(e:Entity {id: $entity_id})"]
        params: dict[str, Any] = {"entity_id": entity_id, "limit": limit}

        if episode_types:
            type_vals = [t.value for t in episode_types]
            clauses.append("ep.type IN $types")
            params["types"] = type_vals

        if time_start:
            clauses.append("ep.timestamp >= $time_start")
            params["time_start"] = time_start.isoformat()

        if time_end:
            clauses.append("ep.timestamp <= $time_end")
            params["time_end"] = time_end.isoformat()

        where = " AND ".join(clauses[1:]) if len(clauses) > 1 else ""
        where_clause = f"WHERE {where}" if where else ""

        query = f"""
            MATCH (ep:Episode), {clauses[0]}
            {where_clause}
            OPTIONAL MATCH (ep)-[:ABOUT]->(e2:Entity)
            RETURN ep, collect(DISTINCT e2.id) as entity_ids
            ORDER BY ep.timestamp DESC
            LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [
                self._record_to_episode(r["ep"], r["entity_ids"])
                for r in records
            ]

    async def query_recent_episodes(
        self,
        *,
        loom_origin: Optional[str] = None,
        limit: int = 20,
    ) -> list[Episode]:
        if loom_origin:
            query = """
                MATCH (ep:Episode)
                WHERE ep.loom_origin = $loom
                OPTIONAL MATCH (ep)-[:ABOUT]->(e:Entity)
                RETURN ep, collect(e.id) as entity_ids
                ORDER BY ep.timestamp DESC
                LIMIT $limit
            """
            params = {"loom": loom_origin, "limit": limit}
        else:
            query = """
                MATCH (ep:Episode)
                OPTIONAL MATCH (ep)-[:ABOUT]->(e:Entity)
                RETURN ep, collect(e.id) as entity_ids
                ORDER BY ep.timestamp DESC
                LIMIT $limit
            """
            params = {"limit": limit}

        async with self.driver.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [
                self._record_to_episode(r["ep"], r["entity_ids"])
                for r in records
            ]

    # ---- Graph traversal --------------------------------------------------

    async def get_related_entities(
        self,
        entity_id: str,
        *,
        max_hops: int = 2,
        limit: int = 10,
    ) -> list[Entity]:
        query = """
            MATCH (start:Entity {id: $id})
            MATCH path = (start)-[*1..$max_hops]-(related:Entity)
            WHERE related.id <> $id
            RETURN DISTINCT related
            ORDER BY related.importance DESC
            LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(
                query, id=entity_id, max_hops=max_hops, limit=limit
            )
            records = await result.data()
            return [self._record_to_entity(r["related"]) for r in records]

    # ---- converters -------------------------------------------------------

    def entity_to_item(self, entity: Entity) -> MemoryItem:
        return MemoryItem(
            id=entity.id,
            kind="entity",
            title=entity.name,
            summary=f"{entity.type.value}: {entity.name}",
            importance=entity.importance,
            resolution_path=ResolutionPath.EXACT,
            properties=entity.properties,
        )

    def episode_to_item(self, episode: Episode) -> MemoryItem:
        return MemoryItem(
            id=episode.id,
            kind="episode",
            title=episode.summary[:80] if episode.summary else episode.id,
            summary=episode.summary,
            content=episode.content,
            timestamp=episode.timestamp,
            importance=episode.significance,
            resolution_path=ResolutionPath.STRUCTURED,
            related_entity_ids=episode.entity_ids,
        )

    @staticmethod
    def _record_to_entity(node: dict[str, Any]) -> Entity:
        props = node.get("properties", "{}")
        if isinstance(props, str):
            # Stored as str repr of dict — parse safely
            try:
                import ast
                props = ast.literal_eval(props)
            except Exception:
                props = {}
        return Entity(
            id=node["id"],
            type=EntityType(node.get("type", "other")),
            name=node.get("name", ""),
            created_at=_parse_dt(node.get("created_at")),
            updated_at=_parse_dt(node.get("updated_at")),
            importance=float(node.get("importance", 0.5)),
            loom_origin=node.get("loom_origin"),
            properties=props if isinstance(props, dict) else {},
        )

    @staticmethod
    def _record_to_episode(
        node: dict[str, Any], entity_ids: list[str]
    ) -> Episode:
        props = node.get("properties", "{}")
        if isinstance(props, str):
            try:
                import ast
                props = ast.literal_eval(props)
            except Exception:
                props = {}
        return Episode(
            id=node["id"],
            type=EpisodeType(node.get("type", "other")),
            timestamp=_parse_dt(node.get("timestamp")),
            summary=node.get("summary", ""),
            content=node.get("content"),
            significance=float(node.get("significance", 0.5)),
            loom_origin=node.get("loom_origin"),
            entity_ids=[eid for eid in (entity_ids or []) if eid],
            properties=props if isinstance(props, dict) else {},
        )


def _parse_dt(val: Any) -> datetime:
    if val is None:
        return datetime.utcnow()
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (ValueError, TypeError):
        return datetime.utcnow()

"""
Neo4j Async Store - Graph-based Knowledge Storage
==================================================
Async Neo4j store for symbolic knowledge graph with:
- Connection pooling
- Cypher query execution
- Batch operations
- Graph algorithms integration

Philosophy:
The Neo4j store manages the SYMBOLIC side of knowledge - discrete entities,
relationships, and graph structure. It complements the vector store by providing
explicit semantic connections that embeddings alone cannot capture.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try to import neo4j, provide graceful fallback
try:
    from neo4j import AsyncGraphDatabase, AsyncSession
    NEO4J_AVAILABLE = True
except ImportError:
    logger.warning("neo4j package not installed. Neo4jStore will operate in mock mode.")
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None
    AsyncSession = None


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "hololoom"
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0


class Neo4jStore:
    """
    Async Neo4j knowledge graph store.

    Manages symbolic knowledge as a graph:
    - Nodes: Entities, concepts, threads
    - Edges: Relationships with typed semantics
    - Properties: Metadata, timestamps, weights

    Usage:
        config = Neo4jConfig(uri="bolt://localhost:7687")
        store = Neo4jStore(config)
        await store.connect()

        await store.add_entity("person:alice", {"name": "Alice", "role": "beekeeper"})
        await store.add_relationship("person:alice", "INSPECTS", "hive:jodi")

        neighbors = await store.get_neighbors("person:alice")
        await store.close()
    """

    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j store.

        Args:
            config: Neo4j configuration
        """
        self.config = config
        self.driver = None
        self.connected = False

        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j not available - using mock mode")
            self._mock_graph = {}  # Simple dict for testing

        logger.info(f"Neo4jStore initialized (uri={config.uri})")

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if not NEO4J_AVAILABLE:
            logger.info("Mock mode: connection simulated")
            self.connected = True
            return

        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout
            )

            # Verify connection
            await self.driver.verify_connectivity()
            self.connected = True

            logger.info("Connected to Neo4j successfully")

            # Create indexes
            await self._create_indexes()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def close(self) -> None:
        """Close connection to Neo4j."""
        if self.driver:
            await self.driver.close()
            self.connected = False
            logger.info("Neo4j connection closed")

    async def _create_indexes(self) -> None:
        """Create indexes for performance."""
        if not NEO4J_AVAILABLE:
            return

        indexes = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX thread_bucket IF NOT EXISTS FOR (t:Thread) ON (t.bucket)",
            "CREATE INDEX entity_timestamp IF NOT EXISTS FOR (e:Entity) ON (e.timestamp)"
        ]

        async with self.driver.session(database=self.config.database) as session:
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")

    async def add_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any],
        labels: Optional[List[str]] = None
    ) -> bool:
        """
        Add entity node to graph.

        Args:
            entity_id: Unique entity identifier
            properties: Entity properties
            labels: Optional additional labels

        Returns:
            True if successful
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        # Mock mode
        if not NEO4J_AVAILABLE:
            self._mock_graph[entity_id] = {
                'properties': properties,
                'labels': labels or ['Entity'],
                'edges': []
            }
            return True

        # Add timestamp if not present
        if 'timestamp' not in properties:
            properties['timestamp'] = datetime.now(timezone.utc).isoformat()

        properties['id'] = entity_id

        # Build labels
        label_str = ":".join(['Entity'] + (labels or []))

        query = f"""
        MERGE (e:{label_str} {{id: $id}})
        SET e += $properties
        RETURN e
        """

        try:
            async with self.driver.session(database=self.config.database) as session:
                result = await session.run(query, id=entity_id, properties=properties)
                await result.consume()
                return True
        except Exception as e:
            logger.error(f"Failed to add entity {entity_id}: {e}")
            return False

    async def add_relationship(
        self,
        source_id: str,
        rel_type: str,
        target_id: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add relationship between entities.

        Args:
            source_id: Source entity ID
            rel_type: Relationship type
            target_id: Target entity ID
            properties: Optional relationship properties

        Returns:
            True if successful
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        # Mock mode
        if not NEO4J_AVAILABLE:
            if source_id in self._mock_graph:
                self._mock_graph[source_id]['edges'].append({
                    'type': rel_type,
                    'target': target_id,
                    'properties': properties or {}
                })
            return True

        props = properties or {}
        if 'timestamp' not in props:
            props['timestamp'] = datetime.now(timezone.utc).isoformat()

        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        RETURN r
        """

        try:
            async with self.driver.session(database=self.config.database) as session:
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    properties=props
                )
                await result.consume()
                return True
        except Exception as e:
            logger.error(f"Failed to add relationship {source_id}-[{rel_type}]->{target_id}: {e}")
            return False

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity properties or None
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        # Mock mode
        if not NEO4J_AVAILABLE:
            if entity_id in self._mock_graph:
                return self._mock_graph[entity_id]['properties']
            return None

        query = """
        MATCH (e:Entity {id: $id})
        RETURN e
        """

        try:
            async with self.driver.session(database=self.config.database) as session:
                result = await session.run(query, id=entity_id)
                record = await result.single()

                if record:
                    return dict(record['e'])
                return None
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None

    async def get_neighbors(
        self,
        entity_id: str,
        rel_type: Optional[str] = None,
        max_hops: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring entities.

        Args:
            entity_id: Source entity ID
            rel_type: Optional relationship type filter
            max_hops: Maximum traversal hops

        Returns:
            List of neighbor entities with relationship info
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        # Mock mode
        if not NEO4J_AVAILABLE:
            if entity_id not in self._mock_graph:
                return []

            neighbors = []
            for edge in self._mock_graph[entity_id]['edges']:
                if rel_type is None or edge['type'] == rel_type:
                    target_id = edge['target']
                    if target_id in self._mock_graph:
                        neighbors.append({
                            'entity': self._mock_graph[target_id]['properties'],
                            'relationship': edge['type'],
                            'properties': edge['properties']
                        })
            return neighbors

        # Build relationship pattern
        rel_pattern = f"[r:{rel_type}]" if rel_type else "[r]"

        query = f"""
        MATCH (source:Entity {{id: $id}})-{rel_pattern*1..{max_hops}}-(neighbor)
        RETURN DISTINCT neighbor, type(r) as rel_type, properties(r) as rel_props
        LIMIT 100
        """

        try:
            async with self.driver.session(database=self.config.database) as session:
                result = await session.run(query, id=entity_id)

                neighbors = []
                async for record in result:
                    neighbors.append({
                        'entity': dict(record['neighbor']),
                        'relationship': record['rel_type'],
                        'properties': record['rel_props'] or {}
                    })

                return neighbors
        except Exception as e:
            logger.error(f"Failed to get neighbors for {entity_id}: {e}")
            return []

    async def search_entities(
        self,
        label: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search entities by label and/or properties.

        Args:
            label: Optional label filter
            properties: Optional property filters
            limit: Maximum results

        Returns:
            List of matching entities
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        # Mock mode
        if not NEO4J_AVAILABLE:
            results = []
            for eid, data in self._mock_graph.items():
                if label and label not in data['labels']:
                    continue
                if properties:
                    match = all(
                        data['properties'].get(k) == v
                        for k, v in properties.items()
                    )
                    if not match:
                        continue
                results.append(data['properties'])
                if len(results) >= limit:
                    break
            return results

        # Build query
        label_str = f":{label}" if label else ""
        where_clauses = []
        params = {'limit': limit}

        if properties:
            for key, value in properties.items():
                where_clauses.append(f"e.{key} = ${key}")
                params[key] = value

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
        MATCH (e:Entity{label_str})
        {where_str}
        RETURN e
        LIMIT $limit
        """

        try:
            async with self.driver.session(database=self.config.database) as session:
                result = await session.run(query, **params)

                entities = []
                async for record in result:
                    entities.append(dict(record['e']))

                return entities
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            return []

    async def batch_add_entities(
        self,
        entities: List[Tuple[str, Dict[str, Any], Optional[List[str]]]]
    ) -> int:
        """
        Batch add entities for efficiency.

        Args:
            entities: List of (entity_id, properties, labels) tuples

        Returns:
            Number of entities added
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        # Mock mode
        if not NEO4J_AVAILABLE:
            for entity_id, properties, labels in entities:
                self._mock_graph[entity_id] = {
                    'properties': properties,
                    'labels': labels or ['Entity'],
                    'edges': []
                }
            return len(entities)

        # Prepare batch data
        batch_data = []
        for entity_id, properties, labels in entities:
            props = properties.copy()
            props['id'] = entity_id
            if 'timestamp' not in props:
                props['timestamp'] = datetime.now(timezone.utc).isoformat()

            batch_data.append({
                'id': entity_id,
                'properties': props,
                'labels': labels or []
            })

        query = """
        UNWIND $batch AS item
        MERGE (e:Entity {id: item.id})
        SET e += item.properties
        WITH e, item.labels AS labels
        CALL {
            WITH e, labels
            UNWIND labels AS label
            CALL apoc.create.addLabels(e, [label]) YIELD node
            RETURN node
        }
        RETURN count(e) as count
        """

        # Fallback if APOC not available
        simple_query = """
        UNWIND $batch AS item
        MERGE (e:Entity {id: item.id})
        SET e += item.properties
        RETURN count(e) as count
        """

        try:
            async with self.driver.session(database=self.config.database) as session:
                try:
                    result = await session.run(query, batch=batch_data)
                except:
                    # Fallback without APOC
                    result = await session.run(simple_query, batch=batch_data)

                record = await result.single()
                return record['count'] if record else 0
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            return 0

    async def run_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute custom Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dicts
        """
        if not self.connected:
            raise RuntimeError("Not connected to Neo4j")

        if not NEO4J_AVAILABLE:
            logger.warning("Mock mode: Cypher query not executed")
            return []

        try:
            async with self.driver.session(database=self.config.database) as session:
                result = await session.run(query, **(parameters or {}))

                records = []
                async for record in result:
                    records.append(dict(record))

                return records
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return []


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*80)
        print("Neo4j Async Store Demo")
        print("="*80 + "\n")

        # Create store
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        store = Neo4jStore(config)

        try:
            # Connect
            print("1. Connecting to Neo4j...")
            await store.connect()
            print("   ✓ Connected\n")

            # Add entities
            print("2. Adding entities...")
            await store.add_entity(
                "person:alice",
                {"name": "Alice", "role": "beekeeper"},
                labels=["Person"]
            )
            await store.add_entity(
                "hive:jodi",
                {"name": "Jodi", "health": "strong"},
                labels=["Hive"]
            )
            print("   ✓ Entities added\n")

            # Add relationship
            print("3. Adding relationship...")
            await store.add_relationship(
                "person:alice",
                "INSPECTS",
                "hive:jodi",
                {"date": "2024-10-13", "notes": "Good brood pattern"}
            )
            print("   ✓ Relationship added\n")

            # Get entity
            print("4. Retrieving entity...")
            alice = await store.get_entity("person:alice")
            print(f"   Alice: {alice}\n")

            # Get neighbors
            print("5. Getting neighbors...")
            neighbors = await store.get_neighbors("person:alice", rel_type="INSPECTS")
            print(f"   Alice inspects: {len(neighbors)} hives")
            for n in neighbors:
                print(f"     - {n['entity'].get('name')}\n")

            # Search
            print("6. Searching entities...")
            hives = await store.search_entities(label="Hive")
            print(f"   Found {len(hives)} hives\n")

        finally:
            # Clean up
            await store.close()
            print("✓ Demo complete!")

    asyncio.run(demo())

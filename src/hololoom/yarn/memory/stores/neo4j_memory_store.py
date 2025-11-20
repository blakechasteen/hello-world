"""
Neo4j Memory Store
==================
MemoryStore implementation using Neo4j graph database.

Stores memories as nodes with rich relationship tracking:
- Temporal threads (time-based connections)
- Entity threads (hive, beekeeper, equipment references)
- Concept threads (topic-based clustering)

This is perfect for beekeeping data where relationships matter!
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

import sys
from pathlib import Path

# Import protocol types
protocol_path = Path(__file__).parent.parent / 'protocol.py'
import importlib.util
spec = importlib.util.spec_from_file_location("protocol", protocol_path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

Memory = protocol.Memory
MemoryQuery = protocol.MemoryQuery
RetrievalResult = protocol.RetrievalResult
Strategy = protocol.Strategy


class Neo4jMemoryStore:
    """
    Memory store backed by Neo4j graph database.

    Memory Model:
    - Each memory is a :Memory node
    - Memories connect via typed relationships:
      - NEXT_IN_TIME (temporal thread)
      - REFERENCES_ENTITY (entity thread)
      - SIMILAR_TO (semantic thread)
      - PART_OF_PATTERN (pattern thread)

    This enables powerful graph queries like:
    - "What happened before/after this memory?"
    - "What memories involve Hive Jodi?"
    - "Find memories similar to this pattern"

    Usage:
        store = Neo4jMemoryStore(
            uri="bolt://localhost:7688",
            username="neo4j",
            password="beekeeper123"
        )

        memory_id = await store.store(memory)
        result = await store.retrieve(query, strategy=Strategy.GRAPH)
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7688",
        username: str = "neo4j",
        password: str = "beekeeper123",
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j memory store.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j package required. Install: pip install neo4j")

        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

        # Connect
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

        # Verify connectivity
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j at {uri}: {e}")

        # Create indexes
        self._setup_schema()

    def _setup_schema(self):
        """Create indexes for performance."""
        with self.driver.session(database=self.database) as session:
            # Index on memory ID for fast lookups
            session.run(
                "CREATE INDEX memory_id_idx IF NOT EXISTS "
                "FOR (m:Memory) ON (m.id)"
            )

            # Index on timestamp for temporal queries
            session.run(
                "CREATE INDEX memory_timestamp_idx IF NOT EXISTS "
                "FOR (m:Memory) ON (m.timestamp)"
            )

            # Index on user_id for filtering
            session.run(
                "CREATE INDEX memory_user_idx IF NOT EXISTS "
                "FOR (m:Memory) ON (m.user_id)"
            )

    def _generate_id(self, text: str) -> str:
        """Generate deterministic ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    async def store(self, memory: Memory) -> str:
        """
        Store a memory in Neo4j.

        Creates:
        1. Memory node with all properties
        2. Temporal link to previous memory (NEXT_IN_TIME)
        3. Entity links if entities found in context
        4. Time thread connection

        Args:
            memory: Memory object to store

        Returns:
            memory_id
        """
        # Generate ID if not set
        if not memory.id:
            memory.id = self._generate_id(memory.text)

        with self.driver.session(database=self.database) as session:
            # Flatten context and metadata for Neo4j (no nested maps allowed)
            import json
            context_json = json.dumps(memory.context) if memory.context else "{}"
            metadata_json = json.dumps(memory.metadata) if memory.metadata else "{}"

            # Store memory node
            query = """
            CREATE (m:Memory {
                id: $id,
                text: $text,
                timestamp: datetime($timestamp),
                user_id: $user_id,
                context_json: $context_json,
                metadata_json: $metadata_json
            })
            RETURN m.id AS id
            """

            result = session.run(
                query,
                id=memory.id,
                text=memory.text,
                timestamp=memory.timestamp.isoformat(),
                user_id=memory.metadata.get('user_id', 'default'),
                context_json=context_json,
                metadata_json=metadata_json
            )

            memory_id = result.single()["id"]

            # Link to previous memory (temporal thread)
            session.run("""
                MATCH (m:Memory {id: $memory_id})
                MATCH (prev:Memory {user_id: $user_id})
                WHERE prev.id <> $memory_id
                  AND prev.timestamp < m.timestamp
                WITH m, prev
                ORDER BY prev.timestamp DESC
                LIMIT 1
                CREATE (prev)-[:NEXT_IN_TIME]->(m)
            """, memory_id=memory_id, user_id=memory.metadata.get('user_id', 'default'))

            # Link to entities if found in context
            if 'hive' in memory.context:
                session.run("""
                    MATCH (m:Memory {id: $memory_id})
                    MERGE (h:Hive {hiveId: $hive_id})
                    CREATE (m)-[:REFERENCES_ENTITY {entity_type: 'hive'}]->(h)
                """, memory_id=memory_id, hive_id=memory.context['hive'])

            return memory_id

    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve memories using graph-based strategies.

        Strategy implementations:
        - TEMPORAL: Follow NEXT_IN_TIME relationships
        - SEMANTIC: Text similarity (basic keyword matching)
        - GRAPH: Follow all relationships (entities, patterns)
        - PATTERN: Detect cycles and clusters
        - FUSED: Combine temporal + graph

        Args:
            query: Memory query
            strategy: Retrieval strategy

        Returns:
            RetrievalResult with scored memories
        """
        with self.driver.session(database=self.database) as session:
            if strategy == Strategy.TEMPORAL:
                return self._retrieve_temporal(session, query)
            elif strategy == Strategy.SEMANTIC:
                return self._retrieve_semantic(session, query)
            elif strategy == Strategy.GRAPH:
                return self._retrieve_graph(session, query)
            elif strategy == Strategy.PATTERN:
                return self._retrieve_pattern(session, query)
            else:  # FUSED
                return self._retrieve_fused(session, query)

    def _retrieve_temporal(self, session, query: MemoryQuery) -> RetrievalResult:
        """Retrieve by temporal proximity."""
        cypher = """
        MATCH (m:Memory)
        WHERE m.user_id = $user_id
        WITH m, duration.inSeconds(m.timestamp, datetime()).seconds AS age
        RETURN m, (1.0 / (1.0 + age / 3600.0)) AS score
        ORDER BY score DESC
        LIMIT $limit
        """

        result = session.run(
            cypher,
            user_id=query.user_id,
            limit=query.limit
        )

        memories = []
        scores = []

        for record in result:
            node = record['m']
            memories.append(self._node_to_memory(node))
            scores.append(record['score'])

        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used=Strategy.TEMPORAL.value,
            metadata={'total_memories': len(memories)}
        )

    def _retrieve_semantic(self, session, query: MemoryQuery) -> RetrievalResult:
        """Retrieve by semantic similarity (keyword matching)."""
        cypher = """
        MATCH (m:Memory)
        WHERE m.user_id = $user_id
          AND toLower(m.text) CONTAINS toLower($query_text)
        RETURN m, 1.0 AS score
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """

        result = session.run(
            cypher,
            user_id=query.user_id,
            query_text=query.text,
            limit=query.limit
        )

        memories = []
        scores = []

        for record in result:
            node = record['m']
            memories.append(self._node_to_memory(node))
            scores.append(record['score'])

        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used=Strategy.SEMANTIC.value,
            metadata={'total_memories': len(memories)}
        )

    def _retrieve_graph(self, session, query: MemoryQuery) -> RetrievalResult:
        """
        Retrieve by graph relationships.

        Follows all relationship types to find connected memories.
        Great for "What else is related to this hive?"
        """
        cypher = """
        MATCH (m:Memory)
        WHERE m.user_id = $user_id
          AND toLower(m.text) CONTAINS toLower($query_text)
        MATCH (m)-[r*1..2]-(related:Memory)
        WHERE related.user_id = $user_id
        WITH related, count(r) AS connection_strength
        RETURN related AS m, toFloat(connection_strength) / 10.0 AS score
        ORDER BY score DESC
        LIMIT $limit
        """

        result = session.run(
            cypher,
            user_id=query.user_id,
            query_text=query.text,
            limit=query.limit
        )

        memories = []
        scores = []

        for record in result:
            node = record['m']
            memories.append(self._node_to_memory(node))
            scores.append(min(record['score'], 1.0))

        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used=Strategy.GRAPH.value,
            metadata={'total_memories': len(memories)}
        )

    def _retrieve_pattern(self, session, query: MemoryQuery) -> RetrievalResult:
        """
        Retrieve by pattern detection.

        Finds temporal loops and clusters.
        """
        # For now, use temporal ordering
        return self._retrieve_temporal(session, query)

    def _retrieve_fused(self, session, query: MemoryQuery) -> RetrievalResult:
        """
        Fused retrieval: Combine temporal + graph.

        Score = 0.3 * temporal_score + 0.7 * graph_score
        """
        cypher = """
        MATCH (m:Memory)
        WHERE m.user_id = $user_id
          AND toLower(m.text) CONTAINS toLower($query_text)

        // Temporal score
        WITH m, duration.inSeconds(m.timestamp, datetime()).seconds AS age
        WITH m, (1.0 / (1.0 + age / 3600.0)) AS temporal_score

        // Graph score
        OPTIONAL MATCH (m)-[r]-(related:Memory)
        WHERE related.user_id = $user_id
        WITH m, temporal_score, count(r) AS connections
        WITH m, temporal_score, toFloat(connections) / 10.0 AS graph_score

        // Fused score
        WITH m, (0.3 * temporal_score + 0.7 * graph_score) AS score
        RETURN m, score
        ORDER BY score DESC
        LIMIT $limit
        """

        result = session.run(
            cypher,
            user_id=query.user_id,
            query_text=query.text,
            limit=query.limit
        )

        memories = []
        scores = []

        for record in result:
            node = record['m']
            memories.append(self._node_to_memory(node))
            scores.append(min(record['score'], 1.0))

        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used=Strategy.FUSED.value,
            metadata={'total_memories': len(memories)}
        )

    def _node_to_memory(self, node) -> Memory:
        """Convert Neo4j node to Memory object."""
        import json

        # Parse JSON fields
        context = json.loads(node.get('context_json', '{}'))
        metadata = json.loads(node.get('metadata_json', '{}'))

        return Memory(
            id=node['id'],
            text=node['text'],
            timestamp=datetime.fromisoformat(str(node['timestamp'])),
            context=context,
            metadata=metadata
        )

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory and its relationships."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (m:Memory {id: $memory_id})
                DETACH DELETE m
                RETURN count(m) AS deleted
            """, memory_id=memory_id)

            return result.single()['deleted'] > 0

    async def health_check(self) -> Dict[str, Any]:
        """Check health of Neo4j connection."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (m:Memory) RETURN count(m) AS count")
                count = result.single()['count']

                return {
                    'status': 'healthy',
                    'backend': 'neo4j',
                    'uri': self.uri,
                    'database': self.database,
                    'memory_count': count
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'neo4j',
                'error': str(e)
            }

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=== Neo4j Memory Store Demo ===\n")

        # Connect to beekeeping Neo4j
        store = Neo4jMemoryStore(
            uri="bolt://localhost:7688",
            username="neo4j",
            password="beekeeper123"
        )

        print("✓ Connected to Neo4j\n")

        # Store some memories
        print("Storing memories...")

        mem1 = Memory(
            id="",
            text="Inspected Hive Jodi - 15 frames of brood, very strong",
            timestamp=datetime.now(),
            context={'hive': 'hive-jodi-primary-001', 'place': 'apiary'},
            metadata={'user_id': 'blake'}
        )

        mem2 = Memory(
            id="",
            text="Applied thymol treatment to Hive Jodi - 35 units",
            timestamp=datetime.now(),
            context={'hive': 'hive-jodi-primary-001', 'place': 'apiary'},
            metadata={'user_id': 'blake'}
        )

        id1 = await store.store(mem1)
        id2 = await store.store(mem2)

        print(f"  Stored: {id1}")
        print(f"  Stored: {id2}\n")

        # Retrieve by different strategies
        query = MemoryQuery(
            text="Hive Jodi",
            user_id="blake",
            limit=5
        )

        print("Retrieving with GRAPH strategy...")
        result = await store.retrieve(query, Strategy.GRAPH)
        print(f"  Found {len(result.memories)} memories")
        for mem, score in zip(result.memories, result.scores):
            print(f"  [{score:.2f}] {mem.text[:50]}...")

        print("\n✓ Demo complete!")

        store.close()

    asyncio.run(demo())
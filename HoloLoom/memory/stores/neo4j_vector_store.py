"""
Neo4j Vector Store - Symbolic Vectors in Hyperspace
====================================================
Neo4j 5.x with vector index support for embeddings.

Combines:
- Graph relationships (symbolic threads)
- Vector similarity (semantic search)
- Both in ONE database

This is the "symbolic vector" hyperspace!

Features:
- Store embeddings as node properties
- Vector index for fast similarity search
- Graph queries for relationship traversal
- Hybrid: "Find similar nodes connected to X"

Requires:
- Neo4j 5.11+ (vector index support)
- pip install neo4j
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import json

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Optional embedder
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    EMBEDDINGS_AVAILABLE = False

# Import protocol
import sys
from pathlib import Path
protocol_path = Path(__file__).parent.parent / 'protocol.py'
import importlib.util
spec = importlib.util.spec_from_file_location("protocol", protocol_path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

Memory = protocol.Memory
MemoryQuery = protocol.MemoryQuery
RetrievalResult = protocol.RetrievalResult
Strategy = protocol.Strategy


class Neo4jVectorStore:
    """
    Neo4j store with vector embeddings.

    The best of both worlds:
    - Graph: "What's connected to Hive Jodi?"
    - Vector: "What's semantically similar to winter prep?"
    - Both: "Similar memories connected to this node?"

    Usage:
        store = Neo4jVectorStore(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )

        # Store with automatic embedding
        mem_id = await store.store(memory)

        # Retrieve with strategy
        results = await store.retrieve(query, strategy=Strategy.GRAPH)  # graph only
        results = await store.retrieve(query, strategy=Strategy.SEMANTIC)  # vector only
        results = await store.retrieve(query, strategy=Strategy.FUSED)  # both!
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_dim: int = 384,
        enable_embeddings: bool = True
    ):
        """
        Initialize Neo4j vector store.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
            embedding_model: SentenceTransformer model name
            vector_dim: Embedding dimension
            enable_embeddings: Generate embeddings automatically
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j package required. Install: pip install neo4j")

        self.uri = uri
        self.database = database
        self.vector_dim = vector_dim
        self.enable_embeddings = enable_embeddings

        # Connect
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

        # Verify connectivity
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j at {uri}: {e}")

        # Initialize embedder if enabled
        if enable_embeddings:
            if not EMBEDDINGS_AVAILABLE:
                raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
            self.embedder = SentenceTransformer(embedding_model)
        else:
            self.embedder = None

        # Setup schema
        self._setup_schema()

    def _setup_schema(self):
        """Create indexes and constraints."""
        with self.driver.session(database=self.database) as session:
            # Constraint on memory ID (uniqueness)
            session.run(
                "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS "
                "FOR (m:Memory) REQUIRE m.id IS UNIQUE"
            )

            # Index on timestamp
            session.run(
                "CREATE INDEX memory_timestamp_idx IF NOT EXISTS "
                "FOR (m:Memory) ON (m.timestamp)"
            )

            # Index on user_id
            session.run(
                "CREATE INDEX memory_user_idx IF NOT EXISTS "
                "FOR (m:Memory) ON (m.user_id)"
            )

            # Vector index (Neo4j 5.11+)
            # This is the hyperspace magic!
            if self.enable_embeddings:
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX memory_embedding_idx IF NOT EXISTS
                        FOR (m:Memory)
                        ON (m.embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.vector_dim},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                except Exception as e:
                    # Vector indexes require Neo4j 5.11+
                    print(f"Warning: Could not create vector index (requires Neo4j 5.11+): {e}")

    def _generate_id(self, text: str, timestamp: datetime) -> str:
        """Generate deterministic ID."""
        content = f"{text}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def store(self, memory: Memory) -> str:
        """
        Store memory with graph relationships and vector embedding.

        Creates:
        1. Memory node with text + embedding property
        2. Temporal link (NEXT_IN_TIME) to previous memory
        3. Entity links (REFERENCES) to entities in context

        Args:
            memory: Memory object

        Returns:
            memory_id
        """
        # Generate ID if needed
        mem_id = memory.id or self._generate_id(memory.text, memory.timestamp)

        # Generate embedding
        embedding = None
        if self.enable_embeddings and self.embedder:
            embedding = self.embedder.encode(memory.text).tolist()

        with self.driver.session(database=self.database) as session:
            # Serialize complex objects
            context_json = json.dumps(memory.context) if memory.context else "{}"
            metadata_json = json.dumps(memory.metadata) if memory.metadata else "{}"
            user_id = memory.metadata.get('user_id', 'default')

            # Create memory node
            if embedding:
                query = """
                CREATE (m:Memory {
                    id: $id,
                    text: $text,
                    timestamp: datetime($timestamp),
                    user_id: $user_id,
                    context_json: $context_json,
                    metadata_json: $metadata_json,
                    embedding: $embedding
                })
                RETURN m.id AS id
                """
                result = session.run(
                    query,
                    id=mem_id,
                    text=memory.text,
                    timestamp=memory.timestamp.isoformat(),
                    user_id=user_id,
                    context_json=context_json,
                    metadata_json=metadata_json,
                    embedding=embedding
                )
            else:
                # No embedding
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
                    id=mem_id,
                    text=memory.text,
                    timestamp=memory.timestamp.isoformat(),
                    user_id=user_id,
                    context_json=context_json,
                    metadata_json=metadata_json
                )

            memory_id = result.single()["id"]

            # Create temporal thread (NEXT_IN_TIME relationship)
            session.run("""
                MATCH (m:Memory {id: $memory_id})
                MATCH (prev:Memory {user_id: $user_id})
                WHERE prev.id <> $memory_id
                  AND prev.timestamp < m.timestamp
                WITH m, prev
                ORDER BY prev.timestamp DESC
                LIMIT 1
                CREATE (prev)-[:NEXT_IN_TIME]->(m)
            """, memory_id=memory_id, user_id=user_id)

            # Create entity links
            entities = memory.context.get('entities', [])
            for entity in entities:
                session.run("""
                    MATCH (m:Memory {id: $memory_id})
                    MERGE (e:Entity {name: $entity})
                    CREATE (m)-[:REFERENCES]->(e)
                """, memory_id=memory_id, entity=entity)

            return memory_id

    async def store_many(self, memories: List[Memory]) -> List[str]:
        """Store multiple memories."""
        ids = []
        for memory in memories:
            mem_id = await self.store(memory)
            ids.append(mem_id)
        return ids

    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (m:Memory {id: $id})
                RETURN m
            """, id=memory_id)

            record = result.single()
            if not record:
                return None

            node = record["m"]
            return self._node_to_memory(node)

    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve memories using selected strategy.

        Strategies:
        - TEMPORAL: Recent memories (graph timestamp ordering)
        - GRAPH: Connected memories (relationship traversal)
        - SEMANTIC: Similar memories (vector search)
        - FUSED: Combine graph + vector (best of both!)
        """
        if strategy == Strategy.TEMPORAL:
            return await self._retrieve_temporal(query)
        elif strategy == Strategy.GRAPH:
            return await self._retrieve_graph(query)
        elif strategy == Strategy.SEMANTIC:
            return await self._retrieve_semantic(query)
        elif strategy == Strategy.FUSED:
            return await self._retrieve_fused(query)
        else:
            return await self._retrieve_fused(query)

    async def _retrieve_temporal(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve recent memories."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (m:Memory {user_id: $user_id})
                RETURN m
                ORDER BY m.timestamp DESC
                LIMIT $limit
            """, user_id=query.user_id, limit=query.limit)

            memories = [self._node_to_memory(record["m"]) for record in result]
            scores = [1.0] * len(memories)  # Equal scores for temporal

            return RetrievalResult(
                memories=memories,
                scores=scores,
                strategy_used="temporal",
                metadata={'total': len(memories)}
            )

    async def _retrieve_graph(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve memories via graph traversal."""
        with self.driver.session(database=self.database) as session:
            # Find memories containing query text, then traverse graph
            result = session.run("""
                MATCH (m:Memory {user_id: $user_id})
                WHERE m.text CONTAINS $query_text
                WITH m
                MATCH path = (m)-[:NEXT_IN_TIME|REFERENCES*0..2]-(related:Memory)
                WHERE related.user_id = $user_id
                RETURN DISTINCT related AS memory, length(path) AS distance
                ORDER BY distance ASC
                LIMIT $limit
            """, user_id=query.user_id, query_text=query.text, limit=query.limit)

            memories = []
            scores = []
            for record in result:
                memories.append(self._node_to_memory(record["memory"]))
                # Score based on graph distance (closer = higher score)
                distance = record["distance"]
                score = 1.0 / (1.0 + distance)
                scores.append(score)

            return RetrievalResult(
                memories=memories,
                scores=scores,
                strategy_used="graph",
                metadata={'total': len(memories)}
            )

    async def _retrieve_semantic(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve memories via vector similarity."""
        if not self.enable_embeddings or not self.embedder:
            # Fall back to text search
            return await self._retrieve_temporal(query)

        # Generate query embedding
        query_embedding = self.embedder.encode(query.text).tolist()

        with self.driver.session(database=self.database) as session:
            # Vector similarity search (Neo4j 5.11+)
            try:
                result = session.run("""
                    MATCH (m:Memory {user_id: $user_id})
                    WHERE m.embedding IS NOT NULL
                    WITH m, vector.similarity.cosine(m.embedding, $query_embedding) AS score
                    WHERE score > 0.5
                    RETURN m, score
                    ORDER BY score DESC
                    LIMIT $limit
                """, user_id=query.user_id, query_embedding=query_embedding, limit=query.limit)

                memories = []
                scores = []
                for record in result:
                    memories.append(self._node_to_memory(record["m"]))
                    scores.append(float(record["score"]))

                return RetrievalResult(
                    memories=memories,
                    scores=scores,
                    strategy_used="semantic",
                    metadata={'total': len(memories)}
                )
            except Exception as e:
                # Vector search not available, fall back
                print(f"Vector search failed: {e}")
                return await self._retrieve_temporal(query)

    async def _retrieve_fused(self, query: MemoryQuery) -> RetrievalResult:
        """
        Retrieve via hybrid graph + vector fusion.

        This is where the magic happens!
        """
        # Get graph results
        graph_result = await self._retrieve_graph(query)

        # Get semantic results
        semantic_result = await self._retrieve_semantic(query)

        # Fuse scores
        fused = {}

        # Add graph results (weight = 0.6)
        for mem, score in zip(graph_result.memories, graph_result.scores):
            fused[mem.id] = {'memory': mem, 'score': 0.6 * score, 'sources': ['graph']}

        # Add semantic results (weight = 0.4)
        for mem, score in zip(semantic_result.memories, semantic_result.scores):
            if mem.id in fused:
                fused[mem.id]['score'] += 0.4 * score
                fused[mem.id]['sources'].append('semantic')
            else:
                fused[mem.id] = {'memory': mem, 'score': 0.4 * score, 'sources': ['semantic']}

        # Sort by fused score
        ranked = sorted(fused.values(), key=lambda x: x['score'], reverse=True)

        # Return top K
        top_k = ranked[:query.limit]
        memories = [item['memory'] for item in top_k]
        scores = [item['score'] for item in top_k]

        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used="fused",
            metadata={
                'total': len(memories),
                'graph_count': len(graph_result.memories),
                'semantic_count': len(semantic_result.memories)
            }
        )

    async def delete(self, memory_id: str) -> bool:
        """Delete memory and its relationships."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (m:Memory {id: $id})
                DETACH DELETE m
                RETURN count(m) AS deleted
            """, id=memory_id)

            return result.single()["deleted"] > 0

    async def health_check(self) -> Dict[str, Any]:
        """Check store health."""
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (m:Memory) RETURN count(m) AS total")
            total = result.single()["total"]

            return {
                'status': 'healthy',
                'database': self.database,
                'uri': self.uri,
                'total_memories': total,
                'embeddings_enabled': self.enable_embeddings
            }

    def _node_to_memory(self, node) -> Memory:
        """Convert Neo4j node to Memory object."""
        context = json.loads(node.get('context_json', '{}'))
        metadata = json.loads(node.get('metadata_json', '{}'))

        return Memory(
            id=node['id'],
            text=node['text'],
            timestamp=datetime.fromisoformat(str(node['timestamp'])),
            context=context,
            metadata=metadata
        )

    def close(self):
        """Close database connection."""
        self.driver.close()

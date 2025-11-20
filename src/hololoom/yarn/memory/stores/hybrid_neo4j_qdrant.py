#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Neo4j + Qdrant Memory Store
===================================
Best of both worlds: Symbolic graphs + Semantic vectors

Architecture:
- Neo4j: Graph relationships, temporal threads, entity links
- Qdrant: Fast vector similarity, multi-scale embeddings
- Fusion: Weighted combination of both strategies

Token Efficiency:
- BARE mode: Neo4j graph only (fast, symbolic)
- FAST mode: Qdrant vectors only (semantic)
- FUSED mode: Hybrid fusion (comprehensive)

This is the HYPERSPACE MEMORY STORE.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

# Direct imports to avoid package hell
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


@dataclass
class Memory:
    """Memory data structure."""
    id: str
    text: str
    timestamp: datetime
    context: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class MemoryQuery:
    """Query specification."""
    text: str
    user_id: str = "default"
    limit: int = 5
    filters: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """Retrieval results with provenance."""
    memories: List[Memory]
    scores: List[float]
    strategy_used: str
    metadata: Dict[str, Any]


class Strategy:
    """Retrieval strategies."""
    TEMPORAL = "temporal"
    GRAPH = "graph"
    SEMANTIC = "semantic"
    FUSED = "fused"


class HybridNeo4jQdrant:
    """
    Hybrid memory store combining Neo4j + Qdrant.

    Capabilities:
    - Store once, retrieve multiple ways
    - Graph queries (symbolic connections)
    - Vector search (semantic similarity)
    - Fusion (weighted combination)

    Usage:
        store = HybridNeo4jQdrant()

        # Store
        await store.store(memory)

        # Retrieve with different strategies
        graph_result = await store.retrieve(query, Strategy.GRAPH)
        vector_result = await store.retrieve(query, Strategy.SEMANTIC)
        fused_result = await store.retrieve(query, Strategy.FUSED)  # Best!
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "hololoom123",
        qdrant_url: str = "http://localhost:6333",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_dim: int = 384,
        collection_name: str = "hololoom_memories"
    ):
        """
        Initialize hybrid store.

        Args:
            neo4j_uri: Neo4j bolt URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            qdrant_url: Qdrant HTTP URL
            embedding_model: SentenceTransformer model
            vector_dim: Embedding dimension
            collection_name: Qdrant collection name
        """
        # Neo4j setup
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        self.neo4j_driver.verify_connectivity()

        # Qdrant setup
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

        # Embedder
        self.embedder = SentenceTransformer(embedding_model)
        self.vector_dim = vector_dim

        # Setup schemas
        self._setup_neo4j()
        self._setup_qdrant()

        print(f"✓ Hybrid store initialized")
        print(f"  - Neo4j: {neo4j_uri}")
        print(f"  - Qdrant: {qdrant_url}")

    def _setup_neo4j(self):
        """Create Neo4j constraints and indexes."""
        with self.neo4j_driver.session() as session:
            # Unique constraint on memory ID
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

    def _setup_qdrant(self):
        """Create Qdrant collection."""
        try:
            self.qdrant_client.get_collection(self.collection_name)
        except:
            # Create collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                )
            )

    def _generate_id(self, text: str, timestamp: datetime) -> str:
        """Generate deterministic ID."""
        content = f"{text}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def store(self, memory: Memory) -> str:
        """
        Store memory in both Neo4j and Qdrant.

        Process:
        1. Generate ID and embedding
        2. Store in Neo4j with graph relationships
        3. Store in Qdrant with vector
        4. Return memory ID

        Args:
            memory: Memory to store

        Returns:
            memory_id
        """
        # Generate ID
        mem_id = memory.id or self._generate_id(memory.text, memory.timestamp)

        # Generate embedding
        embedding = self.embedder.encode(memory.text).tolist()

        # Store in Neo4j
        with self.neo4j_driver.session() as session:
            context_json = json.dumps(memory.context)
            metadata_json = json.dumps(memory.metadata)
            user_id = memory.metadata.get('user_id', 'default')

            # Create memory node
            session.run("""
                CREATE (m:Memory {
                    id: $id,
                    text: $text,
                    timestamp: datetime($timestamp),
                    user_id: $user_id,
                    context_json: $context_json,
                    metadata_json: $metadata_json
                })
            """,
                id=mem_id,
                text=memory.text,
                timestamp=memory.timestamp.isoformat(),
                user_id=user_id,
                context_json=context_json,
                metadata_json=metadata_json
            )

            # Create temporal thread
            session.run("""
                MATCH (m:Memory {id: $memory_id})
                MATCH (prev:Memory {user_id: $user_id})
                WHERE prev.id <> $memory_id
                  AND prev.timestamp < m.timestamp
                WITH m, prev
                ORDER BY prev.timestamp DESC
                LIMIT 1
                CREATE (prev)-[:NEXT_IN_TIME]->(m)
            """, memory_id=mem_id, user_id=user_id)

            # Create entity links
            for entity in memory.context.get('entities', []):
                session.run("""
                    MATCH (m:Memory {id: $memory_id})
                    MERGE (e:Entity {name: $entity})
                    CREATE (m)-[:REFERENCES]->(e)
                """, memory_id=mem_id, entity=entity)

        # Store in Qdrant (convert ID to int for Qdrant)
        # Qdrant wants int or UUID, so convert hash to int
        qdrant_id = int(hashlib.md5(mem_id.encode()).hexdigest()[:15], 16)

        point = PointStruct(
            id=qdrant_id,
            vector=embedding,
            payload={
                'memory_id': mem_id,  # Store original ID in payload
                'text': memory.text,
                'timestamp': memory.timestamp.isoformat(),
                'user_id': memory.metadata.get('user_id', 'default'),
                **memory.context,
                **memory.metadata
            }
        )
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        return mem_id

    async def store_many(self, memories: List[Memory]) -> List[str]:
        """Store multiple memories."""
        ids = []
        for memory in memories:
            mem_id = await self.store(memory)
            ids.append(mem_id)
        return ids

    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: str = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve memories using selected strategy.

        Strategies:
        - TEMPORAL: Recent memories (graph timestamp)
        - GRAPH: Connected memories (relationships)
        - SEMANTIC: Similar memories (vectors)
        - FUSED: Hybrid (graph + vectors)

        Args:
            query: Query specification
            strategy: Retrieval strategy

        Returns:
            RetrievalResult with memories and scores
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
        """Retrieve recent memories from Neo4j."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m:Memory {user_id: $user_id})
                RETURN m
                ORDER BY m.timestamp DESC
                LIMIT $limit
            """, user_id=query.user_id, limit=query.limit)

            memories = []
            for record in result:
                memories.append(self._node_to_memory(record["m"]))

            scores = [1.0] * len(memories)

            return RetrievalResult(
                memories=memories,
                scores=scores,
                strategy_used="temporal",
                metadata={'source': 'neo4j', 'total': len(memories)}
            )

    async def _retrieve_graph(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve via graph traversal from Neo4j."""
        with self.neo4j_driver.session() as session:
            # Extract keywords from query
            keywords = [w.lower() for w in query.text.split() if len(w) > 3]

            # Find memories matching ANY keyword, then traverse graph
            # This is more flexible than exact text matching
            result = session.run("""
                MATCH (m:Memory {user_id: $user_id})
                WHERE any(keyword IN $keywords WHERE toLower(m.text) CONTAINS keyword)
                WITH m,
                     reduce(score = 0, keyword IN $keywords |
                            score + CASE WHEN toLower(m.text) CONTAINS keyword THEN 1 ELSE 0 END) AS match_count
                ORDER BY match_count DESC
                LIMIT 3
                WITH collect(m) AS seed_memories
                UNWIND seed_memories AS m
                MATCH path = (m)-[:NEXT_IN_TIME|REFERENCES*0..2]-(related:Memory)
                WHERE related.user_id = $user_id
                RETURN DISTINCT related AS memory, length(path) AS distance
                ORDER BY distance ASC
                LIMIT $limit
            """,
                user_id=query.user_id,
                keywords=keywords,
                limit=query.limit
            )

            memories = []
            scores = []
            for record in result:
                memories.append(self._node_to_memory(record["memory"]))
                # Score: closer in graph = higher score
                distance = record["distance"]
                score = 1.0 / (1.0 + distance)
                scores.append(score)

            return RetrievalResult(
                memories=memories,
                scores=scores,
                strategy_used="graph",
                metadata={'source': 'neo4j', 'total': len(memories)}
            )

    async def _retrieve_semantic(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve via vector similarity from Qdrant."""
        # Generate query embedding
        query_embedding = self.embedder.encode(query.text).tolist()

        # Search Qdrant
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=query.limit,
            query_filter={
                "must": [
                    {
                        "key": "user_id",
                        "match": {"value": query.user_id}
                    }
                ]
            }
        )

        memories = []
        scores = []
        for result in results.points:
            payload = result.payload
            memory = Memory(
                id=payload.get('memory_id', str(result.id)),  # Use original memory_id
                text=payload['text'],
                timestamp=datetime.fromisoformat(payload['timestamp']),
                context={k: v for k, v in payload.items() if k not in ['memory_id', 'text', 'timestamp', 'user_id']},
                metadata={'user_id': payload.get('user_id', 'default')}
            )
            memories.append(memory)
            scores.append(float(result.score))

        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used="semantic",
            metadata={'source': 'qdrant', 'total': len(memories)}
        )

    async def _retrieve_fused(self, query: MemoryQuery) -> RetrievalResult:
        """
        HYBRID FUSION: Combine graph + semantic.

        This is where the magic happens!

        Weights:
        - Graph (symbolic): 0.6
        - Semantic (vector): 0.4
        """
        # Get graph results
        graph_result = await self._retrieve_graph(query)

        # Get semantic results
        semantic_result = await self._retrieve_semantic(query)

        # Fuse scores
        fused = {}

        # Add graph results (weight = 0.6)
        for mem, score in zip(graph_result.memories, graph_result.scores):
            fused[mem.id] = {
                'memory': mem,
                'score': 0.6 * score,
                'sources': ['graph']
            }

        # Add semantic results (weight = 0.4)
        for mem, score in zip(semantic_result.memories, semantic_result.scores):
            if mem.id in fused:
                # Memory from both sources - combine scores
                fused[mem.id]['score'] += 0.4 * score
                fused[mem.id]['sources'].append('semantic')
            else:
                # Memory only from semantic
                fused[mem.id] = {
                    'memory': mem,
                    'score': 0.4 * score,
                    'sources': ['semantic']
                }

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
                'source': 'neo4j+qdrant',
                'total': len(memories),
                'graph_count': len(graph_result.memories),
                'semantic_count': len(semantic_result.memories)
            }
        )

    def _node_to_memory(self, node) -> Memory:
        """Convert Neo4j node to Memory."""
        context = json.loads(node.get('context_json', '{}'))
        metadata = json.loads(node.get('metadata_json', '{}'))

        return Memory(
            id=node['id'],
            text=node['text'],
            timestamp=datetime.fromisoformat(str(node['timestamp'])),
            context=context,
            metadata=metadata
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check health of both stores."""
        # Neo4j count
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (m:Memory) RETURN count(m) AS total")
            neo4j_count = result.single()["total"]

        # Qdrant count
        qdrant_info = self.qdrant_client.get_collection(self.collection_name)
        qdrant_count = qdrant_info.points_count

        return {
            'status': 'healthy',
            'neo4j': {
                'connected': True,
                'memories': neo4j_count
            },
            'qdrant': {
                'connected': True,
                'memories': qdrant_count
            }
        }

    def close(self):
        """Close connections."""
        self.neo4j_driver.close()

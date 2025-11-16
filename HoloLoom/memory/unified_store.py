"""
Unified Store - Hybrid Graph + Vector Knowledge
================================================
Combines Neo4j (symbolic graph) and Qdrant (semantic vectors) for powerful
hybrid retrieval that leverages both discrete relationships and continuous similarity.

Philosophy:
True intelligence requires BOTH symbolic reasoning (graphs) and fuzzy pattern
matching (vectors). The Unified Store bridges discrete and continuous knowledge,
enabling queries like "find similar concepts within 2 hops of this entity".
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .neo4j_store import Neo4jStore, Neo4jConfig
from .qdrant_store import QdrantStore, QdrantConfig

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid retrieval."""
    entity_id: str
    score: float  # Combined score from graph + vector
    graph_data: Dict[str, Any]  # From Neo4j
    vector_data: Dict[str, Any]  # From Qdrant
    retrieval_path: str  # How it was found (graph, vector, hybrid)


class UnifiedStore:
    """
    Unified hybrid store combining graph and vector search.

    Retrieval Strategies:
    1. **Graph-first**: Start from entity → traverse graph → re-rank by vector similarity
    2. **Vector-first**: Find similar embeddings → expand via graph connections
    3. **Hybrid**: Parallel search → merge and re-rank

    Usage:
        store = UnifiedStore(neo4j_config, qdrant_config)
        await store.connect()

        # Add entity with both graph and vector
        await store.add_knowledge(
            entity_id="doc:001",
            properties={"title": "Thompson Sampling"},
            embedding=vec,
            relationships=[("doc:001", "CITES", "paper:123")]
        )

        # Hybrid search
        results = await store.hybrid_search(
            query_embedding=query_vec,
            start_entities=["topic:RL"],
            max_hops=2,
            top_k=10
        )

        await store.close()
    """

    def __init__(
        self,
        neo4j_config: Neo4jConfig,
        qdrant_config: QdrantConfig,
        fusion_weights: Dict[str, float] = None
    ):
        """
        Initialize unified store.

        Args:
            neo4j_config: Neo4j configuration
            qdrant_config: Qdrant configuration
            fusion_weights: Weights for score fusion (graph, vector)
        """
        self.neo4j = Neo4jStore(neo4j_config)
        self.qdrant = QdrantStore(qdrant_config)

        # Default fusion weights
        self.fusion_weights = fusion_weights or {
            'graph': 0.4,   # Structural relevance
            'vector': 0.6   # Semantic similarity
        }

        logger.info("UnifiedStore initialized")

    async def connect(self) -> None:
        """Connect to both stores."""
        await asyncio.gather(
            self.neo4j.connect(),
            self.qdrant.connect()
        )
        logger.info("UnifiedStore connected")

    async def close(self) -> None:
        """Close both stores."""
        await asyncio.gather(
            self.neo4j.close(),
            self.qdrant.close()
        )
        logger.info("UnifiedStore closed")

    async def add_knowledge(
        self,
        entity_id: str,
        properties: Dict[str, Any],
        embedding: Optional[np.ndarray] = None,
        relationships: Optional[List[Tuple[str, str, str]]] = None,
        labels: Optional[List[str]] = None
    ) -> bool:
        """
        Add knowledge to both graph and vector stores.

        Args:
            entity_id: Unique entity identifier
            properties: Entity properties (stored in both)
            embedding: Optional embedding vector
            relationships: Optional list of (source, type, target) tuples
            labels: Optional Neo4j labels

        Returns:
            True if successful
        """
        # Add to graph
        graph_success = await self.neo4j.add_entity(
            entity_id,
            properties,
            labels=labels
        )

        # Add relationships if provided
        if relationships:
            for source, rel_type, target in relationships:
                await self.neo4j.add_relationship(source, rel_type, target)

        # Add to vector store if embedding provided
        vector_success = True
        if embedding is not None:
            # Combine properties with entity_id for payload
            payload = properties.copy()
            payload['entity_id'] = entity_id

            vector_success = await self.qdrant.upsert_vector(
                entity_id,
                embedding,
                payload=payload
            )

        success = graph_success and vector_success
        if success:
            logger.info(f"Added knowledge: {entity_id} (graph={graph_success}, vector={vector_success})")

        return success

    async def hybrid_search(
        self,
        query_embedding: np.ndarray,
        start_entities: Optional[List[str]] = None,
        max_hops: int = 2,
        top_k: int = 10,
        strategy: str = "hybrid"
    ) -> List[HybridResult]:
        """
        Hybrid search combining graph and vector.

        Args:
            query_embedding: Query vector
            start_entities: Optional starting entities for graph traversal
            max_hops: Maximum graph hops
            top_k: Number of results
            strategy: "graph_first", "vector_first", or "hybrid"

        Returns:
            List of hybrid results with combined scores
        """
        if strategy == "graph_first":
            return await self._graph_first_search(
                query_embedding, start_entities, max_hops, top_k
            )
        elif strategy == "vector_first":
            return await self._vector_first_search(
                query_embedding, max_hops, top_k
            )
        else:  # hybrid
            return await self._hybrid_search(
                query_embedding, start_entities, max_hops, top_k
            )

    async def _graph_first_search(
        self,
        query_embedding: np.ndarray,
        start_entities: List[str],
        max_hops: int,
        top_k: int
    ) -> List[HybridResult]:
        """
        Graph-first: Traverse graph, then re-rank by vector similarity.
        """
        if not start_entities:
            logger.warning("Graph-first search requires start_entities")
            return []

        # 1. Traverse graph from starting entities
        candidate_entities = set()
        graph_data_map = {}

        for start in start_entities:
            neighbors = await self.neo4j.get_neighbors(
                start,
                max_hops=max_hops
            )

            for neighbor in neighbors:
                entity_id = neighbor['entity'].get('id')
                if entity_id:
                    candidate_entities.add(entity_id)
                    graph_data_map[entity_id] = neighbor

        logger.info(f"Graph traversal found {len(candidate_entities)} candidates")

        # 2. Get embeddings and compute similarity
        results = []

        for entity_id in candidate_entities:
            # Get vector data
            vector_data = await self.qdrant.get_vector(entity_id)

            if vector_data and 'vector' in vector_data:
                # Compute similarity
                vec = np.array(vector_data['vector'])
                similarity = float(np.dot(query_embedding, vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vec) + 1e-10
                ))

                # Combine scores (graph distance implicitly captured by presence)
                graph_score = 1.0  # In graph neighborhood
                combined_score = (
                    self.fusion_weights['graph'] * graph_score +
                    self.fusion_weights['vector'] * similarity
                )

                results.append(HybridResult(
                    entity_id=entity_id,
                    score=combined_score,
                    graph_data=graph_data_map.get(entity_id, {}),
                    vector_data=vector_data,
                    retrieval_path="graph_first"
                ))

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    async def _vector_first_search(
        self,
        query_embedding: np.ndarray,
        max_hops: int,
        top_k: int
    ) -> List[HybridResult]:
        """
        Vector-first: Find similar vectors, then expand via graph.
        """
        # 1. Vector search
        vector_results = await self.qdrant.search(
            query_embedding,
            top_k=top_k * 2  # Get more candidates for graph expansion
        )

        logger.info(f"Vector search found {len(vector_results)} candidates")

        # 2. Expand via graph
        expanded_entities = set()
        entity_data = {}

        for vr in vector_results:
            entity_id = vr['id']
            expanded_entities.add(entity_id)

            # Store vector data
            entity_data[entity_id] = {
                'vector': vr,
                'graph': None
            }

            # Get graph neighbors
            neighbors = await self.neo4j.get_neighbors(entity_id, max_hops=1)
            for neighbor in neighbors:
                neighbor_id = neighbor['entity'].get('id')
                if neighbor_id:
                    expanded_entities.add(neighbor_id)
                    if neighbor_id not in entity_data:
                        entity_data[neighbor_id] = {
                            'vector': None,
                            'graph': neighbor
                        }

        logger.info(f"Graph expansion added {len(expanded_entities) - len(vector_results)} entities")

        # 3. Score all entities
        results = []

        for entity_id in expanded_entities:
            data = entity_data.get(entity_id, {})

            # Vector score
            if data.get('vector'):
                vector_score = data['vector']['score']
            else:
                # Need to fetch vector
                vec_data = await self.qdrant.get_vector(entity_id)
                if vec_data and 'vector' in vec_data:
                    vec = np.array(vec_data['vector'])
                    vector_score = float(np.dot(query_embedding, vec) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(vec) + 1e-10
                    ))
                    data['vector'] = {'score': vector_score, 'payload': vec_data['payload']}
                else:
                    vector_score = 0.0

            # Graph score (higher if directly retrieved, lower if expanded)
            graph_score = 1.0 if data.get('vector') else 0.5

            # Combined score
            combined_score = (
                self.fusion_weights['graph'] * graph_score +
                self.fusion_weights['vector'] * vector_score
            )

            results.append(HybridResult(
                entity_id=entity_id,
                score=combined_score,
                graph_data=data.get('graph', {}),
                vector_data=data.get('vector', {}),
                retrieval_path="vector_first"
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    async def _hybrid_search(
        self,
        query_embedding: np.ndarray,
        start_entities: Optional[List[str]],
        max_hops: int,
        top_k: int
    ) -> List[HybridResult]:
        """
        Hybrid: Parallel graph + vector search, then merge.
        """
        # Run both searches in parallel
        tasks = [
            self.qdrant.search(query_embedding, top_k=top_k * 2)
        ]

        # Add graph search if start entities provided
        if start_entities:
            async def graph_search():
                all_neighbors = []
                for start in start_entities:
                    neighbors = await self.neo4j.get_neighbors(start, max_hops=max_hops)
                    all_neighbors.extend(neighbors)
                return all_neighbors

            tasks.append(graph_search())

        search_results = await asyncio.gather(*tasks)

        vector_results = search_results[0]
        graph_results = search_results[1] if len(search_results) > 1 else []

        logger.info(f"Hybrid search: {len(vector_results)} vector + {len(graph_results)} graph results")

        # Merge results
        entity_scores = {}

        # Process vector results
        for vr in vector_results:
            entity_id = vr['id']
            entity_scores[entity_id] = {
                'vector_score': vr['score'],
                'graph_score': 0.0,
                'vector_data': vr,
                'graph_data': {}
            }

        # Process graph results
        for gr in graph_results:
            entity_id = gr['entity'].get('id')
            if entity_id:
                if entity_id not in entity_scores:
                    entity_scores[entity_id] = {
                        'vector_score': 0.0,
                        'graph_score': 0.0,
                        'vector_data': {},
                        'graph_data': gr
                    }
                else:
                    entity_scores[entity_id]['graph_data'] = gr

                entity_scores[entity_id]['graph_score'] = 1.0

        # Compute combined scores
        results = []

        for entity_id, scores in entity_scores.items():
            combined_score = (
                self.fusion_weights['vector'] * scores['vector_score'] +
                self.fusion_weights['graph'] * scores['graph_score']
            )

            results.append(HybridResult(
                entity_id=entity_id,
                score=combined_score,
                graph_data=scores['graph_data'],
                vector_data=scores['vector_data'],
                retrieval_path="hybrid"
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    async def get_entity_full(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete entity data from both stores.

        Args:
            entity_id: Entity identifier

        Returns:
            Combined entity data or None
        """
        # Parallel fetch
        graph_data, vector_data = await asyncio.gather(
            self.neo4j.get_entity(entity_id),
            self.qdrant.get_vector(entity_id),
            return_exceptions=True
        )

        if isinstance(graph_data, Exception):
            graph_data = None
        if isinstance(vector_data, Exception):
            vector_data = None

        if graph_data is None and vector_data is None:
            return None

        return {
            'entity_id': entity_id,
            'graph': graph_data or {},
            'vector': vector_data or {},
            'has_graph': graph_data is not None,
            'has_vector': vector_data is not None
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*80)
        print("Unified Store Demo")
        print("="*80 + "\n")

        # Create unified store
        neo4j_config = Neo4jConfig(uri="bolt://localhost:7687")
        qdrant_config = QdrantConfig(host="localhost", collection_name="demo_unified")

        store = UnifiedStore(neo4j_config, qdrant_config)

        try:
            # Connect
            print("1. Connecting to unified store...")
            await store.connect()
            print("   ✓ Connected to Neo4j and Qdrant\n")

            # Add knowledge
            print("2. Adding hybrid knowledge...")

            vec1 = np.random.randn(384).astype(np.float32)
            vec2 = np.random.randn(384).astype(np.float32)
            vec3 = np.random.randn(384).astype(np.float32)

            await store.add_knowledge(
                "doc:thompson",
                properties={"title": "Thompson Sampling", "year": 1933},
                embedding=vec1,
                labels=["Paper"]
            )

            await store.add_knowledge(
                "doc:bandit",
                properties={"title": "Multi-Armed Bandits", "year": 2010},
                embedding=vec2,
                relationships=[("doc:bandit", "CITES", "doc:thompson")],
                labels=["Paper"]
            )

            await store.add_knowledge(
                "topic:RL",
                properties={"name": "Reinforcement Learning"},
                embedding=vec3,
                relationships=[
                    ("topic:RL", "INCLUDES", "doc:thompson"),
                    ("topic:RL", "INCLUDES", "doc:bandit")
                ],
                labels=["Topic"]
            )

            print("   ✓ Knowledge added\n")

            # Hybrid search
            print("3. Performing hybrid search...")
            query = np.random.randn(384).astype(np.float32)

            results = await store.hybrid_search(
                query_embedding=query,
                start_entities=["topic:RL"],
                max_hops=2,
                top_k=5,
                strategy="hybrid"
            )

            print(f"   Found {len(results)} results:\n")
            for r in results:
                title = r.graph_data.get('entity', {}).get('title') or r.vector_data.get('payload', {}).get('title')
                print(f"     - {title} (score: {r.score:.3f}, path: {r.retrieval_path})")

        finally:
            # Clean up
            await store.close()
            print("\n✓ Demo complete!")

    asyncio.run(demo())

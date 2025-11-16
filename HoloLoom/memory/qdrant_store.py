"""
Qdrant Async Store - Vector-based Semantic Storage
===================================================
Async Qdrant store for dense vector embeddings with:
- Collection management
- Multi-scale embedding support
- Payload filtering
- Batch operations

Philosophy:
The Qdrant store manages the CONTINUOUS side of knowledge - dense embeddings
that capture semantic similarity. It complements the Neo4j graph by providing
fuzzy, context-aware retrieval that symbolic graphs cannot achieve alone.
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)

# Try to import qdrant, provide graceful fallback
try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
        SearchRequest, ScrollRequest
    )
    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("qdrant-client not installed. QdrantStore will operate in mock mode.")
    QDRANT_AVAILABLE = False
    AsyncQdrantClient = None


@dataclass
class QdrantConfig:
    """Configuration for Qdrant connection."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "hololoom_embeddings"
    vector_size: int = 384  # Default for matryoshka max scale
    distance: str = "Cosine"  # Cosine, Euclidean, Dot
    multi_scale: bool = True  # Support multiple embedding scales
    scales: List[int] = None  # [96, 192, 384]

    def __post_init__(self):
        if self.scales is None:
            self.scales = [96, 192, 384]


class QdrantStore:
    """
    Async Qdrant vector store.

    Manages dense embeddings for semantic similarity search:
    - Points: Vectors with payloads (metadata, text, timestamps)
    - Collections: Logical grouping with vector configuration
    - Filters: Payload-based pre-filtering for hybrid search

    Usage:
        config = QdrantConfig(host="localhost", vector_size=384)
        store = QdrantStore(config)
        await store.connect()

        await store.upsert_vector(
            vector_id="doc_001",
            vector=[0.1, 0.2, ...],  # 384-dim
            payload={"text": "Thompson Sampling...", "source": "paper"}
        )

        results = await store.search(query_vector, top_k=5)
        await store.close()
    """

    def __init__(self, config: QdrantConfig):
        """
        Initialize Qdrant store.

        Args:
            config: Qdrant configuration
        """
        self.config = config
        self.client = None
        self.connected = False

        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant not available - using mock mode")
            self._mock_vectors = {}  # Simple dict for testing

        logger.info(f"QdrantStore initialized (host={config.host}, collection={config.collection_name})")

    async def connect(self) -> None:
        """Establish connection to Qdrant."""
        if not QDRANT_AVAILABLE:
            logger.info("Mock mode: connection simulated")
            self.connected = True
            return

        try:
            self.client = AsyncQdrantClient(
                host=self.config.host,
                port=self.config.port,
                timeout=30.0
            )

            # Verify connection
            collections = await self.client.get_collections()
            self.connected = True

            logger.info("Connected to Qdrant successfully")

            # Create collection if needed
            await self._ensure_collection()

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def close(self) -> None:
        """Close connection to Qdrant."""
        if self.client:
            await self.client.close()
            self.connected = False
            logger.info("Qdrant connection closed")

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        if not QDRANT_AVAILABLE:
            return

        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.config.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.config.collection_name}")

                # Determine distance metric
                distance_map = {
                    "Cosine": Distance.COSINE,
                    "Euclidean": Distance.EUCLID,
                    "Dot": Distance.DOT
                }
                distance = distance_map.get(self.config.distance, Distance.COSINE)

                # Create with multi-vector support if needed
                if self.config.multi_scale:
                    # Named vectors for each scale
                    vectors_config = {
                        f"scale_{scale}": VectorParams(
                            size=scale,
                            distance=distance
                        )
                        for scale in self.config.scales
                    }
                else:
                    # Single vector
                    vectors_config = VectorParams(
                        size=self.config.vector_size,
                        distance=distance
                    )

                await self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=vectors_config
                )

                logger.info(f"Collection created: {self.config.collection_name}")
            else:
                logger.info(f"Collection exists: {self.config.collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    async def upsert_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        payload: Optional[Dict[str, Any]] = None,
        multi_scale_vectors: Optional[Dict[int, np.ndarray]] = None
    ) -> bool:
        """
        Upsert vector into collection.

        Args:
            vector_id: Unique vector identifier
            vector: Embedding vector (primary scale)
            payload: Optional metadata
            multi_scale_vectors: Optional dict of {scale: vector}

        Returns:
            True if successful
        """
        if not self.connected:
            raise RuntimeError("Not connected to Qdrant")

        # Mock mode
        if not QDRANT_AVAILABLE:
            self._mock_vectors[vector_id] = {
                'vector': vector.tolist() if isinstance(vector, np.ndarray) else vector,
                'payload': payload or {},
                'multi_scale': multi_scale_vectors or {}
            }
            return True

        # Ensure vector is list
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        # Prepare payload
        full_payload = payload or {}
        if 'timestamp' not in full_payload:
            full_payload['timestamp'] = datetime.now(timezone.utc).isoformat()
        full_payload['id'] = vector_id

        try:
            # Build point
            if self.config.multi_scale and multi_scale_vectors:
                # Named vectors for each scale
                vectors = {
                    f"scale_{scale}": vec.tolist() if isinstance(vec, np.ndarray) else vec
                    for scale, vec in multi_scale_vectors.items()
                }
                # Add primary vector
                max_scale = max(self.config.scales)
                vectors[f"scale_{max_scale}"] = vector
            else:
                # Single vector
                vectors = vector

            point = PointStruct(
                id=vector_id,
                vector=vectors,
                payload=full_payload
            )

            await self.client.upsert(
                collection_name=self.config.collection_name,
                points=[point]
            )

            return True

        except Exception as e:
            logger.error(f"Failed to upsert vector {vector_id}: {e}")
            return False

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        scale: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            top_k: Number of results
            filters: Optional payload filters
            scale: Optional specific scale to search (for multi-scale)

        Returns:
            List of search results with score and payload
        """
        if not self.connected:
            raise RuntimeError("Not connected to Qdrant")

        # Mock mode
        if not QDRANT_AVAILABLE:
            # Simple cosine similarity
            results = []
            query_norm = np.linalg.norm(query_vector)

            for vid, data in self._mock_vectors.items():
                vec = np.array(data['vector'])
                vec_norm = np.linalg.norm(vec)

                if query_norm > 0 and vec_norm > 0:
                    similarity = np.dot(query_vector, vec) / (query_norm * vec_norm)
                else:
                    similarity = 0.0

                results.append({
                    'id': vid,
                    'score': float(similarity),
                    'payload': data['payload']
                })

            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]

        # Ensure vector is list
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # Determine vector name for multi-scale
        if self.config.multi_scale:
            if scale is None:
                scale = max(self.config.scales)
            vector_name = f"scale_{scale}"
        else:
            vector_name = None

        # Build filters
        search_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            search_filter = Filter(must=conditions)

        try:
            search_result = await self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector if vector_name is None else (vector_name, query_vector),
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for hit in search_result:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload or {}
                })

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Get vector by ID.

        Args:
            vector_id: Vector identifier

        Returns:
            Vector data with payload or None
        """
        if not self.connected:
            raise RuntimeError("Not connected to Qdrant")

        # Mock mode
        if not QDRANT_AVAILABLE:
            return self._mock_vectors.get(vector_id)

        try:
            points = await self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=True
            )

            if points:
                point = points[0]
                return {
                    'id': point.id,
                    'vector': point.vector,
                    'payload': point.payload or {}
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get vector {vector_id}: {e}")
            return None

    async def delete_vector(self, vector_id: str) -> bool:
        """
        Delete vector by ID.

        Args:
            vector_id: Vector identifier

        Returns:
            True if successful
        """
        if not self.connected:
            raise RuntimeError("Not connected to Qdrant")

        # Mock mode
        if not QDRANT_AVAILABLE:
            if vector_id in self._mock_vectors:
                del self._mock_vectors[vector_id]
                return True
            return False

        try:
            await self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=[vector_id]
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False

    async def batch_upsert(
        self,
        vectors: List[Tuple[str, np.ndarray, Dict[str, Any]]]
    ) -> int:
        """
        Batch upsert vectors for efficiency.

        Args:
            vectors: List of (vector_id, vector, payload) tuples

        Returns:
            Number of vectors upserted
        """
        if not self.connected:
            raise RuntimeError("Not connected to Qdrant")

        # Mock mode
        if not QDRANT_AVAILABLE:
            for vector_id, vector, payload in vectors:
                self._mock_vectors[vector_id] = {
                    'vector': vector.tolist() if isinstance(vector, np.ndarray) else vector,
                    'payload': payload
                }
            return len(vectors)

        # Prepare points
        points = []
        for vector_id, vector, payload in vectors:
            # Ensure vector is list
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()

            # Add timestamp
            full_payload = payload.copy()
            if 'timestamp' not in full_payload:
                full_payload['timestamp'] = datetime.now(timezone.utc).isoformat()
            full_payload['id'] = vector_id

            point = PointStruct(
                id=vector_id,
                vector=vector,
                payload=full_payload
            )
            points.append(point)

        try:
            # Batch upsert
            await self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )

            return len(points)

        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
            return 0

    async def scroll_all(
        self,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scroll through all vectors (for export/backup).

        Args:
            limit: Maximum vectors to retrieve
            filters: Optional payload filters

        Returns:
            List of vector data
        """
        if not self.connected:
            raise RuntimeError("Not connected to Qdrant")

        # Mock mode
        if not QDRANT_AVAILABLE:
            return list(self._mock_vectors.values())[:limit]

        # Build filters
        scroll_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            scroll_filter = Filter(must=conditions)

        try:
            points, _ = await self.client.scroll(
                collection_name=self.config.collection_name,
                limit=limit,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for point in points:
                results.append({
                    'id': point.id,
                    'payload': point.payload or {}
                })

            return results

        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return []

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dict with collection info
        """
        if not self.connected:
            raise RuntimeError("Not connected to Qdrant")

        if not QDRANT_AVAILABLE:
            return {
                'vectors_count': len(self._mock_vectors),
                'status': 'mock'
            }

        try:
            info = await self.client.get_collection(
                collection_name=self.config.collection_name
            )

            return {
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status,
                'config': {
                    'distance': info.config.params.vectors.distance.value,
                    'size': info.config.params.vectors.size
                }
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*80)
        print("Qdrant Async Store Demo")
        print("="*80 + "\n")

        # Create store
        config = QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="demo_collection",
            vector_size=384,
            multi_scale=True,
            scales=[96, 192, 384]
        )
        store = QdrantStore(config)

        try:
            # Connect
            print("1. Connecting to Qdrant...")
            await store.connect()
            print("   ✓ Connected\n")

            # Create sample embeddings
            print("2. Upserting vectors...")
            vec1 = np.random.randn(384).astype(np.float32)
            vec2 = np.random.randn(384).astype(np.float32)

            await store.upsert_vector(
                "doc_001",
                vec1,
                payload={"text": "Thompson Sampling algorithm", "source": "paper"}
            )
            await store.upsert_vector(
                "doc_002",
                vec2,
                payload={"text": "Bayesian optimization methods", "source": "book"}
            )
            print("   ✓ Vectors upserted\n")

            # Search
            print("3. Searching similar vectors...")
            query = np.random.randn(384).astype(np.float32)
            results = await store.search(query, top_k=2)

            print(f"   Found {len(results)} results:")
            for r in results:
                print(f"     - {r['payload'].get('text')} (score: {r['score']:.3f})\n")

            # Collection info
            print("4. Collection info...")
            info = await store.get_collection_info()
            print(f"   Vectors: {info.get('vectors_count')}")
            print(f"   Status: {info.get('status')}\n")

        finally:
            # Clean up
            await store.close()
            print("✓ Demo complete!")

    asyncio.run(demo())

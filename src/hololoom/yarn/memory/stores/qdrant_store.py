"""
Qdrant Memory Store - Multi-Scale Vector Search
===============================================
Production-grade vector database with multi-scale embeddings.

Features:
- Multi-scale search (96d, 192d, 384d embeddings)
- Payload filtering (user_id, time, place, etc.)
- Efficient similarity search
- Horizontal scaling
"""

import logging
from typing import Dict, List, Optional, TYPE_CHECKING, Any
from datetime import datetime
import hashlib

from ..protocol import Memory, MemoryQuery, RetrievalResult, Strategy

# Optional qdrant import
if TYPE_CHECKING:
    from qdrant_client.models import Filter

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    _HAVE_QDRANT = True
except ImportError:
    QdrantClient = None
    Filter = Any  # Fallback for type hints
    _HAVE_QDRANT = False

# Optional sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    _HAVE_EMBEDDINGS = True
except ImportError:
    SentenceTransformer = None
    _HAVE_EMBEDDINGS = False


class QdrantMemoryStore:
    """
    Qdrant-backed vector store with multi-scale embeddings.
    
    Collections:
    - memories_96: Fast, low-precision (96 dimensions)
    - memories_192: Balanced (192 dimensions)
    - memories_384: High-precision (384 dimensions)
    
    Retrieval:
    - Search at multiple scales
    - Fuse results with weighted scores
    - Filter by user_id, time_range, context
    
    Requires:
    - pip install qdrant-client
    - pip install sentence-transformers
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_prefix: str = "memories",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        scales: List[int] = [96, 192, 384]
    ):
        if not _HAVE_QDRANT:
            raise RuntimeError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            )
        
        if not _HAVE_EMBEDDINGS:
            raise RuntimeError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
        
        # Initialize client
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(url=url)
        
        self.collection_prefix = collection_prefix
        self.scales = scales
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedder
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Create collections for each scale
        self._setup_collections()
        
        self.logger.info(f"Qdrant store initialized: {url} with scales {scales}")
    
    def _setup_collections(self):
        """Create collections for multi-scale vectors."""
        for scale in self.scales:
            collection_name = f"{self.collection_prefix}_{scale}"
            
            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
                self.logger.info(f"Collection {collection_name} already exists")
            except Exception:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=scale,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created collection {collection_name}")
    
    async def store(self, memory: Memory, user_id: str = "default") -> str:
        """
        Store memory with multi-scale vector embeddings.

        Stores a single memory across multiple embedding scales (96d, 192d, 384d)
        for flexible speed/accuracy tradeoffs during retrieval.

        Args:
            memory: Memory object with text and optional pre-computed embedding
            user_id: User identifier for filtering (stored in metadata)

        Returns:
            str: Memory ID (original string format)

        Raises:
            ValueError: If memory validation fails
            RuntimeError: If Qdrant storage fails

        Process:
            1. Validate memory and generate ID
            2. Get or generate 768d embedding vector
            3. Convert string ID to integer (Qdrant requirement)
            4. Store at each scale with truncated vectors
        """
        # ============================================================
        # Validation
        # ============================================================
        if not memory or not memory.text:
            raise ValueError("Memory must have non-empty text")

        # ============================================================
        # ID Generation
        # ============================================================
        mem_id = memory.id or self._generate_id(memory.text, memory.timestamp)
        qdrant_id = self._convert_to_qdrant_id(mem_id)

        # ============================================================
        # Embedding Extraction (with validation)
        # ============================================================
        try:
            full_embedding = self._get_or_generate_embedding(memory)
        except Exception as e:
            self.logger.error(f"✗ Embedding extraction failed: {e}")
            raise

        # ============================================================
        # Multi-Scale Storage
        # ============================================================
        scales_stored = 0
        for scale in self.scales:
            try:
                collection_name = f"{self.collection_prefix}_{scale}"
                vector = full_embedding[:scale]
                payload = self._build_point_payload(mem_id, memory, user_id)

                self.client.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(id=qdrant_id, vector=vector, payload=payload)]
                )
                scales_stored += 1

            except Exception as e:
                self.logger.warning(
                    f"⚠ Failed to store at scale {scale}d: {e}"
                )
                # Continue with other scales (partial success is okay)

        # ============================================================
        # Final Validation
        # ============================================================
        if scales_stored == 0:
            raise RuntimeError(f"Failed to store memory at any scale")

        self.logger.info(
            f"✓ Stored {mem_id[:8]}... at {scales_stored}/{len(self.scales)} scales"
        )
        return mem_id

    async def store_many(self, memories: List[Memory], user_id: str = "default") -> List[str]:
        """
        Store multiple memories in batch.

        Args:
            memories: List of Memory objects to store
            user_id: User identifier for all memories

        Returns:
            List[str]: List of memory IDs (successful stores)
        """
        memory_ids = []
        failures = 0

        for i, memory in enumerate(memories, 1):
            try:
                mem_id = await self.store(memory, user_id=user_id)
                memory_ids.append(mem_id)
            except Exception as e:
                failures += 1
                self.logger.warning(f"⚠ Batch store {i}/{len(memories)} failed: {e}")

        self.logger.info(
            f"✓ Batch complete: {len(memory_ids)}/{len(memories)} stored "
            f"({failures} failures)"
        )
        return memory_ids

    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        # Try to retrieve from the largest scale collection first
        largest_scale = max(self.scales)
        collection_name = f"{self.collection_prefix}_{largest_scale}"
        
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[memory_id],
                with_payload=True
            )
            
            if result and len(result) > 0:
                point = result[0]
                payload = point.payload
                
                # Parse timestamp
                timestamp = datetime.fromisoformat(payload['timestamp'])
                
                # Extract context (remove metadata fields)
                context = {}
                metadata = {}
                for key, value in payload.items():
                    if key in ['text', 'timestamp', 'user_id']:
                        continue
                    elif key in ['user_id']:
                        metadata[key] = value
                    else:
                        context[key] = value
                
                return Memory(
                    id=memory_id,
                    text=payload['text'],
                    timestamp=timestamp,
                    context=context,
                    metadata=metadata
                )
        except Exception as e:
            self.logger.warning(f"Failed to get memory {memory_id}: {e}")
        
        return None
    
    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve memories using multi-scale vector search.
        
        Strategies:
        - TEMPORAL: Filter by recent timestamp
        - SEMANTIC: Multi-scale similarity search
        - FUSED: Weighted fusion of scales (default)
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query.text).tolist()
        
        # Build filter
        filter_conditions = [
            FieldCondition(
                key='user_id',
                match=MatchValue(value=query.user_id)
            )
        ]
        
        # Add filters from query
        if query.filters:
            for key, value in query.filters.items():
                filter_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        if strategy == Strategy.TEMPORAL:
            # Use smallest scale (fastest) and filter by time
            results = self._search_single_scale(
                96, query_embedding[:96], query.limit, query_filter
            )
            return self._results_to_retrieval(results, 'temporal_96d')
        
        elif strategy == Strategy.SEMANTIC:
            # Use largest scale (most accurate)
            results = self._search_single_scale(
                384, query_embedding[:384], query.limit, query_filter
            )
            return self._results_to_retrieval(results, 'semantic_384d')
        
        else:  # FUSED
            # Multi-scale search and fusion
            return self._multi_scale_search(query_embedding, query.limit, query_filter)
    
    def _search_single_scale(
        self,
        scale: int,
        vector: List[float],
        limit: int,
        query_filter: Optional[Filter]
    ) -> List:
        """Search at single scale."""
        collection_name = f"{self.collection_prefix}_{scale}"
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter
        )
        
        return results
    
    def _multi_scale_search(
        self,
        full_embedding: List[float],
        limit: int,
        query_filter: Optional[Filter]
    ) -> RetrievalResult:
        """
        Search at multiple scales and fuse results.
        
        Fusion strategy:
        - 96d: 20% weight (fast, rough)
        - 192d: 30% weight (balanced)
        - 384d: 50% weight (precise)
        """
        weights = {96: 0.2, 192: 0.3, 384: 0.5}
        
        # Search at each scale
        all_results = {}
        for scale in self.scales:
            vector = full_embedding[:scale]
            results = self._search_single_scale(scale, vector, limit * 2, query_filter)
            
            # Weight scores
            for result in results:
                mem_id = result.id
                score = result.score * weights[scale]
                
                if mem_id not in all_results:
                    all_results[mem_id] = {
                        'result': result,
                        'score': score,
                        'scales': [scale]
                    }
                else:
                    all_results[mem_id]['score'] += score
                    all_results[mem_id]['scales'].append(scale)
        
        # Sort by fused score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:limit]
        
        # Convert to memories
        memories = []
        scores = []
        
        for item in sorted_results:
            result = item['result']
            mem = Memory(
                id=result.payload.get('memory_id', str(result.id)),  # Use original memory_id
                text=result.payload.get('text', ''),
                timestamp=self._parse_timestamp(result.payload.get('timestamp')),
                context={k: v for k, v in result.payload.items() if k in ['place', 'time', 'people', 'topics']},
                metadata={
                    'source': 'qdrant',
                    'scales_used': item['scales'],
                    **{k: v for k, v in result.payload.items() if k not in ['text', 'timestamp']}
                }
            )
            memories.append(mem)
            scores.append(item['score'])
        
        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used='multi_scale_fused',
            metadata={
                'backend': 'qdrant',
                'scales': self.scales,
                'weights': weights,
                'total_candidates': len(all_results)
            }
        )
    
    def _results_to_retrieval(self, results: List, strategy_name: str) -> RetrievalResult:
        """Convert Qdrant results to RetrievalResult."""
        memories = []
        scores = []
        
        for result in results:
            mem = Memory(
                id=str(result.id),
                text=result.payload.get('text', ''),
                timestamp=self._parse_timestamp(result.payload.get('timestamp')),
                context={k: v for k, v in result.payload.items() if k in ['place', 'time', 'people', 'topics']},
                metadata={
                    'source': 'qdrant',
                    **{k: v for k, v in result.payload.items() if k not in ['text', 'timestamp']}
                }
            )
            memories.append(mem)
            scores.append(result.score)
        
        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used=strategy_name,
            metadata={'backend': 'qdrant', 'result_count': len(memories)}
        )
    
    async def delete(self, memory_id: str) -> bool:
        """Delete memory from all scale collections."""
        try:
            for scale in self.scales:
                collection_name = f"{self.collection_prefix}_{scale}"
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=[memory_id]
                )
            
            self.logger.info(f"Deleted memory {memory_id} from all scales")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete {memory_id}: {e}")
            return False
    
    async def health_check(self) -> Dict:
        """Check Qdrant connection and collection status."""
        try:
            collection_stats = {}
            for scale in self.scales:
                collection_name = f"{self.collection_prefix}_{scale}"
                info = self.client.get_collection(collection_name)
                collection_stats[f"{scale}d"] = {
                    'points': info.points_count,
                    'vectors': info.vectors_count
                }
            
            return {
                'status': 'healthy',
                'backend': 'qdrant',
                'collections': collection_stats,
                'scales': self.scales,
                'features': ['multi_scale_search', 'vector_similarity', 'payload_filtering']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'qdrant',
                'error': str(e)
            }

    async def recall(self, query: MemoryQuery, limit: int = 10) -> RetrievalResult:
        """
        Recall memories (alias for retrieve with FUSED strategy).

        This method provides compatibility with the MemoryStore protocol.
        """
        query.limit = limit
        return await self.retrieve(query, strategy=Strategy.FUSED)

    # ============================================================
    # Helper Methods
    # ============================================================

    def _get_or_generate_embedding(self, memory: Memory) -> List[float]:
        """
        Extract embedding from Memory or generate if missing.

        Prefers pre-computed embeddings (e.g., from MatryoshkaEmbeddings)
        to avoid duplicate computation. Validates embedding dimensions.

        Args:
            memory: Memory object with optional embedding field

        Returns:
            List[float]: Embedding vector (validated dimensions)

        Raises:
            ValueError: If embedding dimensions are invalid
        """
        if hasattr(memory, 'embedding') and memory.embedding is not None:
            import numpy as np
            # Convert numpy array to list
            if isinstance(memory.embedding, np.ndarray):
                embedding = memory.embedding.tolist()
            else:
                embedding = memory.embedding

            # Validate embedding dimensions
            if not isinstance(embedding, list) or len(embedding) == 0:
                raise ValueError(f"Invalid embedding: expected non-empty list, got {type(embedding)}")

            # Ensure sufficient dimensions for all scales
            max_scale = max(self.scales)
            if len(embedding) < max_scale:
                self.logger.warning(
                    f"⚠ Embedding too small ({len(embedding)}d < {max_scale}d), "
                    f"padding with zeros"
                )
                embedding = embedding + [0.0] * (max_scale - len(embedding))

            self.logger.info(f"✓ Using provided embedding (dim={len(embedding)})")
            return embedding
        else:
            # Fallback: generate embedding
            if not memory.text or not memory.text.strip():
                raise ValueError("Cannot generate embedding: memory text is empty")

            embedding = self.embedder.encode(memory.text).tolist()
            self.logger.info(f"⚠ Generated embedding (dim={len(embedding)})")
            return embedding

    def _convert_to_qdrant_id(self, string_id: str) -> int:
        """
        Convert string ID to integer for Qdrant.

        Qdrant requires integer or UUID IDs. We use MD5 hash truncated
        to 15 hex chars (60 bits) to fit in Python int.

        Args:
            string_id: Original memory ID (string format)

        Returns:
            int: Qdrant-compatible integer ID
        """
        return int(hashlib.md5(string_id.encode()).hexdigest()[:15], 16)

    def _build_point_payload(
        self,
        mem_id: str,
        memory: Memory,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Build Qdrant point payload with metadata.

        Args:
            mem_id: Original memory ID (stored for retrieval)
            memory: Memory object with text, context, metadata
            user_id: User identifier for filtering

        Returns:
            Dict: Payload for Qdrant point
        """
        return {
            'memory_id': mem_id,  # Original string ID
            'text': memory.text,
            'timestamp': memory.timestamp.isoformat(),
            'user_id': memory.metadata.get('user_id', user_id),
            **memory.context,
            **memory.metadata
        }

    def _generate_id(self, text: str, timestamp: datetime) -> str:
        """Generate deterministic ID from text and timestamp."""
        content = f"{text}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp string."""
        if not timestamp_str:
            return datetime.now()
        
        try:
            return datetime.fromisoformat(timestamp_str)
        except Exception:
            return datetime.now()

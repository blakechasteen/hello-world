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
    
    async def store(self, memory: Memory) -> str:
        """
        Store memory with multi-scale embeddings.
        
        Process:
        1. Generate full embedding (384d)
        2. Truncate to each scale (96d, 192d, 384d)
        3. Store in each collection with same ID
        """
        # Generate ID
        mem_id = memory.id or self._generate_id(memory.text, memory.timestamp)
        
        # Generate embedding
        full_embedding = self.embedder.encode(memory.text).tolist()
        
        # Store in each scale
        for scale in self.scales:
            collection_name = f"{self.collection_prefix}_{scale}"
            
            # Truncate embedding to scale
            vector = full_embedding[:scale]
            
            # Prepare payload
            payload = {
                'text': memory.text,
                'timestamp': memory.timestamp.isoformat(),
                'user_id': memory.metadata.get('user_id', 'default'),
                **memory.context,
                **memory.metadata
            }
            
            # Upsert point
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=mem_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
        
        self.logger.info(f"Stored memory {mem_id} at {len(self.scales)} scales")
        return mem_id

    async def store_many(self, memories: List[Memory]) -> List[str]:
        """Store multiple memories (batch operation)."""
        memory_ids = []
        for memory in memories:
            memory_id = await self.store(memory)
            memory_ids.append(memory_id)
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
                id=str(result.id),
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

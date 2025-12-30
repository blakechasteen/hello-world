"""
Qdrant Memory Store - Production-Grade Vector Search with Connection Pooling
============================================================================
Production-grade vector database with multi-scale embeddings and connection pooling.

Features:
- Connection pooling with singleton pattern for efficient resource usage
- Multi-scale search (96d, 192d, 384d embeddings)
- Payload filtering (user_id, time, place, etc.)
- Efficient similarity search with gRPC preference
- Automatic retry logic with exponential backoff
- Health checks and connection monitoring
- Horizontal scaling
"""

import logging
import os
import time
from threading import Lock
from typing import Dict, List, Optional, TYPE_CHECKING, Any
from datetime import datetime
import hashlib

from HoloLoom.utils.security import sanitize_uri
from ..protocol import Memory, MemoryQuery, RetrievalResult, Strategy

# Optional qdrant import
if TYPE_CHECKING:
    from qdrant_client.models import Filter

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
    _HAVE_QDRANT = True
except ImportError:
    QdrantClient = None
    Filter = Any  # Fallback for type hints
    ResponseHandlingException = Exception  # Fallback
    UnexpectedResponse = Exception  # Fallback
    _HAVE_QDRANT = False

# Optional sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    _HAVE_EMBEDDINGS = True
except ImportError:
    SentenceTransformer = None
    _HAVE_EMBEDDINGS = False

logger = logging.getLogger(__name__)


class QdrantMemoryStore:
    """
    Qdrant-backed vector store with connection pooling and multi-scale embeddings.

    Connection Pooling:
    - Singleton client pattern for connection reuse
    - Configurable timeout and retry settings
    - Health checks and monitoring
    - Graceful degradation on connection issues

    Collections:
    - memories_96: Fast, low-precision (96 dimensions)
    - memories_192: Balanced (192 dimensions)
    - memories_384: High-precision (384 dimensions)

    Retrieval:
    - Search at multiple scales
    - Fuse results with weighted scores
    - Filter by user_id, time_range, context

    Environment Variables:
    - QDRANT_HOST: Host URL (default: localhost)
    - QDRANT_PORT: Port number (default: 6333)
    - QDRANT_TIMEOUT: Request timeout in seconds (default: 60)
    - QDRANT_PREFER_GRPC: Use gRPC if available (default: true)
    - QDRANT_API_KEY: API key for authentication (optional)

    Requires:
    - pip install qdrant-client
    - pip install sentence-transformers
    """

    # Class-level client singleton with thread-safe lock
    _client_instance: Optional[QdrantClient] = None
    _embedder_instance: Optional[SentenceTransformer] = None
    _client_lock = Lock()
    _connection_metrics = {
        'total_requests': 0,
        'failed_requests': 0,
        'retry_count': 0,
        'last_health_check': None,
        'health_status': 'unknown'
    }

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_prefix: str = "memories",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        scales: List[int] = [96, 192, 384],
        timeout: Optional[float] = None,
        prefer_grpc: Optional[bool] = None,
        enable_metrics: bool = True
    ):
        if not _HAVE_QDRANT:
            raise RuntimeError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            )

        if not _HAVE_EMBEDDINGS:
            raise RuntimeError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

        # Configuration from environment or parameters
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.timeout = timeout or float(os.getenv("QDRANT_TIMEOUT", "60"))
        self.prefer_grpc = prefer_grpc if prefer_grpc is not None else (
            os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
        )
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.enable_metrics = enable_metrics

        # Build URL if not provided
        # Note: QdrantClient always uses http:// URLs even with prefer_grpc=True
        # The prefer_grpc flag tells the client to use gRPC protocol, not the URL scheme
        if url:
            self.url = url
        else:
            # Always use http:// scheme - gRPC preference is handled by the client
            self.url = f"http://{self.host}:{self.port}"

        self.collection_prefix = collection_prefix
        self.scales = scales

        # Instance logger (references module-level logger)
        self.logger = logger

        # Initialize or reuse client singleton
        self._initialize_client()

        # Initialize or reuse embedder singleton
        self._initialize_embedder(embedding_model)

        # Create collections for each scale
        self._setup_collections()

        logger.info(f"Qdrant store initialized: {sanitize_uri(self.url)} with scales {scales}")

    def _initialize_client(self) -> None:
        """Initialize client singleton with connection pooling."""
        with QdrantMemoryStore._client_lock:
            # Reuse existing healthy client
            if QdrantMemoryStore._client_instance:
                try:
                    # Simple health check
                    QdrantMemoryStore._client_instance.get_collections()
                    self.client = QdrantMemoryStore._client_instance
                    logger.info("Reusing existing Qdrant client connection")
                    return
                except Exception as e:
                    logger.warning(f"Existing Qdrant client unhealthy, recreating: {e}")
                    QdrantMemoryStore._client_instance = None

            # Create new client with retries
            max_retries = 3
            retry_delay = 1.0

            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Connecting to Qdrant at {self.url} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )

                    # Create client with optimal settings
                    client_params = {
                        'timeout': self.timeout,
                        'prefer_grpc': self.prefer_grpc,
                    }

                    if self.api_key:
                        client_params['api_key'] = self.api_key

                    if self.url:
                        client_params['url'] = self.url
                    else:
                        client_params['host'] = self.host
                        client_params['port'] = self.port

                    self.client = QdrantClient(**client_params)

                    # Verify connectivity
                    self.client.get_collections()

                    QdrantMemoryStore._client_instance = self.client

                    if self.enable_metrics:
                        QdrantMemoryStore._connection_metrics['health_status'] = 'healthy'
                        QdrantMemoryStore._connection_metrics['last_health_check'] = time.time()

                    logger.info(f"Successfully connected to Qdrant at {self.url}")
                    return

                except Exception as e:
                    if self.enable_metrics:
                        QdrantMemoryStore._connection_metrics['failed_requests'] += 1

                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Connection attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        QdrantMemoryStore._connection_metrics['health_status'] = 'unhealthy'
                        raise ConnectionError(
                            f"Failed to connect to Qdrant after {max_retries} attempts: {e}"
                        )

    def _initialize_embedder(self, model_name: str) -> None:
        """Initialize embedder singleton for memory efficiency."""
        with QdrantMemoryStore._client_lock:
            if QdrantMemoryStore._embedder_instance is None:
                logger.info(f"Loading embedding model: {model_name}")
                QdrantMemoryStore._embedder_instance = SentenceTransformer(model_name)

            self.embedder = QdrantMemoryStore._embedder_instance
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
    
    def _setup_collections(self):
        """Create collections for multi-scale vectors."""
        for scale in self.scales:
            collection_name = f"{self.collection_prefix}_{scale}"

            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
                logger.info(f"Collection {collection_name} already exists")
            except Exception:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=scale,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection {collection_name}")

    def _execute_with_retry(self, func, *args, **kwargs):
        """
        Execute Qdrant operation with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function execution

        Raises:
            Exception: If all retries fail
        """
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                if self.enable_metrics:
                    QdrantMemoryStore._connection_metrics['total_requests'] += 1

                result = func(*args, **kwargs)
                return result

            except (ResponseHandlingException, UnexpectedResponse, ConnectionError) as e:
                if self.enable_metrics:
                    QdrantMemoryStore._connection_metrics['failed_requests'] += 1
                    QdrantMemoryStore._connection_metrics['retry_count'] += 1

                if attempt < max_retries - 1:
                    logger.warning(
                        f"Qdrant operation failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

            except Exception as e:
                # Non-retryable error
                if self.enable_metrics:
                    QdrantMemoryStore._connection_metrics['failed_requests'] += 1
                raise

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Qdrant connection.

        Returns:
            Dict containing health status and metrics
        """
        health = {
            'status': 'unknown',
            'connected': False,
            'collections': [],
            'metrics': {},
            'last_check': None,
            'error': None
        }

        try:
            # Check client connectivity
            collections = self._execute_with_retry(self.client.get_collections)
            health['status'] = 'healthy'
            health['connected'] = True
            health['collections'] = [c.name for c in collections.collections]

            # Get metrics if enabled
            if self.enable_metrics:
                health['metrics'] = {
                    'total_requests': QdrantMemoryStore._connection_metrics['total_requests'],
                    'failed_requests': QdrantMemoryStore._connection_metrics['failed_requests'],
                    'retry_count': QdrantMemoryStore._connection_metrics['retry_count'],
                    'failure_rate': (
                        QdrantMemoryStore._connection_metrics['failed_requests'] /
                        max(1, QdrantMemoryStore._connection_metrics['total_requests'])
                    )
                }

            health['last_check'] = time.time()

            if self.enable_metrics:
                QdrantMemoryStore._connection_metrics['last_health_check'] = health['last_check']
                QdrantMemoryStore._connection_metrics['health_status'] = 'healthy'

        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            logger.error(f"Qdrant health check failed: {e}")

            if self.enable_metrics:
                QdrantMemoryStore._connection_metrics['health_status'] = 'unhealthy'

        return health

    def get_connection_metrics(self) -> Dict[str, Any]:
        """
        Get connection metrics for monitoring.

        Returns:
            Dict with connection statistics and performance metrics
        """
        metrics = {
            'url': self.url,
            'timeout': self.timeout,
            'prefer_grpc': self.prefer_grpc,
            'health_status': QdrantMemoryStore._connection_metrics.get('health_status', 'unknown'),
            'total_requests': QdrantMemoryStore._connection_metrics.get('total_requests', 0),
            'failed_requests': QdrantMemoryStore._connection_metrics.get('failed_requests', 0),
            'retry_count': QdrantMemoryStore._connection_metrics.get('retry_count', 0),
            'last_health_check': QdrantMemoryStore._connection_metrics.get('last_health_check'),
        }

        # Calculate failure rate
        if metrics['total_requests'] > 0:
            metrics['failure_rate'] = metrics['failed_requests'] / metrics['total_requests']
        else:
            metrics['failure_rate'] = 0.0

        # Add warnings if needed
        if metrics['retry_count'] > metrics['total_requests'] * 0.1:
            metrics['warning'] = f"High retry rate detected ({metrics['retry_count']} retries). Check network stability."

        return metrics
    
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
            logger.error(f"✗ Embedding extraction failed: {e}")
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

                self._execute_with_retry(
                    self.client.upsert,
                    collection_name=collection_name,
                    points=[PointStruct(id=qdrant_id, vector=vector, payload=payload)]
                )
                scales_stored += 1

            except Exception as e:
                logger.warning(
                    f"⚠ Failed to store at scale {scale}d: {e}"
                )
                # Continue with other scales (partial success is okay)

        # ============================================================
        # Final Validation
        # ============================================================
        if scales_stored == 0:
            raise RuntimeError(f"Failed to store memory at any scale")

        logger.info(
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

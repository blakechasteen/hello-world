"""
Simple RAG - Clean wrapper around HoloLoom's memory + LLM integration.

This is a thin wrapper that delegates to existing components:
- HoloLoom: Memory management (experience/recall/reflect)
- WeavingOrchestrator: LLM-based query processing
- Reranker: Cross-encoder precision boost

All complexity is hidden behind a simple API:
    result = await rag.query("question")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncGenerator, TYPE_CHECKING
import logging
import time

from HoloLoom.hololoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query
from HoloLoom.rag.reranking import create_reranker, Reranker

# Custom embedding plugins (graceful degradation if unavailable)
try:
    from HoloLoom.rag.embedding_plugins import (
        EmbeddingProvider,
        MatryoshkaEmbedding,
        validate_embedding_provider,
    )
    EMBEDDING_PLUGINS_AVAILABLE = True
except ImportError:
    EMBEDDING_PLUGINS_AVAILABLE = False
    EmbeddingProvider = None
    MatryoshkaEmbedding = None
    validate_embedding_provider = None

# LLM integration (graceful degradation if unavailable)
try:
    from HoloLoom.weaving_orchestrator_llm import WeavingOrchestrator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

if TYPE_CHECKING:
    from HoloLoom.rag.embedding_plugins import EmbeddingProvider

# Streaming support
try:
    from HoloLoom.rag.streaming import StreamingRAGMixin, StreamToken
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    # Provide fallback mixin
    class StreamingRAGMixin:
        pass
    class StreamToken:
        pass

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """
    Structured result from a RAG query.

    Attributes:
        response: LLM-generated answer
        sources: List of retrieved source texts
        confidence: Confidence score (0.0-1.0)
        reasoning_mode: Mode used ("direct", "verify", "research", "plan_execute")
        metadata: Additional info (latency_ms, cache_hit, tool_used, rerank_latency, etc.)
    """
    response: str
    sources: List[str]
    confidence: float
    reasoning_mode: str = "direct"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"RAGResult(\n"
            f"  response: {self.response[:100]}...\n"
            f"  sources: {len(self.sources)} items\n"
            f"  confidence: {self.confidence:.2f}\n"
            f"  mode: {self.reasoning_mode}\n"
            f")"
        )


class SimpleRAG(StreamingRAGMixin):
    """
    Simple RAG interface for HoloLoom.

    Combines memory management (HoloLoom) with LLM generation
    (WeavingOrchestrator) and reranking (cross-encoder) into a clean,
    zero-config API.

    Usage:
        async with SimpleRAG(enable_reranking=True) as rag:
            # Ingest
            await rag.ingest("Thompson Sampling uses Bayesian statistics")

            # Query (with reranking for better precision)
            result = await rag.query("What is Thompson Sampling?")
            print(result.response)
            print(f"Confidence: {result.confidence}")

    Features:
        - Zero-config initialization
        - Supports text, PDF, image ingestion (delegates to HoloLoom)
        - Returns: response, sources, confidence, reasoning mode
        - Graceful degradation if LLM unavailable
        - Advanced reranking for 10-20% precision boost (optional)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        enable_caching: bool = True,
        enable_reranking: bool = False,
        reranker: str = "cross-encoder",
        rerank_top_k: int = 20,
        embedding_provider: Optional["EmbeddingProvider"] = None,
    ):
        """
        Initialize SimpleRAG with optional configuration.

        Args:
            config: System configuration (defaults to Config.fast())
            llm_provider: LLM provider ("ollama", "anthropic", "openai")
            llm_model: Specific model to use (optional, uses provider defaults)
            enable_caching: Enable query caching for repeated queries
            enable_reranking: Enable cross-encoder reranking for precision boost
            reranker: Reranker type ("cross-encoder" or "noop")
            rerank_top_k: Number of documents to retrieve before reranking
                          (e.g., retrieve top 20, rerank to top 5)
            embedding_provider: Custom embedding provider
                (defaults to MatryoshkaEmbedding). Must implement EmbeddingProvider protocol.

        Example:
            # Basic usage (no reranking)
            rag = SimpleRAG()

            # With reranking enabled
            rag = SimpleRAG(enable_reranking=True, rerank_top_k=20)

            # With custom embeddings
            from HoloLoom.rag.embedding_plugins import HuggingFaceEmbedding
            embedding = HuggingFaceEmbedding("all-mpnet-base-v2")
            rag = SimpleRAG(embedding_provider=embedding)
        """
        self.config = config or Config.fast()
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.enable_caching = enable_caching

        # Reranking configuration
        self.enable_reranking = enable_reranking
        self.rerank_top_k = rerank_top_k
        self.reranker: Optional[Reranker] = None

        # Setup embedding provider
        self._setup_embedding_provider(embedding_provider)

        # Will be initialized in __aenter__
        self.loom: Optional[HoloLoom] = None
        self.orchestrator: Optional[Any] = None

        # Query cache
        self._cache: Dict[str, RAGResult] = {}

        # Reranking stats
        self._rerank_stats = {
            'total_reranks': 0,
            'total_rerank_latency_ms': 0.0,
        }

    def _setup_embedding_provider(self, embedding_provider: Optional["EmbeddingProvider"]) -> None:
        """
        Setup embedding provider with validation and graceful fallback.

        Args:
            embedding_provider: Custom embedding provider or None for default

        Side effects:
            - Sets self.embedding_provider
            - Disables zero-copy for non-Matryoshka embeddings
            - Falls back to MatryoshkaEmbedding on error
        """
        if embedding_provider is None:
            # Use default Matryoshka
            try:
                if EMBEDDING_PLUGINS_AVAILABLE and MatryoshkaEmbedding:
                    self.embedding_provider = MatryoshkaEmbedding()
                    logger.info("✓ Using default MatryoshkaEmbedding (384 dims)")
                else:
                    logger.warning("⚠ Embedding plugins unavailable, using fallback")
                    self.embedding_provider = None
            except Exception as e:
                logger.warning(f"⚠ Failed to initialize MatryoshkaEmbedding: {e}")
                self.embedding_provider = None
        else:
            # Use custom embedding provider
            try:
                # Validate provider
                if EMBEDDING_PLUGINS_AVAILABLE and validate_embedding_provider:
                    if not validate_embedding_provider(embedding_provider):
                        raise ValueError("Embedding provider does not satisfy protocol")

                    # Disable zero-copy for custom embeddings
                    if not isinstance(embedding_provider, MatryoshkaEmbedding):
                        logger.warning("⚠ Zero-copy embeddings disabled with custom provider")
                        self.config.enable_zero_copy_embeddings = False

                    self.embedding_provider = embedding_provider
                    logger.info(
                        f"✓ Using custom {type(embedding_provider).__name__} "
                        f"({embedding_provider.dimension} dims)"
                    )
                else:
                    logger.warning("⚠ Embedding plugins unavailable, ignoring custom provider")
                    self.embedding_provider = None

            except Exception as e:
                logger.warning(f"⚠ Custom embedding provider failed: {e}, falling back to Matryoshka")
                try:
                    if EMBEDDING_PLUGINS_AVAILABLE and MatryoshkaEmbedding:
                        self.embedding_provider = MatryoshkaEmbedding()
                    else:
                        self.embedding_provider = None
                except Exception as e2:
                    logger.error(f"Failed to fallback to MatryoshkaEmbedding: {e2}")
                    self.embedding_provider = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize memory system
        self.loom = HoloLoom(config=self.config)
        await self.loom.__aenter__()
        logger.info("✓ HoloLoom memory system initialized")

        # Initialize reranker (optional)
        if self.enable_reranking:
            try:
                self.reranker = create_reranker(reranker)
                logger.info(f"✓ Reranker initialized ({reranker})")
            except Exception as e:
                logger.warning(f"⚠ Reranker initialization failed: {e}")
                logger.warning("Continuing without reranking (precision boost disabled)")
                self.enable_reranking = False
                self.reranker = None

        # Initialize orchestrator (with LLM if available)
        if LLM_AVAILABLE:
            try:
                self.orchestrator = WeavingOrchestrator(
                    cfg=self.config,
                    shards=[],  # Use loom's memory
                    llm_provider=self.llm_provider,
                    llm=None,  # Auto-initialize based on provider
                )
                await self.orchestrator.__aenter__()
                logger.info(f"✓ LLM orchestrator initialized ({self.llm_provider})")
            except Exception as e:
                logger.warning(f"⚠ LLM orchestrator failed: {e}, using fallback")
                self.orchestrator = None
        else:
            logger.warning("⚠ LLM orchestrator unavailable, install weaving_orchestrator_llm")
            self.orchestrator = None

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit (cleanup)."""
        # Close orchestrator
        if self.orchestrator:
            try:
                await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)
                logger.info("✓ Orchestrator cleaned up")
            except Exception as e:
                logger.error(f"Error closing orchestrator: {e}")

        # Close memory system
        if self.loom:
            try:
                await self.loom.__aexit__(exc_type, exc_val, exc_tb)
                logger.info("✓ HoloLoom cleaned up")
            except Exception as e:
                logger.error(f"Error closing HoloLoom: {e}")

    async def ingest(self, content: Any) -> None:
        """
        Ingest content into the knowledge base.

        Supports any modality (text, PDF, image, etc.) by delegating
        to HoloLoom's multimodal input router.

        Args:
            content: Text string, file path, bytes, or structured data

        Example:
            await rag.ingest("Thompson Sampling balances exploration")
            await rag.ingest("document.pdf")
            await rag.ingest("image.png")

        Raises:
            RuntimeError: If not in async context (not initialized)
        """
        if self.loom is None:
            raise RuntimeError("SimpleRAG not initialized. Use: async with SimpleRAG() as rag:")

        await self.loom.experience(content)
        logger.debug(f"✓ Ingested: {str(content)[:50]}...")

    async def query(
        self,
        question: str,
        mode: str = "verify",
        max_sources: int = 5,
        use_cache: bool = True,
    ) -> RAGResult:
        """
        Query the knowledge base with retrieval + optional reranking + LLM generation.

        Process:
        1. Check cache (optional)
        2. Retrieve relevant memories using HoloLoom.recall()
        3. Rerank using cross-encoder if enabled (improves precision 10-20%)
        4. Generate answer using LLM (if available)
        5. Return: response + sources + confidence + reasoning_mode

        Args:
            question: Query text
            mode: Reasoning mode
                - "direct": Single-pass answer
                - "verify": Answer with verification
                - "research": Multi-query exploration
                - "plan_execute": Goal decomposition (default: "verify")
            max_sources: Maximum number of source documents to return
            use_cache: Use cached results for repeated queries

        Returns:
            RAGResult with response, sources, confidence, and metadata

        Example:
            result = await rag.query("What is Thompson Sampling?")
            print(result.response)              # LLM answer
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Sources: {len(result.sources)}")
            print(f"Reranking latency: {result.metadata.get('rerank_latency_ms')}ms")

        Raises:
            RuntimeError: If not in async context (not initialized)
        """
        if self.loom is None:
            raise RuntimeError("SimpleRAG not initialized. Use: async with SimpleRAG() as rag:")

        # Check cache
        cache_key = (question, mode)
        if use_cache and self.enable_caching and cache_key in self._cache:
            logger.debug(f"✓ Cache hit: {question[:40]}...")
            result = self._cache[cache_key]
            result.metadata['cache_hit'] = True
            return result

        # 1. Retrieve memories
        # If reranking enabled, retrieve more documents than needed
        retrieval_k = self.rerank_top_k if self.enable_reranking else max_sources
        logger.debug(f"Retrieving {retrieval_k} memories for: {question[:50]}...")
        memories = await self.loom.recall(question, limit=retrieval_k)

        # 2. Rerank if enabled
        rerank_latency_ms = 0.0
        if self.enable_reranking and len(memories) > max_sources and self.reranker:
            logger.debug(f"Reranking {len(memories)} documents...")
            start_rerank = time.time()

            try:
                # Extract text from memories
                doc_texts = [m.text for m in memories]

                # Rerank
                reranked_indices = self.reranker.rerank(
                    question,
                    doc_texts,
                    max_sources
                )

                # Reorder memories based on reranking
                reranked_memories = [memories[idx] for idx, _ in reranked_indices]
                memories = reranked_memories

                rerank_latency_ms = (time.time() - start_rerank) * 1000
                self._rerank_stats['total_reranks'] += 1
                self._rerank_stats['total_rerank_latency_ms'] += rerank_latency_ms

                logger.debug(
                    f"✓ Reranked in {rerank_latency_ms:.1f}ms. "
                    f"Top score: {reranked_indices[0][1]:.3f}"
                )

            except Exception as e:
                logger.warning(f"⚠ Reranking failed: {e}, using original order")
                memories = memories[:max_sources]

        # Enforce max_sources limit
        sources = [mem.text for mem in memories[:max_sources]]

        # 3. Generate answer using LLM
        answer = None
        confidence = 0.0
        if self.orchestrator and sources:
            logger.debug(f"Generating answer with LLM ({self.llm_provider})...")
            try:
                # Create query object
                query_obj = Query(text=question)

                # Call orchestrator
                spacetime = await self.orchestrator.weave(query_obj, use_llm=True)

                # Extract answer
                answer = spacetime.response if hasattr(spacetime, 'response') else None
                confidence = spacetime.confidence if hasattr(spacetime, 'confidence') else 0.8

            except Exception as e:
                logger.warning(f"⚠ LLM generation failed: {e}")
                # Fall through to fallback

        # Fallback: Synthesize from sources
        if not answer:
            if sources:
                answer = f"Based on {len(sources)} sources: " + " ".join(sources[:2])
                confidence = 0.6 if sources else 0.2
            else:
                answer = f"No relevant information found for: {question}"
                confidence = 0.1

        # 4. Create result
        result = RAGResult(
            response=answer,
            sources=sources,
            confidence=confidence,
            reasoning_mode=mode,
            metadata={
                'n_sources': len(sources),
                'llm_provider': self.llm_provider if self.orchestrator else None,
                'cache_hit': False,
                'rerank_latency_ms': rerank_latency_ms,
                'reranking_enabled': self.enable_reranking,
            }
        )

        # Cache result
        if self.enable_caching:
            self._cache[cache_key] = result

        logger.debug(
            f"✓ Query complete: confidence={confidence:.2f}, sources={len(sources)}, "
            f"rerank_latency={rerank_latency_ms:.1f}ms"
        )
        return result

    async def batch_query(
        self,
        questions: List[str],
        mode: str = "verify",
        max_sources: int = 5,
    ) -> List[RAGResult]:
        """
        Query multiple questions in batch.

        Efficient batch processing with shared memory system.

        Args:
            questions: List of query strings
            mode: Reasoning mode (applied to all queries)
            max_sources: Maximum sources per query

        Returns:
            List of RAGResult objects

        Example:
            questions = [
                "What is Thompson Sampling?",
                "How does it work?",
                "What are the tradeoffs?"
            ]
            results = await rag.batch_query(questions)
            for result in results:
                print(f"{result.confidence:.2f} {result.response[:50]}")
        """
        logger.info(f"Batch querying {len(questions)} questions...")
        results = []
        for i, question in enumerate(questions, 1):
            logger.debug(f"[{i}/{len(questions)}] {question[:50]}...")
            result = await self.query(question, mode=mode, max_sources=max_sources)
            results.append(result)

        logger.info(f"✓ Batch complete: {len(results)} results")
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics for monitoring.

        Returns:
            Dictionary with metrics from HoloLoom, cache stats, reranking stats, and embedding info

        Example:
            metrics = rag.get_metrics()
            print(f"Memories: {metrics['n_memories']}")
            print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
            print(f"Average rerank latency: {metrics['avg_rerank_latency_ms']:.1f}ms")
            print(f"Embedding provider: {metrics['embedding_provider']}")
        """
        loom_metrics = self.loom.get_metrics() if self.loom else {}

        cache_hits = sum(
            1 for r in self._cache.values()
            if r.metadata.get('cache_hit', False)
        )
        cache_hit_rate = (
            cache_hits / len(self._cache) if self._cache else 0.0
        )

        avg_rerank_latency = (
            self._rerank_stats['total_rerank_latency_ms'] / self._rerank_stats['total_reranks']
            if self._rerank_stats['total_reranks'] > 0 else 0.0
        )

        # Get embedding provider info
        embedding_provider_name = None
        embedding_dimension = None
        if hasattr(self, 'embedding_provider') and self.embedding_provider:
            embedding_provider_name = type(self.embedding_provider).__name__
            embedding_dimension = self.embedding_provider.dimension

        return {
            **loom_metrics,
            'cache_size': len(self._cache),
            'cache_hit_rate': cache_hit_rate,
            'llm_provider': self.llm_provider,
            'llm_available': self.orchestrator is not None,
            'reranking_enabled': self.enable_reranking,
            'total_reranks': self._rerank_stats['total_reranks'],
            'avg_rerank_latency_ms': avg_rerank_latency,
            'embedding_provider': embedding_provider_name,
            'embedding_dimension': embedding_dimension,
        }

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()
        logger.info("✓ Query cache cleared")

    def summary(self) -> str:
        """
        Get human-readable system summary.

        Returns:
            Summary string

        Example:
            print(rag.summary())
        """
        metrics = self.get_metrics()
        rerank_info = ""
        if metrics.get('reranking_enabled'):
            rerank_info = f"Reranking: {metrics.get('total_reranks', 0)} reranks, avg latency {metrics.get('avg_rerank_latency_ms', 0):.1f}ms\n"

        return f"""SimpleRAG System
===============
Memories: {metrics.get('n_memories', 0)}
Cache size: {metrics.get('cache_size', 0)}
Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}
LLM provider: {metrics.get('llm_provider')}
LLM available: {metrics.get('llm_available')}
{rerank_info}"""

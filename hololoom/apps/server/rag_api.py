#!/usr/bin/env python3
from __future__ import annotations
"""
HoloLoom RAG API Server
=======================

FastAPI server exposing the RAG system via REST endpoints.

Features:
- Ingest content into knowledge base
- Query with multiple reasoning modes (direct, verify, research, plan_execute)
- Automatic response caching
- Optional reranking for precision boost
- Health checks and statistics
- CORS for frontend access

Usage:
    # Development (with auto-reload)
    PYTHONPATH=. uvicorn HoloLoom.apps.server.rag_api:app --reload --port 8000

    # Production
    PYTHONPATH=. uvicorn HoloLoom.apps.server.rag_api:app --host 0.0.0.0 --port 8000 --workers 4

View API docs: http://localhost:8000/docs

Created: 2025-11-20
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from time import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Import RAG system
try:
    from hololoom.rag import RAGResult, SimpleRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    RAGResult = None
    SimpleRAG = None

from hololoom.config import Config

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Simple sliding window rate limiter.

    SECURITY: Uses asyncio.Lock for proper async concurrency control.
    This prevents race conditions when multiple requests arrive simultaneously.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()  # SECURITY: Async-safe access

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        async with self._lock:  # SECURITY: Async-safe critical section
            now = time()
            cutoff = now - self.window_seconds

            # Remove old requests
            while self.requests[client_id] and self.requests[client_id][0] < cutoff:
                self.requests[client_id].popleft()

            # Check limit
            if len(self.requests[client_id]) >= self.max_requests:
                return False

            self.requests[client_id].append(now)
            return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time()
        cutoff = now - self.window_seconds
        count = sum(1 for t in self.requests[client_id] if t >= cutoff)
        return max(0, self.max_requests - count)


# ============================================================================
# Request/Response Models
# ============================================================================

class IngestRequest(BaseModel):
    """Request to ingest content."""
    content: str = Field(..., description="Content to ingest (text, file path, etc.)")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional metadata about the content"
    )

    class Config:
        schema_extra = {
            "example": {
                "content": "Thompson Sampling uses Bayesian statistics for exploration-exploitation.",
                "metadata": {"source": "ML book", "chapter": "Bandits"}
            }
        }


class IngestResponse(BaseModel):
    """Response from ingest."""
    status: str = Field(..., description="Status: 'success' or 'error'")
    message: str = Field(..., description="Status message")


class QueryRequest(BaseModel):
    """Request to query RAG system."""
    question: str = Field(..., description="Question to ask")
    mode: str = Field(
        default="verify",
        description="Reasoning mode: 'direct', 'verify', 'research', or 'plan_execute'"
    )
    max_sources: int = Field(
        default=5,
        description="Maximum number of sources to return",
        ge=1,
        le=50
    )
    use_cache: bool = Field(
        default=True,
        description="Use cached results for repeated queries"
    )

    @validator('mode')
    def validate_mode(cls, v):
        allowed = {"direct", "verify", "research", "plan_execute"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "question": "What is Thompson Sampling?",
                "mode": "verify",
                "max_sources": 5
            }
        }


class Source(BaseModel):
    """A source document."""
    text: str = Field(..., description="Source text snippet")
    score: float | None = Field(None, description="Relevance score (0-1)")
    metadata: dict[str, Any] | None = Field(None, description="Source metadata")


class QueryResponse(BaseModel):
    """Response from query."""
    response: str = Field(..., description="LLM-generated answer")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
    sources: list[Source] = Field(..., description="Retrieved sources")
    reasoning_mode: str = Field(..., description="Reasoning mode used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="'healthy' or 'degraded'")
    rag_available: bool = Field(..., description="RAG system available")
    features: list[str] = Field(..., description="Enabled features")
    timestamp: str = Field(..., description="Timestamp of check")


class StatsResponse(BaseModel):
    """System statistics."""
    total_ingests: int = Field(..., description="Total ingest operations")
    total_queries: int = Field(..., description="Total queries")
    cache_hits: int = Field(..., description="Number of cache hits")
    cache_hit_rate: float = Field(..., description="Cache hit rate (0-1)")
    avg_query_latency_ms: float = Field(..., description="Average query latency")
    rag_initialized: bool = Field(..., description="RAG system initialized")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom RAG API",
    description="Level 4 Agentic RAG System with hybrid search, graph RAG, and agentic reasoning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
# TODO: In production, use environment variable for allowed origins:
#   ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Development origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global State
# ============================================================================

# RAG instance (initialized on startup)
rag: SimpleRAG | None = None
config: Config | None = None

# Rate limiting
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# Statistics tracking
stats = {
    'total_ingests': 0,
    'total_queries': 0,
    'cache_hits': 0,
    'query_latencies': deque(maxlen=100),
}


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag, config

    logger.info("=" * 70)
    logger.info("HoloLoom RAG API Starting Up")
    logger.info("=" * 70)

    try:
        if not RAG_AVAILABLE:
            logger.error("❌ RAG module not available")
            return

        # Create config
        config = Config.fast()
        logger.info("✓ Config created (mode: fast)")

        # Initialize RAG
        rag = SimpleRAG(
            config=config,
            llm_provider="ollama",
            enable_caching=True,
            enable_reranking=False,  # Can enable for precision boost
        )
        await rag.__aenter__()
        logger.info("✓ RAG system initialized (HoloLoom + LLM)")
        logger.info("✓ Query caching: ENABLED")
        logger.info("✓ Reranking: DISABLED (set enable_reranking=True to enable)")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        rag = None
        raise

    logger.info("=" * 70)
    logger.info("HoloLoom RAG API Ready")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up RAG system on shutdown."""
    global rag

    logger.info("=" * 70)
    logger.info("HoloLoom RAG API Shutting Down")
    logger.info("=" * 70)

    if rag:
        try:
            await rag.__aexit__(None, None, None)
            logger.info("✓ RAG system cleaned up")
        except Exception as e:
            logger.error(f"⚠ Error during shutdown: {e}")

    logger.info("=" * 70)


# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time()
    response = await call_next(request)
    process_time = time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# ============================================================================
# Helper Functions
# ============================================================================

def get_client_id(request: Request) -> str:
    """Get client ID from request."""
    return request.client.host if request.client else "unknown"


async def check_rag_ready() -> None:
    """Check if RAG is initialized."""
    if not rag or rag.loom is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Server may still be starting."
        )


# ============================================================================
# Routes: Health & Status
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns system status and available features.
    """
    try:
        features = []
        if rag and rag.loom:
            features.append("rag_ingest")
            features.append("rag_query")
            if rag.enable_caching:
                features.append("query_caching")
            if rag.enable_reranking:
                features.append("reranking")
            if rag.orchestrator:
                features.append("llm_generation")

        return HealthResponse(
            status="healthy" if (rag and rag.loom) else "degraded",
            rag_available=rag is not None and rag.loom is not None,
            features=features,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Get system statistics.

    Returns ingestion, query, and cache metrics.
    """
    try:
        total_queries = stats['total_queries']
        cache_hits = stats['cache_hits']
        cache_hit_rate = (cache_hits / total_queries) if total_queries > 0 else 0.0

        avg_latency = 0.0
        if stats['query_latencies']:
            avg_latency = sum(stats['query_latencies']) / len(stats['query_latencies'])

        return StatsResponse(
            total_ingests=stats['total_ingests'],
            total_queries=total_queries,
            cache_hits=cache_hits,
            cache_hit_rate=cache_hit_rate,
            avg_query_latency_ms=avg_latency,
            rag_initialized=rag is not None and rag.loom is not None,
        )
    except Exception as e:
        logger.error(f"Stats request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Routes: Ingest
# ============================================================================

@app.post("/api/rag/ingest", response_model=IngestResponse)
async def ingest_content(request: IngestRequest, req: Request) -> IngestResponse:
    """
    Ingest content into the knowledge base.

    Supports any modality (text, file paths, etc.) through HoloLoom's
    multimodal input router.

    Args:
        request: Ingest request with content and optional metadata

    Returns:
        IngestResponse with status and message

    Raises:
        503: RAG system not initialized
        500: Ingest failed
    """
    # Rate limit
    client_id = get_client_id(req)
    if not await rate_limiter.check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 100 requests per 60 seconds."
        )

    # Check RAG ready
    await check_rag_ready()

    try:
        # Ingest
        await rag.ingest(request.content)
        stats['total_ingests'] += 1

        logger.info(f"✓ Ingested: {request.content[:50]}...")

        return IngestResponse(
            status="success",
            message=f"Successfully ingested content ({len(request.content)} chars)"
        )

    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


# ============================================================================
# Routes: Query
# ============================================================================

@app.post("/api/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, req: Request) -> QueryResponse:
    """
    Query the knowledge base.

    Process:
    1. Check query cache (optional)
    2. Retrieve relevant memories
    3. Rerank using cross-encoder (optional)
    4. Generate answer using LLM
    5. Return response + sources + confidence

    Args:
        request: Query request with question, mode, and options

    Returns:
        QueryResponse with response, sources, confidence, and metadata

    Raises:
        503: RAG system not initialized
        400: Invalid request
        500: Query failed
    """
    # Rate limit
    client_id = get_client_id(req)
    if not await rate_limiter.check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 100 requests per 60 seconds."
        )

    # Check RAG ready
    await check_rag_ready()

    try:
        start_time = time()

        # Query
        rag_result: RAGResult = await rag.query(
            question=request.question,
            mode=request.mode,
            max_sources=request.max_sources,
            use_cache=request.use_cache,
        )

        latency_ms = (time() - start_time) * 1000
        stats['total_queries'] += 1
        stats['query_latencies'].append(latency_ms)

        # Track cache hits
        if rag_result.metadata.get('cache_hit'):
            stats['cache_hits'] += 1

        logger.info(
            f"✓ Query: {request.question[:40]}... "
            f"(mode={request.mode}, latency={latency_ms:.1f}ms, "
            f"confidence={rag_result.confidence:.2f})"
        )

        # Convert sources to response format
        sources = [
            Source(text=s, score=None, metadata=None)
            for s in rag_result.sources
        ]

        # Prepare metadata
        metadata = rag_result.metadata.copy()
        metadata['latency_ms'] = latency_ms
        metadata['cache_hit'] = rag_result.metadata.get('cache_hit', False)

        return QueryResponse(
            response=rag_result.response,
            confidence=rag_result.confidence,
            sources=sources,
            reasoning_mode=rag_result.reasoning_mode,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ============================================================================
# Routes: Advanced (Optional)
# ============================================================================

@app.post("/api/rag/batch-query")
async def batch_query(
    requests_list: list[QueryRequest],
    req: Request
) -> list[QueryResponse]:
    """
    Query multiple questions in a single request.

    Useful for batch processing and research workflows.

    Args:
        requests_list: List of query requests

    Returns:
        List of query responses in same order
    """
    # Rate limit (cost: 1 per query)
    client_id = get_client_id(req)
    for _ in requests_list:
        if not await rate_limiter.check_rate_limit(client_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )

    await check_rag_ready()

    results = []
    for query_req in requests_list:
        try:
            result = await query_rag(query_req, req)
            results.append(result)
        except HTTPException as e:
            results.append(
                QueryResponse(
                    response="",
                    confidence=0.0,
                    sources=[],
                    reasoning_mode="error",
                    metadata={"error": str(e.detail)}
                )
            )

    return results


# ============================================================================
# Root endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "HoloLoom RAG API",
        "version": "1.0.0",
        "description": "Level 4 Agentic RAG System",
        "docs_url": "/docs",
        "status": "ready" if (rag and rag.loom) else "initializing",
        "endpoints": {
            "health": "GET /health",
            "ingest": "POST /api/rag/ingest",
            "query": "POST /api/rag/query",
            "batch_query": "POST /api/rag/batch-query",
            "stats": "GET /api/rag/stats",
        }
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting RAG API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

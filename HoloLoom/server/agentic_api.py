#!/usr/bin/env python3
"""
HoloLoom Agentic API Server
============================
FastAPI server for agentic intelligence integration with VS Code Squad extension.

Provides HTTP endpoints that match the TypeScript interfaces in squad/src/HoloLoomBridge.ts

Usage:
    # Development
    uvicorn HoloLoom.server.agentic_api:app --reload --port 8000

    # Production
    uvicorn HoloLoom.server.agentic_api:app --host 0.0.0.0 --port 8000 --workers 4
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import defaultdict, deque
from time import time

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from HoloLoom.agentic import create_agentic_orchestrator, ReasoningMode, AgenticResult
# from HoloLoom.agentic.codebase_ingestion import CodebaseIndexer, Language as CodeLanguage
# from HoloLoom.agentic.hallucination_detector import HallucinationDetector
# from HoloLoom.agentic.code_verification import CodeVerifier
# from HoloLoom.agentic.ai_slop_detector import AISlopDetector
from HoloLoom.agentic.ml_logic_detector import MLLogicDetector, Language as CodeLanguage
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.alignment.audit_trail import AuditTrail
from HoloLoom.alignment.safety_guardrails import SafetyGuardrails, ActionRequest, RiskLevel
from HoloLoom.alignment.deception_detection import DeceptionDetector

# SaaS API Components (Dec 2025)
try:
    from HoloLoom.saas import create_saas_backend, SaaSBackend
    from HoloLoom.saas.routes import customers_router, api_keys_router
    from HoloLoom.saas.auth import add_rate_limit_headers
    SAAS_AVAILABLE = True
except ImportError as e:
    SAAS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"SaaS components unavailable: {e}")

# Agent Monitoring (Nov 2025)
try:
    from HoloLoom.agentic.monitoring import get_monitor, start_monitoring, stop_monitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("Agent monitoring unavailable (monitoring.py not found)")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiting & Stats Tracking
# ============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.

    Tracks requests per IP and enforces configurable limits.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()  # Async-safe access

    async def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client is within rate limit.

        Args:
            client_id: Client identifier (IP address)

        Returns:
            True if within limit, False if exceeded
        """
        async with self._lock:  # Async-safe critical section
            now = time()
            cutoff = now - self.window_seconds

            # Remove old requests outside window
            while self.requests[client_id] and self.requests[client_id][0] < cutoff:
                self.requests[client_id].popleft()

            # Check if limit exceeded
            if len(self.requests[client_id]) >= self.max_requests:
                return False

            # Record this request
            self.requests[client_id].append(now)
            return True

    async def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client (async-safe)."""
        async with self._lock:  # SECURITY: Ensure consistent read
            now = time()
            cutoff = now - self.window_seconds

            # Count requests in current window
            count = sum(1 for t in self.requests[client_id] if t >= cutoff)
            return max(0, self.max_requests - count)


class ServerStats:
    """
    Track server statistics.

    Monitors uptime, query counts, latencies, and error rates.
    """

    # SECURITY: Max unique keys to prevent unbounded memory growth
    MAX_MODES = 20
    MAX_ERROR_TYPES = 100

    def __init__(self):
        self.start_time = time()
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.latencies: deque = deque(maxlen=1000)  # Last 1000 latencies
        self.queries_by_mode: Dict[str, int] = defaultdict(int)
        self.errors_by_type: Dict[str, int] = defaultdict(int)

    def record_query(self, mode: str, latency_ms: float, success: bool):
        """Record a query completion."""
        self.total_queries += 1
        self.latencies.append(latency_ms)

        # SECURITY: Only track known modes to prevent memory exhaustion
        if len(self.queries_by_mode) < self.MAX_MODES or mode in self.queries_by_mode:
            self.queries_by_mode[mode] += 1

        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1

    def record_error(self, error_type: str):
        """Record an error occurrence."""
        # SECURITY: Only track limited error types to prevent memory exhaustion
        if len(self.errors_by_type) < self.MAX_ERROR_TYPES or error_type in self.errors_by_type:
            self.errors_by_type[error_type] += 1

    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return time() - self.start_time

    def get_avg_latency(self) -> float:
        """Get average latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def get_p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx] if idx < len(sorted_latencies) else sorted_latencies[-1]

    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_queries == 0:
            return 100.0
        return (self.successful_queries / self.total_queries) * 100

    def get_stats_dict(self) -> Dict[str, Any]:
        """Get all stats as dictionary."""
        return {
            "uptime_seconds": self.get_uptime(),
            "uptime_formatted": self._format_uptime(self.get_uptime()),
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": round(self.get_success_rate(), 2),
            "avg_latency_ms": round(self.get_avg_latency(), 2),
            "p95_latency_ms": round(self.get_p95_latency(), 2),
            "queries_by_mode": dict(self.queries_by_mode),
            "errors_by_type": dict(self.errors_by_type)
        }

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime as human-readable string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"


# ============================================================================
# Request/Response Models (Match TypeScript interfaces)
# ============================================================================

class CodeContext(BaseModel):
    """Code context from VS Code editor (matches TypeScript interface)."""
    currentFile: Optional[str] = None
    fileName: Optional[str] = None
    languageId: Optional[str] = None
    selection: Optional[str] = None
    workspace: Optional[str] = None
    diagnostics: Optional[List[Dict]] = None


class QueryRequest(BaseModel):
    """Query request from VS Code extension."""
    text: str = Field(..., description="Query text")
    context: Optional[CodeContext] = Field(None, description="Code context from editor")
    mode: str = Field("verify", description="Reasoning mode: direct, verify, research, plan_execute")
    max_steps: int = Field(5, description="Maximum reasoning steps")

    @validator('text')
    def validate_text_size(cls, v):
        """Validate query text size (max 100KB)."""
        if len(v) > 100_000:  # 100KB limit
            raise ValueError(f"Query text too large: {len(v)} bytes (max 100KB)")
        return v

    @validator('max_steps')
    def validate_max_steps(cls, v):
        """Validate max_steps is reasonable."""
        if v < 1 or v > 20:
            raise ValueError(f"max_steps must be between 1 and 20 (got {v})")
        return v


class VerificationResponse(BaseModel):
    """Verification result (matches TypeScript interface)."""
    verified: bool
    confidence: float
    contradictions: List[str]
    supporting_evidence: List[str]
    suggested_refinements: List[str]


class ReasoningStepResponse(BaseModel):
    """Single reasoning step (matches TypeScript interface)."""
    type: str
    query: Optional[str] = None
    confidence: Optional[float] = None
    finding: Optional[str] = None
    completed: Optional[bool] = None
    tool: Optional[str] = None


class AgenticResponse(BaseModel):
    """
    Agentic reasoning result (matches TypeScript AgenticResult interface).

    This is the main response returned to the VS Code extension.
    """
    response: str
    confidence: float
    reasoning_mode: str
    steps_taken: List[ReasoningStepResponse]
    total_queries: int
    total_duration_ms: float
    verification: Optional[VerificationResponse] = None

    # Additional metadata
    timestamp: str
    query_id: str


class MemoryAddRequest(BaseModel):
    """
    Request model for adding memories.

    SECURITY: Pydantic validation prevents resource exhaustion attacks.
    All fields have explicit size limits.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="Memory text content (max 100KB)"
    )
    episode: str = Field(
        default="default",
        max_length=1000,
        description="Episode/category for the memory (max 1KB)"
    )
    entities: List[str] = Field(
        default_factory=list,
        max_items=100,
        description="List of entities mentioned (max 100 items)"
    )
    motifs: List[str] = Field(
        default_factory=list,
        max_items=100,
        description="List of motifs/patterns (max 100 items)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @validator('entities', 'motifs', each_item=True)
    def validate_list_items(cls, v):
        """SECURITY: Limit individual list item lengths."""
        if len(v) > 1000:
            raise ValueError("List items must be under 1KB")
        return v

    @validator('metadata')
    def validate_metadata_size(cls, v):
        """SECURITY: Limit metadata size to prevent abuse."""
        import json
        if len(json.dumps(v)) > 10000:
            raise ValueError("Metadata must be under 10KB when serialized")
        return v


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom Agentic API",
    description="Agentic intelligence backend for VS Code Squad extension",
    version="1.0.0"
)

# CORS for VS Code extension
# SECURITY: Use environment variable for allowed origins (no wildcard in production)
import os as _cors_os
_cors_origins = _cors_os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080,vscode-webview://*"
).split(",")
_cors_allow_credentials = _cors_os.environ.get("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,  # Only enable if explicitly configured
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ChatOps Observability Routers (Prometheus metrics + WebSocket progress)
try:
    from HoloLoom.chatops.handlers.prometheus_metrics import (
        create_metrics_router, get_metrics_collector
    )
    _metrics_router = create_metrics_router(get_metrics_collector())
    if _metrics_router:
        app.include_router(_metrics_router, prefix="/api")
        logger.info("Prometheus metrics router mounted at /api/metrics")
except ImportError:
    logger.debug("Prometheus metrics not available (missing dependencies)")

try:
    from HoloLoom.chatops.handlers.websocket_progress import (
        create_progress_router, get_global_manager
    )
    _progress_router = create_progress_router(get_global_manager())
    if _progress_router:
        app.include_router(_progress_router)
        logger.info("WebSocket progress router mounted at /ws/progress")
except ImportError:
    logger.debug("WebSocket progress not available (missing dependencies)")

# SaaS API Routers (Dec 2025)
if SAAS_AVAILABLE:
    try:
        app.include_router(customers_router)
        app.include_router(api_keys_router)
        # Add rate limit headers middleware
        app.middleware("http")(add_rate_limit_headers)
        logger.info("SaaS API routers mounted at /api/v1/customers and /api/v1/api-keys")
    except Exception as e:
        logger.warning(f"Failed to mount SaaS routers: {e}")


# Helper function for secure client IP extraction
def _get_client_ip(request: Request) -> str:
    """
    Securely extract client IP address.

    SECURITY: Only trust X-Forwarded-For when BEHIND_PROXY=true.
    When behind a proxy, takes the rightmost IP (closest to trusted proxy).
    This prevents IP spoofing via forged X-Forwarded-For headers.
    """
    import os
    behind_proxy = os.environ.get("BEHIND_PROXY", "false").lower() == "true"

    if behind_proxy:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For format: "client, proxy1, proxy2"
            # Take the rightmost non-empty IP (closest to our trusted proxy)
            ips = [ip.strip() for ip in forwarded_for.split(",") if ip.strip()]
            if ips:
                # Return the first IP (original client) only if we trust the chain
                # In production, you might want to return ips[-1] instead if
                # you don't fully trust the proxy chain
                return ips[0]

    # Default: use direct connection IP
    return request.client.host if request.client else "unknown"


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware.

    Checks rate limit before processing request.
    Returns 429 Too Many Requests if limit exceeded.
    """
    # Skip rate limiting for health checks
    if request.url.path == "/health":
        return await call_next(request)

    # Get client IP (securely handles X-Forwarded-For when behind proxy)
    client_ip = _get_client_ip(request)

    # Check rate limit
    if state.rate_limiter and not await state.rate_limiter.check_rate_limit(client_ip):
        remaining = state.rate_limiter.get_remaining(client_ip)
        logger.warning(f"Rate limit exceeded for {client_ip}")

        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Rate limit exceeded. Try again later.",
                "remaining": remaining,
                "retry_after": state.rate_limiter.window_seconds
            }
        )

    # Process request
    response = await call_next(request)
    return response


# ============================================================================
# SECURITY: Global Exception Handler - Mask Error Details in Production
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that masks error details in production.

    SECURITY: Prevents information disclosure through error messages.
    In production, returns generic error message while logging full details.
    In development, returns full error details for debugging.
    """
    import os
    import traceback
    from fastapi.responses import JSONResponse

    environment = os.environ.get("ENVIRONMENT", "production").lower()
    is_production = environment == "production"

    # Always log full error details server-side
    error_id = f"ERR-{id(exc)}"
    logger.error(
        f"Unhandled exception [{error_id}]: {type(exc).__name__}: {exc}",
        exc_info=True
    )

    if is_production:
        # SECURITY: Return generic error message in production
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal error occurred. Please try again later.",
                "error_id": error_id,
                "support": "If this persists, contact support with the error_id."
            }
        )
    else:
        # Development: Return full error details for debugging
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "type": type(exc).__name__,
                "error_id": error_id,
                "traceback": traceback.format_exc() if environment == "development" else None
            }
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP exception handler that sanitizes error details.

    SECURITY: Even for HTTPException, we sanitize any potential
    sensitive information that may have been accidentally included.
    """
    import os
    import re
    from fastapi.responses import JSONResponse

    environment = os.environ.get("ENVIRONMENT", "production").lower()
    is_production = environment == "production"

    # Patterns that indicate sensitive information in error messages
    sensitive_patterns = [
        r'password[=:]\s*\S+',
        r'api[_-]?key[=:]\s*\S+',
        r'token[=:]\s*\S+',
        r'secret[=:]\s*\S+',
        r'sk-[a-zA-Z0-9]{20,}',  # API keys
        r'bolt://[^@]+:[^@]+@',  # Database connection strings with credentials
    ]

    detail = str(exc.detail) if exc.detail else "An error occurred"

    if is_production:
        # Check if error message contains sensitive information
        for pattern in sensitive_patterns:
            if re.search(pattern, detail, re.IGNORECASE):
                # Log the full error server-side
                error_id = f"ERR-{id(exc)}"
                logger.warning(
                    f"Sanitized sensitive info from error response [{error_id}]: {detail}"
                )
                detail = "An error occurred. Please try again."
                break

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": detail},
        headers=exc.headers
    )


# ============================================================================
# Global State
# ============================================================================

class ServerState:
    """Global server state."""
    orchestrator: Optional[Any] = None
    audit_trail: Optional[AuditTrail] = None
    safety_guardrails: Optional[SafetyGuardrails] = None  # Safety guardrails for risk gating
    deception_detector: Optional[DeceptionDetector] = None  # Deception detection
    config: Optional[Config] = None
    shards: List[MemoryShard] = []
    memory_backend: Optional[Any] = None  # Persistent memory backend
    # codebase_indexer: Optional[CodebaseIndexer] = None  # Codebase knowledge graph
    # hallucination_detector: Optional[HallucinationDetector] = None
    # code_verifier: Optional[CodeVerifier] = None
    # ai_slop_detector: Optional[AISlopDetector] = None  # Comprehensive slop detector
    ml_logic_detector: Optional[MLLogicDetector] = None  # ML-based logic error detector
    rate_limiter: Optional[RateLimiter] = None  # Rate limiting
    stats: Optional[ServerStats] = None  # Server statistics
    monitor: Optional[Any] = None  # Agent monitoring (Nov 2025)
    saas_backend: Optional[Any] = None  # SaaS backend (Dec 2025)


state = ServerState()


# ============================================================================
# Lifecycle
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize server with persistent memory."""
    logger.info("Starting HoloLoom Agentic API server...")

    # Initialize rate limiter and stats
    state.rate_limiter = RateLimiter(max_requests=60, window_seconds=60)
    state.stats = ServerStats()
    logger.info("Rate limiter: 60 requests/minute per IP")

    # Load config
    state.config = Config.fast()
    state.config.enable_agentic_reasoning = True

    # Initialize audit trail
    state.audit_trail = AuditTrail(persist_path="./alignment_logs")

    # Initialize alignment framework (safety guardrails + deception detection)
    # Safety Integration (Dec 2025) - MRF CRITIQUE: Remove testing_mode in production
    try:
        import os
        # SECURITY: Testing mode requires BOTH:
        # 1. HOLOLOOM_TESTING_MODE=true
        # 2. ENVIRONMENT=development
        # This prevents accidental testing mode in production deployments
        environment = os.environ.get("ENVIRONMENT", "production").lower()
        testing_mode_requested = os.environ.get("HOLOLOOM_TESTING_MODE", "").lower() == "true"

        # Only allow testing_mode if ENVIRONMENT is explicitly development
        if testing_mode_requested and environment != "development":
            logger.error("⚠️  SECURITY: HOLOLOOM_TESTING_MODE=true ignored because ENVIRONMENT != development")
            logger.error("   Set ENVIRONMENT=development to enable testing mode")
            testing_mode = False
        else:
            testing_mode = testing_mode_requested and environment == "development"

        if testing_mode:
            logger.warning("⚠️  TESTING MODE ENABLED - Safety approval requirements bypassed!")
            logger.warning("   Set HOLOLOOM_TESTING_MODE=false or ENVIRONMENT=production for production!")

        state.safety_guardrails = SafetyGuardrails(
            testing_mode=testing_mode,  # Safe by default (False unless explicitly enabled)
            enable_adversarial_detection=True,  # Always detect adversarial patterns
        )
        state.deception_detector = DeceptionDetector()
        mode_str = "TESTING MODE (bypasses approvals)" if testing_mode else "PRODUCTION MODE (full safety)"
        logger.info(f"✅ Alignment framework initialized: {mode_str}")
    except Exception as e:
        logger.warning(f"⚠️  Alignment framework initialization failed: {e}")
        logger.warning("   Proceeding without safety gating (NOT RECOMMENDED for production)")

    # ✅ Create persistent memory backend
    try:
        from HoloLoom.memory.backend_factory import create_memory_backend
        from HoloLoom.config import MemoryBackend

        state.config.memory_backend = MemoryBackend.HYBRID  # Use persistent storage
        state.memory_backend = await create_memory_backend(state.config)
        logger.info(f"Memory backend: {state.config.memory_backend.value}")

        # Load existing memories from persistent storage
        state.shards = await _load_from_persistent_backend()
        logger.info(f"Loaded {len(state.shards)} memories from persistent storage")

    except Exception as e:
        logger.warning(f"Persistent backend unavailable: {e}")
        logger.info("Falling back to in-memory storage")
        state.shards = _load_memory_shards()  # Use example shards as fallback

    # Initialize ML logic detector (other detectors disabled due to import issues)
    # state.codebase_indexer = CodebaseIndexer()
    # state.hallucination_detector = HallucinationDetector(state.codebase_indexer)
    # state.code_verifier = CodeVerifier()
    # state.ai_slop_detector = AISlopDetector(state.codebase_indexer)
    state.ml_logic_detector = MLLogicDetector()
    logger.info("ML logic detector initialized (other detectors disabled)")

    # Initialize agent monitoring (Nov 2025)
    if MONITORING_AVAILABLE:
        try:
            state.monitor = get_monitor()
            await start_monitoring()
            logger.info("✅ Agent monitoring initialized (real-time tracking enabled)")
        except Exception as e:
            logger.warning(f"⚠️  Agent monitoring initialization failed: {e}")
            state.monitor = None
    else:
        logger.info("Agent monitoring disabled (monitoring.py not available)")

    # Initialize SaaS backend (Dec 2025)
    if SAAS_AVAILABLE:
        try:
            state.saas_backend = await create_saas_backend()
            app.state.saas_backend = state.saas_backend  # Make available to routes
            logger.info("✅ SaaS backend initialized (SQLite)")
        except Exception as e:
            logger.warning(f"⚠️  SaaS backend initialization failed: {e}")
            state.saas_backend = None
    else:
        logger.info("SaaS backend disabled (saas module not available)")

    # Create orchestrator (lazy init - see get_orchestrator)
    logger.info("HoloLoom server ready!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down HoloLoom server...")

    # Stop agent monitoring
    if MONITORING_AVAILABLE and state.monitor:
        try:
            await stop_monitoring()
            logger.info("Agent monitoring stopped")
        except Exception as e:
            logger.warning(f"Error stopping monitoring: {e}")

    # Close SaaS backend (Dec 2025)
    if SAAS_AVAILABLE and state.saas_backend:
        try:
            await state.saas_backend.close()
            logger.info("SaaS backend closed")
        except Exception as e:
            logger.warning(f"Error closing SaaS backend: {e}")

    # Close orchestrator
    if state.orchestrator:
        await state.orchestrator.close()


# ============================================================================
# Helper Functions
# ============================================================================

def _load_memory_shards() -> List[MemoryShard]:
    """
    Load memory shards from data source (fallback when persistent backend unavailable).

    Returns:
        Example shards for development/fallback
    """
    # Example: Load from a knowledge base
    return [
        MemoryShard(
            id="example_1",
            text="HoloLoom is a neural decision-making system with multi-scale embeddings.",
            episode="hololoom_basics",
            entities=["HoloLoom", "embeddings"],
            motifs=["definition"]
        )
    ]


async def _load_from_persistent_backend() -> List[MemoryShard]:
    """
    Load memories from persistent backend (Neo4j/Qdrant).

    Returns:
        List of MemoryShard objects loaded from storage
    """
    if not state.memory_backend:
        return []

    try:
        # For HYBRID backend, retrieve all stored memories
        # Note: HYBRID backend (Neo4j + Qdrant) uses unified memory protocol
        from HoloLoom.memory.protocol import MemoryQuery

        query = MemoryQuery(
            text="",  # Empty query = retrieve all
            limit=1000  # Adjust based on your needs
        )

        result = await state.memory_backend.retrieve(query)

        # Convert Memory objects to MemoryShard objects
        shards = []
        for memory in result.memories:
            shard = MemoryShard(
                id=memory.id,
                text=memory.text,
                episode=memory.context.get("episode", "default"),
                entities=memory.context.get("entities", []),
                motifs=memory.context.get("motifs", []),
                metadata=memory.metadata
            )
            shards.append(shard)

        return shards

    except Exception as e:
        logger.error(f"Failed to load from persistent backend: {e}")
        return []


async def get_orchestrator():
    """Get or create global orchestrator instance."""
    if state.orchestrator is None:
        logger.info("Initializing agentic orchestrator...")
        state.orchestrator = await create_agentic_orchestrator(
            state.config,
            state.shards,
            enable_verification=True,
            enable_goal_tracking=True,
            audit_trail=state.audit_trail,
            monitor=state.monitor  # Agent monitoring (Nov 2025)
        )
        logger.info("Orchestrator ready!")

    return state.orchestrator


def _format_verification(verification) -> Optional[VerificationResponse]:
    """Format verification result for API response."""
    if verification is None:
        return None

    return VerificationResponse(
        verified=verification.verified,
        confidence=verification.confidence,
        contradictions=verification.contradictions,
        supporting_evidence=verification.supporting_evidence,
        suggested_refinements=verification.suggested_refinements
    )


def _format_steps(steps: List[Dict]) -> List[ReasoningStepResponse]:
    """Format reasoning steps for API response."""
    return [
        ReasoningStepResponse(
            type=step.get("type", "unknown"),
            query=step.get("query"),
            confidence=step.get("confidence"),
            finding=step.get("finding"),
            completed=step.get("completed"),
            tool=step.get("tool_used") or step.get("tool")
        )
        for step in steps
    ]


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Used by VS Code extension to verify server is running.
    """
    return {
        "status": "ok",
        "service": "HoloLoom Agentic API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query", response_model=AgenticResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint for agentic reasoning.

    Matches VS Code extension's HoloLoomBridge.query() expectations.

    Args:
        request: QueryRequest with text, context, mode, max_steps

    Returns:
        AgenticResponse with reasoning results

    Example:
        POST /query
        {
          "text": "Explain this TypeScript code",
          "context": {
            "languageId": "typescript",
            "fileName": "example.ts",
            "selection": "function foo() { ... }"
          },
          "mode": "verify",
          "max_steps": 5
        }
    """
    start_time = datetime.now()
    start_time_ms = time() * 1000  # For latency tracking

    try:
        # Validate request early to surface user errors as 4xx instead of 5xx
        text_value = (request.text or "").strip()
        if not text_value:
            raise HTTPException(status_code=400, detail="Query text must not be empty.")

        mode_map = {
            "direct": ReasoningMode.DIRECT,
            "verify": ReasoningMode.VERIFY,
            "research": ReasoningMode.RESEARCH,
            "plan_execute": ReasoningMode.PLAN_EXECUTE,
        }
        mode_key = (request.mode or "verify").strip().lower()
        if mode_key not in mode_map:
            raise HTTPException(status_code=400, detail=f"Unsupported reasoning mode: {request.mode}")

        # Get orchestrator lazily after validation
        orchestrator = await get_orchestrator()

        mode = mode_map[mode_key]

        # Create query
        query = Query(text=text_value)

        # Add code context to metadata
        if request.context:
            query.metadata = {
                "code_context": request.context.dict(),
                "language": request.context.languageId,
                "file": request.context.fileName,
                "selection": request.context.selection,
                "workspace": request.context.workspace,
            }

        # ========================================================================
        # SAFETY GATING (Alignment Framework Integration)
        # ========================================================================
        if state.safety_guardrails:
            # Import ActionCategory for safety evaluation
            from HoloLoom.alignment.safety_guardrails import ActionCategory

            # Create action request for safety evaluation
            action_request = ActionRequest(
                action="code_analysis" if request.context else "text_query",
                category=ActionCategory.QUERY if not request.context else ActionCategory.CODE_EXECUTION,
                context={
                    "query": text_value,
                    "mode": request.mode,
                    "max_steps": request.max_steps,
                    "has_code_context": request.context is not None,
                    "source": "vscode_extension",
                    "timestamp": start_time.isoformat()
                }
            )

            # Evaluate safety (synchronous method, not async)
            gate_result = state.safety_guardrails.evaluate(action_request)

            # Log safety decision
            logger.info(f"🛡️  Safety Gate: {gate_result.risk_level.value} risk "
                       f"(score={gate_result.safety_score:.2f}, allowed={gate_result.allowed})")

            # Handle high-risk or blocked actions
            if not gate_result.allowed:
                # Blocked by safety guardrails
                error_msg = (
                    f"Query blocked by safety guardrails: {gate_result.reason}. "
                    f"Risk level: {gate_result.risk_level.value} "
                    f"(safety score: {gate_result.safety_score:.2f})"
                )
                logger.warning(f"⚠️  {error_msg}")

                # Return 403 Forbidden (request valid but forbidden)
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "safety_guardrail_blocked",
                        "reason": gate_result.reason,
                        "risk_level": gate_result.risk_level.value,
                        "safety_score": gate_result.safety_score,
                        "message": error_msg
                    }
                )

            # Add safety metadata to query
            if not query.metadata:
                query.metadata = {}
            query.metadata["safety_evaluation"] = {
                "risk_level": gate_result.risk_level.value,
                "safety_score": gate_result.safety_score,
                "allowed": gate_result.allowed
            }
        # ========================================================================
        # END SAFETY GATING
        # ========================================================================

        # Run agentic reasoning (now safety-approved)
        logger.info(f"Query: {request.text[:100]}... (mode={request.mode})")
        result: AgenticResult = await orchestrator.reason(
            query,
            mode=mode,
            max_steps=request.max_steps
        )

        # Extract response text
        response_text = result.spacetime.metadata.get("response", "")
        if not response_text:
            # Fallback: use context or generate generic response
            response_text = f"Processed query with {result.reasoning_mode.value} mode."

        # Calculate latency
        end_time_ms = time() * 1000
        latency_ms = end_time_ms - start_time_ms

        # Track stats
        if state.stats:
            state.stats.record_query(request.mode, latency_ms, success=True)

        # ========================================================================
        # AUDIT TRAIL LOGGING (Alignment Framework Integration)
        # ========================================================================
        if state.audit_trail:
            try:
                # Import required enums
                from HoloLoom.alignment.audit_trail import DecisionType, OutcomeType

                # Log decision (synchronous method with correct signature)
                state.audit_trail.log_decision(
                    decision_type=DecisionType.TOOL_SELECTION,
                    outcome=OutcomeType.APPROVED,
                    reason=f"Agentic reasoning completed successfully in {request.mode} mode",
                    query_text=text_value,
                    action_description=f"agentic_reasoning_{request.mode}",
                    risk_level=query.metadata.get("safety_evaluation", {}).get("risk_level") if query.metadata else None,
                    confidence=result.spacetime.confidence,
                    metadata={
                        "code_context": request.context.dict() if request.context else None,
                        "max_steps": request.max_steps,
                        "reasoning_mode": result.reasoning_mode.value,
                        "steps_taken": len(result.steps_taken),
                        "total_queries": result.total_queries,
                        "timestamp": start_time.isoformat(),
                        "query_id": result.spacetime.query_id,
                        "latency_ms": latency_ms,
                        "verification": result.verification is not None,
                        "safety_score": query.metadata.get("safety_evaluation", {}).get("safety_score") if query.metadata else None
                    }
                )
                logger.debug(f"📝 Logged to audit trail: query_id={result.spacetime.query_id}")
            except Exception as e:
                # Audit logging should never crash the request
                logger.error(f"Failed to log to audit trail: {e}")
        # ========================================================================
        # END AUDIT TRAIL LOGGING
        # ========================================================================

        # Format response (matches TypeScript AgenticResult interface)
        response_obj = AgenticResponse(
            response=response_text,
            confidence=result.spacetime.confidence,
            reasoning_mode=result.reasoning_mode.value,
            steps_taken=_format_steps(result.steps_taken),
            total_queries=result.total_queries,
            total_duration_ms=result.total_duration_ms,
            verification=_format_verification(result.verification),
            timestamp=start_time.isoformat(),
            query_id=result.spacetime.query_id
        )

        # Add safety metadata to response logging (for transparency)
        if query.metadata and "safety_evaluation" in query.metadata:
            logger.info(f"🔒 Safety: {query.metadata['safety_evaluation']}")

        return response_obj

    except HTTPException as e:
        # HTTPException (4xx) - user error, track but don't retry
        if state.stats:
            state.stats.record_query(request.mode, 0, success=False)
            state.stats.record_error(f"HTTP_{e.status_code}")
        raise

    except Exception as e:
        # Server error (5xx) - implement retry logic
        logger.error(f"Query failed: {e}", exc_info=True)

        # Track error
        if state.stats:
            latency_ms = (time() * 1000) - start_time_ms
            state.stats.record_query(request.mode, latency_ms, success=False)
            state.stats.record_error(type(e).__name__)

        # Return error with retry suggestion
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}. Please try again."
        )


@app.get("/stats")
async def get_stats():
    """
    Get server statistics.

    Returns:
        Comprehensive statistics about server performance:
        - Uptime and query counts
        - Success/failure rates
        - Latency metrics (avg, p95)
        - Queries by mode
        - Error breakdown
        - Memory and orchestrator status
    """
    base_stats = {
        "orchestrator_ready": state.orchestrator is not None,
        "memory_shards": len(state.shards),
        "rate_limiter_enabled": state.rate_limiter is not None,
    }

    # Add comprehensive stats if available
    if state.stats:
        base_stats.update(state.stats.get_stats_dict())

    # Add audit trail info
    if state.audit_trail:
        base_stats["audit_trail_entries"] = len(state.audit_trail.logs)

    return base_stats


@app.get("/safety-stats")
async def get_safety_stats():
    """
    Get safety guardrails statistics.

    Useful for monitoring and investor demo.

    Returns:
        Dict with safety metrics
    """
    if not state.safety_guardrails:
        return {
            "enabled": False,
            "message": "Safety guardrails not initialized"
        }

    # SafetyGuardrails stats with proper testing_mode reflection
    # Safety Integration (Dec 2025) - MRF CRITIQUE fix
    import os
    testing_mode = getattr(state.safety_guardrails, 'testing_mode', False)
    env_testing_mode = os.environ.get("HOLOLOOM_TESTING_MODE", "").lower() == "true"

    if testing_mode:
        mode_message = "⚠️ TESTING MODE - approval requirements bypassed (development only)"
    else:
        mode_message = "✅ PRODUCTION MODE - full safety gating active"

    return {
        "enabled": True,
        "testing_mode": testing_mode,
        "env_testing_mode_set": env_testing_mode,
        "adversarial_detection_enabled": state.safety_guardrails.adversarial_detector is not None,
        "deception_detection_enabled": state.deception_detector is not None,
        "decisions_logged": len(state.safety_guardrails.decisions) if hasattr(state.safety_guardrails, 'decisions') else 0,
        "message": mode_message
    }


@app.get("/audit-trail")
async def get_audit_trail(limit: int = 100):
    """
    Get recent audit trail entries.

    Args:
        limit: Maximum number of entries to return

    Returns:
        Recent audit trail entries
    """
    if not state.audit_trail:
        return {"entries": []}

    # Get recent entries
    entries = state.audit_trail.logs[-limit:]

    return {
        "total": len(state.audit_trail.logs),
        "entries": [
            {
                "decision_id": log.decision_id,
                "decision_type": log.decision_type.value,
                "outcome": log.outcome.value,
                "reason": log.reason,
                "timestamp": log.timestamp.isoformat(),
                "confidence": log.confidence,
            }
            for log in entries
        ]
    }


@app.post("/memories/add")
async def add_memory(memory: MemoryAddRequest):
    """
    Add new memory to persistent storage.

    SECURITY: Input validated via MemoryAddRequest Pydantic model with:
    - Text limited to 100KB
    - Episode limited to 1KB
    - Entities/motifs limited to 100 items of 1KB each
    - Metadata limited to 10KB serialized

    Args:
        memory: MemoryAddRequest with text, episode, entities, motifs, metadata

    Returns:
        Success status and memory ID

    Example:
        POST /memories/add
        {
          "text": "Thompson Sampling balances exploration and exploitation",
          "episode": "algorithms",
          "entities": ["Thompson Sampling"],
          "motifs": ["definition"],
          "metadata": {"topic": "ML", "confidence": 0.9}
        }
    """
    try:
        if not state.memory_backend:
            return {
                "success": False,
                "message": "Persistent backend not available",
                "memory_id": None
            }

        from HoloLoom.memory.protocol import Memory

        # Create Memory object - fields already validated by Pydantic
        new_memory = Memory(
            id=f"mem_{datetime.now().timestamp()}",
            text=memory.text,
            context={
                "episode": memory.episode,
                "entities": memory.entities,
                "motifs": memory.motifs
            },
            metadata=memory.metadata
        )

        # Store in persistent backend
        await state.memory_backend.store([new_memory])

        # Also add to in-memory shards for immediate availability
        shard = MemoryShard(
            id=new_memory.id,
            text=new_memory.text,
            episode=memory.episode,
            entities=memory.entities,
            motifs=memory.motifs,
            metadata=memory.metadata
        )
        state.shards.append(shard)

        logger.info(f"Added memory: {new_memory.id}")

        return {
            "success": True,
            "message": "Memory added successfully",
            "memory_id": new_memory.id
        }

    except Exception as e:
        # SECURITY: Log full error internally, return generic message to client
        logger.error(f"Failed to add memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add memory. Please try again or contact support.")


@app.post("/api/remember")
async def api_remember(request: Dict[str, Any]):
    """
    Store content to HoloLoom memory with IDE context.

    Endpoint for Promptly VS Code extension to capture developer notes,
    decisions, and code context into the knowledge graph.

    Args:
        request: Dict with 'content' and optional 'context'
            - content (str): The content to remember
            - context (dict): IDE context (workspace, file, timestamp, etc.)

    Returns:
        Success status and memory ID

    Example:
        POST /api/remember
        {
          "content": "We decided to use PostgreSQL for authentication",
          "context": {
            "workspace": "my-project",
            "file": "src/auth.ts",
            "timestamp": "2025-11-15T10:30:00Z"
          }
        }
    """
    try:
        content = request.get("content")
        if not content:
            raise HTTPException(status_code=400, detail="Missing 'content' field")

        context = request.get("context", {})

        # Import HoloLoom unified API
        from HoloLoom import HoloLoom

        # Create HoloLoom instance with current config
        async with HoloLoom(config=state.config) as loom:
            # Experience the content (stores in knowledge graph)
            memory = await loom.experience(content, context=context)

            logger.info(f"Remembered via /api/remember: {content[:50]}...")

            return {
                "status": "success",
                "message": f"Saved to HoloLoom memory",
                "memory_id": memory.id
            }

    except Exception as e:
        logger.error(f"Failed to remember content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recall")
async def api_recall(request: Dict[str, Any]):
    """
    Search HoloLoom memories using semantic + keyword search.

    Endpoint for Promptly VS Code extension to retrieve relevant memories
    based on a query. Uses hybrid BM25 + semantic search for best results.

    Args:
        request: Dict with 'query' and optional 'k'
            - query (str): Search query
            - k (int): Number of results to return (default: 5)

    Returns:
        List of matching memories with confidence scores

    Example:
        POST /api/recall
        {
          "query": "What database did we choose?",
          "k": 5
        }

        Response:
        {
          "memories": [
            {
              "content": "We decided to use PostgreSQL for authentication",
              "confidence": 0.92,
              "timestamp": "2025-11-15T10:30:00Z",
              "source": "src/auth.ts"
            },
            ...
          ]
        }
    """
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query' field")

        k = request.get("k", 5)

        # Import HoloLoom unified API
        from HoloLoom import HoloLoom

        # Create HoloLoom instance with current config
        async with HoloLoom(config=state.config) as loom:
            # Recall memories
            memories = await loom.recall(query, k=k)

            logger.info(f"Recalled {len(memories)} memories for: {query[:50]}...")

            # Format response to match TypeScript interface
            return {
                "memories": [
                    {
                        "content": m.text,
                        "confidence": m.metadata.get("confidence", 0.85),
                        "timestamp": m.metadata.get("timestamp", "unknown"),
                        "source": m.metadata.get("file", "unknown")
                    }
                    for m in memories
                ]
            }

    except Exception as e:
        logger.error(f"Failed to recall memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/todos")
async def get_todos(workspace_id: Optional[str] = None, limit: int = 50):
    """
    Extract all TODOs from workspace with importance scoring.

    Queries HoloLoom memory for TODO/FIXME/NOTE comments and scores them
    by importance (mention frequency, recency, context).

    Args:
        workspace_id: Optional workspace filter
        limit: Maximum number of TODOs to return (default: 50)

    Returns:
        List of TODOs sorted by importance score

    Example:
        GET /api/todos?limit=20

        Response:
        {
          "todos": [
            {
              "text": "Add rate limiting to auth endpoints",
              "type": "TODO",
              "priority": "HIGH",
              "importance_score": 0.92,
              "mention_count": 3,
              "locations": ["auth.ts:15", "middleware.ts:42"],
              "related_notes": ["Security review needed", "API rate limits"]
            },
            ...
          ],
          "total_count": 47
        }
    """
    try:
        from HoloLoom import HoloLoom
        from collections import defaultdict
        import re

        # Create HoloLoom instance
        async with HoloLoom(config=state.config) as loom:
            # Query for TODO/FIXME/NOTE comments
            # Search broadly for common task markers
            all_results = []
            for keyword in ['TODO', 'FIXME', 'NOTE', 'HACK', 'XXX']:
                results = await loom.recall(keyword, k=100)
                all_results.extend(results)

            logger.info(f"Found {len(all_results)} potential TODO items")

            # Extract and group TODOs
            todo_groups = defaultdict(list)

            for memory in all_results:
                text = memory.text

                # Extract TODO/FIXME/NOTE lines
                todo_pattern = r'(TODO|FIXME|NOTE|HACK|XXX):\s*(.+?)(?:\n|$)'
                matches = re.finditer(todo_pattern, text, re.IGNORECASE)

                for match in matches:
                    todo_type = match.group(1).upper()
                    todo_text = match.group(2).strip()

                    # Get location from metadata
                    file_path = memory.metadata.get('file', 'unknown')

                    # Normalize TODO text for grouping (lowercase, strip punctuation)
                    normalized = todo_text.lower().strip('.,!?')

                    # Group similar TODOs
                    todo_groups[normalized].append({
                        'type': todo_type,
                        'text': todo_text,
                        'location': file_path,
                        'confidence': memory.metadata.get('confidence', 0.5),
                        'timestamp': memory.metadata.get('timestamp', 'unknown')
                    })

            # Score and rank TODOs
            scored_todos = []

            for normalized_text, occurrences in todo_groups.items():
                # Calculate importance score
                mention_count = len(occurrences)
                avg_confidence = sum(o['confidence'] for o in occurrences) / mention_count

                # Importance factors:
                # 1. Mention count (more mentions = more important)
                # 2. Confidence (higher confidence = more important)
                # 3. TODO type (FIXME > TODO > NOTE)
                type_weights = {
                    'FIXME': 1.0,
                    'TODO': 0.8,
                    'HACK': 0.7,
                    'XXX': 0.7,
                    'NOTE': 0.5
                }

                todo_type = occurrences[0]['type']
                type_weight = type_weights.get(todo_type, 0.5)

                # Importance score formula
                importance_score = (
                    (mention_count / 10) * 0.4 +  # Mention frequency (normalized)
                    avg_confidence * 0.3 +          # Average confidence
                    type_weight * 0.3               # Type weight
                )
                importance_score = min(1.0, importance_score)  # Cap at 1.0

                # Determine priority
                if importance_score >= 0.75:
                    priority = 'HIGH'
                elif importance_score >= 0.5:
                    priority = 'MEDIUM'
                else:
                    priority = 'LOW'

                # Get unique locations
                locations = list(set(o['location'] for o in occurrences))

                scored_todos.append({
                    'text': occurrences[0]['text'],  # Use original text (not normalized)
                    'type': todo_type,
                    'priority': priority,
                    'importance_score': round(importance_score, 2),
                    'mention_count': mention_count,
                    'locations': locations[:5],  # Limit to 5 locations
                    'related_notes': []  # Could add related memories here
                })

            # Sort by importance score (descending)
            scored_todos.sort(key=lambda x: x['importance_score'], reverse=True)

            # Limit results
            scored_todos = scored_todos[:limit]

            return {
                'todos': scored_todos,
                'total_count': len(todo_groups)
            }

    except Exception as e:
        logger.error(f"Failed to extract TODOs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/workspace")
async def ingest_workspace(
    workspace_path: str,
    languages: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
):
    """
    Ingest entire workspace into knowledge graph.

    Args:
        workspace_path: Path to workspace root
        languages: Languages to index (e.g., ["python", "typescript"])
        exclude_patterns: Glob patterns to exclude

    Returns:
        Statistics about ingestion

    Example:
        POST /ingest/workspace
        {
          "workspace_path": "/path/to/project",
          "languages": ["python", "typescript"],
          "exclude_patterns": ["**/node_modules/**", "**/.venv/**"]
        }
    """
    try:
        from pathlib import Path
        from HoloLoom.spinningWheel.workspace import WorkspaceSpinner
        from HoloLoom import HoloLoom

        # SECURITY: Path traversal validation
        # Get allowed base directories from environment (comma-separated)
        import os
        allowed_bases_str = os.environ.get("ALLOWED_WORKSPACE_PATHS", "")
        if allowed_bases_str:
            allowed_bases = [Path(p.strip()).resolve() for p in allowed_bases_str.split(",") if p.strip()]
        else:
            # Default: Allow current working directory and home directory
            allowed_bases = [Path.cwd().resolve(), Path.home().resolve()]

        # Resolve the requested path
        requested_path = Path(workspace_path).resolve()

        # Validate path is within allowed bases
        path_allowed = False
        for base in allowed_bases:
            try:
                requested_path.relative_to(base)
                path_allowed = True
                break
            except ValueError:
                continue

        if not path_allowed:
            logger.warning(f"SECURITY: Path traversal attempt blocked: {workspace_path}")
            raise HTTPException(
                status_code=403,
                detail="Path not allowed. Set ALLOWED_WORKSPACE_PATHS environment variable."
            )

        # Additional safety: reject paths with suspicious patterns
        path_str = str(requested_path)
        suspicious_patterns = ["/etc/", "/var/", "/root/", "\\Windows\\", "\\System32\\"]
        for pattern in suspicious_patterns:
            if pattern.lower() in path_str.lower():
                logger.warning(f"SECURITY: Suspicious path blocked: {workspace_path}")
                raise HTTPException(status_code=403, detail="Access to system directories not allowed")

        logger.info(f"Starting workspace ingestion: {requested_path}")

        # Create workspace spinner
        spinner = WorkspaceSpinner()

        # Scan workspace (use validated path)
        shards = await spinner.spin_workspace(
            workspace_path=str(requested_path),
            languages=languages,
            exclude_patterns=exclude_patterns
        )

        logger.info(f"Created {len(shards)} shards from workspace")

        # Store shards in HoloLoom
        async with HoloLoom(config=state.config) as loom:
            for shard in shards:
                await loom.experience(
                    content=shard.text,
                    context=shard.metadata
                )

        # Also add to server's in-memory shards
        state.shards.extend(shards)

        # Calculate statistics
        total_files = len(shards)
        total_elements = sum(shard.metadata.get('element_count', 0) for shard in shards)
        total_comments = sum(shard.metadata.get('comment_count', 0) for shard in shards)
        total_todos = sum(shard.metadata.get('todo_count', 0) for shard in shards)

        logger.info(f"Workspace ingestion complete: {total_files} files, {total_elements} elements, {total_todos} TODOs")

        return {
            "success": True,
            "files_indexed": total_files,
            "code_elements": total_elements,
            "comments": total_comments,
            "todos": total_todos,
            "workspace_path": str(requested_path)  # SECURITY: Return validated path
        }

    except Exception as e:
        # SECURITY: Return generic error message, log details internally
        logger.error(f"Workspace ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Workspace ingestion failed")


# Legacy code (commented out - replaced by WorkspaceSpinner above)
@app.post("/ingest/workspace/legacy")
async def ingest_workspace_legacy(
    workspace_path: str,
    languages: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
):
    """
    Legacy workspace ingestion using codebase indexer (if available).

    This endpoint is kept for backward compatibility but the main
    /ingest/workspace endpoint now uses WorkspaceSpinner.
    """
    try:
        if not state.codebase_indexer:
            raise HTTPException(status_code=500, detail="Codebase indexer not initialized")

        # Convert language strings to Language enum
        lang_map = {
            "python": CodeLanguage.PYTHON,
            "typescript": CodeLanguage.TYPESCRIPT,
            "javascript": CodeLanguage.JAVASCRIPT,
            "java": CodeLanguage.JAVA,
            "cpp": CodeLanguage.CPP,
            "rust": CodeLanguage.RUST,
            "go": CodeLanguage.GO
        }

        parsed_languages = None
        if languages:
            parsed_languages = [lang_map.get(lang.lower()) for lang in languages]
            parsed_languages = [l for l in parsed_languages if l is not None]

        logger.info(f"Starting workspace ingestion: {workspace_path}")
        stats = await state.codebase_indexer.ingest_workspace(
            workspace_path,
            languages=parsed_languages,
            exclude_patterns=exclude_patterns
        )

        # Convert indexed code to memory shards for HoloLoom
        code_shards = state.codebase_indexer.to_memory_shards()
        logger.info(f"Created {len(code_shards)} code memory shards")

        # Add to server's memory
        state.shards.extend(code_shards)

        # If persistent backend available, store shards
        if state.memory_backend:
            from HoloLoom.memory.protocol import Memory
            memories = []
            for shard in code_shards:
                memory = Memory(
                    id=shard.id,
                    text=shard.text,
                    context={
                        "episode": shard.episode,
                        "entities": shard.entities,
                        "motifs": shard.motifs
                    },
                    metadata=shard.metadata
                )
                memories.append(memory)

            await state.memory_backend.store(memories)
            logger.info(f"Stored {len(memories)} code memories to persistent backend")

        # Invalidate orchestrator so it gets recreated with new shards
        if state.orchestrator:
            await state.orchestrator.close()
            state.orchestrator = None

        return {
            "success": True,
            "ingestion_stats": stats,
            "memory_shards_created": len(code_shards),
            "message": f"Ingested {stats['files_processed']} files with {stats['entities_found']} entities"
        }

    except Exception as e:
        logger.error(f"Workspace ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file_path: str, language: str, content: Optional[str] = None):
    """
    Ingest single file into knowledge graph.

    Args:
        file_path: Path to file
        language: Programming language
        content: File content (if None, reads from file_path)

    Returns:
        Parsed entities and statistics

    Example:
        POST /ingest/file
        {
          "file_path": "/path/to/file.py",
          "language": "python",
          "content": "def foo(): pass"
        }
    """
    try:
        if not state.codebase_indexer:
            raise HTTPException(status_code=500, detail="Codebase indexer not initialized")

        # SECURITY: Validate language whitelist (prevents malicious file types)
        lang_map = {
            "python": CodeLanguage.PYTHON,
            "typescript": CodeLanguage.TYPESCRIPT,
            "javascript": CodeLanguage.JAVASCRIPT
        }

        lang_enum = lang_map.get(language.lower())
        if not lang_enum:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}. Allowed: python, typescript, javascript")

        # SECURITY: Path traversal validation (when reading from filesystem)
        validated_path = None
        if not content:
            from pathlib import Path

            # Get allowed base directories from environment (comma-separated)
            allowed_bases_str = os.environ.get("ALLOWED_WORKSPACE_PATHS", "")
            if allowed_bases_str:
                allowed_bases = [Path(p.strip()).resolve() for p in allowed_bases_str.split(",") if p.strip()]
            else:
                # Default: Allow current working directory and home directory
                allowed_bases = [Path.cwd().resolve(), Path.home().resolve()]

            # Resolve the requested path
            requested_path = Path(file_path).resolve()

            # Validate path is within allowed bases
            path_allowed = False
            for base in allowed_bases:
                try:
                    requested_path.relative_to(base)
                    path_allowed = True
                    break
                except ValueError:
                    continue

            if not path_allowed:
                logger.warning(f"SECURITY: Path traversal attempt blocked: {file_path}")
                raise HTTPException(
                    status_code=403,
                    detail="Path not allowed. Set ALLOWED_WORKSPACE_PATHS environment variable."
                )

            # Additional safety: reject paths with suspicious patterns
            path_str = str(requested_path)
            suspicious_patterns = ["/etc/", "/var/", "/root/", "\\Windows\\", "\\System32\\", ".ssh", ".aws", ".env"]
            for pattern in suspicious_patterns:
                if pattern.lower() in path_str.lower():
                    logger.warning(f"SECURITY: Suspicious path blocked: {file_path}")
                    raise HTTPException(status_code=403, detail="Access to system/sensitive directories not allowed")

            validated_path = requested_path

        # If content provided, create temp file with secure handling
        if content:
            import tempfile
            # SECURITY: Use proper suffix based on validated language
            suffix_map = {"python": ".py", "typescript": ".ts", "javascript": ".js"}
            suffix = suffix_map.get(language.lower(), ".txt")

            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                f.write(content)
                temp_path = f.name

            try:
                code_file = await state.codebase_indexer.ingest_file(temp_path, lang_enum)
            finally:
                # Always clean up temp file
                os.unlink(temp_path)
        else:
            code_file = await state.codebase_indexer.ingest_file(str(validated_path), lang_enum)

        return {
            "success": True,
            "file_path": str(validated_path) if validated_path else file_path,
            "language": language,
            "entities_found": len(code_file.entities),
            "imports_found": len(code_file.imports),
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type.value,
                    "line": e.line_number,
                    "signature": e.signature
                }
                for e in code_file.entities
            ]
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # SECURITY: Return generic error message, log details internally
        logger.error(f"File ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="File ingestion failed")


@app.post("/detect/slop")
async def detect_ai_slop(code: str, language: str, file_path: str = "temp"):
    """
    Comprehensive AI slop detection (all common pitfalls).

    Detects:
    - Hallucinations (non-existent functions/classes)
    - Missing error handling
    - Hardcoded secrets/magic numbers
    - Resource leaks
    - Security issues (SQL injection, XSS, command injection)
    - Performance anti-patterns
    - Dead code
    - Naming inconsistencies
    - Missing documentation
    - Incomplete code (TODO, pass statements)
    - Off-by-one errors
    - Timezone issues
    - Copy-paste errors

    Args:
        code: Source code to analyze
        language: Programming language
        file_path: File path for context

    Returns:
        Comprehensive list of all detected issues with fixes

    Example:
        POST /detect/slop
        {
          "code": "def process(data):\\n    result = fetch_data()\\n    return result",
          "language": "python",
          "file_path": "process.py"
        }
    """
    try:
        if not state.ai_slop_detector:
            raise HTTPException(status_code=500, detail="AI slop detector not initialized")

        # Map language string to enum
        lang_map = {
            "python": CodeLanguage.PYTHON,
            "typescript": CodeLanguage.TYPESCRIPT,
            "javascript": CodeLanguage.JAVASCRIPT
        }

        lang_enum = lang_map.get(language.lower())
        if not lang_enum:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        # Detect all issues
        issues = await state.ai_slop_detector.detect_all(code, lang_enum, file_path)
        summary = await state.ai_slop_detector.get_summary(issues)

        return {
            "success": True,
            "language": language,
            "total_issues": len(issues),
            "summary": summary,
            "issues": [
                {
                    "category": issue.category.value,
                    "severity": issue.severity.value,
                    "line": issue.line_number,
                    "column": issue.column,
                    "description": issue.description,
                    "context": issue.context[:200],  # Truncate context
                    "fix": issue.fix_suggestion,
                    "confidence": issue.confidence
                }
                for issue in issues
            ]
        }

    except Exception as e:
        logger.error(f"AI slop detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/logic")
async def detect_logic_errors(code: str, language: str, file_path: str = "temp"):
    """
    ML-based logic error detection.

    Detects subtle logic errors that pattern matching can't catch:
    - Infinite loops (loops with no exit condition)
    - Unreachable code (code after return/break/continue)
    - Logic contradictions (if x and not x)
    - Null/None dereference
    - Division by zero
    - Array out of bounds
    - Missing return statements
    - Constant conditions (always true/false)
    - Wrong operators (= instead of ==)

    Uses hybrid approach:
    - Control flow graph analysis
    - Abstract interpretation for value tracking
    - Symbolic execution for proofs
    - ML model for pattern recognition (future)

    Args:
        code: Source code to analyze
        language: Programming language (python, typescript, javascript)
        file_path: File path for context

    Returns:
        List of logic errors with confidence scores and proofs

    Example:
        POST /detect/logic
        {
          "code": "def divide(a, b):\\n    return a / b",
          "language": "python",
          "file_path": "math_utils.py"
        }

        Response:
        {
          "success": true,
          "language": "python",
          "total_errors": 1,
          "summary": {
            "total_errors": 1,
            "by_type": {"division_by_zero": 1},
            "high_confidence": [
              {
                "type": "division_by_zero",
                "line": 2,
                "description": "Potential division by zero - b not checked"
              }
            ],
            "proven": []
          },
          "errors": [...]
        }
    """
    try:
        if not state.ml_logic_detector:
            raise HTTPException(status_code=500, detail="ML logic detector not initialized")

        # Map language string to enum
        lang_map = {
            "python": CodeLanguage.PYTHON,
            "typescript": CodeLanguage.TYPESCRIPT,
            "javascript": CodeLanguage.JAVASCRIPT
        }

        lang_enum = lang_map.get(language.lower())
        if not lang_enum:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        # Detect logic errors
        errors = await state.ml_logic_detector.detect(code, lang_enum, file_path)
        summary = await state.ml_logic_detector.get_summary(errors)

        return {
            "success": True,
            "language": language,
            "total_errors": len(errors),
            "summary": summary,
            "errors": [
                {
                    "type": error.error_type.value,
                    "line": error.line_number,
                    "column": error.column,
                    "description": error.description,
                    "context": error.context[:200],  # Truncate context
                    "confidence": error.confidence,
                    "fix": error.fix_suggestion,
                    "proof": error.proof
                }
                for error in errors
            ]
        }

    except Exception as e:
        logger.error(f"ML logic detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/hallucinations")
async def detect_hallucinations(code: str, language: str, file_path: str = "temp", strict: bool = False):
    """
    Detect hallucinations in AI-generated code.

    Args:
        code: Source code to analyze
        language: Programming language
        file_path: File path for context
        strict: If True, flag all unrecognized references

    Returns:
        List of hallucinations with suggestions

    Example:
        POST /detect/hallucinations
        {
          "code": "def main():\\n    result = nonexistent_function()\\n    return result",
          "language": "python",
          "strict": false
        }
    """
    try:
        if not state.hallucination_detector:
            raise HTTPException(status_code=500, detail="Hallucination detector not initialized")

        # Map language string to enum
        lang_map = {
            "python": CodeLanguage.PYTHON,
            "typescript": CodeLanguage.TYPESCRIPT,
            "javascript": CodeLanguage.JAVASCRIPT
        }

        lang_enum = lang_map.get(language.lower())
        if not lang_enum:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        result = await state.hallucination_detector.detect_and_explain(
            code,
            lang_enum,
            file_path
        )

        return {
            "success": True,
            "language": language,
            "hallucinations": result["hallucinations"],
            "summary": result["summary"]
        }

    except Exception as e:
        logger.error(f"Hallucination detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/codebase/stats")
async def get_codebase_stats():
    """
    Get statistics about indexed codebase.

    Returns:
        Statistics about entities, files, relationships
    """
    if not state.codebase_indexer:
        return {"error": "Codebase indexer not initialized"}

    return state.codebase_indexer.get_statistics()


@app.post("/verify/code")
async def verify_code(
    code: str,
    language: str,
    file_path: str = "temp",
    check_syntax: bool = True,
    check_types: bool = True,
    check_lint: bool = False
):
    """
    Verify code for errors using compilers and linters.

    Args:
        code: Source code to verify
        language: Programming language
        file_path: File path for context
        check_syntax: Run syntax checker
        check_types: Run type checker
        check_lint: Run linter

    Returns:
        Comprehensive verification results

    Example:
        POST /verify/code
        {
          "code": "def foo():\n    return bar()",
          "language": "python",
          "check_syntax": true,
          "check_types": true,
          "check_lint": false
        }
    """
    try:
        if not state.code_verifier:
            raise HTTPException(status_code=500, detail="Code verifier not initialized")

        result = await state.code_verifier.verify_comprehensive(
            code,
            language,
            file_path
        )

        return {
            "success": True,
            "verification": result
        }

    except Exception as e:
        logger.error(f"Code verification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/codebase/search")
async def search_codebase(
    query: str,
    entity_type: Optional[str] = None,
    fuzzy: bool = True,
    limit: int = 10
):
    """
    Search indexed codebase for entities.

    Args:
        query: Entity name to search for
        entity_type: Filter by type ("function", "class", "method", etc.)
        fuzzy: Allow fuzzy matching
        limit: Maximum results

    Returns:
        Matching entities with metadata

    Example:
        POST /codebase/search
        {
          "query": "authenticate",
          "entity_type": "function",
          "fuzzy": true,
          "limit": 5
        }
    """
    try:
        if not state.codebase_indexer:
            raise HTTPException(status_code=500, detail="Codebase indexer not initialized")

        # Map entity type string
        type_map = {
            "function": "function",
            "class": "class",
            "method": "method",
            "variable": "variable",
            "import": "import",
            "module": "module"
        }

        entity_type_enum = None
        if entity_type:
            from HoloLoom.agentic.codebase_ingestion import EntityType
            mapped_type = type_map.get(entity_type.lower())
            if mapped_type:
                entity_type_enum = EntityType(mapped_type)

        results = await state.codebase_indexer.search_entity(
            query,
            entity_type=entity_type_enum,
            fuzzy=fuzzy
        )

        # Limit results
        results = results[:limit]

        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Codebase search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Graph Visualization
# ============================================================================

@app.get("/api/graph/html")
async def get_graph_html(
    max_nodes: int = 50,
    title: str = "HoloLoom Knowledge Graph",
    highlight_recent: bool = True
):
    """
    Get knowledge graph as interactive HTML visualization.

    Args:
        max_nodes: Maximum nodes to include (default: 50)
        title: Graph title
        highlight_recent: Highlight recently accessed nodes

    Returns:
        HTML document with D3.js force-directed graph

    Example:
        GET /api/graph/html?max_nodes=100&title=My%20Knowledge

    Integration:
        // In VS Code webview
        const response = await fetch('http://localhost:8000/api/graph/html');
        const html = await response.text();
        webview.html = html;
    """
    from fastapi.responses import HTMLResponse
    from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_kg
    from HoloLoom import HoloLoom

    try:
        # Get knowledge graph from HoloLoom
        async with HoloLoom(config=state.config) as loom:
            # Access the knowledge graph
            kg = loom.memory_manager.knowledge_graph

            # Optional: Get recently accessed nodes for highlighting
            highlighted_path = None
            if highlight_recent and hasattr(loom, 'awareness_graph'):
                # Get top 5 most recently activated nodes
                recent_nodes = loom.awareness_graph.get_top_activated(limit=5)
                if recent_nodes:
                    highlighted_path = [node for node, _ in recent_nodes]

            # Render to HTML
            subtitle = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            html = render_knowledge_graph_from_kg(
                kg,
                title=title,
                subtitle=subtitle,
                max_nodes=max_nodes,
                highlighted_path=highlighted_path
            )

            return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"Graph visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/data")
async def get_graph_data(
    max_nodes: int = 50,
    include_metadata: bool = False
):
    """
    Get knowledge graph as JSON data.

    Args:
        max_nodes: Maximum nodes to include
        include_metadata: Include full node/edge metadata

    Returns:
        JSON with nodes and edges arrays

    Example:
        GET /api/graph/data?max_nodes=100

        Response:
        {
          "nodes": [
            {"id": "Thompson Sampling", "label": "Thompson Sampling", "degree": 5, "type": "concept"},
            ...
          ],
          "edges": [
            {"src": "Thompson Sampling", "dst": "exploration", "type": "USES", "weight": 1.0},
            ...
          ],
          "metadata": {
            "total_nodes": 47,
            "total_edges": 128,
            "rendered_nodes": 50,
            "rendered_edges": 95
          }
        }
    """
    from HoloLoom import HoloLoom

    try:
        async with HoloLoom(config=state.config) as loom:
            kg = loom.memory_manager.knowledge_graph

            # Get all nodes (limit to max_nodes)
            all_nodes = list(kg.G.nodes())[:max_nodes]
            node_id_set = set(all_nodes)

            # Build nodes array
            nodes = []
            for node_id in all_nodes:
                node_data = kg.G.nodes.get(node_id, {})
                degree = kg.G.degree(node_id)

                node_obj = {
                    "id": node_id,
                    "label": node_id,
                    "degree": degree,
                    "type": node_data.get('node_type', 'default')
                }

                if include_metadata:
                    node_obj["metadata"] = node_data

                nodes.append(node_obj)

            # Build edges array
            edges = []
            for src, dst, key, data in kg.G.edges(keys=True, data=True):
                if src in node_id_set and dst in node_id_set:
                    edge_obj = {
                        "src": src,
                        "dst": dst,
                        "type": data.get('type', 'unknown'),
                        "weight": data.get('weight', 1.0)
                    }

                    if include_metadata:
                        edge_obj["metadata"] = data

                    edges.append(edge_obj)

            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "total_nodes": kg.G.number_of_nodes(),
                    "total_edges": kg.G.number_of_edges(),
                    "rendered_nodes": len(nodes),
                    "rendered_edges": len(edges)
                }
            }

    except Exception as e:
        logger.error(f"Graph data export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# REST Monitoring Endpoints (Agent Monitoring - Nov 2025)
# ============================================================================

@app.get("/api/monitor/sessions")
async def get_monitor_sessions():
    """
    Get all active agent sessions.

    Returns:
        JSON with list of all active sessions and count

    Example:
        GET /api/monitor/sessions

        Response:
        {
            "sessions": [
                {
                    "agent_id": "agent_abc123",
                    "project": "mythRL",
                    "query": "Explain Thompson Sampling",
                    "mode": "research",
                    "status": "running",
                    "current_step": 2,
                    "total_steps": 5,
                    "feed_line1": "Research query 2/5",
                    "feed_line2": "Exploring tradeoffs..."
                },
                ...
            ],
            "count": 3
        }
    """
    if not MONITORING_AVAILABLE or not state.monitor:
        raise HTTPException(status_code=503, detail="Monitoring unavailable")

    try:
        sessions = state.monitor.get_all_sessions()

        # Serialize sessions
        session_list = []
        for session in sessions:
            session_list.append({
                "agent_id": session.agent_id,
                "project": session.project,
                "query": session.query,
                "mode": session.mode,
                "status": session.status.value,
                "current_step": session.current_step,
                "total_steps": session.total_steps,
                "feed_line1": session.feed_line1,
                "feed_line2": session.feed_line2,
                "files": session.files,
                "start_time": session.start_time,
                "total_duration_ms": session.total_duration_ms
            })

        return {
            "sessions": session_list,
            "count": len(session_list)
        }

    except Exception as e:
        logger.error(f"Failed to get monitor sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monitor/sessions/{agent_id}")
async def get_monitor_session(agent_id: str):
    """
    Get specific agent session with full tree structure.

    Args:
        agent_id: Agent identifier

    Returns:
        JSON with session details and reasoning tree

    Example:
        GET /api/monitor/sessions/agent_abc123

        Response:
        {
            "agent_id": "agent_abc123",
            "project": "mythRL",
            "status": "completed",
            "tree": {
                "node_id": "agent_abc123_step_0",
                "step_type": "initial_answer",
                "children": [...]
            },
            ...
        }
    """
    if not MONITORING_AVAILABLE or not state.monitor:
        raise HTTPException(status_code=503, detail="Monitoring unavailable")

    try:
        session = state.monitor.get_session(agent_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {agent_id} not found")

        # Serialize session with tree
        return {
            "agent_id": session.agent_id,
            "project": session.project,
            "query": session.query,
            "mode": session.mode,
            "status": session.status.value,
            "current_step": session.current_step,
            "total_steps": session.total_steps,
            "feed_line1": session.feed_line1,
            "feed_line2": session.feed_line2,
            "files": session.files,
            "start_time": session.start_time,
            "total_duration_ms": session.total_duration_ms,
            "tree": state.monitor._serialize_tree(session),
            "metadata": session.metadata
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monitor/projects")
async def get_monitor_projects():
    """
    Get list of all active projects.

    Returns:
        JSON with list of project names and count

    Example:
        GET /api/monitor/projects

        Response:
        {
            "projects": ["mythRL", "squad", "elle"],
            "count": 3
        }
    """
    if not MONITORING_AVAILABLE or not state.monitor:
        raise HTTPException(status_code=503, detail="Monitoring unavailable")

    try:
        projects = list(state.monitor.projects.keys())

        return {
            "projects": projects,
            "count": len(projects)
        }

    except Exception as e:
        logger.error(f"Failed to get monitor projects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monitor/projects/{project}")
async def get_monitor_project_agents(project: str):
    """
    Get all agents for a specific project.

    Args:
        project: Project name

    Returns:
        JSON with project name and agent sessions

    Example:
        GET /api/monitor/projects/mythRL

        Response:
        {
            "project": "mythRL",
            "agents": [
                {
                    "agent_id": "agent_abc123",
                    "query": "Explain Thompson Sampling",
                    "status": "running",
                    ...
                },
                ...
            ],
            "count": 2
        }
    """
    if not MONITORING_AVAILABLE or not state.monitor:
        raise HTTPException(status_code=503, detail="Monitoring unavailable")

    try:
        sessions = state.monitor.get_project_agents(project)

        # Serialize sessions
        agent_list = []
        for session in sessions:
            agent_list.append({
                "agent_id": session.agent_id,
                "query": session.query,
                "mode": session.mode,
                "status": session.status.value,
                "current_step": session.current_step,
                "total_steps": session.total_steps,
                "feed_line1": session.feed_line1,
                "feed_line2": session.feed_line2,
                "files": session.files,
                "start_time": session.start_time,
                "total_duration_ms": session.total_duration_ms
            })

        return {
            "project": project,
            "agents": agent_list,
            "count": len(agent_list)
        }

    except Exception as e:
        logger.error(f"Failed to get agents for project {project}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monitor/metrics")
async def get_monitor_metrics():
    """
    Get performance metrics for agent monitoring.

    Returns:
        JSON with performance statistics

    Example:
        GET /api/monitor/metrics

        Response:
        {
            "total_agents_started": 42,
            "total_agents_completed": 38,
            "total_agents_failed": 4,
            "active_agents": 3,
            "avg_latency_ms": 325.7,
            "success_rate": 0.90,
            "projects": ["mythRL", "squad", "elle"],
            "ws_connections": 2
        }
    """
    if not MONITORING_AVAILABLE or not state.monitor:
        raise HTTPException(status_code=503, detail="Monitoring unavailable")

    try:
        metrics = state.monitor.get_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Failed to get monitor metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoints (Agent Monitoring - Nov 2025)
# ============================================================================

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent monitoring.

    Streams agent status updates, step completions, and feed updates.

    Message Types:
        - agent_started: New agent began reasoning
        - agent_step: Agent completed a reasoning step
        - agent_status: Agent status changed (RUNNING/WAITING/COMPLETED/FAILED)
        - agent_feed: Two-line feed update
        - agent_completed: Agent finished with tree structure
        - agent_failed: Agent encountered error

    Example Client (TypeScript):
        const ws = new WebSocket('ws://localhost:8000/ws/monitor');
        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            console.log('Agent update:', message);
        };
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected: {websocket.client}")

    if not MONITORING_AVAILABLE or not state.monitor:
        await websocket.send_json({
            "type": "error",
            "message": "Agent monitoring not available"
        })
        await websocket.close()
        return

    # Register WebSocket connection
    state.monitor.register_ws_connection(websocket)
    logger.info(f"WebSocket registered with monitor ({len(state.monitor.ws_connections)} total)")

    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "message": "Real-time agent monitoring active",
            "active_agents": len(state.monitor.sessions),
            "projects": list(state.monitor.projects.keys())
        })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (ping, etc.)
                data = await websocket.receive_text()

                # Handle client requests
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data == "get_sessions":
                    # Send current sessions
                    sessions = []
                    for session in state.monitor.get_all_sessions():
                        sessions.append({
                            "agent_id": session.agent_id,
                            "project": session.project,
                            "query": session.query,
                            "mode": session.mode,
                            "status": session.status.value,
                            "current_step": session.current_step,
                            "total_steps": session.total_steps
                        })
                    await websocket.send_json({
                        "type": "sessions",
                        "sessions": sessions
                    })
                elif data == "get_metrics":
                    # Send performance metrics
                    metrics = state.monitor.get_metrics()
                    await websocket.send_json({
                        "type": "metrics",
                        "metrics": metrics
                    })

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    finally:
        # Unregister on disconnect
        state.monitor.unregister_ws_connection(websocket)
        logger.info(f"WebSocket unregistered ({len(state.monitor.ws_connections)} remaining)")


# ============================================================================
# Main (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting HoloLoom Agentic API server (development mode)...")
    uvicorn.run(
        "HoloLoom.server.agentic_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

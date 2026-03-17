#!/usr/bin/env python3
"""
HoloLoom Agentic API Server
============================
FastAPI server for agentic intelligence integration with VS Code Squad extension.

Provides HTTP endpoints that match the TypeScript interfaces in squad/src/HoloLoomBridge.ts

Usage:
    # Development
    uvicorn HoloLoom.apps.server.agentic_api:app --reload --port 8000

    # Production
    uvicorn HoloLoom.apps.server.agentic_api:app --host 0.0.0.0 --port 8000 --workers 4
"""

import asyncio
import logging
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from hololoom.apps.server.api.middleware.rate_limiter import RateLimiter, ServerStats
from hololoom.agentic.ml_logic_detector import MLLogicDetector
from hololoom.config import Config
from hololoom.protocols.types import MemoryShard
from hololoom.alignment.audit_trail import AuditTrail
from hololoom.alignment.safety_guardrails import SafetyGuardrails
from hololoom.alignment.deception_detection import DeceptionDetector

# SaaS API Components (Dec 2025)
try:
    from hololoom.saas import create_saas_backend, SaaSBackend
    from hololoom.saas.routes import customers_router, api_keys_router
    from hololoom.saas.auth import add_rate_limit_headers
    SAAS_AVAILABLE = True
except ImportError as e:
    SAAS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"SaaS components unavailable: {e}")

# Agent Monitoring (Nov 2025)
try:
    from hololoom.agentic.monitoring import get_monitor, start_monitoring, stop_monitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("Agent monitoring unavailable (monitoring.py not found)")

# Continuous Validator (Phase 5 - Routing & RBAC)
try:
    from hololoom.routing.learning.continuous_validator import ContinuousValidator
    from hololoom.routing.query_classifier_moonshot import MoonshotQueryClassifier
    CONTINUOUS_VALIDATOR_AVAILABLE = True
except ImportError:
    CONTINUOUS_VALIDATOR_AVAILABLE = False

# Policy Governance / RBAC (Phase 5 - Routing & RBAC)
try:
    from hololoom.agents.policy_governance import (
        GovernancePolicy, PolicyDecision, PolicyEngine,
        RoleBasedAccessControl, TopicGovernance, PolicyTemplates
    )
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response Models (extracted to schemas.py)
# Shared server state (extracted to server_state.py)
from hololoom.apps.server.server_state import state


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
    from hololoom.apps.chatops.handlers.prometheus_metrics import (
        create_metrics_router, get_metrics_collector
    )
    _metrics_router = create_metrics_router(get_metrics_collector())
    if _metrics_router:
        app.include_router(_metrics_router, prefix="/api")
        logger.info("Prometheus metrics router mounted at /api/metrics")
except ImportError:
    logger.debug("Prometheus metrics not available (missing dependencies)")

try:
    from hololoom.apps.chatops.handlers.websocket_progress import (
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

# Extracted Modular Routers (Dec 2025 - W2 SWOT Remediation)
try:
    from hololoom.apps.server.routers import (
        health_router,
        memory_router,
        detection_router,
        monitor_router,
    )
    app.include_router(health_router)
    app.include_router(memory_router)
    app.include_router(detection_router)
    app.include_router(monitor_router)
    logger.info("Modular routers mounted (health, memory, detection, monitor)")
except ImportError as e:
    logger.warning(f"Failed to mount modular routers: {e}")

# Extracted endpoint routers (March 2026 Refactor)
try:
    from hololoom.apps.server.routers.query import router as query_router
    from hololoom.apps.server.routers.ingest import router as ingest_router
    from hololoom.apps.server.routers.codebase import router as codebase_router
    from hololoom.apps.server.routers.graph import router as graph_router
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(codebase_router)
    app.include_router(graph_router)
    logger.info("Extracted routers mounted (query, ingest, codebase, graph)")
except ImportError as e:
    logger.warning(f"Failed to mount extracted routers: {e}")

# Promptly Chat (conversational endpoint with soul + memory)
try:
    from hololoom.apps.server.promptly_chat import router as promptly_chat_router
    app.include_router(promptly_chat_router)
    logger.info("Promptly chat router mounted at /promptly/chat")
except ImportError as e:
    logger.warning(f"Failed to mount Promptly chat router: {e}")

# Elle Chat (calm operational intelligence with soul + memory)
try:
    from hololoom.apps.server.elle_chat import router as elle_chat_router
    app.include_router(elle_chat_router)
    logger.info("Elle chat router mounted at /elle/chat")
except ImportError as e:
    logger.warning(f"Failed to mount Elle chat router: {e}")

# Coz Query API (Company Operating System — structured business data access)
try:
    from hololoom.apps.server.coz_api import router as coz_router
    app.include_router(coz_router)
    logger.info("Coz query router mounted at /coz/*")
except ImportError as e:
    logger.warning(f"Failed to mount Coz query router: {e}")

# Memory Bus (cross-agent observation store)
try:
    from hololoom.apps.server.memory_bus import router as memory_bus_router
    app.include_router(memory_bus_router)
    logger.info("Memory bus router mounted at /memory/*")
except ImportError as e:
    logger.warning(f"Failed to mount memory bus router: {e}")

# Coz Proactive (scheduled insight generation)
try:
    from hololoom.apps.server.coz_proactive import router as coz_proactive_router
    app.include_router(coz_proactive_router)
    logger.info("Coz proactive router mounted at /coz/proactive/*")
except ImportError as e:
    logger.warning(f"Failed to mount Coz proactive router: {e}")

# Department Federation (MCP inter-department routing over HTTP)
try:
    from hololoom.apps.server.department_api import router as department_router
    app.include_router(department_router)
    logger.info("Department federation router mounted at /departments/*")
except ImportError as e:
    logger.warning(f"Failed to mount department federation router: {e}")

# Jenny Generative UI (adaptive visualization runtime)
try:
    from hololoom.apps.server.jenny_api import router as jenny_router, shutdown_runtime as jenny_shutdown
    app.include_router(jenny_router)
    logger.info("Jenny UI router mounted at /jenny/*")
except ImportError as e:
    jenny_shutdown = None
    logger.warning(f"Failed to mount Jenny router: {e}")

# Spatial WebSocket (Stage 3 — conversation visualization in AR/XR)
import os as _os
if _os.environ.get("PROMPTLY_JENNY_SPATIAL", "").lower() == "true":
    try:
        from hololoom.apps.server.spatial_websocket import setup_spatial_routes
        setup_spatial_routes(app)
        logger.info("Spatial WebSocket mounted at /ws/spatial/{room_id}")
    except ImportError as e:
        logger.debug("Spatial WebSocket not available: %s", e)

# Spatial Inspector (Stage 4 — debug consumer for spatial WebSocket)
from pathlib import Path as _Path
_inspector_path = _Path(__file__).parent / "static" / "spatial_inspector.html"
if _inspector_path.exists():
    from fastapi.responses import HTMLResponse as _HTMLResponse

    @app.get("/spatial/{room_id}/inspector")
    async def spatial_inspector(room_id: str):
        """Serve the spatial debug inspector for a given room."""
        html = _inspector_path.read_text()
        # Inject the room_id so it auto-connects
        html = html.replace(
            'value="default"',
            f'value="{room_id}"',
        )
        return _HTMLResponse(content=html)

    logger.info("Spatial inspector mounted at /spatial/{room_id}/inspector")


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


# Global State (imported from server_state.py above)


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
        from hololoom.memory.backend_factory import create_memory_backend
        from hololoom.config import MemoryBackend

        # Use INMEMORY if Neo4j/Qdrant unavailable; override with HOLOLOOM_MEMORY env var
        import os
        _mem_override = os.environ.get("HOLOLOOM_MEMORY", "").lower()
        if _mem_override == "inmemory":
            state.config.memory_backend = MemoryBackend.INMEMORY
        else:
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

    # Initialize UnifiedBus (signal backbone for all execution)
    try:
        from hololoom.apps.server.bus_setup import startup_bus
        await startup_bus()
        logger.info("UnifiedBus initialized (signal backbone active)")
    except Exception as e:
        logger.warning(f"UnifiedBus unavailable: {e}")

    # Initialize Continuous Validator (Phase 5 - Routing & RBAC)
    if CONTINUOUS_VALIDATOR_AVAILABLE:
        try:
            classifier = MoonshotQueryClassifier(enable_semantic_tier=False)
            state.continuous_validator = ContinuousValidator(
                classifier=classifier,
                validation_set_path="./data/validation_set.json",
            )
            state._validator_task = asyncio.create_task(
                state.continuous_validator.start_background_validation(interval_s=3600.0)
            )
            logger.info("Continuous validator started (hourly background validation)")
        except Exception as e:
            logger.warning(f"Continuous validator initialization failed: {e}")
            state.continuous_validator = None
    else:
        logger.info("Continuous validator disabled (routing module not available)")

    # Initialize Policy Governance / RBAC (Phase 5 - Routing & RBAC)
    if GOVERNANCE_AVAILABLE:
        try:
            rbac = RoleBasedAccessControl()
            topic_gov = TopicGovernance()
            state.governance_engine = PolicyEngine(rbac=rbac, topic_governance=topic_gov)
            # Register production policy template
            prod_policy = PolicyTemplates.production()
            state.governance_engine.register_policy(prod_policy)
            logger.info("Policy governance engine initialized (production template)")
        except Exception as e:
            logger.warning(f"Policy governance initialization failed: {e}")
            state.governance_engine = None
    else:
        logger.info("Policy governance disabled (policy_governance module not available)")

    # Governance middleware pipeline (Step 1c — composable chain)
    try:
        from hololoom.apps.server.governance import GovernancePipeline
        from hololoom.apps.server.governance.middleware import (
            RBACMiddleware, SafetyGateMiddleware, ReasoningMiddleware,
            AuditTrailMiddleware, DeceptionMonitorMiddleware,
        )
        from hololoom.apps.server.routers.query import get_orchestrator as _get_orchestrator_fn

        pipeline = GovernancePipeline()
        if state.governance_engine:
            pipeline.add(RBACMiddleware(state.governance_engine))
        if state.safety_guardrails:
            pipeline.add(SafetyGateMiddleware(state.safety_guardrails))
        pipeline.add(ReasoningMiddleware(_get_orchestrator_fn))
        if state.audit_trail:
            pipeline.add(AuditTrailMiddleware(state.audit_trail))
        if state.deception_detector:
            pipeline.add(DeceptionMonitorMiddleware(state.deception_detector, state.audit_trail))
        state.governance_pipeline = pipeline
        logger.info(f"Governance pipeline: {pipeline.middleware_names}")
    except Exception as e:
        logger.warning(f"Governance pipeline initialization failed: {e}")
        state.governance_pipeline = None

    # Initialize SAE activation buffer for interpretability training
    try:
        from hololoom.dark_trace.sae.activation_buffer import init_activation_buffer
        state.activation_buffer = init_activation_buffer(
            flush_dir="./data/sae_activations",
            flush_threshold=512,
        )
        state._activation_flush_task = asyncio.create_task(
            state.activation_buffer.start_background_flush()
        )
        logger.info("SAE activation buffer initialized (flush_threshold=512)")
    except Exception as e:
        logger.warning(f"SAE activation buffer initialization failed: {e}")
        state.activation_buffer = None
        state._activation_flush_task = None

    # Attach server state to app for routers (W2 SWOT Remediation - Dec 2025)
    app.state.server_state = state

    logger.info("HoloLoom server ready!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down HoloLoom server...")

    # Stop continuous validator (Phase 5)
    if state.continuous_validator:
        try:
            state.continuous_validator.stop_background_validation()
            if state._validator_task and not state._validator_task.done():
                state._validator_task.cancel()
                try:
                    await state._validator_task
                except asyncio.CancelledError:
                    pass
            logger.info("Continuous validator stopped")
        except Exception as e:
            logger.warning(f"Error stopping continuous validator: {e}")

    # Stop agent monitoring
    if MONITORING_AVAILABLE and state.monitor:
        try:
            await stop_monitoring()
            logger.info("Agent monitoring stopped")
        except Exception as e:
            logger.warning(f"Error stopping monitoring: {e}")

    # Stop SAE activation buffer
    if hasattr(state, 'activation_buffer') and state.activation_buffer:
        state.activation_buffer.stop()
        if hasattr(state, '_activation_flush_task') and state._activation_flush_task and not state._activation_flush_task.done():
            state._activation_flush_task.cancel()
            try:
                await state._activation_flush_task
            except asyncio.CancelledError:
                pass
        logger.info("SAE activation buffer stopped")

    # Close SaaS backend (Dec 2025)
    if SAAS_AVAILABLE and state.saas_backend:
        try:
            await state.saas_backend.close()
            logger.info("SaaS backend closed")
        except Exception as e:
            logger.warning(f"Error closing SaaS backend: {e}")

    # Close Jenny runtime
    if jenny_shutdown is not None:
        try:
            await jenny_shutdown()
            logger.info("Jenny runtime stopped")
        except Exception as e:
            logger.warning(f"Error stopping Jenny runtime: {e}")

    # Stop UnifiedBus
    try:
        from hololoom.apps.server.bus_setup import shutdown_bus
        await shutdown_bus()
        logger.info("UnifiedBus stopped")
    except Exception as e:
        logger.warning(f"Error stopping UnifiedBus: {e}")

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
            entities=["hololoom", "embeddings"],
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
        from hololoom.memory.protocol import MemoryQuery

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













# ============================================================================
# Main (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting HoloLoom Agentic API server (development mode)...")
    uvicorn.run(
        "hololoom.apps.server.agentic_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

#!/usr/bin/env python3
"""
Health & Monitoring Routes for HoloLoom SaaS API

Endpoints:
- GET /health - Basic health check
- GET /health/detailed - Detailed component status
- GET /health/features - Feature flags status
- GET /metrics - Prometheus-compatible metrics
"""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


# ============================================================================
# Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Basic health check response"""
    status: str  # "healthy", "degraded", "unhealthy"
    backend: str  # "postgresql", "sqlite"
    timestamp: str


class ComponentStatus(BaseModel):
    """Status of a single component"""
    name: str
    status: str  # "ok", "degraded", "error"
    latency_ms: float | None = None
    message: str | None = None


class DetailedHealthResponse(BaseModel):
    """Detailed health check with component status"""
    status: str
    backend: str
    timestamp: str
    components: dict[str, ComponentStatus]
    uptime_seconds: float


class FeatureFlagsResponse(BaseModel):
    """Feature flags status"""
    billing_enabled: bool
    usage_tracking_enabled: bool
    audit_log_enabled: bool
    stripe_configured: bool
    backend_type: str


# ============================================================================
# Module-level state
# ============================================================================

_start_time = time.time()
_request_count = 0
_error_count = 0


def increment_request_count():
    """Increment request counter (call from middleware)"""
    global _request_count
    _request_count += 1


def increment_error_count():
    """Increment error counter (call from error handlers)"""
    global _error_count
    _error_count += 1


# ============================================================================
# GET /health - Basic health check
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic health check",
    description="""
    Quick health check for load balancers and monitoring.

    Returns:
    - status: "healthy" if all systems operational
    - backend: Database backend type (postgresql/sqlite)
    - timestamp: Current UTC timestamp

    Use /health/detailed for component-level status.
    """
)
async def health_check(request: Request) -> HealthResponse:
    """Basic health check"""
    backend = getattr(request.app.state, 'saas_backend', None)

    if backend is None:
        return HealthResponse(
            status="unhealthy",
            backend="none",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    # Quick connectivity check
    try:
        # Try a simple query
        await backend.execute_query("SELECT 1", [])
        status = "healthy"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        status = "degraded"

    return HealthResponse(
        status=status,
        backend=backend.backend_type,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# ============================================================================
# GET /health/detailed - Detailed component status
# ============================================================================

@router.get(
    "/health/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed health check",
    description="""
    Detailed health check with component-level status.

    Components checked:
    - database: Database connectivity and query latency
    - features: Feature flags configuration

    Use this for debugging and detailed monitoring dashboards.
    """
)
async def detailed_health_check(request: Request) -> DetailedHealthResponse:
    """Detailed health check with component status"""
    backend = getattr(request.app.state, 'saas_backend', None)
    components = {}
    overall_status = "healthy"

    # Check database
    if backend:
        try:
            start = time.time()
            await backend.execute_query("SELECT 1", [])
            latency = (time.time() - start) * 1000

            components["database"] = ComponentStatus(
                name="database",
                status="ok",
                latency_ms=round(latency, 2),
                message=f"Backend: {backend.backend_type}"
            )
        except Exception as e:
            components["database"] = ComponentStatus(
                name="database",
                status="error",
                message=str(e)
            )
            overall_status = "degraded"
    else:
        components["database"] = ComponentStatus(
            name="database",
            status="error",
            message="Backend not initialized"
        )
        overall_status = "unhealthy"

    # Check feature configuration
    if backend:
        features_ok = True
        messages = []

        if backend.billing_enabled and not backend.config.stripe_api_key:
            features_ok = False
            messages.append("Billing enabled but Stripe API key not configured")

        components["features"] = ComponentStatus(
            name="features",
            status="ok" if features_ok else "degraded",
            message="; ".join(messages) if messages else "All features properly configured"
        )

        if not features_ok:
            overall_status = "degraded"

    return DetailedHealthResponse(
        status=overall_status,
        backend=backend.backend_type if backend else "none",
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
        uptime_seconds=round(time.time() - _start_time, 2)
    )


# ============================================================================
# GET /health/features - Feature flags status
# ============================================================================

@router.get(
    "/health/features",
    response_model=FeatureFlagsResponse,
    summary="Feature flags status",
    description="""
    Get current feature flags configuration.

    Useful for:
    - Verifying deployment configuration
    - Debugging feature availability
    - Operator dashboards
    """
)
async def feature_flags(request: Request) -> FeatureFlagsResponse:
    """Get feature flags status"""
    backend = getattr(request.app.state, 'saas_backend', None)

    if backend is None:
        return FeatureFlagsResponse(
            billing_enabled=False,
            usage_tracking_enabled=False,
            audit_log_enabled=False,
            stripe_configured=False,
            backend_type="none"
        )

    return FeatureFlagsResponse(
        billing_enabled=backend.billing_enabled,
        usage_tracking_enabled=backend.usage_tracking_enabled,
        audit_log_enabled=backend.audit_log_enabled,
        stripe_configured=bool(backend.config.stripe_api_key),
        backend_type=backend.backend_type
    )


# ============================================================================
# GET /metrics - Prometheus-compatible metrics
# ============================================================================

@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="""
    Prometheus-compatible metrics endpoint.

    Metrics exported:
    - hololoom_saas_up: Service health (1=healthy, 0=unhealthy)
    - hololoom_saas_uptime_seconds: Service uptime
    - hololoom_saas_requests_total: Total requests (if middleware enabled)
    - hololoom_saas_errors_total: Total errors (if middleware enabled)
    - hololoom_saas_feature_enabled: Feature flag status

    Configure Prometheus to scrape this endpoint.
    """,
    response_class=None  # Return plain text
)
async def prometheus_metrics(request: Request):
    """Export Prometheus-compatible metrics"""
    from fastapi.responses import PlainTextResponse

    backend = getattr(request.app.state, 'saas_backend', None)

    lines = []

    # Help and type declarations
    lines.append("# HELP hololoom_saas_up Service health status")
    lines.append("# TYPE hololoom_saas_up gauge")

    # Service health
    if backend:
        try:
            await backend.execute_query("SELECT 1", [])
            lines.append("hololoom_saas_up 1")
        except Exception:
            lines.append("hololoom_saas_up 0")
    else:
        lines.append("hololoom_saas_up 0")

    # Uptime
    lines.append("")
    lines.append("# HELP hololoom_saas_uptime_seconds Service uptime in seconds")
    lines.append("# TYPE hololoom_saas_uptime_seconds counter")
    lines.append(f"hololoom_saas_uptime_seconds {time.time() - _start_time:.2f}")

    # Request counts
    lines.append("")
    lines.append("# HELP hololoom_saas_requests_total Total requests processed")
    lines.append("# TYPE hololoom_saas_requests_total counter")
    lines.append(f"hololoom_saas_requests_total {_request_count}")

    lines.append("")
    lines.append("# HELP hololoom_saas_errors_total Total errors")
    lines.append("# TYPE hololoom_saas_errors_total counter")
    lines.append(f"hololoom_saas_errors_total {_error_count}")

    # Feature flags
    if backend:
        lines.append("")
        lines.append("# HELP hololoom_saas_feature_enabled Feature flag status (1=enabled)")
        lines.append("# TYPE hololoom_saas_feature_enabled gauge")
        lines.append(f'hololoom_saas_feature_enabled{{feature="billing"}} {1 if backend.billing_enabled else 0}')
        lines.append(f'hololoom_saas_feature_enabled{{feature="usage_tracking"}} {1 if backend.usage_tracking_enabled else 0}')
        lines.append(f'hololoom_saas_feature_enabled{{feature="audit_log"}} {1 if backend.audit_log_enabled else 0}')

    # Backend info
    lines.append("")
    lines.append("# HELP hololoom_saas_info Service information")
    lines.append("# TYPE hololoom_saas_info gauge")
    backend_type = backend.backend_type if backend else "none"
    lines.append(f'hololoom_saas_info{{backend="{backend_type}"}} 1')

    return PlainTextResponse(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


# ============================================================================
# GET /ready - Kubernetes readiness probe
# ============================================================================

@router.get(
    "/ready",
    summary="Readiness probe",
    description="""
    Kubernetes readiness probe endpoint.

    Returns 200 if service is ready to accept traffic.
    Returns 503 if service is not ready.

    Use in Kubernetes readiness probe configuration.
    """
)
async def readiness_probe(request: Request):
    """Kubernetes readiness probe"""
    from fastapi.responses import JSONResponse

    backend = getattr(request.app.state, 'saas_backend', None)

    if backend is None:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": "Backend not initialized"}
        )

    try:
        await backend.execute_query("SELECT 1", [])
        return {"ready": True}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": str(e)}
        )


# ============================================================================
# GET /live - Kubernetes liveness probe
# ============================================================================

@router.get(
    "/live",
    summary="Liveness probe",
    description="""
    Kubernetes liveness probe endpoint.

    Returns 200 if service is alive.

    Use in Kubernetes liveness probe configuration.
    """
)
async def liveness_probe():
    """Kubernetes liveness probe"""
    return {"alive": True}

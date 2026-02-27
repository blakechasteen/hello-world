#!/usr/bin/env python3
"""
HoloLoom SaaS API Application

Production-ready FastAPI application for self-hosted HoloLoom SaaS.

Features:
- Customer management (signup, login, profile)
- API key authentication
- Usage tracking (optional)
- Billing integration (optional, Stripe)
- Health monitoring & Prometheus metrics
- Kubernetes readiness/liveness probes

Configuration:
- Set DATABASE_URL for PostgreSQL (production)
- Falls back to SQLite if not set (development)
- Enable/disable features via environment variables

Run:
    # Development (SQLite)
    PYTHONPATH=. uvicorn HoloLoom.saas.app:app --reload --port 8000

    # Production (PostgreSQL)
    DATABASE_URL=postgresql://user:pass@host:5432/db \
    PYTHONPATH=. uvicorn HoloLoom.saas.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from HoloLoom.saas import (
    SaaSBackend,
    SaaSConfig,
    create_saas_backend,
)
from HoloLoom.saas.routes import (
    customers_router,
    api_keys_router,
    health_router,
)
from HoloLoom.saas.routes.health import increment_request_count, increment_error_count

# ============================================================================
# Configuration from Environment
# ============================================================================

# Database
DATABASE_URL = os.getenv("DATABASE_URL")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/hololoom_saas.db")

# Feature flags
ENABLE_BILLING = os.getenv("ENABLE_BILLING", "false").lower() == "true"
ENABLE_USAGE_TRACKING = os.getenv("ENABLE_USAGE_TRACKING", "true").lower() == "true"
ENABLE_AUDIT_LOG = os.getenv("ENABLE_AUDIT_LOG", "true").lower() == "true"

# Stripe (optional)
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# API configuration
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)


# ============================================================================
# Backend Setup
# ============================================================================

def create_config() -> SaaSConfig:
    """Create configuration based on environment"""

    # Parse DATABASE_URL if provided
    if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
        # Parse PostgreSQL URL
        # Format: postgresql://user:pass@host:port/database
        import urllib.parse
        parsed = urllib.parse.urlparse(DATABASE_URL)

        return SaaSConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") or "hololoom_saas",
            user=parsed.username or "hololoom",
            password=parsed.password or "",
            enable_billing=ENABLE_BILLING,
            enable_usage_tracking=ENABLE_USAGE_TRACKING,
            enable_audit_log=ENABLE_AUDIT_LOG,
            stripe_api_key=STRIPE_API_KEY,
            stripe_webhook_secret=STRIPE_WEBHOOK_SECRET,
            fallback_to_sqlite=True,  # Fallback if PostgreSQL unavailable
        )
    else:
        # SQLite mode (development)
        return SaaSConfig(
            sqlite_path=SQLITE_PATH,
            fallback_to_sqlite=True,
            enable_billing=ENABLE_BILLING,
            enable_usage_tracking=ENABLE_USAGE_TRACKING,
            enable_audit_log=ENABLE_AUDIT_LOG,
            stripe_api_key=STRIPE_API_KEY,
            stripe_webhook_secret=STRIPE_WEBHOOK_SECRET,
        )


config = create_config()
backend: Optional[SaaSBackend] = None


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global backend

    # Startup
    logger.info("Starting HoloLoom SaaS API...")
    backend = create_saas_backend(config)
    await backend.connect()

    # Store in app state for routes to access
    app.state.saas_backend = backend

    logger.info(f"Backend connected: {backend.backend_type}")
    logger.info(f"Features: billing={backend.billing_enabled}, "
                f"usage_tracking={backend.usage_tracking_enabled}, "
                f"audit_log={backend.audit_log_enabled}")

    yield

    # Shutdown
    logger.info("Shutting down HoloLoom SaaS API...")
    if backend:
        await backend.close()
    logger.info("Backend closed")


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    app = FastAPI(
        title="HoloLoom SaaS API",
        description="""
HoloLoom SaaS API for self-hosted deployments.

## Features
- **Authentication**: Customer signup, login, API key management
- **Usage Tracking**: Per-query usage metering (optional)
- **Billing**: Stripe integration for subscriptions (optional)
- **Health Monitoring**: Health checks, Prometheus metrics, K8s probes

## Quick Start
1. POST /api/v1/customers/signup - Create account
2. Use returned API key in `X-API-Key` header
3. Make authenticated requests to your protected endpoints

## Self-Hosting
See https://github.com/your-org/hololoom/docs/self-hosting for deployment guides.
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ========================================================================
    # Middleware
    # ========================================================================

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request counting (for metrics)
    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        increment_request_count()
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            increment_error_count()
            raise

    # ========================================================================
    # Routes
    # ========================================================================

    # Health & monitoring (no prefix, standard paths)
    app.include_router(health_router)

    # Customer management (router already has /api/v1/customers prefix)
    app.include_router(
        customers_router,
        tags=["customers"],
    )

    # API key management (router already has /api/v1/api-keys prefix)
    app.include_router(
        api_keys_router,
        tags=["api-keys"],
    )

    # Future: billing_router, dashboard_router, webhooks_router

    # ========================================================================
    # Error Handlers
    # ========================================================================

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions"""
        increment_error_count()
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
            },
        )

    # ========================================================================
    # Root Endpoint
    # ========================================================================

    @app.get("/", tags=["info"])
    async def root():
        """API information and quick start guide"""
        return {
            "name": "HoloLoom SaaS API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": {
                "basic": "/health",
                "detailed": "/health/detailed",
                "features": "/health/features",
                "metrics": "/metrics",
                "k8s_ready": "/ready",
                "k8s_live": "/live",
            },
            "api": {
                "customers": {
                    "signup": f"POST {API_PREFIX}/customers/signup",
                    "login": f"POST {API_PREFIX}/customers/login",
                    "profile": f"GET {API_PREFIX}/customers/profile",
                },
                "api_keys": {
                    "create": f"POST {API_PREFIX}/api-keys",
                    "list": f"GET {API_PREFIX}/api-keys",
                    "revoke": f"DELETE {API_PREFIX}/api-keys/{{key_id}}",
                },
            },
            "features": {
                "billing_enabled": ENABLE_BILLING,
                "usage_tracking_enabled": ENABLE_USAGE_TRACKING,
                "audit_log_enabled": ENABLE_AUDIT_LOG,
            },
            "quick_start": [
                "1. POST /api/v1/customers/signup with email, name, password",
                "2. Save the returned api_secret (shown once!)",
                "3. Use X-API-Key header with your api_secret for authenticated requests",
            ],
        }

    return app


# Create the application instance
app = create_app()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get port from environment
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║               HoloLoom SaaS API Server                       ║
╠══════════════════════════════════════════════════════════════╣
║  Host: {host:<53} ║
║  Port: {port:<53} ║
║  Workers: {workers:<50} ║
║  Reload: {reload!s:<51} ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    Docs:    http://{host}:{port}/docs{' ':<27} ║
║    Health:  http://{host}:{port}/health{' ':<25} ║
║    Metrics: http://{host}:{port}/metrics{' ':<24} ║
╚══════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "HoloLoom.saas.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )

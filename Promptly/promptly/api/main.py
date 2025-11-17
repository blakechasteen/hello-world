"""
Promptly REST API - Main FastAPI Application
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
from datetime import datetime

from .config import get_settings
from .middleware import RateLimitMiddleware, LoggingMiddleware, init_default_auth
from .models.common import HealthResponse, ErrorResponse
from .routes import (
    prompts_router,
    branches_router,
    evaluations_router,
    chains_router,
    plugins_router,
    auth_router,
    history_router,
)
from . import ws_routes

settings = get_settings()

# Track startup time for uptime metrics
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting Promptly API...")

    # Initialize default auth for development
    default_key = init_default_auth()
    print(f"🔑 Development API Key: {default_key}")

    yield

    # Shutdown
    print("👋 Shutting down Promptly API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready REST API for Promptly - Advanced prompt management with versioning, branching, and evaluation",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.rate_limit_per_minute,
    burst=settings.rate_limit_burst
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            message="Validation error",
            detail=str(exc.errors()),
            error_code="VALIDATION_ERROR"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            message="Internal server error",
            detail=str(exc),
            error_code="INTERNAL_ERROR"
        ).model_dump()
    )


# Health check endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - startup_time

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        uptime_seconds=uptime
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(auth_router, prefix=f"{settings.api_prefix}/auth", tags=["Authentication"])
app.include_router(prompts_router, prefix=f"{settings.api_prefix}/prompts", tags=["Prompts"])
app.include_router(branches_router, prefix=f"{settings.api_prefix}/branches", tags=["Branches"])
app.include_router(history_router, prefix=f"{settings.api_prefix}/history", tags=["History"])
app.include_router(evaluations_router, prefix=f"{settings.api_prefix}/evaluations", tags=["Evaluations"])
app.include_router(chains_router, prefix=f"{settings.api_prefix}/chains", tags=["Chains"])
app.include_router(plugins_router, prefix=f"{settings.api_prefix}/plugins", tags=["Plugins"])
app.include_router(ws_routes.router, tags=["WebSocket"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
        log_level=settings.log_level.lower()
    )

"""
HoloLoom API Routes
===================

FastAPI route modules organized by domain.

Modules:
    health: Health check and server statistics
    query: Main query endpoint for agentic reasoning
    memory: Memory storage and retrieval operations
    workspace: Workspace and file ingestion
    analysis: Code analysis and detection endpoints
    visualization: Graph visualization endpoints
    monitoring: Agent monitoring and WebSocket

Usage:
    from fastapi import FastAPI
    from HoloLoom.server.api.routes import include_all_routers

    app = FastAPI()
    include_all_routers(app)
"""

from fastapi import FastAPI

from .health import router as health_router
from .query import router as query_router
from .memory import router as memory_router
from .workspace import router as workspace_router
from .analysis import router as analysis_router
from .visualization import router as visualization_router
from .monitoring import router as monitoring_router


def include_all_routers(app: FastAPI) -> None:
    """Include all route modules in the FastAPI app."""
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(memory_router)
    app.include_router(workspace_router)
    app.include_router(analysis_router)
    app.include_router(visualization_router)
    app.include_router(monitoring_router)


__all__ = [
    "health_router",
    "query_router",
    "memory_router",
    "workspace_router",
    "analysis_router",
    "visualization_router",
    "monitoring_router",
    "include_all_routers",
]

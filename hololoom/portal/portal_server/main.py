"""
Portal Server - Control plane for the Loom.

FastAPI application providing node registry and Loom status endpoints.

Endpoints:
    POST /nodes/register - Register/update a node
    POST /nodes/{node_id}/heartbeat - Node heartbeat
    GET /nodes - List all nodes
    GET /nodes/{node_id} - Get specific node
    DELETE /nodes/{node_id} - Unregister a node
    GET /loom/status - Overall Loom status
    GET /health - Health check
"""

import argparse
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from ..shared.logging import configure_logging, get_logger
from ..shared.types import (
    LoomStatus,
    NodeRecord,
    NodeRegistration,
)
from .config import PortalConfig, load_config
from .registry import NodeRegistry

# Global instances (set during lifespan)
registry: NodeRegistry | None = None
config: PortalConfig | None = None
logger = get_logger(__name__, component="portal")


class HealthResponse(BaseModel):
    status: str
    loom_id: str
    uptime_seconds: float
    timestamp: str


class RegisterResponse(BaseModel):
    message: str
    node: NodeRecord


def verify_secret(x_shared_secret: str = Header(...)) -> str:
    """Verify shared secret from header."""
    if config and x_shared_secret != config.shared_secret:
        raise HTTPException(status_code=401, detail="Invalid shared secret")
    return x_shared_secret


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize registry on startup."""
    global registry, config

    # Load config (can be overridden before startup)
    if config is None:
        config = load_config()

    configure_logging(level=config.log_level, json_format=config.log_json)
    logger.info(f"Starting Portal Server for Loom: {config.loom_id}")

    registry = NodeRegistry(
        loom_id=config.loom_id,
        node_timeout_seconds=config.node_timeout_seconds
    )

    yield

    logger.info("Portal Server shutting down")


def create_app(portal_config: PortalConfig | None = None) -> FastAPI:
    """
    Create FastAPI application with optional config.

    Args:
        portal_config: Configuration (uses defaults if not provided)

    Returns:
        Configured FastAPI app
    """
    global config
    if portal_config:
        config = portal_config

    app = FastAPI(
        title="Portal Server",
        description="Control plane for HoloLoom distributed compute",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(router)

    return app


# Router with all endpoints
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    status = await registry.get_status() if registry else None
    return HealthResponse(
        status="healthy",
        loom_id=config.loom_id if config else "unknown",
        uptime_seconds=status.uptime_seconds if status else 0,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/nodes/register", response_model=RegisterResponse)
async def register_node(
    registration: NodeRegistration,
    _: str = Depends(verify_secret),
):
    """
    Register or update a compute node.

    Nodes should call this on startup and periodically to maintain registration.
    """
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    # Verify the registration secret matches
    if config and registration.shared_secret != config.shared_secret:
        raise HTTPException(status_code=401, detail="Invalid shared secret")

    node = await registry.register(
        node_id=registration.node_id,
        address=registration.address,
        capabilities=registration.capabilities,
    )

    logger.info(
        f"Node registered: {registration.node_id} at {registration.address}",
        extra={"node_id": registration.node_id}
    )

    return RegisterResponse(
        message=f"Node {registration.node_id} registered successfully",
        node=node,
    )


@router.post("/nodes/{node_id}/heartbeat", response_model=NodeRecord)
async def node_heartbeat(
    node_id: str,
    _: str = Depends(verify_secret),
):
    """
    Update node heartbeat.

    Nodes should call this every 30 seconds to stay marked as online.
    """
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    node = await registry.heartbeat(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    return node


@router.get("/nodes", response_model=list[NodeRecord])
async def list_nodes(
    online_only: bool = False,
    _: str = Depends(verify_secret),
):
    """
    List all registered nodes.

    Args:
        online_only: If true, only return online nodes
    """
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    if online_only:
        return await registry.list_online()
    return await registry.list_all()


@router.get("/nodes/{node_id}", response_model=NodeRecord)
async def get_node(
    node_id: str,
    _: str = Depends(verify_secret),
):
    """Get a specific node by ID."""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    node = await registry.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    return node


@router.delete("/nodes/{node_id}")
async def unregister_node(
    node_id: str,
    _: str = Depends(verify_secret),
):
    """Unregister a node."""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    removed = await registry.unregister(node_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    logger.info(f"Node unregistered: {node_id}", extra={"node_id": node_id})
    return {"message": f"Node {node_id} unregistered"}


@router.get("/loom/status", response_model=LoomStatus)
async def loom_status(_: str = Depends(verify_secret)):
    """Get overall Loom status including all nodes."""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    return await registry.get_status()


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Portal Server")
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--host", help="Override bind host")
    parser.add_argument("--port", type=int, help="Override bind port")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply overrides
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port

    # Create app with config
    application = create_app(cfg)

    # Run server
    uvicorn.run(
        application,
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower(),
    )
